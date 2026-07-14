from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from ..options import *
from ..conversions import *


_PETSC_KSP_TYPES = {
    PETScLinearSolverType.CG: "cg",
    PETScLinearSolverType.LGMRES: "lgmres",
    PETScLinearSolverType.BCGS: "bcgs",
    PETScLinearSolverType.PREONLY: "preonly",
}

_PETSC_PC_TYPES = {
    PETScPreconditionerType.NONE: "none",
    PETScPreconditionerType.JACOBI: "jacobi",
    PETScPreconditionerType.ILU: "ilu",
}


@dataclass
class PETScNonlinearSolver:
    """Own the PETSc SNES object and its callback companion objects."""

    snes: object
    residual_callback: Callable
    jacobian_callback: Callable
    options: SolverOptions
    residual_vec: object | None = None
    jacobian_mat: object | None = None

    def initialize_from_x0(self, x0):
        """Allocate PETSc Vec/Mat objects that require the state size."""
        from petsc4py import PETSc

        self.cleanup_work_vectors()
        x0_vec = jaxArrayToPETScVec(x0)
        try:
            self.residual_vec = x0_vec.duplicate()
            self.jacobian_mat = PETSc.Mat().create(PETSc.COMM_WORLD)
            self.snes.setFunction(self.residual_callback, self.residual_vec)
            self.snes.setJacobian(self.jacobian_callback, self.jacobian_mat, self.jacobian_mat)
        finally:
            x0_vec.destroy()
        return self

    def solve(self, x0):
        """Solve with this SNES object and return a PETSc Vec.

        The caller owns the returned Vec and is responsible for destroying it.
        """
        if self.residual_vec is None or self.jacobian_mat is None:
            self.initialize_from_x0(x0)

        x0_vec = jaxArrayToPETScVec(x0)
        x = x0_vec.duplicate()
        try:
            x0_vec.copy(x)
            self.snes.solve(None, x)
            return x
        finally:
            x0_vec.destroy()

    def solve_to_jax(self, x0):
        """Solve and explicitly copy the PETSc Vec result into a JAX array."""
        x = self.solve(x0)
        try:
            result = petscVecToJAX(x).copy()
            result.block_until_ready()
            return result
        finally:
            x.destroy()

    def linear_solve(self, x0):
        pass

    def cleanup_work_vectors(self):
        """Destroy residual/Jacobian objects that depend on vector size."""
        if self.residual_vec is not None:
            self.residual_vec.destroy()
            self.residual_vec = None
        if self.jacobian_mat is not None:
            self.jacobian_mat.destroy()
            self.jacobian_mat = None

    def destroy(self):
        """Destroy all PETSc objects owned by this wrapper."""
        self.cleanup_work_vectors()
        self.snes.destroy()


def _coo_jacobian_function(R: Callable, J: Callable | None):
    """Return a function of x that produces COOData for the SNES Jacobian."""
    if J is None:

        def jacobian_coo_from_residual(x):
            return convertJaxMatToCOOData(jax.jacfwd(R)(x))

        return jacobian_coo_from_residual

    def jacobian_coo(x):
        jacobian = J(x)
        if all(hasattr(jacobian, field) for field in ("shape", "vals", "rows", "cols")):
            return jacobian
        return convertJaxMatToCOOData(jnp.asarray(jacobian))

    return jacobian_coo


def _apply_snes_options(snes, options: SolverOptions):
    snes.setTolerances(
        rtol=options.nonlinear_relative_tol,
        atol=options.nonlinear_absolute_tol,
        max_it=options.nonlinear_max_iter,
    )


def _apply_ksp_options(snes, options: SolverOptions):
    ksp = snes.getKSP()
    ksp.setType(_PETSC_KSP_TYPES[options.linear_solve_type])
    ksp.setTolerances(
        rtol=options.linear_relative_tol,
        atol=options.linear_absolute_tol,
        max_it=options.linear_max_iter,
    )
    pc = ksp.getPC()
    pc.setType(_PETSC_PC_TYPES[options.linear_precond_type])


def build_petsc_snes_from_options(R: Callable, J: Callable | None, options: SolverOptions):
    """Build a PETSc SNES solver from JAX residual/Jacobian functions.

    `R` is expected to be a JAX function of the nonlinear state `x`. If `J` is
    provided it may return either COOData or a dense rank-2 JAX matrix. If `J`
    is `None`, a dense Jacobian is built with `jax.jacfwd(R)` for now.
    """
    from petsc4py import PETSc

    if options.nonlinear_solver_type is not NonlinearSolverType.PETSC_SNES:
        raise TypeError("buildPETScSolverFromOptions only builds PETSc SNES solvers")

    residual_callback = convert_jax_vec_func_to_petsc_vec_func(R)
    jacobian_callback = convert_jax_coo_mat_func_to_petsc_mat_func_pattern_aware(
        _coo_jacobian_function(R, J)
    )

    snes = PETSc.SNES().create(PETSc.COMM_WORLD)
    _apply_snes_options(snes, options)
    _apply_ksp_options(snes, options)

    return PETScNonlinearSolver(
        snes=snes,
        residual_callback=residual_callback,
        jacobian_callback=jacobian_callback,
        options=options,
    )


def update_petsc_snes_callbacks(
    solver: PETScNonlinearSolver,
    R: Callable,
    J: Callable | None,
):
    """Replace residual/Jacobian callbacks on an existing PETSc solver."""
    solver.residual_callback = convert_jax_vec_func_to_petsc_vec_func(R)
    solver.jacobian_callback = convert_jax_coo_mat_func_to_petsc_mat_func_pattern_aware(
        _coo_jacobian_function(R, J)
    )
    if solver.residual_vec is not None and solver.jacobian_mat is not None:
        solver.snes.setFunction(solver.residual_callback, solver.residual_vec)
        solver.snes.setJacobian(
            solver.jacobian_callback,
            solver.jacobian_mat,
            solver.jacobian_mat,
        )
    return solver


def update_petsc_snes_options(solver: PETScNonlinearSolver, options: SolverOptions):
    """Apply new PETSc method/tolerance options to an existing solver."""
    solver.options = options
    _apply_snes_options(solver.snes, options)
    _apply_ksp_options(solver.snes, options)
    return solver


