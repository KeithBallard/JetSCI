from __future__ import annotations

from dataclasses import dataclass, replace
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


# Stores a map from a key to the solver object, allowing reuse between nonlinear solve calls
__solver_dict = {}


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


def build_petsc_solver_with_reuse(
    options: SolverOptions,
    R: jax.tree_util.Partial,
    J: jax.tree_util.Partial,
    x0: jnp.ndarry | None = None,
):
    """Return a solver and SolverOptions containing its dictionary key.

    If `options.solver_key` is `None`, a new PETSc solver is built and stored.
    If a key is present, the existing solver is retrieved and refreshed with
    the latest callbacks and method options.
    """

    validate_solver_options(options)

    if options.solver_key is None:
        solver = solverConstructionBuilding.buildPETScSolverFromOptions(R, J, options)
        solver_key = _new_solver_key()
        __solver_dict[solver_key] = solver
        return solver, replace(options, solver_key=solver_key)

    if options.solver_key not in __solver_dict:
        raise KeyError(f"No PETSc solver found for solver_key={options.solver_key}")

    solver = __solver_dict[options.solver_key]
    solverConstructionBuilding.updatePETScSolverCallbacks(solver, R, J)
    solverConstructionBuilding.updatePETScSolverMethods(solver, options)
    return solver, options


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


def destroy_petsc_solver(solver_key: int):
    """Remove a solver from the dictionary and destroy its PETSc objects."""
    solver = __solver_dict.pop(solver_key)
    solver.destroy()
    return solver