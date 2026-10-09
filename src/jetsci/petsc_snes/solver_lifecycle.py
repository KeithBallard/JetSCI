from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
import warnings

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from ..options import *
from ..conversions import *
from .solver import *


from petsc4py import PETSc


_PETSC_KSP_TYPES = {
    LinearSolverType.PETSC_CG: "cg",
    LinearSolverType.PETSC_GMRES: "gmres",
    LinearSolverType.PETSC_LGMRES: "lgmres",
    LinearSolverType.PETSC_BCGS: "bcgs",
    LinearSolverType.PETSC_PREONLY: "preonly",
    LinearSolverType.PETSC_MINRES: "minres",
}

_PETSC_PC_TYPES = {
    PreconditionerType.PETSC_NONE: "none",
    PreconditionerType.PETSC_JACOBI: "jacobi",
    PreconditionerType.PETSC_ILU: "ilu",
}


# Stores a map from a key to the solver object, allowing reuse between nonlinear solve calls.
# TODO: convert the tuple value to a small dataclass record once the SNES/IFT
# ownership split settles down.
__solver_dict = {}
__solver_idNum = 0


def _new_solver_key():
    global __solver_idNum
    __solver_idNum += (
        1  # do not base on dict size alone, otherwise you'll get overwriting
    )
    return __solver_idNum


def get_petsc_solver_objects_from_key(solver_key: int):
    """Return the live SNES and companion KSP wrappers for `solver_key`."""
    if solver_key not in __solver_dict:
        raise KeyError(f"No PETSc solver found for solver_key={solver_key}")
    return __solver_dict[solver_key]


def _coo_jacobian_function(R: Callable, J: Callable | None):
    """Return a function of x that produces COOData for the SNES Jacobian."""


    #DEBUG PRINT
    #print("starting _coo_jacobian_function in lifecycle")

    if J is None:
        warnings.warn(
            "No Jacobian function J(x) was provided to PETSc SNES. "
            "A dense Jacobian will be materialized via jax.jacfwd(R)(x) and converted to COO data on every step. "
            "This severely degrades performance on large problems. "
            "Fix: Provide an explicit Jacobian function J(x) returning COOData or a sparse matrix.",
            UserWarning,
            stacklevel=2,
        )

        def jacobian_coo_from_residual(x):
            J_dense = jax.jacfwd(R)(x)
            return convert_jax_dense_mat_to_coo_data(J_dense)
            J = jax.jacfwd(R)(x)
            print(J)#we should probably ditch this
            return convert_jax_dense_mat_to_coo_data(J)

        return jacobian_coo_from_residual

    def jacobian_coo(x):

        #DEBUG PRINT
        #print("calling jacobian_coo converted function")

        jacobian = J(x)
        if all(hasattr(jacobian, field) for field in ("shape", "vals", "rows", "cols")):
            return jacobian
        if isinstance(jacobian, jsparse.COO) or (hasattr(jacobian, "data") and hasattr(jacobian, "row") and hasattr(jacobian, "col")):
            return COOData(
                shape=jnp.asarray(jacobian.shape, dtype=jnp.int64),
                vals=jacobian.data,
                rows=jnp.asarray(jacobian.row, dtype=jnp.int32),
                cols=jnp.asarray(jacobian.col, dtype=jnp.int32),
            )
        warnings.warn(
            "Jacobian function returned a dense array instead of COOData for PETSc SNES. "
            "Converting dense matrix to COO format on every evaluation. "
            "Fix: Have J(x) return COOData or a sparse matrix representation to avoid conversion overhead.",
            UserWarning,
            stacklevel=2,
        )
        return convert_jax_dense_mat_to_coo_data(jnp.asarray(jacobian))
        return convert_jax_mat_to_coo_data(jacobian)

    #DEBUG PRINT
    #print("completed _coo_jacobian_function conversion")

    return jacobian_coo


def _apply_snes_options(snes, options: SolverOptions):
    #DEBUG PRINT
    #print("calling solver_lifecycle _apply_snes_options")
    snes.setTolerances(
        rtol=options.nonlinear_relative_tol,
        atol=options.nonlinear_absolute_tol,
        stol=options.nonlinear_step_tol,
        max_it=options.nonlinear_max_iter,
    )


def _apply_ksp_options(snes, options: SolverOptions):
    #DEBUG PRINT
    #print("calling solver_lifecycle _apply_ksp_options")
    ksp = snes.getKSP()
    ksp.setType(_PETSC_KSP_TYPES[options.linear_solver_type])
    if hasattr(PETSc.KSP, "NormType"):
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(
        rtol=options.linear_relative_tol,
        atol=options.linear_absolute_tol,
        max_it=options.linear_max_iter,
    )
    pc = ksp.getPC()
    pc.setType(_PETSC_PC_TYPES[options.linear_preconditioner_type])


def _apply_ksp_options_direct(ksp, options: LinearSolverOptions):
    """Apply KSP/PC options to a standalone PETSc KSP object."""
    ksp.setType(_PETSC_KSP_TYPES[options.linear_solver_type])
    if hasattr(PETSc.KSP, "NormType"):
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(
        rtol=options.linear_relative_tol,
        atol=options.linear_absolute_tol,
        max_it=options.linear_max_iter,
    )
    pc = ksp.getPC()
    pc.setType(_PETSC_PC_TYPES[options.linear_preconditioner_type])


def build_petsc_snes_from_options(
    R: Callable, J: Callable | None, options: SolverOptions
):
    """Build a PETSc SNES solver from JAX residual/Jacobian functions.

    `R` is expected to be a JAX function of the nonlinear state `x`. If `J` is
    provided it may return either COOData or a dense rank-2 JAX matrix. If `J`
    is `None`, a dense Jacobian is built with `jax.jacfwd(R)` for now.
    """
    if options.nonlinear_solver_type is not NonlinearSolverType.PETSC_SNES:
        raise TypeError("build_petsc_snes_from_options only builds PETSc SNES solvers")

    callback_stats = {}
    residual_callback = convert_jax_vec_func_to_petsc_vec_func(
        R,
        stats=callback_stats,
    )
    jacobian_callback_state = PatternAwareMatAssignmentState()
    jacobian_callback = convert_jax_coo_mat_func_to_petsc_mat_func_pattern_aware(
        _coo_jacobian_function(R, J),
        state=jacobian_callback_state,
        stats=callback_stats,
    )

    snes = PETSc.SNES().create(PETSc.COMM_WORLD)
    _apply_snes_options(snes, options)
    _apply_ksp_options(snes, options)

    return PETScNonlinearSolver(
        snes=snes,
        residual_callback=residual_callback,
        jacobian_callback=jacobian_callback,
        options=options,
        jacobian_callback_state=jacobian_callback_state,
        callback_stats=callback_stats,
    )


def build_petsc_internal_ksp_from_options(options: SolverOptions):
    """Build a standalone PETSc KSP wrapper for IFT-style linear solves."""
    if options.nonlinear_solver_type is not NonlinearSolverType.PETSC_SNES:
        raise TypeError("build_petsc_internal_ksp_from_options only builds PETSc KSP solvers")

    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    _apply_ksp_options_direct(ksp, options)

    return PETScLinearSolver(
        ksp=ksp,
        vector_callback=lambda *args, **kwargs: None,
        matrix_callback=lambda *args, **kwargs: None,
        options=options,
    )

#Maybe rename this to 'fetch' since it only builds if there's not one already in memory
def build_petsc_solver_with_reuse(
    options: SolverOptions,
    R: jax.tree_util.Partial,
    J: jax.tree_util.Partial,
    x0: jnp.ndarray | None = None,
):
    """Return a solver and SolverOptions containing its dictionary key.

    If `options.solver_key` is `None`, a new PETSc solver is built and stored.
    If a key is present, the existing solver is retrieved and refreshed with
    the latest callbacks and method options.
    """

    validate_petsc_solver_options(options)

    if options.solver_key is None:
        solver = build_petsc_snes_from_options(R, J, options)
        ksp_for_IFT = build_petsc_internal_ksp_from_options(options)
        solver_key = _new_solver_key()
        __solver_dict[solver_key] = (
            solver,
            ksp_for_IFT,
        )  # this way we hide the KSP since we only need it for the KSP
        return solver, replace(options, solver_key=solver_key)
    else:
        if options.solver_key not in __solver_dict:
            raise KeyError(f"No PETSc solver found for solver_key={options.solver_key}")

        solver = __solver_dict[options.solver_key][0]
        update_petsc_snes_callbacks(solver, R, J)
        update_petsc_snes_options(solver, options)

        return solver, options


def update_petsc_snes_callbacks(
    solver: PETScNonlinearSolver,
    R: Callable,
    J: Callable | None,
):
    """Replace residual/Jacobian callbacks on an existing PETSc solver."""
    if solver.callback_stats is None:
        solver.callback_stats = {}
    solver.residual_callback = convert_jax_vec_func_to_petsc_vec_func(
        R,
        stats=solver.callback_stats,
    )
    if solver.jacobian_callback_state is None:
        solver.jacobian_callback_state = PatternAwareMatAssignmentState()
    solver.jacobian_callback = convert_jax_coo_mat_func_to_petsc_mat_func_pattern_aware(
        _coo_jacobian_function(R, J),
        state=solver.jacobian_callback_state,
        stats=solver.callback_stats,
    )
    if solver.residual_vec is not None:
        solver.snes.setFunction(solver.residual_callback, solver.residual_vec)
    if solver.jacobian_mat is not None:
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


def update_petsc_linear_solver_options(
    solver: PETScLinearSolver,
    options: SolverOptions,
):
    """Apply new PETSc method/tolerance options to an existing KSP wrapper."""
    solver.options = options
    _apply_ksp_options_direct(solver.ksp, options)
    return solver


def destroy_petsc_solver(solver_key: int):
    """Remove a solver from the dictionary and destroy its PETSc objects."""
    solver = __solver_dict.pop(solver_key, None)
    if solver is None:
        return None
    solver[0].destroy()
    solver[1].destroy()
    if hasattr(PETSc, "garbage_cleanup"):
        PETSc.garbage_cleanup()
    return solver

    # careful with this, because it can let you overwriting existing solvers in it's current state.
    # If you have 2 solvers and pop number 1 the next id will be 2 which will overwrite
    # it may be better to move to an increasing number system
