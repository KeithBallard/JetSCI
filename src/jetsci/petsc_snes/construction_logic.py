from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from ..options import (
    NonlinearSolverType,
    PETScLinearSolverType,
    PETScPreconditionerType,
    SolverOptions,
)
from .construction_build import *

# Stores a map from a key to the solver object, allowing reuse between nonlinear solve calls
__solver_dict = {}


def validate_petsc_solver_options(options: SolverOptions) -> None:
    """Validate that selected solver families use compatible method enums."""
    if options.nonlinear_solver_type is NonlinearSolverType.PETSC_SNES:
        if not isinstance(options.linear_solve_type, PETScLinearSolverType):
            raise TypeError(
                "PETSc SNES requires a PETSc linear solver method. "
                f"Got {options.linear_solve_type!r}."
            )
        if not isinstance(options.linear_precond_type, PETScPreconditionerType):
            raise TypeError(
                "PETSc SNES requires a PETSc preconditioner method. "
                f"Got {options.linear_precond_type!r}."
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


def destroy_petsc_solver(solver_key: int):
    """Remove a solver from the dictionary and destroy its PETSc objects."""
    solver = __solver_dict.pop(solver_key)
    solver.destroy()
    return solver
