from __future__ import annotations

import jax
import jax.numpy as jnp

from . import petsc_snes
from . import jax_newton_raph
from .options import *


def build_solver_with_reuse(
    options: SolverOptions,
    R: jax.tree_util.Partial,
    J_x: jax.tree_util.Partial | None = None,
    x_0: jnp.ndarray | None = None,
) -> tuple[Any, SolverOptions]:
    match options.nonlinear_solver_type:
        case NonlinearSolverType.JAX_NEWTON_RAPHSON:
            return jax_newton_raph.build_jax_solver_with_reuse(
                options,
                R,
                J_x,
                x_0,
            )
        case NonlinearSolverType.PETSC_SNES:
            return petsc_snes.build_petsc_solver_with_reuse(
                options,
                R,
                J_x,
                x_0,
            )
        case _:
            raise NotImplementedError(
                f"Nonlinear solver type {options.nonlinear_solver_type} is not implemented."
            )


__linear_solver_dict: dict[int, Any] = {}
__linear_solver_id = 0


def _new_linear_solver_key() -> int:
    global __linear_solver_id
    __linear_solver_id += 1
    return __linear_solver_id


def build_linear_solver_with_reuse(
    options: LinearSolverOptions,
    A: Any,
) -> tuple[Any, LinearSolverOptions]:
    """Construct or retrieve a linear solver matching options backend, binding operator A.

    If options.solver_key is None, a new linear solver is constructed and assigned a key.
    If options.solver_key is present, the existing solver is retrieved and updated with A
    via solver.update_operator(A).
    """
    if options.solver_key is None:
        if options.linear_solver_type.is_jax:
            from .jax_linear.sparse_linear_solve import JAXLinearSolver
            solver = JAXLinearSolver(A=A, options=options)
        elif options.linear_solver_type.is_petsc:
            from .petsc_snes.solver import build_petsc_linear_solver
            solver = build_petsc_linear_solver(A=A, options=options)
        else:
            raise ValueError(f"Unknown linear solver type: {options.linear_solver_type}")

        key = _new_linear_solver_key()
        __linear_solver_dict[key] = solver
        from dataclasses import replace
        return solver, replace(options, solver_key=key)
    else:
        if options.solver_key not in __linear_solver_dict:
            raise KeyError(f"No linear solver found for solver_key={options.solver_key}")
        solver = __linear_solver_dict[options.solver_key]
        solver.update_operator(A)
        return solver, options


def destroy_linear_solver(solver_key: int) -> None:
    """Destroy and remove a linear solver from the registry."""
    solver = __linear_solver_dict.pop(solver_key, None)
    if solver is not None and hasattr(solver, "destroy"):
        solver.destroy()

