from __future__ import annotations

from typing import *

import jax.numpy as jnp
import jax

from .options import *
from .lifecycle import *

#this is where the solver is called from, so this contains running info

def differentiableSolve(solver_options: solverConstructionOptions.SolverOptions, R: Callable, J: Optional[Callable], x_0, *phi) -> tuple[jnp.ndarray, solverConstructionOptions.SolverOptions]:
    if phi:
        R_bar = jax.tree_util.Partial(R, *phi)
        J_bar = None if J is None else jax.tree_util.Partial(J, *phi)
    else:
        R_bar = R
        J_bar = J
    solver, solver_options = solverConstructionLogic.checkSolverExistence(
        solver_options,
        R_bar,
        J_bar,
        x_0,
    )
    x_solution = solver.solve(x_0)
    return x_solution, solver_options

def differentiable_solve(solver_options: SolverOptions, R: Callable, J_x: Optional[Callable], x_0, *args) -> tuple[jnp.ndarray, SolverOptions]:
    if args:
        R_bar = jax.tree_util.Partial(R, *args)
        J_bar = None if J is None else jax.tree_util.Partial(J, *args)
    else:
        R_bar = jax.tree_util.Partial(R)
        J_bar = None if J is None else jax.tree_util.Partial(J)

    if solver_options.nonlinear_solver_type == NonlinearSolverType.JAX_NEWTON_RAPHSON:
        solver = ...
    elif solver_options.nonlinear_solver_type == NonlinearSolverType.PETSC_SNES:
        solver, solver_options = build_petsc_solver_with_reuse(
            solver_options,
            R_bar,
            J_bar,
            x_0,
        )