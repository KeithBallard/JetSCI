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
