from __future__ import annotations

from typing import Callable, Optional, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from .options import (
    LinearSolverOptions,
    SolverOptions,
    NonlinearSolverType,
    LinearSolverType,
    PreconditionerType,
)
from .lifecycle import build_solver_with_reuse, build_linear_solver_with_reuse, __linear_solver_dict
from .conversions import COOData
from .jax_linear import linear_solve, LinearSolverResultInfo


def differentiable_solve(
    solver_options: SolverOptions,
    R: Callable,
    J_x: Optional[Callable],
    x_0: jnp.ndarray,
    *args,
) -> tuple[jnp.ndarray, SolverOptions]:
    """Solve nonlinear problem R(*args, x) = 0 with forward and reverse autodiff.

    Uses Implicit Function Theorem (IFT) differentiation via `jax.lax.custom_linear_solve`
    so that both forward-mode (`jacfwd`, `jvp`) and reverse-mode (`jacrev`, `grad`, `vjp`)
    autodiff work across all nonlinear and linear solver options.
    """
    x_0 = jnp.asarray(x_0)

    if not args:
        R_bar = jax.tree_util.Partial(R)
        J_bar = None if J_x is None else jax.tree_util.Partial(J_x)
        solver, solver_options = build_solver_with_reuse(
            solver_options,
            R_bar,
            J_bar,
            x_0,
        )
        x_solution = solver.solve_to_jax(x_0)
        return x_solution, solver_options

    def _solve_core(*params):
        params_stopped = [jax.lax.stop_gradient(p) for p in params]
        R_primal = jax.tree_util.Partial(R, *params_stopped)
        J_primal = None if J_x is None else jax.tree_util.Partial(J_x, *params_stopped)

        solver, updated_opts = build_solver_with_reuse(
            solver_options,
            R_primal,
            J_primal,
            x_0,
        )
        x_star = jax.lax.stop_gradient(solver.solve_to_jax(x_0))

        if J_x is not None:
            J_mat = J_x(*params, x_star)
            if callable(J_mat):
                A_matvec = J_mat
            elif isinstance(J_mat, (jsparse.COO,)):
                A_matvec = lambda v: jnp.zeros(J_mat.shape[0], dtype=v.dtype).at[J_mat.row].add(J_mat.data * v[J_mat.col])
            else:
                A_matvec = lambda v: J_mat @ v
        else:
            A_matvec = lambda v: jax.jvp(lambda y: R(*params, y), (x_star,), (v,))[1]

        b_val = A_matvec(x_star) - R(*params, x_star)

        def solve_fn(matvec, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r):
                return solver.linear_solve(r, x_star=x_star, transpose=False)
            return jax.pure_callback(_cb, res_info, rhs, vmap_method="sequential")

        def trans_solve_fn(matvec, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r):
                return solver.linear_solve(r, x_star=x_star, transpose=True)
            return jax.pure_callback(_cb, res_info, rhs, vmap_method="sequential")

        is_symmetric = (
            solver_options.linear_solver_type in (
                LinearSolverType.JAX_CG_SCIPY,
                LinearSolverType.JAX_CG_SCIPY_W_INFO,
                LinearSolverType.JAX_CG_JAXOPT,
                LinearSolverType.JAX_CHOLESKY_JAXOPT,
                LinearSolverType.PETSC_CG,
            )
        )

        return jax.lax.custom_linear_solve(
            A_matvec,
            b_val,
            solve=solve_fn,
            transpose_solve=trans_solve_fn,
            symmetric=is_symmetric,
        )

    # Initialize solver if needed to populate options key
    params_stopped = [jax.lax.stop_gradient(p) for p in args]
    R_primal = jax.tree_util.Partial(R, *params_stopped)
    J_primal = None if J_x is None else jax.tree_util.Partial(J_x, *params_stopped)
    _, solver_options = build_solver_with_reuse(
        solver_options,
        R_primal,
        J_primal,
        x_0,
    )

    x_solution = _solve_core(*args)
    return x_solution, solver_options


def differentiable_linear_solve(
    solver_options: SolverOptions,
    A: Any,
    b: jnp.ndarray,
    transpose: bool = False
) -> tuple[jnp.ndarray, LinearSolverResultInfo]:
    """Direct linear solve interface compatible with autodiff."""
    if solver_options is None:
            raise ValueError(
                "solver_options is required for differentiable_linear_solve. "
                "Pass an explicit SolverOptions instance with linear_solve_type set. "
                "Example: SolverOptions(nonlinear_solver_type=..., linear_solve_type=...)"
            )
    return linear_solve(A, b, solver_options=solver_options, transpose=transpose)
    