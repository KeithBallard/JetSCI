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
    solver_options: LinearSolverOptions,
    A: Any,
    b: jnp.ndarray,
    transpose: bool = False,
) -> tuple[jnp.ndarray, LinearSolverOptions]:
    """Solve linear system A x = b (or A^T x = b if transpose=True) with forward and reverse autodiff.

    Uses Implicit Function Theorem (IFT) differentiation via `jax.lax.custom_linear_solve`
    so that autodiff operates via adjoints without unrolling solver iterations.
    """
    if solver_options is None:
        raise ValueError(
            "solver_options is required for differentiable_linear_solve. "
            "Pass an explicit LinearSolverOptions or SolverOptions instance. "
            "Example: LinearSolverOptions(linear_solver_type=LinearSolverType.JAX_CG_SCIPY, "
            "linear_preconditioner_type=PreconditionerType.JAX_NONE)"
        )

    b = jnp.asarray(b)

    is_tracer = (
        isinstance(A, jax.core.Tracer)
        or (hasattr(A, "data") and isinstance(A.data, jax.core.Tracer))
        or (hasattr(A, "vals") and isinstance(A.vals, jax.core.Tracer))
    )

    if not is_tracer:
        solver, solver_options = build_linear_solver_with_reuse(solver_options, A)
    else:
        if solver_options.solver_key is not None and solver_options.solver_key in __linear_solver_dict:
            solver = __linear_solver_dict[solver_options.solver_key]
        else:
            if isinstance(A, (jnp.ndarray, np.ndarray)):
                A_init = np.zeros(A.shape, dtype=A.dtype)
            elif isinstance(A, jsparse.COO):
                A_init = jsparse.COO((np.zeros(A.data.shape, dtype=A.data.dtype), np.asarray(A.row), np.asarray(A.col)), shape=A.shape)
            elif hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
                A_init = COOData(shape=np.asarray(A.shape), vals=np.zeros(A.vals.shape, dtype=A.vals.dtype), rows=np.asarray(A.rows), cols=np.asarray(A.cols))
            else:
                A_init = A
            solver, solver_options = build_linear_solver_with_reuse(solver_options, A_init)

    # Construct differentiable matvec representation
    if callable(A):
        if transpose:
            matvec = lambda v: jax.vjp(A, jnp.zeros_like(b))[1](v)[0]
        else:
            matvec = A
    elif isinstance(A, (jnp.ndarray, np.ndarray)):
        A_arr = jnp.asarray(A)
        matvec = (lambda v: A_arr.T @ v) if transpose else (lambda v: A_arr @ v)
    elif isinstance(A, jsparse.COO):
        if transpose:
            matvec = lambda v: jnp.zeros(A.shape[1], dtype=v.dtype).at[A.col].add(A.data * v[A.row])
        else:
            matvec = lambda v: jnp.zeros(A.shape[0], dtype=v.dtype).at[A.row].add(A.data * v[A.col])
    elif hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
        if transpose:
            matvec = lambda v: jnp.zeros(int(A.shape[1]), dtype=v.dtype).at[A.cols].add(A.vals * v[A.rows])
        else:
            matvec = lambda v: jnp.zeros(int(A.shape[0]), dtype=v.dtype).at[A.rows].add(A.vals * v[A.cols])
    else:
        raise TypeError(f"Unsupported matrix type {type(A)} for differentiable_linear_solve.")

    if isinstance(A, (jnp.ndarray, np.ndarray)):
        def solve_fn(mv, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r, a_mat):
                solver.update_operator(a_mat)
                return solver.solve(r, transpose=transpose)
            return jax.pure_callback(_cb, res_info, rhs, A, vmap_method="sequential")

        def trans_solve_fn(mv, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r, a_mat):
                solver.update_operator(a_mat)
                return solver.solve(r, transpose=not transpose)
            return jax.pure_callback(_cb, res_info, rhs, A, vmap_method="sequential")
    elif isinstance(A, jsparse.COO):
        def solve_fn(mv, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r, data):
                a_coo = jsparse.COO((data, A.row, A.col), shape=A.shape)
                solver.update_operator(a_coo)
                return solver.solve(r, transpose=transpose)
            return jax.pure_callback(_cb, res_info, rhs, A.data, vmap_method="sequential")

        def trans_solve_fn(mv, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r, data):
                a_coo = jsparse.COO((data, A.row, A.col), shape=A.shape)
                solver.update_operator(a_coo)
                return solver.solve(r, transpose=not transpose)
            return jax.pure_callback(_cb, res_info, rhs, A.data, vmap_method="sequential")
    elif hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
        def solve_fn(mv, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r, vals):
                a_coo = COOData(shape=A.shape, vals=vals, rows=A.rows, cols=A.cols)
                solver.update_operator(a_coo)
                return solver.solve(r, transpose=transpose)
            return jax.pure_callback(_cb, res_info, rhs, A.vals, vmap_method="sequential")

        def trans_solve_fn(mv, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r, vals):
                a_coo = COOData(shape=A.shape, vals=vals, rows=A.rows, cols=A.cols)
                solver.update_operator(a_coo)
                return solver.solve(r, transpose=not transpose)
            return jax.pure_callback(_cb, res_info, rhs, A.vals, vmap_method="sequential")
    else:
        def solve_fn(mv, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r):
                return solver.solve(r, transpose=transpose)
            return jax.pure_callback(_cb, res_info, rhs, vmap_method="sequential")

        def trans_solve_fn(mv, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r):
                return solver.solve(r, transpose=not transpose)
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

    x = jax.lax.custom_linear_solve(
        matvec,
        b,
        solve=solve_fn,
        transpose_solve=trans_solve_fn,
        symmetric=is_symmetric,
    )
    return x, solver_options
