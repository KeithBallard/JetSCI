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
from .coo_data import COOData, to_coo_data
from .jax_linear import linear_solve, LinearSolverResultInfo


def linear_solver(
    solver_options: LinearSolverOptions | SolverOptions,
    A: Any,
    b: jnp.ndarray,
    x_0: jnp.ndarray | None = None,
    transpose: bool = False,
) -> tuple[jnp.ndarray, LinearSolverOptions | SolverOptions, LinearSolverResultInfo]:
    """Solve linear system A x = b (or A^T x = b if transpose=True) using JAX or PETSc backend.

    Parameters:
        solver_options: LinearSolverOptions or SolverOptions specifying solver backend,
            algorithm, tolerances, and optional solver_key.
        A: Linear operator as a callable matvec (v -> A @ v), COOData,
            (rows, cols, vals, shape) tuple, or dense jnp.ndarray.
        b: 1D array representing the right-hand side.
        x_0: Optional initial guess array (for iterative solvers).
        transpose: Whether to solve the transposed linear system.

    Returns:
        tuple (x, updated_solver_options, info) where:
            - x is the solution JAX array.
            - updated_solver_options has solver_key populated for reuse in subsequent solves.
            - info is LinearSolverResultInfo containing solver diagnostics.
    """
    if solver_options is None:
        raise ValueError(
            "solver_options is required for linear_solver. "
            "Pass an explicit LinearSolverOptions or SolverOptions instance. "
            "Example: LinearSolverOptions(linear_solver_type=LinearSolverType.JAX_CG_SCIPY, "
            "linear_preconditioner_type=PreconditionerType.JAX_NONE)"
        )

    b = jnp.asarray(b)

    if isinstance(A, tuple) and len(A) == 4:
        A = to_coo_data(A)

    if callable(A) and solver_options.linear_solver_type.is_petsc:
        raise TypeError(
            "PETSc linear solver cannot accept a callable operator. "
            "Callable operators cannot be converted to sparse matrices without evaluation loops (like jacfwd), "
            "which destroys performance. "
            "Fix: Use a matrix-free iterative solver (e.g., LinearSolverType.JAX_CG_SCIPY, LinearSolverType.JAX_GMRES_SCIPY) "
            "or provide an explicit sparse matrix (e.g., COOData or (rows, cols, vals, shape))."
        )

    solver, updated_solver_options = build_linear_solver_with_reuse(solver_options, A)
    x = solver.solve(b, transpose=transpose, x0=x_0)
    info = solver.last_info if hasattr(solver, "last_info") and solver.last_info is not None else LinearSolverResultInfo()
    return x, updated_solver_options, info


def differentiable_linear_solve(
    solver_options: LinearSolverOptions | SolverOptions,
    A: Callable,
    b: Callable,
    x_linearized: jnp.ndarray,
    *args,
    x_0: jnp.ndarray | None = None,
    transpose: bool = False,
) -> tuple[jnp.ndarray, LinearSolverOptions | SolverOptions, LinearSolverResultInfo]:
    """Solve linear system A(x_linearized, *args) x = b(x_linearized, *args) with autodiff.

    Uses Implicit Function Theorem (IFT) differentiation via `jax.lax.custom_linear_solve`
    so that autodiff (forward-mode jvp/jacfwd and reverse-mode vjp/grad/jacrev) operates via
    adjoints without unrolling solver iterations. Compatible with both JAX and PETSc backends.

    Parameters:
        solver_options: LinearSolverOptions for a standalone linear solver, or
            SolverOptions with a solver_key from an already-built nonlinear
            solver. Both sources use the same ``linear_solve`` protocol.
        A: Callable with signature A(x, *args) returning a callable matvec (v -> A @ v),
            COOData, (rows, cols, vals, shape) tuple, or dense array.
        b: Callable with signature b(x, *args) returning a 1D RHS array.
        x_linearized: The point of linearization passed to A and b.
        *args: Additional parameter tensors/values that A and b depend on.
        x_0: Optional initial guess array (defaults to zeros like x_linearized).
        transpose: Whether to solve the transposed linear system.

    Returns:
        tuple (x, updated_solver_options, info). With SolverOptions, the
        supplied A/b must describe the nonlinear solver's current
        linearization; the existing nonlinear solver owns the native KSP.
    """
    if solver_options is None:
        raise ValueError(
            "solver_options is required for differentiable_linear_solve. "
            "Pass an explicit LinearSolverOptions or SolverOptions instance. "
            "Example: LinearSolverOptions(linear_solver_type=LinearSolverType.JAX_CG_SCIPY, "
            "linear_preconditioner_type=PreconditionerType.JAX_NONE)"
        )

    if not callable(A):
        raise TypeError(
            f"Expected A to be a callable with signature A(x, *args), but got {type(A)}. "
            "For non-differentiable solves with static matrices, use `jetsci.linear_solver` "
            "or wrap A as `lambda x, *args: A`."
        )
    if not callable(b):
        raise TypeError(
            f"Expected b to be a callable with signature b(x, *args), but got {type(b)}. "
            "For non-differentiable solves with static vectors, use `jetsci.linear_solver` "
            "or wrap b as `lambda x, *args: b`."
        )

    x_linearized = jnp.asarray(x_linearized)
    if x_0 is None:
        x_0 = jnp.zeros_like(x_linearized)
    else:
        x_0 = jnp.asarray(x_0)

    # Initialize / retrieve solver with stopped gradients to populate solver_key
    x_lin_stopped = jax.lax.stop_gradient(x_linearized)
    params_stopped = [jax.lax.stop_gradient(p) for p in args]
    A_primal = A(x_lin_stopped, *params_stopped)
    if isinstance(A_primal, tuple) and len(A_primal) == 4:
        A_primal = to_coo_data(A_primal)


    #Reconsider if there's another way of doing this
    if callable(A_primal) and solver_options.linear_solver_type.is_petsc:
        raise TypeError(
            "PETSc linear solver cannot accept a callable operator. "
            "Callable operators cannot be converted to sparse matrices without evaluation loops (like jacfwd), "
            "whichdestroys performance. "
            "Fix: Use a matrix-free iterative solver (e.g., LinearSolverType.JAX_CG_SCIPY, LinearSolverType.JAX_GMRES_SCIPY) "
            "or provide an explicit sparse matrix (e.g., COOData or (rows, cols, vals, shape))."
        )

    if isinstance(solver_options, SolverOptions):
        if solver_options.solver_key is None:
            raise TypeError(
                "differentiable_linear_solve with SolverOptions requires an "
                "already-built nonlinear solver (solver_key is missing). Use "
                "LinearSolverOptions to create a standalone KSP/JAX solver."
            )
        if solver_options.nonlinear_solver_type.is_petsc:
            from .petsc_snes.solver_lifecycle import get_petsc_solver_objects_from_key

            solver, _ = get_petsc_solver_objects_from_key(solver_options.solver_key)
        else:
            from .jax_newton_raph.solver import get_jax_solver_from_key

            solver = get_jax_solver_from_key(solver_options.solver_key)
        updated_solver_options = solver_options
    else:
        solver, updated_solver_options = build_linear_solver_with_reuse(
            solver_options, A_primal
        )

    def _solve_core(x_lin_eval, *params_eval):
        A_val = A(x_lin_eval, *params_eval)
        if isinstance(A_val, tuple) and len(A_val) == 4:
            A_val = to_coo_data(A_val)
        b_val = b(x_lin_eval, *params_eval)

        if callable(A_val):
            if not transpose:
                A_matvec = A_val
            else:
                def A_matvec(v):
                    dummy = jnp.zeros_like(v)
                    _, vjp_fn = jax.vjp(A_val, dummy)
                    return vjp_fn(v)[0]
        elif isinstance(A_val, COOData) or (hasattr(A_val, "rows") and hasattr(A_val, "cols") and hasattr(A_val, "vals")):
            if not transpose:
                A_matvec = lambda v: jnp.zeros(int(A_val.shape[0]), dtype=v.dtype).at[A_val.rows].add(A_val.vals * v[A_val.cols])
            else:
                A_matvec = lambda v: jnp.zeros(int(A_val.shape[1]), dtype=v.dtype).at[A_val.cols].add(A_val.vals * v[A_val.rows])
        elif isinstance(A_val, jsparse.COO):
            if not transpose:
                A_matvec = lambda v: jnp.zeros(int(A_val.shape[0]), dtype=v.dtype).at[A_val.row].add(A_val.data * v[A_val.col])
            else:
                A_matvec = lambda v: jnp.zeros(int(A_val.shape[1]), dtype=v.dtype).at[A_val.col].add(A_val.data * v[A_val.row])
        else:
            A_dense = jnp.asarray(A_val)
            if not transpose:
                A_matvec = lambda v: A_dense @ v
            else:
                A_matvec = lambda v: A_dense.T @ v

        #TODO: These matvec_fn aren't being used, nor is the vmap looking quite right
        def solve_fn(matvec_fn, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r, x_linearized_cb):
                return solver.linear_solve(
                    r,
                    x_linearized=x_linearized_cb,
                    x_0=x_0,
                    transpose=transpose,
                )
            return jax.pure_callback(
                _cb, res_info, rhs, x_lin_eval, vmap_method="sequential"
            )

        def trans_solve_fn(matvec_fn, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r, x_linearized_cb):
                return solver.linear_solve(
                    r,
                    x_linearized=x_linearized_cb,
                    x_0=x_0,
                    transpose=not transpose,
                )
            return jax.pure_callback(
                _cb, res_info, rhs, x_lin_eval, vmap_method="sequential"
            )

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

    x_solution = _solve_core(x_linearized, *args)
    info = solver.last_info if hasattr(solver, "last_info") and solver.last_info is not None else LinearSolverResultInfo()
    return x_solution, updated_solver_options, info


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
                return solver.linear_solve(
                    r, x_linearized=x_star, transpose=False
                )
            return jax.pure_callback(_cb, res_info, rhs, vmap_method="sequential")

        def trans_solve_fn(matvec, rhs):
            res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
            def _cb(r):
                return solver.linear_solve(
                    r, x_linearized=x_star, transpose=True
                )
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
