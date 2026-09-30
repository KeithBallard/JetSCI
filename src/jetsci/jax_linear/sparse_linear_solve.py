from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Optional
import warnings
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import jax.lax as lax

from ..options import LinearSolverType, PreconditionerType, LinearSolverOptions, SolverOptions
from ..coo_data import COOData, to_coo_data
from .solve_cg import cg_w_info
from .preconditioners import build_preconditioner

# Optional imports
try:
    import jaxopt.linear_solve
    JAXOPT_AVAILABLE = True
except ImportError:
    JAXOPT_AVAILABLE = False

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    import cupyx.scipy.sparse.linalg as cplinalg
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import pypardiso
    PYPARDISO_AVAILABLE = True
except ImportError:
    PYPARDISO_AVAILABLE = False

try:
    import pyamgx
    PYAMGX_AVAILABLE = True
except ImportError:
    PYAMGX_AVAILABLE = False


@dataclass(frozen=True)
class LinearSolverResultInfo:
    iterations: int = 0
    residual_norm_history: jnp.ndarray | None = None


def _coo_matvec(rows, cols, vals, x):
    return jnp.zeros_like(x).at[rows].add(vals * x[cols])


def _coo_transpose_matvec(rows, cols, vals, x):
    return jnp.zeros_like(x).at[cols].add(vals * x[rows])


def _to_dense_matrix(A: Any, n: int) -> jnp.ndarray:
    if isinstance(A, jnp.ndarray):
        return A
    elif hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
        warnings.warn(
            f"Converting sparse matrix of type {type(A).__name__} with {len(A.vals)} nonzeros "
            f"to a dense {n}x{n} matrix. This allocates O(N^2) memory and will cause OOM or severe "
            f"slowdowns on large problems. Fix: Pass a sparse matrix to a sparse direct solver "
            f"(e.g., SPSOLVE_CUPY, SPSOLVE_PYPARDISO) or use an iterative solver.",
            UserWarning,
            stacklevel=3,
        )
        mat = jnp.zeros((n, n), dtype=A.vals.dtype)
        return mat.at[A.rows, A.cols].add(A.vals)
    elif isinstance(A, jsparse.COO):
        warnings.warn(
            f"Converting jax.experimental.sparse.COO matrix with {A.nse} nonzeros to a dense {n}x{n} "
            f"matrix via .todense(). This allocates O(N^2) memory and will cause OOM or severe "
            f"slowdowns on large problems. Fix: Pass a sparse matrix to a sparse direct solver "
            f"(e.g., SPSOLVE_CUPY, SPSOLVE_PYPARDISO) or use an iterative solver.",
            UserWarning,
            stacklevel=3,
        )
        return A.todense()
    elif callable(A):
        warnings.warn(
            f"Converting callable linear operator to a dense {n}x{n} matrix via jax.jacfwd. "
            f"This evaluates the operator {n} times and allocates O(N^2) memory, drastically "
            f"reducing performance. Fix: Use a matrix-free iterative solver (e.g., CG_JAX_SCIPY, "
            f"GMRES_JAX_SCIPY, BICGSTAB_JAX_SCIPY) instead.",
            UserWarning,
            stacklevel=3,
        )
        return jax.jacfwd(A)(jnp.zeros(n))
    else:
        return jnp.asarray(A)


def _to_matvec(A: Any, transpose: bool = False) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if callable(A):
        if not transpose:
            return A
        else:
            def matvec_t(v):
                dummy = jnp.zeros_like(v)
                _, vjp_fn = jax.vjp(A, dummy)
                return vjp_fn(v)[0]
            return matvec_t
    elif hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
        if not transpose:
            return lambda v: _coo_matvec(A.rows, A.cols, A.vals, v)
        else:
            return lambda v: _coo_transpose_matvec(A.rows, A.cols, A.vals, v)
    elif isinstance(A, jsparse.COO):
        if not transpose:
            return lambda v: _coo_matvec(A.row, A.col, A.data, v)
        else:
            return lambda v: _coo_transpose_matvec(A.row, A.col, A.data, v)
    elif isinstance(A, (jnp.ndarray, np.ndarray)):
        A_jax = jnp.asarray(A)
        if not transpose:
            return lambda v: A_jax @ v
        else:
            return lambda v: A_jax.T @ v
    else:
        raise TypeError(f"Cannot construct matvec from object of type {type(A)}")


# --- External Callbacks ---

def _pypardiso_solve_from_arrays(data, rows, cols, shape, b, transpose: bool):
    A_scipy = scipy.sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=shape,
    )
    A_scipy.sum_duplicates()
    if transpose:
        A_scipy = A_scipy.T.tocsr()
    b_np = np.asarray(b, dtype=np.float64)
    return pypardiso.spsolve(A_scipy, b_np)


def _pypardiso_solve_from_dense(A_dense, b, transpose: bool):
    A_scipy = scipy.sparse.csr_matrix(np.asarray(A_dense, dtype=np.float64))
    if transpose:
        A_scipy = A_scipy.T.tocsr()
    b_np = np.asarray(b, dtype=np.float64)
    return pypardiso.spsolve(A_scipy, b_np)


def _cupy_spsolve_from_arrays(data, rows, cols, shape, b, transpose: bool):
    if CUPY_AVAILABLE:
        try:
            A_cp = cpsparse.csr_matrix(
                (cp.asarray(data), (cp.asarray(rows), cp.asarray(cols))),
                shape=shape,
            )
            A_cp.sum_duplicates()
            A_cp.has_canonical_format = True
            if transpose:
                A_cp = A_cp.T.tocsr()
            b_cp = cp.asarray(b)
            x_cp = cplinalg.spsolve(A_cp, b_cp)
            return np.asarray(x_cp.get())
        except Exception as e:
            warnings.warn(
                f"CuPy spsolve failed with error: {e}. "
                "Falling back to CPU scipy.sparse.linalg.spsolve, which may significantly reduce performance.",
                UserWarning,
                stacklevel=2,
            )
    else:
        warnings.warn(
            "CuPy is not available for SPSOLVE_CUPY. Falling back to CPU scipy.sparse.linalg.spsolve. "
            "Install CuPy (e.g. pip install cupy-cuda12x) for GPU acceleration.",
            UserWarning,
            stacklevel=2,
        )

    A_scipy = scipy.sparse.csr_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=shape,
    )
    A_scipy.sum_duplicates()
    if transpose:
        A_scipy = A_scipy.T.tocsr()
    return scipy.sparse.linalg.spsolve(A_scipy, np.asarray(b))


def _cupy_spsolve_from_dense(A_dense, b, transpose: bool):
    if CUPY_AVAILABLE:
        try:
            A_cp = cpsparse.csr_matrix(cp.asarray(A_dense))
            if transpose:
                A_cp = A_cp.T.tocsr()
            b_cp = cp.asarray(b)
            x_cp = cplinalg.spsolve(A_cp, b_cp)
            return np.asarray(x_cp.get())
        except Exception as e:
            warnings.warn(
                f"CuPy spsolve from dense failed with error: {e}. "
                "Falling back to CPU scipy.sparse.linalg.spsolve, which may significantly reduce performance.",
                UserWarning,
                stacklevel=2,
            )
    else:
        warnings.warn(
            "CuPy is not available for SPSOLVE_CUPY. Falling back to CPU scipy.sparse.linalg.spsolve. "
            "Install CuPy (e.g. pip install cupy-cuda12x) for GPU acceleration.",
            UserWarning,
            stacklevel=2,
        )

    A_scipy = scipy.sparse.csr_matrix(np.asarray(A_dense))
    if transpose:
        A_scipy = A_scipy.T.tocsr()
    return scipy.sparse.linalg.spsolve(A_scipy, np.asarray(b))


def _cupy_splu_from_arrays(data, rows, cols, shape, b, transpose: bool):
    if CUPY_AVAILABLE:
        try:
            A_cp = cpsparse.csr_matrix(
                (cp.asarray(data), (cp.asarray(rows), cp.asarray(cols))),
                shape=shape,
            )
            A_cp.sum_duplicates()
            A_cp.has_canonical_format = True
            lu_cp = cplinalg.splu(A_cp)
            b_cp = cp.asarray(b)
            x_cp = lu_cp.solve(b_cp, trans="T" if transpose else "N")
            return np.asarray(x_cp.get())
        except Exception as e:
            warnings.warn(
                f"CuPy splu failed with error: {e}. "
                "Falling back to CPU scipy.sparse.linalg.splu, which may significantly reduce performance.",
                UserWarning,
                stacklevel=2,
            )
    else:
        warnings.warn(
            "CuPy is not available for LU_CUPY. Falling back to CPU scipy.sparse.linalg.splu. "
            "Install CuPy (e.g. pip install cupy-cuda12x) for GPU acceleration.",
            UserWarning,
            stacklevel=2,
        )

    A_scipy = scipy.sparse.csr_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(cols))),
        shape=shape,
    )
    A_scipy.sum_duplicates()
    lu_scipy = scipy.sparse.linalg.splu(A_scipy.tocsc())
    return lu_scipy.solve(np.asarray(b), trans="T" if transpose else "N")


def _cupy_splu_from_dense(A_dense, b, transpose: bool):
    if CUPY_AVAILABLE:
        try:
            A_cp = cpsparse.csr_matrix(cp.asarray(A_dense))
            lu_cp = cplinalg.splu(A_cp)
            b_cp = cp.asarray(b)
            x_cp = lu_cp.solve(b_cp, trans="T" if transpose else "N")
            return np.asarray(x_cp.get())
        except Exception as e:
            warnings.warn(
                f"CuPy splu from dense failed with error: {e}. "
                "Falling back to CPU scipy.sparse.linalg.splu, which may significantly reduce performance.",
                UserWarning,
                stacklevel=2,
            )
    else:
        warnings.warn(
            "CuPy is not available for LU_CUPY. Falling back to CPU scipy.sparse.linalg.splu. "
            "Install CuPy (e.g. pip install cupy-cuda12x) for GPU acceleration.",
            UserWarning,
            stacklevel=2,
        )

    A_scipy = scipy.sparse.csr_matrix(np.asarray(A_dense))
    lu_scipy = scipy.sparse.linalg.splu(A_scipy.tocsc())
    return lu_scipy.solve(np.asarray(b), trans="T" if transpose else "N")


def _get_operator_shape(A: Any) -> tuple[int, int] | None:
    if isinstance(A, tuple) and len(A) == 4:
        return (int(A[3][0]), int(A[3][1]))
    if hasattr(A, "shape"):
        return (int(A.shape[0]), int(A.shape[1]))
    return None


def _get_operator_sparsity_pattern(A: Any) -> Any:
    if isinstance(A, tuple) and len(A) == 4:
        return (np.asarray(A[0]), np.asarray(A[1]))
    if hasattr(A, "rows") and hasattr(A, "cols"):
        return (np.asarray(A.rows), np.asarray(A.cols))
    elif isinstance(A, jsparse.COO):
        return (np.asarray(A.row), np.asarray(A.col))
    return None


def linear_solve(
    A: Any,
    b: jnp.ndarray,
    solver_options: LinearSolverOptions | None = None,
    solver_type: LinearSolverType | None = None,
    precond_type: PreconditionerType | None = None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    x0: jnp.ndarray | None = None,
    transpose: bool = False,
) -> tuple[jnp.ndarray, LinearSolverResultInfo]:
    """Solve the linear system A x = b (or A^T x = b if transpose=True)."""
    if isinstance(A, tuple) and len(A) == 4:
        A = to_coo_data(A)
    n = b.shape[0]
    if x0 is None:
        x0 = jnp.zeros_like(b)

    if solver_type is None:
        if solver_options is not None and isinstance(solver_options.linear_solver_type, LinearSolverType):
            solver_type = solver_options.linear_solver_type
        else:
            raise ValueError(
                "No linear solver type specified in `linear_solve`. "
                "You must explicitly provide `solver_type` (e.g., `solver_type=LinearSolverType.JAX_CG_SCIPY`) "
                "or pass `solver_options=LinearSolverOptions(...)` with `linear_solver_type` set. "
                "Silently defaulting to a dense solver has been disabled to prevent hidden performance degradation."
            )

    if not solver_type.is_jax:
        raise TypeError(
            f"Expected a JAX LinearSolverType for JAX linear_solve, but got "
            f"solver_type={solver_type.name!r} (PETSc backend). "
            "Fix: Specify a JAX linear solver type (e.g., LinearSolverType.JAX_CG_SCIPY) "
            "or use differentiable_linear_solve with PETSc options."
        )

    if precond_type is None and solver_options is not None:
        if isinstance(solver_options.linear_preconditioner_type, PreconditionerType):
            precond_type = solver_options.linear_preconditioner_type

    rtol = solver_options.linear_relative_tol if solver_options is not None else 1e-10
    atol = solver_options.linear_absolute_tol if solver_options is not None else 1e-10
    maxiter = solver_options.linear_max_iter if solver_options is not None else 1000

    if preconditioner is None and precond_type is not None:
        preconditioner = build_preconditioner(precond_type, A=A, shape=(n, n))

    info = LinearSolverResultInfo()

    match solver_type:
        # --- JAX Native Solvers ---
        case LinearSolverType.JAX_DENSE_INVERSE_JNP:
            A_dense = _to_dense_matrix(A, n)
            A_eff = A_dense.T if transpose else A_dense
            x = jnp.linalg.solve(A_eff, b)
            return x, info

        case LinearSolverType.JAX_CG_SCIPY:
            A_matvec = _to_matvec(A, transpose=transpose)
            x, _ = jax.scipy.sparse.linalg.cg(
                A=A_matvec,
                b=b,
                x0=x0,
                M=preconditioner,
                tol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
            return x, info

        case LinearSolverType.JAX_CG_SCIPY_W_INFO:
            A_matvec = _to_matvec(A, transpose=transpose)
            x, cg_info = cg_w_info(
                A=A_matvec,
                b=b,
                x0=x0,
                M=preconditioner,
                tol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
            info = LinearSolverResultInfo(
                iterations=cg_info["iterations"],
                residual_norm_history=cg_info["residual_norm_history"],
            )
            return x, info

        case LinearSolverType.JAX_GMRES_SCIPY:
            A_matvec = _to_matvec(A, transpose=transpose)
            x, _ = jax.scipy.sparse.linalg.gmres(
                A=A_matvec,
                b=b,
                x0=x0,
                M=preconditioner,
                tol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
            return x, info

        case LinearSolverType.JAX_BICGSTAB_SCIPY:
            A_matvec = _to_matvec(A, transpose=transpose)
            x, _ = jax.scipy.sparse.linalg.bicgstab(
                A=A_matvec,
                b=b,
                x0=x0,
                M=preconditioner,
                tol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
            return x, info

        # --- JAXOpt Solvers ---
        case LinearSolverType.JAX_DENSE_INVERSE_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for JAX_DENSE_INVERSE_JAXOPT")
            A_matvec = _to_matvec(A, transpose=transpose)
            x = jaxopt.linear_solve.solve_inv(matvec=A_matvec, b=b)
            return x, info

        case LinearSolverType.JAX_LU_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for JAX_LU_JAXOPT")
            A_matvec = _to_matvec(A, transpose=transpose)
            x = jaxopt.linear_solve.solve_lu(matvec=A_matvec, b=b)
            return x, info

        case LinearSolverType.JAX_CHOLESKY_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for JAX_CHOLESKY_JAXOPT")
            A_matvec = _to_matvec(A, transpose=transpose)
            x = jaxopt.linear_solve.solve_cholesky(matvec=A_matvec, b=b)
            return x, info

        case LinearSolverType.JAX_CG_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for JAX_CG_JAXOPT")
            A_matvec = _to_matvec(A, transpose=transpose)
            x = jaxopt.linear_solve.solve_cg(
                matvec=A_matvec,
                b=b,
                init=x0,
                tol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
            return x, info

        case LinearSolverType.JAX_GMRES_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for JAX_GMRES_JAXOPT")
            A_matvec = _to_matvec(A, transpose=transpose)
            x = jaxopt.linear_solve.solve_gmres(
                matvec=A_matvec,
                b=b,
                init=x0,
                tol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
            return x, info

        case LinearSolverType.JAX_BICGSTAB_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for JAX_BICGSTAB_JAXOPT")
            A_matvec = _to_matvec(A, transpose=transpose)
            x = jaxopt.linear_solve.solve_bicgstab(
                matvec=A_matvec,
                b=b,
                init=x0,
                tol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
            return x, info

        # --- CuPy / SciPy Sparse Solvers ---
        case LinearSolverType.JAX_SPSOLVE_CUPY:
            res_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
            shape_static = (int(A.shape[0]), int(A.shape[1])) if hasattr(A, "shape") else (n, n)
            if hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
                x = jax.pure_callback(
                    lambda d, r, c, rhs: _cupy_spsolve_from_arrays(d, r, c, shape_static, rhs, transpose),
                    res_info,
                    A.vals, A.rows, A.cols, b,
                    vmap_method="sequential",
                )
            elif isinstance(A, jsparse.COO):
                x = jax.pure_callback(
                    lambda d, r, c, rhs: _cupy_spsolve_from_arrays(d, r, c, shape_static, rhs, transpose),
                    res_info,
                    A.data, A.row, A.col, b,
                    vmap_method="sequential",
                )
            elif isinstance(A, (jnp.ndarray, np.ndarray)):
                warnings.warn(
                    "Dense array passed to sparse direct solver JAX_SPSOLVE_CUPY. "
                    "Converting dense matrix to sparse CSR format on each solve, which degrades performance. "
                    "Fix: Pass a sparse COO matrix (e.g., jsparse.COO or COOData) or use a dense solver "
                    "(e.g., JAX_DENSE_INVERSE_JNP, JAX_LU_JAXOPT).",
                    UserWarning,
                    stacklevel=2,
                )
                A_dense = jnp.asarray(A)
                x = jax.pure_callback(
                    lambda a, rhs: _cupy_spsolve_from_dense(a, rhs, transpose),
                    res_info,
                    A_dense, b,
                    vmap_method="sequential",
                )
            elif callable(A):
                raise TypeError(
                    "JAX_SPSOLVE_CUPY is a sparse direct solver and cannot accept a callable operator. "
                    "Fix: Use a matrix-free iterative solver (e.g., JAX_CG_SCIPY, JAX_GMRES_SCIPY) "
                    "or provide an explicit sparse matrix (e.g., jsparse.COO or COOData)."
                )
            else:
                raise TypeError(
                    f"Unsupported matrix type {type(A)} for JAX_SPSOLVE_CUPY. "
                    "Expected jax.experimental.sparse.COO, COOData, or array."
                )
            return x, info

        case LinearSolverType.JAX_LU_CUPY:
            res_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
            shape_static = (int(A.shape[0]), int(A.shape[1])) if hasattr(A, "shape") else (n, n)
            if hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
                x = jax.pure_callback(
                    lambda d, r, c, rhs: _cupy_splu_from_arrays(d, r, c, shape_static, rhs, transpose),
                    res_info,
                    A.vals, A.rows, A.cols, b,
                    vmap_method="sequential",
                )
            elif isinstance(A, jsparse.COO):
                x = jax.pure_callback(
                    lambda d, r, c, rhs: _cupy_splu_from_arrays(d, r, c, shape_static, rhs, transpose),
                    res_info,
                    A.data, A.row, A.col, b,
                    vmap_method="sequential",
                )
            elif isinstance(A, (jnp.ndarray, np.ndarray)):
                warnings.warn(
                    "Dense array passed to sparse direct solver JAX_LU_CUPY. "
                    "Converting dense matrix to sparse CSR format on each solve, which degrades performance. "
                    "Fix: Pass a sparse COO matrix (e.g., jsparse.COO or COOData) or use a dense solver "
                    "(e.g., JAX_DENSE_INVERSE_JNP, JAX_LU_JAXOPT).",
                    UserWarning,
                    stacklevel=2,
                )
                A_dense = jnp.asarray(A)
                x = jax.pure_callback(
                    lambda a, rhs: _cupy_splu_from_dense(a, rhs, transpose),
                    res_info,
                    A_dense, b,
                    vmap_method="sequential",
                )
            elif callable(A):
                raise TypeError(
                    "JAX_LU_CUPY is a sparse direct solver and cannot accept a callable operator. "
                    "Fix: Use a matrix-free iterative solver (e.g., JAX_CG_SCIPY, JAX_GMRES_SCIPY) "
                    "or provide an explicit sparse matrix (e.g., jsparse.COO or COOData)."
                )
            else:
                raise TypeError(
                    f"Unsupported matrix type {type(A)} for JAX_LU_CUPY. "
                    "Expected jax.experimental.sparse.COO, COOData, or array."
                )
            return x, info

        # --- PyPardiso Solver ---
        case LinearSolverType.JAX_SPSOLVE_PYPARDISO:
            if not PYPARDISO_AVAILABLE:
                raise ImportError("pypardiso is required for JAX_SPSOLVE_PYPARDISO")
            res_info = jax.ShapeDtypeStruct(b.shape, b.dtype)
            shape_static = (int(A.shape[0]), int(A.shape[1])) if hasattr(A, "shape") else (n, n)
            if hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
                x = jax.pure_callback(
                    lambda d, r, c, rhs: _pypardiso_solve_from_arrays(d, r, c, shape_static, rhs, transpose),
                    res_info,
                    A.vals, A.rows, A.cols, b,
                    vmap_method="sequential",
                )
            elif isinstance(A, jsparse.COO):
                x = jax.pure_callback(
                    lambda d, r, c, rhs: _pypardiso_solve_from_arrays(d, r, c, shape_static, rhs, transpose),
                    res_info,
                    A.data, A.row, A.col, b,
                    vmap_method="sequential",
                )
            elif isinstance(A, (jnp.ndarray, np.ndarray)):
                warnings.warn(
                    "Dense array passed to sparse direct solver JAX_SPSOLVE_PYPARDISO. "
                    "Converting dense matrix to sparse CSR format on each solve, which degrades performance. "
                    "Fix: Pass a sparse COO matrix (e.g., jsparse.COO or COOData) or use a dense solver "
                    "(e.g., JAX_DENSE_INVERSE_JNP, JAX_LU_JAXOPT).",
                    UserWarning,
                    stacklevel=2,
                )
                A_dense = jnp.asarray(A)
                x = jax.pure_callback(
                    lambda a, rhs: _pypardiso_solve_from_dense(a, rhs, transpose),
                    res_info,
                    A_dense, b,
                    vmap_method="sequential",
                )
            elif callable(A):
                raise TypeError(
                    "JAX_SPSOLVE_PYPARDISO is a sparse direct solver and cannot accept a callable operator. "
                    "Fix: Use a matrix-free iterative solver (e.g., JAX_CG_SCIPY, JAX_GMRES_SCIPY) "
                    "or provide an explicit sparse matrix (e.g., jsparse.COO or COOData)."
                )
            else:
                raise TypeError(
                    f"Unsupported matrix type {type(A)} for JAX_SPSOLVE_PYPARDISO. "
                    "Expected jax.experimental.sparse.COO, COOData, or array."
                )
            return x, info

        # --- AMGX Solver ---
        case LinearSolverType.JAX_AMGX:
            if not PYAMGX_AVAILABLE:
                raise ImportError("pyamgx is required for AMGX solver")
            raise NotImplementedError("AMGX solver is not configured in this environment.")

        case _:
            raise NotImplementedError(f"Linear solver type {solver_type} is not implemented.")


@dataclass
class JAXLinearSolver:
    """JAX linear solver wrapping operator A, options, and preconditioner."""

    A: Any
    options: LinearSolverOptions
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    last_info: LinearSolverResultInfo | None = None
    _shape: tuple[int, int] | None = None
    _sparsity_pattern: Any = None

    def __post_init__(self):
        if isinstance(self.A, tuple) and len(self.A) == 4:
            self.A = to_coo_data(self.A)
        self._shape = _get_operator_shape(self.A)
        self._sparsity_pattern = _get_operator_sparsity_pattern(self.A)
        if self.preconditioner is None and self.options.linear_preconditioner_type is not PreconditionerType.JAX_NONE:
            shape = self._shape if self._shape is not None else None
            self.preconditioner = build_preconditioner(self.options.linear_preconditioner_type, A=self.A, shape=shape)

    def solve(self, b: jnp.ndarray, transpose: bool = False, x0: jnp.ndarray | None = None) -> jnp.ndarray:
        """Solve A x = b (or A^T x = b if transpose=True)."""
        x, info = linear_solve(
            self.A,
            b,
            solver_options=self.options,
            preconditioner=self.preconditioner,
            x0=x0,
            transpose=transpose,
        )
        self.last_info = info
        return x

    def update_operator(self, A: Any) -> JAXLinearSolver:
        """Update operator values, checking shape and sparsity pattern."""
        if isinstance(A, tuple) and len(A) == 4:
            A = to_coo_data(A)
        new_shape = _get_operator_shape(A)
        new_pattern = _get_operator_sparsity_pattern(A)

        shape_changed = (self._shape is not None and new_shape is not None and self._shape != new_shape)
        pattern_changed = False
        if self._sparsity_pattern is not None and new_pattern is not None:
            old_r, old_c = self._sparsity_pattern
            new_r, new_c = new_pattern
            if len(old_r) != len(new_r) or not (np.array_equal(old_r, new_r) and np.array_equal(old_c, new_c)):
                pattern_changed = True
        elif (self._sparsity_pattern is None) != (new_pattern is None):
            pattern_changed = True

        if shape_changed or pattern_changed:
            reason = "shape changed" if shape_changed else "sparsity pattern changed"
            warnings.warn(
                f"JAXLinearSolver operator {reason} during update_operator. "
                "Rebuilding preconditioner and solver metadata.",
                UserWarning,
                stacklevel=2,
            )
            self._shape = new_shape
            self._sparsity_pattern = new_pattern

        self.A = A
        if self.options.linear_preconditioner_type is not PreconditionerType.JAX_NONE:
            self.preconditioner = build_preconditioner(self.options.linear_preconditioner_type, A=self.A, shape=self._shape)
        return self

    def destroy(self) -> None:
        """Cleanup references."""
        self.A = None
        self.preconditioner = None

