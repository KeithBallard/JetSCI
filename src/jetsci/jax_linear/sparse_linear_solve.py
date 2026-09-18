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

from ..options import JAXLinearSolverType, JAXPreconditionerType, SolverOptions
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


def linear_solve(
    A: Any,
    b: jnp.ndarray,
    solver_options: SolverOptions | None = None,
    solver_type: JAXLinearSolverType | None = None,
    precond_type: JAXPreconditionerType | None = None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    x0: jnp.ndarray | None = None,
    transpose: bool = False,
) -> tuple[jnp.ndarray, LinearSolverResultInfo]:
    """Solve the linear system A x = b (or A^T x = b if transpose=True)."""
    n = b.shape[0]
    if x0 is None:
        x0 = jnp.zeros_like(b)

    if solver_type is None:
        if solver_options is not None and isinstance(solver_options.linear_solve_type, JAXLinearSolverType):
            solver_type = solver_options.linear_solve_type
        elif solver_options is not None and hasattr(solver_options, "linear_solve_type"):
            raise TypeError(
                f"Expected JAXLinearSolverType for JAX linear_solve, but solver_options has "
                f"linear_solve_type={solver_options.linear_solve_type!r}. "
                "Fix: Specify a JAXLinearSolverType (e.g., JAXLinearSolverType.CG_JAX_SCIPY)."
            )
        else:
            raise ValueError(
                "No linear solver type specified in `linear_solve`. "
                "You must explicitly provide `solver_type` (e.g., `solver_type=JAXLinearSolverType.CG_JAX_SCIPY`) "
                "or pass `solver_options=SolverOptions(...)` with `linear_solve_type` set. "
                "Silently defaulting to a dense solver has been disabled to prevent hidden performance degradation."
            )

    if precond_type is None and solver_options is not None:
        if isinstance(solver_options.linear_precond_type, JAXPreconditionerType):
            precond_type = solver_options.linear_precond_type

    rtol = solver_options.linear_relative_tol if solver_options is not None else 1e-10
    atol = solver_options.linear_absolute_tol if solver_options is not None else 1e-10
    maxiter = solver_options.linear_max_iter if solver_options is not None else 1000

    if preconditioner is None and precond_type is not None:
        preconditioner = build_preconditioner(precond_type, A=A, shape=(n, n))

    info = LinearSolverResultInfo()

    match solver_type:
        # --- JAX Native Solvers ---
        case JAXLinearSolverType.DENSE_INVERSE_JNP:
            A_dense = _to_dense_matrix(A, n)
            A_eff = A_dense.T if transpose else A_dense
            x = jnp.linalg.solve(A_eff, b)
            return x, info

        case JAXLinearSolverType.CG_JAX_SCIPY:
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

        case JAXLinearSolverType.CG_JAX_SCIPY_W_INFO:
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

        case JAXLinearSolverType.GMRES_JAX_SCIPY:
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

        case JAXLinearSolverType.BICGSTAB_JAX_SCIPY:
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
        case JAXLinearSolverType.DENSE_INVERSE_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for DENSE_INVERSE_JAXOPT")
            A_matvec = _to_matvec(A, transpose=transpose)
            x = jaxopt.linear_solve.solve_inv(matvec=A_matvec, b=b)
            return x, info

        case JAXLinearSolverType.LU_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for LU_JAXOPT")
            A_matvec = _to_matvec(A, transpose=transpose)
            x = jaxopt.linear_solve.solve_lu(matvec=A_matvec, b=b)
            return x, info

        case JAXLinearSolverType.CHOLESKY_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for CHOLESKY_JAXOPT")
            A_matvec = _to_matvec(A, transpose=transpose)
            x = jaxopt.linear_solve.solve_cholesky(matvec=A_matvec, b=b)
            return x, info

        case JAXLinearSolverType.CG_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for CG_JAXOPT")
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

        case JAXLinearSolverType.GMRES_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for GMRES_JAXOPT")
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

        case JAXLinearSolverType.BICGSTAB_JAXOPT:
            if not JAXOPT_AVAILABLE:
                raise ImportError("jaxopt is required for BICGSTAB_JAXOPT")
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
        case JAXLinearSolverType.SPSOLVE_CUPY:
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
                    "Dense array passed to sparse direct solver SPSOLVE_CUPY. "
                    "Converting dense matrix to sparse CSR format on each solve, which degrades performance. "
                    "Fix: Pass a sparse COO matrix (e.g., jsparse.COO or COOData) or use a dense solver "
                    "(e.g., DENSE_INVERSE_JNP, LU_JAXOPT).",
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
                    "SPSOLVE_CUPY is a sparse direct solver and cannot accept a callable operator. "
                    "Fix: Use a matrix-free iterative solver (e.g., CG_JAX_SCIPY, GMRES_JAX_SCIPY) "
                    "or provide an explicit sparse matrix (e.g., jsparse.COO or COOData)."
                )
            else:
                raise TypeError(
                    f"Unsupported matrix type {type(A)} for SPSOLVE_CUPY. "
                    "Expected jax.experimental.sparse.COO, COOData, or array."
                )
            return x, info

        case JAXLinearSolverType.LU_CUPY:
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
                    "Dense array passed to sparse direct solver LU_CUPY. "
                    "Converting dense matrix to sparse CSR format on each solve, which degrades performance. "
                    "Fix: Pass a sparse COO matrix (e.g., jsparse.COO or COOData) or use a dense solver "
                    "(e.g., DENSE_INVERSE_JNP, LU_JAXOPT).",
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
                    "LU_CUPY is a sparse direct solver and cannot accept a callable operator. "
                    "Fix: Use a matrix-free iterative solver (e.g., CG_JAX_SCIPY, GMRES_JAX_SCIPY) "
                    "or provide an explicit sparse matrix (e.g., jsparse.COO or COOData)."
                )
            else:
                raise TypeError(
                    f"Unsupported matrix type {type(A)} for LU_CUPY. "
                    "Expected jax.experimental.sparse.COO, COOData, or array."
                )
            return x, info

        # --- PyPardiso Solver ---
        case JAXLinearSolverType.SPSOLVE_PYPARDISO:
            if not PYPARDISO_AVAILABLE:
                raise ImportError("pypardiso is required for SPSOLVE_PYPARDISO")
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
                    "Dense array passed to sparse direct solver SPSOLVE_PYPARDISO. "
                    "Converting dense matrix to sparse CSR format on each solve, which degrades performance. "
                    "Fix: Pass a sparse COO matrix (e.g., jsparse.COO or COOData) or use a dense solver "
                    "(e.g., DENSE_INVERSE_JNP, LU_JAXOPT).",
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
                    "SPSOLVE_PYPARDISO is a sparse direct solver and cannot accept a callable operator. "
                    "Fix: Use a matrix-free iterative solver (e.g., CG_JAX_SCIPY, GMRES_JAX_SCIPY) "
                    "or provide an explicit sparse matrix (e.g., jsparse.COO or COOData)."
                )
            else:
                raise TypeError(
                    f"Unsupported matrix type {type(A)} for SPSOLVE_PYPARDISO. "
                    "Expected jax.experimental.sparse.COO, COOData, or array."
                )
            return x, info

        # --- AMGX Solver ---
        case JAXLinearSolverType.AMGX:
            if not PYAMGX_AVAILABLE:
                raise ImportError("pyamgx is required for AMGX solver")
            raise NotImplementedError("AMGX solver is not configured in this environment.")

        case _:
            raise NotImplementedError(f"Linear solver type {solver_type} is not implemented.")
