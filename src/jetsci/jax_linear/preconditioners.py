from __future__ import annotations

from typing import Callable, Any
import warnings
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse

from ..options import JAXPreconditionerType

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    import cupyx.scipy.sparse.linalg as cplinalg
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def build_preconditioner(
    precond_type: JAXPreconditionerType,
    A: Any = None,
    diag: jnp.ndarray | None = None,
    shape: tuple[int, int] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray] | None:
    """Build a preconditioner callable M(r) -> z based on precond_type."""
    if precond_type is JAXPreconditionerType.NONE or precond_type is None:
        return None

    if precond_type is JAXPreconditionerType.JACOBI:
        if diag is not None:
            safe_diag = jnp.where(jnp.abs(diag) < 1e-15, 1.0, diag)
            return lambda r: r / safe_diag

        if A is not None:
            if isinstance(A, jnp.ndarray):
                d = jnp.diag(A)
                safe_diag = jnp.where(jnp.abs(d) < 1e-15, 1.0, d)
                return lambda r: r / safe_diag
            elif hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
                # COOData or similar
                n = int(A.shape[0]) if hasattr(A.shape[0], "__int__") else len(A.rows)
                diag_vals = jnp.where(A.rows == A.cols, A.vals, 0.0)
                d = jnp.zeros(n, dtype=A.vals.dtype).at[A.rows].add(diag_vals)
                safe_diag = jnp.where(jnp.abs(d) < 1e-15, 1.0, d)
                return lambda r: r / safe_diag
            elif isinstance(A, (jsparse.COO,)):
                diag_vals = jnp.where(A.row == A.col, A.data, 0.0)
                d = jnp.zeros(A.shape[0], dtype=A.data.dtype).at[A.row].add(diag_vals)
                safe_diag = jnp.where(jnp.abs(d) < 1e-15, 1.0, d)
                return lambda r: r / safe_diag
            else:
                raise TypeError(
                    f"Cannot extract diagonal for JACOBI preconditioner from operator of type {type(A)}. "
                    "Jacobi preconditioning requires explicit diagonal entries. "
                    "Fix: Provide diagonal entries explicitly via `diag=...`, pass a sparse matrix "
                    "(e.g., jsparse.COO or COOData), or set preconditioning to JAXPreconditionerType.NONE."
                )

        raise ValueError(
            "JACOBI preconditioner requires either `diag` or `A` to extract diagonal elements. "
            "Neither was provided. Fix: Pass diag=... or A=..., or set preconditioning to JAXPreconditionerType.NONE."
        )

    if precond_type is JAXPreconditionerType.ILU_CUPY:
        if A is None:
            raise ValueError(
                "ILU_CUPY preconditioner requires a matrix A. "
                "Fix: Pass a sparse matrix (e.g. jsparse.COO or COOData) to the solver."
            )

        if callable(A):
            raise TypeError(
                f"ILU_CUPY preconditioner requires an explicit sparse matrix representation, got callable {type(A)}. "
                "Fix: Pass a sparse matrix (e.g. jsparse.COO or COOData) or use JAXPreconditionerType.NONE."
            )

        # Convert A to CSR format
        if hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
            data = np.asarray(A.vals)
            rows = np.asarray(A.rows)
            cols = np.asarray(A.cols)
            sh = tuple(int(s) for s in A.shape)
        elif isinstance(A, jsparse.COO):
            data = np.asarray(A.data)
            rows = np.asarray(A.row)
            cols = np.asarray(A.col)
            sh = A.shape
        elif isinstance(A, (jnp.ndarray, np.ndarray)):
            warnings.warn(
                "Dense matrix passed to ILU_CUPY preconditioner. Converting dense matrix to sparse CSR format. "
                "Fix: Pass a sparse matrix representation (e.g. jsparse.COO) directly to avoid conversion overhead.",
                UserWarning,
                stacklevel=2,
            )
            A_sp = scipy.sparse.csr_matrix(np.asarray(A))
            data, rows, cols, sh = A_sp.data, A_sp.nonzero()[0], A_sp.nonzero()[1], A_sp.shape
        elif isinstance(A, scipy.sparse.spmatrix):
            A_coo = A.tocoo()
            data, rows, cols, sh = A_coo.data, A_coo.row, A_coo.col, A_coo.shape
        else:
            raise TypeError(
                f"Unsupported matrix type {type(A)} for ILU preconditioner. "
                "Expected jsparse.COO, COOData, or scipy.sparse matrix."
            )

        if CUPY_AVAILABLE:
            try:
                A_cp = cpsparse.csr_matrix((cp.asarray(data), (cp.asarray(rows), cp.asarray(cols))), shape=sh)
                A_cp.sum_duplicates()
                A_cp.has_canonical_format = True
                ilu_cp = cplinalg.spilu(A_cp, fill_factor=1.0)

                def _cupy_ilu_solve(r_np):
                    r_cp = cp.asarray(r_np)
                    z_cp = ilu_cp.solve(r_cp)
                    return np.asarray(z_cp.get())

                def ilu_precond(r):
                    res_info = jax.ShapeDtypeStruct(r.shape, r.dtype)
                    return jax.pure_callback(_cupy_ilu_solve, res_info, r, vmap_method="sequential")

                return ilu_precond
            except Exception as e:
                warnings.warn(
                    f"CuPy ILU preconditioner failed with error: {e}. "
                    "Falling back to CPU scipy.sparse.linalg.spilu, which may reduce performance.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "CuPy is not available for ILU_CUPY. Falling back to CPU scipy.sparse.linalg.spilu. "
                "Install CuPy for GPU-accelerated ILU preconditioning.",
                UserWarning,
                stacklevel=2,
            )

        # SciPy fallback
        A_scipy = scipy.sparse.csr_matrix((data, (rows, cols)), shape=sh)
        A_scipy.sum_duplicates()
        ilu_scipy = scipy.sparse.linalg.spilu(A_scipy.tocsc(), fill_factor=1.0)

        def _scipy_ilu_solve(r_np):
            return ilu_scipy.solve(np.asarray(r_np))

        def ilu_precond(r):
            res_info = jax.ShapeDtypeStruct(r.shape, r.dtype)
            return jax.pure_callback(_scipy_ilu_solve, res_info, r, vmap_method="sequential")

        return ilu_precond

    raise NotImplementedError(f"Preconditioner {precond_type} is not implemented.")
