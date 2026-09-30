from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class COOData:
    """JAX-visible COO matrix representation.

    Attributes:
        shape: Array or tuple containing the matrix dimensions (M, N).
        vals: Array of non-zero entry values.
        rows: Array of non-zero row indices.
        cols: Array of non-zero column indices.
    """

    shape: jax.Array | tuple[int, int]
    vals: jax.Array
    rows: jax.Array
    cols: jax.Array

    def tree_flatten(self):
        return (self.shape, self.vals, self.rows, self.cols), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def to_coo_data(mat: Any) -> COOData:
    """Normalize a COO representation into a canonical COOData object.

    Supports:
        - `COOData` instance (returned as-is)
        - Tuple `(rows, cols, vals, shape)`
        - `jax.experimental.sparse.COO`
    """
    if isinstance(mat, COOData):
        return mat
    elif isinstance(mat, tuple):
        if len(mat) == 4:
            rows, cols, vals, shape = mat
            return COOData(
                shape=jnp.asarray(shape, dtype=jnp.int64),
                vals=jnp.asarray(vals),
                rows=jnp.asarray(rows, dtype=jnp.int32),
                cols=jnp.asarray(cols, dtype=jnp.int32),
            )
        else:
            raise ValueError(
                f"Expected 4-element tuple (rows, cols, vals, shape) for COO triplet, "
                f"got tuple of length {len(mat)}."
            )
    elif isinstance(mat, jsparse.COO):
        return COOData(
            shape=jnp.asarray(mat.shape, dtype=jnp.int64),
            vals=mat.data,
            rows=jnp.asarray(mat.row, dtype=jnp.int32),
            cols=jnp.asarray(mat.col, dtype=jnp.int32),
        )
    elif hasattr(mat, "rows") and hasattr(mat, "cols") and hasattr(mat, "vals") and hasattr(mat, "shape"):
        return COOData(
            shape=jnp.asarray(mat.shape, dtype=jnp.int64),
            vals=mat.vals,
            rows=jnp.asarray(mat.rows, dtype=jnp.int32),
            cols=jnp.asarray(mat.cols, dtype=jnp.int32),
        )
    else:
        raise TypeError(
            f"Cannot convert object of type {type(mat)} to COOData. "
            "Expected COOData, (rows, cols, vals, shape) tuple, or jax.experimental.sparse.COO."
        )
