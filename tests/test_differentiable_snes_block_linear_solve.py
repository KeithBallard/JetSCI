"""GPU regression test for the differentiable-SNES multi-RHS KSP bridge.

This intentionally exercises the older ``differentiable_snes`` linear hook.
Its multi-RHS contract follows PETSc ``KSP.matSolve``: a dense RHS matrix has
shape ``(n_dofs, n_rhs)`` and each RHS occupies one column.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from jetsci.coo_data import COOData
from jetsci.options import (
    LinearSolverType,
    NonlinearSolverType,
    PreconditionerType,
    SolverOptions,
)
from jetsci.petsc_snes import differentiable_snes
from jetsci.petsc_snes.solver_lifecycle import (
    build_petsc_solver_with_reuse,
    destroy_petsc_solver,
)


jax.config.update("jax_enable_x64", True)


def test_differentiable_snes_block_linear_solve_matches_jax():
    """Compare one and block-RHS solves against JAX and an exact solution.

    Run this only in the CUDA PETSc environment.  A failure in the batched
    assertion identifies the multi-RHS bridge independently of nonlinear FEA
    assembly or implicit differentiation.
    """
    matrix = jnp.array(
        [[6.0, 1.0, 0.5], [1.0, 5.0, 1.0], [0.5, 1.0, 4.0]],
        dtype=jnp.float64,
    )
    rows, cols = jnp.nonzero(matrix)
    coo = COOData(
        shape=jnp.array(matrix.shape, dtype=jnp.int64),
        vals=matrix[rows, cols],
        rows=rows.astype(jnp.int32),
        cols=cols.astype(jnp.int32),
    )

    def residual(phi, x):
        return matrix @ x - phi

    def jacobian(phi, x):
        del phi, x
        return coo

    options = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
        linear_solver_type=LinearSolverType.PETSC_CG,
        linear_preconditioner_type=PreconditionerType.PETSC_JACOBI,
        nonlinear_relative_tol=1e-12,
        nonlinear_absolute_tol=1e-12,
        linear_relative_tol=1e-12,
        linear_absolute_tol=1e-12,
        linear_max_iter=100,
    )
    phi = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float64)
    x0 = jnp.zeros_like(phi)
    solver, options = build_petsc_solver_with_reuse(
        options,
        jax.tree_util.Partial(residual, phi),
        jax.tree_util.Partial(jacobian, phi),
        x0,
    )

    try:
        # Assemble the SNES Jacobian before pointing the companion KSP at it.
        x_star = solver.solve_to_jax(x0)
        primitive = differentiable_snes.DifferentiableSNESPrimitive(
            residual=residual,
            jacobian=jacobian,
            solver_key=options.solver_key,
        )
        _, petsc_linear_solve, _ = differentiable_snes._hooks_from_solver_key(
            options.solver_key,
            primitive,
        )

        rhs = jnp.array([2.0, -1.0, 4.0], dtype=jnp.float64)
        petsc_single = petsc_linear_solve(x_star, phi, rhs)
        petsc_single.block_until_ready()
        jax_single, jax_single_info = jsp.sparse.linalg.cg(
            lambda vector: matrix @ vector,
            rhs,
            tol=1e-12,
            atol=1e-12,
            maxiter=100,
        )
        assert jax_single_info is None
        exact_single = jnp.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(
            np.asarray(petsc_single), np.asarray(exact_single), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(jax_single), np.asarray(exact_single), rtol=1e-10, atol=1e-10
        )

        rhs_rows = jnp.array(
            [
                [2.0, -1.0, 4.0],
                [1.0, 3.0, -2.0],
                [-3.0, 0.5, 1.0],
                [0.25, -4.0, 2.0],
            ],
            dtype=jnp.float64,
        )
        # JAX commonly represents a collection of RHS vectors as rows.  The
        # differentiable-SNES block hook intentionally follows PETSc's column
        # convention instead, so transpose at the call site.
        rhs_columns = rhs_rows.T

        petsc_block = petsc_linear_solve(x_star, phi, rhs_columns)
        petsc_block.block_until_ready()
        jax_batch = jax.vmap(
            lambda active_rhs: jsp.sparse.linalg.cg(
                lambda vector: matrix @ vector,
                active_rhs,
                tol=1e-12,
                atol=1e-12,
                maxiter=100,
            )[0]
        )(rhs_rows).T
        exact_block = jnp.linalg.solve(matrix, rhs_columns)

        np.testing.assert_allclose(
            np.asarray(petsc_block), np.asarray(exact_block), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            np.asarray(jax_batch), np.asarray(exact_block), rtol=1e-10, atol=1e-10
        )
    finally:
        if options.solver_key is not None:
            destroy_petsc_solver(options.solver_key)
