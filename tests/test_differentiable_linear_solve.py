import pytest
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np

import jetsci
from jetsci.options import (
    LinearSolverType,
    PreconditionerType,
    LinearSolverOptions,
    SolverOptions,
    NonlinearSolverType,
)
from jetsci.lifecycle import build_linear_solver_with_reuse, __linear_solver_dict


@pytest.fixture
def linear_system():
    # SPD 3x3 matrix
    A = jnp.array([
        [4.0, 1.0, 0.5],
        [1.0, 5.0, 1.5],
        [0.5, 1.5, 3.0],
    ], dtype=jnp.float64)
    b = jnp.array([2.0, 3.0, 4.0], dtype=jnp.float64)
    expected = jnp.linalg.solve(A, b)
    return A, b, expected


def test_options_backend_validation():
    # Mismatched linear solver and preconditioner backends
    with pytest.raises(ValueError, match="backends must match"):
        LinearSolverOptions(
            linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
            linear_preconditioner_type=PreconditionerType.PETSC_JACOBI,
        )

    with pytest.raises(ValueError, match="backends must match"):
        LinearSolverOptions(
            linear_solver_type=LinearSolverType.PETSC_CG,
            linear_preconditioner_type=PreconditionerType.JAX_JACOBI,
        )

    # Mismatched nonlinear solver and linear solver backends
    with pytest.raises(ValueError, match="requires a PETSc linear solver"):
        SolverOptions(
            nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
            linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
            linear_preconditioner_type=PreconditionerType.JAX_NONE,
        )

    with pytest.raises(ValueError, match="requires a JAX linear solver"):
        SolverOptions(
            nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
            linear_solver_type=LinearSolverType.PETSC_CG,
            linear_preconditioner_type=PreconditionerType.PETSC_NONE,
        )


def test_differentiable_linear_solve_jax_primal(linear_system):
    A, b, expected = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )
    x, updated_opts = jetsci.differentiable_linear_solve(opts, A, b)
    assert jnp.allclose(x, expected, atol=1e-5)
    assert updated_opts.solver_key is not None


def test_differentiable_linear_solve_petsc_primal(linear_system):
    A, b, expected = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.PETSC_CG,
        linear_preconditioner_type=PreconditionerType.PETSC_JACOBI,
    )
    x, updated_opts = jetsci.differentiable_linear_solve(opts, A, b)
    assert jnp.allclose(x, expected, atol=1e-5)
    assert updated_opts.solver_key is not None


def test_differentiable_linear_solve_jax_autodiff(linear_system):
    A, b, _ = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    # Autodiff w.r.t rhs b: dx/db = A^-1
    expected_jac = jnp.linalg.inv(A)

    jac_fwd = jax.jacfwd(lambda rhs: jetsci.differentiable_linear_solve(opts, A, rhs)[0])(b)
    jac_rev = jax.jacrev(lambda rhs: jetsci.differentiable_linear_solve(opts, A, rhs)[0])(b)
    grad_val = jax.grad(lambda rhs: jnp.sum(jetsci.differentiable_linear_solve(opts, A, rhs)[0]))(b)

    assert jnp.allclose(jac_fwd, expected_jac, atol=1e-5)
    assert jnp.allclose(jac_rev, expected_jac, atol=1e-5)
    assert jnp.allclose(grad_val, jnp.sum(expected_jac, axis=0), atol=1e-5)


def test_differentiable_linear_solve_petsc_autodiff(linear_system):
    A, b, _ = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.PETSC_BCGS,
        linear_preconditioner_type=PreconditionerType.PETSC_NONE,
    )

    expected_jac = jnp.linalg.inv(A)

    jac_fwd = jax.jacfwd(lambda rhs: jetsci.differentiable_linear_solve(opts, A, rhs)[0])(b)
    jac_rev = jax.jacrev(lambda rhs: jetsci.differentiable_linear_solve(opts, A, rhs)[0])(b)
    grad_val = jax.grad(lambda rhs: jnp.sum(jetsci.differentiable_linear_solve(opts, A, rhs)[0]))(b)

    assert jnp.allclose(jac_fwd, expected_jac, atol=1e-5)
    assert jnp.allclose(jac_rev, expected_jac, atol=1e-5)
    assert jnp.allclose(grad_val, jnp.sum(expected_jac, axis=0), atol=1e-5)


def test_differentiable_linear_solve_matrix_autodiff():
    # Parameterized matrix A(theta) = theta * A0
    A0 = jnp.array([
        [3.0, 1.0],
        [1.0, 4.0],
    ], dtype=jnp.float64)
    b = jnp.array([2.0, 5.0], dtype=jnp.float64)
    theta = 2.0

    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    def loss(t):
        A = t * A0
        x, _ = jetsci.differentiable_linear_solve(opts, A, b)
        return jnp.sum(x)

    # Analytical derivative: x(theta) = (1/theta) A0^-1 b -> d(sum(x))/dtheta = -1/theta^2 sum(A0^-1 b)
    x0 = jnp.linalg.solve(A0, b)
    expected_dloss = -1.0 / (theta ** 2) * jnp.sum(x0)

    grad_theta = jax.grad(loss)(theta)
    assert jnp.allclose(grad_theta, expected_dloss, atol=1e-5)


def test_linear_solver_key_reuse(linear_system):
    A, b, _ = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    # First call: solver constructed, options populated with solver_key
    x1, opts1 = jetsci.differentiable_linear_solve(opts, A, b)
    assert opts1.solver_key is not None
    cached_solver = __linear_solver_dict[opts1.solver_key]

    # Second call with opts1: reuses existing solver instance
    x2, opts2 = jetsci.differentiable_linear_solve(opts1, A, b)
    assert opts2.solver_key == opts1.solver_key
    assert __linear_solver_dict[opts2.solver_key] is cached_solver
    assert jnp.allclose(x1, x2, atol=1e-6)


def test_linear_solver_sparsity_change_warning():
    # Create initial sparse COO matrix
    row1 = jnp.array([0, 1], dtype=jnp.int32)
    col1 = jnp.array([0, 1], dtype=jnp.int32)
    data1 = jnp.array([2.0, 3.0], dtype=jnp.float64)
    A1 = jsparse.COO((data1, row1, col1), shape=(2, 2))

    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )
    solver, updated_opts = build_linear_solver_with_reuse(opts, A1)

    # In-place update with same sparsity pattern (no warning)
    data1_updated = jnp.array([4.0, 5.0], dtype=jnp.float64)
    A1_updated = jsparse.COO((data1_updated, row1, col1), shape=(2, 2))
    solver.update_operator(A1_updated)

    # Update with changed sparsity pattern (should emit UserWarning)
    row2 = jnp.array([0, 0, 1], dtype=jnp.int32)
    col2 = jnp.array([0, 1, 1], dtype=jnp.int32)
    data2 = jnp.array([4.0, 1.0, 5.0], dtype=jnp.float64)
    A2 = jsparse.COO((data2, row2, col2), shape=(2, 2))

    with pytest.warns(UserWarning, match="sparsity pattern changed"):
        solver.update_operator(A2)
