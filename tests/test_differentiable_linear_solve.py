import pytest
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np

import jetsci
from jetsci.coo_data import COOData, to_coo_data
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


# --- linear_solver Tests (Non-differentiable API) ---

def test_linear_solver_jax_dense(linear_system):
    A, b, expected = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )
    x, updated_opts, info = jetsci.linear_solver(opts, A, b)
    assert jnp.allclose(x, expected, atol=1e-5)
    assert updated_opts.solver_key is not None
    assert info is not None


def test_linear_solver_petsc_dense(linear_system):
    A, b, expected = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.PETSC_CG,
        linear_preconditioner_type=PreconditionerType.PETSC_JACOBI,
    )
    x, updated_opts, info = jetsci.linear_solver(opts, A, b)
    assert jnp.allclose(x, expected, atol=1e-5)
    assert updated_opts.solver_key is not None
    assert info is not None
    assert info.iterations > 0


def test_linear_solver_coo_data(linear_system):
    A, b, expected = linear_system
    # Convert dense to COOData
    rows, cols = jnp.nonzero(A)
    vals = A[rows, cols]
    coo = COOData(shape=jnp.array(A.shape), vals=vals, rows=rows.astype(jnp.int32), cols=cols.astype(jnp.int32))

    opts_jax = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )
    x_jax, _, _ = jetsci.linear_solver(opts_jax, coo, b)
    assert jnp.allclose(x_jax, expected, atol=1e-5)

    opts_petsc = LinearSolverOptions(
        linear_solver_type=LinearSolverType.PETSC_CG,
        linear_preconditioner_type=PreconditionerType.PETSC_NONE,
    )
    x_petsc, _, _ = jetsci.linear_solver(opts_petsc, coo, b)
    assert jnp.allclose(x_petsc, expected, atol=1e-5)


def test_linear_solver_tuple_coo(linear_system):
    A, b, expected = linear_system
    rows, cols = jnp.nonzero(A)
    vals = A[rows, cols]
    coo_tuple = (rows.astype(jnp.int32), cols.astype(jnp.int32), vals, A.shape)

    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )
    x, updated_opts, _ = jetsci.linear_solver(opts, coo_tuple, b)
    assert jnp.allclose(x, expected, atol=1e-5)
    assert updated_opts.solver_key is not None


def test_linear_solver_callable_matvec(linear_system):
    A, b, expected = linear_system
    matvec = lambda v: A @ v

    opts_jax = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )
    x, _, _ = jetsci.linear_solver(opts_jax, matvec, b)
    assert jnp.allclose(x, expected, atol=1e-5)

    # PETSc should fail fast with TypeError on callable matvec
    opts_petsc = LinearSolverOptions(
        linear_solver_type=LinearSolverType.PETSC_CG,
        linear_preconditioner_type=PreconditionerType.PETSC_NONE,
    )
    with pytest.raises(TypeError, match="PETSc linear solver cannot accept a callable operator"):
        jetsci.linear_solver(opts_petsc, matvec, b)


# --- differentiable_linear_solve Tests ---

def test_differentiable_linear_solve_jax_primal(linear_system):
    A, b, expected = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )
    A_fn = lambda x, *args: A
    b_fn = lambda x, *args: b
    x_lin = jnp.zeros_like(b)

    x, updated_opts, info = jetsci.differentiable_linear_solve(opts, A_fn, b_fn, x_lin)
    assert jnp.allclose(x, expected, atol=1e-5)
    assert updated_opts.solver_key is not None
    assert info is not None


def test_differentiable_linear_solve_petsc_primal(linear_system):
    A, b, expected = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.PETSC_CG,
        linear_preconditioner_type=PreconditionerType.PETSC_JACOBI,
    )
    A_fn = lambda x, *args: A
    b_fn = lambda x, *args: b
    x_lin = jnp.zeros_like(b)

    x, updated_opts, info = jetsci.differentiable_linear_solve(opts, A_fn, b_fn, x_lin)
    assert jnp.allclose(x, expected, atol=1e-5)
    assert updated_opts.solver_key is not None
    assert info is not None


def test_differentiable_linear_solve_jax_autodiff_params(linear_system):
    A0, b0, _ = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )
    theta = 2.0
    x_lin = jnp.zeros_like(b0)

    # Parameterized: A(theta) = theta * A0, b(theta) = theta * b0
    # x(theta) = A(theta)^-1 b(theta) = (1/theta A0^-1) (theta b0) = A0^-1 b0 = const w.r.t theta!
    A_fn = lambda x, t: t * A0
    b_fn = lambda x, t: t * b0

    def solve_fn(t):
        x, _, _ = jetsci.differentiable_linear_solve(opts, A_fn, b_fn, x_lin, t)
        return jnp.sum(x)

    grad_t = jax.grad(solve_fn)(theta)
    assert jnp.allclose(grad_t, 0.0, atol=1e-5)

    # Differentiate A only: x(theta) = (1/theta) A0^-1 b0
    # d(sum(x))/dtheta = -1/theta^2 sum(A0^-1 b0)
    b_const = lambda x, t: b0
    def loss_A(t):
        x, _, _ = jetsci.differentiable_linear_solve(opts, A_fn, b_const, x_lin, t)
        return jnp.sum(x)

    x0 = jnp.linalg.solve(A0, b0)
    expected_dloss = -1.0 / (theta ** 2) * jnp.sum(x0)
    assert jnp.allclose(jax.grad(loss_A)(theta), expected_dloss, atol=1e-5)


def test_differentiable_linear_solve_petsc_autodiff_params(linear_system):
    A0, b0, _ = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.PETSC_BCGS,
        linear_preconditioner_type=PreconditionerType.PETSC_NONE,
    )
    theta = 2.0
    x_lin = jnp.zeros_like(b0)

    A_fn = lambda x, t: t * A0
    b_const = lambda x, t: b0

    def loss(t):
        x, _, _ = jetsci.differentiable_linear_solve(opts, A_fn, b_const, x_lin, t)
        return jnp.sum(x)

    x0 = jnp.linalg.solve(A0, b0)
    expected_dloss = -1.0 / (theta ** 2) * jnp.sum(x0)

    grad_fwd = jax.jacfwd(loss)(theta)
    grad_rev = jax.grad(loss)(theta)

    assert jnp.allclose(grad_fwd, expected_dloss, atol=1e-5)
    assert jnp.allclose(grad_rev, expected_dloss, atol=1e-5)


def test_differentiable_linear_solve_autodiff_x_linearized(linear_system):
    A0, b0, _ = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    # A depends on x_linearized: A(x_lin) = (1 + 0.1 * sum(x_lin)) * A0
    A_fn = lambda x, *args: (1.0 + 0.1 * jnp.sum(x)) * A0
    b_fn = lambda x, *args: b0
    x_lin = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)

    def loss(x_l):
        x, _, _ = jetsci.differentiable_linear_solve(opts, A_fn, b_fn, x_l)
        return jnp.sum(x)

    factor = 1.0 + 0.1 * jnp.sum(x_lin)
    x_base = jnp.linalg.solve(A0, b0) / factor
    # d(sum(x))/dx_lin_i = -0.1 / factor^2 * sum(A0^-1 b0)
    expected_grad = -0.1 / (factor ** 2) * jnp.sum(jnp.linalg.solve(A0, b0)) * jnp.ones_like(x_lin)

    grad_x = jax.grad(loss)(x_lin)
    assert jnp.allclose(grad_x, expected_grad, atol=1e-5)


def test_differentiable_linear_solve_coo_data():
    # 2x2 SPD matrix
    rows = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    cols = jnp.array([0, 1, 1, 0], dtype=jnp.int32)
    shape = (2, 2)
    b0 = jnp.array([2.0, 5.0], dtype=jnp.float64)
    theta = 3.0
    x_lin = jnp.zeros_like(b0)

    # COOData-producing function
    def A_fn(x, t):
        vals = jnp.array([t * 3.0, t * 4.0, 1.0, 1.0], dtype=jnp.float64)
        return COOData(shape=shape, vals=vals, rows=rows, cols=cols)

    b_fn = lambda x, t: b0

    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    x, updated_opts, info = jetsci.differentiable_linear_solve(opts, A_fn, b_fn, x_lin, theta)
    assert updated_opts.solver_key is not None

    # Autodiff w.r.t theta
    def loss(t):
        sol, _, _ = jetsci.differentiable_linear_solve(opts, A_fn, b_fn, x_lin, t)
        return jnp.sum(sol)

    grad_t = jax.grad(loss)(theta)
    assert not jnp.isnan(grad_t)


def test_differentiable_linear_solve_callable_matvec():
    # Matrix-free operator
    A0 = jnp.array([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    b0 = jnp.array([2.0, 5.0], dtype=jnp.float64)
    theta = 2.0
    x_lin = jnp.zeros_like(b0)

    def A_fn(x, t):
        return lambda v: t * (A0 @ v)

    b_fn = lambda x, t: b0

    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    x, _, _ = jetsci.differentiable_linear_solve(opts, A_fn, b_fn, x_lin, theta)
    expected = jnp.linalg.solve(theta * A0, b0)
    assert jnp.allclose(x, expected, atol=1e-5)

    # Autodiff
    def loss(t):
        sol, _, _ = jetsci.differentiable_linear_solve(opts, A_fn, b_fn, x_lin, t)
        return jnp.sum(sol)

    grad_t = jax.grad(loss)(theta)
    expected_grad = -1.0 / (theta ** 2) * jnp.sum(jnp.linalg.solve(A0, b0))
    assert jnp.allclose(grad_t, expected_grad, atol=1e-5)


def test_linear_solver_key_reuse(linear_system):
    A, b, _ = linear_system
    opts = LinearSolverOptions(
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    # First call: solver constructed, options populated with solver_key
    x1, opts1, _ = jetsci.linear_solver(opts, A, b)
    assert opts1.solver_key is not None
    cached_solver = __linear_solver_dict[opts1.solver_key]

    # Second call with opts1: reuses existing solver instance
    x2, opts2, _ = jetsci.linear_solver(opts1, A, b)
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
