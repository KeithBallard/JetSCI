import pytest
import jax
import jax.numpy as jnp

import jetsci
from jetsci.options import (
    NonlinearSolverType,
    LinearSolverType,
    PreconditionerType,
    SolverOptions,
)
from jetsci.jax_newton_raph import JAXNewtonRaphsonSolver, build_jax_solver_with_reuse


def residual_fn(x):
    # Coupled nonlinear system:
    # x0^3 + x1 - 2 = 0
    # x0 + x1^3 - 2 = 0
    # Solution is [1.0, 1.0]
    return jnp.array([x[0]**3 + x[1] - 2.0, x[0] + x[1]**3 - 2.0])


def jacobian_fn(x):
    return jnp.array([
        [3.0 * x[0]**2, 1.0],
        [1.0, 3.0 * x[1]**2],
    ])


@pytest.mark.parametrize("linear_solver", [
    LinearSolverType.JAX_DENSE_INVERSE_JNP,
    LinearSolverType.JAX_CG_SCIPY,
    LinearSolverType.JAX_GMRES_SCIPY,
    LinearSolverType.JAX_BICGSTAB_SCIPY,
    LinearSolverType.JAX_LU_JAXOPT,
    LinearSolverType.JAX_CHOLESKY_JAXOPT,
    LinearSolverType.JAX_CG_JAXOPT,
    LinearSolverType.JAX_GMRES_JAXOPT,
    LinearSolverType.JAX_BICGSTAB_JAXOPT,
    LinearSolverType.JAX_SPSOLVE_CUPY,
    LinearSolverType.JAX_LU_CUPY,
    LinearSolverType.JAX_SPSOLVE_PYPARDISO,
])
def test_newton_raphson_solvers(linear_solver):
    opts = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solver_type=linear_solver,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
        nonlinear_max_iter=25,
        nonlinear_relative_tol=1e-8,
        nonlinear_absolute_tol=1e-8,
    )

    x0 = jnp.array([0.9, 0.9])
    solver = JAXNewtonRaphsonSolver(
        residual_func=residual_fn,
        jacobian_func=jacobian_fn,
        options=opts,
    )

    sol = solver.solve_to_jax(x0)
    expected = jnp.array([1.0, 1.0])
    assert jnp.allclose(sol, expected, atol=1e-5), f"Failed for {linear_solver.name}"


def test_newton_raphson_ad_jacobian():
    # Test with jacobian_func=None (automatic autodiff Jacobian)
    opts = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solver_type=LinearSolverType.JAX_GMRES_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    x0 = jnp.array([0.5, 0.5])
    solver = JAXNewtonRaphsonSolver(
        residual_func=residual_fn,
        jacobian_func=None,
        options=opts,
    )

    sol = solver.solve_to_jax(x0)
    expected = jnp.array([1.0, 1.0])
    assert jnp.allclose(sol, expected, atol=1e-5)


def test_newton_raphson_companion_linear_solve():
    opts = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solver_type=LinearSolverType.JAX_DENSE_INVERSE_JNP,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    solver = JAXNewtonRaphsonSolver(
        residual_func=residual_fn,
        jacobian_func=jacobian_fn,
        options=opts,
    )

    x_star = jnp.array([1.0, 1.0])
    J_star = jacobian_fn(x_star)
    rhs = jnp.array([4.0, 2.0])

    sol_fwd = solver.linear_solve(rhs, x_star=x_star, transpose=False)
    expected_fwd = jnp.linalg.solve(J_star, rhs)
    assert jnp.allclose(sol_fwd, expected_fwd, atol=1e-5)

    sol_trans = solver.linear_solve(rhs, x_star=x_star, transpose=True)
    expected_trans = jnp.linalg.solve(J_star.T, rhs)
    assert jnp.allclose(sol_trans, expected_trans, atol=1e-5)


def test_lifecycle_reuse():
    opts = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    x0 = jnp.array([0.8, 0.8])
    solver1, opts1 = build_jax_solver_with_reuse(opts, residual_fn, None, x0)
    assert opts1.solver_key is not None

    solver2, opts2 = build_jax_solver_with_reuse(opts1, residual_fn, None, x0)
    assert opts2.solver_key == opts1.solver_key
    assert solver1 is solver2
