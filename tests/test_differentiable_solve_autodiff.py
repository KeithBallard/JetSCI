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


def residual(phi, x):
    # Linearized or nonlinear operator:
    # A = [[4.0, 1.0, 0.5], [0.25, 3.0, 1.5], [0.75, 0.5, 2.5]]
    # R(phi, x) = A @ x - phi
    A = jnp.array([
        [4.0, 1.0, 0.5],
        [0.25, 3.0, 1.5],
        [0.75, 0.5, 2.5],
    ], dtype=jnp.float64)
    return A @ x - phi


def nonlinear_residual(p, x):
    # Fully nonlinear coupled residual:
    # [x0^3 + x1 - p0, x0 + x1^3 - p1]
    return jnp.array([
        x[0]**3 + x[1] - p[0],
        x[0] + x[1]**3 - p[1],
    ], dtype=jnp.float64)


def test_jax_newton_autodiff_forward_reverse():
    p = jnp.array([2.0, 2.0], dtype=jnp.float64)
    x0 = jnp.array([0.9, 0.9], dtype=jnp.float64)

    opts = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solver_type=LinearSolverType.JAX_CG_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    # Primal solve
    sol, _ = jetsci.differentiable_solve(opts, nonlinear_residual, None, x0, p)
    assert jnp.allclose(sol, jnp.array([1.0, 1.0]), atol=1e-5)

    # Forward-mode jacobian (jacfwd)
    jac_fwd = jax.jacfwd(lambda param: jetsci.differentiable_solve(opts, nonlinear_residual, None, x0, param)[0])(p)

    # Reverse-mode jacobian (jacrev)
    jac_rev = jax.jacrev(lambda param: jetsci.differentiable_solve(opts, nonlinear_residual, None, x0, param)[0])(p)

    # JVP
    p_dot = jnp.array([1.0, 0.0], dtype=jnp.float64)
    _, x_dot = jax.jvp(lambda param: jetsci.differentiable_solve(opts, nonlinear_residual, None, x0, param)[0], (p,), (p_dot,))

    # Scalar loss gradient (grad)
    grad_val = jax.grad(lambda param: jnp.sum(jetsci.differentiable_solve(opts, nonlinear_residual, None, x0, param)[0]))(p)

    # Analytical expected Jacobian: J^-1 where J = [[3, 1], [1, 3]] -> J^-1 = [[3/8, -1/8], [-1/8, 3/8]]
    expected_jac = jnp.array([[0.375, -0.125], [-0.125, 0.375]])
    assert jnp.allclose(jac_fwd, expected_jac, atol=1e-5)
    assert jnp.allclose(jac_rev, expected_jac, atol=1e-5)
    assert jnp.allclose(x_dot, expected_jac @ p_dot, atol=1e-5)
    assert jnp.allclose(grad_val, jnp.sum(expected_jac, axis=0), atol=1e-5)


def test_jax_and_petsc_equivalence():
    phi = jnp.array([4.0, 9.0, 16.0], dtype=jnp.float64)
    x0 = jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64)

    jax_opts = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solver_type=LinearSolverType.JAX_GMRES_SCIPY,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    petsc_opts = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
        linear_solver_type=LinearSolverType.PETSC_BCGS,
        linear_preconditioner_type=PreconditionerType.PETSC_NONE,
    )

    sol_jax, _ = jetsci.differentiable_solve(jax_opts, residual, None, x0, phi)
    sol_petsc, _ = jetsci.differentiable_solve(petsc_opts, residual, None, x0, phi)
    assert jnp.allclose(sol_jax, sol_petsc, atol=1e-5)

    jac_jax = jax.jacfwd(lambda p: jetsci.differentiable_solve(jax_opts, residual, None, x0, p)[0])(phi)
    jac_petsc = jax.jacfwd(lambda p: jetsci.differentiable_solve(petsc_opts, residual, None, x0, p)[0])(phi)
    assert jnp.allclose(jac_jax, jac_petsc, atol=1e-5)

    grad_jax = jax.grad(lambda p: jnp.sum(jetsci.differentiable_solve(jax_opts, residual, None, x0, p)[0]))(phi)
    grad_petsc = jax.grad(lambda p: jnp.sum(jetsci.differentiable_solve(petsc_opts, residual, None, x0, p)[0]))(phi)
    assert jnp.allclose(grad_jax, grad_petsc, atol=1e-5)


def test_multi_parameter_differentiable_solve():
    # R(p1, p2, x) = [x0^3 + p1 * x1 - p2, x0 + x1^3 - 2.0]
    def multi_param_residual(p1, p2, x):
        return jnp.array([
            x[0]**3 + p1 * x[1] - p2,
            x[0] + x[1]**3 - 2.0,
        ])

    p1 = jnp.array(1.0, dtype=jnp.float64)
    p2 = jnp.array(2.0, dtype=jnp.float64)
    x0 = jnp.array([0.9, 0.9], dtype=jnp.float64)

    opts = SolverOptions(
        nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solver_type=LinearSolverType.JAX_LU_JAXOPT,
        linear_preconditioner_type=PreconditionerType.JAX_NONE,
    )

    sol, _ = jetsci.differentiable_solve(opts, multi_param_residual, None, x0, p1, p2)
    assert jnp.allclose(sol, jnp.array([1.0, 1.0]), atol=1e-5)

    # Gradients w.r.t p1 and p2 simultaneously
    def loss(p_a, p_b):
        x, _ = jetsci.differentiable_solve(opts, multi_param_residual, None, x0, p_a, p_b)
        return jnp.sum(x)

    g1, g2 = jax.grad(loss, argnums=(0, 1))(p1, p2)
    assert jnp.isfinite(g1)
    assert jnp.isfinite(g2)
    # At solution x*=[1, 1], J = [[3, 1], [1, 3]], J^-1 = [[3/8, -1/8], [-1/8, 3/8]]
    # sum(J^-1) row sum is [2/8, 2/8] = [0.25, 0.25]
    # dR/dp1 = [1, 0] -> dp1 grad = - (0.25 * 1 + 0.25 * 0) = -0.25
    # dR/dp2 = [-1, 0] -> dp2 grad = - (0.25 * -1 + 0.25 * 0) = +0.25
    assert jnp.allclose(g1, -0.25, atol=1e-5)
    assert jnp.allclose(g2, 0.25, atol=1e-5)
