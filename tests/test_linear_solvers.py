import pytest
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import numpy as np

import jetsci
from jetsci.options import JAXLinearSolverType, JAXPreconditionerType, SolverOptions
from jetsci.jax_linear import (
    linear_solve,
    cg_w_info,
    build_preconditioner,
    JAXOPT_AVAILABLE,
    CUPY_AVAILABLE,
    PYPARDISO_AVAILABLE,
)


@pytest.fixture
def spd_system():
    A = jnp.array([
        [4.0, 1.0, 0.5],
        [1.0, 5.0, 1.5],
        [0.5, 1.5, 3.0],
    ], dtype=jnp.float64)
    b = jnp.array([4.0, 9.0, 16.0], dtype=jnp.float64)
    expected = jnp.linalg.solve(A, b)
    return A, b, expected


@pytest.fixture
def nonsymmetric_system():
    A = jnp.array([
        [4.0, 1.0, 0.5],
        [0.25, 3.0, 1.5],
        [0.75, 0.5, 2.5],
    ], dtype=jnp.float64)
    b = jnp.array([4.0, 9.0, 16.0], dtype=jnp.float64)
    expected_fwd = jnp.linalg.solve(A, b)
    expected_trans = jnp.linalg.solve(A.T, b)
    return A, b, expected_fwd, expected_trans


@pytest.mark.parametrize("solver_type", [
    JAXLinearSolverType.DENSE_INVERSE_JNP,
    JAXLinearSolverType.CG_JAX_SCIPY,
    JAXLinearSolverType.CG_JAX_SCIPY_W_INFO,
    JAXLinearSolverType.GMRES_JAX_SCIPY,
    JAXLinearSolverType.BICGSTAB_JAX_SCIPY,
    JAXLinearSolverType.DENSE_INVERSE_JAXOPT,
    JAXLinearSolverType.LU_JAXOPT,
    JAXLinearSolverType.CHOLESKY_JAXOPT,
    JAXLinearSolverType.CG_JAXOPT,
    JAXLinearSolverType.GMRES_JAXOPT,
    JAXLinearSolverType.BICGSTAB_JAXOPT,
    JAXLinearSolverType.SPSOLVE_CUPY,
    JAXLinearSolverType.LU_CUPY,
    JAXLinearSolverType.SPSOLVE_PYPARDISO,
])
def test_all_solvers_spd(spd_system, solver_type):
    A, b, expected = spd_system
    opts = SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solve_type=solver_type,
        linear_precond_type=JAXPreconditionerType.NONE,
    )
    sol, info = linear_solve(A, b, solver_options=opts)
    assert jnp.allclose(sol, expected, atol=1e-5), f"Failed for {solver_type.name}"


@pytest.mark.parametrize("solver_type", [
    JAXLinearSolverType.DENSE_INVERSE_JNP,
    JAXLinearSolverType.GMRES_JAX_SCIPY,
    JAXLinearSolverType.BICGSTAB_JAX_SCIPY,
    JAXLinearSolverType.DENSE_INVERSE_JAXOPT,
    JAXLinearSolverType.LU_JAXOPT,
    JAXLinearSolverType.GMRES_JAXOPT,
    JAXLinearSolverType.BICGSTAB_JAXOPT,
    JAXLinearSolverType.SPSOLVE_CUPY,
    JAXLinearSolverType.LU_CUPY,
    JAXLinearSolverType.SPSOLVE_PYPARDISO,
])
def test_nonsymmetric_transpose_solve(nonsymmetric_system, solver_type):
    A, b, expected_fwd, expected_trans = nonsymmetric_system
    opts = SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solve_type=solver_type,
        linear_precond_type=JAXPreconditionerType.NONE,
    )

    # Forward solve
    sol_fwd, _ = linear_solve(A, b, solver_options=opts, transpose=False)
    assert jnp.allclose(sol_fwd, expected_fwd, atol=1e-5), f"Forward solve failed for {solver_type.name}"

    # Transpose solve
    sol_trans, _ = linear_solve(A, b, solver_options=opts, transpose=True)
    assert jnp.allclose(sol_trans, expected_trans, atol=1e-5), f"Transpose solve failed for {solver_type.name}"


@pytest.mark.parametrize("precond_type", [
    JAXPreconditionerType.NONE,
    JAXPreconditionerType.JACOBI,
    JAXPreconditionerType.ILU_CUPY,
])
def test_preconditioners_with_cg(spd_system, precond_type):
    A, b, expected = spd_system
    opts = SolverOptions(
        nonlinear_solver_type=jetsci.NonlinearSolverType.JAX_NEWTON_RAPHSON,
        linear_solve_type=JAXLinearSolverType.CG_JAX_SCIPY,
        linear_precond_type=precond_type,
    )
    sol, _ = linear_solve(A, b, solver_options=opts)
    assert jnp.allclose(sol, expected, atol=1e-5), f"Preconditioner {precond_type.name} failed"


def test_cg_w_info_diagnostics(spd_system):
    A, b, expected = spd_system
    sol, info = cg_w_info(lambda v: A @ v, b, maxiter=50, tol=1e-10)
    assert jnp.allclose(sol, expected, atol=1e-5)
    assert "iterations" in info
    assert "residual_norm_history" in info
    assert info["iterations"] > 0
    assert len(info["residual_norm_history"]) == 50


def test_sparse_coo_input(spd_system):
    A_dense, b, expected = spd_system
    coo = jsparse.COO.fromdense(A_dense)

    sol_cg, _ = linear_solve(coo, b, solver_type=JAXLinearSolverType.CG_JAX_SCIPY)
    assert jnp.allclose(sol_cg, expected, atol=1e-5)

    sol_cupy, _ = linear_solve(coo, b, solver_type=JAXLinearSolverType.SPSOLVE_CUPY)
    assert jnp.allclose(sol_cupy, expected, atol=1e-5)

    sol_pardiso, _ = linear_solve(coo, b, solver_type=JAXLinearSolverType.SPSOLVE_PYPARDISO)
    assert jnp.allclose(sol_pardiso, expected, atol=1e-5)


def test_linear_solve_requires_solver_type(spd_system):
    A, b, _ = spd_system
    with pytest.raises(ValueError, match="No linear solver type specified in `linear_solve`"):
        linear_solve(A, b)


def test_sparse_solvers_reject_callable(spd_system):
    A, b, _ = spd_system
    matvec = lambda v: A @ v
    with pytest.raises(TypeError, match="cannot accept a callable operator"):
        linear_solve(matvec, b, solver_type=JAXLinearSolverType.SPSOLVE_PYPARDISO)

    with pytest.raises(TypeError, match="cannot accept a callable operator"):
        linear_solve(matvec, b, solver_type=JAXLinearSolverType.SPSOLVE_CUPY)


def test_sparse_solvers_warn_on_dense_array(spd_system):
    A, b, expected = spd_system
    with pytest.warns(UserWarning, match="Dense array passed to sparse direct solver"):
        sol, _ = linear_solve(A, b, solver_type=JAXLinearSolverType.SPSOLVE_PYPARDISO)
        assert jnp.allclose(sol, expected, atol=1e-5)


def test_preconditioner_validation():
    with pytest.raises(ValueError, match="JACOBI preconditioner requires either `diag` or `A`"):
        build_preconditioner(JAXPreconditionerType.JACOBI)

    with pytest.raises(TypeError, match="Cannot extract diagonal for JACOBI preconditioner"):
        build_preconditioner(JAXPreconditionerType.JACOBI, A=lambda v: v)

    with pytest.raises(TypeError, match="ILU_CUPY preconditioner requires an explicit sparse matrix"):
        build_preconditioner(JAXPreconditionerType.ILU_CUPY, A=lambda v: v)


def test_dense_solver_warns_on_sparse_conversion(spd_system):
    A_dense, b, expected = spd_system
    coo = jsparse.COO.fromdense(A_dense)
    with pytest.warns(UserWarning, match="Converting jax.experimental.sparse.COO matrix .* to a dense"):
        sol, _ = linear_solve(coo, b, solver_type=JAXLinearSolverType.DENSE_INVERSE_JNP)
        assert jnp.allclose(sol, expected, atol=1e-5)

