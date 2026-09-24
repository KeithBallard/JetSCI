from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Any
import warnings
import jax
import jax.lax as lax
import jax.numpy as jnp

from ..options import SolverOptions, NonlinearSolverType, JAXLinearSolverType, JAXPreconditionerType
from ..jax_linear import linear_solve, LinearSolverResultInfo


def validate_jax_solver_options(options: SolverOptions) -> None:
    """Validate that selected solver options are compatible with JAX Newton-Raphson."""
    if options.nonlinear_solver_type is NonlinearSolverType.JAX_NEWTON_RAPHSON:
        if not isinstance(options.linear_solve_type, JAXLinearSolverType):
            raise TypeError(
                "JAX Newton-Raphson requires a JAX linear solver type. "
                f"Got {options.linear_solve_type!r}."
            )
        if not isinstance(options.linear_precond_type, JAXPreconditionerType):
            raise TypeError(
                "JAX Newton-Raphson requires a JAX preconditioner type. "
                f"Got {options.linear_precond_type!r}."
            )


@dataclass
class JAXNewtonRaphsonSolver:
    """JAX-only Newton-Raphson nonlinear solver matching JetSCI solver interface."""

    residual_func: Callable
    jacobian_func: Callable | None
    options: SolverOptions

    def __post_init__(self):
        if self.jacobian_func is None and self.options.linear_solve_type in (
            JAXLinearSolverType.DENSE_INVERSE_JNP,
            JAXLinearSolverType.SPSOLVE_CUPY,
            JAXLinearSolverType.LU_CUPY,
            JAXLinearSolverType.SPSOLVE_PYPARDISO,
        ):
            warnings.warn(
                f"No Jacobian function J(x) provided to JAX Newton-Raphson with linear solver "
                f"'{self.options.linear_solve_type.name}'. A dense Jacobian will be materialized via "
                f"jax.jacfwd(R)(x) on every iteration. This scales as O(N^2) memory and O(N) evaluations, "
                f"severely degrading performance for large systems. "
                f"Fix: Provide an explicit Jacobian function J(x) returning a sparse matrix, or use a "
                f"matrix-free linear solver (e.g., JAXLinearSolverType.CG_JAX_SCIPY).",
                UserWarning,
                stacklevel=2,
            )

    def solve(self, x0: jnp.ndarray) -> jnp.ndarray:
        """Solve nonlinear system R(x) = 0 starting from x0."""
        R = self.residual_func
        J = self.jacobian_func
        opts = self.options

        R0 = R(x0)
        norm_0 = jnp.linalg.norm(R0)

        def while_cond(val):
            nl_iter, x, R_x, norm_0_val = val
            norm_R = jnp.linalg.norm(R_x)
            rel_err = jnp.where(norm_0_val > 0, norm_R / norm_0_val, 0.0)
            return (
                (nl_iter < opts.nonlinear_max_iter)
                & (rel_err > opts.nonlinear_relative_tol)
                & (norm_R > opts.nonlinear_absolute_tol)
            )

        def while_body(val):
            nl_iter, x, R_x, norm_0_val = val
            if J is not None:
                J_op = J(x)
            else:
                if opts.linear_solve_type in (
                    JAXLinearSolverType.DENSE_INVERSE_JNP,
                    JAXLinearSolverType.SPSOLVE_CUPY,
                    JAXLinearSolverType.LU_CUPY,
                    JAXLinearSolverType.SPSOLVE_PYPARDISO,
                ):
                    J_op = jax.jacfwd(R)(x)
                else:
                    J_op = lambda v: jax.jvp(R, (x,), (v,))[1]

            delta_x, _ = linear_solve(
                J_op,
                -R_x,
                solver_options=opts,
            )
            x_next = x + delta_x
            R_next = R(x_next)
            return (nl_iter + 1, x_next, R_next, norm_0_val)

        init_val = (0, x0, R0, norm_0)
        _, x_sol, _, _ = lax.while_loop(while_cond, while_body, init_val)
        return x_sol

    def solve_to_jax(self, x0: jnp.ndarray) -> jnp.ndarray:
        """Solve and return solution as a JAX array."""
        return self.solve(x0)

    def linear_solve(
        self,
        rhs: jnp.ndarray,
        x_star: jnp.ndarray | None = None,
        transpose: bool = False,
    ) -> jnp.ndarray:
        """Perform linear solve J(x*) dx = rhs (or J(x*)^T lam = rhs) for IFT differentiation."""
        R = self.residual_func
        J = self.jacobian_func
        opts = self.options

        if x_star is None:
            raise ValueError("x_star (solution point) is required for linear_solve in IFT")

        if J is not None:
            J_op = J(x_star)
        else:
            if opts.linear_solve_type in (
                JAXLinearSolverType.DENSE_INVERSE_JNP,
                JAXLinearSolverType.SPSOLVE_CUPY,
                JAXLinearSolverType.LU_CUPY,
                JAXLinearSolverType.SPSOLVE_PYPARDISO,
            ):
                J_mat = jax.jacfwd(R)(x_star)
                J_op = J_mat
            else:
                J_op = lambda v: jax.jvp(R, (x_star,), (v,))[1]

        sol, _ = linear_solve(
            J_op,
            rhs,
            solver_options=opts,
            transpose=transpose,
        )
        return sol

    def destroy(self):
        """Cleanup resources."""
        pass


__jax_solver_dict: dict[int, JAXNewtonRaphsonSolver] = {}
__jax_solver_id = 0


def _new_jax_solver_key() -> int:
    global __jax_solver_id
    __jax_solver_id += 1
    return __jax_solver_id


def build_jax_solver_with_reuse(
    options: SolverOptions,
    R: jax.tree_util.Partial,
    J: jax.tree_util.Partial | None = None,
    x0: jnp.ndarray | None = None,
) -> tuple[JAXNewtonRaphsonSolver, SolverOptions]:
    """Return a JAX Newton-Raphson solver and updated SolverOptions."""
    validate_jax_solver_options(options)

    if options.solver_key is None:
        solver = JAXNewtonRaphsonSolver(
            residual_func=R,
            jacobian_func=J,
            options=options,
        )
        solver_key = _new_jax_solver_key()
        __jax_solver_dict[solver_key] = solver
        return solver, replace(options, solver_key=solver_key)
    else:
        if options.solver_key not in __jax_solver_dict:
            raise KeyError(f"No JAX solver found for solver_key={options.solver_key}")
        solver = __jax_solver_dict[options.solver_key]
        solver.residual_func = R
        solver.jacobian_func = J
        solver.options = options
        return solver, options
