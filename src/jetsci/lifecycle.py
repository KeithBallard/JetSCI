from . import petsc_snes

def build_solver_with_reuse(
    options: SolverOptions,
    R: jax.tree_util.Partial,
    J: jax.tree_util.Partial,
    x0: jnp.ndarry | None = None,
) -> tuple[jnp.ndarray, SolverOptions]:
    match options.nonlinear_solver_type:
        case NonlinearSolverType.JAX_NEWTON_RAPHSON:
            return jax_netwon_raph
        case NonlinearSolverType.PETSC_SNES:
            return petsc_snes.build_petsc_solver_with_reuse(
                solver_options,
                R_bar,
                J_bar,
                x_0,
            )