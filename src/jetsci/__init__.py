import jax

jax.config.update("jax_enable_x64", True)

import pathlib
jax.config.update("jax_compilation_cache_dir", str(pathlib.Path(__file__).parent.resolve() / "__jax_cache__"))

from .options import *
from .solve import *
from .lifecycle import *
from .jax_linear import (
    linear_solve,
    LinearSolverResultInfo,
    cg_w_info,
    build_preconditioner,
    JAXOPT_AVAILABLE,
    CUPY_AVAILABLE,
    PYPARDISO_AVAILABLE,
    PYAMGX_AVAILABLE,
)
from .jax_newton_raph import JAXNewtonRaphsonSolver

__all__ = [
    "NonlinearSolverType",
    "PETScPreconditionerType",
    "JAXPreconditionerType",
    "PETScLinearSolverType",
    "JAXLinearSolverType",
    "SolverOptions",
    "differentiable_solve",
    "differentiable_linear_solve",
    "build_solver_with_reuse",
    "linear_solve",
    "LinearSolverResultInfo",
    "cg_w_info",
    "build_preconditioner",
    "JAXNewtonRaphsonSolver",
    "JAXOPT_AVAILABLE",
    "CUPY_AVAILABLE",
    "PYPARDISO_AVAILABLE",
    "PYAMGX_AVAILABLE",
]
