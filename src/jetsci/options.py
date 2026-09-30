from __future__ import annotations

from typing import *
from enum import Enum, auto
from dataclasses import dataclass


class NonlinearSolverType(Enum):
    JAX_NEWTON_RAPHSON = auto()
    PETSC_SNES         = auto()

    @property
    def is_jax(self) -> bool:
        return self is NonlinearSolverType.JAX_NEWTON_RAPHSON

    @property
    def is_petsc(self) -> bool:
        return self is NonlinearSolverType.PETSC_SNES


class PreconditionerType(Enum):
    JAX_NONE     = auto()
    JAX_JACOBI   = auto()
    JAX_ILU_CUPY = auto()

    PETSC_NONE   = auto()
    PETSC_JACOBI = auto()
    PETSC_ILU    = auto()

    @property
    def is_jax(self) -> bool:
        return self.name.startswith("JAX_")

    @property
    def is_petsc(self) -> bool:
        return self.name.startswith("PETSC_")


class LinearSolverType(Enum):
    # JAX Solvers
    JAX_DENSE_INVERSE_JNP    = auto()
    JAX_CG_SCIPY             = auto()
    JAX_CG_SCIPY_W_INFO      = auto()
    JAX_GMRES_SCIPY          = auto()
    JAX_BICGSTAB_SCIPY       = auto()
    JAX_DENSE_INVERSE_JAXOPT = auto()
    JAX_LU_JAXOPT            = auto()
    JAX_CHOLESKY_JAXOPT      = auto()
    JAX_CG_JAXOPT            = auto()
    JAX_GMRES_JAXOPT         = auto()
    JAX_BICGSTAB_JAXOPT      = auto()
    JAX_SPSOLVE_CUPY         = auto()
    JAX_LU_CUPY              = auto()
    JAX_AMGX                 = auto()
    JAX_SPSOLVE_PYPARDISO    = auto()

    # PETSc Solvers
    PETSC_CG                 = auto()
    PETSC_GMRES              = auto()
    PETSC_LGMRES             = auto()
    PETSC_BCGS               = auto()
    PETSC_PREONLY            = auto()
    PETSC_MINRES             = auto()

    @property
    def is_jax(self) -> bool:
        return self.name.startswith("JAX_")

    @property
    def is_petsc(self) -> bool:
        return self.name.startswith("PETSC_")


@dataclass(eq=True, frozen=True, kw_only=True)
class LinearSolverOptions:
    linear_solver_type:         LinearSolverType
    linear_preconditioner_type: PreconditionerType

    linear_max_iter:            int = 1000
    linear_relative_tol:        float = 1e-14
    linear_absolute_tol:        float = 1e-10

    solver_key:                 int | None = None
    petsc_mat_type:             str = "aijcusparse"

    def __post_init__(self):
        if not isinstance(self.linear_solver_type, LinearSolverType):
            raise TypeError(
                f"linear_solver_type must be an instance of LinearSolverType, got {self.linear_solver_type!r}. "
                "Fix: Specify an explicit LinearSolverType (e.g., LinearSolverType.JAX_CG_SCIPY or LinearSolverType.PETSC_CG)."
            )
        if not isinstance(self.linear_preconditioner_type, PreconditionerType):
            raise TypeError(
                f"linear_preconditioner_type must be an instance of PreconditionerType, got {self.linear_preconditioner_type!r}. "
                "Fix: Specify an explicit PreconditionerType (e.g., PreconditionerType.JAX_NONE or PreconditionerType.PETSC_NONE)."
            )
        if self.linear_solver_type.is_jax != self.linear_preconditioner_type.is_jax:
            backend_s = "JAX" if self.linear_solver_type.is_jax else "PETSc"
            backend_p = "JAX" if self.linear_preconditioner_type.is_jax else "PETSc"
            raise ValueError(
                f"Linear solver type and preconditioner type backends must match. "
                f"Got linear_solver_type={self.linear_solver_type.name} ({backend_s} backend) and "
                f"linear_preconditioner_type={self.linear_preconditioner_type.name} ({backend_p} backend). "
                f"Fix: Use a matching {backend_s} preconditioner type."
            )


@dataclass(eq=True, frozen=True, kw_only=True)
class SolverOptions(LinearSolverOptions):
    nonlinear_solver_type:  NonlinearSolverType

    nonlinear_max_iter:     int = 10
    nonlinear_relative_tol: float = 1e-10 # Convergence if ||R(x)|| < absolute_tol * ||R(x_0)||
    nonlinear_absolute_tol: float = 1e-8 # Convergence if ||R(x)|| < absolute_tol
    nonlinear_step_tol:     float = 0.0 # Convergence if ||delta x|| < step_tol * ||x||

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.nonlinear_solver_type, NonlinearSolverType):
            raise TypeError(
                f"nonlinear_solver_type must be an instance of NonlinearSolverType, got {self.nonlinear_solver_type!r}. "
                "Fix: Specify NonlinearSolverType.JAX_NEWTON_RAPHSON or NonlinearSolverType.PETSC_SNES."
            )
        if self.nonlinear_solver_type.is_jax and not self.linear_solver_type.is_jax:
            raise ValueError(
                f"Nonlinear solver type {self.nonlinear_solver_type.name} requires a JAX linear solver and preconditioner. "
                f"Got linear_solver_type={self.linear_solver_type.name}. "
                "Fix: Pass a JAX linear solver (e.g., LinearSolverType.JAX_CG_SCIPY) and preconditioner."
            )
        if self.nonlinear_solver_type.is_petsc and not self.linear_solver_type.is_petsc:
            raise ValueError(
                f"Nonlinear solver type {self.nonlinear_solver_type.name} requires a PETSc linear solver and preconditioner. "
                f"Got linear_solver_type={self.linear_solver_type.name}. "
                "Fix: Pass a PETSc linear solver (e.g., LinearSolverType.PETSC_CG) and preconditioner."
            )
