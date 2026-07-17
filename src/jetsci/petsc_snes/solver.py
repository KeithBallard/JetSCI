from dataclasses import dataclass

from petsc4py import PETSc

from ..conversions import *
from ..options import *

@dataclass
class PETScNonlinearSolver:
    """Own the PETSc SNES object and its callback companion objects."""

    snes: object
    residual_callback: Callable
    jacobian_callback: Callable
    options: SolverOptions
    residual_vec: object | None = None
    jacobian_mat: object | None = None

    def __post_init__(self):
        """Setup snes and Mat/Vec."""
        self.residual_vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.jacobian_mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self.jacobian_mat.setType('aijcusparse')
        self.snes.setFunction(self.residual_callback, self.residual_vec)
        self.snes.setJacobian(self.jacobian_callback, self.jacobian_mat, self.jacobian_mat)


    def _ensure_size(self, x0: jnp.ndarray):
        """Ensure the vector is the correct size"""
        if self.residual_vec.getType() is None:
            self.residual_vec.setType("cuda")
            self.residual_vec.setSizes((PETSc.DECIDE, x0.shape[0]))
            self.residual_vec.setUp()


    def solve(self, x0: jnp.ndarray):
        """Solve with this SNES object and return a PETSc Vec.

        The caller owns the returned Vec and is responsible for destroying it.
        """
        self._ensure_size(x0)

        x0_vec = jax_array_to_petsc_vec(x0)
        x = x0_vec.duplicate()
        try:
            x0_vec.copy(x)
            self.snes.solve(None, x)
            return x
        finally:
            x0_vec.destroy()

    def solve_to_jax(self, x0):
        """Solve and explicitly copy the PETSc Vec result into a JAX array."""
        x = self.solve(x0)
        try:
            result = petsc_vec_to_jax_array(x).copy()
            result.block_until_ready()
            return result
        finally:
            x.destroy()

    def linear_solve(self, x0):
        pass

    def cleanup_work_vectors(self):
        """Destroy residual/Jacobian objects that depend on vector size."""
        if self.residual_vec is not None:
            self.residual_vec.destroy()
            self.residual_vec = None
        if self.jacobian_mat is not None:
            self.jacobian_mat.destroy()
            self.jacobian_mat = None

    def destroy(self):
        """Destroy all PETSc objects owned by this wrapper."""
        self.cleanup_work_vectors()
        self.snes.destroy()


def validate_petsc_solver_options(options: SolverOptions) -> None:
    """Validate that selected solver families use compatible method enums."""
    if options.nonlinear_solver_type is NonlinearSolverType.PETSC_SNES:
        if not isinstance(options.linear_solve_type, PETScLinearSolverType):
            raise TypeError(
                "PETSc SNES requires a PETSc linear solver method. "
                f"Got {options.linear_solve_type!r}."
            )
        if not isinstance(options.linear_precond_type, PETScPreconditionerType):
            raise TypeError(
                "PETSc SNES requires a PETSc preconditioner method. "
                f"Got {options.linear_precond_type!r}."
            )
