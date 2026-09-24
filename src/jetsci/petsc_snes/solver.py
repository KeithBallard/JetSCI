from dataclasses import dataclass

from petsc4py import PETSc

from ..conversions import *
from ..options import *

@dataclass
class PETScLinearSolver:
    """Owns the PETSc KSP object to be used in automatic differentiation.

    This is the linear-solve companion to `PETScNonlinearSolver`. It owns the
    live PETSc KSP plus its work vectors and operator matrix so that the IFT
    derivative solve can reuse PETSc directly instead of falling back to JAX.
    """
    ksp: object
    vector_callback: Callable
    matrix_callback: Callable
    options: SolverOptions
    vector_data: object | None = None
    matrix_data: object | None = None
    operator_matrix: object | None = None

    working_vector: object | None = None


    def __post_init__(self):
        """Setup the KSP work objects."""
        #DEBUG PRINT
        print("PETScLinearSolver __post_init__ called")
        
        self.vector_data = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.working_vector = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.matrix_data = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self.matrix_data.setType('aijcusparse')
        self.operator_matrix = self.matrix_data
        self.ksp.setOperators(self.operator_matrix, self.operator_matrix)

        #DEBUG PRINT
        print("PETScLinearSolver __post_init__ complete")

    def _ensure_size(self, x0: jnp.ndarray):
        """Ensure the vector and matrix storage are the correct size."""
        if self.vector_data.getType() is None:
            self.vector_data.setType("cuda")
            self.vector_data.setSizes((PETSc.DECIDE, x0.shape[0]))
            self.vector_data.setUp()
        if self.working_vector.getType() is None:
            self.working_vector.setType("cuda")
            self.working_vector.setSizes((PETSc.DECIDE, x0.shape[0]))
            self.working_vector.setUp()
        if self.matrix_data.getType() is None:
            self.matrix_data.setSizes((x0.shape[0], x0.shape[0]))
            self.matrix_data.setUp()
            self.operator_matrix = self.matrix_data
            self.ksp.setOperators(self.operator_matrix, self.operator_matrix)


    def update_operator(self, operator_matrix):
        """Replace the linear operator used by the KSP."""

        #DEBUG PRINT
        print("PETScLinearSolver update operator complete")

        if self.matrix_data is not None and self.matrix_data is not operator_matrix:
            self.matrix_data.destroy()
        self.operator_matrix = operator_matrix
        self.matrix_data = operator_matrix
        self.ksp.setOperators(operator_matrix, operator_matrix)

        #DEBUG PRINT
        print("PETScLinearSolver update_operator complete")

        return self


    def linear_solve(self, rhs: jnp.ndarray):
        """Solve with this KSP object and return a PETSc Vec.

        The caller owns the returned Vec and is responsible for destroying it.
        """

        #DEBUG PRINT
        print("PETScLinearSolver linear_solve called")

        self._ensure_size(rhs)

        #DEBUG PRINT
        print("PETScLinearSolver linear_solve: converting jax rhs to rhsVec")

        rhs_vec = jax_array_to_petsc_vec(rhs)

        #DEBUG PRINT
        print("PETScLinearSolver linear_solve: finished converting jax rhs to rhsVec")


        try:
            #DEBUG PRINT
            print("PETScLinearSolver linear_solve: copying x into rhsVec")
            rhs_vec.copy(self.working_vector)

            #DEBUG PRINT
            print("PETScLinearSolver linear_solve: finished copying x into rhsVec")


            #DEBUG PRINT
            print("PETScLinearSolver linear_solve: calling self.ksp.solve")

            self.ksp.solve(rhs_vec, self.working_vector)

            #DEBUG PRINT
            print("PETScLinearSolver linear_solve: finished self.ksp.solve") 


            #DEBUG PRINT
            print("PETScLinearSolver linear_solve complete")

            return self.working_vector
        finally:
            rhs_vec.destroy()

    def block_linear_solve(self,rhs_block):


        #DEBUG PRINT
        print("PETScLinearSolver block_linear_solve")

        output_block = rhs_block.duplicate()
        self.ksp.matSolve(rhs_block,output_block)

        #DEBUG PRINT
        print("PETScLinearSolver block_linear_solve complete")

        return output_block

    def transpose_linear_solve(self, rhs: jnp.ndarray):
        """Solve adjoint problem with this KSP object and return a PETSc Vec.

        The caller owns the returned Vec and is responsible for destroying it.
        """

        #DEBUG PRINT
        print("PETScLinearSolver transpose_linear_solve")

        self._ensure_size(rhs)

        rhs_vec = jax_array_to_petsc_vec(rhs)

        try:
            rhs_vec.copy(self.working_vector)
            self.ksp.solveTranspose(rhs_vec, self.working_vector)

            #DEBUG PRINT
            print("PETScLinearSolver block_linear_solve complete")

            return self.working_vector

        
        finally:
            rhs_vec.destroy()


    def solve(self, rhs: jnp.ndarray):
        """Alias for `linear_solve`."""
        return self.linear_solve(rhs)
    
    def solve_transpose(self, rhs: jnp.ndarray):
        """Alias for `transpose_linear_solve`."""
        return self.transpose_linear_solve(rhs)


    def solve_to_jax(self, rhs):
        """Solve and explicitly copy the PETSc Vec result into a JAX array."""

        #DEBUG PRINT
        print("PETScLinearSolver solve_to_jax")

        x = self.linear_solve(rhs)
        try:
            #DEBUG PRINT
            print("PETScLinearSolver solve_to_jax: starting petsc_vec_to_jax_array(x).copy()")

            result = petsc_vec_to_jax_array(x).copy()
    

            result.block_until_ready()

            #DEBUG PRINT
            print("PETScLinearSolver solve_to_jax: finished petsc_vec_to_jax_array(x).copy()")

            return result
        finally:
            pass

    def solve_transpose_to_jax(self, rhs):
        """Solve transpose and explicitly copy the PETSc Vec result into a JAX array."""

        #DEBUG PRINT
        print("PETScLinearSolver solve_transpose_to_jax")

        x = self.transpose_linear_solve(rhs)
        try:

            #DEBUG PRINT
            print("PETScLinearSolver solve_transpose_to_jax: starting petsc_vec_to_jax_array(x).copy()")
            result = petsc_vec_to_jax_array(x).copy()
            result.block_until_ready()

            #DEBUG PRINT
            print("PETScLinearSolver solve_transpose_to_jax: finished starting petsc_vec_to_jax_array(x).copy()")

            return result
        finally:
            pass

    def cleanup_work_vectors(self):
        """Destroy work objects that depend on vector size."""
        if self.vector_data is not None:
            self.vector_data.destroy()
            self.vector_data = None
        if self.working_vector is not None:
            self.working_vector.destroy()
            self.working_vector = None
        if self.matrix_data is not None:
            self.matrix_data.destroy()
            self.matrix_data = None
        self.operator_matrix = None

    def destroy(self):
        """Destroy all PETSc objects owned by this wrapper."""

        #DEBUG PRINT
        print("PETScLinearSolver detroy called")

        self.cleanup_work_vectors()
        self.ksp.destroy()

@dataclass
class PETScNonlinearSolver:
    """Own the PETSc SNES object and its callback companion objects."""

    snes: object
    residual_callback: Callable
    jacobian_callback: Callable
    options: SolverOptions
    residual_vec: object | None = None
    jacobian_mat: object | None = None
    jacobian_callback_state: object | None = None
    callback_stats: dict | None = None
    diagnostics: bool = False
    last_diagnostics: dict | None = None

    #this is to avoid that recasting error Chennie was getting
    workingVector: object | None = None 

    def __post_init__(self):
        """Setup snes and Mat/Vec."""

        #DEBUG PRINT
        print("PETScNonlinearSolver __post_init__ called")

        self.residual_vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.workingVector = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        self.jacobian_mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
        self.jacobian_mat.setType('aijcusparse')
        self.snes.setFunction(self.residual_callback, self.residual_vec)
        self.snes.setJacobian(self.jacobian_callback, self.jacobian_mat, self.jacobian_mat)

        #DEBUG PRINT
        print("PETScNonlinearSolver __post_init__ complete")


    def _ensure_size(self, x0: jnp.ndarray):
        """Ensure the vector is the correct size"""
        if self.residual_vec.getType() is None:
            self.residual_vec.setType("cuda")
            self.residual_vec.setSizes((PETSc.DECIDE, x0.shape[0]))
            self.residual_vec.setUp()

        if self.workingVector.getType() is None:
            self.workingVector.setType("cuda")
            self.workingVector.setSizes((PETSc.DECIDE, x0.shape[0]))
            self.workingVector.setUp()

    def solve(self, x0: jnp.ndarray):
        """Solve with this SNES object and return a PETSc Vec.

        The caller owns the returned Vec and is responsible for destroying it.
        """

        #DEBUG PRINT
        print("Inside PETScNonlinearSolver solve")

        solve_start = perf_counter()
        self._ensure_size(x0)

        conversion_start = perf_counter()

        #DEBUG PRINT
        print("PETScNonlinearSolver solve: converting x0 JAX array to petscVec")
        x0_vec = jax_array_to_petsc_vec(x0)

        #DEBUG PRINT
        print("PETScNonlinearSolver solve: finished converting x0 to petscVec")
        
        conversion_time = perf_counter() - conversion_start



        try:
            copy_start = perf_counter()

           
            #DEBUG PRINT
            print("PETScNonlinearSolver solve: copying x0 into workingVector petscVec")    
            x0_vec.copy(self.workingVector)

            
            #DEBUG PRINT
            print("PETScNonlinearSolver solve: finished copying x0 into workingVector petscVec")    


            copy_time = perf_counter() - copy_start
            petsc_start = perf_counter()

            #DEBUG PRINT
            print("PETScNonlinearSolver solve: calling self.snes.solve")  
            self.snes.solve(None, self.workingVector)

            #DEBUG PRINT
            print("PETScNonlinearSolver solve: finished calling self.snes.solve")  


            petsc_time = perf_counter() - petsc_start
            self.last_diagnostics = {
                "total_s": perf_counter() - solve_start,
                "input_conversion_s": conversion_time,
                "initial_copy_s": copy_time,
                "snes_solve_s": petsc_time,
                "snes_iterations": self.snes.getIterationNumber(),
                "snes_function_norm": self.snes.getFunctionNorm(),
                "snes_ksp_iterations": self.snes.getKSP().getIterationNumber(),
                "callback_stats": dict(self.callback_stats or {}),
            }

            #DEBUG PRINT
            print("finished PETScNonlinearSolver solve")

            if self.diagnostics:
                print("PETSc SNES diagnostics:", self.last_diagnostics)
            return self.workingVector
        finally:
            x0_vec.destroy()  #I wonder, can we just keep using x0_vec and updating it?

    def solve_to_jax(self, x0):
        """Solve and explicitly copy the PETSc Vec result into a JAX array."""

        #DEBUG PRINT
        print("PETScNonlinearSolver solve_to_jax starting")

        
        x = self.solve(x0)

        #DEBUG PRINT
        print("PETScNonlinearSolver solve_to_jax: completed solve call")


        try:

            #DEBUG PRINT
            print("PETScNonlinearSolver solve_to_jax: starting petsc_vec_to_jax_array.copy()")

            result = petsc_vec_to_jax_array(x).copy()

            #DEBUG PRINT
            print("PETScNonlinearSolver solve_to_jax: finished petsc_vec_to_jax_array.copy()")

            result.block_until_ready()
            return result
        finally:
            x.destroy()

    def linear_solve(
        self,
        rhs: jnp.ndarray,
        x_star: jnp.ndarray | None = None,
        transpose: bool = False,
    ) -> jnp.ndarray:
        """Perform linear solve J(x*) dx = rhs (or J(x*)^T lam = rhs) for IFT differentiation."""
        ksp = self.snes.getKSP()
        rhs_vec = jax_array_to_petsc_vec(rhs)
        out_vec = rhs_vec.duplicate()
        try:
            if transpose:
                ksp.solveTranspose(rhs_vec, out_vec)
            else:
                ksp.solve(rhs_vec, out_vec)
            result = petsc_vec_to_jax_array(out_vec).copy()
            result.block_until_ready()
            return result
        finally:
            rhs_vec.destroy()
            out_vec.destroy()

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

        #DEBUG PRINT
        print("PETScNonlinearSolver destroy self called")

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
