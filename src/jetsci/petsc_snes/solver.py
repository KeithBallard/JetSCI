from __future__ import annotations

import warnings
from typing import Callable, Any
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from petsc4py import PETSc

from ..conversions import *
from ..options import (
    LinearSolverType,
    PreconditionerType,
    LinearSolverOptions,
    SolverOptions,
    NonlinearSolverType,
)
from ..coo_data import COOData, to_coo_data
from ..jax_linear import LinearSolverResultInfo
from ..petsc_ksp.linear_methods import (
    init_matrix_from_COOData,
    init_ksp,
    update_matrix_values,
)
from ..petsc_ksp.options import PETScMethodOptions


_PETSC_KSP_NAMES = {
    LinearSolverType.PETSC_CG: "cg",
    LinearSolverType.PETSC_GMRES: "gmres",
    LinearSolverType.PETSC_LGMRES: "lgmres",
    LinearSolverType.PETSC_BCGS: "bcgs",
    LinearSolverType.PETSC_PREONLY: "preonly",
    LinearSolverType.PETSC_MINRES: "minres",
}

_PETSC_PC_NAMES = {
    PreconditionerType.PETSC_NONE: "none",
    PreconditionerType.PETSC_JACOBI: "jacobi",
    PreconditionerType.PETSC_ILU: "ilu",
}


def _to_petsc_coo_and_metadata(A: Any) -> tuple[COOData, tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    if callable(A):
        raise TypeError(
            "PETSc linear solver cannot accept a callable operator. "
            "Callable operators cannot be converted to sparse matrices without evaluation loops (like jacfwd), "
            "which destroys performance. "
            "Fix: Use a matrix-free iterative solver (e.g., LinearSolverType.JAX_CG_SCIPY, LinearSolverType.JAX_GMRES_SCIPY) "
            "or provide an explicit sparse matrix (e.g., jsparse.COO or COOData)."
        )
    if isinstance(A, (jnp.ndarray, np.ndarray)):
        warnings.warn(
            "Dense array passed to PETSc linear solver. Converting dense matrix to COO format on every solve, "
            "which degrades performance and consumes extra memory. "
            "Fix: Pass a sparse COO matrix (e.g., jsparse.COO or COOData) directly.",
            UserWarning,
            stacklevel=3,
        )
        coo_data = convert_jax_dense_mat_to_coo_data(jnp.asarray(A))
    elif isinstance(A, tuple) and len(A) == 4:
        coo_data = to_coo_data(A)
    elif hasattr(A, "rows") and hasattr(A, "cols") and hasattr(A, "vals"):
        coo_data = A if isinstance(A, COOData) else COOData(
            shape=jnp.asarray(A.shape, dtype=jnp.int64),
            vals=A.vals,
            rows=jnp.asarray(A.rows, dtype=jnp.int32),
            cols=jnp.asarray(A.cols, dtype=jnp.int32),
        )
    elif isinstance(A, jsparse.COO):
        coo_data = COOData(
            shape=jnp.asarray(A.shape, dtype=jnp.int64),
            vals=A.data,
            rows=jnp.asarray(A.row, dtype=jnp.int32),
            cols=jnp.asarray(A.col, dtype=jnp.int32),
        )
    else:
        raise TypeError(
            f"Unsupported matrix type {type(A)} for PETSc linear solver. "
            "Expected COOData, (rows, cols, vals, shape) tuple, jsparse.COO, or dense array."
        )

    shape = (int(coo_data.shape[0]), int(coo_data.shape[1]))
    pattern = (np.asarray(coo_data.rows), np.asarray(coo_data.cols))
    return coo_data, shape, pattern


from ..conversions.mat_func_conversions import (
    _assign_petsc_mat_from_coo_data_prealloc,
    _assign_petsc_mat_from_coo_data_no_prealloc,
)


def _setup_petsc_linear_solver(
    solver: PETScLinearSolver,
    coo_data: COOData,
    options: LinearSolverOptions,
) -> None:
    mat_type = getattr(options, "petsc_mat_type", "aijcusparse")
    matrix = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    _assign_petsc_mat_from_coo_data_prealloc(
        matrix,
        coo_data,
        mat_type=mat_type,
    )
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOperators(matrix, matrix)
    ksp.setType(_PETSC_KSP_NAMES.get(options.linear_solver_type, "cg"))
    ksp.setTolerances(
        rtol=options.linear_relative_tol,
        atol=options.linear_absolute_tol,
        max_it=options.linear_max_iter,
    )
    pc = ksp.getPC()
    pc.setType(_PETSC_PC_NAMES.get(options.linear_preconditioner_type, "none"))
    ksp.setUp()

    solver.matrix_data = matrix
    solver.operator_matrix = matrix
    solver.ksp = ksp


def build_petsc_linear_solver(A: Any, options: LinearSolverOptions) -> PETScLinearSolver:
    """Build a standalone PETScLinearSolver from operator A and LinearSolverOptions."""
    coo_data, shape, pattern = _to_petsc_coo_and_metadata(A)
    solver = PETScLinearSolver(
        ksp=None,
        options=options,
        _coo_data=coo_data,
        _shape=shape,
        _sparsity_pattern=pattern,
    )
    _setup_petsc_linear_solver(solver, coo_data, options)
    return solver


@dataclass
class PETScLinearSolver:
    """Owns the PETSc KSP object to be used in linear solve and automatic differentiation.

    Can be constructed standalone via build_petsc_linear_solver(A, options) or
    as the companion to PETScNonlinearSolver.
    """

    ksp: object
    options: LinearSolverOptions
    vector_callback: Callable | None = None
    matrix_callback: Callable | None = None
    vector_data: object | None = None
    matrix_data: object | None = None
    operator_matrix: object | None = None
    working_vector: object | None = None
    last_info: LinearSolverResultInfo | None = None
    _coo_data: COOData | None = None
    _shape: tuple[int, int] | None = None
    _sparsity_pattern: Any | None = None

    def __post_init__(self):
        """Setup the KSP work objects if not already built."""
        self.working_vector = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
        if self.ksp is not None and self.matrix_data is None:
            self.vector_data = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
            self.matrix_data = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
            self.matrix_data.setType(getattr(self.options, "petsc_mat_type", "aijcusparse"))
            self.operator_matrix = self.matrix_data
            self.ksp.setOperators(self.operator_matrix, self.operator_matrix)

    def _ensure_size(self, x0: jnp.ndarray):
        """Ensure the vector and matrix storage are the correct size."""
        if self.vector_data is not None and self.vector_data.getType() is None:
            self.vector_data.setType("cuda")
            self.vector_data.setSizes((PETSc.DECIDE, x0.shape[0]))
            self.vector_data.setUp()
        if self.working_vector is not None and self.working_vector.getType() is None:
            self.working_vector.setType("cuda")
            self.working_vector.setSizes((PETSc.DECIDE, x0.shape[0]))
            self.working_vector.setUp()
        if self.matrix_data is not None and self.matrix_data.getType() is None:
            self.matrix_data.setSizes((x0.shape[0], x0.shape[0]))
            self.matrix_data.setUp()
            self.operator_matrix = self.matrix_data
            self.ksp.setOperators(self.operator_matrix, self.operator_matrix)

    def update_operator(self, operator_matrix):
        """Replace or update the linear operator used by the KSP."""
        if hasattr(operator_matrix, "setType"):
            # Direct PETSc Mat passed
            if self.matrix_data is not None and self.matrix_data is not operator_matrix:
                self.matrix_data.destroy()
            self.operator_matrix = operator_matrix
            self.matrix_data = operator_matrix
            self.ksp.setOperators(operator_matrix, operator_matrix)
            return self

        # Operator passed as JAX matrix or COOData
        new_coo, new_shape, new_pattern = _to_petsc_coo_and_metadata(operator_matrix)

        shape_changed = (self._shape is not None and new_shape is not None and self._shape != new_shape)
        pattern_changed = False
        if self._sparsity_pattern is not None and new_pattern is not None:
            old_r, old_c = self._sparsity_pattern
            new_r, new_c = new_pattern
            if len(old_r) != len(new_r) or not (np.array_equal(old_r, new_r) and np.array_equal(old_c, new_c)):
                pattern_changed = True
        elif (self._sparsity_pattern is None) != (new_pattern is None):
            pattern_changed = True

        if shape_changed or pattern_changed or self.matrix_data is None:
            reason = "shape changed" if shape_changed else "sparsity pattern changed"
            warnings.warn(
                f"PETScLinearSolver operator {reason} during update_operator. "
                "Rebuilding PETSc matrix and KSP solver.",
                UserWarning,
                stacklevel=2,
            )
            self.cleanup_work_vectors()
            if self.ksp is not None:
                self.ksp.destroy()
                self.ksp = None
            self._coo_data = new_coo
            self._shape = new_shape
            self._sparsity_pattern = new_pattern
            _setup_petsc_linear_solver(self, new_coo, self.options)
        else:
            self._coo_data = new_coo
            _assign_petsc_mat_from_coo_data_no_prealloc(self.matrix_data, new_coo)
            self.ksp.setOperators(self.matrix_data, self.matrix_data)

        return self

    def solve(self, rhs: jnp.ndarray, transpose: bool = False, x0: jnp.ndarray | None = None) -> jnp.ndarray:
        """Solve A x = rhs (or A^T x = rhs if transpose=True), returning a JAX array."""
        if transpose:
            return self.solve_transpose_to_jax(rhs, x0=x0)
        return self.solve_to_jax(rhs, x0=x0)

    def linear_solve(
        self,
        rhs: jnp.ndarray,
        *,
        x_linearized: jnp.ndarray | None = None,
        x_0: jnp.ndarray | None = None,
        transpose: bool = False,
    ) -> jnp.ndarray:
        """Solve with the common differentiable-linear-solve protocol.

        The standalone KSP already owns its operator, so ``x_linearized`` is
        accepted for interface compatibility but does not alter the solve.
        """
        del x_linearized
        return self.solve(rhs, transpose=transpose, x0=x_0)

    def transpose_linear_solve(self, rhs: jnp.ndarray):
        """Solve adjoint problem with this KSP object and return a JAX array."""
        return self.solve(rhs, transpose=True)

    def solve_to_jax(self, rhs: jnp.ndarray, x0: jnp.ndarray | None = None) -> jnp.ndarray:
        """Solve and explicitly copy the PETSc Vec result into a JAX array."""
        self._ensure_size(rhs)
        rhs_vec = jax_array_to_petsc_vec(rhs)
        x = self.working_vector if self.working_vector is not None else rhs_vec.duplicate() #TODO it refactored working vector, check that it did so correctly
        if x0 is not None:
            x0_vec = jax_array_to_petsc_vec(x0)
            x0_vec.copy(x)
            x0_vec.destroy()
            self.ksp.setInitialGuessNonzero(True)
        else:
            self.ksp.setInitialGuessNonzero(False)
        try:
            self.ksp.solve(rhs_vec, x)
            iters = self.ksp.getIterationNumber()
            self.last_info = LinearSolverResultInfo(iterations=iters)
            result = petsc_vec_to_jax_array(x).copy()
            result.block_until_ready()
            return result
        finally:
            rhs_vec.destroy()
            if self.working_vector is None:
                x.destroy()

    def solve_transpose_to_jax(self, rhs: jnp.ndarray, x0: jnp.ndarray | None = None) -> jnp.ndarray:
        """Solve transpose and explicitly copy the PETSc Vec result into a JAX array."""
        self._ensure_size(rhs)
        rhs_vec = jax_array_to_petsc_vec(rhs)
        x = self.working_vector if self.working_vector is not None else rhs_vec.duplicate()
        if x0 is not None:
            x0_vec = jax_array_to_petsc_vec(x0)
            x0_vec.copy(x)
            x0_vec.destroy()
            self.ksp.setInitialGuessNonzero(True)
        else:
            self.ksp.setInitialGuessNonzero(False)
        try:
            self.ksp.solveTranspose(rhs_vec, x)
            iters = self.ksp.getIterationNumber()
            self.last_info = LinearSolverResultInfo(iterations=iters)
            result = petsc_vec_to_jax_array(x).copy()
            result.block_until_ready()
            return result
        finally:
            rhs_vec.destroy()
            if self.working_vector is None:
                x.destroy()

    def block_linear_solve(self, rhs_block):
        output_block = rhs_block.duplicate()
        self.ksp.matSolve(rhs_block, output_block)
        return output_block

    def cleanup_work_vectors(self):
        """Destroy work objects."""
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
        self.cleanup_work_vectors()
        if self.ksp is not None:
            self.ksp.destroy()
            self.ksp = None

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
        """Solve nonlinear problem with this SNES object and return a PETSc Vec.

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

        print("!!!!!!!!!!!!!!!!!!!!!!!!! x passed to nonlinearsolver:",jnp.linalg.norm(x0))

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
        *,
        x_linearized: jnp.ndarray | None = None,
        x_0: jnp.ndarray | None = None,
        transpose: bool = False,
    ) -> jnp.ndarray:
        """Solve with this SNES object's current KSP linearization.

        ``x_linearized`` names the state at which the SNES Jacobian was
        assembled. The PETSc operator is already owned by SNES, so the value
        is not recomputed here. ``x_0`` is an optional KSP initial guess.
        """
        del x_linearized
        ksp = self.snes.getKSP()
        rhs_vec = jax_array_to_petsc_vec(rhs)
        out_vec = rhs_vec.duplicate()
        try:
            if x_0 is None:
                ksp.setInitialGuessNonzero(False)
            else:
                x0_vec = jax_array_to_petsc_vec(x_0)
                try:
                    x0_vec.copy(out_vec)
                finally:
                    x0_vec.destroy()
                ksp.setInitialGuessNonzero(True)
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
        if not options.linear_solver_type.is_petsc:
            raise TypeError(
                "PETSc SNES requires a PETSc linear solver method. "
                f"Got {options.linear_solver_type!r}."
            )
        if not options.linear_preconditioner_type.is_petsc:
            raise TypeError(
                "PETSc SNES requires a PETSc preconditioner method. "
                f"Got {options.linear_preconditioner_type!r}."
            )
