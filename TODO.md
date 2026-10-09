
- Add other JAX nonlinear methods
- Add better error handling for PETSc's nonlinear solver. It just fails silently right now

## Performance audit (2026-10-09)

Prioritize these JetSCI implementation improvements before test-level tuning:

1. Remove duplicate PETSc Vec allocation, sizing, input-copy, and cleanup code
   in `petsc_snes/solver.py`. The duplicate work also leaks the first
   `linear_rhs_vec`/`linear_out_vec` pair and can double-destroy temporary
   RHS vectors in the standalone KSP path.
2. Remove unconditional debug prints from residual, Jacobian, and solution
   conversion callbacks. They run in solver hot paths and serialize Python I/O.
3. Use fixed-pattern COO preallocation by default for fixed FE meshes. The
   current pattern-aware check performs three GPU comparisons followed by
   `.item()` host synchronizations on every Jacobian callback.
4. Cache the ctypes binding for `MatSetValuesCOO` instead of loading/configuring
   it on every Jacobian value update.
5. Remove duplicate PETSc callback refreshes and avoid resetting KSP operators
   when the existing SNES Jacobian Mat is already installed.
6. Refactor the generic differentiable solve so its primal result is the SNES
   solution directly; do not perform an extra KSP solve merely to reconstruct
   that already-known primal state.
7. Redesign the multi-RHS `KSP.matSolve` bridge around reusable PETSc dense RHS
   and output matrices. The current path allocates/duplicates matrices and
   converts the result per call, so it does not yet realize the intended
   batching benefit.
8. Add value/pattern-aware reuse for JAX preconditioners and sparse-direct
   factorizations. `JAXLinearSolver.update_operator` currently rebuilds a
   preconditioner on every value update.

Notes: `pure_callback` paths host-stage data and execute vmapped calls
sequentially. Replacing them with `buffer_callback` alone did not improve the
benchmark, likely because the experimental path creates temporary PETSc
wrappers per solve; any device-native replacement should use persistent native
objects and be profiled on the target GPU/Linux environment.
