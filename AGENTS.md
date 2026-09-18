# Agent Guidelines & Design Principles

## Anti-Pattern Rules: Fail-Fast, Explicit Configuration, and Zero Silent Degradation

When authoring or modifying code in `JetSCI` and `fea-in-jax`, strictly adhere to the following principles:

### 1. No Silent Defaulting of Critical Options
- **Never silently substitute default solver types or algorithms** when an option is `None`.
  - *Anti-Pattern*: `if solver_type is None: solver_type = JAXLinearSolverType.DENSE_INVERSE_JNP`
  - *Required*: Raise a descriptive `ValueError` or `TypeError` stating that the option is required, what values are accepted, and how to configure it.
- **Tolerances and Numerical Parameters**: If optional tolerances default to fallback values, document them clearly and issue a `UserWarning` if unconfigured inputs could mask test configuration mistakes or unintended behavior.

### 2. No Silent Conversions Between Sparse, Dense, and Matrix-Free Forms
- **Sparse Solvers Require Sparse Data**: Direct sparse solvers (e.g., `SPSOLVE_CUPY`, `LU_CUPY`, `SPSOLVE_PYPARDISO`, PETSc AIJ) must NEVER silently convert dense matrices or callable operators into sparse formats without warning or raising an error.
  - If a **callable operator** is passed to a sparse direct solver: **Raise `TypeError`** immediately. Callable operators cannot be converted to sparse matrices without evaluation loops (like `jacfwd`), which destroys performance. Instruct the user to use a matrix-free iterative solver (e.g. `CG_JAX_SCIPY`, `GMRES_JAX_SCIPY`).
  - If a **dense array** is passed to a sparse solver: Emit an explicit `UserWarning` explaining the conversion overhead and memory penalty, and advise passing a native sparse representation (`jsparse.COO` or `COOData`).
- **Dense Solvers Must Not Silently Convert Sparse/Callable**:
  - If `DENSE_INVERSE_JNP` or a dense solver receives a sparse matrix or callable operator, emit an explicit `UserWarning` detailing that $O(N^2)$ memory allocation is taking place and provide recommendations for sparse or iterative solvers.

### 3. No Silent Fallbacks / Exception Swallowing
- **Never use bare `except Exception: pass`** to fall back between GPU and CPU or between different solvers (e.g., CuPy -> SciPy).
  - If a fallback is necessary, log an explicit `UserWarning` detailing the original failure exception and warning about the performance degradation.
  - Fail fast with informative exceptions when mandatory dependencies or capabilities are missing.

### 4. Meaningful, Actionable Error Messages
- All exceptions must include:
  1. What was expected and what was received.
  2. The performance, correctness, or testing implications.
  3. Actionable instructions on how to resolve the issue (e.g., specify `solver_type=...` or use `JAXLinearSolverType.CG_JAX_SCIPY`).
