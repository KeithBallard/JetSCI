# JetSCI Differentiable Solver Architecture & Unified API Design Document

## 1. Executive Summary & Motivation

`JetSCI` provides differentiable linear and nonlinear solvers for scientific computing systems of equations integrated with Google JAX. Originally, JetSCI had distinct backend implementations—JAX-native solvers (`jax_linear`, `jax_newton_raph`) and PETSc-based solvers (`petsc_ksp`, `petsc_snes`)—which suffered from fragmented front-end options, divergent solver lifecycle patterns, backend-specific enums (`JAXLinearSolverType`, `PETScLinearSolverType`, etc.), and inconsistent differentiation interfaces.

This redesign unifies the public front-end API across all backends into a single cohesive, fail-fast, differentiable architecture. Key achievements include:
1. **Unified Options Hierarchy**: `LinearSolverOptions` serves as the base configuration for linear solves; `SolverOptions` inherits from `LinearSolverOptions` and adds nonlinear configuration parameters.
2. **Backend Prefixing**: All linear solver types and preconditioner types are consolidated into single enums (`LinearSolverType` and `PreconditionerType`) with explicit backend prefixes (`JAX_*` and `PETSC_*`), eliminating silent backend confusion.
3. **Fail-Fast Backend Alignment**: Strict validation at initialization ensures that linear solver, preconditioner, and nonlinear solver types all share the same backend, failing fast with actionable error messages.
4. **Standalone Linear Solver Lifecycle with Key-Based Reuse**: Internal factory `build_linear_solver_with_reuse(options, A)` constructs standalone linear solvers with identical object APIs across JAX and PETSc (`solve(b, transpose=False)`, `update_operator(A)`, `destroy()`), allowing persistent resource reuse.
5. **Operator Mutability and Sparsity Tracking**: `update_operator(A)` dynamically checks shape and sparsity pattern preservation. In-place matrix updates are performed if the pattern is preserved; underlying solver structures are rebuilt with an informative `UserWarning` if shape or non-zero sparsity patterns change.
6. **Pure Adjoint / Implicit Function Theorem Differentiation**: Linear and nonlinear systems are differentiated via `jax.lax.custom_linear_solve` rather than unrolling solver iterations, ensuring mathematically exact derivatives and constant memory scaling for both forward-mode (`jacfwd`, `jvp`) and reverse-mode (`jacrev`, `grad`, `vjp`) automatic differentiation.

---

## 2. API Architecture & Options Hierarchy

### 2.1 Unified Enums

Previously, `jetsci.options` defined separate enums for JAX and PETSc:
- `JAXLinearSolverType` vs. `PETScLinearSolverType`
- `JAXPreconditionerType` vs. `PETScPreconditionerType`

These have been completely replaced with unified enums using backend prefixes:

```python
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
```

### 2.2 Configuration Dataclasses: Inheritance and Prefix Convention

To maintain a consistent naming scheme across both linear and nonlinear solver options, all linear-related fields consistently use the `linear_` prefix:

```python
@dataclass(frozen=True)
class LinearSolverOptions:
    linear_solver_type: LinearSolverType
    linear_preconditioner_type: PreconditionerType
    linear_max_iter: int = 100
    linear_relative_tol: float = 1e-5
    linear_absolute_tol: float = 1e-50
    solver_key: int | None = None
    petsc_mat_type: str = "aijcusparse"

    def __post_init__(self):
        # Fail-fast validation: Linear solver and preconditioner backends must match
        ...

@dataclass(frozen=True)
class SolverOptions(LinearSolverOptions):
    nonlinear_solver_type: NonlinearSolverType = NonlinearSolverType.JAX_NEWTON_RAPHSON
    nonlinear_max_iter: int = 50
    nonlinear_relative_tol: float = 1e-6
    nonlinear_absolute_tol: float = 1e-12

    def __post_init__(self):
        super().__post_init__()
        # Fail-fast validation: Nonlinear solver backend must match linear solver backend
        ...
```

### 2.3 Backend Validation Rules
1. `linear_solver_type` and `linear_preconditioner_type` are mandatory (no silent defaulting).
2. If `linear_solver_type` has prefix `JAX_`, `linear_preconditioner_type` must have prefix `JAX_`.
3. If `linear_solver_type` has prefix `PETSC_`, `linear_preconditioner_type` must have prefix `PETSC_`.
4. If `nonlinear_solver_type` is specified (in `SolverOptions`), its backend must match the linear solver backend:
   - `JAX_NEWTON_RAPHSON` requires `linear_solver_type.is_jax`
   - `PETSC_SNES` requires `linear_solver_type.is_petsc`

Violations immediately raise a clear, actionable `ValueError` detailing the incompatible backends and required fix.

---

## 3. Solver Lifecycle & Standalone Linear Solver Architecture

### 3.1 Front-End Solve Signatures

The top-level user entrypoints are symmetric between linear and nonlinear solves:

```python
# Nonlinear differentiable solve
def differentiable_solve(
    solver_options: SolverOptions,
    R: Callable,
    J_x: Optional[Callable],
    x_0: jnp.ndarray,
    *args,
) -> tuple[jnp.ndarray, SolverOptions]:
    ...

# Linear differentiable solve
def differentiable_linear_solve(
    solver_options: LinearSolverOptions,
    A: Any,
    b: jnp.ndarray,
    transpose: bool = False,
) -> tuple[jnp.ndarray, LinearSolverOptions]:
    ...
```

Both routines return a tuple of `(solution, updated_options)`. The updated options contain `solver_key`, which can be passed to subsequent solve calls to reuse pre-allocated solver infrastructure and memory.

### 3.2 Internal Standalone Linear Solvers

Standalone linear solvers are constructed internally via:
```python
build_linear_solver_with_reuse(options: LinearSolverOptions, A: Any) -> tuple[LinearSolver, LinearSolverOptions]
```
If `options.solver_key is None`, a new solver instance is constructed, assigned an integer key, and cached in the internal registry `__linear_solver_dict`. If `options.solver_key` is present, the existing solver is retrieved and updated via `solver.update_operator(A)`.

### 3.3 Unified Linear Solver Protocol

Both `JAXLinearSolver` and `PETScLinearSolver` adhere to an identical contract:

| Method | Signature | Description |
|---|---|---|
| `solve` | `solve(b: jnp.ndarray, transpose: bool = False) -> jnp.ndarray` | Solves $A x = b$ (or $A^T x = b$ if `transpose=True`). Returns JAX array. |
| `update_operator` | `update_operator(A: Any) -> self` | Updates the linear operator. Verifies shape and sparsity pattern match; updates in-place if unchanged; rebuilds underlying solver and issues `UserWarning` if shape/pattern changed. |
| `destroy` | `destroy() -> None` | Destroys allocated C/GPU/CUDA resources (e.g., PETSc Mat/Vec/KSP/PC). |

### 3.4 Operator Sparsity Pattern Tracking

In iterative and direct solvers, changing matrix sparsity requires reallocating symbolic factorizations, non-zero index structures, or GPU preallocations.
- In `update_operator(A)`:
  - Shape and non-zero indices (for sparse matrices like `jsparse.COO` or `COOData`) are compared against cached values from construction.
  - If identical: matrix values are updated in-place with zero memory allocation.
  - If modified: a `UserWarning` is raised:
    `UserWarning: Operator sparsity pattern or shape changed in <SolverClass>. Rebuilding underlying solver objects. This degrades performance.`
    and underlying preconditioners and solver objects are rebuilt.

---

## 4. Automatic Differentiation via Adjoint / Implicit Function Theorem (IFT)

### 4.1 Principle: Zero Unrolling of Solver Iterations

Differentiating an iterative solver by unrolling its iterations via standard reverse-mode AD leads to:
1. $O(K)$ memory overhead where $K$ is the number of solver iterations.
2. Numerical inaccuracy if iterations terminate early based on tolerances.
3. Incompatibility with external, non-JAX compiled C/GPU solvers such as PETSc or CuPy.

Instead, JetSCI implements differentiation via the Implicit Function Theorem (IFT) using `jax.lax.custom_linear_solve`.

### 4.2 Mathematical Formulation

For a linear system $A(\theta) x = b(\theta)$:
- **Primal problem**:
  $$A x = b$$
- **Forward-Mode Tangent (JVP)**:
  Differentiating $A x = b$ with respect to a scalar perturbation gives:
  $$\dot{A} x + A \dot{x} = \dot{b} \implies A \dot{x} = \dot{b} - \dot{A} x$$
  The tangent $\dot{x}$ is obtained by solving the original linear system with RHS $\dot{b} - \dot{A} x$.
- **Reverse-Mode Cotangent (VJP / Adjoint)**:
  For a scalar objective $L(x)$, with cotangent $\bar{x} = \nabla_x L$:
  $$A^T \lambda = \bar{x}$$
  The adjoint $\lambda$ is obtained by solving the transpose linear system. Gradients then propagate through $\bar{b} = \lambda$ and $\bar{A} = -\lambda x^T$.

### 4.3 JAX Implementation

Inside `differentiable_linear_solve`:
```python
def solve_fn(mv, rhs):
    res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
    def _cb(r):
        return solver.solve(r, transpose=transpose)
    return jax.pure_callback(_cb, res_info, rhs, vmap_method="sequential")

def trans_solve_fn(mv, rhs):
    res_info = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
    def _cb(r):
        return solver.solve(r, transpose=not transpose)
    return jax.pure_callback(_cb, res_info, rhs, vmap_method="sequential")

x = jax.lax.custom_linear_solve(
    matvec,
    b,
    solve=solve_fn,
    transpose_solve=trans_solve_fn,
    symmetric=is_symmetric,
)
```
- `matvec` is constructed directly from $A$ using differentiable JAX operations (`lambda v: A @ v`, or coordinate indexing for `jsparse.COO`), allowing JAX to trace $\dot{A} x$ in forward mode and $\bar{A}$ in reverse mode.
- `solve_fn` and `trans_solve_fn` invoke `solver.solve(...)` through `jax.pure_callback`. Because `custom_linear_solve` treats `solve_fn` and `trans_solve_fn` as exact solution oracles and never differentiates through them directly, the solver iterations are never unrolled.
- Both forward-mode (`jacfwd`, `jvp`) and reverse-mode (`jacrev`, `grad`, `vjp`) work seamlessly with identical behavior across JAX and PETSc backends.

---

## 5. Compliance with Anti-Pattern Rules (`AGENTS.md`)

| Rule | Enforcement in JetSCI |
|---|---|
| **1. No Silent Defaulting** | `linear_solver_type` and `linear_preconditioner_type` must be explicitly provided in `LinearSolverOptions`. Passing `None` or leaving them unspecified raises `ValueError`. |
| **2. No Silent Format Conversions** | Passing a dense array to a sparse direct solver (e.g. `JAX_SPSOLVE_CUPY`, `JAX_SPSOLVE_PYPARDISO`) emits a prominent `UserWarning`. Passing a callable operator to a sparse direct solver immediately raises `TypeError`. |
| **3. No Silent Fallbacks** | No bare `try...except Exception: pass`. If GPU solvers or external libraries are unavailable, explicit informative exceptions are raised with guidance on required dependencies. |
| **4. Meaningful, Actionable Errors** | All error messages clearly state: (a) what was expected, (b) what was received, and (c) an actionable fix with code examples. |
