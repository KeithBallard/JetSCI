# JetSCI

`jetsci` is a library for sparse linear and nonlinear solver utilities using JAX.

## Features

*   **GPU Acceleration**: Native support for hardware acceleration via JAX and CuPy.
*   **Differentiability**: Fully compatible with JAX's composable function transformations (`jacfwd`, `jacrev`, `jvp`, `grad`), enabling gradient-based optimization and machine learning workflows.
*   **Extensible Solver Configurations**: Unified interface for both PETSc (`PETSC_SNES`) and native JAX Newton-Raphson (`JAX_NEWTON_RAPHSON`) nonlinear solvers.
*   **Comprehensive Linear Solver Suite**: 14 selectable linear solver options across JAX-native, JAXOpt, CuPy, and PyPardiso backends:
    *   *JAX Native*: `DENSE_INVERSE_JNP`, `CG_JAX_SCIPY`, `CG_JAX_SCIPY_W_INFO` (with iteration & residual history), `GMRES_JAX_SCIPY`, `BICGSTAB_JAX_SCIPY`
    *   *JAXOpt*: `DENSE_INVERSE_JAXOPT`, `LU_JAXOPT`, `CHOLESKY_JAXOPT`, `CG_JAXOPT`, `GMRES_JAXOPT`, `BICGSTAB_JAXOPT`
    *   *External / Sparse*: `SPSOLVE_CUPY`, `LU_CUPY`, `SPSOLVE_PYPARDISO`
*   **Preconditioners**: `NONE`, `JACOBI` (diagonal inverse), and `ILU_CUPY` (CuPy incomplete LU factorization).
*   **Unified Differentiable Interface**: `differentiable_solve` transparently applies the Implicit Function Theorem (IFT) across all nonlinear solvers and linear solver backends.

## Project Structure

*   `src/jetsci`: Core solver library source code.
    *   `jax_linear/`: Implementation of JAX-native linear solvers, preconditioners, and CG diagnostic tracking.
    *   `jax_newton_raph/`: JAX-native Newton-Raphson nonlinear solver with lifecycle reuse and companion adjoint solves.
    *   `petsc_snes/`: PETSc SNES nonlinear solver with zero-copy DLPack memory exchange.
    *   `solve.py`: Unified `differentiable_solve` and `differentiable_linear_solve` functions.
*   `tests`: Comprehensive unit and autodiff test suite.

## Getting Started

`jetsci` can be installed in two modes depending on your hardware and performance requirements:

> [!IMPORTANT]
> **GPU-enabled installation (Path B) is highly recommended.** The primary performance advantages of JAX and `jetsci` come from hardware (GPU) acceleration. While CPU-only installations are extremely useful for quick testing, local prototyping, and basic debugging, they do not provide high performance.

*   **Path A: CPU-Only Installation** (Quick & easy; suited for rapid local testing and debugging, but lacks high performance)
*   **Path B: GPU & PETSc Installation (Recommended)** (High-performance; utilizes JAX/CuPy on GPUs and compiles PETSc with CUDA support for GPU-accelerated nonlinear (`PETSC_SNES`) and linear (`PETSC_KSP`) solver methods)

---

### Prerequisites

Regardless of the installation path, ensure you have:
*   **Python 3.10+** (Python 3.12 is fully supported)
*   An active virtual environment (recommended)

---

### Path A: CPU-Only Installation (Testing & Debugging)

This path is intended for quick local tests, basic development, and debugging of solver logic, or if you do not have access to an NVIDIA GPU. It is not designed for performance-critical workloads.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KeithBallard/JetSCI.git
   cd JetSCI
   ```

2. **Set up and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the library and test dependencies in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```
   This installs standard CPU-only JAX, NumPy, SciPy, and Pytest.

---

### Path B: GPU & PETSc Installation (Advanced)

This path enables high-performance sparse GPU solvers by compiling PETSc with CUDA support and building `petsc4py` inside your environment. 

> [!IMPORTANT]
> To prevent runtime linking errors (such as missing `libcublas.so` or `libcurand.so`), your system's **host CUDA Toolkit version must match your Python package CUDA version** (e.g., CUDA 12.x).

#### Step 1: Install CUDA Toolkit on the Host
If you don't already have the CUDA Toolkit installed on your host system, install it using your package manager. For example, on Ubuntu 24.04:

```bash
# Download and install the CUDA repository keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA Toolkit (matching your target JAX version, e.g., CUDA 12)
sudo apt install cuda-toolkit-12-9
```
*Note: Make sure to CUDA compilers (e.g. nvcc) is in your path, if not append `/usr/local/cuda/bin` to `PATH` and `/usr/local/cuda/lib64` to `LD_LIBRARY_PATH`.*

#### Step 2: Set up Virtual Environment & Install Python Dependencies
1. **Clone the repository:**
   ```bash
   git clone https://github.com/KeithBallard/JetSCI.git
   cd JetSCI
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install GPU-enabled Python dependencies:**
   Depending on your CUDA version (12 or 13), run:
   ```bash
   # For CUDA 12.x
   pip install -e ".[cuda12,dev]"

   # For CUDA 13.x (Note: make sure the version matches your CUDA toolkit version)
   pip install -e ".[cuda13,dev]"
   ```
   You can check the installed versions using `pip freeze`.

#### Step 3: Clone and Compile PETSc
To use PETSc's advanced solver configurations on the GPU, you must compile PETSc with CUDA enabled.

1. **Clone the PETSc repository:**
   ```bash
   git clone https://gitlab.com/petsc/petsc.git petsc
   cd petsc
   git checkout main  # Or a specific release tag/branch
   ```

2. **Configure PETSc:**
   We configure PETSc with CUDA support and disable MPI (if running on a single GPU). We also download LAPACK/BLAS dependencies automatically.

   *   **Debug Mode (Default):** Includes full assertion checks, memory tracing, and debugging symbols. Ideal for development and troubleshooting:
       ```bash
       ./configure --with-cuda --with-mpi=0 --download-f2cblaslapack=1 --with-debugging=1
       ```
       *(Note: `--with-debugging=1` is PETSc's default when omitted, producing `PETSC_ARCH=arch-linux-c-debug`)*.

   *   **Release Mode (Optimized):** Disables internal debugging assertions and enables compiler optimizations (`-O3`), recommended for high performance and production benchmarking:
       ```bash
       ./configure --with-cuda --with-mpi=0 --download-f2cblaslapack=1 --with-debugging=0 --COPTFLAGS="-O3" --CXXOPTFLAGS="-O3" --CUDAOPTFLAGS="-O3"
       ```
       *(Produces `PETSC_ARCH=arch-linux-c-opt`)*.

3. **Build PETSc:**
   After configuration completes, compile PETSc using the command corresponding to your chosen build mode:

   *   **For Debug Mode:**
       ```bash
       make PETSC_DIR=/absolute/path/to/petsc PETSC_ARCH=arch-linux-c-debug all
       ```

   *   **For Release Mode:**
       ```bash
       make PETSC_DIR=/absolute/path/to/petsc PETSC_ARCH=arch-linux-c-opt all
       ```
   *(Be sure to replace `/absolute/path/to/petsc` with your actual absolute path to the petsc directory, which can be acquired via `pwd`)*.

#### Step 4: Build and Install `petsc4py`
Once PETSc is successfully compiled, you can build and install its Python bindings (`petsc4py`) inside your virtual environment.

1. **Set PETSc Environment Variables:**
   Ensure these variables point to your compiled PETSc installation matching your build architecture:
   ```bash
   export PETSC_DIR=/absolute/path/to/petsc

   # For Debug Mode:
   export PETSC_ARCH=arch-linux-c-debug

   # OR for Release Mode:
   export PETSC_ARCH=arch-linux-c-opt
   ```

2. **Navigate to Python Bindings:**
   Inside the cloned `petsc` repository:
   ```bash
   cd src/binding/petsc4py/
   ```

3. **Install in your Virtual Environment:**
   ```bash
   python -m pip install .
   ```

#### Step 5: Verify the Installation
To confirm that PETSc, JAX, and `petsc4py` are integrated successfully, return to your active shell and verify:

1. **Test import and file location:**
   ```bash
   python -c "from petsc4py import PETSc; print('petsc4py successfully installed at:', PETSc.__file__)"
   ```

2. **Ensure dynamic libraries are resolved:**
   Ensure your shell has the system CUDA path in `LD_LIBRARY_PATH` so compiled objects can find GPU runtime libraries:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
   ```

---

## Usage Example
 
```python
import jax
import jax.numpy as jnp
import jetsci
from jetsci.options import (
    NonlinearSolverType,
    JAXLinearSolverType,
    JAXPreconditionerType,
    PETScLinearSolverType,
    PETScPreconditionerType,
    SolverOptions,
)

# Define parameterized residual: R(p, x) = 0
def residual(p, x):
    return jnp.array([
        x[0]**3 + x[1] - p[0],
        x[0] + x[1]**3 - p[1],
    ])

p = jnp.array([2.0, 2.0], dtype=jnp.float64)
x0 = jnp.array([0.9, 0.9], dtype=jnp.float64)

# 1. Solve using JAX Newton-Raphson with CG or GMRES
jax_opts = SolverOptions(
    nonlinear_solver_type=NonlinearSolverType.JAX_NEWTON_RAPHSON,
    linear_solve_type=JAXLinearSolverType.CG_JAX_SCIPY,
    linear_precond_type=JAXPreconditionerType.JACOBI,
)
sol_jax, _ = jetsci.differentiable_solve(jax_opts, residual, None, x0, p)
grad_jax = jax.grad(lambda param: jnp.sum(jetsci.differentiable_solve(jax_opts, residual, None, x0, param)[0]))(p)

# 2. Solve using PETSc SNES with BCGS
petsc_opts = SolverOptions(
    nonlinear_solver_type=NonlinearSolverType.PETSC_SNES,
    linear_solve_type=PETScLinearSolverType.BCGS,
    linear_precond_type=PETScPreconditionerType.NONE,
)
sol_petsc, _ = jetsci.differentiable_solve(petsc_opts, residual, None, x0, p)
grad_petsc = jax.grad(lambda param: jnp.sum(jetsci.differentiable_solve(petsc_opts, residual, None, x0, param)[0]))(p)

print("JAX Solution:", sol_jax)      # [1.0, 1.0]
print("PETSc Solution:", sol_petsc)  # [1.0, 1.0]
print("JAX Gradient:", grad_jax)     # [0.25, 0.25]
print("PETSc Gradient:", grad_petsc) # [0.25, 0.25]
```

---

## Running Tests

> [!IMPORTANT]
> **Run `pytest` from the `JetSCI` root directory.**  
> The test configuration (`[tool.pytest.ini_options]`) is defined in [`pyproject.toml`](file:///home/user/JetSCI/pyproject.toml), which configures `pythonpath = ["src", "tests"]` and `testpaths = ["tests"]`. Executing `pytest` from the repository root ensures that `pyproject.toml` is detected, test modules are discovered properly, and `src` is placed on Python's module search path.

To run the test suite:

```bash
# Run the full pytest suite (all tests)
pytest

# Or run individual test suites with verbose output:
pytest tests/test_linear_solvers.py -v
pytest tests/test_jax_newton_raphson.py -v
pytest tests/test_differentiable_solve_autodiff.py -v
```

