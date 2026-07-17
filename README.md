# JetSCI

`jetsci` is a library for sparse linear and nonlinear solver utilities using JAX. It provides high-performance solver structures separated from the Finite Element Analysis (FEA) pipeline.

## Features

*   **GPU Acceleration**: Native support for hardware acceleration via JAX and CuPy.
*   **Differentiability**: Fully compatible with JAX's composable function transformations, enabling gradient-based optimization and machine learning workflows.
*   **Extensible Solver Configurations**: Standard interfaces for Newton-Raphson, Conjugate Gradient (CG), GMRES, BiCGSTAB, and direct solvers (LU, Cholesky).

## Project Structure

*   `src/jetsci`: Core solver library source code, including solver options, solver classes, and linear/nonlinear interfaces.
*   `tests`: Unit tests and integration tests.

## Getting Started

### Prerequisites

*   Python 3.10+
*   [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (optional, strictly for GPU acceleration)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd JetSCI
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    For development, install the package in editable mode with test dependencies:
    ```bash
    pip install -e ".[dev]"
    ```

4. **TODO document petsc4py**
`pip install "jax[cuda12]"`
`pip freeze` (note version)
`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb`
`sudo dpkg -i cuda-keyring_1.1-1_all.deb`
`sudo apt update`
`sudo apt install cuda-toolkit-12-9` (for example, match pip freeze version)
`git clone https://gitlab.com/petsc/petsc.git petsc`
`cd petsc`
`git checkout main`
`./configure --with-cuda --with-mpi=0 --download-f2cblaslapack=1`
`make PETSC_DIR=/home/user/petsc PETSC_ARCH=arch-linux-c-debug all`

`export PETSC_DIR=/path/to/petsc`
`export PETSC_ARCH=your-petsc-architecture`
`cd ./src/binding/petsc4py/`
`python -m pip install .`
`python -c "from petsc4py import PETSc; print(PETSc.__file__)"` (verify)

## Running Tests

To run the test suite:

```bash
pytest tests
```
