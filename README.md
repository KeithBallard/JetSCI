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

## Running Tests

To run the test suite:

```bash
pytest tests
```
