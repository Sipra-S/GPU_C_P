# CPU Correlation Matrix Benchmark Code

This repository contains a Python implementation for benchmarking CPU-based correlation matrix computation for large-scale time-series datasets.

Developed for **DS3294 – DS Practice Project #9**

Made by: **Sipra Subhadarsini Sahoo**

---

# Project Description

The code benchmarks correlation matrix computation using two CPU computation strategies:

1. Single-threaded baseline implementation
2. NumPy vectorised parallel computation

The project measures:
- execution runtime,
- peak memory usage,
- parallel speedup,
- and correlation matrix generation performance.

---

# Mathematical Background

For `N` time series of length `L`, the computational complexity scales as:

```text
O(N² · L)
```

The correlation matrix is computed as:

```text
C = (1 / L) ZZᵀ
```

where:
- `Z` is the standardized dataset,
- and `C` is the Pearson correlation matrix.

---

# Implemented Methods

## 1. Single-threaded Baseline

A loop-based implementation using explicit nested iteration.

### Features
- Sequential CPU execution
- Baseline reference implementation
- Simple and interpretable computation

---

## 2. NumPy Parallel Computation

A vectorised implementation using NumPy matrix multiplication.

### Features
- BLAS-backed optimized linear algebra
- Internally parallelized execution
- Faster computation for large datasets

---

# File Structure

```text
.
├── cpu_correlation.py
└── README.md
```

---

# Requirements

## Python Version

Recommended:

```text
Python 3.9+
```

---

# Dependencies

Install required libraries using:

```bash
pip install numpy matplotlib
```

---

# Library Usage

## NumPy

Used for:
- dataset generation,
- vectorised computation,
- matrix multiplication,
- and numerical operations.

---

## Matplotlib

Used for:
- runtime visualization,
- memory usage plots,
- speedup analysis,
- and correlation heatmaps.

---

# Running the Code

Execute:

```bash
python cpu_correlation.py
```

or run directly inside:
- Google Colab
- Jupyter Notebook
- VS Code
- or any Python IDE.

---

# Benchmark Experiments

The script benchmarks multiple dataset sizes:

```python
experiments = [
    (64, 1000),
    (256, 1000),
    (512, 2000),
    (1024, 2000),
]
```

where:
- first value = number of time series (`N`)
- second value = series length (`L`)

---

# Generated Outputs

The code automatically generates:

## Runtime Plot

```text
runtime_vs_n.png
```

---

## Memory Usage Plot

```text
memory_vs_n.png
```

---

## Parallel Speedup Plot

```text
speedup_plot.png
```

---

## Correlation Matrix Heatmap

```text
correlation_heatmap.png
```

---

# Features of the Code

- Automatic dataset generation
- Runtime benchmarking
- Peak memory profiling
- Numerical correctness verification
- Automated visualization
- Google Colab compatibility
- Scalable benchmarking framework

---

# Technologies Used

- Python
- NumPy
- Matplotlib

---

# Notes

- Larger datasets significantly increase computational cost due to quadratic scaling.
- NumPy internally utilizes optimized BLAS routines for faster matrix operations.
- Float32 precision is used for efficient memory usage.

---

# Conclusion

This implementation demonstrates:
- CPU-based scientific computing,
- performance benchmarking,
- scalable correlation matrix computation,
- and the benefits of vectorised numerical libraries for high-dimensional data processing.
