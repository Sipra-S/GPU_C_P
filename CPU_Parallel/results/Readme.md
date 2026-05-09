# CPU Correlation Matrix Benchmark

A benchmarking framework for analysing CPU-based correlation matrix computation using:
- single-threaded execution,
- and NumPy vectorised parallel computation.

Developed for **DS3294 – DS Practice Project #9**

Made by: **Sipra Subhadarsini Sahoo**

---

# Project Overview

This project investigates the computational performance of CPU-based correlation matrix estimation for large-scale time-series datasets.

The benchmark compares:
- execution runtime,
- memory consumption,
- parallel speedup,
- and correlation matrix generation

across increasing dataset sizes.

---

# Mathematical Background

For `N` time series of length `L`, the correlation matrix computation scales as:

```text
O(N² · L)
```

The correlation matrix is computed using:

```text
C = (1 / L) ZZᵀ
```

where:
- `Z` represents standardized time-series data,
- and `C` is the Pearson correlation matrix.

---

# Implemented CPU Strategies

## 1. Single-threaded Baseline

A pure loop-based implementation using explicit nested iteration.

### Features
- Sequential execution
- Baseline performance reference
- High computational cost for large `N`

---

## 2. NumPy Parallel Computation

A vectorised implementation using NumPy matrix multiplication backed by optimized BLAS libraries.

### Features
- Internally parallelized linear algebra
- Significant runtime acceleration
- Efficient memory usage
- Scalable computation

---

# Benchmark Configuration

```text
Experiments:
N = 64,   L = 1000
N = 256,  L = 1000
N = 512,  L = 2000
N = 1024, L = 2000
```

---

# Generated Benchmark Plots

The benchmark automatically generates:

```text
runtime_vs_n.png
memory_vs_n.png
speedup_plot.png
correlation_heatmap.png
```

---

# 1. Runtime vs N

<p align="center">
  <img src="images/runtime_vs_n.png" width="650">
</p>

## Analysis

- Runtime increases rapidly with dataset size due to quadratic complexity.
- The single-threaded implementation becomes significantly slower for large `N`.
- NumPy parallel computation maintains substantially lower runtime.

## Observation

Vectorised BLAS-backed matrix multiplication provides major acceleration over explicit loop-based computation.

---

# 2. Peak Memory vs N

<p align="center">
  <img src="images/memory_vs_n.png" width="650">
</p>

## Analysis

- Memory usage increases with dataset dimensionality.
- NumPy parallel computation requires additional memory allocation for optimized matrix operations.
- Memory growth remains manageable across tested configurations.

## Observation

Both implementations scale reasonably in memory usage, though larger correlation matrices naturally demand higher RAM.

---

# 3. Parallel Speedup

<p align="center">
  <img src="images/speedup_plot.png" width="650">
</p>

## Analysis

- Speedup increases significantly with larger problem sizes.
- NumPy parallel computation achieves more than 100× acceleration for large datasets.
- Larger matrices benefit more strongly from optimized vectorised computation.

## Observation

Parallel numerical libraries become increasingly advantageous as computational workload grows.

---

# 4. Correlation Matrix Heatmap

<p align="center">
  <img src="images/correlation_heatmap.png" width="650">
</p>

## Analysis

- The diagonal elements show perfect self-correlation (`r = 1`).
- Off-diagonal regions remain close to zero because the generated time series are random and statistically independent.
- The heatmap validates numerical correctness of the correlation computation.

## Observation

The generated correlation matrix demonstrates expected statistical behaviour for randomly generated datasets.

---

# Repository Structure

```text
.
├── cpu_correlation.py
├── README.md
└── images
    ├── runtime_vs_n.png
    ├── memory_vs_n.png
    ├── speedup_plot.png
    └── correlation_heatmap.png
```

---

# Installation

Install dependencies:

```bash
pip install numpy matplotlib
```

---

# Running the Benchmark

Run:

```bash
python cpu_correlation.py
```

or in Google Colab/Jupyter Notebook directly.

---

# Technologies Used

- Python
- NumPy
- Matplotlib

---

# Key Findings

- Explicit loop-based CPU correlation computation scales poorly for large datasets.
- NumPy vectorised computation dramatically improves performance.
- Parallel numerical libraries provide substantial acceleration for matrix-heavy scientific workloads.
- Correlation matrix computation becomes increasingly resource-intensive as dimensionality grows.

---

# Conclusion

This project demonstrates:
- CPU parallel numerical computation,
- performance benchmarking of large-scale matrix operations,
- memory and runtime scaling behaviour,
- and the effectiveness of optimized linear algebra libraries for scientific computing.
