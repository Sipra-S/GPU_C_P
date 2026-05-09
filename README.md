# CPU–GPU Parallel Correlation Matrix Benchmark

This repository contains a benchmarking framework for computing large-scale \(N \times N\) correlation matrices from parallel time series data using GPU and CPU parallel computation strategies.

Developed for **DS3294 – DS Practice Project #9**.

Made by: **Sipra Subhadarsini Sahoo**

---

# Project Overview

Computing correlation matrices for thousands of time series is computationally intensive. For \(N\) time series of length \(T\), the computational complexity scales as:

\[
O(N^2 \cdot T)
\]

As the dimensionality increases, both runtime and memory requirements grow rapidly.

This project explores how parallel computation on CPUs and GPUs can accelerate correlation matrix estimation while addressing memory and scalability challenges.

---

# Implemented Parallel Computation Strategies

## 1. Parallel CPU Computation

A CPU-based parallel implementation designed to utilize multiple processing cores for correlation matrix estimation.

### Features
- Parallel workload distribution
- Multi-core processing
- NumPy-based computation
- Runtime and memory benchmarking

### Goal
Evaluate how efficiently CPU parallelism scales for large datasets.

---

## 2. GPU Full-Matrix Computation

A PyTorch CUDA implementation that computes the entire correlation matrix in one massively parallel GPU operation.

### Pipeline
- Transfer data to GPU
- Standardize time series (Z-score normalization)
- Compute correlations using matrix multiplication:

\[
C = \frac{1}{T-1}XX^T
\]

via `torch.mm`.

### Advantages
- Extremely fast for large datasets
- Maximizes GPU throughput

### Limitations
- High VRAM consumption
- May fail for extremely large \(N\)

---

## 3. GPU Blockwise Computation

A memory-efficient GPU strategy that computes the correlation matrix in smaller sub-blocks.

### Features
- Chunked matrix processing
- Reduced VRAM footprint
- Handles larger datasets beyond full-GPU memory limits

### Trade-off
- Slightly slower than full GPU computation
- Significantly more memory stable

---

# Benchmarking Features

The project benchmarks and compares:

- Parallel CPU computation
- GPU full-matrix computation
- GPU blockwise computation

across varying:
- number of time series \(N\),
- sequence lengths \(T\),
- and block sizes.

---

# Memory Profiling

## GPU VRAM Tracking

Peak GPU memory usage is measured using:
- `torch.cuda.max_memory_allocated()`
- `torch.cuda.reset_peak_memory_stats()`

This enables precise comparison between:
- full GPU computation,
- and blockwise GPU computation.

---

# CPU–GPU Transfer Overhead Analysis

The project additionally investigates:
- CPU → GPU transfer latency
- Synchronization overhead
- End-to-end execution cost

to identify:
- when GPU acceleration becomes beneficial,
- and when transfer overhead dominates performance gains.

---

# Numerical Verification

All implementations are verified against NumPy’s standard:

```python
np.corrcoef()
```

to ensure numerical consistency and correctness.

---

# Automated Visualization

The repository includes Matplotlib-based visualization tools that generate comparative performance dashboards.

Generated plots include:
- Execution Time Comparison
- Peak Memory Usage
- CPU vs GPU Scaling
- GPU Transfer Overhead
- Parallel Speedup Analysis

---

# Research Objectives

This project investigates:

- When does GPU acceleration outperform CPU parallel computation?
- How does runtime scale with increasing \(N\)?
- What are the memory trade-offs between computation strategies?
- How effective is blockwise GPU computation under VRAM constraints?
- How do empirical results compare with theoretical complexity?

---

# Technologies Used

- Python
- NumPy
- PyTorch (CUDA)
- Multiprocessing
- Matplotlib

---

# How to Run

## Prerequisites

Install the required libraries:

```bash
pip install torch numpy matplotlib
```

CUDA-enabled PyTorch is recommended for GPU benchmarking.

---

## Execution

Run the benchmark script:

```bash
python gpu_benchmark.py
```

---

# Expected Outcome

A systematic comparison of CPU and GPU parallelization strategies for large-scale correlation matrix computation, providing practical insight into:
- scalability,
- memory efficiency,
- transfer overhead,
- and high-performance parallel computing.
