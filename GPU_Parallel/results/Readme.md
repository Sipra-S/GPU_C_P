
# GPU Parallel Correlation Matrix Benchmark

A GPU-accelerated benchmarking framework for large-scale correlation matrix computation using PyTorch CUDA.

Developed for **DS3294 – DS Practice Project #9**

Made by: **Sipra Subhadarsini Sahoo**

---

# Project Overview

This project investigates GPU-based parallel computation strategies for estimating large-scale correlation matrices from parallel time series data.

For `N` time series of length `T`, the computational complexity scales as:

```text
O(N² · T)
```

The project implements and benchmarks:

- Full GPU Parallel Correlation Computation
- Blockwise GPU Parallel Correlation Computation

while analysing:
- execution speed,
- GPU memory usage,
- and CPU → GPU transfer overhead.

---

# Implemented GPU Strategies

## 1. Full GPU Parallel Computation

Computes the entire correlation matrix in a single GPU matrix multiplication operation using:

```text
C = (1 / (T - 1)) XXᵀ
```

### Advantages
- Extremely fast
- Maximizes GPU parallel throughput

### Limitations
- High VRAM consumption
- Limited by GPU memory capacity

---

## 2. Blockwise GPU Parallel Computation

Computes the correlation matrix in smaller sub-blocks to reduce peak VRAM usage.

### Advantages
- Memory efficient
- Handles larger datasets

### Trade-off
- Slightly slower due to chunked computation

---

# Benchmarking Features

The framework measures:

- Execution Time
- Peak GPU VRAM Usage
- CPU → GPU Transfer Time

for both GPU computation strategies.

---

# Benchmark Results

The following benchmark was performed for:

```text
N = 5000
T = 1000
Block Size = 1024
```

---

# Execution Time Comparison

<p align="center">
  <img src="images/execution_time_comparison.png" width="500">
</p>

### Observation

- Blockwise GPU computation performs comparably to full GPU computation.
- GPU parallel matrix multiplication achieves extremely low execution times.
- The overhead introduced by blockwise computation remains small.

---

# Peak VRAM Usage

<p align="center">
  <img src="images/vram_usage_comparison.png" width="500">
</p>

### Observation

- Full GPU computation uses lower VRAM for this dataset size.
- Blockwise computation introduces additional intermediate allocations.
- For larger datasets, blockwise computation becomes essential to avoid VRAM overflow.

---

# CPU → GPU Transfer Time

<p align="center">
  <img src="images/transfer_time_comparison.png" width="500">
</p>

### Observation

- Data transfer overhead is relatively small compared to total computation time.
- GPU acceleration remains highly beneficial for large-scale matrix operations.

---

# Technologies Used

- Python
- PyTorch (CUDA)
- NumPy
- Matplotlib

---

# Repository Structure

```text
.
├── gpu_benchmark.py
├── README.md
└── images
    ├── execution_time_comparison.png
    ├── vram_usage_comparison.png
    └── transfer_time_comparison.png
```

---

# How to Run

## Install Dependencies

```bash
pip install torch numpy matplotlib
```

CUDA-enabled PyTorch is strongly recommended.

---

## Run Benchmark

```bash
python gpu_benchmark.py
```

---

# Example Experiment

```python
run_experiment(
    N=5000,
    T=1000,
    block_size=1024
)
```

---

# Expected Outcome

This project demonstrates:
- large-scale GPU parallel computation,
- efficient matrix operations using CUDA,
- memory-aware blockwise processing,
- and practical benchmarking of GPU scientific workloads.
