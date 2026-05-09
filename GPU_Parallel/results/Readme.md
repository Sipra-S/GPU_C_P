# GPU Correlation Matrix Benchmark Results

This repository presents benchmarking results for GPU-accelerated correlation matrix computation using PyTorch CUDA.

Developed for **DS3294 – DS Practice Project #9**

Made by: **Sipra Subhadarsini Sahoo**

---

# Experiment Overview

The experiment benchmarks two GPU parallel processing strategies for computing large-scale correlation matrices:

1. **GPU Full-Matrix Computation**
2. **GPU Blockwise Computation**

The benchmark evaluates:
- execution time,
- GPU VRAM usage,
- and CPU → GPU transfer overhead.

---

# Experimental Configuration

```text
Number of Time Series (N) : 5000
Length of Each Series (T) : 1000
Block Size                : 1024
Data Type                 : float32
Framework                 : PyTorch CUDA
```

---

# Correlation Matrix Complexity

The computational complexity of correlation matrix estimation scales as:

```text
O(N² · T)
```

which becomes computationally expensive for large-scale datasets.

---

# Benchmark Results

## 1. Execution Time Comparison

<p align="center">
  <img src="images/execution_time_comparison.png" width="500">
</p>

### Analysis

- Both GPU strategies achieve extremely fast execution times.
- Blockwise GPU computation performs slightly faster in this benchmark configuration.
- The difference in runtime is minimal, demonstrating efficient GPU parallelism.

### Observation

GPU-based matrix multiplication using CUDA significantly accelerates correlation matrix computation for large datasets.

---

## 2. Peak GPU VRAM Usage

<p align="center">
  <img src="images/vram_usage_comparison.png" width="500">
</p>

### Analysis

- Blockwise GPU computation consumes more VRAM in this experiment due to intermediate block allocations.
- Full GPU computation shows lower memory usage for the chosen dataset size.
- As dataset dimensionality increases further, blockwise computation becomes necessary to prevent GPU memory overflow.

### Observation

Memory-efficient computation strategies become increasingly important for very large correlation matrices.

---

## 3. CPU → GPU Transfer Time

<p align="center">
  <img src="images/transfer_time_comparison.png" width="500">
</p>

### Analysis

- Data transfer overhead between CPU and GPU remains relatively small.
- Transfer times are nearly identical for both strategies because the same dataset is transferred to GPU memory.
- Computation time dominates overall runtime for large-scale workloads.

### Observation

GPU acceleration remains advantageous despite transfer overhead for high-dimensional numerical computation.

---

# Key Findings

- GPU parallel computation provides extremely fast correlation matrix estimation.
- Full GPU computation is highly efficient for moderate dataset sizes.
- Blockwise GPU computation enables scalable processing under memory constraints.
- CPU → GPU transfer overhead is comparatively small relative to matrix computation cost.
- GPU-based matrix multiplication is highly effective for large-scale scientific computing workloads.

---

# Technologies Used

- Python
- PyTorch CUDA
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

# Running the Benchmark

## Install Dependencies

```bash
pip install torch numpy matplotlib
```

---

## Run the Script

```bash
python gpu_benchmark.py
```

---

# Conclusion

This benchmark demonstrates the effectiveness of GPU parallel processing for large-scale correlation matrix estimation and highlights the trade-offs between:
- execution speed,
- memory consumption,
- and scalable GPU computation strategies.
