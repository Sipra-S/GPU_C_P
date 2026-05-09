# GPU Parallel Processing

This folder contains the implementation of GPU-accelerated correlation matrix computation using PyTorch CUDA.

The project benchmarks large-scale correlation matrix estimation using parallel GPU processing strategies and analyses:
- execution performance,
- GPU memory usage,
- and scalability for high-dimensional datasets.

Developed for **DS3294 – DS Practice Project #9**

Made by: **Sipra Subhadarsini Sahoo**

---

# Folder Contents

```text
gpu_parallel_processing/
│
├── gpu_benchmark.py
├── README.md
└── images/
    ├── execution_time_comparison.png
    ├── vram_usage_comparison.png
    └── transfer_time_comparison.png
```

---

# Implemented GPU Strategies

The code implements two GPU computation approaches:

## 1. Full GPU Parallel Computation

Computes the entire correlation matrix in a single GPU matrix multiplication operation.

### Features
- Fastest execution
- Fully parallelized GPU computation
- Efficient CUDA matrix operations

### Limitation
- High VRAM consumption for large datasets

---

## 2. Blockwise GPU Parallel Computation

Computes the correlation matrix in smaller matrix blocks.

### Features
- Reduced peak VRAM usage
- Memory-efficient processing
- Scalable for larger datasets

### Trade-off
- Slightly increased computation overhead

---

# Mathematical Background

For `N` time series of length `T`, the computational complexity scales as:

```text
O(N² · T)
```

The correlation matrix is computed using:

```text
C = (1 / (T - 1)) XXᵀ
```

where:
- `X` represents standardized time-series data,
- and `C` is the resulting correlation matrix.

---

# Requirements

## Python Version

Recommended:

```text
Python 3.9+
```

---

# Dependencies

Install the required libraries using:

```bash
pip install torch numpy matplotlib
```

---

# CUDA Requirements

To use GPU acceleration:

- NVIDIA GPU
- CUDA Toolkit installed
- CUDA-compatible PyTorch version

---

# Installing PyTorch with CUDA

Visit:

```text
https://pytorch.org/get-started/locally/
```

Example installation command:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Replace `cu121` with the CUDA version supported by your system.

---

# Verifying CUDA Installation

Run:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected output:

```text
True
NVIDIA GPU Name
```

---

# Running the Benchmark

Execute:

```bash
python gpu_benchmark.py
```

---

# Example Configuration

Inside `gpu_benchmark.py`:

```python
run_experiment(
    N=5000,
    T=1000,
    block_size=1024
)
```

where:
- `N` = number of time series
- `T` = length of each series
- `block_size` = GPU block size for chunked computation

---

# Generated Outputs

The script automatically generates:

## 1. Execution Time Plot

```text
execution_time_comparison.png
```

---

## 2. VRAM Usage Plot

```text
vram_usage_comparison.png
```

---

## 3. CPU → GPU Transfer Time Plot

```text
transfer_time_comparison.png
```

---

# Benchmark Metrics

The project benchmarks:

- GPU execution time
- Peak VRAM usage
- CPU → GPU transfer overhead
- Performance of:
  - full GPU computation
  - blockwise GPU computation

---

# Technologies Used

- Python
- PyTorch CUDA
- NumPy
- Matplotlib

---

# Notes

- CUDA-enabled PyTorch is required for GPU acceleration.
- Blockwise computation becomes useful for very large datasets that exceed available GPU memory.
- Float32 precision is used for efficient GPU computation.

---

# Conclusion

This implementation demonstrates:
- large-scale GPU parallel processing,
- efficient matrix computation using CUDA,
- VRAM-aware scientific computing,
- and benchmarking of GPU acceleration techniques for high-dimensional numerical workloads.
