"""
GPU Parallel Correlation Matrix Benchmark
DS3294 - DS Practice Project #9

This project benchmarks large-scale correlation matrix computation
using two GPU parallel processing strategies:

1. Full-Matrix GPU Parallel Computation
2. Blockwise GPU Parallel Computation

The script measures:
- Execution Time
- Peak GPU VRAM Usage
- GPU Scalability

and visualizes performance using Matplotlib.
"""

import torch
import numpy as np
import time
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1: GPU DEVICE DETECTION
# =============================================================================
def get_device():
    """
    Detect CUDA-enabled GPU and return device object.
    """
    if torch.cuda.is_available():

        device = torch.device("cuda")

        gpu_name = torch.cuda.get_device_name(0)

        total_mem = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        print("\n[GPU DETECTED]")
        print(f"Device       : {gpu_name}")
        print(f"VRAM         : {total_mem:.2f} GB")

    else:
        raise RuntimeError(
            "CUDA-compatible GPU not detected."
        )

    return device


# =============================================================================
# SECTION 2: SYNTHETIC DATA GENERATION
# =============================================================================
def generate_time_series(
    N: int,
    T: int,
    seed: int = 42
):
    """
    Generate synthetic float32 time-series dataset.
    """

    rng = np.random.default_rng(seed)

    data = rng.standard_normal((N, T)).astype(np.float32)

    size_mb = data.nbytes / (1024 ** 2)

    print("\n[DATASET]")
    print(f"Time Series  : {N}")
    print(f"Length       : {T}")
    print(f"Dataset Size : {size_mb:.2f} MB")

    return data


# =============================================================================
# SECTION 3: FULL GPU PARALLEL CORRELATION
# =============================================================================
def gpu_correlation_full(
    data: np.ndarray,
    device: torch.device
):
    """
    Compute full N × N correlation matrix on GPU.
    """

    print("\n==================================================")
    print("RUNNING FULL GPU PARALLEL COMPUTATION")
    print("==================================================")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Transfer data to GPU
    # -------------------------------------------------------------------------
    transfer_start = time.perf_counter()

    X = torch.tensor(
        data,
        device=device,
        dtype=torch.float32
    )

    torch.cuda.synchronize()

    transfer_end = time.perf_counter()

    transfer_time = transfer_end - transfer_start

    # -------------------------------------------------------------------------
    # Z-score normalization
    # -------------------------------------------------------------------------
    mean = X.mean(dim=1, keepdim=True)

    std = X.std(
        dim=1,
        keepdim=True,
        unbiased=False
    )

    std = torch.clamp(std, min=1e-8)

    Z = (X - mean) / std

    # -------------------------------------------------------------------------
    # Correlation matrix computation
    # -------------------------------------------------------------------------
    T = data.shape[1]

    corr_matrix = torch.mm(Z, Z.T) / T

    corr_matrix = torch.clamp(
        corr_matrix,
        -1.0,
        1.0
    )

    torch.cuda.synchronize()

    end = time.perf_counter()

    execution_time = end - start

    peak_vram = (
        torch.cuda.max_memory_allocated(0)
        / (1024 ** 2)
    )

    print(f"Execution Time     : {execution_time:.4f} s")
    print(f"Transfer Time      : {transfer_time:.4f} s")
    print(f"Peak VRAM Usage    : {peak_vram:.2f} MB")

    return (
        corr_matrix,
        execution_time,
        peak_vram,
        transfer_time
    )


# =============================================================================
# SECTION 4: BLOCKWISE GPU PARALLEL CORRELATION
# =============================================================================
def gpu_correlation_blockwise(
    data: np.ndarray,
    device: torch.device,
    block_size: int = 512
):
    """
    Compute correlation matrix in GPU blocks
    to reduce VRAM consumption.
    """

    print("\n==================================================")
    print("RUNNING BLOCKWISE GPU PARALLEL COMPUTATION")
    print("==================================================")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Transfer data to GPU
    # -------------------------------------------------------------------------
    transfer_start = time.perf_counter()

    X = torch.tensor(
        data,
        device=device,
        dtype=torch.float32
    )

    torch.cuda.synchronize()

    transfer_end = time.perf_counter()

    transfer_time = transfer_end - transfer_start

    # -------------------------------------------------------------------------
    # Standardization
    # -------------------------------------------------------------------------
    N, T = data.shape

    mean = X.mean(dim=1, keepdim=True)

    std = X.std(
        dim=1,
        keepdim=True,
        unbiased=False
    )

    std = torch.clamp(std, min=1e-8)

    Z = (X - mean) / std

    # -------------------------------------------------------------------------
    # Blockwise correlation matrix computation
    # -------------------------------------------------------------------------
    corr_matrix = torch.zeros(
        (N, N),
        device=device,
        dtype=torch.float32
    )

    num_blocks = (
        N + block_size - 1
    ) // block_size

    for i in range(num_blocks):

        i_start = i * block_size
        i_end = min(i_start + block_size, N)

        Zi = Z[i_start:i_end, :]

        for j in range(i, num_blocks):

            j_start = j * block_size
            j_end = min(j_start + block_size, N)

            Zj = Z[j_start:j_end, :]

            block_corr = torch.mm(Zi, Zj.T) / T

            corr_matrix[
                i_start:i_end,
                j_start:j_end
            ] = block_corr

            if i != j:

                corr_matrix[
                    j_start:j_end,
                    i_start:i_end
                ] = block_corr.T

    corr_matrix = torch.clamp(
        corr_matrix,
        -1.0,
        1.0
    )

    torch.cuda.synchronize()

    end = time.perf_counter()

    execution_time = end - start

    peak_vram = (
        torch.cuda.max_memory_allocated(0)
        / (1024 ** 2)
    )

    print(f"Execution Time     : {execution_time:.4f} s")
    print(f"Transfer Time      : {transfer_time:.4f} s")
    print(f"Peak VRAM Usage    : {peak_vram:.2f} MB")

    return (
        corr_matrix,
        execution_time,
        peak_vram,
        transfer_time
    )


# =============================================================================
# SECTION 5: PERFORMANCE VISUALIZATION
# =============================================================================
def plot_comparisons(
    N,
    T,
    times,
    memories,
    transfers
):
    """
    Generate GPU benchmarking visualizations.
    """

    labels = [
        "GPU Full",
        "GPU Blockwise"
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    fig.suptitle(
        f"GPU Benchmarking | N={N}, T={T}",
        fontsize=15
    )

    # -------------------------------------------------------------------------
    # Execution Time Plot
    # -------------------------------------------------------------------------
    axes[0].bar(labels, times)

    axes[0].set_title(
        "Execution Time"
    )

    axes[0].set_ylabel(
        "Seconds"
    )

    # -------------------------------------------------------------------------
    # VRAM Usage Plot
    # -------------------------------------------------------------------------
    axes[1].bar(labels, memories)

    axes[1].set_title(
        "Peak VRAM Usage"
    )

    axes[1].set_ylabel(
        "Memory (MB)"
    )

# =============================================================================
# SECTION 6: MAIN EXPERIMENT DRIVER
# =============================================================================
def run_experiment(
    N: int,
    T: int,
    block_size: int = 1024
):

    print("\n" + "=" * 65)
    print("GPU CORRELATION MATRIX BENCHMARK")
    print("=" * 65)

    device = get_device()

    data = generate_time_series(N, T)

    # -------------------------------------------------------------------------
    # Full GPU Computation
    # -------------------------------------------------------------------------
    (
        _,
        gpu_full_time,
        gpu_full_mem,
        gpu_full_transfer
    ) = gpu_correlation_full(
        data,
        device
    )

    # -------------------------------------------------------------------------
    # Blockwise GPU Computation
    # -------------------------------------------------------------------------
    (
        _,
        gpu_block_time,
        gpu_block_mem,
        gpu_block_transfer
    ) = gpu_correlation_blockwise(
        data,
        device,
        block_size
    )

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    plot_comparisons(
        N,
        T,
        times=[
            gpu_full_time,
            gpu_block_time
        ],
        memories=[
            gpu_full_mem,
            gpu_block_mem
        ],
        transfers=[
            gpu_full_transfer,
            gpu_block_transfer
        ]
    )

    torch.cuda.empty_cache()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    run_experiment(
        N=5000,
        T=1000,
        block_size=1024
    )
