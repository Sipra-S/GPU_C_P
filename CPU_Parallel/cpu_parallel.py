"""
DS3294 Project #9 — CPU Correlation Matrix Benchmark
Google Colab Compatible Version
"""

import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_series, length, seed=42):
    """Generate random time-series dataset."""

    rng = np.random.default_rng(seed)

    data = rng.standard_normal(
        (n_series, length)
    ).astype(np.float32)

    return data


# =============================================================================
# MEMORY TRACKING
# =============================================================================

def peak_memory_mb():
    """Return peak memory usage in MB."""

    _, peak = tracemalloc.get_traced_memory()

    return peak / (1024 ** 2)


# =============================================================================
# SINGLE-THREADED IMPLEMENTATION
# =============================================================================

def corr_single_thread(data):
    """Single-threaded baseline."""

    n, length = data.shape

    mean = data.mean(axis=1, keepdims=True)

    std = data.std(axis=1, keepdims=True) + 1e-8

    z = (data - mean) / std

    corr = np.empty((n, n), dtype=np.float32)

    for i in range(n):

        for j in range(i, n):

            c = float(np.dot(z[i], z[j])) / length

            corr[i, j] = c
            corr[j, i] = c

    return corr


# =============================================================================
# NUMPY PARALLEL IMPLEMENTATION
# =============================================================================

def corr_numpy_parallel(data):
    """
    NumPy vectorised parallel computation.
    Uses BLAS backend internally.
    """

    mean = data.mean(axis=1, keepdims=True)

    std = data.std(axis=1, keepdims=True) + 1e-8

    z = (data - mean) / std

    corr = (z @ z.T) / data.shape[1]

    return corr.astype(np.float32)


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark(fn, data, label):
    """Benchmark execution time and memory."""

    tracemalloc.start()

    start = time.perf_counter()

    result = fn(data)

    elapsed = time.perf_counter() - start

    memory = peak_memory_mb()

    tracemalloc.stop()

    print(
        f"[{label}] Time = {elapsed:.4f} s | "
        f"Peak Memory = {memory:.2f} MB"
    )

    return result, elapsed, memory


# =============================================================================
# CORRECTNESS CHECK
# =============================================================================

def verify(reference, other, label, atol=1e-3):
    """Verify numerical correctness."""

    max_error = np.max(np.abs(reference - other))

    status = "PASS" if max_error < atol else "FAIL"

    print(
        f"Correctness [{label}] : "
        f"{status} | Max Error = {max_error:.2e}"
    )


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_results(results, corr_matrix):
    """Generate plots."""

    Ns = [r["n"] for r in results]

    t_st = [r["t_st"] for r in results if r["t_st"] is not None]

    st_ns = [r["n"] for r in results if r["t_st"] is not None]

    t_np = [r["t_np"] for r in results]

    m_st = [r["m_st"] for r in results if r["m_st"] is not None]

    m_np = [r["m_np"] for r in results]

    # =========================================================================
    # Runtime Plot
    # =========================================================================
    plt.figure(figsize=(7, 5))

    plt.plot(st_ns, t_st, "o-", label="Single-thread")

    plt.plot(Ns, t_np, "s-", label="NumPy Parallel")

    plt.xscale("log", base=2)

    plt.yscale("log")

    plt.xlabel("N")

    plt.ylabel("Runtime (s)")

    plt.title("Runtime vs N")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig("runtime_vs_n.png", dpi=300)

    plt.show()

    # =========================================================================
    # Memory Plot
    # =========================================================================
    plt.figure(figsize=(7, 5))

    x = np.arange(len(Ns))

    width = 0.35

    st_mem = m_st + [0] * (len(Ns) - len(m_st))

    plt.bar(
        x - width / 2,
        st_mem,
        width,
        label="Single-thread"
    )

    plt.bar(
        x + width / 2,
        m_np,
        width,
        label="NumPy Parallel"
    )

    plt.xticks(
        x,
        [f"N={n}" for n in Ns]
    )

    plt.ylabel("Peak Memory (MB)")

    plt.title("Peak Memory vs N")

    plt.legend()

    plt.tight_layout()

    plt.savefig("memory_vs_n.png", dpi=300)

    plt.show()

    # =========================================================================
    # Speedup Plot
    # =========================================================================
    speedups = []

    for r in results:

        if r["t_st"] is not None:

            speedups.append(
                r["t_st"] / r["t_np"]
            )

    valid_ns = [
        r["n"] for r in results
        if r["t_st"] is not None
    ]

    plt.figure(figsize=(7, 5))

    plt.plot(valid_ns, speedups, "o-")

    plt.xscale("log", base=2)

    plt.xlabel("N")

    plt.ylabel("Speedup")

    plt.title("Parallel Speedup")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig("speedup_plot.png", dpi=300)

    plt.show()

    # =========================================================================
    # Correlation Heatmap
    # =========================================================================
    plt.figure(figsize=(7, 6))

    plt.imshow(
        corr_matrix,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(label="Correlation")

    plt.title("Correlation Matrix Heatmap")

    plt.tight_layout()

    plt.savefig("correlation_heatmap.png", dpi=300)

    plt.show()


# =============================================================================
# RUN EXPERIMENT
# =============================================================================

def run_experiment(n_series, length):
    """Run benchmark experiment."""

    print("\n" + "=" * 60)

    print(
        f"N = {n_series}, Length = {length}"
    )

    print("=" * 60)

    data = generate_data(
        n_series,
        length
    )

    row = {"n": n_series}

    # =========================================================================
    # Single-thread baseline
    # =========================================================================
    if n_series <= 512:

        ref, t_st, m_st = benchmark(
            corr_single_thread,
            data,
            "Single-thread"
        )

        row["t_st"] = t_st
        row["m_st"] = m_st

    else:

        print(
            "[Single-thread skipped for large N]"
        )

        ref = corr_numpy_parallel(data)

        row["t_st"] = None
        row["m_st"] = None

    # =========================================================================
    # NumPy parallel
    # =========================================================================
    np_result, t_np, m_np = benchmark(
        corr_numpy_parallel,
        data,
        "NumPy Parallel"
    )

    row["t_np"] = t_np
    row["m_np"] = m_np
    row["corr_matrix"] = np_result

    print()

    verify(
        ref,
        np_result,
        "Parallel vs Reference"
    )

    return row


# =============================================================================
# MAIN
# =============================================================================

experiments = [
    (64, 1000),
    (256, 1000),
    (512, 2000),
    (1024, 2000),
]

all_results = []

for n, L in experiments:

    result = run_experiment(n, L)

    all_results.append(result)

print("\nBenchmark Complete.")

plot_results(
    all_results,
    all_results[-1]["corr_matrix"]
)
