"""Parallel processing utilities for kernel preparation.

This module provides utilities for parallelizing CPU-intensive kernel preparation
(compression, preprocessing) while keeping metadata manipulation (TOC updates) sequential.
"""

import os
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from typing import Any


def get_worker_count(max_workers: int | None = None) -> int:
    """Determine the number of worker threads to use.

    Args:
        max_workers: Explicit worker count, or None for auto-detection

    Returns:
        Number of worker threads (minimum 1)
    """
    if max_workers is not None:
        return max(1, max_workers)

    # Auto-detect: use all available cores
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return max(1, cpu_count)


def parallel_prepare_kernels(
    archive: Any,
    kernels: list[tuple[str, str, bytes, dict[str, Any] | None]],
    executor: Executor | None = None,
) -> list[Any]:
    """Prepare multiple kernels in parallel using provided executor.

    This is the map phase of map/reduce compression. Each kernel is prepared
    (compressed/preprocessed) independently in parallel, then the results can
    be added to the archive sequentially.

    Args:
        archive: PackedKernelArchive instance
        kernels: List of (relative_path, gfx_arch, hsaco_data, metadata) tuples
        executor: Executor for parallel execution. If None, runs sequentially.

    Returns:
        List of PreparedKernel objects in the same order as input
    """
    if not kernels:
        return []

    # Sequential path when no executor provided
    if executor is None:
        return [
            archive.prepare_kernel(relative_path, gfx_arch, hsaco_data, metadata)
            for relative_path, gfx_arch, hsaco_data, metadata in kernels
        ]

    # Parallel preparation using provided executor
    # Submit all tasks
    future_to_index = {}
    for i, (relative_path, gfx_arch, hsaco_data, metadata) in enumerate(kernels):
        future = executor.submit(
            archive.prepare_kernel, relative_path, gfx_arch, hsaco_data, metadata
        )
        future_to_index[future] = i

    # Collect results in original order
    results = [None] * len(kernels)
    for future in as_completed(future_to_index):
        index = future_to_index[future]
        results[index] = future.result()

    return results
