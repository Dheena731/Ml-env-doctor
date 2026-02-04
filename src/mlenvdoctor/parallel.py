"""Parallel execution utilities for independent operations."""

import concurrent.futures
from typing import Callable, Iterable, List, TypeVar

from .logger import logger

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 4,
    timeout: float | None = None,
) -> List[R]:
    """
    Run a function in parallel on multiple items.

    Args:
        func: Function to execute
        items: Iterable of items to process
        items_list: List of items to process
        max_workers: Maximum number of parallel workers
        timeout: Maximum time to wait for all tasks (None = no timeout)

    Returns:
        List of results in the same order as input items

    Example:
        def check_library(name: str) -> bool:
            return importlib.util.find_spec(name) is not None

        results = run_parallel(check_library, ["torch", "transformers", "peft"])
    """
    items_list = list(items)

    if not items_list:
        return []

    # Use ThreadPoolExecutor for I/O-bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {executor.submit(func, item): item for item in items_list}

        results: List[R] = []
        completed = 0

        # Process completed tasks
        for future in concurrent.futures.as_completed(future_to_item, timeout=timeout):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
            except Exception as e:
                logger.error(f"Error processing {item}: {e}")
                # Re-raise to maintain error behavior
                raise

        if completed != len(items_list):
            raise RuntimeError(f"Only {completed}/{len(items_list)} tasks completed")

        return results


def run_parallel_with_results(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 4,
    timeout: float | None = None,
) -> List[tuple[T, R | Exception]]:
    """
    Run a function in parallel and return results with original items.

    Unlike run_parallel, this catches exceptions and returns them as results.

    Args:
        func: Function to execute
        items: Iterable of items to process
        max_workers: Maximum number of parallel workers
        timeout: Maximum time to wait for all tasks

    Returns:
        List of (item, result_or_exception) tuples

    Example:
        def check_library(name: str) -> bool:
            if name == "bad":
                raise ValueError("Bad library")
            return True

        results = run_parallel_with_results(check_library, ["torch", "bad", "peft"])
        # Returns: [("torch", True), ("bad", ValueError(...)), ("peft", True)]
    """
    items_list = list(items)

    if not items_list:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, item): item for item in items_list}

        results: List[tuple[T, R | Exception]] = []

        for future in concurrent.futures.as_completed(future_to_item, timeout=timeout):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append((item, result))
            except Exception as e:
                results.append((item, e))

        return results
