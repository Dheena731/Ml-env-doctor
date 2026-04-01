"""Parallel execution utilities for independent operations."""

import concurrent.futures
from typing import Callable, Iterable, List, TypeVar, cast

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

    # Use ThreadPoolExecutor for I/O-bound operations.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(func, item): index for index, item in enumerate(items_list)
        }
        results: List[R | None] = [None] * len(items_list)
        completed = 0

        for future in concurrent.futures.as_completed(future_to_index, timeout=timeout):
            index = future_to_index[future]
            item = items_list[index]
            try:
                results[index] = future.result()
                completed += 1
            except Exception as e:
                logger.error(f"Error processing {item}: {e}")
                raise

        if completed != len(items_list):
            raise RuntimeError(f"Only {completed}/{len(items_list)} tasks completed")

        return [cast(R, result) for result in results]


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
        future_to_index = {
            executor.submit(func, item): index for index, item in enumerate(items_list)
        }
        results: List[tuple[T, R | Exception] | None] = [None] * len(items_list)

        for future in concurrent.futures.as_completed(future_to_index, timeout=timeout):
            index = future_to_index[future]
            item = items_list[index]
            try:
                results[index] = (item, future.result())
            except Exception as e:
                results[index] = (item, e)

        return [cast(tuple[T, R | Exception], result) for result in results]
