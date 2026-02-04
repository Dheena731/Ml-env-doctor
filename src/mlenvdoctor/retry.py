"""Retry logic for transient failures."""

import functools
import time
from typing import Any, Callable, Optional, TypeVar

from .exceptions import DiagnosticError
from .logger import logger

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to retry a function on failure.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry

    Returns:
        Decorated function with retry logic

    Example:
        @retry(max_attempts=3, delay=1.0, exceptions=(ConnectionError,))
        def fetch_data():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")

            # All attempts failed
            if last_exception:
                raise DiagnosticError(
                    f"{func.__name__} failed after {max_attempts} attempts",
                    f"Last error: {last_exception}",
                ) from last_exception

            # Should never reach here, but satisfy type checker
            raise RuntimeError("Retry logic failed unexpectedly")

        return wrapper

    return decorator


def retry_network(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator specifically for network operations.

    Retries on network-related exceptions with exponential backoff.
    """
    return retry(
        max_attempts=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
        ),
    )(func)
