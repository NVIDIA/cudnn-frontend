import cudnn
from contextlib import contextmanager
from typing import Optional, List, Union, Callable
from functools import wraps
import warnings


def graph_cache(key_fn, maxsize=256):
    """Custom caching decorator that uses a provided key function

    Args:
        key_fn: Function that generates cache key from the input arguments
        maxsize: Maximum size of the cache
    """

    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            if key in cache:
                return cache[key]

            result = func(*args, **kwargs)
            if len(cache) >= maxsize:
                # Remove oldest item if cache is full
                cache.pop(next(iter(cache)))
            cache[key] = result
            return result

        return wrapper

    return decorator


def jit(
    heur_modes: Union[List[cudnn.heur_mode], cudnn.heur_mode] = cudnn.heur_mode.A,
    **kwargs,
) -> Callable:
    """
    Decorator that automatically builds a graph with specified heuristic modes.

    Args:
        heur_modes: Single heuristic mode or list of modes for graph building.
        **kwargs: Additional configuration options for graph building.

    Returns:
        Callable: Decorated context manager function that returns (graph, tensor_uids).

    Example:
        >>> handle = cudnn.create_handle()
        >>> @cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.B])
        ... def my_graph():
        ...     with graph(handle) as g:
        ...         X = g.tensor(name="X", dim=[8, 64, 56, 56],
        ...                     stride=[56*56*64, 1, 56*64, 64])
        ...         return g, [X]  # Return graph and list of tensors to get UIDs for
    """
    if not isinstance(heur_modes, list):
        heur_modes = [heur_modes]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            g, tensors = func(*args, **kwargs)  # Get the result
            if g.get_execution_plan_count() <= 0:
                g.build(heur_modes)  # Build the graph
            return g, [t.get_uid() for t in tensors]  # Convert tensors to UIDs

        return wrapper

    return decorator


@contextmanager
def graph(
    handle: object,
    name: str = "cudnn_graph",
    io_data_type: cudnn.data_type = cudnn.data_type.HALF,
    intermediate_data_type: cudnn.data_type = cudnn.data_type.FLOAT,
    compute_data_type: cudnn.data_type = cudnn.data_type.FLOAT,
) -> cudnn.pygraph:
    """
    Context manager for creating and managing a CUDNN graph object.

    Args:
        handle: CUDNN handle created with cudnn.create_handle().
        name: Name of the graph for debugging purposes.
        io_data_type: Data type for input/output tensors.
        compute_data_type: Data type for computation.

    Yields:
        Tuple[cudnn.pygraph, List]: (graph object, list of tensors to get UIDs for)
    """
    g = cudnn.pygraph(
        handle=handle,
        name=name,
        io_data_type=io_data_type,
        intermediate_data_type=intermediate_data_type,
        compute_data_type=compute_data_type,
    )

    yield g, []
