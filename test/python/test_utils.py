import torch
import functools


# decorator function to fork the RNG and set the seed for each tests
def torch_fork_set_rng(seed=None):
    def decorator_(func):
        @functools.wraps(func)
        def wrapper_(*args, **kwargs):
            with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
                if seed is not None:
                    torch.manual_seed(seed)
                return func(*args, **kwargs)

        return wrapper_

    return decorator_
