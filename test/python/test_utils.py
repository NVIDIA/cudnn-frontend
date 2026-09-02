# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import functools

# Repeats for bitwise-determinism checks. A reduction-order race is timing-dependent, so a
# single repeat proves little; overridable for bisecting a flaky one.
DETERMINISM_REPEATS = int(os.environ.get("DETERMINISM_REPEATS", "8"))


def bitwise_bits(t: torch.Tensor) -> torch.Tensor:
    """View a tensor as raw bytes, so comparisons are bitwise rather than by value.

    torch.equal on floats would call +0.0 and -0.0 equal -- a difference a change in
    reduction order can produce.
    """
    return t.contiguous().view(torch.uint8)


def assert_bitwise_runs(launch, repeats=DETERMINISM_REPEATS, label=""):
    """``launch()`` returns a tuple of freshly-written output tensors.  Launch
    ``repeats`` times back to back (single sync at the end) and require every
    run to match run 0 bit for bit (barrier/fence races are timing-dependent;
    any mismatching bit is a failure, there is no tolerance)."""
    # DETERMINISM_REPEATS is settable, and below 2 there is nothing to compare run 0
    # against -- the loop below would be empty and the assertion would pass vacuously.
    assert repeats >= 2, f"{label}: assert_bitwise_runs needs repeats >= 2, got {repeats}"
    runs = [launch() for _ in range(repeats)]
    torch.cuda.synchronize()
    for out in runs[0]:
        assert torch.isfinite(out.float()).all(), f"{label}: non-finite output in run 0"
    for r, outs in enumerate(runs[1:], start=1):
        for i, (a, b) in enumerate(zip(runs[0], outs)):
            assert torch.equal(bitwise_bits(a), bitwise_bits(b)), f"{label}: output {i} differs between run 0 and run {r}"


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
