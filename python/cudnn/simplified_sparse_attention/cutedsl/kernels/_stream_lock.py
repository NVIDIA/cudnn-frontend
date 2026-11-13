# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize host enqueue sequences that reuse stream-local scratch."""

from functools import wraps
from inspect import signature
from threading import Lock, RLock

import torch

_locks = {}
_locks_guard = Lock()


def stream_lock(device=None):
    stream = torch.cuda.current_stream(device)
    key = (stream.device.index, stream.cuda_stream)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = RLock()
    return lock


def serialize_workspace(function):
    tensor_name = next(iter(signature(function).parameters))

    @wraps(function)
    def wrapped(*args, **kwargs):
        tensor = args[0] if args else kwargs[tensor_name]
        # Stream order alone cannot stop host threads interleaving producers
        # and consumers of shared scratch. Hold through the final queued read.
        with stream_lock(tensor.device):
            return function(*args, **kwargs)

    return wrapped
