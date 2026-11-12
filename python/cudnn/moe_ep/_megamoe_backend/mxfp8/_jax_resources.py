# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit NVSHMEM ownership for the JAX MegaMoE launch sequence."""

import math

import numpy as np

from cudnn._cake.jax import ExternalTensorRef, LaunchContext
from cudnn._device import _ck, ensure_current_context
from cudnn.frost.device import device_context


def round_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


class SymmetricResources:
    """One collectively allocated slab; never place its storage in XLA operands."""

    def __init__(self, kernels, *, device, rank, world_size):
        import cuda.bindings.driver as driver

        try:
            import nvshmem.core as core
        except (ImportError, OSError) as exc:
            raise ImportError('JAX MoeEP requires NVSHMEM; install "nvidia-cudnn-frontend[comm]" and initialize its runtime') from exc

        try:
            from cuda.core import Device
        except ImportError:
            from cuda.core.experimental import Device

        ensure_current_context(device=device)
        status = core.init_status()
        if int(status) not in (2, 3, 4):
            raise RuntimeError("initialize NVSHMEM on the selected GPU before preparing JAX MoeEP")
        if (int(core.my_pe()), int(core.n_pes())) != (rank, world_size):
            raise ValueError("JAX MoeEP rank/world_size must match the initialized NVSHMEM world")
        self.device = device
        self._core = core
        self._closed = False
        self.regions = {}
        size = 0

        def add(name, shape, dtype):
            nonlocal size
            size = round_up(size, 256)
            self.regions[name] = (size, tuple(shape), dtype)
            size += math.prod(shape) * np.dtype(dtype).itemsize

        config = kernels["training_forward"].config
        capacity, hidden, top_k = config.max_tokens_per_rank, config.hidden, config.top_k
        add("activation", (capacity, hidden), "uint8")
        add("activation_sf", (round_up(capacity, 128), round_up(hidden // 32, 16)), "uint8")
        add("indices", (capacity, top_k), "int32")
        add("scores", (capacity, top_k), "float32")
        add("output", (capacity, hidden), "bfloat16")
        add("dprob", (capacity, top_k), "float32")
        add("overflow", (1,), "int32")
        add("beta", (config.num_experts,), "float32")
        for phase, prepared in kernels.items():
            local, shared = prepared.kernel.get_workspace_sizes()
            add(phase + ".local", (int(local),), "uint8")
            add(phase + ".shared", (int(shared),), "uint8")
        self.nbytes = round_up(size, 256)
        self.buffer = core.buffer(self.nbytes)
        self.address = int(self.buffer.handle)
        if self.address % 256 or int(self.buffer.device_id) != device:
            core.free(self.buffer)
            raise RuntimeError("NVSHMEM returned misaligned storage or storage on another GPU")
        try:
            self.peers = tuple(core.get_peer_buffer(self.buffer, peer) for peer in range(world_size))
            if any(peer is None or int(peer.handle) == 0 for peer in self.peers):
                raise NotImplementedError("JAX MegaMoE requires direct peer access to every NVSHMEM PE")
            self.offsets = tuple(int(peer.handle) - self.address for peer in self.peers)
            self.context = LaunchContext(device, owners=(self.buffer, self.peers))
        except Exception:
            core.free(self.buffer)
            raise
        try:
            with device_context(self.device):
                stream = Device(device).create_stream()
                _ck(*driver.cuMemsetD8Async(self.address, 0, self.nbytes, driver.CUstream(int(stream.handle))))
                core.barrier_all(stream=stream)
                stream.sync()
        except Exception:
            self.close()
            raise

    def ref(self, name):
        offset, shape, dtype = self.regions[name]
        return ExternalTensorRef(self.address, shape, dtype, self.nbytes, offset)

    def close(self):
        """Wait for local launches, then free collectively on every PE."""
        if self._closed:
            return
        with device_context(self.device):
            self.context.close()
            self._core.free(self.buffer)
            self._closed = True
