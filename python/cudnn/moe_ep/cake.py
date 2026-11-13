# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental standalone Cake MoK training API."""

from importlib import import_module

from cudnn._buffer_utils import check_buffers_disjoint
from cudnn._cake import loader

__all__ = ["prepare_mok_training", "create_mok_workspace", "quantize_mok_weights"]


def _guard_accumulators(adapter):
    from dataclasses import fields, is_dataclass
    from functools import wraps
    from inspect import signature

    import torch

    backward = adapter.backward
    parameters = signature(backward)

    def tensors(name, value):
        if isinstance(value, torch.Tensor):
            yield name, value
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                yield from tensors(f"{name}[{index}]", item)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                yield from tensors(f"{name}.{field.name}", getattr(value, field.name))

    @wraps(backward)
    def checked_backward(*args, **kwargs):
        arguments = parameters.bind(*args, **kwargs).arguments
        accumulators = arguments.pop("weight_grad_accumulators", None)
        if accumulators is not None:
            if not isinstance(accumulators, (tuple, list)) or len(accumulators) != 6 or any(not isinstance(t, torch.Tensor) for t in accumulators):
                raise ValueError("weight_grad_accumulators must contain six tensors")
            outputs = list(tensors("weight_grad_accumulators", accumulators))
            inputs = [pair for name, value in arguments.items() for pair in tensors(name, value)]
            check_buffers_disjoint(outputs, inputs)
        return backward(*args, **kwargs)

    adapter.backward = checked_backward
    return adapter


def _api(device=None):
    arch = loader.device_arch(device)
    if arch != "sm_100a":
        raise NotImplementedError("the Frontend MoK adapter is qualified for SM100 only")
    loader.check_toolkit(arch, "MoK training")
    package = loader.load_package("cake_mok_bf16")
    return import_module(package.__name__ + ".api")


def prepare_mok_training(*, device=None, ep_size=1, local_experts=4, topk=2, clamped_swiglu=False, fp32_wgrad=False):
    """Prepare native MoK scheduling, forward, recompute, and backward calls.

    The initial Frontend contract is one rank, four routed experts, and top-2.
    Warm each precision/activation variant before CUDA graph capture.
    """
    import torch

    if any(type(v) is not int for v in (ep_size, local_experts, topk)) or (ep_size, local_experts, topk) != (1, 4, 2):
        raise NotImplementedError("the initial MoK adapter supports EP1, four routed experts, and top-2")
    if type(clamped_swiglu) is not bool or type(fp32_wgrad) is not bool:
        raise TypeError("clamped_swiglu and fp32_wgrad must be bool")
    with torch.cuda.device(device):
        adapter = _api(device).prepare_mok_bf16(
            ep_size=ep_size,
            local_experts=local_experts,
            topk=topk,
            clamped_swiglu=clamped_swiglu,
            fp32_wgrad=fp32_wgrad,
        )
        return _guard_accumulators(adapter)


def create_mok_workspace(*, group, device, num_local_tokens, hidden_size, topk=2, **options):
    """Allocate the NCCL workspace outside capture and return ``(config, workspace)``.

    Retain the workspace until all eager work and graph replays finish.
    Compute calls allocate their outputs and temporary buffers separately.
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_backend(group) != "nccl":
        raise ValueError("MoK requires an initialized NCCL process group")
    if dist.get_world_size(group) != 1 or type(topk) is not int or topk != 2:
        raise NotImplementedError("the initial MoK adapter supports one NCCL rank and top-2")
    if type(num_local_tokens) is not int or num_local_tokens < 0:
        raise ValueError("num_local_tokens must be a nonnegative int")
    if type(hidden_size) is not int or hidden_size <= 0 or hidden_size % 256:
        raise ValueError("hidden_size must be a positive multiple of 256")
    options.setdefault("imbalance_comm_sms", None)
    with torch.cuda.device(device):
        return _api(device).create_mok_bf16_workspace(
            group=group,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_size,
            topk=topk,
            **options,
        )


def quantize_mok_weights(gate, up, down):
    """Allocate native MXFP8 routed weights in forward and backward orientations.

    Call after every weight update. Shared-expert weights remain BF16.
    """
    import torch

    if any(not isinstance(w, torch.Tensor) for w in (gate, up, down)):
        raise TypeError("weights must be Torch tensors")
    if any(w.device != gate.device or not w.is_cuda or w.dtype != torch.bfloat16 or not w.is_contiguous() or w.data_ptr() % 16 for w in (gate, up, down)):
        raise ValueError("weights must be contiguous, aligned CUDA BF16 tensors on one device")
    if gate.ndim != 3 or up.shape != gate.shape or down.shape != (gate.shape[0], gate.shape[2], gate.shape[1]):
        raise ValueError("expected gate/up[E,I,H] and down[E,H,I]")
    if gate.shape[0] != 4 or gate.shape[1] <= 0 or gate.shape[1] % 256 or gate.shape[2] <= 0 or gate.shape[2] % 512:
        raise ValueError("MXFP8 requires four experts, I a positive multiple of 256, and H a positive multiple of 512")
    with torch.cuda.device(gate.device):
        return _api(gate.device).quantize_mok_mxfp8_weights(gate, up, down)
