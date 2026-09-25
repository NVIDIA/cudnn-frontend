# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN-accelerated drop-in for FLA 0.5.2's plain ``GatedMLP``.

The adapter is deliberately narrow.  It replaces only the arithmetic of the
stock, local, bias-free BF16 ``swish`` module with
``cudnn.gemm.ops.swiglu_mlp``.  Anything whose semantics may differ -- tensor
parallelism, tensor/module subclasses, quantization, LoRA, parametrizations,
hooks, non-compact storage, another FLA version, or graph compilation -- calls
the original FLA ``forward`` unchanged.
"""

from __future__ import annotations

import functools
from importlib import metadata

import torch

import cudnn

_SUPPORTED_FLA_VERSION = "0.5.2"
_DECLINE_ERRORS = (NotImplementedError, cudnn.cudnnGraphNotSupportedError, ImportError)
_LAST = {"path": None}

_MODULE_HOOK_ATTRS = (
    "_backward_hooks",
    "_backward_pre_hooks",
    "_forward_hooks",
    "_forward_pre_hooks",
)
_GLOBAL_MODULE_HOOK_ATTRS = (
    "_global_backward_hooks",
    "_global_backward_pre_hooks",
    "_global_forward_hooks",
    "_global_forward_pre_hooks",
)
_TENSOR_HOOK_ATTRS = ("_backward_hooks", "_post_accumulate_grad_hooks")


def last_path() -> str | None:
    """The route the most recent shimmed MLP call took, for telemetry/tests."""
    return _LAST["path"]


def _installed_fla_version() -> str | None:
    try:
        return metadata.version("flash-linear-attention")
    except metadata.PackageNotFoundError:
        return None


def _supports_installed_fla() -> bool:
    return _installed_fla_version() == _SUPPORTED_FLA_VERSION


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and compiler.is_compiling():
        return True
    dynamo = getattr(torch, "_dynamo", None)
    return bool(dynamo is not None and dynamo.is_compiling())


def _is_cuda_autocast_enabled() -> bool:
    try:
        return torch.is_autocast_enabled("cuda")
    except TypeError:  # torch versions whose no-argument form means CUDA
        return torch.is_autocast_enabled()


def _cuda_autocast_dtype():
    if not _is_cuda_autocast_enabled():
        return None
    try:
        return torch.get_autocast_dtype("cuda")
    except (AttributeError, TypeError):
        return torch.get_autocast_gpu_dtype()


def _is_cuda_tensor(tensor) -> bool:
    return tensor.is_cuda


def _device_capability(device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _call_native(x, gate_weight, up_weight, down_weight):
    # Keep importing cudnn.fla independent of the optional GEMM/CuTeDSL stack;
    # an unavailable native op is a typed decline to the original FLA method.
    from cudnn.gemm.ops import swiglu_mlp

    return swiglu_mlp(x, gate_weight, up_weight, down_weight)


def _has_hooks(obj, attrs) -> bool:
    return any(bool(getattr(obj, attr, None)) for attr in attrs)


def _has_global_module_hooks() -> bool:
    module_impl = torch.nn.modules.module
    return _has_hooks(module_impl, _GLOBAL_MODULE_HOOK_ATTRS)


def _is_parametrized(module) -> bool:
    parametrize = getattr(torch.nn.utils, "parametrize", None)
    return bool(parametrize is not None and parametrize.is_parametrized(module))


def _plain_parameter_decline(parameter, *, shape, device) -> str | None:
    if type(parameter) is not torch.nn.Parameter:
        return "tensor-subclass"
    if type(parameter.data) is not torch.Tensor or parameter.layout is not torch.strided:
        return "tensor-subclass"
    if parameter.dtype is not torch.bfloat16:
        return "non-bf16"
    if not _is_cuda_tensor(parameter) or parameter.device != device:
        return "nonlocal-device"
    if tuple(parameter.shape) != tuple(shape):
        return "shape"
    if not parameter.is_contiguous():
        return "noncontiguous"
    if not parameter.is_leaf or parameter.grad_fn is not None:
        return "nonlocal-parameter"
    if _has_hooks(parameter, _TENSOR_HOOK_ATTRS):
        return "hooks"
    return None


def _module_decline(module, *, expected_type, expected_parameter_names) -> str | None:
    if _is_parametrized(module):
        return "parametrized"
    if type(module) is not expected_type:
        return "custom-module"
    if "forward" in module.__dict__:
        return "custom-forward"
    if set(module._parameters) != set(expected_parameter_names) or module._buffers or module._modules:
        return "custom-module"
    if _has_hooks(module, _MODULE_HOOK_ATTRS):
        return "hooks"
    return None


def _decline_reason(self, x, *, gated_mlp_cls, swiglu_linear_cls, fla_version) -> str | None:
    # This check must precede all tensor/device introspection so torch.compile
    # traces the incumbent implementation rather than specializing this adapter.
    if _is_compiling() or torch.jit.is_scripting() or torch.jit.is_tracing():
        return "compile"
    if _cuda_autocast_dtype() not in (None, torch.bfloat16):
        # Stock FLA runs all three Linear modules under autocast.  Entering the
        # BF16-only fused op would instead bypass a different requested dtype.
        # BF16 autocast preserves the admitted operands/results and remains fast.
        return "autocast"
    if fla_version != _SUPPORTED_FLA_VERSION:
        return "fla-version"
    if type(self) is not gated_mlp_cls or "forward" in self.__dict__:
        return "custom-gated-mlp"
    if self._parameters or self._buffers or set(self._modules) != {"gate_proj", "up_proj", "down_proj", "swiglu_linear"}:
        return "custom-gated-mlp"
    if _is_parametrized(self) or _has_hooks(self, _MODULE_HOOK_ATTRS) or _has_global_module_hooks():
        return "hooks-or-parametrization"
    if getattr(self, "hidden_act", None) != "swish" or getattr(self, "fuse_swiglu", None) is not True:
        return "variant"

    projections = (self.gate_proj, self.up_proj, self.down_proj)
    for projection in projections:
        reason = _module_decline(projection, expected_type=torch.nn.Linear, expected_parameter_names=("weight", "bias"))
        if reason is not None:
            return reason
        if projection.bias is not None:
            return "bias"

    reason = _module_decline(self.swiglu_linear, expected_type=swiglu_linear_cls, expected_parameter_names=())
    if reason is not None:
        return reason

    if type(x) is not torch.Tensor or x.layout is not torch.strided:
        return "tensor-subclass"
    if x.dtype is not torch.bfloat16:
        return "non-bf16"
    if not _is_cuda_tensor(x):
        return "non-cuda"
    if x.dim() < 2 or x.numel() == 0 or not x.is_contiguous():
        return "shape-or-layout"
    if _has_hooks(x, _TENSOR_HOOK_ATTRS):
        return "hooks"
    if _device_capability(x.device)[0] != 10:
        return "non-sm100"

    hidden = getattr(self, "hidden_size", None)
    intermediate = getattr(self, "intermediate_size", None)
    if not isinstance(hidden, int) or not isinstance(intermediate, int) or hidden <= 0 or intermediate <= 0:
        return "shape"
    if hidden % 8 or intermediate % 8 or x.shape[-1] != hidden:
        return "shape"
    expected_linears = (
        (self.gate_proj, hidden, intermediate),
        (self.up_proj, hidden, intermediate),
        (self.down_proj, intermediate, hidden),
    )
    for projection, in_features, out_features in expected_linears:
        if projection.in_features != in_features or projection.out_features != out_features:
            return "shape"
        reason = _plain_parameter_decline(projection.weight, shape=(out_features, in_features), device=x.device)
        if reason is not None:
            return reason
    return None


def make_gated_mlp_forward(real_forward, gated_mlp_cls, swiglu_linear_cls):
    """Wrap FLA's class method with a fail-closed cuDNN SwiGLU-MLP path.

    The wrapper patches the class, so it covers both existing and future
    instances.  FLA 0.5.2 accepts arbitrary ``**kwargs`` and ignores them; the
    native path deliberately does the same, while fallback forwards the exact
    mapping to ``real_forward``.
    """

    fla_version = _installed_fla_version()

    @functools.wraps(real_forward)
    def forward(self, x, **kwargs):
        reason = _decline_reason(
            self,
            x,
            gated_mlp_cls=gated_mlp_cls,
            swiglu_linear_cls=swiglu_linear_cls,
            fla_version=fla_version,
        )
        if reason is not None:
            _LAST["path"] = f"fallback:{reason}"
            return real_forward(self, x, **kwargs)
        # Mark the selected route before entering the custom autograd op.
        # Non-reentrant checkpointing deliberately raises an internal
        # control-flow exception while replaying the op; it must pass through
        # untouched without being misreported as a native execution failure.
        _LAST["path"] = "native"
        try:
            out = _call_native(x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)
        except _DECLINE_ERRORS as error:
            _LAST["path"] = f"fallback:{type(error).__name__}"
            return real_forward(self, x, **kwargs)
        except (RuntimeError, ValueError) as error:
            _LAST["path"] = f"error:{type(error).__name__}"
            raise
        return out

    forward.__cudnn_fla_target__ = "gated_mlp"
    return forward


__all__ = ["last_path", "make_gated_mlp_forward"]
