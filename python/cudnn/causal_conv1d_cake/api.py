# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared deterministic Cake width-four causal convolution with SiLU."""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

from cudnn._buffer_utils import check_buffers_disjoint


def _geometry(tokens, channels, sequences):
    if any(type(n) is not int or not 0 < n < 2**31 for n in (tokens, channels, sequences)) or channels % 8 or sequences > tokens:
        raise ValueError("tokens/channels/sequences must be positive int32, channels divisible by 8, and sequences <= tokens")


def _tensor(tensor, name, shape, dtype, device, alignment=16):
    import torch

    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {device}")
    if not tensor.is_contiguous() or tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")


@functools.lru_cache(maxsize=None)
def _validator(arch, nvcc, cuda):
    from cudnn._cake.loader import JitSpec

    source = Path(__file__).with_name("validate.cu")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    spec = JitSpec(
        "causal_conv1d",
        f"validate_{digest}_{arch}",
        nvcc=nvcc,
        cuda=cuda,
        flags=["-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC", f"-arch={arch}"],
        include_dirs=[],
        sources=[str(source)],
        ldflags=["-lcudart"],
    )
    return spec.build_and_load().validate


class CausalConv1dPlan:
    """Prepared kernels for contiguous [tokens, channels] data and strict packed prefixes."""

    def __init__(self, tokens, channels, sequences, *, weight_dtype=None, bias_dtype=None, device=None):
        import torch

        from cudnn._cake import loader

        _geometry(tokens, channels, sequences)
        self.tokens, self.channels, self.sequences = tokens, channels, sequences
        self.shape = (tokens, channels)
        self.weight_dtype = torch.float32 if weight_dtype is None else weight_dtype
        self.bias_dtype = bias_dtype
        if self.weight_dtype not in (torch.float32, torch.bfloat16) or bias_dtype not in (None, torch.float32, torch.bfloat16):
            raise ValueError("weight and bias must be FP32 or BF16")
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        if device.type != "cuda":
            raise ValueError("device must be CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        arch = loader.device_arch(self.device)
        cuda = loader.check_toolkit(arch, "causal convolution")
        package = loader.load_package("cake_causal_conv1d_det_train")
        jit = package.cake_jit
        self._flags = package._flags
        with torch.cuda.device(self.device):
            forward = package._route(
                arch, jit.forward_route(bias=bias_dtype is not None, bf16_taps=self.weight_dtype == torch.bfloat16), needs="forward", C=channels, S=tokens
            )
            backward = package._route(
                arch,
                jit.backward_route(bias=bias_dtype is not None, fp32_grad=True, bf16_taps_or_dweight=self.weight_dtype == torch.bfloat16),
                needs="backward",
                C=channels,
                S=tokens,
            )
            self._scalars, self._launches = [], []
            self.work_words = 0
            for direction, names in enumerate((forward, backward)):
                main = jit.MODULES[names[0]]
                scalars, grid = package._launch(main, tokens, channels, sequences, device_index=self.device.index)
                self._scalars.append(package._scalars(scalars))
                if direction == 0:
                    self.work_words = max(jit.counter_words(jit.MODULES[name], scalars["n_ch_tiles"]) for name in names)
                else:
                    self._partial_rows = jit.partial_rows(grid[0], scalars["n_ch_tiles"])
                    self._partial_taps = int(main["workspace_rule"]["taps"])
                    self._partial_bytes = jit.partial_bytes(main, channels, self._partial_rows)
                    self._scalars[1]["main_ctas"] = grid[0]
                    names = (names[0], main["reduce"])
                launches = []
                for index, name in enumerate(names):
                    record = jit.MODULES[name]
                    launch_grid = grid if index == 0 else jit.launch_grid(record, device_index=self.device.index, C=channels, S=tokens)
                    grids = dict(zip(("grid_x", "grid_y", "grid_z"), launch_grid))
                    arguments = tuple((kind, grids[key] if kind == "grid" else record["logical"][key]) for kind, key in record["arg_plan"])
                    launches.append((getattr(jit.load(name, arch), record["ffi_entry"]), arguments))
                self._launches.append(launches)
            self._validate_prefix = _validator(arch, loader.find_nvcc(), cuda)

    def get_workspace_size(self):
        return max(4 * self.work_words, self._partial_bytes)

    def _inputs(self, x, weight, cu_seqlens, bias, workspace):
        import torch

        _tensor(x, "x", self.shape, torch.bfloat16, self.device)
        _tensor(weight, "weight", (self.channels, 4), self.weight_dtype, self.device)
        _tensor(cu_seqlens, "cu_seqlens", (self.sequences + 1,), torch.int32, self.device, 4)
        if self.bias_dtype is None:
            if bias is not None:
                raise ValueError("this plan has no bias")
        else:
            _tensor(bias, "bias", (self.channels,), self.bias_dtype, self.device)
        if not isinstance(workspace, torch.Tensor) or workspace.ndim != 1 or workspace.numel() < self.get_workspace_size():
            raise ValueError(f"workspace needs at least {self.get_workspace_size()} uint8 elements")
        _tensor(workspace, "workspace", tuple(workspace.shape), torch.uint8, self.device)
        return {"x": x, "w": weight, "cu": cu_seqlens, "b": bias}

    def _run(self, direction, values):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            self._validate_prefix(values["cu"], self.tokens)
            for entry, arguments in self._launches[direction]:
                entry(*(key if kind == "grid" else values[key] for kind, key in arguments))

    def forward(self, x, weight, cu_seqlens, out, workspace, *, bias=None, stream=None):
        import torch

        from cudnn._torch_stream import stream_context

        values = self._inputs(x, weight, cu_seqlens, bias, workspace)
        _tensor(out, "out", self.shape, torch.bfloat16, self.device)
        readers = [("x", x), ("weight", weight), ("cu_seqlens", cu_seqlens)] + ([] if bias is None else [("bias", bias)])
        check_buffers_disjoint([("out", out), ("workspace", workspace)], readers)
        work = workspace[: 4 * self.work_words].view(torch.int32)
        values.update(self._scalars[0], y=out, work=work, flags=self._flags(weight, b=bias))
        with torch.cuda.device(self.device), stream_context(stream, self.device):
            work.zero_()
            self._run(0, values)

    def backward(self, grad_out, x, weight, cu_seqlens, grad_x, grad_weight, workspace, *, bias=None, grad_bias=None, stream=None):
        import torch

        from cudnn._torch_stream import stream_context

        values = self._inputs(x, weight, cu_seqlens, bias, workspace)
        _tensor(grad_out, "grad_out", self.shape, torch.bfloat16, self.device)
        _tensor(grad_x, "grad_x", self.shape, torch.bfloat16, self.device)
        _tensor(grad_weight, "grad_weight", (self.channels, 4), self.weight_dtype, self.device)
        if self.bias_dtype is None:
            if grad_bias is not None:
                raise ValueError("this plan has no bias gradient")
        else:
            _tensor(grad_bias, "grad_bias", (self.channels,), self.bias_dtype, self.device)
        writers = [("grad_x", grad_x), ("grad_weight", grad_weight), ("workspace", workspace)] + ([] if grad_bias is None else [("grad_bias", grad_bias)])
        readers = [("grad_out", grad_out), ("x", x), ("weight", weight), ("cu_seqlens", cu_seqlens)] + ([] if bias is None else [("bias", bias)])
        check_buffers_disjoint(writers, readers)
        partial = workspace[: self._partial_bytes].view(torch.float32).view(self._partial_rows, self.channels, self._partial_taps)
        values.update(
            self._scalars[1], dy=grad_out, dx=grad_x, dw=grad_weight, db=grad_bias, partial=partial, flags=self._flags(weight, grad_weight, bias, grad_bias)
        )
        with torch.cuda.device(self.device), stream_context(stream, self.device):
            self._run(1, values)
