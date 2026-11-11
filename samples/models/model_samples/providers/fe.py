# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters to cuDNN Frontend entries.

``fe.gdn_decode_fused`` calls ``cudnn.GdnDecodeFused``, the fused decode step of a gated delta-rule linear-attention layer
(``docs/operations/linear_attention/gdn_decode_flashinfer_backend.md``): one launch per layer and step.  The entry takes the
layer's head counts and hidden width as a geometry, so one adapter serves every family; the state and conv-history pools are
the session's and are updated in place.  Only the public API is used, and ``cudnn`` is imported when the block is asked
whether it serves, never at module import.

The entry compiles once per (rows, operand strides) per process; ``compile`` builds operands with the layout the session uses
(``split_qkvz`` and ``state.RecurrentSlots``), so the first step of a run finds the kernel compiled.
"""

from __future__ import annotations

from ..state import RecurrentSlots
from ..weights import ScaledTensor
from . import BlockBase, GdnParams, split_qkvz

#: the files of the package this adapter runs, relative to the ``cudnn`` package: the source hash of ``check_environment``
SOURCE_FILES = (
    "__init__.py",
    "linear_attention/frost/gdn_decode_fused.py",
    "linear_attention/frost/kernel/gdn_decode_fused_config.py",
    "linear_attention/frost/kernel/gdn_decode_fused_f16.py",
    "linear_attention/frost/kernel/gdn_geometry.py",
)
#: the first cuDNN Frontend version that carries the fused decode step, and the CuTe DSL floor its page states
FIRST_VERSION = "1.33.0"
DSL_FLOOR = (4, 8, 0)


#: one entry per (device, geometry, epsilon) of the process: the entry compiles once per rows count, and every session shares it
_ENTRIES: dict = {}
#: whether a kernel of this adapter was compiled in this process (its first launch's reservation is then already on the device)
_WARM: list = []


class GdnDecodeFusedBlock(BlockBase):
    provider = "cuDNN FE"
    kernels = ("gdn_decode_fused",)
    per_row = (0, 1, 2, 3)
    state_args = (5,)
    compiles = True

    def _cudnn(self):
        import cudnn  # before torch in a sample's own imports; here torch is already loaded by the sample

        return cudnn

    def check_support(self, sequences: int, tokens: int) -> None:
        cudnn = self._cudnn()
        try:
            cudnn.GdnDecodeFused
        except AttributeError:
            raise NotImplementedError(
                f"needs nvidia-cudnn-frontend with the fused linear-attention decode step (>= {FIRST_VERSION}); this is {cudnn.__version__}: "
                "install from the checkout (see README)"
            ) from None
        if tokens != 1:
            raise NotImplementedError(f"cudnn.GdnDecodeFused is the one-token decode step; this run has {tokens} tokens per sequence")
        import importlib.metadata as metadata

        try:
            installed = metadata.version("nvidia-cutlass-dsl")
        except metadata.PackageNotFoundError:
            installed = None
        floor = ".".join(str(n) for n in DSL_FLOOR)
        if installed is None or tuple(int(n) for n in installed.split(".")[:3] if n.isdigit()) < DSL_FLOOR:
            raise NotImplementedError(f"cudnn.GdnDecodeFused needs nvidia-cutlass-dsl >= {floor}; this is {installed or 'not installed'}")
        if self.facts.total_bytes and "device" in self.options:
            self._build(self.options["device"])  # on a real device: the entry's own typed decline (a part that holds no form) is a SKIP

    def reserve_bytes(self) -> int:
        """Device memory the kernel's first launch reserves outside PyTorch's allocator, as the family states it per GPU class."""
        if _WARM:  # already launched in this process: the reservation is part of what the device reports as used
            return 0
        reserve = getattr(self.model, "first_launch_reserve", {})
        return int(reserve.get(f"cc{self.facts.cc[0]}.x", 0))

    def _build(self, device):
        import torch

        g = self.model.geometry
        device = torch.device(device)
        key = (str(device), g.la_key_heads, g.la_value_heads, g.hidden, self.model.norm_eps)
        if key not in _ENTRIES:
            cudnn = self._cudnn()
            from cudnn.linear_attention.frost import GdnGeometry

            _ENTRIES[key] = cudnn.GdnDecodeFused(device, eps=self.model.norm_eps, geometry=GdnGeometry(g.la_key_heads, g.la_value_heads, g.hidden))
        return _ENTRIES[key]

    def compile(self, sequences: int, tokens: int) -> None:
        import torch

        g = self.model.geometry
        device = torch.device(self.options.get("device", "cuda"))
        rows, D, NV = sequences * tokens, g.la_head_dim, g.la_value_heads
        qkv = (2 * g.la_key_heads + NV) * D
        bf16 = dict(dtype=torch.bfloat16, device=device)
        x, z = split_qkvz(torch.zeros((rows, qkv + NV * D), **bf16), qkv, NV, D)
        slots = RecurrentSlots(torch.zeros((rows + 1, NV, D, D), dtype=torch.float32, device=device), torch.zeros((rows + 1, g.la_conv_width - 1, qkv), **bf16))
        params = GdnParams(
            g.la_key_heads,
            NV,
            D,
            torch.zeros((qkv, g.la_conv_width), **bf16),
            torch.zeros(NV, dtype=torch.float32, device=device),
            torch.zeros(NV, **bf16),
            torch.ones(D, **bf16),
            torch.zeros((2 * NV, g.hidden), **bf16),
            torch.ones(1, dtype=torch.float32, device=device),
            1.0,
        )
        slot_ids = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        try:
            self._build(device).compile(*self._operands(x, z, torch.zeros((rows, g.hidden), **bf16), slot_ids, params, slots))
        except NotImplementedError as error:  # the plan found the part and the versions served: what is left is this adapter's own layout
            raise RuntimeError(f"cudnn.GdnDecodeFused declined the operands this adapter built: {error}") from error
        _WARM.append(rows)

    @staticmethod
    def _operands(x, z, hidden, slot_ids, p, slots):
        """The entry's positional operands: ``(rows, x, conv_state, conv_weight, a_log, dt_bias, state, slot_ids, z, norm_weight,
        out_scale, hidden, ba_weight)``."""
        return (x.shape[0], x, slots.conv, p.conv_weight, p.a_log, p.dt_bias, slots.state, slot_ids, z, p.norm_weight, p.out_scale, hidden, p.ba_weight)

    def __call__(self, x, z, hidden, slot_ids, params, slots):
        import torch

        try:
            qout = self._build(x.device)(*self._operands(x, z, hidden, slot_ids, params, slots))
        except NotImplementedError as error:
            raise RuntimeError(f"cudnn.GdnDecodeFused declined the operands of this step: {error}") from error
        return ScaledTensor(qout.view(torch.uint8).view(x.shape[0], -1), params.out_scale_value)
