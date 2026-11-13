# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters to cuDNN Frontend entries.

``fe.gdn_decode_fused`` calls ``cudnn.GdnDecodeFused``, the fused decode step of a gated delta-rule linear-attention layer
(``docs/operations/linear_attention/gdn_decode_flashinfer_backend.md``): one launch per layer and step.  The entry takes the
layer's head counts and hidden width as a geometry, so one adapter serves every family; the state and conv-history pools are
the session's and are updated in place.  Only the public API is used, and ``cudnn`` is imported when the block is asked
whether it serves, never at module import.

The entry compiles once per (rows, operand strides) per process; ``compile`` builds operands with the layout the session uses
(``split_qkvz`` and ``state.RecurrentSlots``) and launches the compiled form once on them, so the first step of a run finds the
kernel compiled and launched.  That first launch is measured: on some GPU classes it reserves device memory outside PyTorch's
allocator (``first_launch``), which the plan's memory sentence allows for per class before anything is loaded.

The provider row says in which form the entry is called (``row.form``: fields of the entry's own form, under the entry's names).
The adapter passes exactly those to the entry (``FORM_ARGUMENTS``) and nothing else that fixes the kernel, so the row is the
whole statement; ``launch_facts`` hands the kernel check the entry's own answers for a row count -- the symbols of the kernels it
compiled, its form and its launch shape (the entry's ``kernel_symbols_for`` / ``form_for`` / ``launch_shape_for``).
"""

from __future__ import annotations

import re

from ..census import Facts, Kernel
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
    "frost/launch_facts.py",
)
#: a field of the entry's form a provider row may name -> the constructor argument of ``cudnn.GdnDecodeFused`` that sets it
FORM_ARGUMENTS = {"PDL": "pdl", "PDL_RELEASE": "pdl_release"}
#: the entry's three launch facts: what the kernel check reads
LAUNCH_FACTS = ("kernel_symbols_for", "form_for", "launch_shape_for")
#: the first cuDNN Frontend version that carries the fused decode step, and the CuTe DSL floor its page states
FIRST_VERSION = "1.33.0"
DSL_FLOOR = (4, 8, 0)


def serves_floor(installed: str, floor: tuple) -> bool:
    """Whether a version string is at or above ``floor`` (three numbers).  The first three fields are read by their leading
    digits, so "4.8.1rc1" is 4.8.1 and serves a floor of 4.8.0; a pre-release or development build of the floor itself
    ("4.8.0rc1", "4.8.0.dev3") is below it."""
    public = installed.split("+")[0]
    numbers = [int(digits.group()) if (digits := re.match(r"\d+", field)) else 0 for field in public.split(".")[:3]]
    release = tuple(numbers + [0] * (3 - len(numbers)))
    before = re.match(r"\.?(a|b|rc|dev)", re.sub(r"^\d+(\.\d+)*", "", public), re.IGNORECASE)
    return release > tuple(floor) or (release == tuple(floor) and not before)


#: one entry per (device, geometry, epsilon, form) of the process: the entry compiles once per rows count, and every session shares it
_ENTRIES: dict = {}
#: rows -> device bytes the first launch of that form took outside PyTorch's allocator in this process, as measured
_FIRST_LAUNCH: dict = {}


def _outside_pool(device, launch) -> int:
    """Device bytes one launch took outside PyTorch's allocator: what the device reports as newly used across it, less what
    PyTorch's pool grew by.  Other processes on the GPU count in the first term."""
    import torch

    torch.cuda.synchronize(device)
    free, pool = torch.cuda.mem_get_info(device)[0], torch.cuda.memory_reserved(device)
    launch()
    torch.cuda.synchronize(device)
    return (free - torch.cuda.mem_get_info(device)[0]) - (torch.cuda.memory_reserved(device) - pool)


class GdnDecodeFusedBlock(BlockBase):
    provider = "cuDNN FE"
    kernels = (Kernel("gdn_decode_fused", "cuDNN FE", "exact", launches=1, shape="entry"),)  # one launch per layer and step
    per_row = (0, 1, 2, 3)
    state_args = (5,)
    compiles = True

    def _cudnn(self):
        import cudnn  # before torch in a sample's own imports; here torch is already loaded by the sample

        return cudnn

    def _arguments(self) -> dict:
        """The constructor arguments the row's form stands for.  A field this adapter cannot pass is a wrong table."""
        unknown = [name for name, _ in self.row.form if name not in FORM_ARGUMENTS]
        if unknown:
            raise ValueError(
                f"provider row {self.row.key}: its form names {', '.join(unknown)}; this adapter passes {', '.join(FORM_ARGUMENTS)} to cudnn.GdnDecodeFused"
            )
        return {FORM_ARGUMENTS[name]: value for name, value in self.row.form}

    def check_support(self, sequences: int, tokens: int) -> None:
        cudnn = self._cudnn()
        try:
            cudnn.GdnDecodeFused
        except AttributeError:
            raise NotImplementedError(
                f"needs nvidia-cudnn-frontend with the fused linear-attention decode step (>= {FIRST_VERSION}); this is {cudnn.__version__}: "
                "install from the checkout (see README)"
            ) from None
        self._arguments()
        if self.options.get("check_kernels") and not all(hasattr(cudnn.GdnDecodeFused, fact) for fact in LAUNCH_FACTS):
            raise NotImplementedError(
                f"the kernel check needs the launch facts of cudnn.GdnDecodeFused ({', '.join(LAUNCH_FACTS)}); this nvidia-cudnn-frontend "
                f"{cudnn.__version__} has none: install from the checkout (see README), or run without --check-kernels"
            )
        if tokens != 1:
            raise NotImplementedError(f"cudnn.GdnDecodeFused is the one-token decode step; this run has {tokens} tokens per sequence")
        import importlib.metadata as metadata

        try:
            installed = metadata.version("nvidia-cutlass-dsl")
        except metadata.PackageNotFoundError:
            installed = None
        floor = ".".join(str(n) for n in DSL_FLOOR)
        if installed is None or not serves_floor(installed, DSL_FLOOR):
            raise NotImplementedError(f"cudnn.GdnDecodeFused needs nvidia-cutlass-dsl >= {floor}; this is {installed or 'not installed'}")
        if self.facts.total_bytes and "device" in self.options:
            self._build(self.options["device"])  # on a real device: the entry's own typed decline (a part that holds no form) is a SKIP

    def reserve_bytes(self) -> int:
        """Device memory the kernel's first launch reserves outside PyTorch's allocator: the family's allowance for the GPU
        class, until a form was launched in this process (the reservation is then part of what the device reports as used)."""
        if _FIRST_LAUNCH:
            return 0
        reserve = getattr(self.model, "first_launch_reserve", {})
        return int(reserve.get(f"cc{self.facts.cc[0]}.x", 0))

    def first_launch(self) -> dict:
        return dict(_FIRST_LAUNCH)

    def _build(self, device):
        import torch

        g = self.model.geometry
        device = torch.device(device)
        key = (str(device), g.la_key_heads, g.la_value_heads, g.hidden, self.model.norm_eps, tuple(self.row.form))
        if key not in _ENTRIES:
            cudnn = self._cudnn()
            from cudnn.linear_attention.frost import GdnGeometry

            geometry = GdnGeometry(g.la_key_heads, g.la_value_heads, g.hidden)
            _ENTRIES[key] = cudnn.GdnDecodeFused(device, eps=self.model.norm_eps, geometry=geometry, **self._arguments())
        self._instance = list(_ENTRIES).index(key) + 1
        return _ENTRIES[key]

    def serving(self):
        """The entry instance of this adapter (one per device, geometry, epsilon and form of the process) and the row's form."""
        instance = getattr(self, "_instance", None)
        return (f"{self.entry} #{instance}" if instance else self.entry), tuple(self.row.form)

    def launch_facts(self, rows: int) -> Facts:
        """The entry's own answers for ``rows``: the symbols of the kernels it has compiled for that row count, its form as plain
        data and its launch shape.  The entry's typed declines pass through: a row that claims a row count the entry does not
        serve is a wrong table."""
        entry = self._build(self.options.get("device", "cuda"))
        return Facts(tuple(entry.kernel_symbols_for(rows)), entry.form_for(rows), tuple(entry.launch_shape_for(rows)))

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
        entry, operands = self._build(device), self._operands(x, z, torch.zeros((rows, g.hidden), **bf16), slot_ids, params, slots)
        try:
            entry.compile(*operands)
            if rows not in _FIRST_LAUNCH:
                _FIRST_LAUNCH[rows] = _outside_pool(device, lambda: entry(*operands))
        except NotImplementedError as error:  # the plan found the part and the versions served: what is left is this adapter's own layout
            raise RuntimeError(f"cudnn.GdnDecodeFused declined the operands this adapter built: {error}") from error

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
