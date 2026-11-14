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

``fe.attention_gate_quant_decode`` calls ``cudnn.AttentionGateQuantDecode``, the output gate of an attention layer fused with
the static 8-bit quantisation of the out-projection's input (``docs/operations/attention/gate_quant_decode.md``): one launch
per attention layer and step.  One compiled kernel serves every row count and layer.  The adapter hands the entry an output
buffer of its own per call: the entry's own buffer is one for every row count and is overwritten by its next call.  Its
kernel rounds the gated value at other points than the reference chain does, so the adapter names the reference block it is
compared with (``twin``): the gate written at the kernel's rounding points.

A provider row says in which form its entry is called (``row.form``: fields of the entry's own form, under the entry's names).
An adapter passes exactly those to the entry and nothing else that fixes the kernel, so the row is the whole statement;
``launch_facts`` hands the kernel check the entry's own answers for a row count -- the symbols of the kernels it compiled, its
form and its launch shape (the entry's ``kernel_symbols_for`` / ``form_for`` / ``launch_shape_for``).  ``serving`` says what
served the last call: the entry instance ``__call__`` launched and the row's fields as that instance states them.  On a real
device an adapter asks its entry, while the run is planned, for the launch of the run's row count: an entry that declines what its row claims is a
``ValueError`` -- the provider table is wrong, which a sample reports as an ``ERROR``, never as a ``SKIP``.

**First use is part of the plan.**  An entry gates the installed toolchain where it builds its kernel (the CuTe DSL's target
list, on some GPU classes its compiler), not where it is constructed.  So on a CUDA device an adapter builds the kernel of
the run's row count on scratch operands while the run is planned (``first_use``), before anything is loaded: what the
entry needs of the installation is then a ``SKIP`` that names the fix, in the entry's own words, and the build after the
load finds its kernel built.  A decline at that point is the installation's only where this adapter can name a
requirement the installation does not meet (``_unmet``: the part's target by cuDNN Frontend's public probe, and what the
entry's page states beyond it); any other decline is of the operands this adapter built -- a ``ValueError``, an ``ERROR``
before the load, because a faulty adapter must not read as an unsupported GPU.

**The memory sentence comes first.**  The scratch operands of a first use are as large as the run's own (rows + 1 slots of
recurrent state for the fused step), so the first use is not made where a block is asked whether it serves
(``check_support`` allocates nothing): a plan makes it as a step of its own (``Plan.first_use``), after the memory
sentence has said that the run fits and before anything is loaded.  A run that does not fit is a ``SKIP`` with its
estimate, and nothing was allocated for it.
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
    "attention_gate/frost/gate_quant_decode.py",
    "attention_gate/frost/kernel/gate_quant_decode_config.py",
    "attention_gate/frost/kernel/gate_quant_decode_f16.py",
)
#: a field of the entry's form a provider row may name -> the constructor argument of ``cudnn.GdnDecodeFused`` that sets it
FORM_ARGUMENTS = {"PDL": "pdl", "PDL_RELEASE": "pdl_release"}
#: the fields of the form of ``cudnn.AttentionGateQuantDecode`` a provider row may name: ``THREADS`` goes into the entry's
#: config (``cfg=Cfg(THREADS=...)``), ``PDL_RELEASE`` is its launch knob ``pdl_release=``
GATE_FORM_FIELDS = ("THREADS", "PDL_RELEASE")
#: the entry's three launch facts: what the kernel check reads
LAUNCH_FACTS = ("kernel_symbols_for", "form_for", "launch_shape_for")
#: the first cuDNN Frontend version that carries the fused decode step and the gate entry, and the CuTe DSL floor their pages state
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


#: the entries of the process, one per (entry, device, geometry, form): an entry compiles once per form, and every session shares it
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


class EntryBlock(BlockBase):
    """An adapter to a cuDNN Frontend entry that states its launch: what such adapters share.  A subclass names the entry
    (``name``), builds it from the row's form (``_build``) and notes in ``__call__`` the entry object it launched and the
    row count (``_served``); the call log and the kernel check read the rest from here."""

    provider = "cuDNN FE"
    compiles = True
    name = ""  # the entry, as ``cudnn`` exports it
    what = ""  # what the entry is, for the sentence of an older package
    _served = None  # (entry instance, rows) of the last call
    _answer = None  # (entry instance, rows, what ``serving`` says of them): asked of an instance once, not once per call

    def _cudnn(self):
        import cudnn  # before torch in a sample's own imports; here torch is already loaded by the sample

        return cudnn

    def _build(self, device):
        raise NotImplementedError

    def _check_package(self) -> None:
        """The declines of the installed packages, each a ``NotImplementedError`` (a SKIP) that names the fix: a cuDNN
        Frontend without the entry, one whose entry states no launch facts where the run checks its kernels, a CuTe DSL
        below the entry's floor."""
        cudnn = self._cudnn()
        entry = getattr(cudnn, self.name, None)
        if entry is None:
            raise NotImplementedError(
                f"needs nvidia-cudnn-frontend with {self.what} (>= {FIRST_VERSION}); this is {cudnn.__version__}: install from the checkout (see README)"
            )
        if self.options.get("check_kernels") and not all(hasattr(entry, fact) for fact in LAUNCH_FACTS):
            raise NotImplementedError(
                f"the kernel check needs the launch facts of {self.entry} ({', '.join(LAUNCH_FACTS)}); this nvidia-cudnn-frontend "
                f"{cudnn.__version__} has none: install from the checkout (see README), or run without --check-kernels"
            )
        import importlib.metadata as metadata

        try:
            installed = metadata.version("nvidia-cutlass-dsl")
        except metadata.PackageNotFoundError:
            installed = None
        floor = ".".join(str(n) for n in DSL_FLOOR)
        if installed is None or not serves_floor(installed, DSL_FLOOR):
            raise NotImplementedError(f"{self.entry} needs nvidia-cutlass-dsl >= {floor}; this is {installed or 'not installed'}")

    def _check_claim(self, entry, rows: int) -> None:
        """On a real device: the row claims this run, so the entry must state a launch for its row count.  One that declines
        is a wrong table -- a ``ValueError``, which a sample reports as an ``ERROR`` and never as a ``SKIP``."""
        if not hasattr(entry, "launch_shape_for"):
            return
        try:
            entry.launch_shape_for(rows)
        except (ValueError, NotImplementedError) as error:
            cc = self.facts.cc
            raise ValueError(
                f"the provider table is wrong: the row of {self.row.key} claims this run ({rows} row{'s' if rows != 1 else ''} on compute "
                f"capability {cc[0]}.{cc[1]}), and {self.entry} declines it: {error}"
            ) from None

    #: the device types on which the plan makes the entry's first use: a kernel can be built for a CUDA device only (a host
    #: test names its own device here to run the step on a stand-in entry)
    first_use_devices = ("cuda",)

    def _scratch(self, sequences: int, tokens: int):
        """``(entry, operands)``: the entry of this row and scratch operands in the layout a step hands it."""
        raise NotImplementedError

    def _unmet(self) -> bool:
        """Whether the installation lacks something the entry needs on this GPU class, as far as this adapter can tell without
        the entry: the CuTe DSL's target of the part, by cuDNN Frontend's public probe.  A subclass adds what its entry's
        page states beyond that."""
        from cudnn.frost import cutedsl_runtime_available

        return not cutedsl_runtime_available(tuple(self.facts.cc))

    def _declined(self, error, what: str) -> str:
        """The sentence for a typed decline of the entry at ``what``: the installation's where this adapter can name an unmet
        requirement, else the operands this adapter built."""
        if self._unmet():
            return f"{self.entry} declines this installation at {what}: {error}"
        return f"{self.entry} declined the operands this adapter built for {what}: {error}"

    def first_use(self, sequences: int, tokens: int) -> None:
        """While the run is planned, on a real CUDA device: build the entry's kernel for the run's row count on scratch operands.
        The entry gates the installed toolchain there, so its decline comes before the load: a ``NotImplementedError`` (a
        SKIP, in the entry's own words) where the installation does not meet a requirement this adapter can name
        (``_unmet``), a ``ValueError`` (an ERROR) for any other decline -- then the operands are this adapter's.

        The scratch operands are the size of the run (the fused step's recurrent state at 65,535 rows is 192 GiB), so this
        is not part of ``check_support``: a plan makes it after its memory sentence has held (``Plan.first_use``), and a
        run that does not fit the GPU is a SKIP with its estimate before anything is allocated for it."""
        import torch

        if not self.facts.total_bytes or "device" not in self.options:  # a plan on stated device facts builds nothing
            return
        if torch.device(self.options["device"]).type not in self.first_use_devices:
            return
        entry, operands = self._scratch(sequences, tokens)
        try:
            entry.compile(*operands)
        except NotImplementedError as error:
            if self._unmet():
                raise
            raise ValueError(self._declined(error, "its first use")) from None

    def serving(self):
        """What served the last call of this block: the entry instance ``__call__`` launched (its place among the entries of
        its kind in this process: one per device, geometry and form) and the fields of the row's form as THAT instance states
        them for the call's row count (its ``form_for``).  Read from the object that was called, never from the row: an entry
        built without the row's knobs, or a call served from another instance, shows here.  Before the first call nothing
        has served and nothing was passed."""
        if self._served is None:
            return self.entry, ()
        entry, rows = self._served
        if self._answer is None or self._answer[0] is not entry or self._answer[1] != rows:
            place = next((n for n, built in enumerate((e for e in _ENTRIES.values() if type(e) is type(entry)), 1) if built is entry), 0)
            try:
                stated = entry.form_for(rows)
            except (ValueError, NotImplementedError):  # no form stated for rows it has just served: the log shows none
                stated = {}
            form = tuple((name, stated[name]) for name, _ in self.row.form if name in stated)
            self._answer = (entry, rows, (f"{self.entry} #{place}" if place else self.entry, form))
        return self._answer[2]

    def launch_facts(self, rows: int) -> Facts:
        """The entry's own answers for ``rows``: the symbols of the kernels it has compiled for that row count, its form as plain
        data and its launch shape.  The entry's typed declines pass through: a row that claims a row count the entry does not
        serve is a wrong table."""
        entry = self._build(self.options.get("device", "cuda"))
        return Facts(tuple(entry.kernel_symbols_for(rows)), entry.form_for(rows), tuple(entry.launch_shape_for(rows)))


class GdnDecodeFusedBlock(EntryBlock):
    name = "GdnDecodeFused"
    what = "the fused linear-attention decode step"
    kernels = (Kernel("gdn_decode_fused", "cuDNN FE", "exact", launches=1, shape="entry"),)  # one launch per layer and step
    per_row = (0, 1, 2, 3)
    state_args = (5,)

    def _arguments(self) -> dict:
        """The constructor arguments the row's form stands for.  A field this adapter cannot pass is a wrong table."""
        unknown = [name for name, _ in self.row.form if name not in FORM_ARGUMENTS]
        if unknown:
            raise ValueError(
                f"provider row {self.row.key}: its form names {', '.join(unknown)}; this adapter passes {', '.join(FORM_ARGUMENTS)} to cudnn.GdnDecodeFused"
            )
        return {FORM_ARGUMENTS[name]: value for name, value in self.row.form}

    def check_support(self, sequences: int, tokens: int) -> None:
        self._check_package()
        self._arguments()
        if tokens != 1:
            raise NotImplementedError(f"cudnn.GdnDecodeFused is the one-token decode step; this run has {tokens} tokens per sequence")
        if self.facts.total_bytes and "device" in self.options:
            # on a real device: the entry's own typed decline (a part that holds no form) is a SKIP; once it is built, a row
            # count it does not serve is a wrong table.  Its first use, where it gates the installed toolchain, allocates
            # scratch operands of the run's size and is made after the memory sentence has held (``first_use``)
            self._check_claim(self._build(self.options["device"]), sequences * tokens)

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
        return _ENTRIES[key]

    def _scratch(self, sequences: int, tokens: int):
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
        return self._build(device), self._operands(x, z, torch.zeros((rows, g.hidden), **bf16), slot_ids, params, slots)

    def compile(self, sequences: int, tokens: int) -> None:
        import torch

        device, rows = torch.device(self.options.get("device", "cuda")), sequences * tokens
        entry, operands = self._scratch(sequences, tokens)
        try:
            entry.compile(*operands)  # built while the run was planned (``first_use``): the entry finds its kernel
            if rows not in _FIRST_LAUNCH:
                _FIRST_LAUNCH[rows] = _outside_pool(device, lambda: entry(*operands))
        except NotImplementedError as error:  # after the plan nothing skips: the sentence says whose the decline is
            raise RuntimeError(self._declined(error, "its first launch")) from error

    @staticmethod
    def _operands(x, z, hidden, slot_ids, p, slots):
        """The entry's positional operands: ``(rows, x, conv_state, conv_weight, a_log, dt_bias, state, slot_ids, z, norm_weight,
        out_scale, hidden, ba_weight)``."""
        return (x.shape[0], x, slots.conv, p.conv_weight, p.a_log, p.dt_bias, slots.state, slot_ids, z, p.norm_weight, p.out_scale, hidden, p.ba_weight)

    def __call__(self, x, z, hidden, slot_ids, params, slots):
        import torch

        entry = self._build(x.device)
        self._served = (entry, int(x.shape[0]))  # what the call log of a traced step reads back (``serving``)
        try:
            qout = entry(*self._operands(x, z, hidden, slot_ids, params, slots))
        except NotImplementedError as error:
            raise RuntimeError(f"cudnn.GdnDecodeFused declined the operands of this step: {error}") from error
        return ScaledTensor(qout.view(torch.uint8).view(x.shape[0], -1), params.out_scale_value)


class AttentionGateQuantDecodeBlock(EntryBlock):
    """``cudnn.AttentionGateQuantDecode``: ``code = e4m3(bf16(attn / (1 + exp(-gate))) * (1 / scale))`` per element of the
    attention output, one launch per attention layer and step."""

    name = "AttentionGateQuantDecode"
    what = "the attention gate + quantise decode step"
    kernels = (Kernel("gate_quant", "cuDNN FE", "exact", launches=1, shape="entry"),)  # one launch per attention layer and step
    per_row = (0, 1)
    _twin = None
    _scales = None  # scale value -> the one-element fp32 tensor the kernel reads: built once per value, never inside a later step

    def _width(self) -> int:
        g = self.model.geometry
        return g.attn_heads * g.attn_head_dim

    def _arguments(self) -> dict:
        """The constructor arguments the row's form stands for.  A field this adapter cannot pass is a wrong table."""
        unknown = [name for name, _ in self.row.form if name not in GATE_FORM_FIELDS]
        if unknown:
            raise ValueError(
                f"provider row {self.row.key}: its form names {', '.join(unknown)}; this adapter passes {', '.join(GATE_FORM_FIELDS)} "
                "to cudnn.AttentionGateQuantDecode"
            )
        form, arguments = dict(self.row.form), {}
        if "THREADS" in form:
            from cudnn.attention_gate.frost import Cfg

            arguments["cfg"] = Cfg(THREADS=form["THREADS"])
        if "PDL_RELEASE" in form:
            arguments["pdl_release"] = form["PDL_RELEASE"]
        return arguments

    def check_support(self, sequences: int, tokens: int) -> None:
        self._check_package()
        self._arguments()
        if self.facts.total_bytes and "device" in self.options:
            self._check_claim(self._build(self.options["device"]), sequences * tokens)  # the first use follows the memory sentence

    def _unmet(self) -> bool:
        """Beyond the part's target: on compute capability 10.7 the entry's page asks for the CUDA 13 flavour of the CuTe DSL
        (the CUDA 12 flavour lists the target and cannot assemble it)."""
        if super()._unmet():
            return True
        if tuple(self.facts.cc) != (10, 7):
            return False
        from cutlass.base_dsl.version_info import CUDA_VERSION  # the compiler the installed CuTe DSL bound; the entry has imported the DSL by now

        return int(CUDA_VERSION.major) < 13

    def _build(self, device):
        import torch

        device = torch.device(device)
        key = ("attention_gate_quant_decode", str(device), self._width(), tuple(self.row.form))
        if key not in _ENTRIES:
            try:
                _ENTRIES[key] = self._cudnn().AttentionGateQuantDecode(device, self._width(), **self._arguments())
            except NotImplementedError as error:  # the entry declines the width the family's table hands it
                raise ValueError(f"the provider table is wrong: {self.entry} declines the geometry of the row of {self.row.key}: {error}") from None
        return _ENTRIES[key]

    def _scale(self, value: float, device):
        import torch

        if self._scales is None:
            self._scales = {}
        key = (float(value), str(device))
        if key not in self._scales:
            self._scales[key] = torch.tensor([float(value)], dtype=torch.float32, device=device)
        return self._scales[key]

    def twin(self, reference):
        """The reference block this adapter is compared with: the gate at the kernel's rounding points (the gated value as a
        quotient, rounded once to bf16), not the reference chain's product of a rounded sigmoid."""
        from . import torch_ref

        if self._twin is None:
            self._twin = torch_ref.GateQuantizeQuotient(reference.row, self.model, self.facts, self.options)
        return self._twin

    def _scratch(self, sequences: int, tokens: int):
        import torch

        device = torch.device(self.options.get("device", "cuda"))
        rows, width = sequences * tokens, self._width()
        attn = torch.zeros((rows, width), dtype=torch.bfloat16, device=device)
        out = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=device)
        return self._build(device), (attn, torch.zeros_like(attn), self._scale(1.0, device), out)

    def compile(self, sequences: int, tokens: int) -> None:
        entry, operands = self._scratch(sequences, tokens)
        try:
            entry.compile(*operands)  # built while the run was planned (``first_use``): the entry finds its kernel
            entry(*operands)  # the first launch of the compiled kernel, before the first step
        except NotImplementedError as error:  # after the plan nothing skips: the sentence says whose the decline is
            raise RuntimeError(self._declined(error, "its first launch")) from error

    def __call__(self, o, gate, quant):
        import torch

        if quant.fmt != "fp8":
            raise ValueError(f"cudnn.AttentionGateQuantDecode writes static 8-bit codes; this block is asked for {quant.fmt!r}")
        entry, rows = self._build(o.device), int(o.shape[0])
        self._served = (entry, rows)  # what the call log of a traced step reads back (``serving``)
        out = torch.empty((rows, o.shape[1]), dtype=torch.float8_e4m3fn, device=o.device)  # this call's own: never the entry's one buffer
        try:
            entry(o, gate, self._scale(quant.scale, o.device), out)
        except NotImplementedError as error:
            raise RuntimeError(f"cudnn.AttentionGateQuantDecode declined the operands of this step: {error}") from error
        return ScaledTensor(out.view(torch.uint8), float(quant.scale))
