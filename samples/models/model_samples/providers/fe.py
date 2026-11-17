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

``fe.gdn_verify_fused`` calls ``cudnn.GdnVerifyFused``, the fused VERIFY step of the same layer (the same page): one launch
per layer and step for ``T`` tokens per sequence at once, the recurrent state after every token written to a slot of its
own.  The step shape is the entry's compile-time fact, so the row names the tokens per sequence as a field of the form
(``TOKENS``) beside the two launch knobs, one row per count; the adapter passes the knobs to the entry and holds the slot
table of every call to the row's ``TOKENS`` (a table of another count is a ``ValueError``: the wiring hands the block a
step its row does not name).  It is handed the step's slot table (``state.TokenSlots``: the slot of every (sequence,
token), the first row of every sequence, the tokens the last step kept) and the pools in the layout of
``state.RecurrentSlots`` at that many tokens.  The output is a buffer of the call's own.  The entry's facts take the step
shape ``(sequences, tokens)``; the name of its kernel carries neither the launch knobs nor the tokens (they sit behind
the cut of the symbol), so the form on the objects is what shows them.  Its determinism row says that one sequence alone
is not byte-equal to the same sequence among several (``batch_note``): a run prints it, and gates no batch invariance
on this row.

``fe.attention_gate_quant_decode`` calls ``cudnn.AttentionGateQuantDecode``, the output gate of an attention layer fused with
the static 8-bit quantisation of the out-projection's input (``docs/operations/attention/gate_quant_decode.md``): one launch
per attention layer and step.  One compiled kernel serves every row count and layer.  The adapter hands the entry an output
buffer of its own per call: the entry's own buffer is one for every row count and is overwritten by its next call.  Its
kernel rounds the gated value at other points than the reference chain does, so the adapter names the reference block it is
compared with (``twin``): the gate written at the kernel's rounding points.

``fe.add_norm_quant_decode`` calls ``cudnn.AddNormQuantDecode``, the residual add, the RMSNorm and the quantisation of the
next block's operand in one launch (``docs/operations/normalizations/add_norm_quant_decode.md``): one launch per call of the
norm block.  The entry is one compiled kernel per FORM -- which residual mode, which outputs -- and a step calls the block
at several sites, each in a form of its own: the row names them (``ProviderRow.sites``), the family's step hands the site
over as the block's last operand, and the adapter holds one entry instance per form and serves each call with the instance
of its site.  It decides nothing: a call whose operands do not fit the form the table names for its site (a residual where
the form starts the stream, codes of another format than the consumer reads) is a ``ValueError``.  The kernel updates the
residual stream in place, as the form says; the adapter hands it buffers of its own for everything else it writes, per
call.  What a consumer on a fallback row needs beyond the kernel's bytes is that consumer's to launch: the block scales
stay in the kernel's 128 x 4 layout (``weights.TiledBlockScaledTensor``), and where the form of a site ends at the bf16 rows
the consumer's quantised operand is handed over unquantised (``weights.Unquantized``).  The kernel normalises in fp32 and
rounds once, so the adapter names its twin: the norm at the kernel's rounding points.

``fe.fp8_projection_decode`` calls ``cudnn.Fp8ProjectionDecode``, the multiply of an 8-bit activation by an 8-bit weight on
the stored codes of both (``docs/operations/gemm_fusions/fp8_projection_decode.md``): one launch per call.  An entry is one
weight shape at one split, and the in-projection of a layer has another shape in each kind of layer: the row names the
shape per call site, and the adapter holds one entry instance per site.  The row's form is the whole kernel -- the split,
the weight tile, the ring depth, the tensor-memory columns and their release, the launch knobs -- and the adapter passes
it as given, so the entry never runs on its own pick.  The weight is the merged codes the loader kept for this block
(``weights.Dequantized.stored``; ``stored_weights`` asks the loader for them, and where no other block of the run reads
the weight's bf16 copy the loader makes none), the scale the fp32 product of the two static scales as one element on the
device, formed on the host once per pair of values, and the output a buffer of the call's own.  The kernel accumulates
in fp32 in a fixed order and rounds once, as the reference multiply does in its own precision: the reference block of
the same name is its twin.

``fe.swiglu_fp4_decode`` calls ``cudnn.SwigluFp4Decode``, the merged gate / up multiply of an MLP on the packed 4-bit codes
of both operands with ``SiLU(gate) * up`` and the 4-bit quantisation of the result in its epilogue
(``docs/operations/gemm_fusions/swiglu_fp4_decode.md``): one launch per call, and nothing of the product is stored.  The
row's form is the whole kernel -- the token tile of its row count, the weight stages issued ahead of the wait, the L2
policy of the weight stream, the cap of the persistent grid, the launch attribute -- and the adapter passes it as one
config, so the entry never runs on its own default; the token tile follows the row count, so the table has one row per
tile and the adapter one entry instance per row.  The kernel reads the weight in a layout of its own, the rows in tiles
of 64 gate + 64 up channels with the block scales in the 128 x 4 layout: the adapter builds it once per weight at load
with the entry's own helper (``stored_layout``, which the loader calls for a block that says ``stored_weights``), and
that tensor is the one form of the stored weight the load keeps.  The operand is the norm's 4-bit output: from the
norm's kernel it comes in the layout this kernel reads and is handed over as it is; from a block that is not a kernel
(a reference chain, or one row of a batch in a check) the block scales come row by row, and bringing them into the
layout is this block's own launch, made when it reads its operand.  The two scalars are one fp32 element each on the
device, formed on the host once per value: the fp32 product of the two static scales, and the global scale of the
output (the fp32 reciprocal of the consumer's static scale).  The outputs are buffers of the call's own and stay in the
kernel's layout (``weights.TiledBlockScaledTensor``): a consumer on a fallback row gathers when it reads them.  The
kernel accumulates in fp32 over the whole input width and rounds three times, so the adapter names its twin: the
reference pair of the block evaluated in fp64 with the entry's error bound.

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
before the load, because a faulty adapter must not read as an unsupported GPU.  The fused gate / up entry is built from
the row count alone, and its row count is held in front of the build (the entry states a launch for it, or the table
is wrong): there the adapter has built nothing a decline could be about, and one of that entry's requirements -- the
names its kernel is written on -- has no public probe this adapter could ask.  Every typed decline of that build is
therefore the entry's answer about the installation, and the ``SKIP``.

**The memory sentence comes first.**  The scratch operands of a first use are as large as the run's own (rows + 1 slots of
recurrent state for the fused step), so the first use is not made where a block is asked whether it serves
(``check_support`` allocates nothing): a plan makes it as a step of its own (``Plan.first_use``), after the memory
sentence has said that the run fits and before anything is loaded.  A run that does not fit is a ``SKIP`` with its
estimate, and nothing was allocated for it.
"""

from __future__ import annotations

import re

from ..census import Facts, Kernel
from ..state import RecurrentSlots, TokenSlots
from ..weights import BLOCK, TILE_COLUMNS, TILE_ROWS, BlockScaledTensor, InterleavedBlockScaledTensor, ScaledTensor, TiledBlockScaledTensor, tile_scales
from . import BlockBase, GdnParams, split_qkvz

#: the files of the package this adapter runs, relative to the ``cudnn`` package: the source hash of ``check_environment``
SOURCE_FILES = (
    "__init__.py",
    "linear_attention/frost/gdn_decode_fused.py",
    "linear_attention/frost/kernel/gdn_decode_fused_config.py",
    "linear_attention/frost/kernel/gdn_decode_fused_f16.py",
    "linear_attention/frost/kernel/gdn_geometry.py",
    "linear_attention/frost/gdn_verify_fused.py",
    "linear_attention/frost/kernel/gdn_verify_fused_config.py",
    "linear_attention/frost/kernel/gdn_verify_fused_f16.py",
    "frost/launch_facts.py",
    "attention_gate/frost/gate_quant_decode.py",
    "attention_gate/frost/kernel/gate_quant_decode_config.py",
    "attention_gate/frost/kernel/gate_quant_decode_f16.py",
    "norm/frost/add_norm_quant_decode.py",
    "norm/frost/kernel/add_norm_quant_decode_config.py",
    "norm/frost/kernel/add_norm_quant_decode_f16.py",
    "gemm/frost/projection_decode/fp8_projection_decode.py",
    "gemm/frost/projection_decode/kernel/fp8_projection_decode_config.py",
    "gemm/frost/projection_decode/kernel/fp8_projection_decode_e4m3.py",
    "gemm/frost/swiglu_decode/swiglu_fp4_decode.py",
    "gemm/frost/swiglu_decode/kernel/swiglu_fp4_decode_config.py",
    "gemm/frost/swiglu_decode/kernel/swiglu_fp4_decode_nvfp4.py",
)
#: a field of the entry's form a provider row may name -> the constructor argument of ``cudnn.GdnDecodeFused`` that sets it
FORM_ARGUMENTS = {"PDL": "pdl", "PDL_RELEASE": "pdl_release"}
#: the field of the form of ``cudnn.GdnVerifyFused`` that is the step's shape and no constructor argument: the tokens per
#: sequence.  A row of that entry names it (the entry compiles one kernel per count, and the kernel's name does not say
#: which); the adapter holds every call's slot table to it
VERIFY_TOKENS = "TOKENS"
#: the fields of the form of ``cudnn.AttentionGateQuantDecode`` a provider row may name: ``THREADS`` goes into the entry's
#: config (``cfg=Cfg(THREADS=...)``), ``PDL_RELEASE`` is its launch knob ``pdl_release=``
GATE_FORM_FIELDS = ("THREADS", "PDL_RELEASE")
#: a field of the form of ``cudnn.AddNormQuantDecode`` a provider row may name -> the constructor argument that sets it; the
#: entry has no default for the first three (its forms are peers), so every call site of a row names them
NORM_FORM_ARGUMENTS = {"ADD": "add", "BF16_OUT": "bf16", "QUANT": "quant", "STORE_RESIDUAL": "store_residual", "PDL": "pdl", "PDL_RELEASE": "pdl_release"}
NORM_FORM_REQUIRED = ("ADD", "BF16_OUT", "QUANT")
#: the quantisation of a consumer's operand (``providers.Quant.fmt``) -> the entry's name for the codes that serve it
NORM_QUANT = {"fp8": "e4m3_static", "nvfp4": "nvfp4"}
#: the fields of the form of ``cudnn.Fp8ProjectionDecode`` a provider row may name, by how the adapter passes them: the
#: constructor arguments (``SPLIT_K`` is a numerics attribute of the entry, the other two its launch knobs), the fields of
#: its performance form (``form=ProjectionForm(...)``, passed as a whole: the entry then never takes its own pick) and the
#: weight's shape (``geometry=ProjectionGeometry(...)``), which a row names per call site.  A row names at least the split,
#: the weight tile, the ring depth and the shape: the entry is called on an explicit kernel, never on a default
PROJECTION_ARGUMENTS = {"SPLIT_K": "split_k", "PDL": "pdl", "PDL_RELEASE": "pdl_release"}
PROJECTION_FORM_FIELDS = ("MMA_M", "STAGES", "TMEM_COLS", "TOKEN_TILE", "PREWAIT_WEIGHTS", "HOLD_TMEM")
PROJECTION_GEOMETRY = ("OUT_FEATURES", "IN_FEATURES")
PROJECTION_REQUIRED = ("SPLIT_K", "MMA_M", "STAGES", *PROJECTION_GEOMETRY)
#: the fields of the form of ``cudnn.SwigluFp4Decode`` a provider row names, by how the adapter passes them: the fields of
#: its config (``cfg=Cfg(...)``, passed as a whole: the entry then never takes a default of its own) and its launch knob
#: (``pdl=``).  A row names every one of them: the token tile of its row count, the weight stages ahead of the wait, the
#: L2 policy of the weight stream, the cap of the persistent grid, the launch attribute
SWIGLU_CONFIG_FIELDS = ("TOKEN_TILE", "PREWAIT_STAGES", "L2_EVICT_FIRST", "MAX_CLUSTERS")
SWIGLU_ARGUMENTS = {"PDL": "pdl"}
SWIGLU_REQUIRED = (*SWIGLU_CONFIG_FIELDS, *SWIGLU_ARGUMENTS)
#: the entry's three launch facts: what the kernel check reads
LAUNCH_FACTS = ("kernel_symbols_for", "form_for", "launch_shape_for")
#: the first cuDNN Frontend version that carries the entries of this module, and the CuTe DSL floor their pages state
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
#: the (sequences, tokens) step shapes of the fused verify step that were launched in this process
_VERIFY_LAUNCHED: set = set()


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

    def _build(self, device, site: str = ""):  # ``site``: the call site, for an entry that has one form per site
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
            entry.launch_shape_for(*self._step_of(rows))
        except (ValueError, NotImplementedError) as error:
            cc = self.facts.cc
            raise ValueError(
                f"the provider table is wrong: the row of {self.row.key} claims this run ({rows} row{'s' if rows != 1 else ''} on compute "
                f"capability {cc[0]}.{cc[1]}), and {self.entry} declines it: {error}"
            ) from None

    def _step_of(self, rows: int) -> tuple:
        """What the entry's launch facts are asked with for a run of ``rows`` rows: the row count.  An adapter whose entry
        states its launch per step shape says which shape that is."""
        return (rows,)

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

    def _sites(self) -> tuple:
        """The call sites of the row that the model's wiring has (every site of the row for a model that names none): the
        forms a run builds.  A row without sites is called in its one form."""
        named = getattr(self.model, "sites", None)
        called = set(named()) if callable(named) else None
        return tuple(site for site, _ in self.row.sites if called is None or site in called) if self.row.sites else ("",)

    def _stated(self, entry, rows: int) -> dict:
        """The form an entry instance states for ``rows``, as the row's fields are looked up in it: the entry's own
        ``form_for``.  An adapter whose row names fields the entry states one level down says where."""
        return entry.form_for(*self._step_of(rows))

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
                stated = self._stated(entry, rows)
            except (ValueError, NotImplementedError):  # no form stated for rows it has just served: the log shows none
                stated = {}
            form = tuple((name, stated[name]) for name, _ in self._served_form() if name in stated)
            self._answer = (entry, rows, (f"{self.entry} #{place}" if place else self.entry, form))
        return self._answer[2]

    def _served_form(self) -> tuple:
        """The fields the call log reads from the instance that served the last call: those of the row's form.  An adapter
        whose row names call sites answers with the form of the last call's site."""
        return tuple(self.row.form)

    def launch_facts(self, rows: int, site: str = "") -> Facts:
        """The entry's own answers for ``rows``: the symbols of the kernels it has compiled for that row count, its form as plain
        data and its launch shape.  The entry's typed declines pass through: a row that claims a row count the entry does not
        serve is a wrong table."""
        entry = self._build(self.options.get("device", "cuda"))
        step = self._step_of(rows)
        return Facts(tuple(entry.kernel_symbols_for(*step)), self._stated(entry, rows), tuple(entry.launch_shape_for(*step)))


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


class GdnVerifyFusedBlock(EntryBlock):
    """``cudnn.GdnVerifyFused``: the fused linear-attention step for ``T`` tokens per sequence in one launch per layer -- the
    conv, the recurrence and the gated norm of every token in turn, the state after token ``t`` written to the slot of
    (sequence, ``t``), the conv history shifted by the step's rows; in the form the row names, ``TOKENS`` included."""

    name = "GdnVerifyFused"
    what = "the fused linear-attention verify step"
    kernels = (Kernel("gdn_verify_fused", "cuDNN FE", "exact", launches=1, shape="entry"),)  # one launch per layer and step
    per_row = (0, 1, 2, 3)
    state_args = (5,)
    batch_note = (
        "cudnn.GdnVerifyFused states replay-identical bytes and no batch invariance between one sequence and several: one sequence takes "
        "another association of the gated norm's sum than two and more, so the last bits of its output differ by design "
        '(docs/operations/determinism.md: "Replay only")'
    )

    def _arguments(self) -> dict:
        """The constructor arguments the row's form stands for.  A field this adapter cannot pass, or a form that does not
        name the tokens per sequence, is a wrong table."""
        unknown = [name for name, _ in self.row.form if name not in FORM_ARGUMENTS and name != VERIFY_TOKENS]
        if unknown or VERIFY_TOKENS not in dict(self.row.form):
            what = f"names {', '.join(unknown)}" if unknown else f"does not name {VERIFY_TOKENS}"
            raise ValueError(
                f"provider row {self.row.key}: its form {what}; this adapter passes {', '.join(FORM_ARGUMENTS)} to cudnn.GdnVerifyFused "
                f"and holds every step to the row's {VERIFY_TOKENS}, the tokens per sequence"
            )
        return {FORM_ARGUMENTS[name]: value for name, value in self.row.form if name in FORM_ARGUMENTS}

    def _tokens(self) -> int:
        return int(dict(self.row.form)[VERIFY_TOKENS])

    def _step_of(self, rows: int) -> tuple:
        """The step shape the entry's facts take: ``(sequences, tokens)`` of a run of ``rows`` rows at the row's tokens."""
        return (rows // self._tokens(), self._tokens())

    def check_support(self, sequences: int, tokens: int) -> None:
        self._check_package()
        self._arguments()
        if tokens != self._tokens():  # the row's tokens range selected a run its form does not name
            raise ValueError(
                f"the provider table is wrong: the row of {self.row.key} says {VERIFY_TOKENS} {self._tokens()} and was selected for a run of {tokens} tokens per sequence"
            )
        if self.facts.total_bytes and "device" in self.options:
            # on a real device: the entry's own typed decline (a part that holds no form) is a SKIP; once it is built, a step
            # shape it does not serve is a wrong table.  Its first use follows the memory sentence (``first_use``)
            self._check_claim(self._build(self.options["device"]), sequences * tokens)

    def reserve_bytes(self) -> int:
        """Device memory the kernel's first launch reserves outside PyTorch's allocator: the family's allowance for the GPU
        class, until a fused step was launched in this process."""
        if _FIRST_LAUNCH or _VERIFY_LAUNCHED:
            return 0
        reserve = getattr(self.model, "first_launch_reserve", {})
        return int(reserve.get(f"cc{self.facts.cc[0]}.x", 0))

    def _build(self, device):
        import torch

        g = self.model.geometry
        device = torch.device(device)
        key = ("gdn_verify_fused", str(device), g.la_key_heads, g.la_value_heads, g.hidden, self.model.norm_eps, tuple(self.row.form))
        if key not in _ENTRIES:
            cudnn = self._cudnn()
            from cudnn.linear_attention.frost import GdnGeometry

            geometry = GdnGeometry(g.la_key_heads, g.la_value_heads, g.hidden)
            _ENTRIES[key] = cudnn.GdnVerifyFused(device, eps=self.model.norm_eps, geometry=geometry, **self._arguments())
        return _ENTRIES[key]

    def _scratch(self, sequences: int, tokens: int):
        import torch

        g = self.model.geometry
        device = torch.device(self.options.get("device", "cuda"))
        rows, D, NV = sequences * tokens, g.la_head_dim, g.la_value_heads
        qkv = (2 * g.la_key_heads + NV) * D
        bf16 = dict(dtype=torch.bfloat16, device=device)
        x, z = split_qkvz(torch.zeros((rows, qkv + NV * D), **bf16), qkv, NV, D)
        slots = RecurrentSlots(
            torch.zeros((rows + 1, NV, D, D), dtype=torch.float32, device=device),
            torch.zeros((sequences + 1, g.la_conv_width - 2 + tokens, qkv), **bf16),
            tokens,
        )
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
        table = TokenSlots.build(sequences, tokens, device)
        return self._build(device), self._operands(x, z, torch.zeros((rows, g.hidden), **bf16), table, params, slots)

    def compile(self, sequences: int, tokens: int) -> None:
        entry, operands = self._scratch(sequences, tokens)
        try:
            entry.compile(*operands)  # built while the run was planned (``first_use``): the entry finds its kernel
            entry(*operands)  # the first launch of the compiled kernel, before the first step
            _VERIFY_LAUNCHED.add((sequences, tokens))
        except NotImplementedError as error:  # after the plan nothing skips: the sentence says whose the decline is
            raise RuntimeError(self._declined(error, "its first launch")) from error

    @staticmethod
    def _operands(x, z, hidden, table, p, slots):
        """The entry's positional operands: ``(qkv, z, hidden, ba_weight, A_log, dt_bias, ssm_state, state_indices, cu_seqlens,
        num_accepted_tokens, norm_weight, out_scale, conv_state, conv_weight)``."""
        return (
            x,
            z,
            hidden,
            p.ba_weight,
            p.a_log,
            p.dt_bias,
            slots.state,
            table.indices,
            table.cu_seqlens,
            table.accepted,
            p.norm_weight,
            p.out_scale,
            slots.conv,
            p.conv_weight,
        )

    def __call__(self, x, z, hidden, table, params, slots):
        import torch

        if not isinstance(table, TokenSlots):
            raise ValueError(
                f"cudnn.GdnVerifyFused takes the slot table of a step of several tokens per sequence; this block is handed {type(table).__name__}: "
                "the session's state was built for one token per sequence"
            )
        if table.tokens != self._tokens():
            raise ValueError(
                f"provider row {self.row.key}: its form says {VERIFY_TOKENS} {self._tokens()}; the step hands it a slot table of {table.tokens} tokens per sequence"
            )
        entry, rows = self._build(x.device), int(x.shape[0])
        self._served = (entry, rows)  # what the call log of a traced step reads back (``serving``)
        out = torch.empty((rows, params.value_heads, params.head_dim), dtype=torch.float8_e4m3fn, device=x.device)  # this call's own
        try:
            entry(*self._operands(x, z, hidden, table, params, slots), out)
        except NotImplementedError as error:
            raise RuntimeError(f"cudnn.GdnVerifyFused declined the operands of this step: {error}") from error
        return ScaledTensor(out.view(torch.uint8).view(rows, -1), params.out_scale_value)


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


class AddNormQuantDecodeBlock(EntryBlock):
    """``cudnn.AddNormQuantDecode``: per row ``y = bf16(v * rsqrt(mean(v^2) + eps) * (1 + weight))`` with ``v = branch +
    residual`` (the residual updated in place) or ``v = branch`` (the residual stream started from it), and the codes of
    quantising ``y``; one launch per call, in the form the row names for the call's site."""

    name = "AddNormQuantDecode"
    what = "the add + RMSNorm + quantise decode step"
    kernels = (Kernel("add_norm_quant", "cuDNN FE", "exact", launches=1, shape="entry"),)  # one launch per call of the block
    per_row = (0, 1)
    state_args = (1,)  # the residual stream: updated in place where the form adds to it
    _twin = None
    _scales = None  # (value, reciprocal, device) -> the one-element fp32 tensor the kernel reads: built once, never inside a later step
    _site = ""  # of the last call

    def _arguments(self, site: str) -> dict:
        """The constructor arguments the form of ``site`` stands for.  A field this adapter cannot pass, or a form without
        one the entry has no default for, is a wrong table."""
        form = self.row.site_form(site)
        unknown = [name for name, _ in form if name not in NORM_FORM_ARGUMENTS]
        missing = [name for name in NORM_FORM_REQUIRED if name not in dict(form)]
        if unknown or missing:
            what = f"names {', '.join(unknown)}" if unknown else f"does not name {', '.join(missing)}"
            raise ValueError(
                f"provider row {self.row.key}: its form at {site or 'its one call site'} {what}; this adapter passes "
                f"{', '.join(NORM_FORM_ARGUMENTS)} to cudnn.AddNormQuantDecode, which has no default for {', '.join(NORM_FORM_REQUIRED)}"
            )
        return {NORM_FORM_ARGUMENTS[name]: value for name, value in form}

    def check_support(self, sequences: int, tokens: int) -> None:
        self._check_package()
        for site in [site for site, _ in self.row.sites] or [""]:
            self._arguments(site)
        if self.facts.total_bytes and "device" in self.options:
            for site in self._sites():  # the first use, where the entry gates the toolchain, follows the memory sentence
                self._check_claim(self._build(self.options["device"], site), sequences * tokens)

    def _unmet(self) -> bool:
        """Beyond the part's target: on compute capability 10.7 the entry's page asks for a CuTe DSL whose compiler is CUDA 13.4
        or later (the CUDA 12 flavour lists the target and cannot assemble it)."""
        if super()._unmet():
            return True
        if tuple(self.facts.cc) != (10, 7):
            return False
        from cutlass.base_dsl.version_info import CUDA_VERSION  # the compiler the installed CuTe DSL bound

        return (int(CUDA_VERSION.major), int(CUDA_VERSION.minor)) < (13, 4)

    def _build(self, device, site: str = ""):
        import torch

        device = torch.device(device)
        hidden, eps = self.model.geometry.hidden, self.model.norm_eps
        key = ("add_norm_quant_decode", str(device), hidden, eps, tuple(self.row.site_form(site)))
        if key not in _ENTRIES:
            arguments = self._arguments(site)
            try:
                _ENTRIES[key] = self._cudnn().AddNormQuantDecode(device, hidden, eps, **arguments)
            except (NotImplementedError, TypeError, ValueError) as error:  # the entry declines the width or the form the table hands it
                where = f" at {site}" if site else ""
                raise ValueError(
                    f"the provider table is wrong: {self.entry} declines the geometry or the form of the row of {self.row.key}{where}: {error}"
                ) from None
        return _ENTRIES[key]

    def _scale(self, value: float, device, reciprocal: bool = False):
        """The one-element fp32 tensor the kernel reads its scale from: ``value``, or -- for 4-bit codes, whose kernel takes
        the global scale a consumer divides by -- its fp32 reciprocal, correctly rounded.  Formed on the host."""
        import torch

        if self._scales is None:
            self._scales = {}
        key = (float(value), reciprocal, str(device))
        if key not in self._scales:
            scale = torch.tensor([float(value)], dtype=torch.float32)
            self._scales[key] = (torch.ones(1, dtype=torch.float32) / scale if reciprocal else scale).to(device)
        return self._scales[key]

    def _outputs(self, form: dict, rows: int, device) -> dict:
        """This call's own buffers for what the form writes: never the entry's, which are one set per row count and are
        overwritten by its next call.  The block scales of 4-bit codes take whole tiles of 128 rows; the kernel writes
        the bytes of the live rows, and nothing else of that buffer is read."""
        import torch

        from ..weights import BLOCK, TILE_ROWS

        hidden, outputs = self.model.geometry.hidden, {}
        if form.get("BF16_OUT"):
            outputs["out"] = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
        if form.get("QUANT") == "e4m3_static":
            outputs["codes"] = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device)
        elif form.get("QUANT") == "nvfp4":
            outputs["codes"] = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=device)
            outputs["scales"] = torch.empty((-(-rows // TILE_ROWS) * TILE_ROWS, hidden // BLOCK), dtype=torch.float8_e4m3fn, device=device)
        return outputs

    def twin(self, reference):
        """The reference block this adapter is compared with: the norm at the kernel's rounding points, writing what the
        form of each call site writes."""
        from . import torch_ref

        if self._twin is None:
            self._twin = torch_ref.AddNormQuantizeBracket(reference.row, self.model, self.facts, self.options)
            self._twin.forms = {site: dict(self.row.site_form(site)) for site in [site for site, _ in self.row.sites] or [""]}
        return self._twin

    def first_use(self, sequences: int, tokens: int) -> None:
        """While the run is planned, on a real CUDA device: build the kernel of every form the run calls (the entry's build
        takes no operand).  Declines as ``EntryBlock.first_use``."""
        import torch

        if not self.facts.total_bytes or "device" not in self.options:  # a plan on stated device facts builds nothing
            return
        if torch.device(self.options["device"]).type not in self.first_use_devices:
            return
        for site in self._sites():
            try:
                self._build(self.options["device"], site).compile()
            except NotImplementedError as error:
                if self._unmet():
                    raise
                raise ValueError(self._declined(error, "its first use")) from None

    def compile(self, sequences: int, tokens: int) -> None:
        import torch

        device, rows, hidden = torch.device(self.options.get("device", "cuda")), sequences * tokens, self.model.geometry.hidden
        for site in self._sites():
            entry, form = self._build(device, site), dict(self.row.site_form(site))
            x = torch.zeros((rows, hidden), dtype=torch.bfloat16, device=device)
            residual = torch.zeros_like(x) if form.get("ADD") or form.get("STORE_RESIDUAL") else None
            scale = self._scale(1.0, device) if form.get("QUANT") else None
            try:
                entry.compile()  # built while the run was planned (``first_use``): the entry finds its kernel
                entry(x, residual, torch.zeros(hidden, dtype=torch.bfloat16, device=device), scale, **self._outputs(form, rows, device))
            except NotImplementedError as error:  # after the plan nothing skips: the sentence says whose the decline is
                raise RuntimeError(self._declined(error, "its first launch")) from error

    def launch_facts(self, rows: int, site: str = "") -> Facts:
        """The answers of the entry instance of ``site`` for ``rows``: the symbol of the kernel of that form (one per form,
        whatever the row count), the form and the launch shape."""
        entry = self._build(self.options.get("device", "cuda"), site)
        return Facts(tuple(entry.kernel_symbols_for(rows)), entry.form_for(rows), tuple(entry.launch_shape_for(rows)))

    def _served_form(self) -> tuple:
        return tuple(self.row.site_form(self._site))

    def compared(self, out):
        """What the tie-out compares of a call: the codes the kernel wrote (with their block scales, row by row), the normed
        rows where the form writes them, and the residual stream.  A quantised operand the consumer forms itself
        (``Unquantized``) is not the kernel's output and is left out."""
        from ..weights import Unquantized

        q, normed, residual = out
        return (None if isinstance(q, Unquantized) else q, normed, residual)

    def __call__(self, branch, residual, weight, quant, site=""):
        import torch

        from ..weights import TiledBlockScaledTensor, Unquantized

        entry, form = self._build(branch.device, site), dict(self.row.site_form(site))
        rows, device, where = int(branch.shape[0]), branch.device, f" at {site}" if site else ""
        self._served, self._site = (entry, rows), site  # what the call log of a traced step reads back (``serving``)
        if quant is not None and quant.fmt not in NORM_QUANT:
            raise ValueError(f"cudnn.AddNormQuantDecode writes static 8-bit or 4-bit codes; this block is asked for {quant.fmt!r}")
        if form["QUANT"] is not None and (quant is None or NORM_QUANT[quant.fmt] != form["QUANT"]):
            asked = "no quantised operand" if quant is None else f"a {quant.fmt} operand"
            raise ValueError(f"provider row {self.row.key}: its form{where} writes {form['QUANT']} codes; the step asks for {asked} there")
        starts = residual is None
        if bool(form["ADD"]) == starts or (starts and not form.get("STORE_RESIDUAL")):
            has = "no residual stream yet" if starts else "a residual stream"
            raise ValueError(
                f"provider row {self.row.key}: its form{where} is ADD {form['ADD']}, STORE_RESIDUAL {form.get('STORE_RESIDUAL', False)}; the step has {has} there"
            )
        if starts:
            residual = torch.empty_like(branch)  # the stream starts here: the kernel writes the branch into it
        scale = None
        if form["QUANT"] is not None:
            scale = self._scale(quant.scale, device, reciprocal=form["QUANT"] == "nvfp4")
        try:
            out, codes, scales = entry(branch, residual, weight, scale, **self._outputs(form, rows, device))
        except NotImplementedError as error:
            raise RuntimeError(f"cudnn.AddNormQuantDecode declined the operands of this step: {error}") from error
        if form["QUANT"] == "e4m3_static":
            q = ScaledTensor(codes.view(torch.uint8), float(quant.scale))
        elif form["QUANT"] == "nvfp4":
            q = TiledBlockScaledTensor(codes, scales.view(torch.uint8), float(quant.scale))
        else:  # the form ends at the bf16 rows: a consumer that reads a quantised operand quantises them itself
            q = None if quant is None else Unquantized(out, quant.fmt, float(quant.scale))
        return q, out, residual


class Fp8ProjectionDecodeBlock(EntryBlock):
    """``cudnn.Fp8ProjectionDecode``: ``out[t, n] = bf16(scale * sum_k activation[t, k] * weight[n, k])`` on the 8-bit codes
    of both operands, fp32 accumulation in a fixed order; one launch per call, from the entry instance of the call's site
    (one per weight shape), in the form the row names."""

    name = "Fp8ProjectionDecode"
    what = "the FP8 projection decode step at these weight shapes"
    kernels = (Kernel("fp8_projection", "cuDNN FE", "exact", launches=1, shape="entry"),)  # one launch per call of the block
    stored_weights = True  # the weight is multiplied on its stored 8-bit codes: the loader keeps them for this block
    _scales = None  # (activation scale, weight scale, device) -> the one-element fp32 tensor the kernel reads: built once per pair
    _site = ""  # of the last call

    def _arguments(self, site: str) -> dict:
        """The constructor arguments the form of ``site`` stands for.  A field this adapter cannot pass, or a form that
        leaves the split, the weight tile, the ring depth or the weight's shape to the entry, is a wrong table."""
        form = self.row.site_form(site)
        known = (*PROJECTION_ARGUMENTS, *PROJECTION_FORM_FIELDS, *PROJECTION_GEOMETRY)
        unknown = [name for name, _ in form if name not in known]
        missing = [name for name in PROJECTION_REQUIRED if name not in dict(form)]
        if unknown or missing:
            what = f"names {', '.join(unknown)}" if unknown else f"does not name {', '.join(missing)}"
            raise ValueError(
                f"provider row {self.row.key}: its form at {site or 'its one call site'} {what}; this adapter passes {', '.join(known)} to "
                f"cudnn.Fp8ProjectionDecode, and never leaves {', '.join(PROJECTION_REQUIRED)} to the entry's own pick"
            )
        from cudnn.gemm import ProjectionForm, ProjectionGeometry

        fields = dict(form)
        arguments = {PROJECTION_ARGUMENTS[name]: value for name, value in form if name in PROJECTION_ARGUMENTS}
        arguments["geometry"] = ProjectionGeometry(*(fields[name] for name in PROJECTION_GEOMETRY))
        arguments["form"] = ProjectionForm(**{name: fields[name] for name in PROJECTION_FORM_FIELDS if name in fields})
        return arguments

    def check_support(self, sequences: int, tokens: int) -> None:
        self._check_package()
        for site in [site for site, _ in self.row.sites] or [""]:
            self._arguments(site)
        rule = self.options.get("merge_rule", "largest_scale")
        if rule != "largest_scale":  # the entry reads ONE scale element per call: a weight merged with a scale per row is not its operand
            raise NotImplementedError(
                f"cudnn.Fp8ProjectionDecode takes one scale per weight; the merge rule {rule!r} keeps one per weight row: load with the merge "
                f'rule "largest_scale", or select another declared row of the block with --provider'
            )
        if self.facts.total_bytes and "device" in self.options:
            for site in self._sites():  # the first use, where the entry gates the toolchain, follows the memory sentence
                self._check_claim(self._build(self.options["device"], site), sequences * tokens)

    def _unmet(self) -> bool:
        """Beyond the part's target: the entry's page asks for a CuTe DSL that has the layer its kernel is written on."""
        if super()._unmet():
            return True
        import importlib.util

        try:
            return importlib.util.find_spec("cutlass.cute.experimental") is None
        except (ImportError, ValueError):
            return True

    def _build(self, device, site: str = ""):
        import torch

        device = torch.device(device)
        key = ("fp8_projection_decode", str(device), tuple(self.row.site_form(site)))
        if key not in _ENTRIES:
            arguments = self._arguments(site)
            try:
                _ENTRIES[key] = self._cudnn().Fp8ProjectionDecode(device, **arguments)
            except (NotImplementedError, TypeError, ValueError) as error:  # the entry declines the shape, the split or the form the table hands it
                where = f" at {site}" if site else ""
                raise ValueError(
                    f"the provider table is wrong: {self.entry} declines the weight shape or the form of the row of {self.row.key}{where}: {error}"
                ) from None
        return _ENTRIES[key]

    def _stated(self, entry, rows: int) -> dict:
        """The entry's form for ``rows`` with the two fields of its geometry beside the others: a row names the weight's
        shape per call site (``OUT_FEATURES``, ``IN_FEATURES``), and the entry states them one level down (``GEOM``)."""
        form = entry.form_for(rows)
        return {**form, **form.get("GEOM", {})}

    def _served_form(self) -> tuple:
        return tuple(self.row.site_form(self._site))

    def launch_facts(self, rows: int, site: str = "") -> Facts:
        """The answers of the entry instance of ``site`` for ``rows``: the symbol of the kernel it compiled for that row
        count, its form and its launch shape (one thread block per weight tile: the grid follows the weight's shape)."""
        entry = self._build(self.options.get("device", "cuda"), site)
        return Facts(tuple(entry.kernel_symbols_for(rows)), self._stated(entry, rows), tuple(entry.launch_shape_for(rows)))

    def _scale(self, activation: float, weight: float, device):
        """The one fp32 element the kernel reads its scale from: the fp32 product of the activation's static scale and the
        weight's scale, formed on the host once per pair of values and never inside a later step."""
        import torch

        if self._scales is None:
            self._scales = {}
        key = (float(activation), float(weight), str(device))
        if key not in self._scales:
            product = torch.tensor([float(activation)], dtype=torch.float32) * torch.tensor([float(weight)], dtype=torch.float32)
            self._scales[key] = product.to(device)
        return self._scales[key]

    def first_use(self, sequences: int, tokens: int) -> None:
        """While the run is planned, on a real CUDA device: build the kernel of the run's row count at every weight shape
        the run calls (the entry's build takes no operand).  Declines as ``EntryBlock.first_use``."""
        import torch

        if not self.facts.total_bytes or "device" not in self.options:  # a plan on stated device facts builds nothing
            return
        if torch.device(self.options["device"]).type not in self.first_use_devices:
            return
        for site in self._sites():
            try:
                self._build(self.options["device"], site).compile(sequences * tokens)
            except NotImplementedError as error:
                if self._unmet():
                    raise
                raise ValueError(self._declined(error, "its first use")) from None

    def compile(self, sequences: int, tokens: int) -> None:
        import torch

        device, rows = torch.device(self.options.get("device", "cuda")), sequences * tokens
        for site in self._sites():
            entry, fields = self._build(device, site), dict(self.row.site_form(site))
            n, k = (fields[name] for name in PROJECTION_GEOMETRY)
            weight = torch.zeros((n, k), dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
            activation = torch.zeros((rows, k), dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
            out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
            try:
                entry.compile(rows)  # built while the run was planned (``first_use``): the entry finds its kernel
                entry(weight, activation, self._scale(1.0, 1.0, device), out)  # the first launch of the compiled kernel, before the first step
            except NotImplementedError as error:  # after the plan nothing skips: the sentence says whose the decline is
                raise RuntimeError(self._declined(error, "its first launch")) from error

    def __call__(self, qx, weight, site=""):
        import torch

        stored = getattr(weight, "stored", None)
        if not isinstance(stored, ScaledTensor) or isinstance(stored.scale, torch.Tensor):
            has = "one scale per weight row" if isinstance(stored, ScaledTensor) else "no stored 8-bit codes"
            raise ValueError(
                f"cudnn.Fp8ProjectionDecode multiplies on the stored 8-bit weight with its one scale; the weight of this call has {has}: "
                "the weights of this session were loaded for another plan (plan every session of a run before the first load: Session.plan_table)"
            )
        if not isinstance(qx, ScaledTensor):
            raise ValueError(f"cudnn.Fp8ProjectionDecode multiplies static 8-bit codes; this block is handed {type(qx).__name__}")
        device, rows = qx.codes.device, int(qx.codes.shape[0])
        entry = self._build(device, site)
        self._served, self._site = (entry, rows), site  # what the call log of a traced step reads back (``serving``)
        out = torch.empty((rows, stored.codes.shape[0]), dtype=torch.bfloat16, device=device)  # this call's own
        scale = self._scale(qx.scale, stored.scale, device)
        try:
            entry(stored.codes.view(torch.float8_e4m3fn), qx.codes.view(torch.float8_e4m3fn), scale, out)
        except NotImplementedError as error:
            raise RuntimeError(f"cudnn.Fp8ProjectionDecode declined the operands of this step: {error}") from error
        return out


class SwigluFp4DecodeBlock(EntryBlock):
    """``cudnn.SwigluFp4Decode``: ``gate`` and ``up`` = ``bf16(alpha * sum_k activation[t, k] * weight[row, k])`` on the packed
    4-bit codes of both operands, ``v = bf16(silu(gate) * up)``, and per block of 16 channels the block scale and the
    4-bit codes of ``v``; one launch per call, in the form the row names, on the weight in the kernel's own layout."""

    name = "SwigluFp4Decode"
    what = "the fused 4-bit gate / up decode step"
    kernels = (Kernel("swiglu_fp4", "cuDNN FE", "exact", launches=1, shape="entry"),)  # one launch per call of the block
    stored_weights = True  # the weight is multiplied on its stored 4-bit codes: the loader keeps them for this block, in ``stored_layout``
    _twin = None
    _scalars = None  # (kind, values, device) -> the one-element fp32 tensor the kernel reads: built once per value, never inside a later step

    def _geometry(self):
        from cudnn.gemm import SwigluGeometry

        return SwigluGeometry(self.model.geometry.hidden, self.model.geometry.mlp)

    def _arguments(self) -> dict:
        """The constructor arguments the row's form stands for: the config as a whole and the launch knob.  A field this
        adapter cannot pass, or a form that leaves one of them to the entry's default, is a wrong table."""
        form = dict(self.row.form)
        unknown = [name for name in form if name not in SWIGLU_REQUIRED]
        missing = [name for name in SWIGLU_REQUIRED if name not in form]
        if unknown or missing:
            what = f"names {', '.join(unknown)}" if unknown else f"does not name {', '.join(missing)}"
            raise ValueError(
                f"provider row {self.row.key}: its form {what}; this adapter passes {', '.join(SWIGLU_REQUIRED)} to cudnn.SwigluFp4Decode, "
                "and never leaves one of them to the entry's own default"
            )
        from cudnn.gemm.frost.swiglu_decode import Cfg

        cfg = Cfg(GEOM=self._geometry(), **{name: form[name] for name in SWIGLU_CONFIG_FIELDS})
        return {"cfg": cfg, **{SWIGLU_ARGUMENTS[name]: form[name] for name in SWIGLU_ARGUMENTS}}

    def check_support(self, sequences: int, tokens: int) -> None:
        self._check_package()
        self._arguments()
        if self.facts.total_bytes and "device" in self.options:
            self._check_claim(self._build(self.options["device"]), sequences * tokens)  # the first use follows the memory sentence

    def _unmet(self) -> bool:
        """Beyond the part's target: the entry's page asks for a CuTe DSL whose compiler is CUDA 13.4 or later, on every
        class it serves (its bytes were qualified with that compiler).  A CuTe DSL that does not say which compiler it
        bound is not unmet by that: the entry declines nothing on a value it could not read, and neither does this."""
        if super()._unmet():
            return True
        try:
            from cutlass.base_dsl.version_info import CUDA_VERSION  # the compiler the installed CuTe DSL bound; the entry has imported the DSL by now

            compiler = (int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
        except Exception:  # noqa: BLE001 -- whatever keeps the value from being read: no statement about the compiler
            return False
        return compiler < (13, 4)

    def _build(self, device):
        import torch

        device = torch.device(device)
        g = self.model.geometry
        key = ("swiglu_fp4_decode", str(device), g.hidden, g.mlp, tuple(self.row.form))
        if key not in _ENTRIES:
            arguments = self._arguments()
            try:
                _ENTRIES[key] = self._cudnn().SwigluFp4Decode(device, self._geometry(), **arguments)
            except (NotImplementedError, TypeError, ValueError) as error:  # the entry declines the widths or the form the table hands it
                raise ValueError(f"the provider table is wrong: {self.entry} declines the geometry or the form of the row of {self.row.key}: {error}") from None
        return _ENTRIES[key]

    def stored_layout(self, merged):
        """The merged gate / up weight (every gate row, then every up row; block scales row by row) in the layout the
        kernel multiplies on, built once per weight at load by the entry's own helper: the rows in tiles of 64 gate + 64 up
        channels, the block scales of those rows in the 128 x 4 layout.  A permutation of bytes: nothing is requantised.
        The loader keeps what this returns as the stored form of the weight, and nothing else of it."""
        from cudnn.gemm.frost.swiglu_decode import interleave_gate_up

        if not isinstance(merged, BlockScaledTensor):
            raise ValueError(f"cudnn.SwigluFp4Decode multiplies on a merged 4-bit gate / up weight; the loader handed its layout {type(merged).__name__}")
        try:
            codes, scales = interleave_gate_up(merged.codes, tile_scales(merged.block_scales))
        except (TypeError, ValueError) as error:  # a weight whose shape the kernel's layout cannot hold
            raise ValueError(f"the provider table is wrong: {self.entry} cannot lay out the weight of the row of {self.row.key}: {error}") from None
        return InterleavedBlockScaledTensor(codes, scales.view(merged.block_scales.shape), float(merged.scale))

    def twin(self, reference):
        """The reference block this adapter is compared with: the block's reference pair evaluated in fp64 with the error
        bound of the entry's contract, not the pair's own bf16 product."""
        from . import torch_ref

        if self._twin is None:
            self._twin = torch_ref.GateUpFp4Bound(reference.row, self.model, self.facts, self.options)
        return self._twin

    def _scalar(self, device, product=None, reciprocal=None):
        """One fp32 element the kernel reads, formed on the host once per value: the fp32 product of two static scales
        (``product``: the pair), or the fp32 reciprocal of one, correctly rounded (``reciprocal``: the global scale of 4-bit
        codes whose consumer reads ``code * block scale * scale``)."""
        import torch

        from .torch_ref import fp32_product

        if self._scalars is None:
            self._scalars = {}
        key = ("product", *(float(v) for v in product), str(device)) if product is not None else ("reciprocal", float(reciprocal), str(device))
        if key not in self._scalars:
            if product is not None:
                value = fp32_product(*product)
            else:
                value = torch.ones(1, dtype=torch.float32) / torch.tensor([float(reciprocal)], dtype=torch.float32)
            self._scalars[key] = value.to(device)
        return self._scalars[key]

    def _outputs(self, rows: int, device):
        """This call's own buffers: the packed codes of the rows, and whole bands and tiles of the 128 x 4 layout for their
        block scales (the kernel writes the bytes of the live rows; nothing else of that buffer is read)."""
        import torch

        width = self.model.geometry.mlp
        out = torch.empty((rows, width // 2), dtype=torch.uint8, device=device)
        scales = torch.empty((-(-rows // TILE_ROWS) * TILE_ROWS, -(-(width // BLOCK) // TILE_COLUMNS) * TILE_COLUMNS), dtype=torch.uint8, device=device)
        return out, scales

    def first_use(self, sequences: int, tokens: int) -> None:
        """While the run is planned, on a real CUDA device: build the kernel of the run's row count.  The entry's build is
        handed that row count and nothing else, and the row count is held in front of it (``_check_claim``: an entry that
        states no launch for it is a wrong table).  Nothing of this adapter's is then left for the build to decline, so
        every typed decline of it is the entry's answer about the installation -- its compiler, the part's target, the
        names the kernel is written on -- and passes through as it is: a ``NotImplementedError``, the SKIP in the entry's
        own words.  (``EntryBlock.first_use`` turns a decline no requirement explains into an ERROR about the operands
        the adapter built; this adapter builds none here, and one of this entry's requirements has no public probe.)"""
        import torch

        if not self.facts.total_bytes or "device" not in self.options:  # a plan on stated device facts builds nothing
            return
        if torch.device(self.options["device"]).type not in self.first_use_devices:
            return
        entry, rows = self._build(self.options["device"]), sequences * tokens
        self._check_claim(entry, rows)
        entry.compile(rows)

    def compile(self, sequences: int, tokens: int) -> None:
        import torch

        device, rows, g = torch.device(self.options.get("device", "cuda")), sequences * tokens, self.model.geometry
        entry = self._build(device)
        weight = torch.zeros((2 * g.mlp, g.hidden // 2), dtype=torch.uint8, device=device)
        weight_scale = torch.zeros((2 * g.mlp, g.hidden // BLOCK), dtype=torch.uint8, device=device)
        activation = torch.zeros((rows, g.hidden // 2), dtype=torch.uint8, device=device)
        activation_scale = torch.zeros((-(-rows // TILE_ROWS) * TILE_ROWS, g.hidden // BLOCK), dtype=torch.uint8, device=device)
        one = self._scalar(device, product=(1.0, 1.0))
        try:
            entry.compile(rows)  # built while the run was planned (``first_use``): the entry finds its kernel
            entry(weight, weight_scale, activation, activation_scale, one, one, *self._outputs(rows, device))  # the first launch, before the first step
        except NotImplementedError as error:  # after the plan nothing skips: the sentence says whose the decline is
            raise RuntimeError(self._declined(error, "its first launch")) from error

    @staticmethod
    def _operand(qx):
        """The 4-bit operand as the kernel reads it: ``(packed codes, block scales in the 128 x 4 layout)``.  From a kernel
        on the stored formats it comes that way and is handed over as it is.  From a block that is not a kernel -- a
        reference chain, or one row of a batch in a check -- the block scales come row by row, and this block brings
        them into the layout when it reads them: its own launch, never the producer's."""
        if isinstance(qx, TiledBlockScaledTensor):
            return qx.codes, qx.tiled_scales
        if isinstance(qx, BlockScaledTensor):
            return qx.codes, tile_scales(qx.block_scales)
        raise ValueError(f"cudnn.SwigluFp4Decode multiplies packed 4-bit codes with their block scales; this block is handed {type(qx).__name__}")

    def __call__(self, qx, weight, quant):
        stored = getattr(weight, "stored", None)
        if not isinstance(stored, InterleavedBlockScaledTensor):
            has = "it in the checkpoint's row order" if isinstance(stored, BlockScaledTensor) else "no stored 4-bit codes"
            raise ValueError(
                f"cudnn.SwigluFp4Decode multiplies on the stored 4-bit gate / up weight in its own row order; the weight of this call has {has}: "
                "the weights of this session were loaded for another plan (plan every session of a run before the first load: Session.plan_table)"
            )
        if quant is None or quant.fmt != "nvfp4":
            asked = "no quantised operand" if quant is None else repr(quant.fmt)
            raise ValueError(f"cudnn.SwigluFp4Decode writes 4-bit codes with their block scales; this block is asked for {asked}")
        codes, scales = self._operand(qx)
        device, rows = codes.device, int(codes.shape[0])
        entry = self._build(device)
        self._served = (entry, rows)  # what the call log of a traced step reads back (``serving``)
        out, out_scales = self._outputs(rows, device)
        alpha, scale = self._scalar(device, product=(qx.scale, stored.scale)), self._scalar(device, reciprocal=quant.scale)
        try:
            entry(stored.codes, stored.tiled_scales, codes, scales, alpha, scale, out, out_scales)
        except NotImplementedError as error:
            raise RuntimeError(f"cudnn.SwigluFp4Decode declined the operands of this step: {error}") from error
        return TiledBlockScaledTensor(out, out_scales, float(quant.scale))
