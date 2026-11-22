# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library kernels of FlashInfer as providers of model blocks: the multiplies the target kernel set runs on a library, each on
the stored format of its weight.

``flashinfer.mm_fp4`` calls ``flashinfer.gemm.mm_fp4(backend="cute-dsl")``, the multiply of a 4-bit activation by a 4-bit
weight on the packed codes of both: one launch per call.  The library picks the kernel of a call itself, by the shape and
the GPU: one of two kernel classes of one module, in a grid of its own choice.  That pick is a function of the call as long
as nothing has tuned the operation, so the row says ``TUNED False`` and the adapter calls the library outside any tuning
context; which class and which launch shape the call then gets is data of the family's kernel census, held exactly by
the kernel check.  ``TUNED`` is a statement about this process: the library's tuner is tuning, or has tuned the operation
here.  A tuning store the caller attached to the library is not seen by it; a tactic replayed from one is held by the
kernel class and the launch shape alone, where the census has a section for the run.

The operand is the 4-bit output of the block before it: from a kernel on the stored formats it comes with its block
scales in the 128 x 4 layout this kernel reads and is handed over as it is; from a block that is not a kernel the scales
come row by row, and bringing them into the layout is this block's own launch.  The weight is the stored 4-bit weight
with its block scales in that layout (``weights.TiledBlockScaledWeight``), built once per weight at load
(``stored_layout``) and kept as the weight's one stored form; the loader holds it to the bytes it was made from.

``torch_ref.quantize+flashinfer.mm_fp4`` is the same multiply for a block whose operand reaches it as bf16 rows -- the
output head, whose operand the target quantises with a launch of its own.  The block quantises them with the reference
quantiser when it reads them (PyTorch's launches, declared as that family) and multiplies with the library.

``flashinfer.bmm_fp8`` calls ``flashinfer.gemm.bmm_fp8(backend="auto")``, the multiply of an 8-bit activation by an 8-bit
weight on the stored codes of both.  Which kernel serves it is the choice of the library among its own: its tuner's where
one has run, else its first candidate.  That choice is never a row and never gated; the kernel check holds the launch as
one of the kernels this call can launch, and prints its name beside the pick the census records.

**The package.**  FlashInfer is no dependency of cuDNN Frontend and none of the default stack.  It is imported inside the
functions of this module that run on a device, never at module import and never by a plan on stated device facts.  A plan
for a CUDA device reads the installed version from the distribution's metadata: a missing package, or a version that is not
validated (``VALIDATED``: the versions the launch records of the census were recorded with), is a ``NotImplementedError``
-- a ``SKIP`` before the load that names the version, the requirement file and the stack that runs without the package.

**First use is part of the plan.**  The library builds what a call needs when it is first made: its CuTe DSL kernels in the
process, its CUTLASS multiply as a module compiled with the installed CUDA compiler and ninja (minutes, once per machine).
The 4-bit kernel is built per range of row counts.  So an adapter makes the call on scratch operands while the run is
planned (``first_use``) -- at the run's row count and, for a run of several sequences, at the rows of one sequence, which
is what the batch-invariance check calls -- after the memory sentence has held and before anything is loaded, and a sample
announces it (``first_use_note``).  A refusal of the operands this adapter built is a ``ValueError`` -- an ``ERROR``,
because a faulty adapter must not read as an unsupported installation; whatever else keeps the LIBRARY from importing,
building or launching there is the installation's, and the ``SKIP``.  A fault of this adapter's own code is neither: it
is raised as it is.
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
from typing import Optional

from ..census import Kernel
from ..weights import BLOCK, BlockScaledTensor, Dequantized, ScaledTensor, TiledBlockScaledTensor, TiledBlockScaledWeight, Unquantized, tile_scales
from . import PYTORCH_CHAIN, BlockBase

#: the distribution these adapters import, and the versions of it this area has validated: those the launch records of the
#: family's kernel census were recorded with.  ``samples/models/constraints-native.txt`` pins the same; a host test holds
#: the three to each other.  The one pin that is not a version of that record is the library's kernel-binary package
#: below: it joined the list when its files were held to the files the record's runs read
DISTRIBUTION = "flashinfer-python"
VALIDATED = ("0.7.2.post1",)
REQUIREMENTS = "samples/models/requirements-flashinfer.txt"
CONSTRAINTS = "samples/models/constraints-native.txt"
#: the stack that runs the cuDNN Frontend rows of the target kernel set without this package: what a decline points to
WITHOUT = "--stack fe-entries runs the cuDNN Frontend rows of this stack without the package"
#: the name under which the library's tuner keeps the 4-bit multiply
FP4_OPERATION = "fp4_gemm"
#: the fields a row of the 4-bit multiply names: the backend the adapter passes, and whether the library's tuner may have
#: chosen the kernel (a row says False: the library's untuned pick)
MM_FP4_FORM = ("BACKEND", "TUNED")
BMM_FP8_FORM = ("BACKEND",)
#: the library's kernel-binary package: its distribution, the module the library looks for, its requirement file and the
#: index its versions are published on.  The library reads a package of its OWN version alone, so the validated version
#: of the package is the library's (``VALIDATED``); the constraints file pins it beside the library
BINARIES_DISTRIBUTION = "flashinfer-cubin"
BINARIES_MODULE = "flashinfer_cubin"
BINARIES_REQUIREMENTS = "samples/models/requirements-flashinfer-cubin.txt"
BINARIES_INDEX = "https://flashinfer.ai/whl"
#: the command that installs that package at the validated version: the index is named for this one command alone
BINARIES_INSTALL = f"pip install --no-deps --index-url {BINARIES_INDEX} -c {CONSTRAINTS} -r {BINARIES_REQUIREMENTS}"
#: the two variables of the library these adapters read or set: the directory it reads kernel binaries from in the place
#: of its package, and its switch against downloading one (any text that is not empty refuses)
BINARIES_DIRECTORY = "FLASHINFER_CUBIN_DIR"
NO_DOWNLOAD = "FLASHINFER_NO_DOWNLOAD"


def installed() -> Optional[str]:
    """The installed version of the distribution, read from its metadata (nothing is imported); ``None``: not installed."""
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def check_package(entry: str) -> str:
    """The validated version that is installed.  ``NotImplementedError`` (a SKIP) for a missing package and for a version
    that is not validated: the sentence names it, the fix and the stack that runs without the package."""
    version, validated = installed(), " or ".join(VALIDATED)
    fix = f"pip install -c {CONSTRAINTS} -r {REQUIREMENTS}"
    if version is None:
        raise NotImplementedError(f"{entry} needs {DISTRIBUTION} {validated}, which is not installed: {fix}; {WITHOUT}")
    if version not in VALIDATED:
        raise NotImplementedError(
            f"{entry} needs a validated {DISTRIBUTION} ({validated}: the launch records of its kernels are data of that version); this is {version}: {fix}; {WITHOUT}"
        )
    return version


def refuse_downloads() -> None:
    """Switch the library's own download of kernel binaries off for this process and for every process it starts: the
    variable the library reads at every load, set where the environment does not hold it or holds it empty (the library
    tests the text's truth).  A run then reads kernel binaries from the package or from a directory filled beforehand, and
    a file that is not there is the library's refusal, never a fetch."""
    if not os.environ.get(NO_DOWNLOAD):
        os.environ[NO_DOWNLOAD] = "1"


class LibraryBlock(BlockBase):
    """An adapter to a FlashInfer multiply on the stored format of its weight: what the three share.  A subclass names the
    fields of its row's form (``fields``), makes the call (``__call__``; it notes what it passed: ``_served``) and builds
    scratch operands for the first use (``_scratch``)."""

    provider = "FlashInfer"
    stored_weights = True  # the weight is multiplied on its stored codes: the loader keeps them for this block
    fields: tuple = ()
    #: the device types on which the plan makes the first use (a host test names its own device to run it on a stand-in library)
    first_use_devices = ("cuda",)
    _served = None  # the fields of the row's form as the last call passed them and the library's tuner stood: the call log's line
    _passed = ()  # the keywords of the last library call as (FIELD, value), written by ``_call`` from what it handed over
    _first = False  # inside the plan's first use (``first_use``): a failure of the library call is the installation's then
    _scalars = None  # (value, device) -> the one-element fp32 tensor a call reads: built once per value, never inside a later step

    def _library(self):
        """The library, imported on a device only (a plan on stated device facts never comes here) -- and never before its
        download of kernel binaries is switched off for the process: the one place every row reaches the library."""
        refuse_downloads()
        import flashinfer
        import flashinfer.autotuner  # noqa: F401 -- the tuner's state is read through the package (``_tuned``)

        return flashinfer

    def _form(self) -> dict:
        """The row's form.  A field this adapter does not pass, or a form that leaves one of its fields out, is a wrong table."""
        form = dict(self.row.form)
        unknown = [name for name in form if name not in self.fields]
        missing = [name for name in self.fields if name not in form]
        if unknown or missing:
            what = f"names {', '.join(unknown)}" if unknown else f"does not name {', '.join(missing)}"
            raise ValueError(f"provider row {self.row.key}: its form {what}; this adapter's rows name {', '.join(self.fields)} for {self.entry}")
        return form

    def _launches_here(self) -> bool:
        """Whether this plan is for a device the library launches on: a real CUDA device.  A plan on stated device facts,
        or for another device, states the table and asks nothing of the installation."""
        import torch

        return bool(self.facts.total_bytes) and "device" in self.options and torch.device(self.options["device"]).type in self.first_use_devices

    def check_support(self, sequences: int, tokens: int) -> None:
        self._form()
        if self._launches_here():
            check_package(self.entry)

    @property
    def first_use_note(self) -> str:
        if not self._launches_here():  # a plan on stated device facts makes no first use: nothing to announce
            return ""
        return (
            f"first use of FlashInfer {installed() or '(not installed)'}: it builds the kernels of its multiplies now, in this process and with the "
            'installed CUDA compiler and ninja (minutes on the first run of a machine, nothing on a later one; the family README, "What the first run does")'
        )

    def _scratch(self, device, rows: int) -> tuple:
        """Scratch operands of the run's row count, in the layout a step hands the block: the arguments of ``__call__``."""
        raise NotImplementedError

    def first_use(self, sequences: int, tokens: int) -> None:
        """While the run is planned, on a real CUDA device: the call on scratch operands, at the run's row count and -- for a
        run of several sequences -- at the rows of one sequence, which the batch-invariance check calls (the library builds
        the kernel of the 4-bit multiply per range of row counts).  ``ValueError`` (an ERROR) where the library refuses the
        operands this adapter built; ``NotImplementedError`` (a SKIP) for whatever else keeps the library itself from
        importing, building or launching on this installation (``_call``).  The scratch operands and everything else of
        this adapter's own code are outside that: a fault there is raised as it is, never read as the installation's."""
        import torch

        if not self._launches_here():
            return
        device = torch.device(self.options["device"])
        for rows in dict.fromkeys((sequences * tokens, tokens)):
            operands = self._scratch(device, rows)
            self._first = True
            try:
                self(*operands)
            except (TypeError, ValueError) as error:  # the library's own checks of what it was handed
                first = (str(error).splitlines() or [type(error).__name__])[0][:200]
                raise ValueError(f"{self.entry} declined the operands this adapter built for its first use: {first}") from None
            finally:
                self._first = False
                self._served = None  # a first use is no call of a step: the call log starts empty

    def _call(self, name: str, *operands, **passed):
        """The library call ``flashinfer.gemm.<name>`` itself, and what it was passed by keyword (``_passed``: what the call
        log reads back).  During the plan's first use the import, the build and the launch are the library's first on this
        installation: whatever keeps it from them, but its own check of the operands, is a ``NotImplementedError``."""
        self._passed = tuple((key.upper(), value) for key, value in passed.items())
        if not self._first:
            return getattr(self._library().gemm, name)(*operands, **passed)
        try:
            return getattr(self._library().gemm, name)(*operands, **passed)
        except (TypeError, ValueError, NotImplementedError):
            raise
        except Exception as error:  # noqa: BLE001 -- whatever keeps the library from importing, building or launching: the installation's
            raise NotImplementedError(
                f"{self.entry}: FlashInfer could not make its first use on this installation ({type(error).__name__}); it builds its kernels "
                f'there and needs the CUDA compiler and ninja (the family README, "What the first run does"); {WITHOUT}'
            ) from None

    def _scalar(self, device, *values: float):
        """One fp32 element on the device, formed on the host once per value: a scale, or the fp32 product of two."""
        import torch

        if self._scalars is None:
            self._scalars = {}
        key = (*(float(value) for value in values), str(device))
        if key not in self._scalars:
            element = torch.tensor([float(values[0])], dtype=torch.float32)
            for value in values[1:]:
                element = element * torch.tensor([float(value)], dtype=torch.float32)
            self._scalars[key] = element.to(device)
        return self._scalars[key]

    def serving(self):
        """What served the last call: the library function, and the fields of the row's form as that call passed them and
        as the library's tuner stood behind it.  Before the first call nothing was passed."""
        return (self.entry, ()) if self._served is None else (self.entry, self._served)


class MmFp4Block(LibraryBlock):
    """``flashinfer.gemm.mm_fp4(backend="cute-dsl")``: ``out[t, n] = bf16(alpha * sum_k activation[t, k] * weight[n, k])`` on
    the packed 4-bit codes of both operands with their block scales in the 128 x 4 layout; one launch per call, outside
    any tuning context.  This is the row of the MLP's down multiply."""

    kernels = (Kernel("fi_mm_fp4", "FlashInfer", "exact", launches=1),)  # one launch per call; its class and shape are census data
    fields = MM_FP4_FORM

    def _shape(self) -> tuple:
        """``(out features, in features)`` of the weight this block multiplies: the MLP's down projection."""
        g = self.model.geometry
        return g.hidden, g.mlp

    def stored_layout(self, merged):
        """The stored 4-bit weight in the layout the kernel multiplies on, built once per weight at load: the packed codes as
        they are, the block scales of their rows in the 128 x 4 layout.  A permutation of the scale bytes: nothing is
        requantised.  The loader keeps what this returns as the stored form of the weight, and nothing else of it."""
        if not isinstance(merged, BlockScaledTensor):
            raise ValueError(f"{self.entry} multiplies on a stored 4-bit weight; the loader handed its layout {type(merged).__name__}")
        return TiledBlockScaledWeight(merged.codes, tile_scales(merged.block_scales), float(merged.scale))

    def _operand(self, qx) -> tuple:
        """The 4-bit operand as the kernel reads it: ``(packed codes, block scales in the 128 x 4 layout, global scale)``.
        From a kernel on the stored formats it comes that way and is handed over as it is; from a block that is not a
        kernel the block scales come row by row, and this block brings them into the layout when it reads them."""
        if isinstance(qx, TiledBlockScaledTensor):
            return qx.codes, qx.tiled_scales, qx.scale
        if isinstance(qx, BlockScaledTensor):
            return qx.codes, tile_scales(qx.block_scales), qx.scale
        raise ValueError(f"{self.entry} multiplies packed 4-bit codes with their block scales; this block is handed {type(qx).__name__}")

    def _tuned(self) -> bool:
        """Whether the library's tuner may have chosen the kernel of the call that has just returned: it is tuning, or has
        tuned this operation in this process (a tuned winner outlives the context it was tuned in).  A tuning store the
        caller attached to the library is not read here: the tuner states none of it."""
        tuner = self._library().autotuner.AutoTuner.get()
        return bool(tuner.is_tuning_mode) or FP4_OPERATION in tuner.stats.tuned_op_total_configs

    def _scratch(self, device, rows: int) -> tuple:
        import torch

        n, k = self._shape()

        def zeros(leading: int) -> BlockScaledTensor:
            codes = torch.zeros((leading, k // 2), dtype=torch.uint8, device=device)
            return BlockScaledTensor(codes, torch.zeros((leading, k // BLOCK), dtype=torch.uint8, device=device), 1.0)

        return zeros(rows), Dequantized(None, 1.0, self.stored_layout(zeros(n)))

    def __call__(self, qx, weight, site=""):
        import torch

        stored = getattr(weight, "stored", None)
        if not isinstance(stored, TiledBlockScaledWeight):
            has = "it in the checkpoint's layout" if isinstance(stored, BlockScaledTensor) else "no stored 4-bit codes"
            raise ValueError(
                f"{self.entry} multiplies on the stored 4-bit weight with its block scales in the 128 x 4 layout; the weight of this call has {has}: "
                "the weights of this session were loaded for another plan (plan every session of a run before the first load: Session.plan_table)"
            )
        codes, scales, scale = self._operand(qx)
        backend, e4m3 = self._form()["BACKEND"], torch.float8_e4m3fn
        alpha = self._scalar(codes.device, scale, stored.scale)  # the fp32 product of the two global scales
        out = self._call("mm_fp4", codes, stored.codes.t(), scales.view(e4m3), stored.tiled_scales.view(e4m3).t(), alpha, torch.bfloat16, backend=backend)
        self._served = (*self._passed, ("TUNED", self._tuned()))  # what the call log of a traced step reads back (``serving``)
        return out


class QuantizeMmFp4Block(MmFp4Block):
    """The same multiply for a block whose operand reaches it as bf16 rows without codes: the output head.  The block
    quantises them with the reference quantiser when it reads them -- PyTorch's launches, this block's own -- and
    multiplies with the library: one launch of the library's kernel per call behind them."""

    kernels = (*PYTORCH_CHAIN, *MmFp4Block.kernels)

    def _shape(self) -> tuple:
        g = self.model.geometry
        return g.vocab, g.hidden

    def _operand(self, qx) -> tuple:
        return super()._operand(qx.quantized() if isinstance(qx, Unquantized) else qx)


class BmmFp8Block(LibraryBlock):
    """``flashinfer.gemm.bmm_fp8(backend="auto")``: ``out[t, n] = bf16(scale * sum_k activation[t, k] * weight[n, k])`` on
    the 8-bit codes of both operands; the kernel is the library's choice among its own (one launch, and a second where a
    split pick of cuBLASLt reduces its slices).  This is the row of a layer's out-projection."""

    kernels = (Kernel("cublaslt_fp8_gemm", "FlashInfer", "exact", launches=1), Kernel("cublas", "cuBLASLt"))
    fields = BMM_FP8_FORM

    def _scratch(self, device, rows: int) -> tuple:
        import torch

        g = self.model.geometry
        weight = ScaledTensor(torch.zeros((g.hidden, g.la_out), dtype=torch.uint8, device=device), 1.0)
        return ScaledTensor(torch.zeros((rows, g.la_out), dtype=torch.uint8, device=device), 1.0), Dequantized(None, 1.0, weight)

    def __call__(self, qx, weight, site=""):  # one call at every call site
        import torch

        stored = getattr(weight, "stored", None)
        if not isinstance(stored, ScaledTensor) or isinstance(stored.scale, torch.Tensor):
            has = "one scale per weight row" if isinstance(stored, ScaledTensor) else "no stored 8-bit codes"
            raise ValueError(
                f"{self.entry} multiplies on the stored 8-bit weight with its one scale; the weight of this call has {has}: "
                "the weights of this session were loaded for another plan (plan every session of a run before the first load: Session.plan_table)"
            )
        if not isinstance(qx, ScaledTensor):
            raise ValueError(f"{self.entry} multiplies static 8-bit codes; this block is handed {type(qx).__name__}")
        backend, e4m3, device = self._form()["BACKEND"], torch.float8_e4m3fn, qx.codes.device
        a, b = qx.codes.view(e4m3).unsqueeze(0), stored.codes.view(e4m3).t().unsqueeze(0)
        out = self._call("bmm_fp8", a, b, self._scalar(device, qx.scale), self._scalar(device, stored.scale), torch.bfloat16, backend=backend)
        self._served = self._passed
        return out.view(qx.codes.shape[0], stored.codes.shape[0])
