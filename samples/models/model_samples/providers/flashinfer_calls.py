# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Library kernels of FlashInfer as providers of model blocks: three multiplies, each on the stored format of its weight,
and the attention core over the paged attention cache.

``flashinfer.mm_fp4`` calls ``flashinfer.gemm.mm_fp4(backend="cute-dsl")``, the multiply of a 4-bit activation by a 4-bit
weight on the packed codes of both: one launch per call.  The library picks the kernel of a call itself, by the shape and
the GPU: one of two kernel classes of one module, in a grid of its own choice.  That pick is a function of the call as long
as nothing has tuned the operation, so the row says ``TUNED False`` and the adapter calls the library outside any tuning
context; which class and which launch shape the call then gets is the library's own, and a run prints the
kernel that was launched on the row's line.  ``TUNED`` is a statement about this process: the library's tuner is tuning, or has tuned the operation
here.  A tuning store the caller attached to the library is not seen by it; a tactic replayed from one shows in the
kernel a run prints, and is not gated.

The operand is the 4-bit output of the block before it: from a kernel on the stored formats it comes with its block
scales in the 128 x 4 layout this kernel reads and is handed over as it is; from a block that is not a kernel the scales
come row by row, and bringing them into the layout is this block's own launch.  The weight is the stored 4-bit weight
with its block scales in that layout (``weights.TiledBlockScaledWeight``), built once per weight at load
(``stored_layout``) and kept as the weight's one stored form; the loader holds it to the bytes it was made from.

``flashinfer.mm_fp4_head`` is the same call for the output head: the same kernel module on the vocabulary's weight, whose
shape the scratch call of its first use needs.  Its operand is what the quantise block in front of it hands over; the
adapter reads it as the down multiply's does and launches nothing in front of the library's kernel.

``flashinfer.bmm_fp8`` calls ``flashinfer.gemm.bmm_fp8(backend="auto")``, the multiply of an 8-bit activation by an 8-bit
weight on the stored codes of both.  Which kernel serves it is the choice of the library among its own: its tuner's where
one has run, else its first candidate.  That choice is never a row and never gated; the kernel check holds the launch as
one of the kernels this call can launch, and prints its name on the row's line.

``flashinfer.decode_attention`` calls ``flashinfer.decode.trtllm_batch_decode_with_kv_cache``, the library's paged decode
attention, with the library's own backend for these GPU classes (``DECODE_ATTENTION_BACKEND``): the attention of the rows
of a step over the keys and values of their sequences, on the pool of pages the attention preparation wrote, read
through the state's page table and lengths (``state.PagedKvCache.attended``).  One launch per call and nothing else: the
adapter hands the library views and host constants, an output buffer of the call's own, one workspace per device and a
zero counter buffer, and converts nothing on the device.  The library checks almost none of what it is handed and
returns other bytes for a table of another dtype, a strided view or a table off its width, so the adapter holds its own
operands on the host before every call (``ValueError``).  Which kernel the library launches follows the run's maximum
length, the sequences, the tokens per sequence and the GPU: the row is of the call, the kernel check holds the launch by
the family's name, and a run prints the kernel it launched.  The library documents no numerics contract for this kernel:
its band is the bound of a stated model, judged against the reference chain with the operand terms of that bound
(``twin``, ``tie_out``: ``attention_bf16_weights``), and it states no batch invariance (``batch_note``).

**The package.**  FlashInfer is no dependency of cuDNN Frontend and none of the default stack.  It is imported inside the
functions of this module that run on a device, never at module import and never by a plan on stated device facts.  A plan
for a CUDA device reads the installed version from the distribution's metadata: a missing package, or a version that is not
validated (``VALIDATED``: the versions this stack was run and checked with), is a ``NotImplementedError``
-- a ``SKIP`` before the load that names the version, the requirement file and the stack that runs without the package.

**Kernel binaries are never fetched by a run.**  The attention kernels are precompiled binaries the library reads from
files: from the directory ``FLASHINFER_CUBIN_DIR`` names, else from its kernel-binary package (``BINARIES_DISTRIBUTION``)
where that has the library's own version, else from a cache directory -- and it downloads a file it does not find.  These
adapters switch that download off for the process before they reach the library (``refuse_downloads``, in the one method
every row reaches it through).  The library itself does not always refuse a file it lacks: where the missing binary is
not the first one its kernel selection reads, it prints lines with a path of its installation from its own code and
launches another kernel.  So the attention's adapter does not rely on the refusal: before its first call it holds every
kernel binary the library's table names for the call's data types, head width, paging and page on the GPU's class --
the table in the header of the library's own kernel set, never a list of this module, and more binaries than the
library's selection can reach for the call -- to the SHA-256 of that set's manifest, and the manifest to the SHA-256
the library itself names for it (``hold_binaries``): a file that is missing or is not the manifest's, or a manifest
that is not the library's, is the ``SKIP`` that names what is missing and the fix (``missing_binaries``).  Two states
of the installation make the library
print a local path at its import -- the package at another version than the library's, and the directory variable beside
the package -- and one more is undecidable from the metadata: each is a ``SKIP`` at plan time, before the library is
imported (``check_binaries``).

**The library's own lines.**  The library logs through loggers of its own, at the level its variable
``FLASHINFER_LOGGING_LEVEL`` names when it is imported and at ``info`` where the variable names none: a first build then
prints lines of the library on standard error, some of them with a path of its cache directory.  These adapters set the
variable to ``warning`` for the process before they import the library, in the same method as the refusal of a download
and where the environment names no level -- the variable unset, empty, or blanks alone (``warnings_only``).  A level
the environment names stays, and the library then prints those lines.  Its warnings and errors reach the streams either
way.  A text there that names no level is not the sample's to correct: the library's import raises on it, so a plan
declines it before the library is imported, in a sentence that names the variable (``check_level``: a ``ValueError``,
the run's ``ERROR``, and never the sentence of operands an adapter built).

**First use is part of the plan.**  The library builds what a call needs when it is first made: its CuTe DSL kernels in the
process, its CUTLASS multiply as a module compiled with the installed CUDA compiler and ninja (minutes, once per machine).
The 4-bit kernel is built per range of row counts.  The attention builds a launcher with the same compiler and reads
kernel binaries at the first call of every step shape, the adapter having held them all before the first.  So an
adapter makes the call on scratch operands while the run is planned (``first_use``) -- at the run's row count and, for a
run of several sequences, at the rows of one sequence, which is what the batch-invariance check calls -- after the
memory sentence has held and before anything is loaded, and a sample announces it (``first_use_note``).  A refusal of
the operands this adapter built is a ``ValueError`` -- an ``ERROR``, because a faulty adapter must not read as an
unsupported installation; whatever else keeps the LIBRARY from importing, building or launching there is the
installation's, and the ``SKIP``.  A fault of this adapter's own code is neither: it is raised as it is.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.metadata as metadata
import importlib.util
import logging
import os
import re
import sys
from typing import Dict, Optional, Tuple

from ..census import FAMILIES, Kernel
from ..state import PagedKvCache, PageTable
from ..weights import BLOCK, BlockScaledTensor, Dequantized, ScaledTensor, TiledBlockScaledTensor, TiledBlockScaledWeight, tile_scales
from . import BlockBase

#: the distribution these adapters import, and the versions of it this area has validated: those the stack ``native`` was
#: last run and checked with.  ``samples/models/constraints-native.txt`` pins the same, and the library's kernel-binary
#: package below at the library's own version; a host test holds them to each other
DISTRIBUTION = "flashinfer-python"
VALIDATED = ("0.7.2.post1",)
REQUIREMENTS = "samples/models/requirements-flashinfer.txt"
CONSTRAINTS = "samples/models/constraints-native.txt"
#: the stack that runs the cuDNN Frontend rows of ``native`` without this package: what a decline points to
WITHOUT = "--stack fe-entries runs the cuDNN Frontend rows of this stack without the package"
#: the name under which the library's tuner keeps the 4-bit multiply
FP4_OPERATION = "fp4_gemm"
#: the fields a row of the 4-bit multiply names: the backend the adapter passes, and whether the library's tuner may have
#: chosen the kernel (a row says False: the library's untuned pick)
MM_FP4_FORM = ("BACKEND", "TUNED")
BMM_FP8_FORM = ("BACKEND",)
#: the field a row of the paged decode attention names: the layout of the two pool halves
DECODE_ATTENTION_FORM = ("KV_LAYOUT",)
#: the value of that call's ``backend`` argument: the library's own backend for these GPU classes, whose kernels are
#: precompiled binaries, and the name of the folder its kernel set stands in.  Every call passes it: the library's default
#: chooses a backend by the GPU, and an explicit value does not move with a release of the library
DECODE_ATTENTION_BACKEND = "trtllm-gen"
#: the library's kernel-binary package: its distribution, the module the library looks for, its requirement file and the
#: index its versions are published on.  The library reads a package of its OWN version alone, so the validated version
#: of the package is the library's (``VALIDATED``); the constraints file pins it beside the library
BINARIES_DISTRIBUTION = "flashinfer-cubin"
BINARIES_MODULE = "flashinfer_cubin"
BINARIES_REQUIREMENTS = "samples/models/requirements-flashinfer-cubin.txt"
BINARIES_INDEX = "https://flashinfer.ai/whl"
#: the command that installs that package at the validated version: the index is named for this one command alone
BINARIES_INSTALL = f"pip install --no-deps --index-url {BINARIES_INDEX} -c {CONSTRAINTS} -r {BINARIES_REQUIREMENTS}"
#: the three variables of the library these adapters read or set: the directory it reads kernel binaries from in the
#: place of its package, its switch against downloading one (any text that is not empty refuses), and the level of its
#: own loggers, which it reads once, when it is imported (the name of a level of the interpreter's logging module)
BINARIES_DIRECTORY = "FLASHINFER_CUBIN_DIR"
NO_DOWNLOAD = "FLASHINFER_NO_DOWNLOAD"
LOGGING_LEVEL = "FLASHINFER_LOGGING_LEVEL"
#: how the library's refusal of a file that is not on the machine begins -- the manifest or the header at the first use of
#: its launcher, a kernel binary at the first call of a step shape -- and the module of its loader, whose callback raises
#: the first where no caller can catch it.  Read to tell that refusal from any other failure, never printed: each text
#: goes on with a local path
MISSING_FILE, MISSING_KERNEL = "Artifact not found locally", "Failed to load cubin for"
LOADER = "flashinfer.jit.cubin_loader"
#: the files of a kernel set, in its folder under the library's kernel-binary directory: the manifest (a line per file:
#: its SHA-256 and its name), the header whose table names every kernel of the set, and the binary of a kernel
BINARIES_MANIFEST, BINARIES_HEADER, BINARY = "checksums.txt", "include/flashInferMetaInfo.h", "{}.cubin"
#: (folder of a kernel set, kernel family, compute capability) -> what ``hold_binaries`` found there, once per process
_HELD: dict = {}
#: the workspace the paged decode attention is handed: the size every run of this stack was made with.  The library
#: makes no size check of it, and a smaller one has not been run
WORKSPACE_BYTES = 394 * 1024 * 1024
#: device -> the one workspace of that device in this process, built at the first plan that needs it and handed by every
#: adapter of the device
_WORKSPACES: dict = {}


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
            f"{entry} needs {DISTRIBUTION} {validated}, the version this stack was run and checked with; this is {version}: {fix}; {WITHOUT}"
        )
    return version


def refuse_downloads() -> None:
    """Switch the library's own download of kernel binaries off for this process and for every process it starts: the
    variable the library reads at every load, set where the environment does not hold it or holds it empty (the library
    tests the text's truth).  A run then reads kernel binaries from the package or from a directory filled beforehand, and
    a file that is not there is the library's refusal, never a fetch."""
    if not os.environ.get(NO_DOWNLOAD):
        os.environ[NO_DOWNLOAD] = "1"


def warnings_only() -> None:
    """Keep the library's informational lines off the streams of this process and of every process it starts: the level
    of the library's own loggers, which it reads from this variable when it is imported, set to ``warning`` where the
    environment names none.  An empty text and a text of blanks alone name none: the library takes the text for the
    name of a level, and its import raises on either.  A level that is named there stays -- the library then prints
    lines of its own at a first build, some with a path of its cache directory.  Its warnings and errors reach the
    streams either way.  A text that names no level does not come here: a plan declines it (``check_level``)."""
    if not os.environ.get(LOGGING_LEVEL, "").strip():
        os.environ[LOGGING_LEVEL] = "warning"


def check_level(entry: str) -> None:
    """At plan time, before the library is imported: the text of the library's logging-level variable, where it is
    neither empty nor blanks alone.  The library upper-cases that text and hands it to ``setLevel`` of a logger of its
    own when it is imported, and that raises on a text that is no level name of the interpreter's ``logging`` module.
    The same call on a logger of this function's own decides here, so every text the library takes is taken.
    ``ValueError`` (an ERROR: the text was set outside the sample) that names the variable and never its text."""
    text = os.environ.get(LOGGING_LEVEL, "")
    if not text.strip():
        return
    try:
        logging.Logger(LOGGING_LEVEL).setLevel(text.upper())
    except (TypeError, ValueError):
        raise ValueError(
            f"{entry}: the environment's {LOGGING_LEVEL} names no level of Python's logging module, and FlashInfer takes its text for one when it is "
            "imported; name a level there (debug, info, warning, error or critical), or unset the variable and this sample sets it to warning"
        ) from None


def binaries_package() -> Tuple[Optional[str], bool]:
    """The kernel-binary package as the installation has it, asked both ways and with nothing imported: the version its
    distribution's metadata states (``None``: no such distribution), and whether the module the library looks for can be
    found (the library decides by the module alone)."""
    try:
        version = metadata.version(BINARIES_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        version = None
    try:
        found = importlib.util.find_spec(BINARIES_MODULE) is not None
    except (ImportError, ValueError):  # a module of that name without a spec: nothing the library could read
        found = False
    return version, found


def check_binaries(entry: str) -> None:
    """At plan time, before the library is imported: the states of the kernel-binary package in which the library's import
    would print a local path, or in which what it will read cannot be told.  ``NotImplementedError`` (a SKIP) that names
    the state and the fix, never a path and never the variable's value:

    * the distribution's metadata and the module disagree -- one is there without the other;
    * the package is installed at another version than the library's: the library ignores it with a warning;
    * the directory variable is set while the package's module can be found: the library takes the directory with a
      warning.

    With neither the metadata nor the module nothing is declined here: the library then reads the directory the variable
    names, or its cache directory, and what is missing there is found before the first call (``hold_binaries``)."""
    version, found = binaries_package()
    validated = " or ".join(VALIDATED)
    if version is None and found:
        raise NotImplementedError(
            f"{entry}: the module {BINARIES_MODULE} can be imported and no distribution {BINARIES_DISTRIBUTION} is installed, so the version of the "
            f"kernel binaries the library would read is not stated; install the package: {BINARIES_INSTALL}; {WITHOUT}"
        )
    if version is not None and not found:
        raise NotImplementedError(
            f"{entry}: the distribution {BINARIES_DISTRIBUTION} {version} is installed and its module {BINARIES_MODULE} cannot be found; install the "
            f"package again: {BINARIES_INSTALL}; {WITHOUT}"
        )
    if version is not None and version not in VALIDATED:
        raise NotImplementedError(
            f"{entry} needs the kernel-binary package {BINARIES_DISTRIBUTION} at the library's version, {validated}; this is {version}, which the "
            f"library ignores: {BINARIES_INSTALL}; {WITHOUT}"
        )
    if found and os.environ.get(BINARIES_DIRECTORY):
        raise NotImplementedError(
            f"{entry}: {BINARIES_DIRECTORY} is set and the kernel-binary package {BINARIES_DISTRIBUTION} is installed, and the library would read "
            f"the directory in the place of the package; unset the variable to read the package, or uninstall the package to read the directory; {WITHOUT}"
        )


def missing_binaries(entry: str, found: str) -> str:
    """The sentence of a SKIP for kernel binaries that are not on the machine as their manifest lists them (``found``: what
    is missing, in words without a path).  It names the fix that is still to be made, decided from what the
    installation holds: the package that is not installed; the directory the variable names; or, with the package
    installed at the library's version, that package itself, to be installed anew -- the command that installs a
    missing package leaves an installed one as it is, so the sentence names none."""
    version, module = binaries_package()
    lacks = (
        f"does not hold the kernel binaries the library's table names for this call's data types, head width, paging and page on this GPU's class "
        f"as their manifest lists them ({found})"
    )
    if module and version in VALIDATED:
        return f'{entry}: the installed {BINARIES_DISTRIBUTION} {version} {lacks}; install it anew (the family README, "What the first run does"); {WITHOUT}'
    if os.environ.get(BINARIES_DIRECTORY):
        return (
            f'{entry}: the directory {BINARIES_DIRECTORY} names {lacks}; fill it (the family README, "What the first run does") or install the '
            f"package: {BINARIES_INSTALL}; {WITHOUT}"
        )
    return f"{entry} needs the kernel-binary package {BINARIES_DISTRIBUTION} {' or '.join(VALIDATED)}, which is not installed: {BINARIES_INSTALL}; {WITHOUT}"


def kernel_set(header: str, family: "re.Pattern", cc: Tuple[int, int]) -> Dict[str, str]:
    """``{kernel name: SHA-256 of its binary}`` for the kernels of ``family`` that the table of a kernel set's header offers
    a GPU of compute capability ``cc``.  The header declares one record type and lists one record per kernel in the
    order of that type's fields; three fields are read, each found by its name: the architecture (``mSM``), the
    kernel's name (``mFuncName``) and the SHA-256 (``sha256``).  A kernel serves a GPU whose architecture is its own
    (``kSM_<major><minor>``) or the family of its major class (``kSM_<major>0f``): the rule of the library's launcher on
    the 10.x classes.  ``ValueError`` for a text that is no such table."""
    declared = re.search(r"struct\s+\w+\s*\{(.*?)\}\s*;", header, re.S)
    fields = [line.split()[-1].rstrip(";") for line in declared.group(1).splitlines() if line.strip()] if declared else []
    architecture, name, digest = (fields.index(field) for field in ("mSM", "mFuncName", "sha256"))  # ValueError: no such field
    serves = (f"kSM_{cc[0]}{cc[1]}", f"kSM_{cc[0]}0f")
    kernels = {}
    for line in header[declared.end() :].splitlines():
        if line.startswith("{"):
            record = [value.strip().strip('"') for value in line.strip().strip("{},").split(", ")]
            if len(record) != len(fields):
                raise ValueError(f"a record of the header's table has {len(record)} values, its type {len(fields)} fields")
            if record[architecture] in serves and family.match(record[name]):
                kernels[record[name]] = record[digest]
    return kernels


def hold_binaries(folder: str, family: "re.Pattern", cc: Tuple[int, int], manifest: Optional[str] = None) -> str:
    """Hold the kernel binaries in ``folder``, the folder of a kernel set, that the table of the set's header names for the
    kernel family ``family`` -- a call's data types, head width, paging and page -- on a GPU of compute capability
    ``cc``: the header is the file the set's manifest lists; every kernel the header's table names for them
    (``kernel_set``) has a line in the manifest that states the table's SHA-256; and its binary is in the folder with
    that SHA-256.  Returns ``""`` where all of it holds, else what does not, in words that name no path.  The set is
    more than a run reads, and more than the library's selection can reach for the call -- which of these kernels a
    step shape reads is the library's selection, which nothing here models -- so that no call of a run can meet a
    binary that is not there.  The manifest itself is held to ``manifest``, the SHA-256 the library names for it, where
    one is given: the library holds it to the same constant when it first reads it, and reports a difference in a line
    that holds a local path.  Read once per process: the manifest, the header and the binaries (46 files and 28 MB at
    the published widths on compute capability 10.0)."""
    key = (folder, family.pattern, tuple(cc), manifest)
    if key not in _HELD:
        _HELD[key] = _held_binaries(folder, family, cc, manifest)
    return _HELD[key]


def _held_binaries(folder: str, family: "re.Pattern", cc: Tuple[int, int], named: Optional[str]) -> str:
    def read(name: str) -> Optional[bytes]:
        try:
            with open(os.path.join(folder, name), "rb") as file:
                return file.read()
        except OSError:
            return None

    manifest = read(BINARIES_MANIFEST)
    if manifest is None:
        return "their manifest is not there"
    listed = dict(reversed(line.split(None, 1)) for line in manifest.decode("utf-8", "replace").splitlines() if len(line.split(None, 1)) == 2)
    header = read(BINARIES_HEADER)
    if header is None or hashlib.sha256(header).hexdigest() != listed.get(BINARIES_HEADER):
        return "the header of the kernel set is not the file their manifest lists"
    try:
        kernels = kernel_set(header.decode("utf-8", "replace"), family, cc)
    except ValueError:
        return "the header of the kernel set is not the table of kernels this adapter reads"
    if not kernels:
        return "the header of the kernel set names no kernel of this call for this GPU"
    wrong = 0
    for name, digest in kernels.items():
        binary = read(BINARY.format(name))
        wrong += binary is None or not (hashlib.sha256(binary).hexdigest() == listed.get(BINARY.format(name)) == digest)
    if wrong:
        return f"{wrong} of {len(kernels)} missing or changed"
    if named is not None and hashlib.sha256(manifest).hexdigest() != named:
        return "their manifest is not the file the library names"
    return ""


def _kernel_set_folder(library) -> Tuple[str, str]:
    """Where the library reads the kernel set of the paged decode attention, and the SHA-256 it names for that set's
    manifest, by its own answer: the kernel-binary directory it resolved at its import; among its artifact paths, the one
    that ends in the backend's folder; and, among its checksums, the one of the manifest in that folder."""
    paths = [getattr(library.artifacts.ArtifactPath, field.name) for field in dataclasses.fields(library.artifacts.ArtifactPath)]
    found = [path for path in paths if isinstance(path, str) and path.strip("/").endswith(f"fmha/{DECODE_ATTENTION_BACKEND}")]
    if len(found) != 1:
        raise LookupError(f"the library names {len(found)} folders of its attention kernel set")
    named = getattr(library.artifacts.CheckSumHash, "map_checksums", {}).get(f"{found[0].rstrip('/')}/{BINARIES_MANIFEST}")
    if not isinstance(named, str):
        raise LookupError("the library names no SHA-256 for the manifest of its attention kernel set")
    return os.path.join(str(library.jit.env.FLASHINFER_CUBIN_DIR), found[0]), named


@contextlib.contextmanager
def _first_use_reports():
    """While a first use reaches the library: the interpreter's report of ONE error is kept off the streams -- the library's
    ``RuntimeError`` of a file that is not on the machine, raised inside its loader's callback, where the interpreter can
    only report it as an exception it ignored (the call then raises an error of its own, which the first use turns into
    the SKIP).  That report holds a local path.  Every other report is handed on to the hook that was there."""
    previous = sys.unraisablehook

    def hook(unraisable):
        error, raised_in = unraisable.exc_value, getattr(unraisable.object, "__module__", "")
        if type(error) is RuntimeError and raised_in == LOADER and str(error).startswith(MISSING_FILE):
            return
        previous(unraisable)

    sys.unraisablehook = hook
    try:
        yield
    finally:
        sys.unraisablehook = previous


class LibraryBlock(BlockBase):
    """An adapter to a FlashInfer call: what the rows of the library share.  A subclass names the module of its call
    (``module``) and the fields of its row's form (``fields``), makes the call (``__call__``; it notes what it passed:
    ``_served``) and builds scratch operands for the first use (``_scratch``).  The multiplies' adapters multiply on the
    stored format of their weight."""

    provider = "FlashInfer"
    stored_weights = True  # the weight is multiplied on its stored codes: the loader keeps them for this block
    captures = True  # a library call is launches on the operands it is handed; what it reads of the host it reads once, at its first call
    module = "gemm"  # of the library: where the call of this adapter lives
    fields: tuple = ()
    #: the device types on which the plan makes the first use (a host test names its own device to run it on a stand-in library)
    first_use_devices = ("cuda",)
    _served = None  # the fields of the row's form as the last call passed them and the library's tuner stood: the call log's line
    _passed = ()  # the keywords of the last library call as (FIELD, value), written by ``_call`` from what it handed over
    _first = False  # inside the plan's first use (``first_use``): a failure of the library call is the installation's then
    _scalars = None  # (value, device) -> the one-element fp32 tensor a call reads: built once per value, never inside a later step

    def _library(self):
        """The library, imported on a device only (a plan on stated device facts never comes here) -- and never before its
        download of kernel binaries is switched off for the process and the level of its own lines is set: the one place
        every row reaches the library."""
        refuse_downloads()
        warnings_only()
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
            check_level(self.entry)

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

    def _scratches(self, device, sequences: int, tokens: int):
        """The scratch operands of every call the plan's first use makes, one after the other: at the run's row count and,
        for a run of several sequences, at the rows of one sequence."""
        for rows in dict.fromkeys((sequences * tokens, tokens)):
            yield self._scratch(device, rows)

    def first_use(self, sequences: int, tokens: int) -> None:
        """While the run is planned, on a real CUDA device: the call on scratch operands, at the run's row count and -- for a
        run of several sequences -- at the rows of one sequence, which the batch-invariance check calls (the library builds
        the kernel of the 4-bit multiply per range of row counts).  ``ValueError`` (an ERROR) where the library refuses the
        operands this adapter built; ``NotImplementedError`` (a SKIP) for whatever else keeps the library itself from
        importing, building or launching on this installation (``_call``).  The scratch operands and everything else of
        this adapter's own code are outside that: a fault there is raised as it is, never read as the installation's.
        In front of the first call a row holds what it needs of the installation (``_hold``: the attention's kernel binaries)."""
        import torch

        if not self._launches_here():
            return
        self._hold()
        device = torch.device(self.options["device"])
        for operands in self._scratches(device, sequences, tokens):
            self._first = True
            try:
                self(*operands)
            except (TypeError, ValueError) as error:  # the library's own checks of what it was handed
                first = (str(error).splitlines() or [type(error).__name__])[0][:200]
                raise ValueError(f"{self.entry} declined the operands this adapter built for its first use: {first}") from None
            finally:
                self._first = False
                self._served = None  # a first use is no call of a step: the call log starts empty

    def _hold(self) -> None:
        """What a row holds of the installation in front of its first call: nothing, for a row whose kernels the library builds."""

    def _call(self, name: str, *operands, **passed):
        """The library call ``flashinfer.<module>.<name>`` itself, and the fields of the row's form as it was passed them by
        keyword, in the row's order (``_passed``: what the call log reads back)."""
        stated = {key.upper(): value for key, value in passed.items()}
        self._passed = tuple((name, stated[name]) for name in self.fields if name in stated)
        return self._reach(lambda: getattr(getattr(self._library(), self.module), name)(*operands, **passed))

    def _reach(self, use):
        """``use()``, one use of the library.  During the plan's first use the import, the build and the launch are the
        library's first on this installation: whatever keeps it from them, but its own check of the operands, is a
        ``NotImplementedError`` (``_unavailable``), and the interpreter's report of a file the library's loader did not
        find stays off the streams (``_first_use_reports``).  In a step nothing is renamed."""
        if not self._first:
            return use()
        try:
            with _first_use_reports():
                return use()
        except (TypeError, ValueError, NotImplementedError):
            raise
        except Exception as error:  # noqa: BLE001 -- whatever keeps the library from importing, building or launching: the installation's
            raise NotImplementedError(self._unavailable(error)) from None

    def _unavailable(self, error: Exception) -> str:
        """The sentence of the SKIP for what kept the library from its first use: the type of the exception, never its text."""
        return (
            f"{self.entry}: FlashInfer could not make its first use on this installation ({type(error).__name__}); it builds its kernels "
            f'there and needs the CUDA compiler and ninja (the family README, "What the first run does"); {WITHOUT}'
        )

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

    # one launch per call; the id stands for the two kernel classes of the library's multiply, and which one a call launches is
    # the library's function of the operands' shape: printed with the kernel check, never derived
    kernels = (Kernel("fi_mm_fp4", "FlashInfer", "exact", launches=1, variants=True),)
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


class MmFp4HeadBlock(MmFp4Block):
    """The same multiply for the output head: one launch of the library's kernel per call, on the 4-bit operand the quantise
    block in front of it hands over.  What differs from the down multiply is the weight's shape."""

    def _shape(self) -> tuple:
        """``(out features, in features)`` of the weight this block multiplies: the output head's."""
        g = self.model.geometry
        return g.vocab, g.hidden


class BmmFp8Block(LibraryBlock):
    """``flashinfer.gemm.bmm_fp8(backend="auto")``: ``out[t, n] = bf16(scale * sum_k activation[t, k] * weight[n, k])`` on
    the 8-bit codes of both operands; the kernel is the library's choice among its own (one launch, and a second where a
    split pick of cuBLASLt reduces its slices).  This is the row of a layer's out-projection."""

    # the id stands for the kernels the library chooses among for this call: printed with the kernel check, never derived
    kernels = (Kernel("cublaslt_fp8_gemm", "FlashInfer", "exact", launches=1, variants=True), Kernel("cublas", "cuBLASLt"))
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


class DecodeAttentionBlock(LibraryBlock):
    """``flashinfer.decode.trtllm_batch_decode_with_kv_cache``, the library's paged decode attention, with the library's own
    backend for these GPU classes (``DECODE_ATTENTION_BACKEND``): per row of a step and query head the attention over
    the keys and values of the row's sequence up to the row's own position, ``[rows, heads x D]`` bf16; one launch per
    attention layer and step.  The cache is the pool of pages the attention preparation wrote: the adapter
    hands the library the pool's two halves, and the page-table rows and the lengths of the step as the state holds them
    for the run's maximum length (``state.PagedKvCache.attended``).  It converts nothing on the device and keeps nothing
    between two steps but what it built with the plan: the workspace of its device and a zero counter buffer."""

    module = "decode"
    # one launch per call; the id stands for the variants of one kernel, and which one a call launches is the library's
    # function of the run's maximum length, the sequences, the tokens per sequence and the GPU: printed, never derived
    kernels = (Kernel("fi_decode_attention", "FlashInfer", "exact", launches=1, variants=True),)
    fields = DECODE_ATTENTION_FORM
    stored_weights = False  # the block has no weight
    per_row = (0, 1)
    state_args = (1,)  # the layer's cache: the call writes nothing, and the tie-out holds the pool byte for byte
    cache_page = 64  # the positions of a page of the pool this adapter hands the library: the preparation's
    attended_tables = True  # ... which the kernel reads through the state's page table and lengths
    batch_note = (
        "the library states no batch invariance for this call: the kernel it launches follows the sequences, the tokens per sequence and the run's "
        "maximum length, and the bytes of one sequence alone can differ from the same sequence among several"
    )
    workspace_bytes = WORKSPACE_BYTES
    _twin = None
    _counters = None  # (device, sequences) -> the zero counter buffer of that sequence count, of the library's size
    _step = (0, 0, 0)  # (sequences, tokens per sequence, maximum length) of the call that is being made

    def check_support(self, sequences: int, tokens: int) -> None:
        super().check_support(sequences, tokens)
        if self._launches_here():
            check_binaries(self.entry)

    @property
    def first_use_note(self) -> str:
        if not self._launches_here():
            return ""
        return (
            f"first use of the paged decode attention of FlashInfer {installed() or '(not installed)'}: this sample switches the library's download "
            f"of kernel binaries off for the process ({NO_DOWNLOAD}), so they are read from its kernel-binary package or from a directory filled "
            "beforehand and never fetched; it holds the kernel binaries of this GPU to their manifest, and the library then builds its launcher with "
            'the installed CUDA compiler and ninja and reads those of every step shape of the run (the family README, "What the first run does")'
        )

    def twin(self, reference):
        """The block this adapter is compared with: the reference chain itself, which also keeps the operand terms of the
        band's bound.  One twin per adapter: it keeps the terms of its last call."""
        from . import torch_ref

        if self._twin is None:
            self._twin = torch_ref.AttentionBound(reference.row, self.model, self.facts, self.options)
        return self._twin

    def reserve_bytes(self) -> int:
        """The workspace the library's kernel is handed, for the memory sentence of the plan: one per device and process,
        so it is counted until it is built (it is then part of what the device reports as used)."""
        import torch

        if _WORKSPACES and "device" in self.options and _device_key(torch.device(self.options["device"])) in _WORKSPACES:
            return 0
        return self.workspace_bytes

    def _workspace(self, device):
        """The one workspace of ``device`` in this process: zero bytes, built at the first plan that needs it."""
        import torch

        key = _device_key(device)
        if key not in _WORKSPACES:
            _WORKSPACES[key] = torch.zeros(self.workspace_bytes, dtype=torch.uint8, device=device)
        return _WORKSPACES[key]

    def _counter(self, device, sequences: int):
        """The counter buffer of a call of ``sequences`` sequences: zero bytes of the library's size for that count, built
        once per count (the kernel resets its counters at the end of every launch)."""
        import torch

        if self._counters is None:
            self._counters = {}
        key = (_device_key(device), sequences)
        if key not in self._counters:
            heads, sms = self.model.geometry.attn_heads, int(self.facts.sm_count)
            size = self._reach(lambda: self._library().utils.get_trtllm_gen_multi_ctas_kv_counter_bytes(sequences, heads, sms))
            self._counters[key] = torch.zeros(int(size), dtype=torch.uint8, device=device)
        return self._counters[key]

    def _output(self, q):
        """The output buffer of one call, the call's own: ``[rows, heads, D]`` bf16, contiguous."""
        import torch

        return torch.empty(tuple(q.shape), dtype=torch.bfloat16, device=q.device)

    def _limit(self) -> int:
        """The run's maximum length, which the kernel is told beside its operands: a constant of the plan's options."""
        limit = self.options.get("max_model_len")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(
                f"{self.entry} is told the run's maximum length beside its operands, a whole number of positions; the plan of this block names "
                f"max_model_len={limit!r}"
            )
        return limit

    def _held(self, q, blocks, lengths, pools, out, sequences: int, width: int) -> None:
        """What the library does not check and reads all the same: ``ValueError`` that names the operand.  The page-table
        rows and the lengths are int32 on the query's device, contiguous, each starting 16 bytes aligned, of exactly
        ``[sequences, width]`` and ``[sequences]`` -- the width the state's for the run's maximum length; the output is
        contiguous; the last axis of both pool halves has stride 1."""
        import torch

        for name, table, shape in (("the page table", blocks, (sequences, width)), ("the lengths", lengths, (sequences,))):
            wrong = ""
            if table.dtype != torch.int32:
                wrong = f"of dtype {table.dtype}: the kernel reads 32-bit words"
            elif table.device != q.device:
                wrong = f"on {table.device}, the query on {q.device}"
            elif tuple(table.shape) != shape:
                wrong = f"of shape {tuple(table.shape)}: this call reads {shape}"
            elif not table.is_contiguous():
                wrong = "a strided view: the kernel reads adjacent words"
            elif table.data_ptr() % 16:
                wrong = "not 16 bytes aligned"
            if wrong:
                raise ValueError(f"{self.entry}: {name} of this step is {wrong}")
        if not out.is_contiguous():
            raise ValueError(f"{self.entry}: the output buffer of this call is a strided view: the kernel writes adjacent rows")
        for name, pool in zip(("key", "value"), pools):
            if pool.stride(-1) != 1:
                raise ValueError(f"{self.entry}: the {name} half of the pool has stride {pool.stride(-1)} on its last axis: the kernel reads adjacent elements")

    def _scratches(self, device, sequences: int, tokens: int):
        """One scratch step per form the run calls: its sequences at its tokens per sequence, and one sequence alone, which
        the batch-invariance check calls -- each at the run's maximum length, on a zero pool of the pages one step holds
        and the tables of a state of that size."""
        import torch

        g, page, limit = self.model.geometry, self.cache_page, self._limit()
        pages = -(-tokens // page)
        for count in dict.fromkeys((sequences, 1)):
            cos_sin = torch.zeros((tokens, g.rotary_dim), dtype=torch.bfloat16, device=device)
            table = PageTable.build(count, tokens, tokens, page, pages, cos_sin, device, limit)
            pool = torch.zeros((count * pages, g.attn_kv_heads, page, 2 * g.attn_head_dim), dtype=torch.bfloat16, device=device)
            yield torch.zeros((count * tokens, g.attn_heads, g.attn_head_dim), dtype=torch.bfloat16, device=device), PagedKvCache(pool, pages, table, tokens), 0

    def _hold(self) -> None:
        """Before this adapter's first call of the library: the kernel binaries the library's table names for this
        call's data types, head width, paging and page on the GPU's class are on the machine as their manifest lists
        them, and the manifest is the file the library names (``hold_binaries``; the family is the kernel this
        block declares, the class the plan's), or the run is the ``SKIP`` of ``missing_binaries``.  The library is
        imported for it and nothing of it is called: a run that passes cannot meet a missing binary later."""
        self._first = True
        try:
            folder, manifest = self._reach(lambda: _kernel_set_folder(self._library()))
        finally:
            self._first = False
        found = hold_binaries(folder, dict(FAMILIES)[self.kernels[0].id], tuple(self.facts.cc), manifest)
        if found:
            raise NotImplementedError(missing_binaries(self.entry, found))

    def _unavailable(self, error: Exception) -> str:
        """A file the library did not find with its download refused is the SKIP that names what is missing and the fix;
        whatever else kept it from its first use is the installation's, as for every row."""
        if type(error) is RuntimeError and str(error).startswith((MISSING_FILE, MISSING_KERNEL)):
            sequences, tokens, limit = self._step
            return missing_binaries(self.entry, f"the library found a file missing at {sequences} x {tokens} rows at maximum length {limit}")
        return super()._unavailable(error)

    def __call__(self, q, cache, position):
        if not isinstance(cache, PagedKvCache) or cache.page != self.cache_page:
            has = f"pages of {cache.page} positions" if isinstance(cache, PagedKvCache) else type(cache).__name__
            raise ValueError(
                f"{self.entry} reads an attention cache that is a pool of pages of {self.cache_page} positions; this block is handed {has}: "
                "the session's state was built for another plan"
            )
        limit, form = self._limit(), self._form()
        rows, tokens = int(q.shape[0]), cache.tokens
        sequences = rows // tokens
        if position + tokens > limit:  # the library neither declines nor cuts a length past the maximum it is told
            raise ValueError(
                f"{self.entry}: a step of {tokens} token{'s' if tokens > 1 else ''} at position {position} passes the run's maximum length {limit}"
            )
        self._step = (sequences, tokens, limit)
        blocks, lengths = cache.attended(position)  # the position reaches the kernel as rows of the state's tables, through the cache alone
        pools, out = (cache.k_pool, cache.v_pool), self._output(q)
        self._held(q, blocks, lengths, pools, out, sequences, PageTable.width(cache.page, cache.pages, limit))
        self._call(
            "trtllm_batch_decode_with_kv_cache",
            query=q,
            kv_cache=pools,
            workspace_buffer=self._workspace(q.device),
            block_tables=blocks,
            seq_lens=lengths,
            max_seq_len=limit,
            bmm1_scale=int(q.shape[-1]) ** -0.5,  # the chain's scale, as a Python number: a tensor would be a multiply on the device
            bmm2_scale=1.0,
            out=out,
            kv_layout=form["KV_LAYOUT"],
            backend=DECODE_ATTENTION_BACKEND,
            q_len_per_req=tokens,
            multi_ctas_kv_counter_buffer=self._counter(q.device, sequences),
        )
        self._served = self._passed  # what the call log of a traced step reads back (``serving``)
        return out.view(rows, -1)  # the buffer the kernel wrote, as the rows the gate block reads


def _device_key(device) -> str:
    """A device as one name whether or not it states its index: ``cuda`` is the current CUDA device."""
    import torch

    index = device.index
    if index is None:
        index = torch.cuda.current_device() if device.type == "cuda" else 0
    return f"{device.type}:{index}"
