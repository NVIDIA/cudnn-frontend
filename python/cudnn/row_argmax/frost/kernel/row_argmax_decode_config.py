# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the row argmax DECODE kernel (``row_argmax_decode_f16.py``): the limits it serves, the form of one kernel body as a
frozen ``Cfg``, the kinds of id table a call can bind, and the two tables a reader audits (the barrier ledger, the shared-memory
table).  Importable without the CuTe DSL and without torch (the host entry validates a ``Cfg`` before the DSL version gate runs).

The op, per row of a bf16 ``[rows, width]`` slab::

    column  = the LOWEST column whose value is greater than -inf and not below any other value of the row;
              0 when the row holds no such value (every element is -inf or a NaN)
    result  = column                 without an id table   (int32)
            = ids[column]            with one              (a bit copy, in the table's dtype)

A NaN is never chosen and never hides another element; ``+0`` and ``-0`` are equal, so the lower column wins.  The op is exact:
no arithmetic, no rounding, one result in any evaluation order.

One thread block per row; ``THREADS`` threads in ``WARPS`` full warps; a thread owns ``PASSES`` vectors of ``VEC`` elements, the
vectors ``thread + THREADS x pass`` of its row, and scans them in ascending column order.  The kernel body runs WHOLE passes: a
width is whole passes of ``THREADS`` threads (``VEC x THREADS`` elements each), so no thread ever skips a vector.

WHAT CHANGES A RESULT: the values of the row and, with a table, the table.  WHAT NEVER DOES: the row count, a row's position,
the other rows, the row stride of the slab, who owns the output.

THE FORM IS IN THE KERNEL SYMBOL.  ``str(cfg)`` is a compact text (``w65536``) and the table kind a second one (``none``,
``int32``, ``int64``); the DSL appends both to the kernel's name, so two forms never share a symbol.  ``repr(cfg)`` stays the
field list.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

NAME = "the row argmax decode step"

VEC = 8  # elements per thread per pass: one 128-bit load of the bf16 row
ELEMENT_BYTES = 2  # bf16
VEC_BYTES = VEC * ELEMENT_BYTES
WARP = 32
THREADS = 1024  # threads per thread block: 32 full warps
MAX_PASSES = 8  # vectors per thread, at most: the scan is unrolled
#: the structural limits of ``width``: whole 16-byte vectors, at most ``MAX_PASSES`` per thread
MAX_WIDTH = VEC * THREADS * MAX_PASSES
#: one pass of the thread block: a served width is whole passes
PASS_WIDTH = VEC * THREADS
#: the widths the entry serves
SERVED_WIDTH: Tuple[int, ...] = (65536,)
#: the served row ceiling per call (the ceiling of the other row decode steps of this library).  Whether a call of this many rows
#: crosses a scheduling wave depends on the part: one wave holds ``CTAS_PER_SM`` x the SM count rows.
MAX_ROWS = 1024
#: threads one SM holds: ``SM_THREADS // THREADS`` thread blocks co-reside
SM_THREADS = 2048
#: the kinds of id table a call binds: none (the result is the column), or a table of that integer dtype.  The kind is a load and
#: a store width of the kernel, so each kind is one compiled kernel.
TABLE_KINDS: Tuple[str, ...] = ("none", "int32", "int64")
#: the dtype of a call's result per table kind, by name, and its element size in bytes
RESULT_DTYPE: Dict[str, str] = {"none": "int32", "int32": "int32", "int64": "int64"}
INT_BYTES: Dict[str, int] = {"int32": 4, "int64": 8}


@dataclass(frozen=True)
class Cfg:
    """The form of one kernel body: every field is a compile-time constant.  ``str`` is the compact form text the kernel symbol
    carries (the module docstring); ``repr`` is the field list."""

    WIDTH: int

    # --- derived (read-only) ---
    @property
    def THREADS(self) -> int:
        return THREADS

    @property
    def WARPS(self) -> int:
        return THREADS // WARP

    @property
    def PASSES(self) -> int:
        """Vectors per thread: the passes of the thread block over the row."""
        return self.WIDTH // PASS_WIDTH

    @property
    def SMEM_WORDS(self) -> int:
        """The fp32 words of the one shared-memory buffer: a maximum and an index per warp."""
        return 2 * self.WARPS

    @property
    def SMEM_BYTES(self) -> int:
        return 4 * self.SMEM_WORDS

    @property
    def CTAS_PER_SM(self) -> int:
        """Thread blocks of this form one SM holds: what one scheduling wave runs per SM."""
        return SM_THREADS // self.THREADS

    def __str__(self) -> str:
        return f"w{self.WIDTH}"


def validate_width(width) -> int:
    """``width`` when it is structurally a row width of this kernel: an ``int`` (``TypeError`` otherwise) that is a multiple of 8
    in 8 .. 65536 (``ValueError`` otherwise).  Whether the entry SERVES a structurally valid width is ``SERVED_WIDTH``."""
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError(f"{NAME}: width must be an int, got {type(width).__name__}")
    if width < VEC or width > MAX_WIDTH or width % VEC:
        raise ValueError(
            f"{NAME}: width must be a multiple of {VEC} in {VEC} .. {MAX_WIDTH} (whole 16-byte vectors of the bf16 row, at most "
            f"{MAX_PASSES} per thread of a {THREADS}-thread block); got {width}"
        )
    return width


def validate_cfg(cfg) -> Cfg:
    """``cfg`` when it is a form this kernel body has: ``TypeError`` for another type, the declines of :func:`validate_width`,
    ``NotImplementedError`` for a width that is not whole passes (the body has no pass in which only some threads own a vector)."""
    if not isinstance(cfg, Cfg):
        raise TypeError(f"{NAME}: cfg must be a row_argmax_decode_config.Cfg, got {type(cfg).__name__}")
    validate_width(cfg.WIDTH)
    if cfg.WIDTH % PASS_WIDTH:
        raise NotImplementedError(
            f"{NAME}: the kernel body runs whole passes of {THREADS} threads ({PASS_WIDTH} elements each); width {cfg.WIDTH} is not a "
            f"multiple of {PASS_WIDTH}"
        )
    return cfg


def validate_rows(rows) -> int:
    """``rows`` when a call of that many rows is served: an ``int`` (``TypeError``), at least one (``ValueError``), at most
    ``MAX_ROWS`` (``NotImplementedError``)."""
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise TypeError(f"{NAME}: rows must be an int, got {type(rows).__name__}")
    if rows < 1:
        raise ValueError(f"{NAME}: rows must be at least 1, got {rows}")
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_ROWS} rows per call; got {rows}")
    return rows


def validate_table_kind(kind) -> str:
    """``kind`` when it names a kind of id table: one of ``TABLE_KINDS`` (``ValueError`` otherwise)."""
    if not isinstance(kind, str) or kind not in TABLE_KINDS:
        raise ValueError(f"{NAME}: the id table kind must be one of {TABLE_KINDS}, got {kind!r}")
    return kind


def smem_table(cfg: Cfg) -> Tuple[Dict[str, Any], ...]:
    """The kernel's shared-memory buffers, one row each (the kernel docstring's table as data).  ONE buffer: per warp an fp32
    maximum and the bits of an index, written once by lane 0 of its warp before the thread-block barrier and read once behind it.
    It is never refilled inside a launch, so nothing releases it."""
    return (
        dict(
            name="warp_maxima_and_indices",
            dtype="float32",
            elements=cfg.SMEM_WORDS,
            bytes=cfg.SMEM_BYTES,
            lane_stride_bytes=4,
            writers="lane 0 of each warp, before the thread-block barrier",
            readers=f"lanes 0 .. {cfg.WARPS - 1} of every warp, behind the thread-block barrier",
            refilled=False,
        ),
    )


def barrier_ledger(cfg: Cfg) -> Dict[str, int]:
    """The one thread-block barrier: every thread of the block takes part and every thread issues it exactly once, so the two
    counts are equal by construction (``participants == issuing_lanes == THREADS``).  No other synchronization object exists."""
    return dict(participants=cfg.THREADS, issuing_lanes=WARP * cfg.WARPS)
