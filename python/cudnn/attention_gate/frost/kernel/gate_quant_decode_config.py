# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the attention gate + quantize DECODE kernel (``gate_quant_decode_f16.py``): the limits it serves and the compile-time knobs of
its one kernel body.  Importable without the CuTe DSL (the host entry validates a ``Cfg`` before the DSL version gate runs), so nothing here
imports ``cutlass``.

The op (the output gate of an attention block fused with the static FP8 quantization of the following projection's input, decode-time row
counts): ``out[r, c] = e4m3( bf16( attn[r, c] / (1 + exp(-gate[r, c])) ) * (1 / scale) )`` for ``attn`` / ``gate`` ``[rows, width]`` bf16,
``scale`` one fp32 element, ``out`` ``[rows, width]`` E4M3.  One launch, an element-wise map: thread ``i`` of the flat grid owns the ``VEC``
elements ``VEC i .. VEC i + VEC - 1`` of the row-major slab, so the element count ``N = rows x width`` is a RUNTIME kernel argument and one
compiled kernel serves every row count and width.

There is NO numerics knob: the kernel has no reduction, so nothing here can change an fp32 association -- every output byte is a function of
its own (attn, gate) pair and ``scale``.  The knobs are performance-only and bit-neutral (the suite compares their bytes):

* ``THREADS`` (128, 256): threads per CTA, hence the grid ``ceil(N / VEC / THREADS)``.  Picked by ``default_cfg`` from plan-time facts (the
  element count, the part's SM count and compute capability) through the data table ``THREADS_PICKS``; one row today (256 everywhere).  A row
  is added with its measured table.
* ``PDL_RELEASE`` (default off): ``griddepcontrol.launch_dependents`` as the kernel's first statement.  CALLER OBLIGATION when on: every grid
  that can start before this one completes (a grid launched with the programmatic-stream-serialization attribute, and transitively its own
  dependents) issues its ``griddepcontrol.wait`` before it reads ``out`` and before it writes ``attn``, ``gate`` or ``scale``.  Without such a
  grid on the stream the trigger is inert.

Limits (the host entry declines the rest by name): ``rows`` in 1 .. ``MAX_ROWS``; ``width`` a positive multiple of ``VEC`` up to ``MAX_WIDTH``
(the widest slab the suite runs on a device; the index arithmetic itself reaches ``ELEMENT_LIMIT`` elements).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

NAME = "gate_quant_decode"

VEC = 8  # elements per thread: one 128-bit load of each bf16 operand, eight one-byte stores of the E4M3 codes
MAX_ROWS = 160  # the served row ceiling per call (decode and verify step row counts; a taller slab is unmeasured)
SERVED_THREADS = (128, 256)
DEFAULT_THREADS = 256
#: the element index ``(block x THREADS + thread) x VEC`` is a signed 32-bit value compared with ``N``: the last CTA's largest index is
#: below ``N + VEC x THREADS``, so ``N <= 2^31 - VEC x max(THREADS)`` keeps every index representable for every served thread count.
ELEMENT_LIMIT = 2**31 - VEC * max(SERVED_THREADS)
#: the widest row the suite runs on a device (160 x 32768 elements = 2,560 CTAs of 256 threads); far inside ``ELEMENT_LIMIT``.
MAX_WIDTH = 32768

#: ``(cc_lo, cc_hi, max_elements, THREADS)``, first match wins; ``cc`` as ``10 x major + minor``.  DATA: a row is added with its measured table.
THREADS_PICKS: Tuple[Tuple[int, int, int, int], ...] = ((0, 999, ELEMENT_LIMIT, DEFAULT_THREADS),)


@dataclass(frozen=True)
class Cfg:
    """The compile-time knobs of the kernel body (both performance-only and bit-neutral; the module docstring)."""

    THREADS: int = DEFAULT_THREADS
    PDL_RELEASE: bool = False


def validate_cfg(cfg) -> Cfg:
    """``cfg`` when it is a served ``Cfg``: ``TypeError`` for another type, ``ValueError`` naming the knob outside its vocabulary."""
    if not isinstance(cfg, Cfg):
        raise TypeError(f"{NAME}: cfg must be a gate_quant_decode_config.Cfg, got {type(cfg).__name__}")
    if isinstance(cfg.THREADS, bool) or not isinstance(cfg.THREADS, int) or cfg.THREADS not in SERVED_THREADS:
        raise ValueError(f"{NAME}: THREADS must be one of {SERVED_THREADS}; got {cfg.THREADS!r}")
    if not isinstance(cfg.PDL_RELEASE, bool):
        raise ValueError(f"{NAME}: PDL_RELEASE must be a bool; got {cfg.PDL_RELEASE!r}")
    return cfg


def validate_width(width) -> int:
    """``width`` when the kernel serves it: an ``int`` (``TypeError`` otherwise) that is a positive multiple of ``VEC`` up to ``MAX_WIDTH``
    (``NotImplementedError`` otherwise: a row that is not whole threads needs a tail path, a wider one is not covered on a device)."""
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError(f"{NAME}: width must be an int, got {type(width).__name__}")
    if width < VEC or width % VEC or width > MAX_WIDTH:
        raise NotImplementedError(
            f"{NAME} serves row widths that are a positive multiple of {VEC} elements up to {MAX_WIDTH} (one 128-bit load per thread; the widest "
            f"slab covered on a device); got {width}"
        )
    return width


def validate_rows(rows: int) -> int:
    """``rows`` when the kernel serves it: at least one (``ValueError``), at most ``MAX_ROWS`` (``NotImplementedError``)."""
    if rows < 1:
        raise ValueError(f"{NAME}: a call holds at least one row; got {rows}")
    if rows > MAX_ROWS:
        raise NotImplementedError(
            f"{NAME} serves at most {MAX_ROWS} rows per call (decode and verify step row counts; a taller slab is unmeasured); got {rows}"
        )
    return rows


def blocks_for(elements: int, threads: int) -> int:
    """CTAs of the flat grid: ``ceil(elements / VEC / threads)`` -- the launcher's own arithmetic (``elements`` is a multiple of ``VEC``)."""
    return (elements // VEC + threads - 1) // threads


def last_index(elements: int, threads: int) -> int:
    """The largest element index any thread of the grid computes (the last thread of the last CTA; it exits on ``index >= elements``)."""
    return (blocks_for(elements, threads) * threads - 1) * VEC


def pick_threads(elements: int, sm_count: Optional[int] = None, cc: Optional[Tuple[int, int]] = None) -> int:
    """``THREADS`` for a slab of ``elements`` on a part: the first matching row of ``THREADS_PICKS``.  ``sm_count`` and ``cc`` are plan-time
    facts of the device (``None`` = unknown: every compute capability matches); ``sm_count`` is carried so that a row keyed on the wave count
    needs no signature change.  Performance-only: every pick writes the same bytes."""
    code = None if cc is None else 10 * int(cc[0]) + int(cc[1])
    for cc_lo, cc_hi, max_elements, threads in THREADS_PICKS:
        if elements <= max_elements and (code is None or cc_lo <= code <= cc_hi):
            return threads
    return DEFAULT_THREADS


def default_cfg(rows: int, width: int, sm_count: Optional[int] = None, cc: Optional[Tuple[int, int]] = None) -> Cfg:
    """The pick for a ``[rows, width]`` slab on a part (``pick_threads``), the release off."""
    return Cfg(THREADS=pick_threads(validate_rows(rows) * validate_width(width), sm_count, cc), PDL_RELEASE=False)
