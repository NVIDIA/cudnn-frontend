# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the FP8 row quantization DECODE kernel (``row_quantize_fp8_decode_f16.py``): the limits it serves, the form of one
kernel body as a frozen ``Cfg``, and the two tables a reader audits (the barrier ledger, the shared-memory table).  Importable
without the CuTe DSL and without torch (the host entry validates a ``Cfg`` before the DSL version gate runs).

The op, per row of a bf16 ``[rows, width]`` slab (fp32 arithmetic on the bf16 values)::

    amax    = max |x_j|                                  over the row
    scale   = max( amax / 448, fp32(1 / (448 * 512)) )   one correctly rounded fp32 division, one exact maximum
    inv     = 1 / scale                                  one correctly rounded fp32 division
    code_j  = e4m3( clamp(x_j * inv, -448, 448) )        one fp32 multiplication, nearest-even to E4M3, saturating at +-448
    codes   <- code_j, one byte per element              scales <- scale, one fp32 value per row

One thread block per row.  A thread owns whole VECTORS of ``VEC`` = 8 consecutive elements (one 128-bit load, one 64-bit store):
thread ``t`` owns the vectors ``t``, ``t + THREADS``, ``t + 2 * THREADS`` of its row -- as many PASSES as the row needs, at most
three.  ``THREADS`` is the row's vector count rounded up to full warps, at most 1024; a thread whose vector of the last pass lies
past the row owns none there: it loads and stores nothing for it and holds ``+0``.  The row maximum is one block reduction over
the threads' maxima; everything behind it is element-wise.

WHAT CHANGES BYTES: the row's values.  WHAT NEVER DOES: the row count, a row's position, the other rows, ``WIDTH`` beyond the
values it holds (it moves addresses and the thread plan; a maximum is exact in any order), ``PDL`` and ``PDL_RELEASE`` (control
instructions).

THE FORM IS IN THE KERNEL SYMBOL.  ``str(cfg)`` is a compact text (``w17408_release_wait``: the width and the launch arm); the
DSL appends it to the kernel's name, so two forms never share a symbol and the form is readable inside the first 100 characters.
``repr(cfg)`` stays the field list.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

NAME = "the FP8 row quantization decode step"

VEC = 8  # elements per vector: one 128-bit load of the bf16 row, one 64-bit store of its codes
WARP = 32
MAX_THREADS = 1024  # threads of one thread block
MAX_PASSES = 3  # vectors one thread owns, at most
#: the structural limits of ``width``: whole vectors (a multiple of ``VEC``), at most ``MAX_PASSES`` vectors per thread at
#: ``MAX_THREADS`` threads.
MAX_WIDTH = VEC * MAX_THREADS * MAX_PASSES
#: the widths the entry serves.
SERVED_WIDTH: Tuple[int, ...] = (17408,)
#: the served row ceiling per call (the ceiling of the other row decode steps of this library).  Above the rows one scheduling
#: wave holds on the served parts (at most ``CTAS_PER_SM`` x the SM count), so a call of this many rows crosses waves.
MAX_ROWS = 1024
#: the most threads one SM of a served part holds: AT MOST ``SM_THREADS // THREADS`` thread blocks co-reside.  A bound over the
#: served parts, not a count: a part whose SM holds fewer threads holds fewer blocks, and the registers of the compiled kernel
#: bind first when it holds more than 32 per thread at 1024 threads (two such blocks need more than an SM's register file),
#: and then ONE thread block occupies an SM.
SM_THREADS = 2048
ELEMENT_BYTES = 2  # bf16
SCALE_BYTES = 4  # one fp32 scale per row


@dataclass(frozen=True)
class Cfg:
    """The form of one kernel body: every field is a compile-time constant and the whole config is the compile key.  ``str`` is
    the compact form text the kernel symbol carries (the module docstring); ``repr`` is the field list."""

    WIDTH: int
    PDL: bool
    PDL_RELEASE: bool

    # --- derived (read-only) ---
    @property
    def VECS(self) -> int:
        """Vectors of ``VEC`` elements per row."""
        return self.WIDTH // VEC

    @property
    def THREADS(self) -> int:
        """Threads per thread block: the row's vectors rounded up to full warps, at most ``MAX_THREADS``."""
        return min(MAX_THREADS, WARP * ((self.VECS + WARP - 1) // WARP))

    @property
    def WARPS(self) -> int:
        return self.THREADS // WARP

    @property
    def PASSES(self) -> int:
        """Vectors one thread owns, at most: pass ``p`` of thread ``t`` is vector ``t + p * THREADS``."""
        return (self.VECS + self.THREADS - 1) // self.THREADS

    def live_threads(self, p: int) -> int:
        """Threads that own a vector in pass ``p``: every thread of a full pass, the first ``VECS - p * THREADS`` of the last."""
        return max(0, min(self.THREADS, self.VECS - p * self.THREADS))

    @property
    def SMEM_BYTES(self) -> int:
        """Shared memory of one thread block: one fp32 word per warp (the warp maxima of the row maximum)."""
        return 4 * self.WARPS

    @property
    def CTAS_PER_SM(self) -> int:
        """Thread blocks of this form one SM holds AT MOST (the thread limit; the compiled kernel's registers can bind first): the
        most one scheduling wave runs per SM."""
        return SM_THREADS // self.THREADS

    @property
    def LAUNCH_ARM(self) -> str:
        if self.PDL:
            return "release_wait" if self.PDL_RELEASE else "wait"
        return "release" if self.PDL_RELEASE else "plain"

    def __str__(self) -> str:
        return f"w{self.WIDTH}_{self.LAUNCH_ARM}"


def validate_width(width) -> int:
    """``width`` when it is structurally a row width of this kernel: an ``int`` (``TypeError`` otherwise) that is a multiple of
    8 in 8 .. 24576 (``ValueError`` otherwise).  Whether the entry SERVES a structurally valid width is ``SERVED_WIDTH``."""
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError(f"{NAME}: width must be an int, got {type(width).__name__}")
    if width < VEC or width > MAX_WIDTH or width % VEC:
        raise ValueError(
            f"{NAME}: width must be a multiple of {VEC} in {VEC} .. {MAX_WIDTH} (whole {VEC}-element vectors, at most {MAX_PASSES} vectors per "
            f"thread at {MAX_THREADS} threads); got {width}"
        )
    return width


def validate_cfg(cfg) -> Cfg:
    """``cfg`` when it is a form this kernel body has: ``TypeError`` for another type or a field of the wrong type, ``ValueError``
    naming the argument outside its structure or the inconsistent pair.  Messages name the host entry's arguments (``width``,
    ``pdl``, ``pdl_release``)."""
    if not isinstance(cfg, Cfg):
        raise TypeError(f"{NAME}: cfg must be a row_quantize_fp8_decode_config.Cfg, got {type(cfg).__name__}")
    validate_width(cfg.WIDTH)
    for name, value in (("pdl", cfg.PDL), ("pdl_release", cfg.PDL_RELEASE)):
        if not isinstance(value, bool):
            raise TypeError(f"{NAME}: {name} must be a bool, got {type(value).__name__}")
    if cfg.PDL_RELEASE and not cfg.PDL:
        raise ValueError(f"{NAME}: pdl_release releases dependents in front of this launch's own wait and needs pdl=True")
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


def smem_table(cfg: Cfg) -> Tuple[Dict[str, Any], ...]:
    """The kernel's shared-memory buffers, one row each: ONE.  Written once and read once per launch, so nothing releases it."""
    return (
        dict(
            name="warp maxima",
            dtype="fp32",
            elems=cfg.WARPS,
            writer="lane 0 of each warp, its own word, before the thread-block barrier",
            reader="behind the barrier: the lanes below the warp count of every warp, word = lane",
            refilled_in_a_launch=False,
        ),
    )


def barrier_ledger(cfg: Cfg) -> Dict[str, int]:
    """The kernel's synchronization objects by name with their participant count: ONE thread-block barrier (inside the block
    reduction of the row maximum), issued once by every thread of the block.  No mbarrier."""
    return {"the thread-block barrier of the row maximum": cfg.THREADS}
