# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the FP4 row quantization DECODE kernel (``row_quantize_fp4_decode_f16.py``): the limits it serves, the form of one kernel
body as a frozen ``Cfg``, and the two tables a reader audits (the barrier ledger, the shared-memory table) -- both EMPTY.  Importable
without the CuTe DSL and without torch (the host entry validates a ``Cfg`` before the DSL version gate runs).

The op, per block of sixteen consecutive elements of a row of a bf16 ``[rows, width]`` slab (fp32 arithmetic on bf16 values)::

    amax          = max |x_j|                                  over the block's sixteen elements
    byte, value   = e4m3( scale * (amax * fp32(1/6)) )         the block scale: nearest-even, saturating at 448
    encode        = 0                                          when value == 0
                  = rcp( value * rcp(scale) )                  otherwise: two approximate reciprocals
    code_j        = e2m1( x_j * encode )                       nearest-even on {0, 0.5, 1, 1.5, 2, 3, 4, 6}, saturating at +-6, sign kept
    codes         <- two codes per byte, the even element in the low nibble
    scales        <- byte, at the block's position in the 128 x 4 scale layout over (rows, blocks)

``scale`` is the GLOBAL scale of the operand: one fp32 value, the reciprocal of the consuming multiply's static input scale.

One thread block per row, ONE THREAD PER BLOCK of sixteen elements, so ``THREADS = width / 16`` threads in ``WARPS`` full warps.  A
thread reads its own sixteen values and the shared ``scale``; nothing is reduced across threads.

WHAT CHANGES BYTES: the sixteen values of a block and the value of ``scale``.  ``WIDTH`` moves addresses only.  WHAT NEVER DOES: the
row count, a row's position, the other rows, the other blocks of the row, ``PDL`` and ``PDL_RELEASE`` (control instructions).

THE FORM IS IN THE KERNEL SYMBOL.  ``str(cfg)`` is a compact text (``w5120_wait_release``: the width and the launch arm); the DSL
appends it to the kernel's name, so two forms never share a symbol and the form is readable inside the first 100 characters.
``repr(cfg)`` stays the field list.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

NAME = "the FP4 row quantization decode step"

BLOCK = 16  # elements per block scale = elements per thread: two 128-bit loads of the bf16 row, one 64-bit store of its codes
WARP = 32
#: the structural limits of ``width``: full warps of one-block threads (a multiple of ``BLOCK x WARP``), at most 1024 threads.  Every
#: such width is also whole scale atoms (four scale columns each): the scale buffer has no padded column.
WIDTH_STEP = BLOCK * WARP
MAX_WIDTH = BLOCK * 1024
#: the widths the entry serves.
SERVED_WIDTH: Tuple[int, ...] = (5120,)
#: the served row ceiling per call (the ceiling of the other row decode steps of this library).  Whether a call of this many
#: rows crosses a scheduling wave depends on the part: one wave holds ``CTAS_PER_SM`` x the SM count rows.
MAX_ROWS = 1024
#: the element index ``row x width + column`` is a signed 32-bit value: ``MAX_ROWS x MAX_WIDTH`` stays below it (an invariant of these
#: constants, never a decline of a call).
ROW_ADDRESS_LIMIT = 2**31 - 1
#: threads one SM holds: ``SM_THREADS // THREADS`` thread blocks co-reside (the kernel has no shared memory, and threads bind before
#: registers here).
SM_THREADS = 2048
#: the 128 x 4 scale layout: rows of one band, scale columns of one atom, bytes of one atom.
SF_TILE_ROWS = 128
SF_ATOM_COLS = 4
SF_ATOM_BYTES = SF_TILE_ROWS * SF_ATOM_COLS


@dataclass(frozen=True)
class Cfg:
    """The form of one kernel body: every field is a compile-time constant and the whole config is the compile key.  ``str`` is
    the compact form text the kernel symbol carries (the module docstring); ``repr`` is the field list."""

    WIDTH: int
    PDL: bool
    PDL_RELEASE: bool

    # --- derived (read-only) ---
    @property
    def THREADS(self) -> int:
        return self.WIDTH // BLOCK

    @property
    def WARPS(self) -> int:
        return self.THREADS // WARP

    @property
    def SMEM_BYTES(self) -> int:
        """The launch requests no shared memory."""
        return 0

    @property
    def SF_BLOCKS(self) -> int:
        """Block scales per row."""
        return self.WIDTH // BLOCK

    @property
    def SF_ATOMS(self) -> int:
        """Scale atoms (four scale columns each) per band of 128 rows."""
        return (self.SF_BLOCKS + SF_ATOM_COLS - 1) // SF_ATOM_COLS

    @property
    def CTAS_PER_SM(self) -> int:
        """Thread blocks of this form one SM holds: what one scheduling wave runs per SM."""
        return SM_THREADS // self.THREADS

    @property
    def LAUNCH_ARM(self) -> str:
        if self.PDL:
            return "wait_release" if self.PDL_RELEASE else "wait"
        return "release" if self.PDL_RELEASE else "plain"

    def __str__(self) -> str:
        return f"w{self.WIDTH}_{self.LAUNCH_ARM}"


def validate_width(width) -> int:
    """``width`` when it is structurally a row width of this kernel: an ``int`` (``TypeError`` otherwise) that is a multiple of 512 in
    512 .. 16384 (``ValueError`` otherwise).  Whether the entry SERVES a structurally valid width is ``SERVED_WIDTH``."""
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError(f"{NAME}: width must be an int, got {type(width).__name__}")
    if width < WIDTH_STEP or width > MAX_WIDTH or width % WIDTH_STEP:
        raise ValueError(
            f"{NAME}: width must be a multiple of {WIDTH_STEP} in {WIDTH_STEP} .. {MAX_WIDTH} (full warps of one-block threads, {BLOCK} elements "
            f"per block scale, at most {MAX_WIDTH // BLOCK} threads); got {width}"
        )
    return width


def validate_cfg(cfg) -> Cfg:
    """``cfg`` when it is a form this kernel body has: ``TypeError`` for another type or a field of the wrong type, ``ValueError``
    naming the argument outside its structure or the inconsistent pair.  Messages name the host entry's arguments (``width``, ``pdl``,
    ``pdl_release``)."""
    if not isinstance(cfg, Cfg):
        raise TypeError(f"{NAME}: cfg must be a row_quantize_fp4_decode_config.Cfg, got {type(cfg).__name__}")
    validate_width(cfg.WIDTH)
    for name, value in (("pdl", cfg.PDL), ("pdl_release", cfg.PDL_RELEASE)):
        if not isinstance(value, bool):
            raise TypeError(f"{NAME}: {name} must be a bool, got {type(value).__name__}")
    if cfg.PDL_RELEASE and not cfg.PDL:
        raise ValueError(f"{NAME}: pdl_release releases dependents after this launch's own wait and needs pdl=True")
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


def scale_rows(rows: int) -> int:
    """Rows of the 128 x 4 scale buffer that holds ``rows`` rows -- ``ceil(rows / 128) x 128``."""
    return (rows + SF_TILE_ROWS - 1) // SF_TILE_ROWS * SF_TILE_ROWS


def scale_byte_offset(cfg: Cfg, row, block):
    """The byte of block scale ``(row, block)`` in the scale buffer -- the host twin of the kernel's index: bands of 128 rows one
    after the other, ``SF_ATOMS`` atoms of 512 bytes each; inside a band the 128 x 4 atom rule."""
    from cudnn.frost.tile_dsl.sf_layout import sf_band_offset  # plain Python: no DSL import

    return sf_band_offset(row, block, cfg.SF_ATOMS)


def smem_table(cfg: Cfg) -> Tuple[Dict[str, Any], ...]:
    """The kernel's shared-memory buffers, one row each: NONE.  The launch requests no shared memory; every value lives in a
    thread's registers.  Nothing is refilled inside a launch, so nothing is released."""
    return ()


def barrier_ledger(cfg: Cfg) -> Dict[str, int]:
    """The kernel's synchronization objects: NONE.  No barrier, no warp shuffle: a thread owns its whole block, so no value
    crosses a thread.  The only ordering between launches is the dependent-launch pair of the launch arm."""
    return {}
