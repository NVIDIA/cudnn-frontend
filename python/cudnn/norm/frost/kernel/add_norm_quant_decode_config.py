# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the add + RMSNorm + quantize DECODE kernel (``add_norm_quant_decode_f16.py``): the limits it serves, the form of one
kernel body as a frozen ``Cfg``, and the two tables a reader audits (the barrier ledger, the shared-memory table).  Importable without
the CuTe DSL and without torch (the host entry validates a ``Cfg`` before the DSL version gate runs).

The op, per row of an ``[rows, hidden]`` slab (fp32 arithmetic on bf16 operands)::

    v      = x + residual                 (ADD)             residual <- bf16(v), in place
    v      = x                            (not ADD)         residual <- x under STORE_RESIDUAL
    y      = bf16( v * rsqrt(mean(v^2) + eps) * (1 + weight) )
    out    = y                                              (BF16_OUT)
    codes  = e4m3( y * (1 / scale) )                        (QUANT == "e4m3_static")
    codes, scales = nvfp4( y ; global scale )               (QUANT == "nvfp4": 16-element blocks, E4M3 block scales, 128 x 4 layout)

One thread block per row, ``VEC`` contiguous elements per thread, so ``THREADS = hidden / VEC`` threads in ``WARPS`` full warps.  The
row's sum of squares is a fixed tree whose shape depends on ``hidden`` alone (a fused multiply-add chain over a thread's ``VEC``
elements, a butterfly over the 32 lanes of a warp, a butterfly over the warps); everything behind it is element-wise.

WHAT CHANGES BYTES: ``HIDDEN`` (the tree and the divisor), ``EPS``, ``ADD`` (the function that is normalized) and ``QUANT`` (which
codes exist).  WHAT NEVER DOES: the row count, a row's position, the other rows, ``BF16_OUT`` and ``STORE_RESIDUAL`` (they select
which buffers are written; ``y`` is rounded to bf16 in registers in every form), ``PDL`` and ``PDL_RELEASE`` (control instructions).

``QUANT_FORMS`` is the vocabulary of ``QUANT`` besides ``None``; a later form is appended to it and to the kernel as one more
compile-time arm, with every existing form's rendering unchanged (dead kernel operands are ``None`` and are dropped from the
specialized kernel, so an appended arm adds no parameter to today's forms).

THE FORM IS IN THE KERNEL SYMBOL.  ``str(cfg)`` is a compact text (``h5120_add_bf16_e4m3_static_wait_release_eps358637bd``: the
width, the residual mode, the outputs, the launch arm and the fp32 bits of ``eps``); the DSL appends it to the kernel's name, so two
forms never share a symbol and the form is readable inside the first 100 characters.  ``repr(cfg)`` stays the field list.
"""

import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

NAME = "the add + RMSNorm + quantize decode step"

VEC = 8  # elements per thread: one 128-bit load of each bf16 operand
WARP = 32
#: the structural limits of ``hidden``: full warps of ``VEC``-element threads (a multiple of ``VEC x WARP``), at most 1024 threads.
HIDDEN_STEP = VEC * WARP
MAX_HIDDEN = VEC * 1024
#: the widths the entry serves (each is its own byte class: the sum tree and the divisor follow the width).
SERVED_HIDDEN: Tuple[int, ...] = (5120,)
#: the served row ceiling per call.  Above the rows one scheduling wave holds on every served part (``CTAS_PER_SM`` x the SM count),
#: so the suite's largest cell crosses a wave boundary.
MAX_ROWS = 1024
#: the element index ``row x hidden + column`` is a signed 32-bit value.
ROW_ADDRESS_LIMIT = 2**31 - 1
#: threads one SM holds: ``SM_THREADS // THREADS`` thread blocks co-reside (threads bind before registers and shared memory here).
SM_THREADS = 2048
#: NVFP4: elements per block scale; the 128 x 4 scale layout (rows of one scale tile, scale columns of one atom, bytes of one atom).
SF_BLOCK = 16
SF_TILE_ROWS = 128
SF_ATOM_COLS = 4
SF_ATOM_BYTES = SF_TILE_ROWS * SF_ATOM_COLS
#: the vocabulary of ``Cfg.QUANT`` besides ``None``.  A literal: an appended form is a visible diff.
QUANT_FORMS: Tuple[str, ...] = ("e4m3_static", "nvfp4")


def eps_bits(eps: float) -> int:
    """The fp32 bit pattern of ``eps`` -- the constant the kernel adds (``0x358637BD`` for ``1e-6``)."""
    return struct.unpack("<I", struct.pack("<f", eps))[0]


@dataclass(frozen=True)
class Cfg:
    """The form of one kernel body: every field is a compile-time constant and the whole config is the compile key.  ``str`` is
    the compact form text the kernel symbol carries (the module docstring); ``repr`` is the field list."""

    HIDDEN: int
    EPS: float
    ADD: bool
    STORE_RESIDUAL: bool
    BF16_OUT: bool
    QUANT: Optional[str]
    PDL: bool
    PDL_RELEASE: bool

    # --- derived (read-only) ---
    @property
    def THREADS(self) -> int:
        return self.HIDDEN // VEC

    @property
    def WARPS(self) -> int:
        return self.THREADS // WARP

    @property
    def SMEM_BYTES(self) -> int:
        """The kernel's one shared-memory buffer: one fp32 slot per warp."""
        return 4 * self.WARPS

    @property
    def SF_BLOCKS(self) -> int:
        """NVFP4 block scales per row."""
        return self.HIDDEN // SF_BLOCK

    @property
    def SF_K_TILES(self) -> int:
        """Scale atoms (four scale columns each) per row of the 128 x 4 layout."""
        return (self.SF_BLOCKS + SF_ATOM_COLS - 1) // SF_ATOM_COLS

    @property
    def CTAS_PER_SM(self) -> int:
        """Thread blocks of this form one SM holds: what one scheduling wave runs per SM."""
        return SM_THREADS // self.THREADS

    @property
    def RESIDUAL_MODE(self) -> str:
        return "add" if self.ADD else ("store" if self.STORE_RESIDUAL else "none")

    @property
    def LAUNCH_ARM(self) -> str:
        if self.PDL:
            return "wait_release" if self.PDL_RELEASE else "wait"
        return "release" if self.PDL_RELEASE else "plain"

    def __str__(self) -> str:
        outputs = "_".join(part for part in ("bf16" if self.BF16_OUT else None, self.QUANT) if part) or "nothing"
        try:
            eps = f"{eps_bits(self.EPS):08x}"
        except (struct.error, OverflowError, TypeError):
            eps = "invalid"
        return f"h{self.HIDDEN}_{self.RESIDUAL_MODE}_{outputs}_{self.LAUNCH_ARM}_eps{eps}"


def validate_hidden(hidden) -> int:
    """``hidden`` when it is structurally a row width of this kernel: an ``int`` (``TypeError`` otherwise) that is a multiple of 256 in
    256 .. 8192 (``ValueError`` otherwise).  Whether the entry SERVES a structurally valid width is ``SERVED_HIDDEN``."""
    if isinstance(hidden, bool) or not isinstance(hidden, int):
        raise TypeError(f"{NAME}: hidden must be an int, got {type(hidden).__name__}")
    if hidden < HIDDEN_STEP or hidden > MAX_HIDDEN or hidden % HIDDEN_STEP:
        raise ValueError(
            f"{NAME}: hidden must be a multiple of {HIDDEN_STEP} in {HIDDEN_STEP} .. {MAX_HIDDEN} (full warps of {VEC}-element threads, at most "
            f"{MAX_HIDDEN // VEC} threads); got {hidden}"
        )
    return hidden


def validate_eps(eps) -> float:
    """``eps`` as a ``float`` when the kernel's constant -- its fp32 value -- is finite and positive; ``ValueError`` otherwise."""
    ok = isinstance(eps, (int, float)) and not isinstance(eps, bool) and math.isfinite(eps) and eps > 0
    if ok:
        try:
            ok = 0 < eps_bits(float(eps)) < 0x7F800000
        except (struct.error, OverflowError):
            ok = False
    if not ok:
        raise ValueError(f"{NAME}: eps must be a finite positive float (its fp32 value is the kernel's constant), got {eps!r}")
    return float(eps)


def validate_cfg(cfg) -> Cfg:
    """``cfg`` when it is a form this kernel body has: ``TypeError`` for another type or a field of the wrong type, ``ValueError``
    naming the argument outside its vocabulary or the inconsistent pair.  Messages name the host entry's arguments (``hidden``,
    ``eps``, ``add``, ``store_residual``, ``bf16``, ``quant``, ``pdl``, ``pdl_release``)."""
    if not isinstance(cfg, Cfg):
        raise TypeError(f"{NAME}: cfg must be an add_norm_quant_decode_config.Cfg, got {type(cfg).__name__}")
    validate_hidden(cfg.HIDDEN)
    validate_eps(cfg.EPS)
    for name, value in (("add", cfg.ADD), ("store_residual", cfg.STORE_RESIDUAL), ("bf16", cfg.BF16_OUT), ("pdl", cfg.PDL), ("pdl_release", cfg.PDL_RELEASE)):
        if not isinstance(value, bool):
            raise TypeError(f"{NAME}: {name} must be a bool, got {type(value).__name__}")
    if cfg.ADD and cfg.STORE_RESIDUAL:
        raise ValueError(f"{NAME}: store_residual initializes the residual from x and cannot be combined with add")
    if cfg.QUANT is not None and (not isinstance(cfg.QUANT, str) or cfg.QUANT not in QUANT_FORMS):
        raise ValueError(f"{NAME}: quant must be None or one of {QUANT_FORMS}; got {cfg.QUANT!r}")
    if not cfg.BF16_OUT and cfg.QUANT is None:
        raise ValueError(f"{NAME}: the form writes no output: set bf16=True or a quant form")
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
    """NVFP4: rows of the 128 x 4 scale buffer that holds ``rows`` rows -- ``ceil(rows / 128) x 128``."""
    return (rows + SF_TILE_ROWS - 1) // SF_TILE_ROWS * SF_TILE_ROWS


def scale_byte_offset(cfg: Cfg, row: int, block: int) -> int:
    """NVFP4: the byte of block scale ``(row, block)`` in the scale buffer -- the host twin of the kernel's index.  Scale tiles of 128
    rows are laid out one after the other, ``SF_K_TILES`` atoms of 512 bytes each; inside a tile the byte is the 128 x 4 atom rule."""
    from cudnn.frost.tile_dsl.sf_layout import sf_atom_offset  # plain Python: no DSL import

    return (row // SF_TILE_ROWS) * (cfg.SF_K_TILES * SF_ATOM_BYTES) + sf_atom_offset(row % SF_TILE_ROWS, block)


def smem_table(cfg: Cfg) -> Tuple[Dict[str, Any], ...]:
    """The kernel's shared-memory buffers, one row each (the kernel docstring's table as data).  ONE buffer: an fp32 slot per warp,
    written once by lane 0 of its warp before the thread-block barrier and read once behind it by lanes ``0 .. WARPS - 1`` of every
    warp.  It is never refilled inside a launch, so nothing releases it."""
    return (
        dict(
            name="warp_sums",
            dtype="float32",
            elements=cfg.WARPS,
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
