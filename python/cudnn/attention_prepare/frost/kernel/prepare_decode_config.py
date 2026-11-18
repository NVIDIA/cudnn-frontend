# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config of the attention preparation DECODE kernel (``prepare_decode_f16.py``): the limits it serves, the form of one kernel body
as a frozen ``Cfg``, and the two tables a reader audits (the barrier ledger, the shared-memory table).  Importable without the CuTe
DSL and without torch (the host entry validates a ``Cfg`` before the DSL version gate runs).

The op, per token ``t`` of a decode step and per head row of ``HEAD_DIM`` bf16 elements (fp32 arithmetic on bf16 operands)::

    n      = bf16( x * rsqrt(mean(x^2) + eps) * (1 + weight) )         the RMSNorm of one head row, one rounding to bf16
    out[r]        = bf16( n[r] * cos[p, r] - n[r + HALF] * sin[p, r] )        r < HALF = ROTARY_DIM / 2: the rotary part
    out[r + HALF] = bf16( n[r + HALF] * cos[p, r] + n[r] * sin[p, r] )        p = the token's position for rotary index r
    out[e]        = n[e]                                                      e >= ROTARY_DIM: the pass-through part

    q_out[t, h]    = out of the query row  q[t, h]  with q_weight          gate_out[t, h] = gate[t, h]   (a bit copy)
    k_pool[slot_t] = out of the key row    k[t, h]  with k_weight          v_pool[slot_t] = v[t, h]      (a bit copy)

One thread block per (token, head) of ``HEADS`` query rows and ``KV_HEADS`` key / value rows; ``THREADS = HEAD_DIM / 2`` threads in
``WARPS`` full warps, thread ``i`` owns the two elements ``2 i``, ``2 i + 1`` of its row.  The row's sum of squares is ONE fixed tree
(a fused step over a thread's two elements, a butterfly over the 32 lanes of a warp with the offsets 16, 8, 4, 2, 1, a butterfly
over the warp sums); everything behind it is element-wise except the rotary part, where a thread reads its partner element from
the thread ``HALF / 2`` lanes away.

WHAT CHANGES BYTES: the geometry (the tree and the divisor), ``EPS``, ``POSITIONS`` with its ``SECTION`` (which position row a
rotary index reads; and which product of the second rotary half is the fused one -- ``n[r + HALF] * cos`` under ``"3d"``,
``n[r] * sin`` under ``"1d"``: the same bytes unless a product underflows).  WHAT NEVER DOES: the token count, a token's place in the call, the other tokens, ``PAGE_SIZE`` and the two
integer dtypes (they select addresses and load widths), ``PDL`` and ``PDL_RELEASE`` (control instructions).

THE FORM IS IN THE KERNEL SYMBOL.  ``str(cfg)`` is a compact text (``h24_4_d256_r64_3d_11_11_10_p64_i64_i64_plain_eps358637bd``:
the geometry, the position form with its section, the page size, the position and slot dtypes, the launch arm and the fp32 bits
of ``eps``); the DSL appends it to the kernel's name, so two KERNELS never share a symbol and the form is readable inside the first
100 characters.  (Two ``eps`` floats with one fp32 value are two compile keys of one kernel: the config keeps the caller's float,
the text and the kernel hold its fp32 value.)  ``repr(cfg)`` stays the field list.
"""

import math
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

NAME = "the attention preparation decode step"

WARP = 32
PAIR = 2  # elements per thread: one 32-bit access of each bf16 operand
ELEMENT_BYTES = 2  # bf16
#: the geometries the entry serves as ``(HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM)`` (each head width is its own byte class: the sum
#: tree and the divisor follow it).
SERVED_GEOMETRIES: Tuple[Tuple[int, int, int, int], ...] = ((24, 4, 256, 64),)
#: the page sizes (rows of one page of both pools) the entry serves; a compile-time constant of the kernel (the in-kernel divide).
SERVED_PAGE_SIZES: Tuple[int, ...] = (64, 16)
#: the position forms: one position per token, or three rows (temporal, height, width) selected per rotary index.
POSITION_FORMS: Tuple[str, ...] = ("1d", "3d")
#: the integer dtypes of ``positions`` and ``slot_mapping``, by name (so that this module imports without torch).
INT_DTYPES: Tuple[str, ...] = ("int32", "int64")
INT_BYTES: Dict[str, int] = {"int32": 4, "int64": 8}
#: the served token ceiling per call.  Above the thread blocks one scheduling wave holds on every served part divided by the
#: programs of a token (``CTAS_PER_SM`` x the SM count / ``PROGRAMS``), so the suite's largest cell crosses wave boundaries.
MAX_ROWS = 1024
#: threads one SM holds: ``SM_THREADS // THREADS`` thread blocks co-reside.
SM_THREADS = 2048


def eps_bits(eps: float) -> int:
    """The fp32 bit pattern of ``eps`` -- the constant the kernel adds (``0x358637BD`` for ``1e-6``)."""
    return struct.unpack("<I", struct.pack("<f", eps))[0]


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class AttentionPrepareGeometry:
    """The geometry of one attention layer's preparation: ``HEADS`` query heads (each with a gate row), ``KV_HEADS`` key / value
    heads, ``HEAD_DIM`` elements per head row, and the leading ``ROTARY_DIM`` elements of a row that are rotated (two halves of
    ``ROTARY_DIM / 2``)."""

    HEADS: int
    KV_HEADS: int
    HEAD_DIM: int
    ROTARY_DIM: int

    def __str__(self) -> str:
        return f"h{self.HEADS}_{self.KV_HEADS}_d{self.HEAD_DIM}_r{self.ROTARY_DIM}"


@dataclass(frozen=True)
class Cfg:
    """The form of one kernel body: every field is a compile-time constant and the whole config is the compile key.  ``str`` is the
    compact form text the kernel symbol carries (the module docstring); ``repr`` is the field list."""

    GEOMETRY: AttentionPrepareGeometry
    EPS: float
    POSITIONS: str
    SECTION: Optional[Tuple[int, int, int]]
    PAGE_SIZE: int
    POSITION_DTYPE: str
    SLOT_DTYPE: str
    PDL: bool
    PDL_RELEASE: bool

    # --- derived (read-only) ---
    @property
    def HEADS(self) -> int:
        return self.GEOMETRY.HEADS

    @property
    def KV_HEADS(self) -> int:
        return self.GEOMETRY.KV_HEADS

    @property
    def HEAD_DIM(self) -> int:
        return self.GEOMETRY.HEAD_DIM

    @property
    def ROTARY_DIM(self) -> int:
        return self.GEOMETRY.ROTARY_DIM

    @property
    def HALF(self) -> int:
        """Rotary indices: element ``r`` rotates with element ``r + HALF``."""
        return self.ROTARY_DIM // 2

    @property
    def PROGRAMS(self) -> int:
        """Thread blocks per token: one per query head, then one per key / value head."""
        return self.HEADS + self.KV_HEADS

    @property
    def THREADS(self) -> int:
        return self.HEAD_DIM // PAIR

    @property
    def WARPS(self) -> int:
        return self.THREADS // WARP

    @property
    def SMEM_BYTES(self) -> int:
        """The kernel's one shared-memory buffer: one fp32 slot per warp."""
        return 4 * self.WARPS

    @property
    def CTAS_PER_SM(self) -> int:
        """Thread blocks of this form one SM holds: what one scheduling wave runs per SM."""
        return SM_THREADS // self.THREADS

    @property
    def POSITION_ROWS(self) -> int:
        return 3 if self.POSITIONS == "3d" else 1

    @property
    def LAUNCH_ARM(self) -> str:
        if self.PDL:
            return "release_wait" if self.PDL_RELEASE else "wait"
        return "release" if self.PDL_RELEASE else "plain"

    def __str__(self) -> str:
        positions = self.POSITIONS if self.SECTION is None else "_".join([str(self.POSITIONS), *(str(n) for n in self.SECTION)])
        try:
            eps = f"{eps_bits(self.EPS):08x}"
        except (struct.error, OverflowError, TypeError):
            eps = "invalid"
        dtypes = "_".join("i" + str(d)[3:] for d in (self.POSITION_DTYPE, self.SLOT_DTYPE))
        return f"{self.GEOMETRY}_{positions}_p{self.PAGE_SIZE}_{dtypes}_{self.LAUNCH_ARM}_eps{eps}"


def validate_geometry(geometry) -> AttentionPrepareGeometry:
    """``geometry`` when the entry serves it: an :class:`AttentionPrepareGeometry` of ``int`` fields (``TypeError`` otherwise) that
    are positive (``ValueError``), whose ``HEAD_DIM`` and ``ROTARY_DIM / 2`` are powers of two and which is one of
    ``SERVED_GEOMETRIES`` (``NotImplementedError`` otherwise, naming the rule or the served set)."""
    if not isinstance(geometry, AttentionPrepareGeometry):
        raise TypeError(f"{NAME}: geometry must be an AttentionPrepareGeometry, got {type(geometry).__name__}")
    fields = (("HEADS", geometry.HEADS), ("KV_HEADS", geometry.KV_HEADS), ("HEAD_DIM", geometry.HEAD_DIM), ("ROTARY_DIM", geometry.ROTARY_DIM))
    for name, value in fields:
        if not _is_int(value):
            raise TypeError(f"{NAME}: geometry.{name} must be an int, got {type(value).__name__}")
    for name, value in fields:
        if value < 1:
            raise ValueError(f"{NAME}: geometry.{name} must be positive, got {value}")
    if geometry.ROTARY_DIM % 2 or geometry.ROTARY_DIM > geometry.HEAD_DIM:
        raise ValueError(f"{NAME}: geometry.ROTARY_DIM must be even and at most HEAD_DIM (two rotary halves inside a head row), got {geometry.ROTARY_DIM}")
    if not _is_pow2(geometry.HEAD_DIM) or not _is_pow2(geometry.ROTARY_DIM // 2):
        raise NotImplementedError(
            f"{NAME} serves a HEAD_DIM and a ROTARY_DIM / 2 that are powers of two (full warps of two-element threads; the rotary "
            f"partner one lane group away); got HEAD_DIM {geometry.HEAD_DIM}, ROTARY_DIM {geometry.ROTARY_DIM}"
        )
    as_tuple = (geometry.HEADS, geometry.KV_HEADS, geometry.HEAD_DIM, geometry.ROTARY_DIM)
    if as_tuple not in SERVED_GEOMETRIES:
        served = ", ".join(str(g) for g in SERVED_GEOMETRIES)
        raise NotImplementedError(f"{NAME} serves the geometries (HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM) {served} (validated); got {as_tuple}")
    return geometry


def validate_eps(eps) -> float:
    """``eps`` as a ``float`` when it is a Python ``float`` or ``int`` whose fp32 value -- the kernel's constant -- is finite and
    positive; ``ValueError`` otherwise, naming the type it got (a value of another type is declined, never converted)."""
    ok = isinstance(eps, (int, float)) and not isinstance(eps, bool)
    if ok:
        try:  # an int too large for a float raises OverflowError from the first conversion: declined like every other value
            ok = math.isfinite(eps) and eps > 0 and 0 < eps_bits(float(eps)) < 0x7F800000
        except (struct.error, OverflowError):
            ok = False
    if not ok:
        raise ValueError(
            f"{NAME}: eps must be a finite positive float (a Python float or int; its fp32 value is the kernel's constant), "
            f"got {type(eps).__name__} {eps!r}"
        )
    return float(eps)


def validate_positions(positions, mrope_section, half: int) -> Tuple[str, Optional[Tuple[int, int, int]]]:
    """``(positions, section)`` when the pair is a position form: ``"1d"`` without a section, or ``"3d"`` with a section
    ``(t, h, w)`` of three positive ``int`` that sum to ``half`` (the rotary indices: ``ROTARY_DIM / 2``); ``ValueError`` otherwise."""
    if not isinstance(positions, str) or positions not in POSITION_FORMS:
        raise ValueError(f"{NAME}: positions must be one of {POSITION_FORMS} (required: the forms are peers); got {positions!r}")
    if positions == "1d":
        if mrope_section is not None:
            raise ValueError(f"{NAME}: mrope_section belongs to positions='3d'; positions='1d' reads one position per token, pass None")
        return positions, None
    ok = isinstance(mrope_section, (tuple, list)) and len(mrope_section) == 3 and all(_is_int(n) and n > 0 for n in mrope_section)
    if not ok:
        raise ValueError(f"{NAME}: positions='3d' needs mrope_section = (t, h, w), three positive ints; got {mrope_section!r}")
    if sum(mrope_section) != half:
        raise ValueError(f"{NAME}: mrope_section must sum to ROTARY_DIM / 2 = {half} (one position row per rotary index); got {tuple(mrope_section)}")
    return positions, tuple(int(n) for n in mrope_section)


def validate_page_size(page_size) -> int:
    """``page_size`` when the entry serves it: a positive ``int`` that is a power of two (``ValueError`` otherwise: the kernel's
    divide is a shift) and one of ``SERVED_PAGE_SIZES`` (``NotImplementedError`` otherwise)."""
    if not _is_int(page_size) or page_size < 1:
        raise ValueError(f"{NAME}: page_size must be a positive int (the rows of one page of both pools; a compile-time constant), got {page_size!r}")
    if not _is_pow2(page_size):
        raise ValueError(f"{NAME}: page_size must be a power of two (the slot is divided by it with a shift), got {page_size}")
    if page_size not in SERVED_PAGE_SIZES:
        raise NotImplementedError(f"{NAME} serves page sizes {', '.join(str(p) for p in SERVED_PAGE_SIZES)} (validated); got {page_size}")
    return page_size


def validate_int_dtype(name: str, value) -> str:
    """``value`` when it names an integer dtype of ``positions`` / ``slot_mapping``: ``"int32"`` or ``"int64"``; ``ValueError``
    otherwise (a name, never a framework dtype object: the config imports without one)."""
    if not isinstance(value, str) or value not in INT_DTYPES:
        raise ValueError(f"{NAME}: {name} must be one of {INT_DTYPES} (a name; a compile-time load width), got {value!r}")
    return value


def validate_cfg(cfg) -> Cfg:
    """``cfg`` when it is a form this kernel body has: ``TypeError`` for another type or a flag that is not a ``bool``, then the
    declines of the field validators.  Messages name the host entry's arguments.  The structural couplings of the body are pinned
    here: the block is full warps of two-element threads with a power-of-two warp count (the row sum's second level), and the
    rotary part is exactly warp 0 (every lane of that warp has its partner ``HALF / 2`` lanes away)."""
    if not isinstance(cfg, Cfg):
        raise TypeError(f"{NAME}: cfg must be a prepare_decode_config.Cfg, got {type(cfg).__name__}")
    validate_geometry(cfg.GEOMETRY)
    validate_eps(cfg.EPS)
    positions, section = validate_positions(cfg.POSITIONS, cfg.SECTION, cfg.HALF)
    if section != cfg.SECTION:
        raise TypeError(f"{NAME}: SECTION must be a tuple of three ints or None, got {cfg.SECTION!r}")
    validate_page_size(cfg.PAGE_SIZE)
    validate_int_dtype("position_dtype", cfg.POSITION_DTYPE)
    validate_int_dtype("slot_dtype", cfg.SLOT_DTYPE)
    for name, value in (("pdl", cfg.PDL), ("pdl_release", cfg.PDL_RELEASE)):
        if not isinstance(value, bool):
            raise TypeError(f"{NAME}: {name} must be a bool, got {type(value).__name__}")
    if cfg.PDL_RELEASE and not cfg.PDL:
        raise ValueError(f"{NAME}: pdl_release releases dependents in front of this launch's own wait and needs pdl=True")
    if cfg.THREADS % WARP or not _is_pow2(cfg.WARPS) or cfg.WARPS > WARP or cfg.HALF != WARP:
        raise NotImplementedError(
            f"{NAME}: the kernel body needs HEAD_DIM / 2 threads in a power-of-two count of full warps (at most {WARP}) and a rotary "
            f"part of exactly one warp (ROTARY_DIM {2 * WARP}); got HEAD_DIM {cfg.HEAD_DIM}, ROTARY_DIM {cfg.ROTARY_DIM}"
        )
    return cfg


def validate_rows(rows) -> int:
    """``rows`` (the tokens of a call) when a call of that many is served: an ``int`` (``TypeError``), at least one
    (``ValueError``), at most ``MAX_ROWS`` (``NotImplementedError``)."""
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise TypeError(f"{NAME}: rows must be an int, got {type(rows).__name__}")
    if rows < 1:
        raise ValueError(f"{NAME}: rows must be at least 1, got {rows}")
    if rows > MAX_ROWS:
        raise NotImplementedError(f"{NAME} serves at most {MAX_ROWS} rows (tokens) per call; got {rows}")
    return rows


def position_row_of(cfg: Cfg, rotary_index: int) -> int:
    """The position row rotary index ``r`` reads: row 0 under ``"1d"``; under ``"3d"`` with the section ``(t, h, w)`` row 1 when
    ``r % 3 == 1 and r < 3 h``, row 2 when ``r % 3 == 2 and r < 3 w``, row 0 otherwise.  The host twin of the kernel's select."""
    if cfg.SECTION is None:
        return 0
    _, h, w = cfg.SECTION
    if rotary_index % 3 == 1 and rotary_index < 3 * h:
        return 1
    if rotary_index % 3 == 2 and rotary_index < 3 * w:
        return 2
    return 0


def position_row_masks(cfg: Cfg) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """``((h0_mask, w0_mask), (h1_mask, w1_mask))``: per element of a thread's pair (rotary index ``2 j`` and ``2 j + 1`` for lane group index
    ``j = lane % (HALF / 2)``) the bit masks over ``j`` of the indices that read position row 1 and position row 2
    (:func:`position_row_of`).  Compile-time constants of the kernel: the select needs no division."""
    masks = []
    for element in range(PAIR):
        h = sum(1 << j for j in range(cfg.HALF // PAIR) if position_row_of(cfg, PAIR * j + element) == 1)
        w = sum(1 << j for j in range(cfg.HALF // PAIR) if position_row_of(cfg, PAIR * j + element) == 2)
        masks.append((h, w))
    return masks[0], masks[1]


def smem_table(cfg: Cfg) -> Tuple[Dict[str, Any], ...]:
    """The kernel's shared-memory buffers, one row each (the kernel docstring's table as data).  ONE buffer: an fp32 slot per warp,
    written once by lane 0 of its warp before the thread-block barrier and read once behind it by EVERY lane (slot ``lane mod
    WARPS``).  It is never refilled inside a launch, so nothing releases it."""
    return (
        dict(
            name="warp_sums",
            dtype="float32",
            elements=cfg.WARPS,
            bytes=cfg.SMEM_BYTES,
            lane_stride_bytes=4,
            writers="lane 0 of each warp, before the thread-block barrier",
            readers=f"every lane, slot lane mod {cfg.WARPS}, behind the thread-block barrier",
            refilled=False,
        ),
    )


def barrier_ledger(cfg: Cfg) -> Dict[str, int]:
    """The one thread-block barrier: every thread of the block takes part and every thread issues it exactly once, so the two
    counts are equal by construction (``participants == issuing_lanes == THREADS``).  No other synchronization object exists."""
    return dict(participants=cfg.THREADS, issuing_lanes=WARP * cfg.WARPS)
