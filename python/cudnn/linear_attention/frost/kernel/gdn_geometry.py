# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-rank geometry of a Gated DeltaNet sub-layer that the fused GDN decode (``gdn_decode_fused_f16.py``) and verify
(``gdn_verify_fused_f16.py``) step kernels serve: the K-head and V-head counts and the model hidden width of ONE tensor-parallel rank,
``Geometry(K_HEADS, V_HEADS, HIDDEN)``.  Importable without the CuTe DSL (the configs and the host entries read it before any DSL
import), so nothing here imports ``cutlass``.

What is geometry and what is not.  The recurrence, the lane map, the bank map, the shared-memory budget and the barrier tables of both
kernels depend on the head dim (``HEAD_DIM`` 128) and the conv width (``CONV_WIDTH`` 4) only; those two stay pinned module constants
(another value is the host entries' typed decline).  The head counts and the hidden width reach the kernels through the compile-time
``Cfg`` (``Cfg.GEOM``) -- the QKV channel layout (``QKV_CHANNELS``, ``K_CHANNEL0``, ``V_CHANNEL0``), the output row (``OUT_ROW``), the
V-heads-per-K-head ratio (``VK_RATIO``: the K head of V head ``h`` is ``h // VK_RATIO``, the reader-ticket limit ``VK_RATIO x
CTAS_PER_HEAD - 1``), the BA projection's chunk counts (``BA_CHUNKS_DECODE`` of 1024 elements for the 128-thread decode partition,
``BA_CHUNKS_VERIFY`` of 2048 elements for the 256-virtual-thread verify partition, whose last chunk is a HALF chunk when ``HIDDEN %
2048 == 1024``: ``ba_verify_half_live``) -- so every form is compiled per geometry (the ``Cfg`` is the compile key) and the default
geometry's renderings are byte-identical to the ones built before the geometry was a parameter (every site reads the same Python
integers at trace time).

Envelope (``ValueError`` otherwise, at construction): ``1 <= K_HEADS <= 16`` (the reader-ticket layout ``counts[row * 512 + kh * 32]``
holds 16 K heads per row), ``1 <= V_HEADS <= 128`` with ``V_HEADS % K_HEADS == 0`` (every K head serves a whole group of V heads),
``HIDDEN`` a positive multiple of 1024 (the decode BA partition's chunk; the verify partition's half chunk).  The default geometry
(``DEFAULT_GEOMETRY``: 16 K heads, 48 V heads, hidden 5120 -- the dense 27B-class model's GDN sub-layer at tensor-parallel rank 1 the
kernels were written and validated against) keeps every existing caller and test unchanged; a tensor-parallel shard of a larger dense
model is a geometry of its own (e.g. hidden 8192 with 1 K head and 8 V heads per rank at 16-way tensor parallelism, 2 / 16 at 8-way,
4 / 32 at 4-way; hidden 4096 with 4 K heads and 16 V heads at 4-way).
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Optional

# --- pinned: every slab / box / lane map / norm term of both kernels (another value is the host entries' typed decline) ---
HEAD_DIM = 128
CONV_WIDTH = 4
# the reader ticket of a (row, K head) conv history: counts[row * TICKET_STRIDE + kh * 32] -- the reference layout; holds MAX_K_HEADS
TICKET_STRIDE = 512
_TICKET_STRIDE = TICKET_STRIDE  # read by the Geometry.TICKET_STRIDE property (same name as the module constant)
MAX_K_HEADS = TICKET_STRIDE // 32  # 16
MAX_V_HEADS = 128
# the BA projection partitions: decode = 128 threads x 8 elements per chunk; verify = 256 virtual threads x 8 elements per chunk,
# every lane of the BA warpgroup carrying virtual threads ``lane`` (the chunk's first half) and ``lane + 128`` (its second half)
BA_DECODE_CHUNK = 1024
BA_VERIFY_CHUNK = 2048
BA_VERIFY_HALF = BA_VERIFY_CHUNK // 2  # 1024
HIDDEN_GRANULE = BA_DECODE_CHUNK  # = BA_VERIFY_HALF: a hidden width is a whole number of decode chunks and of verify half chunks


@dataclass(frozen=True)
class Geometry:
    """The head counts and hidden width of one tensor-parallel rank's GDN sub-layer (see the module docstring for the envelope)."""

    K_HEADS: int
    V_HEADS: int
    HIDDEN: int

    def __post_init__(self):
        msg = geometry_error(self.K_HEADS, self.V_HEADS, self.HIDDEN)
        if msg is not None:
            raise ValueError(f"Geometry: {msg}")
        for name in ("K_HEADS", "V_HEADS", "HIDDEN"):  # any Integral (a numpy int from a config) is held as a plain int: one hash, one repr
            object.__setattr__(self, name, int(getattr(self, name)))

    # --- the QKV slab [rows, QKV_CHANNELS] = Q (K_HEADS x 128) | K (K_HEADS x 128) | V (V_HEADS x 128) ---
    @property
    def QKV_CHANNELS(self) -> int:
        return 2 * self.K_HEADS * HEAD_DIM + self.V_HEADS * HEAD_DIM

    @property
    def K_CHANNEL0(self) -> int:
        """The first K channel of the QKV slab."""
        return self.K_HEADS * HEAD_DIM

    @property
    def V_CHANNEL0(self) -> int:
        """The first V channel of the QKV slab."""
        return 2 * self.K_HEADS * HEAD_DIM

    @property
    def OUT_ROW(self) -> int:
        """Outputs per token row (``V_HEADS x 128``)."""
        return self.V_HEADS * HEAD_DIM

    @property
    def VK_RATIO(self) -> int:
        """V heads per K head: the K head of V head ``h`` is ``h // VK_RATIO``; ``VK_RATIO x CTAS_PER_HEAD`` CTAs read one K head's conv
        history per row (the reader-ticket limit is one less)."""
        return self.V_HEADS // self.K_HEADS

    @property
    def BA_CHUNKS_DECODE(self) -> int:
        """Chunks of the decode BA partition: thread ``t`` of 128 sums elements ``chunk * 1024 + 8 t + j`` of every chunk."""
        return self.HIDDEN // BA_DECODE_CHUNK

    @property
    def BA_CHUNKS_VERIFY(self) -> int:
        """Chunks of the verify BA partition (2048 elements each; the last one a half chunk when ``HIDDEN % 2048 == 1024``): virtual
        thread ``v`` of 256 sums elements ``chunk * 2048 + 8 v + j`` while ``chunk * 2048 + 8 v < HIDDEN``."""
        return (self.HIDDEN + BA_VERIFY_CHUNK - 1) // BA_VERIFY_CHUNK

    def ba_verify_half_live(self, chunk: int) -> bool:
        """Whether verify chunk ``chunk`` has its SECOND half (the ``lane + 128`` virtual threads, elements ``chunk * 2048 + 1024 ..``):
        every chunk but a trailing half chunk.  The ONE predicate of the second virtual thread's weight loads, hidden loads and dot chains
        (the kernel reads it at every one of those sites, so an input is never loaded without being accumulated)."""
        return chunk * BA_VERIFY_CHUNK + BA_VERIFY_HALF < self.HIDDEN

    @property
    def BA_VERIFY_HALF_LIVE(self) -> tuple:
        """``ba_verify_half_live`` per chunk, as a tuple."""
        return tuple(self.ba_verify_half_live(c) for c in range(self.BA_CHUNKS_VERIFY))

    @property
    def TICKET_STRIDE(self) -> int:
        """``counts`` entries per row (the per-(row, K head) reader ticket sits at ``kh * 32``)."""
        return _TICKET_STRIDE

    def describe(self) -> str:
        """The geometry in words, for the typed declines' messages."""
        return f"{self.K_HEADS} K heads, {self.V_HEADS} V heads x {HEAD_DIM}, hidden {self.HIDDEN}"


def _is_int(value) -> bool:
    """An integer in the envelope's sense: any ``numbers.Integral`` (a Python or numpy int), never a bool."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def geometry_error(k_heads, v_heads, hidden) -> Optional[str]:
    """Why ``(k_heads, v_heads, hidden)`` is outside the served envelope (``None`` = served): a positive int each, at most ``MAX_K_HEADS``
    K heads (the reader-ticket layout), at most ``MAX_V_HEADS`` V heads, a whole number of V heads per K head, the hidden width a
    multiple of ``HIDDEN_GRANULE``."""
    for name, value in (("K_HEADS", k_heads), ("V_HEADS", v_heads), ("HIDDEN", hidden)):
        if not _is_int(value) or value < 1:
            return f"{name} must be a positive int, got {value!r}"
    if k_heads > MAX_K_HEADS:
        return f"K_HEADS={k_heads} exceeds the {MAX_K_HEADS} K heads the reader-ticket layout (counts[row * {TICKET_STRIDE} + kh * 32]) holds"
    if v_heads > MAX_V_HEADS:
        return f"V_HEADS={v_heads} exceeds the served maximum of {MAX_V_HEADS}"
    if v_heads % k_heads:
        return f"V_HEADS={v_heads} must be a multiple of K_HEADS={k_heads} (every K head serves a whole group of V heads)"
    if hidden % HIDDEN_GRANULE:
        return f"HIDDEN={hidden} must be a multiple of {HIDDEN_GRANULE} (the BA projection's chunk granule)"
    return None


def geometry_fields_off(k_heads, v_heads, hidden) -> Dict[str, int]:
    """The fields of ``(k_heads, v_heads, hidden)`` the envelope refuses, by name (empty = served) -- for a decline that names them."""
    off: Dict[str, int] = {}
    if not _is_int(k_heads) or not 1 <= k_heads <= MAX_K_HEADS:
        off["K_HEADS"] = k_heads
    if not _is_int(v_heads) or not 1 <= v_heads <= MAX_V_HEADS or ("K_HEADS" not in off and v_heads % k_heads):
        off["V_HEADS"] = v_heads
    if not _is_int(hidden) or hidden < 1 or hidden % HIDDEN_GRANULE:
        off["HIDDEN"] = hidden
    return off


#: the geometry the kernels were written and validated against (the dense 27B-class model's GDN sub-layer at tensor-parallel rank 1):
#: the default of every ``Cfg``, entry and probe, so existing callers and tests are unchanged
DEFAULT_GEOMETRY = Geometry(K_HEADS=16, V_HEADS=48, HIDDEN=5120)
