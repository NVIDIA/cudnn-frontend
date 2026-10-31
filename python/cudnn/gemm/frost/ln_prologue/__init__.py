# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Configure and launch LayerNorm-prologue gated dual GEMMs.

Protocol: python/cudnn/frost/README.md#layernorm-prologue-dual-gemm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect
import os
from typing import Any, Optional

#: The LayerNorm width (= K) is streamed in 64-element stages; the pairing needs 128-channel value / gate blocks.
K_STAGE = 64
CHANNEL_BLOCK = 128
PAIR_TILE_ROWS = 256
#: Output ``k`` padding: the channel-major TMA store needs a 16-byte-aligned row stride (bf16 -> 8 elements).
K_PAD_MULTIPLE = 8
MIN_K_BLOCK = 8

DIAG_ARMS = ("", "one_pass_stats", "no_ln_norm", "no_ln", "no_epi_math", "ln_first_tile_only")
PASS_ORDERS = ("major", "interleaved")
PAIRINGS = ("columns", "accumulators")
A0_MAJORS = ("k", "mn")
A0_PROLOGUES = ("mn_major", "transpose")
#: The MN-major TMA box is 64 rows of ONE ``bi``: a pair tile's ``kb`` must be >= 64 (``n_res > 32``) in the accumulators arm.
MIN_N_RES_MN_MAJOR = 33
#: The same constraint on the LAUNCHED ``kb`` (a caller's ``k_block`` below 64 cannot serve the accumulators arm; the columns arm admits 8).
MIN_K_BLOCK_MN_MAJOR = 64
LN_PASS_FORMS = ("registers", "smem", "smem_rolled", "smem_rolled_pf", "smem_rolled_tree")
LN_PASS_FORMS_REORDERED = ("smem_rolled_tree",)
EPI_MATHS = ("scalar", "packed", "packed_hoist", "packed_biasfold")
EPI_MATHS_PACKED = ("packed", "packed_hoist", "packed_biasfold")
MIN_BLOCKS_PER_MP_DOMAIN = (0, 1)
LN_ONCE_FORMS = ("", "block", "passes")
RING_STAGES_BY_ARCH = {
    "sm107": {"columns": 9, "accumulators": 6},
    "sm100": {"columns": 6, "accumulators": 4},
}
#: The sm107 body's ring depths (the historical name; :func:`ring_stages` is the per-arch accessor).
RING_STAGES = RING_STAGES_BY_ARCH["sm107"]
SMEM_BUDGET_BYTES_BY_ARCH = {"sm107": 320 * 1024, "sm100": 227 * 1024}
TCGEN05_SMEM_DESC_LEGACY_ADDR_LIMIT = 262144
TCGEN05_SMEM_DESC_VERSION = 1
A_STAGE_BYTES = (PAIR_TILE_ROWS // 2) * K_STAGE * 2  #: one 128-row x 64-channel bf16 A stage per operand (16 KiB)
B_STAGE_BYTES = {
    "columns": (PAIR_TILE_ROWS // 2) * K_STAGE * 2,
    "accumulators": (PAIR_TILE_ROWS // 4) * K_STAGE * 2,
}  #: each CTA's half of the N tile per B operand
#: The rendered kernel bodies, one per arch family (``sm<NNN>_dual_gemm.py`` beside this file); ``build_ln_dual_gemm`` picks by name.
BODIES = {"sm107": "sm107_dual_gemm.py", "sm100": "sm100_dual_gemm.py"}
ARCHS_BY_CC = {(10, 0): "sm100", (10, 3): "sm100", (10, 7): "sm107"}
ARCH_GATE_CC = {"sm100": (10, 0), "sm107": (10, 7)}


def body_arch_for_cc(cc) -> str:
    """The body (``"sm100"`` / ``"sm107"``) for a device's compute capability; a typed decline (``NotImplementedError``) for a part neither body
    was rendered for -- never a silent fallback to another arch's budget."""
    key = tuple(int(x) for x in cc)
    if key not in ARCHS_BY_CC:
        served = ", ".join(f"{a}.{b}" for a, b in sorted(ARCHS_BY_CC))
        raise NotImplementedError(f"the LayerNorm-prologue dual GEMM is rendered for compute capabilities {served} only today; found {key[0]}.{key[1]}")
    return ARCHS_BY_CC[key]


def ring_stages(arch: str, pairing: str) -> int:
    """The ``arch`` body's operand ring depth for ``pairing`` (:data:`RING_STAGES_BY_ARCH`; a typed decline for an unknown arch or pairing)."""
    if arch not in RING_STAGES_BY_ARCH:
        raise ValueError(f"the LayerNorm-prologue dual GEMM has a rendered body for {sorted(RING_STAGES_BY_ARCH)} only, not {arch!r}")
    if pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
    return RING_STAGES_BY_ARCH[arch][pairing]


def max_k(arch: str, pairing: str) -> int:
    """The largest LayerNorm width the ``arch`` body serves for ``pairing``: every K stage of a tile stays resident in the ring, so
    ``K <= 64 x ring_stages`` -- sm107 576 (columns) / 384 (accumulators), sm100 384 / 256."""
    return K_STAGE * ring_stages(arch, pairing)


def body_path(arch: str = "sm107") -> str:
    """The rendered body for ``arch`` (a typed decline for an arch family without one -- never a silent fallback to another arch's body)."""
    if arch not in BODIES:
        raise ValueError(f"the LayerNorm-prologue dual GEMM has a rendered body for {sorted(BODIES)} only, not {arch!r}")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), BODIES[arch])


def smem_desc_wide_start_address_required(arch: str) -> bool:
    """Check whether the architecture needs wide shared-memory descriptor addresses."""
    if arch not in SMEM_BUDGET_BYTES_BY_ARCH:
        raise ValueError(f"the LayerNorm-prologue dual GEMM has a rendered body for {sorted(SMEM_BUDGET_BYTES_BY_ARCH)} only, not {arch!r}")
    return SMEM_BUDGET_BYTES_BY_ARCH[arch] > TCGEN05_SMEM_DESC_LEGACY_ADDR_LIMIT


def _tcgen05_smem_desc_build_is_versioned() -> bool:
    """Whether the installed DSL's ``Tcgen05SmemDesc.build`` takes ``version`` (nvidia-cutlass-dsl 4.8.0+; the released 4.7.0 does not)."""
    from cutlass.experimental.primitives import Tcgen05SmemDesc

    return "version" in inspect.signature(Tcgen05SmemDesc.build).parameters


def tcgen05_smem_desc_requirement_error(arch: str) -> Optional[str]:
    """Check the DSL descriptor-address support required by the architecture."""
    if not smem_desc_wide_start_address_required(arch) or _tcgen05_smem_desc_build_is_versioned():
        return None
    from cudnn.frost.buffers import cutedsl_state

    _, version = cutedsl_state()
    found = "unknown version" if version is None else " ".join(version)
    return (
        f"the LayerNorm-prologue dual GEMM's {arch} body places operand ring roots at or above {TCGEN05_SMEM_DESC_LEGACY_ADDR_LIMIT} B of SMEM "
        f"and needs the CuTe DSL's versioned tcgen05 SMEM-descriptor builder (Tcgen05SmemDesc.build with a 'version' parameter: "
        f"nvidia-cutlass-dsl >= 4.8.0); found {found}"
    )


class Tcgen05SmemDescBuilder:
    """Build architecture-compatible tcgen05 shared-memory descriptors."""

    __slots__ = ("arch", "path", "_decline")

    def __init__(self, arch: str, path: str, decline: Optional[str] = None):
        self.arch, self.path, self._decline = arch, path, decline

    def __call__(self, start_address, *, leading_byte_offset: int, stride_byte_offset: int, layout):
        if self.path == "declined":
            raise NotImplementedError(self._decline)
        from cutlass.experimental.primitives import Tcgen05SmemDesc

        if self.path == "versioned":
            return Tcgen05SmemDesc.build(
                start_address=start_address,
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                version=TCGEN05_SMEM_DESC_VERSION,
                layout=layout,
            )
        return Tcgen05SmemDesc.build(start_address=start_address, leading_byte_offset=leading_byte_offset, stride_byte_offset=stride_byte_offset, layout=layout)

    def __repr__(self) -> str:
        return f"Tcgen05SmemDescBuilder(arch={self.arch!r}, path={self.path!r})"


def tcgen05_smem_desc_builder(arch: str) -> Tcgen05SmemDescBuilder:
    """Resolve the ``arch`` body's ring-root descriptor builder against the installed DSL (:class:`Tcgen05SmemDescBuilder`): the versioned
    build() where it exists, the floor DSL's non-versioned build() on a body whose roots stay below the line, a declining builder otherwise."""
    if _tcgen05_smem_desc_build_is_versioned():
        return Tcgen05SmemDescBuilder(arch, "versioned")
    msg = tcgen05_smem_desc_requirement_error(arch)
    return Tcgen05SmemDescBuilder(arch, "legacy") if msg is None else Tcgen05SmemDescBuilder(arch, "declined", msg)


@dataclass(frozen=True)
class LnDualGemmParams:
    """Compile-time shape of one LayerNorm-prologue gated dual GEMM artifact (the template's ``FROST_TEMPLATE_PARAMS``)."""

    k: int = 128
    c_out: int = 128
    passes: int = 2
    has_bias: bool = True
    want_stats: bool = True
    has_mask: bool = True
    skip_masked_tiles: bool = True
    eps: float = 1e-5
    clc_stages: int = 2
    diag: str = ""
    pass_order: str = "major"
    pairing: str = "columns"
    #: operand 0's GMEM major (:data:`A0_MAJORS`): "mn" = its rows are MN-major (the K3 ``y`` rows: the inner index contiguous), "k" = K-major.
    a0_major: str = "k"
    #: the MN-major LayerNorm form (:data:`A0_PROLOGUES`; accumulators arm only).  Performance only: both forms compute the same function.
    a0_prologue: str = "mn_major"
    affine_fold: bool = False
    fuse_residual: bool = False
    #: the TMA warp prefetches the tile's A boxes into L2 before waiting for a ring slot (performance only).
    prefetch_a_l2: bool = False
    ln_pass_form: str = "registers"
    #: :data:`EPI_MATHS` -- the gated epilogue's arithmetic form (performance only).
    epi_math: str = "scalar"
    min_blocks_per_mp: int = 0
    ln_once: str = ""
    ln_once_b_stages: int = 0

    @property
    def dual(self) -> bool:
        return self.pairing == "accumulators"

    @property
    def n_out(self) -> int:
        """The GEMM N of the artifact: 2 c_out (value | gate interleaved) in the columns arm, c_out per GEMM in the accumulators arm."""
        return self.c_out if self.dual else 2 * self.c_out

    @property
    def num_a_operands(self) -> int:
        return 2 if self.dual else 1

    @property
    def num_b_operands(self) -> int:
        return 2 if self.dual else 1

    @property
    def k_stages(self) -> int:
        return self.k // K_STAGE

    @property
    def n_tiles(self) -> int:
        """GEMM N tiles per row block: ``c_out / 128`` in BOTH arms (the columns arm's 256-wide tile holds one value block + one gate block; the
        accumulators arm's two GEMMs each run a 128-wide tile)."""
        return self.c_out // CHANNEL_BLOCK


def ln_once_n_tiles_per_item(p: LnDualGemmParams) -> int:
    """N tiles one work item spans under the lever: every one (``"block"``), one (``"passes"``); 1 when the lever is off."""
    return p.n_tiles if p.ln_once == "block" else 1


def ln_once_tiles_per_item(p: LnDualGemmParams) -> int:
    """GEMM tiles per work item: ``passes x n_tiles_per_item`` under the lever (the LayerNorm multiplicity it removes), 1 when off."""
    return p.passes * ln_once_n_tiles_per_item(p) if p.ln_once else 1


def _ln_once_a_slots_at(p: LnDualGemmParams, arch: str, b_stages: int) -> int:
    """The slot count the self-referential predicate gives with a B ring of ``b_stages`` stages (the shipped ring's bytes are the budget)."""
    ring = ring_stages(arch, p.pairing)
    off_bytes = ring * (p.num_a_operands * A_STAGE_BYTES + p.num_b_operands * B_STAGE_BYTES[p.pairing])
    b_ring_bytes = b_stages * p.num_b_operands * B_STAGE_BYTES[p.pairing]
    region_bytes = p.num_a_operands * p.k_stages * A_STAGE_BYTES
    return 2 if 2 * region_bytes + b_ring_bytes <= off_bytes else 1


def ln_once_b_stages_effective(p: LnDualGemmParams, arch: str = "sm107") -> int:
    """Return the effective B-ring depth for LayerNorm reuse."""
    ring = ring_stages(arch, p.pairing)
    want = int(p.ln_once_b_stages or 0)
    if not p.ln_once or want <= 0 or want >= ring:
        return ring
    if _ln_once_a_slots_at(p, arch, ring) == 2 or _ln_once_a_slots_at(p, arch, want) != 2:
        return ring
    return want


def ln_once_a_slots(p: LnDualGemmParams, arch: str = "sm107") -> int:
    """Return the resident A-slot count for LayerNorm reuse."""
    if not p.ln_once:
        return 0
    return _ln_once_a_slots_at(p, arch, ln_once_b_stages_effective(p, arch))


_SM100_UNALLOCATABLE_MIN_K = 4 * K_STAGE


def sm100_register_form_unallocatable(p: LnDualGemmParams, arch: str) -> bool:
    """Reject SM100 register forms that ptxas cannot allocate."""
    if arch != "sm100" or p.ln_pass_form != "registers" or p.epi_math == "scalar" or p.k < _SM100_UNALLOCATABLE_MIN_K:
        return False
    if p.dual:
        return True
    return bool(p.want_stats) and p.epi_math in ("packed_hoist", "packed_biasfold")


def validate_ln_dual_gemm_params(p: LnDualGemmParams, arch: str = "sm107") -> None:
    """Typed declines for every value a caller can trip (never a module-level assert -- engine contract S 7); ``arch`` names the body whose
    ring bounds K (:func:`max_k`): the same record can be valid for the Rubin body and a decline for the Blackwell one."""
    if not isinstance(p, LnDualGemmParams):
        raise TypeError(f"expected LnDualGemmParams, got {type(p).__name__}")
    if arch not in BODIES:
        raise ValueError(f"the LayerNorm-prologue dual GEMM has a rendered body for {sorted(BODIES)} only, not {arch!r}")
    if p.k <= 0 or p.k % K_STAGE != 0:
        raise ValueError(f"the LayerNorm width K must be a positive multiple of {K_STAGE} (one TMA / MMA stage), got {p.k}")
    if p.c_out <= 0 or p.c_out % CHANNEL_BLOCK != 0:
        raise ValueError(f"c_out must be a positive multiple of {CHANNEL_BLOCK} (one value block + one gate block per 256-wide N tile), got {p.c_out}")
    if p.passes < 1:
        raise ValueError(f"passes must be >= 1, got {p.passes}")
    if not (p.eps > 0.0):
        raise ValueError(f"eps must be > 0, got {p.eps}")
    if not (1 <= p.clc_stages <= 8):
        raise ValueError(f"clc_stages must be in [1, 8], got {p.clc_stages}")
    if p.diag not in DIAG_ARMS:
        raise ValueError(f"diag must be one of {DIAG_ARMS}, got {p.diag!r}")
    if p.pass_order not in PASS_ORDERS:
        raise ValueError(f"pass_order must be one of {PASS_ORDERS}, got {p.pass_order!r}")
    if p.pass_order == "interleaved" and (p.passes & (p.passes - 1)):
        raise ValueError(f"pass_order='interleaved' folds the pass into the tile index with a shift and a mask: passes must be a power of two, got {p.passes}")
    if p.pairing not in PAIRINGS:
        raise ValueError(f"pairing must be one of {PAIRINGS}, got {p.pairing!r}")
    if p.a0_major not in A0_MAJORS:
        raise ValueError(f"a0_major must be one of {A0_MAJORS}, got {p.a0_major!r}")
    if p.a0_prologue not in A0_PROLOGUES:
        raise ValueError(f"a0_prologue must be one of {A0_PROLOGUES}, got {p.a0_prologue!r}")
    if p.ln_pass_form not in LN_PASS_FORMS:
        raise ValueError(f"ln_pass_form must be one of {LN_PASS_FORMS}, got {p.ln_pass_form!r}")
    if p.epi_math not in EPI_MATHS:
        raise ValueError(f"epi_math must be one of {EPI_MATHS}, got {p.epi_math!r}")
    if isinstance(p.min_blocks_per_mp, bool) or p.min_blocks_per_mp not in MIN_BLOCKS_PER_MP_DOMAIN:
        raise ValueError(
            f"min_blocks_per_mp must be 0 (no .minnctapersm directive: the rendering as shipped, the register split dropped by ptxas) or 1 (the hint "
            f"under which ptxas honours the body's setmaxnreg split); 2 or more would cap the allocation at 65536 / (2 x 384) = 80 registers with the "
            f"split impossible -- got {p.min_blocks_per_mp!r}"
        )
    if p.ln_once not in LN_ONCE_FORMS:
        raise ValueError(f"ln_once must be one of {LN_ONCE_FORMS}, got {p.ln_once!r}")
    if isinstance(p.ln_once_b_stages, bool) or not isinstance(p.ln_once_b_stages, int) or p.ln_once_b_stages < 0:
        raise ValueError(f"ln_once_b_stages must be a non-negative integer (0 = the body's shipped B ring), got {p.ln_once_b_stages!r}")
    if p.ln_once_b_stages and not p.ln_once:
        raise ValueError(
            f"ln_once_b_stages={p.ln_once_b_stages} shallows the B ring under the LayerNorm-once item only (the item's resident A region is what the freed "
            "bytes buy); with ln_once='' the ring is the shipped one -- leave the field 0"
        )
    if p.ln_once_b_stages and p.pairing in PAIRINGS and arch in BODIES and p.ln_once_b_stages > ring_stages(arch, p.pairing):
        raise ValueError(
            f"ln_once_b_stages={p.ln_once_b_stages} is deeper than the {arch} body's shipped {ring_stages(arch, p.pairing)}-deep ring (pairing {p.pairing!r}): "
            "the field can only shallow the ring"
        )
    if p.pairing in PAIRINGS and p.k > max_k(arch, p.pairing):
        raise ValueError(
            f"the {arch} body keeps every K stage of a tile resident in its {ring_stages(arch, p.pairing)}-deep operand ring (pairing {p.pairing!r}): "
            f"K must be <= {max_k(arch, p.pairing)}, got {p.k}"
        )
    if sm100_register_form_unallocatable(p, arch):
        raise ValueError(
            f"the sm100 body cannot render K={p.k} with ln_pass_form='registers' and epi_math={p.epi_math!r}"
            f"{' (want_stats=True)' if not p.dual else ''} for pairing {p.pairing!r}: ptxas fails to allocate the LayerNorm row held in registers plus the "
            "packed epilogue at the flat 168-register entry count (C7600 'Register allocation failed with register count of 168' on sm_100a; the sm107 body "
            "renders the same record with hundreds of spilled words).  Use a re-read LayerNorm form (ln_pass_form='smem_rolled' / 'smem_rolled_pf': the "
            "measured picks at this width on both architectures) or the scalar epilogue"
        )
    if p.ln_once:
        if ln_once_tiles_per_item(p) < 2:
            raise ValueError(
                f"ln_once={p.ln_once!r} makes a work item of ONE GEMM tile here (passes {p.passes} x {ln_once_n_tiles_per_item(p)} N tile(s) at c_out {p.c_out}, "
                f"pairing {p.pairing!r}): the LayerNorm already runs once per row block and the resident A region would be dead weight -- leave the lever off"
            )
        if p.pass_order != "major":
            raise ValueError(
                f"ln_once={p.ln_once!r} folds the pass into the work item's sub-tile loop; pass_order={p.pass_order!r} would fold it a second time into "
                "the scheduler's M axis and every tile of pass 1 would read pass 0's rows -- use pass_order='major'"
            )
        if p.prefetch_a_l2:
            raise ValueError(
                f"ln_once={p.ln_once!r} loads a row block's A operand(s) once per item; the per-tile L2 prefetch (prefetch_a_l2) has no tile to prefetch for"
            )
        if (p.passes & (p.passes - 1)) or (p.n_tiles & (p.n_tiles - 1)):
            raise ValueError(
                f"ln_once={p.ln_once!r} decodes the item's sub-tile with a shift and a mask: passes ({p.passes}) and the N tiles per row block "
                f"({p.n_tiles} = c_out / {CHANNEL_BLOCK}) must be powers of two"
            )
        if p.diag == "ln_first_tile_only":
            raise ValueError(
                "diag='ln_first_tile_only' bounds the LayerNorm-once lever from the per-tile body; under ln_once the LayerNorm already runs once per item -- nothing to relieve"
            )
    if p.dual:
        if p.has_mask or p.skip_masked_tiles:
            raise ValueError("pairing='accumulators' has no output mask: has_mask and skip_masked_tiles must be False")
        if p.passes != 1:
            raise ValueError(f"pairing='accumulators' has one output: passes must be 1, got {p.passes}")
        if p.a0_major != "mn":
            raise ValueError("pairing='accumulators' is written for an MN-major operand 0 (the channel-major y rows); a0_major must be 'mn'")
        if p.ln_pass_form != "registers" and p.a0_prologue != "mn_major":
            raise ValueError(
                f"ln_pass_form={p.ln_pass_form!r} re-reads operand 0's MN-major stage in every LayerNorm pass; a0_prologue='transpose' overwrites the "
                "channel rows other lanes still read (the register form reads everything before its LN_SYNC barrier) -- use a0_prologue='mn_major'"
            )
    else:
        if p.a0_major != "k":
            raise ValueError("pairing='columns' (the single K-major operand) takes a0_major='k'")
        if p.affine_fold or p.fuse_residual:
            raise ValueError("affine_fold / fuse_residual belong to the accumulators arm (pairing='accumulators')")


@dataclass(frozen=True)
class RowBlockGeometry:
    """The pair-tile decomposition of ``(bn, n_res)`` rows -- every number the kernel's host call needs."""

    n_res: int
    bn: int
    n_pad: int  #: output k extent (a multiple of 8)
    kb: int  #: k's per 256-row pair tile (power of two)
    lkb: int  #: log2(kb)
    r_i: int  #: bi's per pair tile (= 256 // kb)
    n_kb: int  #: k blocks per bi row
    n_ib: int  #: bi blocks (ceil(bn / r_i))
    kbc: int  #: k's per CTA box (min(kb, 128))
    rc: int  #: bi's per CTA box (128 // kbc)

    @property
    def pair_tiles(self) -> int:
        return self.n_ib * self.n_kb

    @property
    def m_sched(self) -> int:
        """The flat M the template's scheduler walks: one 256-row pair tile per (bi block, k block)."""
        return self.pair_tiles * PAIR_TILE_ROWS

    @property
    def rows_computed(self) -> int:
        return self.m_sched

    @property
    def rows_valid(self) -> int:
        return self.bn * self.n_res


def default_k_block(n_res: int) -> int:
    """The rule: ``kb = 256`` for ``n_res >= 256`` (one bi per pair tile), else the next power of two >= max(n_res, 8)."""
    n_res = int(n_res)
    if n_res >= PAIR_TILE_ROWS:
        return PAIR_TILE_ROWS
    kb = MIN_K_BLOCK
    while kb < n_res:
        kb *= 2
    return kb


def validate_k_block(k_block) -> int:
    """A caller's ``k_block``: a power of two in ``[MIN_K_BLOCK, PAIR_TILE_ROWS]`` (ValueError names the domain -- a malformed declaration)."""
    if isinstance(k_block, bool) or not isinstance(k_block, int):
        raise ValueError(f"k_block must be an int power of two in [{MIN_K_BLOCK}, {PAIR_TILE_ROWS}], got {k_block!r}")
    if k_block < MIN_K_BLOCK or k_block > PAIR_TILE_ROWS or k_block & (k_block - 1):
        raise ValueError(f"k_block must be a power of two in [{MIN_K_BLOCK}, {PAIR_TILE_ROWS}] (k's per 256-row pair tile), got {k_block}")
    return k_block


def row_block_geometry(n_res: int, bn: int, k_block: Optional[int] = None) -> RowBlockGeometry:
    """The pair-tile decomposition: ``kb`` = :func:`default_k_block` (``None``) or the caller's :func:`validate_k_block`-checked ``k_block``
    (a performance knob -- the same bits under every admitted value; the accumulators arm additionally needs ``kb >= MIN_K_BLOCK_MN_MAJOR``)."""
    n_res, bn = int(n_res), int(bn)
    if n_res < 1 or bn < 1:
        raise ValueError(f"n_res and bn must be >= 1, got n_res={n_res}, bn={bn}")
    kb = default_k_block(n_res) if k_block is None else validate_k_block(k_block)
    lkb = kb.bit_length() - 1
    r_i = PAIR_TILE_ROWS // kb
    n_kb = -(-n_res // kb)
    n_ib = -(-bn // r_i)
    kbc = min(kb, PAIR_TILE_ROWS // 2)
    rc = (PAIR_TILE_ROWS // 2) // kbc
    n_pad = -(-n_res // K_PAD_MULTIPLE) * K_PAD_MULTIPLE
    return RowBlockGeometry(n_res=n_res, bn=bn, n_pad=n_pad, kb=kb, lkb=lkb, r_i=r_i, n_kb=n_kb, n_ib=n_ib, kbc=kbc, rc=rc)


def interleave_value_gate(w_value, w_gate):
    """Interleave 128-channel value and gate weight blocks."""
    import torch

    if w_value.shape != w_gate.shape or w_value.ndim != 2:
        raise ValueError(f"value / gate weights must both be [c_out, K], got {tuple(w_value.shape)} and {tuple(w_gate.shape)}")
    c_out, k = w_value.shape
    if c_out % CHANNEL_BLOCK != 0:
        raise ValueError(f"c_out must be a multiple of {CHANNEL_BLOCK}, got {c_out}")
    v = w_value.reshape(c_out // CHANNEL_BLOCK, CHANNEL_BLOCK, k)
    g = w_gate.reshape(c_out // CHANNEL_BLOCK, CHANNEL_BLOCK, k)
    return torch.cat([v, g], dim=1).reshape(2 * c_out, k)


def interleave_value_gate_bias(b_value, b_gate):
    """The ``[c_out]`` value and gate biases in the interleaved ``[2 * c_out]`` order of :func:`interleave_value_gate`."""
    return interleave_value_gate(b_value.reshape(-1, 1), b_gate.reshape(-1, 1)).reshape(-1)


@dataclass(frozen=True)
class LnDualGemmPlan:
    """One compiled artifact.  Plan-time only; :func:`run_ln_dual_gemm` launches."""

    params: LnDualGemmParams
    module: Any
    launch: Any  # the cute-compiled ``_host``


@lru_cache(maxsize=None)
def build_ln_dual_gemm(params: LnDualGemmParams, arch: str = "sm107") -> LnDualGemmPlan:
    """Load the ``arch`` body (:data:`BODIES`; :func:`body_arch_for_cc` names it from the device) specialized for ``params`` and cute-compile it."""
    from cudnn.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
    from cudnn.frost.template_loader import load_template

    if arch not in BODIES:
        raise ValueError(f"the LayerNorm-prologue dual GEMM has a rendered body for {sorted(BODIES)} only, not {arch!r}")
    msg = (
        cutedsl_requirement_error("the LayerNorm-prologue dual GEMM")
        or cutedsl_arch_requirement_error(ARCH_GATE_CC[arch])
        or tcgen05_smem_desc_requirement_error(arch)
    )
    if msg:
        raise NotImplementedError(msg)
    validate_ln_dual_gemm_params(params, arch)
    mod = load_template(body_path(arch), params, tag="ln_prologue_dual_gemm")
    return LnDualGemmPlan(params=params, module=mod, launch=mod.compile())


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def run_ln_dual_gemm(
    plan: LnDualGemmPlan,
    z,  # [bn * n_res, K] bf16 contiguous (pair-major rows)
    w,  # [passes, 2 * c_out, K] bf16 contiguous, rows interleaved per pass (interleave_value_gate)
    out,  # [passes, c_out, bn, n_pad] bf16 contiguous (channel-major planes; k padded to n_pad)
    ln_weight,  # [K] bf16
    ln_bias,  # [K] bf16
    *,
    n_res: int,
    bn: int,
    bias=None,  # [passes, 2 * c_out] bf16 interleaved, required iff params.has_bias
    mask=None,  # [bn * n_res] bf16 per-row mask, required iff params.has_mask
    stats=None,  # [bn * n_res, 2] fp32 (mean, rstd), required iff params.want_stats
    stream,
    k_block: Optional[int] = None,  # k's per 256-row pair tile (row_block_geometry's rule when None; performance only, bitwise the same)
) -> RowBlockGeometry:
    """Launch.  No allocation, no conversion; every check is cheap host arithmetic.  Returns the geometry used."""
    import torch
    from cuda.bindings import driver as cuda

    p = plan.params
    _check(not p.dual, "this artifact was compiled with pairing='accumulators'; launch it through run_ln_dual_gemm_tail")
    g = row_block_geometry(n_res, bn, k_block)
    rows = g.rows_valid
    bf16 = torch.bfloat16
    _check(
        z.dtype == bf16 and z.is_contiguous() and tuple(z.shape) == (rows, p.k),
        f"z must be a contiguous bf16 [{rows}, {p.k}] tensor, got {tuple(z.shape)} {z.dtype}",
    )
    _check(
        w.dtype == bf16 and w.is_contiguous() and tuple(w.shape) == (p.passes, p.n_out, p.k),
        f"w must be a contiguous bf16 [{p.passes}, {p.n_out}, {p.k}] tensor (rows interleaved per 128-channel block), got {tuple(w.shape)} {w.dtype}",
    )
    _check(
        out.dtype == bf16 and out.is_contiguous() and tuple(out.shape) == (p.passes, p.c_out, bn, g.n_pad),
        f"out must be a contiguous bf16 [{p.passes}, {p.c_out}, {bn}, {g.n_pad}] tensor (channel-major, k padded to {g.n_pad}), got {tuple(out.shape)} {out.dtype}",
    )
    for name, t_ in (("ln_weight", ln_weight), ("ln_bias", ln_bias)):
        _check(
            t_.dtype == bf16 and t_.is_contiguous() and tuple(t_.shape) == (p.k,),
            f"{name} must be a contiguous bf16 [{p.k}] tensor, got {tuple(t_.shape)} {t_.dtype}",
        )
    if p.has_bias:
        _check(bias is not None, "this artifact was compiled with has_bias=True; bias must be bound (no silent fallback)")
        _check(
            bias.dtype == bf16 and bias.is_contiguous() and tuple(bias.shape) == (p.passes, p.n_out),
            f"bias must be a contiguous bf16 [{p.passes}, {p.n_out}] tensor (interleaved like the weight rows), got {tuple(bias.shape)} {bias.dtype}",
        )
    else:
        _check(bias is None, "this artifact was compiled with has_bias=False; bias must be None")
    if p.has_mask:
        _check(mask is not None, "this artifact was compiled with has_mask=True; mask must be bound")
        _check(
            mask.dtype == bf16 and mask.is_contiguous() and tuple(mask.shape) == (rows,),
            f"mask must be a contiguous bf16 [{rows}] tensor, got {tuple(mask.shape)} {mask.dtype}",
        )
    else:
        _check(mask is None, "this artifact was compiled with has_mask=False; mask must be None")
    if p.want_stats:
        _check(stats is not None, "this artifact was compiled with want_stats=True; stats must be bound")
        _check(
            stats.dtype == torch.float32 and stats.is_contiguous() and tuple(stats.shape) == (rows, 2),
            f"stats must be a contiguous fp32 [{rows}, 2] tensor, got {tuple(stats.shape)} {stats.dtype}",
        )
    else:
        _check(stats is None, "this artifact was compiled with want_stats=False; stats must be None")
    for name, t_ in (("z", z), ("w", w), ("out", out)):
        _check(t_.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned (a TMA operand)")
    for name, t_ in (("ln_weight", ln_weight), ("ln_bias", ln_bias), ("bias", bias)):
        _check(t_ is None or t_.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned (read with 16-byte vector loads)")
    _check(mask is None or mask.data_ptr() % 4 == 0, "mask must be 4-byte aligned (read as aligned 32-bit words)")
    _check(stats is None or stats.data_ptr() % 8 == 0, "stats must be 8-byte aligned (written as 8-byte (mean, rstd) pairs)")
    geom = (g.m_sched, p.n_out, p.k, p.passes, g.n_res, g.bn, g.n_pad, g.lkb, g.n_kb, g.kbc, g.rc)
    plan.launch(
        geom,
        z.view(-1),
        None,
        w.view(-1),
        None,
        out.view(-1),
        ln_weight,
        ln_bias,
        None,
        None,
        bias.view(-1) if bias is not None else None,
        mask if mask is not None else None,
        stats.view(-1) if stats is not None else None,
        None,
        cuda.CUstream(int(stream)),
    )
    return g


def fold_output_affine(w_value, b_value, ln_weight, ln_bias):
    """Fold LayerNorm affine parameters into projection weights and bias."""
    import torch

    w64 = w_value.double()
    g64 = ln_weight.double()
    be64 = ln_bias.double()
    if w64.ndim != 2 or w64.shape[1] != g64.numel() or be64.numel() != g64.numel():
        raise ValueError(f"fold_output_affine: W must be [c_out, K] with K == len(gamma) == len(beta), got {tuple(w64.shape)}, {g64.numel()}, {be64.numel()}")
    w_folded = (w64 * g64[None, :]).to(torch.bfloat16).contiguous()
    b_fold = w64 @ be64
    if b_value is not None:
        b_fold = b_fold + b_value.double()
    return w_folded, b_fold.to(torch.bfloat16).contiguous()


def run_ln_dual_gemm_tail(
    plan: LnDualGemmPlan,
    a_value,  # [c_out_in = K, bn, n_pad] bf16 contiguous: operand 0, CHANNEL-major (its pair rows are MN-major)
    a_gate,  # [bn * n_res, K] bf16 contiguous: operand 1, pair-major rows (K-major)
    w_value,  # [c_out, K] bf16 contiguous
    w_gate,  # [c_out, K] bf16 contiguous
    out,  # [bn * n_res, c_out] bf16 contiguous (pair-major)
    ln_w_value,  # [K] bf16 (gamma of operand 0; ignored by the kernel under affine_fold, still bound)
    ln_b_value,  # [K] bf16
    ln_w_gate,  # [K] bf16 (gamma of operand 1)
    ln_b_gate,  # [K] bf16
    *,
    n_res: int,
    bn: int,
    bias=None,  # [2 * c_out] bf16 interleaved per 128-block [b_value; b_gate] (interleave_value_gate_bias), required iff params.has_bias
    stats=None,  # [bn * n_res, 2] fp32 (mean, rstd) of operand 0's LayerNorm, required iff params.want_stats
    residual=None,  # [bn * n_res, c_out] bf16 pair-major, required iff params.fuse_residual
    stream,
    k_block: Optional[int] = None,  # k's per 256-row pair tile (row_block_geometry's rule when None; >= MIN_K_BLOCK_MN_MAJOR here; bitwise the same)
) -> RowBlockGeometry:
    """Launch the accumulators arm (K3).  No allocation, no conversion; every check is cheap host arithmetic.  Returns the geometry."""
    import torch
    from cuda.bindings import driver as cuda

    p = plan.params
    _check(p.dual, "this artifact was compiled with pairing='columns'; launch it through run_ln_dual_gemm")
    g = row_block_geometry(n_res, bn, k_block)
    _check(
        g.n_res >= MIN_N_RES_MN_MAJOR,
        f"the accumulators arm loads the MN-major operand in 64-row boxes of one bi: n_res must be >= {MIN_N_RES_MN_MAJOR} (kb >= 64), got {n_res}",
    )
    _check(
        g.kb >= MIN_K_BLOCK_MN_MAJOR,
        f"the accumulators arm loads the MN-major operand in 64-row boxes of one bi: k_block must be >= {MIN_K_BLOCK_MN_MAJOR}, got {g.kb}",
    )
    rows = g.rows_valid
    bf16 = torch.bfloat16
    _check(
        a_value.dtype == bf16 and a_value.is_contiguous() and tuple(a_value.shape) == (p.k, bn, g.n_pad),
        f"a_value must be a contiguous bf16 [{p.k}, {bn}, {g.n_pad}] tensor (channel-major, the inner index padded to {g.n_pad}), got {tuple(a_value.shape)} {a_value.dtype}",
    )
    _check(
        a_gate.dtype == bf16 and a_gate.is_contiguous() and tuple(a_gate.shape) == (rows, p.k),
        f"a_gate must be a contiguous bf16 [{rows}, {p.k}] tensor (pair-major rows), got {tuple(a_gate.shape)} {a_gate.dtype}",
    )
    for name, t_ in (("w_value", w_value), ("w_gate", w_gate)):
        _check(
            t_.dtype == bf16 and t_.is_contiguous() and tuple(t_.shape) == (p.c_out, p.k),
            f"{name} must be a contiguous bf16 [{p.c_out}, {p.k}] tensor, got {tuple(t_.shape)} {t_.dtype}",
        )
    _check(
        out.dtype == bf16 and out.is_contiguous() and tuple(out.shape) == (rows, p.c_out),
        f"out must be a contiguous bf16 [{rows}, {p.c_out}] tensor (pair-major), got {tuple(out.shape)} {out.dtype}",
    )
    for name, t_ in (("ln_w_value", ln_w_value), ("ln_b_value", ln_b_value), ("ln_w_gate", ln_w_gate), ("ln_b_gate", ln_b_gate)):
        _check(
            t_.dtype == bf16 and t_.is_contiguous() and tuple(t_.shape) == (p.k,),
            f"{name} must be a contiguous bf16 [{p.k}] tensor, got {tuple(t_.shape)} {t_.dtype}",
        )
    if p.has_bias:
        _check(bias is not None, "this artifact was compiled with has_bias=True; bias must be bound (no silent fallback)")
        _check(
            bias.dtype == bf16 and bias.is_contiguous() and tuple(bias.shape) == (2 * p.c_out,),
            f"bias must be a contiguous bf16 [{2 * p.c_out}] tensor (interleaved [value block; gate block] per 128 channels), got {tuple(bias.shape)} {bias.dtype}",
        )
    else:
        _check(bias is None, "this artifact was compiled with has_bias=False; bias must be None")
    if p.want_stats:
        _check(stats is not None, "this artifact was compiled with want_stats=True; stats must be bound")
        _check(
            stats.dtype == torch.float32 and stats.is_contiguous() and tuple(stats.shape) == (rows, 2),
            f"stats must be a contiguous fp32 [{rows}, 2] tensor, got {tuple(stats.shape)} {stats.dtype}",
        )
    else:
        _check(stats is None, "this artifact was compiled with want_stats=False; stats must be None")
    if p.fuse_residual:
        _check(residual is not None, "this artifact was compiled with fuse_residual=True; residual must be bound")
        _check(
            residual.dtype == bf16 and residual.is_contiguous() and tuple(residual.shape) == (rows, p.c_out),
            f"residual must be a contiguous bf16 [{rows}, {p.c_out}] tensor (pair-major), got {tuple(residual.shape)} {residual.dtype}",
        )
    else:
        _check(residual is None, "this artifact was compiled with fuse_residual=False; residual must be None")
    for name, t_ in (("a_value", a_value), ("a_gate", a_gate), ("w_value", w_value), ("w_gate", w_gate), ("out", out), ("residual", residual)):
        _check(t_ is None or t_.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned (a TMA operand or a 16-byte vector load)")
    for name, t_ in (("ln_w_value", ln_w_value), ("ln_b_value", ln_b_value), ("ln_w_gate", ln_w_gate), ("ln_b_gate", ln_b_gate), ("bias", bias)):
        _check(t_ is None or t_.data_ptr() % 16 == 0, f"{name} must be 16-byte aligned (read with 16-byte vector loads)")
    _check(stats is None or stats.data_ptr() % 8 == 0, "stats must be 8-byte aligned (written as 8-byte (mean, rstd) pairs)")
    geom = (g.m_sched, p.n_out, p.k, p.passes, g.n_res, g.bn, g.n_pad, g.lkb, g.n_kb, g.kbc, g.rc)
    plan.launch(
        geom,
        a_value.view(-1),
        a_gate.view(-1),
        w_value.view(-1),
        w_gate.view(-1),
        out.view(-1),
        ln_w_value,
        ln_b_value,
        ln_w_gate,
        ln_b_gate,
        bias,
        None,
        stats.view(-1) if stats is not None else None,
        residual.view(-1) if residual is not None else None,
        cuda.CUstream(int(stream)),
    )
    return g
