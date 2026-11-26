# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer batch_invariant/{generator,blocks/operands}.py (e2d163fb3d46).

"""Legal tile, cluster, and K-fold placements for the GEMM kernels."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Tuple

from ._formats import BLOCK_K, BLOCK_SCALED, B_ELEM_BITS_OF, ELEM_BITS_OF, FP8_BLOCKSCALE_FMTS, MMA_K_OF, MX_MIXED_FMTS, chunk_unit, k_granule

K_BLOCK_VALUES = {"mxfp4w8a": (128, 256), "mxfp4w8a_k64": (128, 256)}
SWAP_K_BLOCK = {"mxfp4w8a_k64": 256}

TILES_N = (32, 64, 128, 192, 256)
#: The decode tiles: tile N 8 / 16 on the 1-CTA instruction (kind::f16 admits N >= 8 step 8; kind::f8f6f4 N >= 16) and 16
#: on the 2-CTA one. More CTAs on the weight stream at small M (bytes in flight = CTAs x stages x B-stage bytes).
TILES_N_DENSE_NARROW = (8, 16)
#: Pruned from the ENUMERATED space (still well-formed, still compilable; never within 3 % of a grid point's best record):
#: (tile, 2-CTA) pairs. (128, 8) 1-CTA: a 128-row A tile for eight output columns; (256, 16) 2-CTA: the pair's per-k-tile
#: hand-shake for 16 columns.
PRUNED_DENSE = frozenset({((128, 8), False), ((256, 16), True)})
TILES_M = {
    False: (64, 128),
    True: (128, 256),
}  # 1-CTA / 2-CTA MMA (mma tiler M; the CTA holds half for 2-CTA)
CLUSTERS = ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (1, 4))
RASTERS = ("m", "n")
SWIZZLES = (1, 2, 4, 8)
#: The in-CTA chunk fold (``Knobs.cta_chunks``): a record of the split class ``kchunk[C]`` in which ONE CTA (pair) owns an
#: output tile for all C chunks: the mainloop folds chunk c into a fresh TMEM accumulator (``D_0 = +0``, the chunk's K steps
#: ascending), and the epilogue warps fold the completed chunk partial into a TMEM running sum in chunk order from +0 with
#: ``add.rn``, then apply the chain once after the last chunk -- the workspace fixup's fold, with the partials in TMEM instead
#: of an fp32 round trip through HBM and no second launch. The running sum takes one accumulator's worth of TMEM columns,
#: so the record exists where ``(stages + 1) x tile N`` (plus the scale-factor columns of a block-scaled class) fit the
#: TMEM: tile N <= CTA_CHUNKS_MAX_TN[format] (192 under the 512-column budget of the sm_100 block-scaled class, 256 otherwise).
CTA_CHUNKS_MAX_TN = {
    "bf16": 256,
    "e4m3": 256,
    "e5m2": 256,
    "mxfp8": 192,
    "mxfp8_k64": 256,
    "nvfp4": 192,
    "nvfp4_k128": 256,
    "mxfp4": 192,
    "mxfp4_k128": 256,
    "mxfp4w8a": 192,
    "mxfp4w8a_k64": 256,
    "e4m3_bs": 128,
}
#: The block-scaled instruction fixes MMA M at 128 (1-CTA) / 256 (2-CTA): its scale-factor atom spans 128 rows, so no
#: 64-row CTA tile exists; tile N below 64 would need sub-tile scale-factor addressing; a scale-factor tile multicasts to
#: at most 4 CTAs.
TILES_N_BLOCKSCALED = (64, 128, 192, 256)
TILES_M_BLOCKSCALED = {False: (128,), True: (256,)}
#: The block-scale fp8 class (``e4m3_bs``: the dense ``kind::f8f6f4`` body with a per-block promotion by the epilogue warps): the
#: 128-row CTA tile of the one-row-per-thread promotion (MMA M 128 1-CTA / 256 2-CTA), tile N dividing the 128-column weight
#: scale block (the register running sum bounds it at 128), the 16-wide tile on the 1-CTA instruction only; no split variant,
#: no in-CTA fold, no swapped record.
TILES_N_FP8BS = (16, 32, 64, 128, 256)
TILES_M_FP8BS = TILES_M_BLOCKSCALED
MAX_CLUSTER_DIM_BLOCKSCALED = 4
#: In-cluster split-K (block-scaled body): the ``cluster_k`` CTAs of a (cluster_k, 1, 1) cluster each fold one K chunk and
#: the partials are folded in chunk order through distributed shared memory in the epilogue; the record computes the class
#: ``kchunk[cluster_k]`` exactly as the workspace split does. 1-CTA MMA, no M / N cluster, tile N bounded by the mailbox.
CLUSTER_K = (1, 2, 4)
MAX_TILE_N_CLUSTER_K = {2: 128, 4: 64}
#: The decode specialisation (block-scaled body): operands swapped (the kernel computes C^T with the weights as the MMA's
#: A operand on the 128-row axis and the tokens as B) with one token tile holding every token: the narrow tiles 8 / 16 / 32
#: on the 1-CTA instruction, or the 64-wide tile on either CTA group. The block-scaled step is symmetric in its operands
#: (checked bit for bit), so the records compute the same class. No N cluster (the narrow scale-factor tile is not
#: multicast), no in-cluster split.
TILES_N_NARROW = (8, 16, 32)
SWAP_WIDE_TILE_N = 64
#: The 64-wide swapped token tile exists for the UE8M0 classes (their scale-factor atom covers the tile's K extent of one stage);
#: the NVFP4 classes keep the narrow token tiles only.
SWAP_WIDE_FMTS = (
    "mxfp8",
    "mxfp8_k64",
    "mxfp4",
    "mxfp4_k128",
    "mxfp4w8a",
    "mxfp4w8a_k64",
)
#: B-reuse (the sm_107 instructions only): a 512-row tiler on the 2-CTA 256-row instruction; each CTA pair issues two
#: instructions per K step on the two halves of the tile, the B fragment loaded into the collector buffer by the first and
#: consumed by the second: half the B smem traffic per MMA. One accumulator per half, the same K steps in order from +0:
#: a placement of the class. Dense records only (no swap, no in-cluster split, no in-CTA chunk fold).
TILE_M_BREUSE = 512
BREUSE_FMTS = ("nvfp4_k128", "mxfp8_k64", "mxfp4_k128", "mxfp4w8a_k64")


@dataclass(frozen=True)
class Knobs:
    """Instruction shape, K-fold implementation, and tile scheduling parameters."""

    mma_tiler_mn: Tuple[int, int] = (256, 256)
    cluster_shape_mn: Tuple[int, int] = (2, 1)
    use_2cta_instrs: bool = True
    use_tma_store: bool = True  # the only store path that compiles with dynamic shapes
    raster_along: str = "m"
    swizzle_size: int = 1
    grid_clusters: int = 0  # persistent grid size in clusters; 0 = the launch device's co-resident cluster count
    cta_chunks: bool = False  # the in-CTA chunk fold (split class): one CTA folds every chunk of its tile
    cluster_k: int = 1  # in-cluster split-K factor (block-scaled body): the record computes the class kchunk[cluster_k]
    swap_ab: bool = False  # the decode specialisation (block-scaled body): operands swapped, one token tile for all tokens
    k_block: int = 0  # K elements per smem stage: 0 = the format's default, else a value of K_BLOCK_VALUES[in_fmt] (a lowering choice)

    # named views
    @property
    def tile_m(self) -> int:
        return self.mma_tiler_mn[0]

    @property
    def tile_n(self) -> int:
        return self.mma_tiler_mn[1]

    @property
    def cluster_m(self) -> int:
        return self.cluster_shape_mn[0]

    @property
    def cluster_n(self) -> int:
        return self.cluster_shape_mn[1]

    @property
    def cta_tile_mn(self) -> Tuple[int, int]:
        return (self.tile_m // (2 if self.use_2cta_instrs else 1), self.tile_n)

    @property
    def b_reuse(self) -> bool:
        """The B-reuse record: tile M = 2 x the instruction's M (512 on the 2-CTA instruction; the sm_107 block-scaled
        classes only)."""
        return self.tile_m == TILE_M_BREUSE

    @property
    def inst_m(self) -> int:
        """The MMA instruction's M: the tile M, or half of it for the B-reuse record (two instructions per K step)."""
        return self.tile_m // 2 if self.b_reuse else self.tile_m

    def well_formed(self, in_fmt: str = "bf16") -> bool:
        """Check legal instruction, cluster and tile combinations for this format."""
        two = self.use_2cta_instrs
        cm, cn = self.cluster_shape_mn
        cl_ok = cm in (1, 2, 4, 8, 16) and cn in (1, 2, 4, 8, 16) and cm * cn <= 16 and (not two or cm % 2 == 0)
        ok = (
            cl_ok
            and self.raster_along in RASTERS
            and self.swizzle_size in SWIZZLES
            and self.use_tma_store
            and (self.k_block == 0 or self.k_block in K_BLOCK_VALUES.get(in_fmt, ()))
        )
        if in_fmt in FP8_BLOCKSCALE_FMTS:
            return (
                ok
                and self.tile_m in TILES_M_FP8BS[two]
                and self.tile_n in TILES_N_FP8BS
                and (self.tile_n >= 32 or not two)
                and self.cluster_k == 1
                and not self.swap_ab
                and not self.cta_chunks
            )
        if in_fmt in BLOCK_SCALED:
            tm_ok = self.tile_m in TILES_M_BLOCKSCALED[two] or (self.b_reuse and in_fmt in BREUSE_FMTS and two and not self.swap_ab and self.cluster_k == 1)
            ok = ok and tm_ok and cm <= MAX_CLUSTER_DIM_BLOCKSCALED and cn <= MAX_CLUSTER_DIM_BLOCKSCALED and self.cluster_k in CLUSTER_K
            if self.swap_ab:
                ok = (
                    ok
                    and cn == 1
                    and self.cluster_k == 1
                    and ((self.tile_n in TILES_N_NARROW and not two) or (self.tile_n == SWAP_WIDE_TILE_N and in_fmt in SWAP_WIDE_FMTS))
                )
            else:
                ok = ok and self.tile_n in TILES_N_BLOCKSCALED
            if self.cluster_k > 1:
                ok = ok and cm == 1 and cn == 1 and not two and self.tile_n <= MAX_TILE_N_CLUSTER_K[self.cluster_k]
            if self.cta_chunks:
                ok = ok and self.cluster_k == 1 and not self.swap_ab and not self.b_reuse and self.tile_n <= CTA_CHUNKS_MAX_TN[in_fmt]
            return ok
        tm_ok = self.tile_m in TILES_M[two]
        # the decode tiles: N 16 on both CTA groups; N 8 on the 1-CTA kind::f16 instruction only
        tn_ok = (32 <= self.tile_n <= 256 and self.tile_n % 32 == 0) or (
            self.tile_n in TILES_N_DENSE_NARROW and (self.tile_n >= 16 or (not two and in_fmt == "bf16"))
        )
        ok = ok and tm_ok and tn_ok and self.cluster_k == 1 and not self.swap_ab
        if self.cta_chunks:
            ok = ok and self.tile_n <= CTA_CHUNKS_MAX_TN[in_fmt]  # the running sum fits next to the accumulator stages
        return ok


def can_implement(
    knobs: Knobs,
    M: int,
    N: int,
    K: int,
    b_major: str = "n",
    chunks: int = 1,
    in_fmt: str = "bf16",
) -> bool:
    """The body's acceptance: well-formed knobs for the format (TMA store only, so out-of-bounds tiles are fine), 16-byte
    alignment of each operand's contiguous extent (the K chunk for A and a K-major B, N for an N-major B and for C), a
    whole number of chunks; on ``kind::f8f6f4`` with an MN-major B the N step of the 2-CTA instruction is 32, so the
    16-wide decode tile of the 2-CTA pair needs a K-major weight."""
    if in_fmt not in MMA_K_OF or b_major not in ("n", "k"):
        return False
    if K % chunks:
        return False
    if knobs.cta_chunks and chunks == 1:
        return False  # the in-CTA chunk fold is a record of the split classes only
    if in_fmt != "bf16" and knobs.tile_n < 32 and knobs.use_2cta_instrs and b_major == "n":
        return False
    Kc, al_in = K // chunks, 128 // B_ELEM_BITS_OF[in_fmt]
    # the output is stored directly (no fp32 workspace) by the unsplit class and by the in-CTA chunk fold
    al_out = 8 if (chunks == 1 or knobs.cta_chunks) else 4
    if in_fmt in BLOCK_SCALED:
        # the block-scaled instruction takes K-major A and B only; K is a whole number of scale blocks (whole 128-element
        # runs for the sm_100 mixed class: `k_granule`); a chunk is a whole number of steps AND of scale-factor atom
        # columns (`chunk_unit`: 128 elements for the MX classes)
        if b_major != "k" or K % k_granule(in_fmt) or (chunks > 1 and Kc % chunk_unit(in_fmt)):
            return False
        if chunks > 1 and knobs.k_block and Kc % knobs.k_block:
            return False  # a chunk is whole k-tiles of the record's k-block
        if knobs.swap_ab:
            # the decode specialisation: the kernel's M is N (the weights), its N is M (the tokens), all tokens in one
            # tile (the scale-factor atom read from its start); the M-major output needs N aligned. The mixed classes'
            # split classes take it too (one chunk's partial through the workspace like the dense record's; not the
            # in-cluster / in-CTA folds); the other block-scaled classes keep their unsplit-only swapped records
            split_ok = chunks == 1 or (in_fmt in MX_MIXED_FMTS and not (knobs.cluster_k > 1 or knobs.cta_chunks))
            return split_ok and knobs.tile_n >= M and N % 8 == 0 and Kc % al_in == 0
        if knobs.cluster_k > 1:
            # the in-cluster split computes exactly the class kchunk[cluster_k]: its chunks are whole k-tiles (one smem
            # stage), so K splits into cluster_k equal k-tile ranges; the output is stored directly. Its launch puts one
            # cluster per output tile on the grid's z axis, which the hardware caps (MAX_GRID_Z): above that tile count
            # the record is not a placement of the shape and the workspace split (the same class) takes over
            k_tile = knobs.k_block or 1024 // ELEM_BITS_OF[in_fmt]
            tiles = -(-M // knobs.tile_m) * -(-N // knobs.tile_n)
            return chunks == knobs.cluster_k and K % (k_tile * chunks) == 0 and N % 8 == 0 and Kc % al_in == 0 and tiles <= 65535
    if in_fmt in FP8_BLOCKSCALE_FMTS:
        # the block-scale class: a K-major weight (the NT operator), K a whole number of scale blocks (= k-tiles), no split
        # variant, no in-CTA fold; the output is stored directly
        return b_major == "k" and chunks == 1 and not knobs.cta_chunks and K % BLOCK_K == 0 and K % al_in == 0 and N % al_out == 0
    return Kc % al_in == 0 and (N if b_major == "n" else Kc) % al_in == 0 and N % al_out == 0


@functools.lru_cache(maxsize=None)
def knob_product(
    swizzles=SWIZZLES,
    clusters=CLUSTERS,
    cta_chunks: bool = False,
    in_fmt: str = "bf16",
) -> tuple[Knobs, ...]:
    """Every well-formed knob combination, before the kernel's shape-dependent acceptance: 888 records for bf16 (720 at the
    tiles N 32..256 plus 168 decode-tile records), 840 for the fp8 formats (no 8-wide tile on ``kind::f8f6f4``).
    ``cta_chunks=True`` adds the in-CTA chunk-fold twin of every record under the TMEM budget, which only the split
    classes accept."""
    out = []
    bs = in_fmt in BLOCK_SCALED
    for two in (False, True):
        tms = TILES_M_BLOCKSCALED[two] + ((TILE_M_BREUSE,) if two and in_fmt in BREUSE_FMTS else ()) if bs else TILES_M[two]
        tns = TILES_N_BLOCKSCALED + TILES_N_NARROW if bs else TILES_N + TILES_N_DENSE_NARROW
        for tm in tms:
            for tn in tns:
                for cl in clusters:
                    if two and cl[0] % 2:
                        continue
                    for raster in RASTERS:
                        for sw in swizzles:
                            for ck in CLUSTER_K if bs else (1,):
                                for swap in (False, True) if bs else (False,):
                                    for cc in (False, True) if cta_chunks else (False,):
                                        k = Knobs(
                                            (tm, tn),
                                            cl,
                                            two,
                                            True,
                                            raster,
                                            sw,
                                            0,
                                            cc,
                                            ck,
                                            swap,
                                            SWAP_K_BLOCK.get(in_fmt, 0) if swap else 0,
                                        )
                                        if k.well_formed(in_fmt) and (bs or (k.mma_tiler_mn, two) not in PRUNED_DENSE):
                                            out.append(k)
    return tuple(out)
