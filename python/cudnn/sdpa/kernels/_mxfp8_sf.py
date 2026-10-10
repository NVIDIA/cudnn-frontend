# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""MXFP8 scale-factor (E8M0 bytes in cuDNN's F8_128x4 order) TMA descriptors and the cga2 peer split of an SF slab -- ONE
spelling for every SDPA kernel that block-scales.

The atom byte rule (which byte inside a 512-B atom holds scale ``(r, c)``) is ``cudnn.frost.tile_dsl.sf_layout``; what
this module owns is the GMEM geometry of the SF tensors the SDPA graph carries and how a TMA box walks them:

* **rowwise SF** (Q, K, dO -- an operand quantized along its contraction axis D; the backward's dS along its own
  contraction axis): the F8_128x4 atom grid of ``[b*h*S/128 s-tiles] x [D/128 planes]`` is laid out with the s-tile as the
  OUTER index, so one (b, h, s_tile)'s ``sf_smem_size`` bytes (all its D planes) are CONTIGUOUS.  A 5-D descriptor
  ``[128-B row, rows, tiles, heads, batches]`` with a ``[128, box_rows, 1, 1, 1]`` box fetches ``box_rows`` 128-byte rows of
  one tile -- :func:`build_rowwise_sf_desc`.
* **columnwise SF** (V -- quantized along S, the BMM2 contraction axis): the same atom rule applied to the TRANSPOSED
  scale matrix ``[D, S/32]`` lays the grid out ``(D/128) x (b*h*S/128)`` row-major -- the D-plane index is the OUTER one,
  and the planes of one tile sit a whole plane of ``b*h*tiles`` atoms apart, a stride that GROWS WITH S.
  At d = 128 there is exactly one plane and the two layouts coincide, which is why the d128
  and d192x128 kernels read V through the rowwise form and why a d256 kernel was right at S = 128 and wrong from S = 256
  on until it took :func:`build_columnwise_sf_desc`.  THD packs both planes of a (head, sequence-tile) contiguously
  (per-tile strides); dense takes the plane-major strides.

**Peer split (cga2).**  Each CTA of a pair TMA-loads HALF of a K / V SF slab into the shared ring (the tensor TMA's
self-multicast delivers the whole slab to both), so the TMA-LDG warp offsets its box by ``bytes_per_peer`` in SMEM and by
``rows_per_peer`` 128-byte rows in the descriptor -- :func:`sf_peer_split`.  The columnwise (V) split walks D-PLANES, not
rows: the kernel keeps ``planes_per_peer = num_planes // cta_mma`` and passes it as ``planes_per_box``.

Every function is plain Python that traces inside the ``@cute.jit`` host (``tmap.create_tensor_map_tiled`` is a DSL op)
or the ``@cute.kernel`` body, mirroring the closures the sm107 forwards used to carry; ``num_tiles`` / ``num_batches`` may
be Python ints (dense: derived from ``problem_size``) or traced ``Int32`` (THD: the packed per-sequence-tile totals).
The op sequence is the forwards' original one, in order, so their sm_107a cubins are byte-identical before and after the
lift (the gate of this refactor; ``test_mxfp8_sf_desc_shared.py``).  MEASURED 2026-09-30: cubin, PTX and clean-MLIR md5s
of both forwards at dense and causal+SWA640 (4 builds) identical at dd3235c3 and after the lift (md5 records retained
internally).  Re-check that byte-identity -- dump both forwards' sm_107a cubins before and after and compare md5s, on any
box whose DSL knows sm_107a -- before and after ANY edit to the builders below.
"""

from typing import NamedTuple

import cutlass
from cutlass.experimental.cuda import tensor_map as tmap

#: One TMA row of an SF slab: 128 bytes -- the 5-D SF descriptors' innermost extent (a 16-byte-unit stride of 8), and the
#: unit the cga2 peer split of a rowwise slab is counted in.
SF_TMA_ROW_BYTES = 128


class SfPeerSplit(NamedTuple):
    bytes_per_peer: int  # SMEM offset of this CTA's half of the slab: cta_in_pair * bytes_per_peer
    rows_per_peer: int  # descriptor row coordinate of that half: cta_in_pair * rows_per_peer (128-byte rows)


def sf_tma_rows(sf_smem_size: int) -> int:
    """128-byte TMA rows of an SF slab of ``sf_smem_size`` bytes -- the box height of a whole-slab (Q) load."""
    if sf_smem_size <= 0 or sf_smem_size % SF_TMA_ROW_BYTES != 0:
        raise ValueError(f"an SF slab is a whole number of {SF_TMA_ROW_BYTES}-byte TMA rows; got {sf_smem_size} bytes")
    return sf_smem_size // SF_TMA_ROW_BYTES


def sf_peer_split(sf_smem_size: int, cta_mma: int) -> SfPeerSplit:
    """How a rowwise K / V SF slab of ``sf_smem_size`` bytes splits across the ``cta_mma`` CTAs of a cluster pair.

    cga1 (``cta_mma == 1``) is the whole slab (offsets fold to 0 at the call site: ``cta_in_pair`` is 0).  Raises when the
    split is not a whole number of TMA rows -- a box cannot start mid-row, so such a geometry has no correct peer offset."""
    if cta_mma < 1 or sf_smem_size % cta_mma != 0:
        raise ValueError(f"an SF slab of {sf_smem_size} bytes does not split evenly across CTA_MMA={cta_mma} peers")
    bytes_per_peer = sf_smem_size // cta_mma
    if bytes_per_peer % SF_TMA_ROW_BYTES != 0:
        raise ValueError(
            f"a peer's share of the SF slab must be whole {SF_TMA_ROW_BYTES}-byte TMA rows; got {bytes_per_peer} bytes ({sf_smem_size} / {cta_mma})"
        )
    return SfPeerSplit(bytes_per_peer, bytes_per_peer // SF_TMA_ROW_BYTES)


def build_rowwise_sf_desc(sf_tensor, *, num_tiles, sf_smem_size: int, num_rows_box: int, num_heads, num_batches):
    """5-D TMA descriptor over a ROWWISE SF tensor (per-tile contiguous atoms): ``[128 B, rows, tiles, heads, batches]``,
    box ``[128, num_rows_box, 1, 1, 1]``.

    ``sf_tensor``: the packed uint8 SF tensor; ``num_tiles``: s-tiles per (b, h) (dense ``ceil(S / TILE)``) or the packed
    per-sequence-tile total with ``num_batches = 1`` (THD); ``sf_smem_size``: one tile's bytes (= ``TILE * ceil128(D) / 32``);
    ``num_rows_box``: the box height -- the whole slab for Q (:func:`sf_tma_rows`), a peer's share for K / V at cga2
    (:func:`sf_peer_split`).  Strides are in 16-byte units, as TMA counts them.  (Paged KV needs no page base: the Rubin
    and SM100 forwards pass ``num_batches = n_pages``, ``num_tiles = page_size / 128`` and address the page as the batch
    coordinate -- the K / V scale-factor POOLS page with K / V.)"""
    sf_base = cutlass.Int64(sf_tensor.iterator.toint())
    tile_stride_16 = sf_smem_size // 16
    return tmap.create_tensor_map_tiled(
        global_address=sf_base,
        dtype=cutlass.Uint8,
        global_dims=[
            SF_TMA_ROW_BYTES,
            sf_smem_size // SF_TMA_ROW_BYTES,
            num_tiles,
            num_heads,
            num_batches,
        ],
        global_strides=[
            SF_TMA_ROW_BYTES // 16,
            tile_stride_16,
            cutlass.Int64(num_tiles) * tile_stride_16,
            cutlass.Int64(num_heads) * cutlass.Int64(num_tiles) * tile_stride_16,
        ],
        box_dims=[SF_TMA_ROW_BYTES, num_rows_box, 1, 1, 1],
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )


def build_columnwise_sf_desc(
    sf_tensor, *, num_tiles, num_heads, num_batches, num_planes: int, planes_per_box: int, sf_bytes_per_block: int, sf_smem_size: int, thd_varlen: bool
):
    """5-D TMA descriptor over a COLUMNWISE SF tensor (D-plane-major atoms): ``[128 B, rows-of-one-atom, planes, tiles,
    heads * batches]``, box ``[128, atom rows, planes_per_box, 1, 1]`` -- ``planes_per_box`` planes of one tile per load
    (a cga2 peer's D-planes; every plane at cga1).

    Dense: the plane stride is a whole plane of ``batches * heads * tiles`` atoms (``sf_bytes_per_block`` each) -- it grows
    with S -- and the tile stride is one atom.  THD (``thd_varlen``): both planes of a (head, sequence-tile) are packed
    contiguously, so the plane stride is one atom and the tile stride the whole ``sf_smem_size`` slab.  ``num_tiles`` /
    ``num_batches`` as in :func:`build_rowwise_sf_desc`."""
    v_sf_groups = cutlass.Int64(num_batches) * cutlass.Int64(num_heads) * num_tiles
    if thd_varlen:
        plane_stride_16 = sf_bytes_per_block // 16
        tile_stride_16 = sf_smem_size // 16
    else:
        plane_stride_16 = (v_sf_groups * sf_bytes_per_block) // 16
        tile_stride_16 = sf_bytes_per_block // 16
    return tmap.create_tensor_map_tiled(
        global_address=cutlass.Int64(sf_tensor.iterator.toint()),
        dtype=cutlass.Uint8,
        global_dims=[
            SF_TMA_ROW_BYTES,
            sf_bytes_per_block // SF_TMA_ROW_BYTES,
            num_planes,
            num_tiles,
            num_heads * num_batches,
        ],
        global_strides=[
            SF_TMA_ROW_BYTES // 16,
            plane_stride_16,
            tile_stride_16,
            cutlass.Int64(num_tiles) * tile_stride_16,
        ],
        box_dims=[SF_TMA_ROW_BYTES, sf_bytes_per_block // SF_TMA_ROW_BYTES, planes_per_box, 1, 1],
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )


def build_ds_sf_atom_desc(sf_tensor, *, inner_tiles, outer_tiles, num_bh, sf_atom_bytes: int = 512):
    """5-D TMA descriptor over a dS scale-factor WORKSPACE tensor -- ``[B * H_chunk, outer_tiles, inner_tiles, 512]`` bytes, one
    F8_128x4 atom per (outer tile, inner tile) -- as ``[128 B, 4 rows, inner, outer, bh]`` with a ``[128, 4, 1, 1, 1]`` box: ONE
    atom per TMA op, for the store that publishes the atoms the backward's compute lanes quantized in SMEM AND for the loader of
    the stage-3 block-scale GEMM that consumes them (one spelling, both ends).

    The MXFP8 d=256 backward writes two such tensors (``config_sm107.sf_workspace_bytes``): ``sf_ds_dk`` is
    ``[B, H_chunk, S_kv/128, S_q/128, 512]`` -- atom (kv_tile, q_tile), rows = kv within the tile, columns = q-blocks -- so
    ``inner_tiles = S_q / 128`` (q tiles), ``outer_tiles = S_kv / 128``; ``sf_ds_dq`` is ``[B, H_chunk, S_q/128, S_kv/128, 512]`` --
    atom (q_tile, kv_tile), rows = q within the tile, columns = kv-blocks -- so ``inner_tiles = S_kv / 128``, ``outer_tiles =
    S_q / 128``.  Coordinates at the op: ``(0, 0, inner, outer, b * H_chunk + h)`` (the head axis is the launch's CHUNK-local one,
    exactly the payload workspace's).  Strides are in 16-byte units, as TMA counts them; a Python-int or traced ``Int32`` tile
    count is accepted like the rowwise builder's ``num_tiles``.  Unswizzled: the atom IS the byte order the consumer's UTCCP
    reads (``tile_dsl.sf_layout``), so a permutation here would need an un-permuting consumer."""
    if sf_atom_bytes % SF_TMA_ROW_BYTES:
        raise ValueError(f"an SF atom is a whole number of {SF_TMA_ROW_BYTES}-byte TMA rows; got {sf_atom_bytes}")
    atom_rows = sf_atom_bytes // SF_TMA_ROW_BYTES
    atom_stride_16 = sf_atom_bytes // 16
    inner_stride_16 = cutlass.Int64(inner_tiles) * atom_stride_16
    return tmap.create_tensor_map_tiled(
        global_address=cutlass.Int64(sf_tensor.iterator.toint()),
        dtype=cutlass.Uint8,
        global_dims=[SF_TMA_ROW_BYTES, atom_rows, inner_tiles, outer_tiles, num_bh],
        global_strides=[SF_TMA_ROW_BYTES // 16, atom_stride_16, inner_stride_16, cutlass.Int64(outer_tiles) * inner_stride_16],
        box_dims=[SF_TMA_ROW_BYTES, atom_rows, 1, 1, 1],
        swizzle=tmap.TensorMapSwizzle.none,
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
