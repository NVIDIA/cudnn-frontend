# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""blk128 sparse layouts from open-source video sparse-attention systems.

For the tile-based systems block occupancy follows the released tile
geometry, padding rule, resolution, and frame count and does not depend on
sample content, so those layouts are generated here.  Q/K/V are synthetic in
every case.

* ``ssta_hunyuan``: HunyuanVideo-1.5 SSTA (hunyuanvideo_1_5_transformer.py
  attn_param): (6,8,8) latent tiles, zero padding, (3,3,3) sliding window OR
  per-head top-k, text tokens appended.
* ``sta_hunyuan``: Sliding Tile Attention with the per-head windows of the
  HunyuanVideo mask search, no top-k.
* ``vsa_wan``: FastVideo Video Sparse Attention on a Wan2.1 latent: (4,4,4) or
  (4,8,8) cubes, ``construct_variable_block_sizes`` padding, top-k cubes per
  query cube at the given sparsity (random scores).

Every logical tile or cluster with ``v`` valid tokens becomes ``ceil(v / 128)``
blk128 blocks (full blocks first, then one partial block), and tile adjacency
is expanded to block adjacency.  Pad tokens never take part in the attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

BLOCK = 128


@dataclass
class Layout:
    """One sparse pattern.  ``block_sizes`` is the KV partition; ``q_block_sizes``
    the query partition (``None``: the two sides share ``block_sizes``)."""

    name: str
    desc: str
    H: int
    q2k_idx: torch.Tensor  # [1, H, nQ, kmax] int32
    q2k_cnt: torch.Tensor  # [1, H, nQ] int32
    block_sizes: torch.Tensor  # [nKV] int32
    meta: Dict = field(default_factory=dict)
    q_block_sizes: Optional[torch.Tensor] = None  # [nQ] int32

    @property
    def nB(self) -> int:
        return int(self.block_sizes.shape[0])

    @property
    def nQ(self) -> int:
        return int(self.q2k_cnt.shape[-1])

    @property
    def T(self) -> int:
        return self.nQ * BLOCK

    @property
    def seqlen_k(self) -> int:
        return self.nB * BLOCK

    @property
    def q_sizes(self) -> torch.Tensor:
        return self.block_sizes if self.q_block_sizes is None else self.q_block_sizes

    @property
    def independent(self) -> bool:
        return self.q_block_sizes is not None

    def to(self, device) -> "Layout":
        qbs = None if self.q_block_sizes is None else self.q_block_sizes.to(device)
        return Layout(self.name, self.desc, self.H, self.q2k_idx.to(device), self.q2k_cnt.to(device), self.block_sizes.to(device), dict(self.meta), qbs)

    def stats(self) -> Dict:
        bs = self.block_sizes.to(torch.int64).cpu()
        qs = self.q_sizes.to(torch.int64).cpu()
        idx = self.q2k_idx.to(torch.int64).cpu()
        cnt = self.q2k_cnt.to(torch.int64).cpu()
        H, nQ, kmax = idx.shape[1], idx.shape[2], idx.shape[3]
        nB = self.nB
        active = torch.arange(kmax).view(1, 1, 1, kmax) < cnt.unsqueeze(-1)
        kv_rows_per_q = (bs[idx.clamp(0, nB - 1)] * active).sum(-1)
        useful_pairs = int((kv_rows_per_q * qs.view(1, 1, nQ)).sum())
        whole_pairs = int(cnt.sum()) * BLOCK * BLOCK
        return dict(
            H=H,
            nB=nB,
            nQ=nQ,
            T=self.T,
            seqlen_k=self.seqlen_k,
            kmax=kmax,
            independent_partitions=self.independent,
            pad_frac_blk128=1.0 - int(bs.sum()) / self.seqlen_k,
            q_pad_frac_blk128=1.0 - int(qs.sum()) / self.T,
            partial_block_frac=int(((bs > 0) & (bs < BLOCK)).sum()) / nB,
            q_partial_block_frac=int(((qs > 0) & (qs < BLOCK)).sum()) / nQ,
            selected_block_pairs=int(cnt.sum()),
            useful_pairs=useful_pairs,
            whole_block_pairs=whole_pairs,
            pair_efficiency=useful_pairs / max(whole_pairs, 1),
            useful_tflop_fwd=4.0 * useful_pairs * 128 / 1e12,
        )


def _adj_to_q2k(adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """[H, nB, nB] bool -> padded (idx [1,H,nB,kmax], cnt [1,H,nB])."""
    H, nQ, nK = adj.shape
    cnt = adj.sum(-1).to(torch.int32)
    kmax = min(nK, max(8, (int(cnt.max()) + 7) // 8 * 8))
    order = torch.argsort(adj.to(torch.int8), dim=-1, descending=True, stable=True)
    idx = order[..., :kmax].to(torch.int32)
    idx = torch.where(torch.arange(kmax).view(1, 1, kmax) < cnt.unsqueeze(-1), idx, torch.zeros_like(idx))
    assert int(cnt.min()) >= 1, "a query block attends no KV block"
    return idx.unsqueeze(0).contiguous(), cnt.unsqueeze(0).contiguous()


def _groups_to_blocks(group_valid: Sequence[int]) -> Tuple[List[List[int]], List[int]]:
    blocks_of_group: List[List[int]] = []
    sizes: List[int] = []
    for v in group_valid:
        ids = []
        v = int(v)
        while v > 0:
            take = min(BLOCK, v)
            ids.append(len(sizes))
            sizes.append(take)
            v -= take
        blocks_of_group.append(ids)
    return blocks_of_group, sizes


def _group_mask_to_layout(name: str, desc: str, H: int, group_valid: Sequence[int], group_adj: torch.Tensor, meta: Dict) -> Layout:
    blocks_of_group, sizes = _groups_to_blocks(group_valid)
    nB, G = len(sizes), len(group_valid)
    blk_group = torch.empty(nB, dtype=torch.long)
    for g, ids in enumerate(blocks_of_group):
        for b in ids:
            blk_group[b] = g
    if group_adj.shape[0] == 1 and H > 1:
        group_adj = group_adj.expand(H, G, G)
    adj = group_adj.cpu()[:, blk_group][:, :, blk_group]
    idx, cnt = _adj_to_q2k(adj)
    return Layout(name, desc, H, idx, cnt, torch.tensor(sizes, dtype=torch.int32), dict(meta, num_groups=G))


def _ceil_sizes(dim_len: int, tile: int) -> List[int]:
    n = math.ceil(dim_len / tile)
    out = [tile] * n
    out[-1] = dim_len - (n - 1) * tile
    return out


def _tile_valid_counts(canvas: Tuple[int, int, int], tile: Tuple[int, int, int]) -> np.ndarray:
    """Valid voxels per 3D tile in (t, h, w) tile order; matches FastVideo's
    construct_variable_block_sizes and the HunyuanVideo-1.5 zero padding."""
    ts = [np.asarray(_ceil_sizes(c, t)) for c, t in zip(canvas, tile)]
    return (ts[0][:, None, None] * ts[1][None, :, None] * ts[2][None, None, :]).reshape(-1)


def _sta_window_mask(ntiles: Tuple[int, int, int], win: Tuple[int, int, int]) -> torch.Tensor:
    """Clamped-centre 3D window over tiles (hyvideo create_sta_3d_mask_optimize)."""
    nt, nh, nw = ntiles
    i = np.arange(nt * nh * nw)
    qt, qh, qw = i // (nh * nw), (i % (nh * nw)) // nw, i % nw
    ct = np.clip(qt, win[0] // 2, (nt - 1) - win[0] // 2)
    ch = np.clip(qh, win[1] // 2, (nh - 1) - win[1] // 2)
    cw = np.clip(qw, win[2] // 2, (nw - 1) - win[2] // 2)
    m = np.abs(ct[:, None] - qt[None, :]) <= win[0] // 2
    m &= np.abs(ch[:, None] - qh[None, :]) <= win[1] // 2
    m &= np.abs(cw[:, None] - qw[None, :]) <= win[2] // 2
    return torch.from_numpy(m)


def _topk_mask(H: int, G: int, k: int, gen: torch.Generator) -> torch.Tensor:
    scores = torch.randn(H, G, G, generator=gen)
    top = scores.topk(min(k, G), dim=-1).indices
    m = torch.zeros(H, G, G, dtype=torch.bool)
    m.scatter_(-1, top, True)
    return m


HY15 = dict(tile=(6, 8, 8), win=(3, 3, 3), topk=64, vae_t=4, vae_s=16)
STA_WINDOWS_HY = [(3, 3, 3), (3, 3, 5), (3, 6, 10), (5, 6, 10), (3, 3, 3), (3, 6, 10), (5, 6, 10), (3, 3, 5)]


def ssta_hunyuan(
    width_px: int,
    height_px: int,
    frames: int,
    H: int = 16,
    seed: int = 0,
    text_valid: int = 356,
    topk: Optional[int] = None,
    windows: Optional[Sequence[Tuple[int, int, int]]] = None,
    name: Optional[str] = None,
) -> Layout:
    cfg = HY15
    t = (frames - 1) // cfg["vae_t"] + 1
    h, w = height_px // cfg["vae_s"], width_px // cfg["vae_s"]
    tile, win = cfg["tile"], cfg["win"]
    if topk is None:
        topk = cfg["topk"] if t > 31 else cfg["topk"] // 2
    ntiles = tuple(math.ceil(c / s) for c, s in zip((t, h, w), tile))
    Gv = int(np.prod(ntiles))
    tile_tokens = int(np.prod(tile))
    text_tiles = math.ceil(text_valid / tile_tokens)
    text_valid_per_tile = [tile_tokens] * text_tiles
    text_valid_per_tile[-1] = text_valid - tile_tokens * (text_tiles - 1)
    G = Gv + text_tiles
    gen = torch.Generator().manual_seed(seed)
    if windows is None:
        sta = _sta_window_mask(ntiles, win).unsqueeze(0)
    else:
        sta = torch.stack([_sta_window_mask(ntiles, tuple(windows[hh % len(windows)])) for hh in range(H)])
    adj = torch.zeros(H, G, G, dtype=torch.bool)
    adj[:, :Gv, :Gv] = sta if sta.shape[0] == H else sta.expand(H, Gv, Gv)
    if topk > 0:
        adj[:, :Gv, :Gv] |= _topk_mask(H, Gv, topk, gen)
    adj[:, :, Gv:] = True  # video rows attend the text tiles
    adj[:, Gv:, :] = True  # text rows attend everything
    group_valid = list(map(int, _tile_valid_counts((t, h, w), tile))) + text_valid_per_tile
    desc = (
        f"{width_px}x{height_px} {frames}f: latent {t}x{h}x{w} -> tiles {ntiles} of {tile}"
        + (f", top-{topk}/head" if topk else "")
        + f", {text_valid} text tokens"
    )
    return _group_mask_to_layout(name or f"ssta_hy15_{height_px}p_{frames}f", desc, H, group_valid, adj, dict(latent=(t, h, w), tile=tile, topk=topk))


def sta_hunyuan(H: int = 24, frames: int = 117, text_valid: int = 256, seed: int = 0) -> Layout:
    return ssta_hunyuan(1280, 720, frames, H=H, seed=seed, text_valid=text_valid, topk=0, windows=STA_WINDOWS_HY, name=f"sta_hy_720p_{frames}f")


def vsa_wan(width_px: int, height_px: int, frames: int, tile: Tuple[int, int, int] = (4, 8, 8), sparsity: float = 0.875, H: int = 12, seed: int = 0) -> Layout:
    t = (frames - 1) // 4 + 1
    h, w = height_px // 16, width_px // 16
    ntiles = tuple(math.ceil(c / s) for c, s in zip((t, h, w), tile))
    G = int(np.prod(ntiles))
    topk = max(1, min(math.ceil((1 - sparsity) * G), G))
    adj = _topk_mask(H, G, topk, torch.Generator().manual_seed(seed))
    tile_tokens = int(np.prod(tile))
    desc = f"Wan2.1 {width_px}x{height_px} {frames}f: latent {t}x{h}x{w} -> tiles {ntiles} of {tile}, sparsity {sparsity} (top-{topk}/{G} per head)"
    return _group_mask_to_layout(
        f"vsa_wan_{height_px}p_{frames}f_t{tile_tokens}",
        desc,
        H,
        list(map(int, _tile_valid_counts((t, h, w), tile))),
        adj,
        dict(latent=(t, h, w), tile=tile, sparsity=sparsity),
    )


def build(spec: str) -> Layout:
    """``ssta:WxHxF[:heads]`` | ``sta[:heads]`` | ``vsa:WxHxF:tile_tokens[:sparsity[:heads]]``."""
    kind, *rest = spec.split(":")
    if kind == "ssta":
        w, h, f = map(int, rest[0].split("x"))
        return ssta_hunyuan(w, h, f, H=int(rest[1]) if len(rest) > 1 else 16)
    if kind == "sta":
        return sta_hunyuan(H=int(rest[0]) if rest else 24)
    if kind == "vsa":
        w, h, f = map(int, rest[0].split("x"))
        tile = {64: (4, 4, 4), 256: (4, 8, 8), 128: (2, 8, 8)}[int(rest[1]) if len(rest) > 1 else 256]
        sparsity = float(rest[2]) if len(rest) > 2 else 0.875
        return vsa_wan(w, h, f, tile=tile, sparsity=sparsity, H=int(rest[3]) if len(rest) > 3 else 12)
    raise ValueError(spec)


# The layouts the benchmark runs by default: HunyuanVideo-1.5 SSTA at two
# resolutions and two lengths, Sliding Tile Attention, FastVideo VSA with
# (4,4,4) and (4,8,8) cubes.  Padding ranges from 0.7% to 59%.
BENCHMARK_LAYOUTS = [
    "ssta:848x480x121:16",
    "ssta:848x480x241:16",
    "ssta:1280x720x121:16",
    "ssta:1280x720x241:16",
    "sta:24",
    "vsa:832x480x81:64:0.875:12",
    "vsa:1280x720x81:256:0.875:40",
    "vsa:1280x720x81:64:0.875:40",
]
