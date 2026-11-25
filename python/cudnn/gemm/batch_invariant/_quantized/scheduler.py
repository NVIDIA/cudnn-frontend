# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer's flashinfer/gemm/batch_invariant/blocks/scheduler.py
# (e2d163fb3d46) to the cuDNN Frontend persistent tile scheduler.

"""Persistent cluster grid with an exact (unpadded) raster over the output tiles."""

from __future__ import annotations

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Int32,
    dsl_user_op,
    extract_mlir_values,
    min,
    new_from_mlir_values,
)

from cudnn._cutlass_helpers.static_persistent_tile_scheduler import WorkTileInfo


class SchedParams:
    """Cached divisors for the cluster raster and its optional swizzle bands."""

    @dsl_user_op
    def __init__(
        self,
        num_tiles_mnl,
        cluster_shape_mn: Tuple[int, int],
        swizzle: int,
        raster_m: bool,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        self.cluster_shape_mn = tuple(int(x) for x in cluster_shape_mn)
        cm = cute.ceil_div(num_tiles_mnl[0], self.cluster_shape_mn[0], loc=loc, ip=ip)
        cn = cute.ceil_div(num_tiles_mnl[1], self.cluster_shape_mn[1], loc=loc, ip=ip)
        L = num_tiles_mnl[2]
        # a static batch extent of one (the unsplit class, the in-CTA chunk fold's output) needs no batch decode
        self.static_l1 = isinstance(L, int) and L == 1
        S, F = (cm, cn) if raster_m else (cn, cm)
        self.swizzle = swizzle
        self.raster_m = raster_m
        per_l = S * F
        self.total = per_l * Int32(L)
        self.fdd_S = cute.fast_divmod_create_divisor_v2(S, loc=loc, ip=ip)
        self._fields = ["total", "fdd_S"]
        if swizzle > 1:
            self.full_bands = F // swizzle
            self.fdd_band = cute.fast_divmod_create_divisor_v2(swizzle * S, loc=loc, ip=ip)
            self._fields.extend(("full_bands", "fdd_band"))
        if not self.static_l1:
            self.fdd_l = cute.fast_divmod_create_divisor_v2(per_l, loc=loc, ip=ip)
            self._fields.append("fdd_l")

    def __extract_mlir_values__(self) -> list:
        values, self._pos = [], []
        for name in self._fields:
            v = extract_mlir_values(getattr(self, name))
            values += v
            self._pos.append(len(v))
        return values

    def __new_from_mlir_values__(self, values: list) -> "SchedParams":
        new = object.__new__(SchedParams)
        new.swizzle = self.swizzle
        new.raster_m = self.raster_m
        new.cluster_shape_mn = self.cluster_shape_mn
        new.static_l1 = self.static_l1
        new._fields = list(self._fields)
        rest = list(values)
        for name, n in zip(self._fields, self._pos, strict=True):
            setattr(new, name, new_from_mlir_values(getattr(self, name), rest[:n]))
            rest = rest[n:]
        assert not rest, len(rest)
        return new

    def grid_shape(self, max_active_clusters, *, loc=None, ip=None):
        """``(cluster m, cluster n, persistent clusters)`` with the cluster count ``min(tiles, max_active_clusters)``, both
        runtime values."""
        return (*self.cluster_shape_mn, min(self.total, Int32(max_active_clusters)))


class TileScheduler:
    """The persistent scheduler over :class:`SchedParams`: cluster ``z`` of the grid takes the linear indices ``z, z + G,
    z + 2 G, ...`` (``G`` the launched cluster count) until the valid-work bound; ``create / initial_work_tile_info /
    get_current_work / advance_to_next_work / num_tiles_executed`` as the role loops expect. The current tile is decoded
    when the scheduler is created and on every advance (not on every read), so a scheduler created before the barrier
    init has its first tile ready when the role loop starts."""

    def __init__(
        self,
        params: SchedParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster,
        num_tiles_executed: Int32,
        current_work: Optional[WorkTileInfo] = None,
    ):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self._current_work_linear_idx = current_work_linear_idx
        self.cta_id_in_cluster = cta_id_in_cluster
        self._num_tiles_executed = num_tiles_executed
        self._current_work = current_work if current_work is not None else self._work_for(current_work_linear_idx)

    def __extract_mlir_values__(self) -> list:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        values.extend(extract_mlir_values(self._current_work))
        values.extend(extract_mlir_values(self.params))
        return values

    def __new_from_mlir_values__(self, values: list) -> "TileScheduler":
        assert len(values) >= 10
        return TileScheduler(
            new_from_mlir_values(self.params, values[10:]),
            new_from_mlir_values(self.num_persistent_clusters, [values[0]]),
            new_from_mlir_values(self._current_work_linear_idx, [values[1]]),
            new_from_mlir_values(self.cta_id_in_cluster, values[2:5]),
            new_from_mlir_values(self._num_tiles_executed, [values[5]]),
            new_from_mlir_values(self._current_work, values[6:10]),
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: SchedParams,
        block_idx,
        grid_dim,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "TileScheduler":
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(params.cluster_shape_mn, loc=loc, ip=ip)
        bidx, bidy, bidz = block_idx
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        return TileScheduler(
            params,
            Int32(num_persistent_clusters),
            Int32(bidz),
            cta_id_in_cluster,
            Int32(0),
        )

    @cute.jit
    def _decode(self, idx: Int32):
        """``(slow, fast, l)`` of linear index ``idx`` (module docstring): the batch slice first, then the band and its
        offset; a full band splits its offset by ``s``, the remainder band by ``S`` (fast index first); ``s = 1`` is one
        divmod by ``S``."""
        p = self.params
        l = Int32(0)
        r = idx
        if cutlass.const_expr(not p.static_l1):
            l, r = divmod(idx, p.fdd_l)
        if cutlass.const_expr(p.swizzle == 1):
            f, sl = divmod(r, p.fdd_S)
        else:
            band, e = divmod(r, p.fdd_band)
            sl_full, f_full = e // p.swizzle, e % p.swizzle
            f_rem, sl_rem = divmod(e, p.fdd_S)
            remainder = band == p.full_bands
            sl = sl_rem if remainder else sl_full
            f = band * p.swizzle + (f_rem if remainder else f_full)
        return sl, f, l

    def _work_for(self, idx: Int32, *, loc=None, ip=None) -> WorkTileInfo:
        p = self.params
        is_valid = idx < p.total
        sl, f, l = self._decode(idx)
        m, n = (sl, f) if cutlass.const_expr(p.raster_m) else (f, sl)
        cm, cn = p.cluster_shape_mn
        coord = (
            cute.arch.make_warp_uniform(Int32(m) * Int32(cm) + self.cta_id_in_cluster[0]),
            cute.arch.make_warp_uniform(Int32(n) * Int32(cn) + self.cta_id_in_cluster[1]),
            cute.arch.make_warp_uniform(Int32(l) + self.cta_id_in_cluster[2]),
        )
        return WorkTileInfo(coord, cute.arch.make_warp_uniform(is_valid))

    def get_current_work(self) -> WorkTileInfo:
        return self._current_work

    def initial_work_tile_info(self) -> WorkTileInfo:
        return self._current_work

    @dsl_user_op
    @cute.jit
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None) -> None:
        self._current_work_linear_idx += Int32(advance_count) * Int32(self.num_persistent_clusters)
        self._num_tiles_executed += Int32(1)
        self._current_work = self._work_for(self._current_work_linear_idx, loc=loc, ip=ip)

    @property
    @cute.jit
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed


def build_grid(st, c: cute.Tensor, max_active_clusters):
    """Place the output tiles on at most the resident number of clusters."""
    gc = cute.zipped_divide(c, tiler=cute.slice_(st.cta_tile_shape_mnk, (None, None, 0)))
    num_ctas_mnl = gc[(0, (None, None, None))].shape
    params = SchedParams(num_ctas_mnl, st.cluster_shape_mn, st.knobs.swizzle_size, st.knobs.raster_along == "m")
    return params, params.grid_shape(max_active_clusters)
