# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Place output tiles and batch or K-chunk planes on a persistent cluster grid."""

import cutlass.cute as cute

from cudnn._cutlass_helpers.static_persistent_tile_scheduler import PersistentTileSchedulerParams as SchedParams, StaticPersistentTileScheduler as TileScheduler

__all__ = ["SchedParams", "TileScheduler", "build_grid"]


def build_grid(st, c: cute.Tensor, max_active_clusters):
    tiles = cute.zipped_divide(c, tiler=cute.slice_(st.cta_tile_shape_mnk, (None, None, 0)))
    params = SchedParams(tiles[(0, (None, None, None))].shape, (*st.cluster_shape_mn, 1))
    return params, params.get_grid_shape(max_active_clusters)
