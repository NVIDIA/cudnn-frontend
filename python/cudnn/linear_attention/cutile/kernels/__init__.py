# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kernel libraries backing the python execution engines.

``gdn``: the chunked Gated DeltaNet (GDN) cuTile kernels behind
``GdnCuTileEngine`` (forward, backward, and the standalone building blocks —
``chunk_gated_delta_rule*``, ``chunk_local_cumsum``).

``kda``: the chunked Kimi Delta Attention (KDA) cuTile kernels behind
``KdaCuTileEngine`` (``chunk_kda*``, ``chunk_kda_fwd_intra``,
``chunk_kda_bwd``, ``chunk_local_cumsum``).

``common``: the glue kernels both pipelines share.

Both kernel modules carry the same twelve sections in the same order —
helpers, then one device-kernel group per pipeline stage, then the host
launchers mirroring those groups, then the pipeline drivers. A ``*_kernel``
name is a ``@ct.kernel``; the launcher that grids and launches it drops the
suffix.

All public wrappers take THD (token-packed) tensors — ``[total_T, H, D]``
values and ``[total_T, H]`` / ``[total_T, H, K]`` gates — with required
``cu_seqlens`` / ``chunk_indices``, and are called only by the engines.
"""
