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

"""Linear-attention kernels built on Cutlass primitives.

GDN: ``gdn_prefill_f16.py`` (prefill, optional per-chunk H output) with
``gdn_prefill_config.py``, and ``gdn_bprop_f16.py`` (backward) with
``gdn_bprop_config.py``.  The tile primitives come from the shared
``cudnn.frost.tile_dsl``; ``thd.py`` lives in ``..common``.

KDA: ``kda_prefill_f16.py`` (prefill, BT=16, per-key-channel decay) with
``kda_prefill_config.py``; ``kda_bprop_f16.py`` (backward) is a STUB.

GDN-2: ``gdn2_prefill_f16.py`` (prefill, BT=16, per-key erase + per-value
write gates) with ``gdn2_prefill_config.py``; ``gdn2_bprop_f16.py`` is a
STUB.  KDA / GDN-2 do not store per-chunk H in the forward (small chunk);
the backward recomputes."""
