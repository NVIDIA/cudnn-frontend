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

"""Linear-attention kernels built on Cutlass primitives.  Per family (``gdn``, ``kda``, ``gdn2``): ``*_prefill_f16.py``,
``*_bprop_f16.py``, ``*_recompute_f16.py`` (state / checkpoint-only recompute), ``*_bprop_summary_f16.py`` (the reverse
state-gradient recurrence), ``*_summary_f16.py`` (fused H+M piece summary) and ``*_chain_prologue_f16.py``, each with its
``*_config.py``.  GDP shares the GDN kernels except its d_v = 64 backward fork ``gdp_bprop_v64_f16.py``; GDN also has the
chunk-factor T pass ``gdn_tinv_f16.py``.  KDA / GDN-2 store no per-chunk H in the forward; the backward recomputes."""
