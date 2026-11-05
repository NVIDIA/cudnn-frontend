# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from linear_attention.determinism import LinearAttentionDeterminism


class TestKDA(LinearAttentionDeterminism):
    variant = "kda"
