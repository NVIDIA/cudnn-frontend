# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cudnn.gnn import mha_gat_v2

from gnn._mha_test_utils import MhaTestSuite, graph_data

__all__ = ["graph_data"]


class TestMhaGatV2(MhaTestSuite):
    variant = "gat_v2"
    op = staticmethod(mha_gat_v2)
    weight_size = 8
