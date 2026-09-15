# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cudnn.gnn import mha_gat

from gnn._mha_test_utils import MhaTestSuite, graph_data

__all__ = ["graph_data"]


class TestMhaGat(MhaTestSuite):
    variant = "gat"
    op = staticmethod(mha_gat)
    weight_size = 16
