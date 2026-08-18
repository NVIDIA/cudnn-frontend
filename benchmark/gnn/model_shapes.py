# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass(frozen=True)
class GnnBenchmarkShape:
    num_src_nodes: int
    num_dst_nodes: int
    degree: int
    feature_dim: int

    @property
    def num_edges(self) -> int:
        return self.num_dst_nodes * self.degree


MODEL_SHAPES = {
    "small": GnnBenchmarkShape(num_src_nodes=4_096, num_dst_nodes=4_096, degree=16, feature_dim=64),
    "medium": GnnBenchmarkShape(num_src_nodes=65_536, num_dst_nodes=65_536, degree=32, feature_dim=128),
    "large": GnnBenchmarkShape(num_src_nodes=262_144, num_dst_nodes=262_144, degree=64, feature_dim=256),
}
