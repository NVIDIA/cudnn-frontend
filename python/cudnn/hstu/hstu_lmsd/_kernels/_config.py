# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch configuration for the HSTU LMSD kernels."""

from dataclasses import dataclass

WARP_SIZE = 32


@dataclass(frozen=True)
class HSTULMSDFwdConfig:
    vector_size: int = 4
    threads_per_row: int = WARP_SIZE
    rows_per_cta: int = 4
    grid_ctas_per_sm: int = 192
    min_blocks_per_mp: int = 8

    @classmethod
    def from_hidden_size(cls, hidden_size: int) -> "HSTULMSDFwdConfig":
        if hidden_size >= 960:
            return cls(vector_size=8, rows_per_cta=4, grid_ctas_per_sm=192, min_blocks_per_mp=6)
        return cls()

    def launch_grid(self, num_rows: int, multiprocessor_count: int) -> int:
        row_blocks = (num_rows + self.rows_per_cta - 1) // self.rows_per_cta
        return min(row_blocks, multiprocessor_count * self.grid_ctas_per_sm)


@dataclass(frozen=True)
class HSTULMSDBwdConfig:
    vector_size: int
    threads_per_row: int
    rows_per_cta: int = 1
    grid_ctas_per_sm: int = 64
    min_blocks_per_mp: int = 12

    @classmethod
    def from_hidden_size(cls, hidden_size: int) -> "HSTULMSDBwdConfig":
        if hidden_size <= 256:
            return cls(vector_size=8, threads_per_row=WARP_SIZE)
        if hidden_size <= 512:
            return cls(vector_size=8, threads_per_row=2 * WARP_SIZE)
        return cls(vector_size=8, threads_per_row=4 * WARP_SIZE, min_blocks_per_mp=8)

    def workspace_rows(self, multiprocessor_count: int) -> int:
        return multiprocessor_count * self.grid_ctas_per_sm

    def launch_grid(self, num_rows: int, multiprocessor_count: int) -> int:
        return min(num_rows, self.workspace_rows(multiprocessor_count))


@dataclass(frozen=True)
class HSTULMSDGradReduceConfig:
    threads: int = 256
    columns_per_cta: int = 4
    prefetch_batch: int = 16

    @property
    def warps(self) -> int:
        return self.threads // WARP_SIZE

    @property
    def rows_per_warp(self) -> int:
        return WARP_SIZE // self.columns_per_cta

    @property
    def rows_per_cta(self) -> int:
        return self.threads // self.columns_per_cta
