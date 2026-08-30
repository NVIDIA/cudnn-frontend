# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Device expansion of per-expert SF atoms into fixed-capacity WGrad ABI."""

from __future__ import annotations

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32


class Mxfp8TrainingScaleExpandKernel:
    """Expand 128-row SF extents into 256-row fixed-capacity segments."""

    _threads = 256
    _atom_bytes = 512
    _neutral_e8m0 = 127

    def __init__(
        self,
        *,
        non_k_size: int,
        expert_count: int,
        source_sf_padding: int,
    ) -> None:
        self.non_k_size = int(non_k_size)
        self.expert_count = int(expert_count)
        self.source_sf_padding = int(source_sf_padding)
        if self.non_k_size <= 0 or self.non_k_size % 128:
            raise ValueError("WGrad scale non-K size must be divisible by 128")
        if self.expert_count <= 0:
            raise ValueError("WGrad scale expansion requires experts")
        if self.source_sf_padding <= 0 or self.source_sf_padding % 128:
            raise ValueError("WGrad source SF padding must be a positive multiple of 128")

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        valid_counts: cute.Tensor,
        expert_offsets: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        output_bytes = cute.size(output)
        self._kernel(
            source,
            valid_counts,
            expert_offsets,
            output,
        ).launch(
            grid=[
                output_bytes // self._threads,
                1,
                1,
            ],
            block=[self._threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def _kernel(
        self,
        source: cute.Tensor,
        valid_counts: cute.Tensor,
        expert_offsets: cute.Tensor,
        output: cute.Tensor,
    ) -> None:
        linear = cute.arch.block_idx()[0] * Int32(self._threads) + cute.arch.thread_idx()[0]

        atom_bytes: cutlass.Constexpr[int] = self._atom_bytes
        non_k_atoms: cutlass.Constexpr[int] = self.non_k_size // 128
        atom = linear // Int32(atom_bytes)
        byte_in_atom = linear % Int32(atom_bytes)
        value = cutlass.Uint8(self._neutral_e8m0)
        target_atom_base = Int32(0)
        source_atom_base = Int32(0)
        previous_end = Int32(0)

        for expert in cutlass.range_constexpr(self.expert_count):
            end = Int32(expert_offsets[expert])
            target_token_atoms = (end - previous_end) // Int32(128)
            source_token_atoms = ((Int32(valid_counts[expert]) + Int32(self.source_sf_padding - 1)) // Int32(self.source_sf_padding)) * Int32(
                self.source_sf_padding // 128
            )
            target_atom_count = Int32(non_k_atoms) * target_token_atoms
            in_expert = (atom >= target_atom_base) & (atom < target_atom_base + target_atom_count)
            if in_expert & (target_token_atoms > Int32(0)):
                relative_atom = atom - target_atom_base
                hidden_atom = relative_atom // target_token_atoms
                token_atom = relative_atom % target_token_atoms
                if token_atom < source_token_atoms:
                    source_atom = source_atom_base + hidden_atom * source_token_atoms + token_atom
                    value = source[source_atom * Int32(atom_bytes) + byte_in_atom]
            target_atom_base += target_atom_count
            source_atom_base += Int32(non_k_atoms) * source_token_atoms
            previous_end = end

        output[linear] = value


__all__ = ["Mxfp8TrainingScaleExpandKernel"]
