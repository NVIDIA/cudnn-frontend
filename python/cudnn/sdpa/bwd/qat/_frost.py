# SPDX-License-Identifier: Apache-2.0

"""Prepared, allocation-free FROST QAT backward orchestration.

Version/capability checks in api.py run before importing this module.
Q/delta and KV preprocessing reuse the public Triton quantizers; dV/dS
uses CuTe DSL and dQ/dK use explicit output-buffer batched GEMMs.
"""

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import torch

from ._frost_kernel import compile as compile_core
from ._interface import _workspace_tensor
from ._nvfp4 import fake_quantize_kv, fake_quantize_q
from ._workspace import _align_up


def workspace_layout(heads: int, sequence: int, head_chunk: int):
    """Q/K/V in native BSHD, raw delta in BHS, chunk-local BF16 dS."""
    entries, offset = [], 0
    for shape, dtype in (
        *((((1, sequence, heads, 128), torch.bfloat16),) * 3),
        ((1, heads, sequence), torch.float32),
        ((1, head_chunk, sequence, sequence), torch.bfloat16),
    ):
        offset = _align_up(offset)
        entries.append((offset, shape, dtype))
        offset += math.prod(shape) * dtype.itemsize
    return tuple(entries), _align_up(offset)


@dataclass(frozen=True)
class PreparedBackward:
    """Own compiled launchers, never change module globals or compile at execute."""

    heads: int
    sequence: int
    head_chunk: int
    entries: tuple
    quant_q: object
    quant_kv: object
    core: object
    source_strides: tuple
    fake_strides: tuple

    @classmethod
    def compile(cls, heads: int, sequence: int, head_chunk: int):
        entries, _ = workspace_layout(heads, sequence, head_chunk)
        source = (heads * sequence * 128, sequence * 128, 128, 1)
        # Logical BHSD views over the quantizers' BSHD destination.
        fake = (heads * sequence * 128, 128, heads * 128, 1)
        grid = (sequence // 32, heads, 1)
        q_kernel = fake_quantize_q.warmup(
            torch.bfloat16,
            torch.bfloat16,
            *source,
            *fake,
            heads,
            sequence,
            32,
            128,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            grid=grid,
            num_warps=4,
            num_stages=2,
        )
        kv_kernel = fake_quantize_kv.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            *source,
            *fake,
            heads,
            sequence,
            32,
            128,
            grid=grid,
            num_warps=4,
            num_stages=2,
        )
        core = compile_core(1, heads, heads, sequence, sequence, qh_chunk=head_chunk)
        # Index CompiledKernel now: materializes launch handles without a launch.
        # Execute bypasses Triton's JITFunction/cache-key construction entirely.
        return cls(heads, sequence, head_chunk, entries, q_kernel[grid], kv_kernel[grid], core, source, fake)

    @torch.no_grad()
    def execute(self, q, k, v, o, do, lse, dq, dk, dv, workspace, scale):
        fake_q, fake_k, fake_v, delta, ds = (_workspace_tensor(workspace, entry) for entry in self.entries)
        stream = torch.cuda.current_stream(q.device).cuda_stream
        self.quant_q(
            q,
            fake_q,
            *self.source_strides,
            *self.fake_strides,
            self.heads,
            self.sequence,
            32,
            128,
            o,
            do,
            delta,
            stream=stream,
        )
        self.quant_kv(
            k,
            v,
            fake_k,
            fake_v,
            *self.source_strides,
            *self.fake_strides,
            self.heads,
            self.sequence,
            32,
            128,
            stream=stream,
        )
        # Only metadata views, including the BSHD views consumed by TMA.
        do_view, dv_view = do.permute(0, 2, 1, 3), dv.permute(0, 2, 1, 3)
        q_heads, k_heads = fake_q[0].permute(1, 0, 2), fake_k[0].permute(1, 0, 2)
        for base in range(0, self.heads, self.head_chunk):
            self.core(
                fake_q,
                do_view,
                fake_k,
                fake_v,
                dv_view,
                ds,
                lse,
                delta,
                (1, self.heads, self.heads, self.sequence, self.sequence, self.head_chunk),
                scale,
                scale * math.log2(math.e),
                1.0,
                1.0,
                1.0,
                scale,
                cutlass.Int32(base),
                cutlass.Int32(self.sequence),
                cuda.CUstream(stream),
            )
            selected = slice(base, base + self.head_chunk)
            # B=1 is intentional: no flattening of noncompact B/H dimensions
            # and no implicit reshape copies in torch.matmul's batching path.
            torch.bmm(ds[0].transpose(1, 2), k_heads[selected], out=dq[0, selected])
            torch.bmm(ds[0], q_heads[selected], out=dk[0, selected])
