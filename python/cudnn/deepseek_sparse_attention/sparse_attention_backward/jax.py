# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX API for DeepSeek sparse-attention backward on SM90 and SM100+."""

from __future__ import annotations

from typing import Any, Optional

import jax

from ... import data_type
from ..._jax import JaxApiBase, JaxTensorDesc, TupleDict
from ..._jax.compiler import compile_options_for_target
from .op import BLOCK_TILE, SparseAttentionBackwardOp

_SUPPORTED_COMPUTE_CAPABILITIES = (90, 100, 103, 107)
_SUPPORTED_COMPUTE_CAPABILITY_FAMILIES = (90, 100)


class SparseAttentionBackward(JaxApiBase):
    """JAX callable specialized from sparse-attention tensor metadata."""

    def __init__(
        self,
        sample_q: Any,
        sample_kv: Any,
        sample_out: Any,
        sample_dout: Any,
        sample_lse: Any,
        sample_attn_sink: Any,
        sample_topk_idxs: Any,
        sample_dq: Any | None = None,
        sample_dkv: Any | None = None,
        sample_d_sink: Any | None = None,
        sample_topk_length: Any | None = None,
        softmax_scale: Optional[float] = None,
        block_tile: int = BLOCK_TILE,
        target_compute_capability: int | None = None,
    ) -> None:
        self.q_desc = self._to_tensor_desc(sample_q, "sample_q")
        self.kv_desc = self._to_tensor_desc(sample_kv, "sample_kv")
        self.out_desc = self._to_tensor_desc(sample_out, "sample_out")
        self.dout_desc = self._to_tensor_desc(sample_dout, "sample_dout")
        self.lse_desc = self._to_tensor_desc(sample_lse, "sample_lse")
        self.attn_sink_desc = self._to_tensor_desc(sample_attn_sink, "sample_attn_sink")
        self.topk_idxs_desc = self._to_tensor_desc(sample_topk_idxs, "sample_topk_idxs")
        self.topk_length_desc = None if sample_topk_length is None else self._to_tensor_desc(sample_topk_length, "sample_topk_length")

        self.dq_desc = self._output_desc(
            sample_dq,
            source=self.q_desc,
            cudnn_dtype=self.q_desc.cudnn_dtype,
            shape=self.q_desc.shape,
            name="sample_dq",
        )
        self.dkv_desc = self._output_desc(
            sample_dkv,
            source=self.kv_desc,
            cudnn_dtype=self.kv_desc.cudnn_dtype,
            shape=self.kv_desc.shape,
            name="sample_dkv",
        )
        self.d_sink_desc = self._output_desc(
            sample_d_sink,
            source=self.attn_sink_desc,
            cudnn_dtype=data_type.FLOAT,
            shape=self.attn_sink_desc.shape,
            name="sample_d_sink",
            init_value=0.0,
        )
        self._op = SparseAttentionBackwardOp(
            q=self.q_desc,
            kv=self.kv_desc,
            output=self.out_desc,
            doutput=self.dout_desc,
            lse=self.lse_desc,
            attn_sink=self.attn_sink_desc,
            topk_idxs=self.topk_idxs_desc,
            topk_length=self.topk_length_desc,
            dq=self.dq_desc,
            dkv=self.dkv_desc,
            d_sink=self.d_sink_desc,
            softmax_scale=softmax_scale,
            block_tile=block_tile,
        )
        self.target_compute_capability = target_compute_capability
        self.compute_capability: int | None = None

    @staticmethod
    def _output_desc(
        sample: Any | None,
        *,
        source: JaxTensorDesc,
        cudnn_dtype: data_type,
        shape: tuple[int, ...],
        name: str,
        init_value: bool | int | float | None = None,
    ) -> JaxTensorDesc:
        if sample is not None:
            desc = JaxApiBase._to_tensor_desc(sample, name, init_value=init_value)
        else:
            desc = source.compact_like(
                cudnn_dtype=cudnn_dtype,
                shape=shape,
                name=name,
                init_value=init_value,
            )
        return desc

    def check_support(self) -> bool:
        self._op.check_support()
        compute_capability = self._resolve_compute_capability(
            self.target_compute_capability,
            _SUPPORTED_COMPUTE_CAPABILITIES,
            "SparseAttentionBackward",
        )
        family = self._compute_capability_family(compute_capability, _SUPPORTED_COMPUTE_CAPABILITY_FAMILIES)
        if family == 100 and self.q_desc.cudnn_dtype != data_type.BFLOAT16:
            raise ValueError("SparseAttentionBackward on SM100+ currently requires bfloat16 inputs")
        self.compute_capability = compute_capability
        return True

    def __call__(
        self,
        q: Any,
        kv: Any,
        out: Any,
        dout: Any,
        lse: Any,
        attn_sink: Any,
        topk_idxs: Any,
        topk_length: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        if (topk_length is None) != (self.topk_length_desc is None):
            raise ValueError("topk_length presence must match sample_topk_length")

        inputs = (q, kv, out, dout, lse, attn_sink, topk_idxs)
        q_desc = self.q_desc.with_divisibility(
            (None, None, self._op.head_dim)
        )
        kv_desc = self.kv_desc.with_divisibility(
            (None, self._op.head_dim)
        )
        out_desc = self.out_desc.with_divisibility(
            (None, None, self._op.head_dim_v)
        )
        dout_desc = self.dout_desc.with_divisibility(
            (None, None, self._op.head_dim_v)
        )
        input_descs = (
            q_desc,
            kv_desc,
            out_desc,
            dout_desc,
            self.lse_desc,
            self.attn_sink_desc,
            self.topk_idxs_desc,
        )
        if topk_length is not None:
            inputs += (topk_length,)
            input_descs += (self.topk_length_desc,)

        dq_desc = self.dq_desc.with_divisibility(
            (None, None, self._op.head_dim)
        )
        dkv_desc = self.dkv_desc.with_divisibility(
            (None, self._op.head_dim)
        )

        dq, dkv, d_sink = self._call_kernel(
            inputs,
            launch=self._launch_kernel,
            output_descs=(dq_desc, dkv_desc, self.d_sink_desc),
            input_descs=input_descs,
            workspace_descs=self._workspace_descs(),
            compile_options=compile_options_for_target(self.compute_capability),
        )
        return TupleDict(dq=dq, dkv=dkv, d_sink=d_sink)

    def _workspace_descs(self) -> tuple[JaxTensorDesc, ...]:
        if self._architecture_family == 100:
            q_rounded = _round_up(self._op.total_seqlen_q, 8)
            kv_rounded = _round_up(self._op.total_seqlen_kv, 8)
            head_dim_rounded = _round_up(self._op.head_dim, 8)
            return (
                self.q_desc.compact_like(
                    cudnn_dtype=data_type.UINT8,
                    shape=(1, self._op.num_heads, q_rounded, 8),
                    name="workspace_lse_odo",
                    init_value=0,
                ),
                self.q_desc.compact_like(
                    cudnn_dtype=data_type.UINT8,
                    shape=(1, 1, kv_rounded, head_dim_rounded * 4),
                    name="workspace_dkv",
                    init_value=0,
                ),
            )

        q_rounded = _round_up(self._op.total_seqlen_q, BLOCK_TILE)
        kv_rounded = _round_up(self._op.total_seqlen_kv, BLOCK_TILE)
        head_dim_rounded = _round_up(self._op.head_dim, 32)
        workspaces = (
            self.q_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=(1, q_rounded, self._op.num_heads),
                name="workspace_dpsum",
            ),
            self.q_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=(1, q_rounded, self._op.num_heads),
                name="workspace_lse_log2",
            ),
            self.q_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=(1, 1, kv_rounded * head_dim_rounded),
                name="workspace_dkv_accum",
                init_value=0.0,
            ),
        )
        if self.topk_length_desc is None:
            workspaces += (
                self.q_desc.compact_like(
                    cudnn_dtype=data_type.INT32,
                    shape=(1,),
                    name="workspace_dummy_topk_length",
                ),
            )
        return workspaces

    @property
    def _architecture_family(self) -> int:
        if self.compute_capability is None:
            raise RuntimeError("check_support() must resolve the compute capability before lowering")
        family = self._compute_capability_family(self.compute_capability, _SUPPORTED_COMPUTE_CAPABILITY_FAMILIES)
        if family is None:
            raise RuntimeError(f"No sparse-attention backward kernel for SM{self.compute_capability}")
        return family

    def _launch_kernel(
        self,
        stream: Any,
        *arguments: Any,
    ) -> None:
        workspace_count = len(self._workspace_descs())
        input_count = len(arguments) - 3 - workspace_count
        inputs = arguments[:input_count]
        outputs = arguments[input_count : input_count + 3]
        workspaces = arguments[input_count + 3 :]
        if self._architecture_family == 90:
            self._launch_sm90(inputs, outputs, workspaces, stream)
        else:
            self._launch_sm100(inputs, outputs, workspaces, stream)

    def _launch_sm100(
        self,
        inputs: tuple[Any, ...],
        outputs: tuple[Any, ...],
        workspaces: tuple[Any, ...],
        stream: Any,
    ) -> None:
        import cutlass

        from .dsa_bwd_sm100 import FlashAttentionDSABackwardSm100

        q, kv, out, dout, lse, attn_sink, topk_idxs, *optional = inputs
        topk_length = optional[0] if optional else None
        dq, dkv, d_sink = outputs
        workspace_lse_odo, workspace_dkv = workspaces
        kernel = FlashAttentionDSABackwardSm100(
            head_dim=self._op.head_dim,
            head_dim_v=self._op.head_dim_v,
            block_tile=self._op.block_tile,
            max_topk=self._op.max_topk,
        )
        problem_shape = (
            cutlass.Int32(self._op.total_seqlen_q),
            cutlass.Int32(self._op.total_seqlen_kv),
            cutlass.Int32(self._op.head_dim),
            (cutlass.Int32(self._op.num_heads), cutlass.Int32(1)),
        )
        kernel(
            problem_shape,
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            topk_length,
            dq,
            dkv,
            d_sink,
            workspace_lse_odo,
            workspace_dkv,
            cutlass.Float32(self._op.softmax_scale),
            stream,
        )

    def _launch_sm90(
        self,
        inputs: tuple[Any, ...],
        outputs: tuple[Any, ...],
        workspaces: tuple[Any, ...],
        stream: Any,
    ) -> None:
        import cutlass

        from .dsa_bwd_sm90 import (
            FlashAttentionDSABackwardSm90,
            _FlashAttentionDSABackwardPostprocessSm90,
            _FlashAttentionDSABackwardPreprocessSm90,
        )

        q, kv, out, dout, lse, attn_sink, topk_idxs, *optional = inputs
        dq, dkv, d_sink = outputs
        workspace_dpsum, workspace_lse_log2, workspace_dkv_accum, *optional_workspaces = workspaces
        topk_length = optional[0] if optional else optional_workspaces[0]

        q4 = _prepend_unit_dim(q)
        kv4 = _as_bshd_kv(kv)
        out4 = _prepend_unit_dim(out)
        dout4 = _prepend_unit_dim(dout)
        lse4 = _prepend_unit_dim(lse)
        topk4 = _prepend_unit_dim(topk_idxs)
        topk_length4 = _prepend_unit_dim(topk_length)
        dq4 = _prepend_unit_dim(dq)
        dkv4 = _as_bshd_kv(dkv)

        dtype = q.element_type
        preprocess = _FlashAttentionDSABackwardPreprocessSm90(
            dtype,
            self._op.head_dim_v,
            90,
            BLOCK_TILE,
            num_threads=256,
        )
        preprocess(
            out4,
            dout4,
            workspace_dpsum,
            lse4,
            workspace_lse_log2,
            attn_sink,
            d_sink,
            None,
            None,
            None,
            stream,
        )

        backward = FlashAttentionDSABackwardSm90(
            dtype,
            self._op.head_dim,
            self._op.head_dim_v,
            self._op.num_heads,
            tile_m=BLOCK_TILE,
            tile_n=BLOCK_TILE,
            KV_stage=1,
            PdS_stage=1,
            SdP_swapAB=False,
            dKV_swapAB=False,
            dQ_swapAB=False,
            num_threads=256,
            have_topk_length=self.topk_length_desc is not None,
            max_topk=self._op.max_topk,
        )
        backward(
            q4,
            kv4,
            dout4,
            workspace_lse_log2,
            workspace_dpsum,
            dq4,
            workspace_dkv_accum,
            topk4,
            topk_length4,
            cutlass.Float32(self._op.softmax_scale),
            stream,
        )

        hdim_chunk = 64 if self._op.head_dim == 576 else min(128, self._op.head_dim)
        postprocess = _FlashAttentionDSABackwardPostprocessSm90(
            dtype,
            hdim_chunk=hdim_chunk,
            tile_n=BLOCK_TILE,
            head_dim=self._op.head_dim,
            num_threads=hdim_chunk,
            N_hdim_chunks=self._op.head_dim // hdim_chunk,
        )
        postprocess(
            workspace_dkv_accum,
            dkv4,
            cutlass.Int32(self._op.total_seqlen_kv),
            stream,
        )


def _prepend_unit_dim(tensor: Any) -> Any:
    """Return a CuTe view with a leading batch dimension of one."""

    import cutlass.cute as cute

    leading_stride = tensor.shape[0] * tensor.stride[0]
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout((1, *tensor.shape), stride=(leading_stride, *tensor.stride)),
    )


def _as_bshd_kv(tensor: Any) -> Any:
    """Return a CuTe ``(1, sequence, 1, head_dim)`` MQA view."""

    import cutlass.cute as cute

    leading_stride = tensor.shape[0] * tensor.stride[0]
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(
            (1, tensor.shape[0], 1, tensor.shape[1]),
            stride=(leading_stride, tensor.stride[0], tensor.stride[0], tensor.stride[1]),
        ),
    )


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@jax.jit(static_argnames=("softmax_scale", "block_tile", "target_compute_capability"))
def sparse_attention_backward_wrapper(
    q: Any,
    kv: Any,
    out: Any,
    dout: Any,
    lse: Any,
    attn_sink: Any,
    topk_idxs: Any,
    softmax_scale: Optional[float] = None,
    topk_length: Any | None = None,
    block_tile: int = BLOCK_TILE,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Compute DeepSeek sparse-attention gradients from JAX arrays."""

    return SparseAttentionBackward(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        sample_topk_length=topk_length,
        softmax_scale=softmax_scale,
        block_tile=block_tile,
        target_compute_capability=target_compute_capability,
    )(q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length)


__all__ = ["SparseAttentionBackward", "sparse_attention_backward_wrapper"]
