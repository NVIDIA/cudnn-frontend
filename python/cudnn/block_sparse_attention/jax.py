# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""JAX adapters for the CuTe DSL block-sparse attention kernels."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from .. import data_type
from .._cute_compiler import compile_options_for_target
from .._jax import JaxApiBase, JaxTensorDesc, TupleDict
from .._jax.layout import mode_from_layout, to_public_axes
from .op import (
    BlockSparseAttentionBackwardOp,
    BlockSparseAttentionForwardOp,
    SUPPORTED_BACKWARD_COMPUTE_CAPABILITIES,
    SUPPORTED_FORWARD_COMPUTE_CAPABILITIES,
    ceil_div,
    compute_capability_family,
)


_DATA_LAYOUTS = ("bhsd", "bshd")


def _data_mode(layout: str) -> tuple[int, ...]:
    if not isinstance(layout, str):
        raise TypeError(f"layout must be a string, got {type(layout).__name__}")
    normalized = "".join(
        character for character in layout.lower() if character.isalpha()
    )
    if normalized not in _DATA_LAYOUTS:
        raise ValueError(f"layout must be one of {_DATA_LAYOUTS}, got {layout!r}")
    return mode_from_layout(normalized.upper(), kernel_axes="BHSD")


def _shape_sample(
    shape: tuple[int, ...], dtype: Any, mode: tuple[int, ...] | None = None
) -> Any:
    return jax.ShapeDtypeStruct(to_public_axes(shape, mode), dtype)


def _optional_signature(
    value: Any | None, desc: JaxTensorDesc | None, name: str
) -> None:
    if (value is None) != (desc is None):
        expected = "omitted" if desc is None else "provided"
        raise ValueError(f"{name} must be {expected} for this specialized callable")
    if desc is not None:
        JaxApiBase._check_tensor_signature(value, desc)


def _default_forward_sparse_block_size(target: int) -> int:
    return 64 if compute_capability_family(target) in {90, 120} else 128


def _default_backward_sparse_block_size(target: int) -> int:
    return 64 if compute_capability_family(target) == 90 else 128


def _default_bucket_size(
    target: int, sparse_block_size: int, num_q_blocks: int, num_heads: int
) -> int:
    if compute_capability_family(target) == 90:
        return 384
    if sparse_block_size == 64:
        if num_q_blocks >= 3000:
            return 1024
        return 1088 if num_q_blocks < 2048 or num_q_blocks >= 8192 else 1152
    if num_q_blocks >= 4096 and num_heads <= 1:
        return 256
    if num_q_blocks >= 2048:
        return 512
    return 384


def _round_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def _sm100_blk64_use_clc(
    *,
    batch: int,
    heads: int,
    seqlen_q: int,
    block_sparse_num: int,
    has_variable_block_nums: bool,
) -> bool:
    if has_variable_block_nums:
        return True
    num_m_blocks = ceil_div(seqlen_q, 64)
    if num_m_blocks >= 8192 and block_sparse_num >= 512:
        return True
    if heads == 1:
        return False
    return (
        num_m_blocks >= 128
        and batch * heads * num_m_blocks >= 512
        and block_sparse_num <= (64 if heads == 2 else 128)
    )


def _resolve_sm100_blk64_use_clc(
    *,
    kv_splits: int,
    requested: bool | None,
    batch: int,
    heads: int,
    seqlen_q: int,
    block_sparse_num: int,
    has_variable_block_nums: bool,
) -> bool:
    if kv_splits > 1:
        return False
    if requested is not None:
        return requested
    return _sm100_blk64_use_clc(
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        block_sparse_num=block_sparse_num,
        has_variable_block_nums=has_variable_block_nums,
    )


def _sm100_blk64_auto_kv_splits(block_sparse_num: int) -> int:
    if block_sparse_num >= 900:
        return 8
    if block_sparse_num >= 450:
        return 4
    if block_sparse_num >= 256:
        return 2
    return 1


def _sm100_blk64_uses_int64_kv_strides(desc: JaxTensorDesc) -> bool:
    coord_stride_limit = 1 << 27
    batch, heads, seqlen_k, _ = desc.shape
    stride_b, stride_h, stride_s, stride_d = desc.stride
    rank6_shape = (64, 64, 2, heads, ceil_div(seqlen_k, 64), batch)
    rank6_stride = (
        stride_s,
        stride_d,
        64 * stride_d,
        stride_h,
        64 * stride_s,
        stride_b,
    )
    if any(stride < 0 or stride > jnp.iinfo(jnp.int32).max for stride in rank6_stride):
        return True
    return (rank6_shape[4] > 1 and rank6_stride[4] >= coord_stride_limit) or (
        rank6_shape[5] > 1 and rank6_stride[5] >= coord_stride_limit
    )


class BlockSparseAttentionForward(JaxApiBase):
    """JAX callable specialized from block-sparse attention metadata."""

    def __init__(
        self,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        sample_q2k_block_index: Any,
        *,
        sample_block_sizes: Any | None = None,
        sample_q2k_block_nums: Any | None = None,
        sample_o: Any | None = None,
        sample_lse: Any | None = None,
        block_sparse_num: int | None = None,
        sparse_block_size: int | None = None,
        allow_empty_block_nums: bool = False,
        softmax_scale: float | None = None,
        pack_gqa: bool | None = None,
        layout: str = "bhsd",
        kv_splits: int | str = 1,
        use_clc: bool | None = None,
        target_compute_capability: int | None = None,
    ) -> None:
        self.layout = layout
        self.data_mode = _data_mode(layout)
        self.compute_capability = self._resolve_compute_capability(
            target_compute_capability,
            SUPPORTED_FORWARD_COMPUTE_CAPABILITIES,
            "BlockSparseAttentionForward",
        )
        self.compute_capability_family = compute_capability_family(
            self.compute_capability
        )
        self.q_desc = self._to_tensor_desc(sample_q, "sample_q", mode=self.data_mode)
        self.k_desc = self._to_tensor_desc(sample_k, "sample_k", mode=self.data_mode)
        self.v_desc = self._to_tensor_desc(sample_v, "sample_v", mode=self.data_mode)
        self.block_index_desc = self._to_tensor_desc(
            sample_q2k_block_index, "sample_q2k_block_index"
        )
        self.block_sizes_desc = (
            None
            if sample_block_sizes is None
            else self._to_tensor_desc(sample_block_sizes, "sample_block_sizes")
        )
        self.block_nums_desc = (
            None
            if sample_q2k_block_nums is None
            else self._to_tensor_desc(sample_q2k_block_nums, "sample_q2k_block_nums")
        )

        if self.q_desc.ndim != 4 or self.v_desc.ndim != 4:
            raise ValueError("sample_q and sample_v must be rank 4")
        batch, num_q_heads, seqlen_q, head_dim = self.q_desc.shape
        value_dim = self.v_desc.shape[3]
        canonical_output_shape = (batch, num_q_heads, seqlen_q, value_dim)
        if sample_o is None:
            self.o_desc = self._to_tensor_desc(
                _shape_sample(
                    canonical_output_shape, self.q_desc.dtype, self.data_mode
                ),
                "sample_o",
                mode=self.data_mode,
            )
        else:
            self.o_desc = self._to_tensor_desc(
                sample_o, "sample_o", mode=self.data_mode
            )
        if sample_lse is None:
            self.lse_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct((batch, num_q_heads, seqlen_q), jnp.float32),
                "sample_lse",
            )
        else:
            self.lse_desc = self._to_tensor_desc(sample_lse, "sample_lse")

        self.sparse_block_size = (
            _default_forward_sparse_block_size(self.compute_capability)
            if sparse_block_size is None
            else sparse_block_size
        )
        capacity = (
            self.block_index_desc.shape[3] if self.block_index_desc.ndim == 4 else 0
        )
        self.block_sparse_num = (
            capacity if block_sparse_num is None else block_sparse_num
        )
        self.softmax_scale = (
            head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
        )
        if isinstance(kv_splits, str):
            if kv_splits != "auto":
                raise ValueError("kv_splits string value must be 'auto'")
            if not (
                self.compute_capability_family == 100 and self.sparse_block_size == 64
            ):
                raise ValueError(
                    "kv_splits='auto' is available only on the SM100-family blk64 path"
                )
            auto_block_count = (
                self.block_sparse_num if self.block_sparse_num > 0 else capacity
            )
            self.kv_splits = _sm100_blk64_auto_kv_splits(auto_block_count)
        else:
            self.kv_splits = kv_splits
        self.use_clc = use_clc
        self._op = BlockSparseAttentionForwardOp(
            q=self.q_desc,
            k=self.k_desc,
            v=self.v_desc,
            block_index=self.block_index_desc,
            output=self.o_desc,
            lse=self.lse_desc,
            block_sizes=self.block_sizes_desc,
            block_nums=self.block_nums_desc,
            block_sparse_num=self.block_sparse_num,
            sparse_block_size=self.sparse_block_size,
            softmax_scale=self.softmax_scale,
            pack_gqa=pack_gqa,
            allow_empty_block_nums=allow_empty_block_nums,
            kv_splits=kv_splits,
            use_clc=use_clc,
            target_compute_capability=self.compute_capability,
        )
        self.check_support()

    def check_support(self) -> bool:
        return self._op.check_support()

    def __call__(
        self,
        q: Any,
        k: Any,
        v: Any,
        q2k_block_index: Any,
        block_sizes: Any | None = None,
        q2k_block_nums: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        self._check_tensor_signature(q, self.q_desc, mode=self.data_mode)
        self._check_tensor_signature(k, self.k_desc, mode=self.data_mode)
        self._check_tensor_signature(v, self.v_desc, mode=self.data_mode)
        self._check_tensor_signature(q2k_block_index, self.block_index_desc)
        _optional_signature(block_sizes, self.block_sizes_desc, "block_sizes")
        _optional_signature(q2k_block_nums, self.block_nums_desc, "q2k_block_nums")

        inputs: list[Any] = [q, k, v, q2k_block_index]
        bindings: list[tuple[JaxTensorDesc, tuple[int, ...] | None]] = [
            (self.q_desc, self.data_mode),
            (self.k_desc, self.data_mode),
            (self.v_desc, self.data_mode),
            (self.block_index_desc, None),
        ]
        if self.block_sizes_desc is not None:
            inputs.append(block_sizes)
            bindings.append((self.block_sizes_desc, None))

        effective_block_nums = q2k_block_nums
        effective_block_nums_desc = self.block_nums_desc
        needs_materialized_block_nums = (
            self.compute_capability_family == 90 and effective_block_nums is None
        )
        if needs_materialized_block_nums:
            block_nums_shape = (
                self._op.batch,
                self.block_index_desc.shape[1],
                self._op.num_q_blocks,
            )
            effective_block_nums = jnp.full(
                block_nums_shape, self.block_sparse_num, dtype=jnp.int32
            )
            effective_block_nums_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct(block_nums_shape, jnp.int32),
                "effective_q2k_block_nums",
            )
        if effective_block_nums is not None:
            inputs.append(effective_block_nums)
            bindings.append((effective_block_nums_desc, None))

        split_offsets_desc = None
        if self.kv_splits > 1:
            if effective_block_nums is None:
                valid_kv = jnp.full(
                    (
                        self._op.batch,
                        self.block_index_desc.shape[1],
                        self._op.num_q_blocks,
                    ),
                    self.block_sparse_num,
                    dtype=jnp.int32,
                )
            else:
                valid_kv = jnp.maximum(effective_block_nums.astype(jnp.int32), 0)
            split_ids = jnp.arange(self.kv_splits + 1, dtype=jnp.int32)
            quotient = valid_kv // self.kv_splits
            remainder_even = valid_kv - quotient * self.kv_splits
            even_offsets = (
                quotient[..., None] * split_ids
                + (remainder_even[..., None] * split_ids + self.kv_splits - 1)
                // self.kv_splits
            )
            aligned_base = (quotient // 8) * 8
            aligned_remainder = valid_kv - aligned_base * self.kv_splits
            aligned_offsets = aligned_base[..., None] * split_ids + jnp.minimum(
                aligned_remainder[..., None],
                split_ids * 8,
            )
            split_offsets = jnp.where(
                (aligned_base == 0)[..., None],
                even_offsets,
                jnp.minimum(aligned_offsets, valid_kv[..., None]),
            ).astype(jnp.int32)
            split_offsets_desc = self._to_tensor_desc(
                jax.ShapeDtypeStruct(split_offsets.shape, split_offsets.dtype),
                "split_offsets",
            )
            inputs.append(split_offsets)
            bindings.append((split_offsets_desc, None))

        self._forward_has_effective_block_nums = effective_block_nums is not None
        self._forward_has_split_offsets = split_offsets_desc is not None
        self._forward_input_count = len(inputs)

        workspace_descs: tuple[JaxTensorDesc, ...] = ()
        if self.kv_splits > 1:
            partial_shape = (
                self._op.batch,
                self.kv_splits * self._op.num_q_heads,
                self._op.seqlen_q,
                self._op.value_dim,
            )
            partial_lse_shape = partial_shape[:3]
            partial_o_desc = self.q_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=partial_shape,
                name="partial_o_workspace",
            )
            partial_lse_desc = self.lse_desc.compact_like(
                cudnn_dtype=data_type.FLOAT,
                shape=partial_lse_shape,
                name="partial_lse_workspace",
            )
            workspace_descs = (partial_o_desc, partial_lse_desc)

        o, lse = self._call_kernel(
            tuple(inputs),
            launch=self._launch,
            output_descs=(self.o_desc, self.lse_desc),
            workspace_descs=workspace_descs,
            input_spec=tuple(
                self._to_tensor_spec(desc, mode=mode) for desc, mode in bindings
            ),
            output_spec=(
                self._to_tensor_spec(self.o_desc, mode=self.data_mode),
                self._to_tensor_spec(self.lse_desc),
            ),
            compile_options=compile_options_for_target(self.compute_capability),
        )
        return TupleDict(o_tensor=o, lse_tensor=lse)

    def _launch(self, stream: Any, *arguments: Any) -> None:
        import cutlass
        import cutlass.cute as cute

        inputs = list(arguments[: self._forward_input_count])
        buffers = list(arguments[self._forward_input_count :])
        q, k, v, block_index = inputs[:4]
        cursor = 4
        block_sizes = None
        if self.block_sizes_desc is not None:
            block_sizes = inputs[cursor]
            cursor += 1
        block_nums = None
        if self._forward_has_effective_block_nums:
            block_nums = inputs[cursor]
            cursor += 1
        split_offsets = None
        if self._forward_has_split_offsets:
            split_offsets = inputs[cursor]
            cursor += 1
        if cursor != len(inputs):
            raise RuntimeError("Unexpected BSA forward kernel inputs")

        output, lse = buffers[:2]
        if self.kv_splits > 1:
            partial_output, partial_lse = buffers[2:]
            kernel_output, kernel_lse = partial_output, partial_lse
        else:
            kernel_output, kernel_lse = output, lse

        if self.compute_capability_family == 90:
            from .csrc.fwd.sm90_blk64.bsa_fwd_sm90 import (
                BlockSparseAttnForwardSm90Blk64,
            )

            def select(tensor: Any, modes: tuple[int, ...]) -> Any:
                return cute.make_tensor(
                    tensor.iterator, cute.select(tensor.layout, mode=modes)
                )

            q_kernel = select(q, (2, 3, 1, 0))
            k_kernel = select(k, (2, 3, 1, 0))
            v_kernel = select(v, (3, 2, 1, 0))
            o_kernel = select(kernel_output, (2, 3, 1, 0))
            lse_kernel = select(kernel_lse, (2, 1, 0))
            index_kernel = select(block_index, (3, 2, 1, 0))
            nums_kernel = select(block_nums, (2, 1, 0))
            if block_sizes is None:
                sizes_kernel = nums_kernel
            elif self.block_sizes_desc.ndim == 1:
                sizes_kernel = cute.make_tensor(
                    block_sizes.iterator,
                    cute.make_layout(
                        (
                            self._op.num_kv_blocks,
                            self.block_index_desc.shape[1],
                            self._op.batch,
                        ),
                        stride=(block_sizes.stride[0], 0, 0),
                    ),
                )
            elif self.block_sizes_desc.ndim == 2:
                sizes_kernel = cute.make_tensor(
                    block_sizes.iterator,
                    cute.make_layout(
                        (
                            self._op.num_kv_blocks,
                            self.block_index_desc.shape[1],
                            self._op.batch,
                        ),
                        stride=(block_sizes.stride[1], 0, block_sizes.stride[0]),
                    ),
                )
            else:
                sizes_kernel = select(block_sizes, (2, 1, 0))
            split_kernel = (
                nums_kernel
                if split_offsets is None
                else select(split_offsets, (3, 2, 1, 0))
            )
            kernel = BlockSparseAttnForwardSm90Blk64(
                gqa_ratio=self._op.gqa_ratio,
                head_dim=self._op.head_dim,
                value_dim=self._op.value_dim,
                blocksparse_blocksize_q=64,
                blocksparse_blocksize_k=64,
                dtype=q.element_type,
                acc_dtype=cutlass.Float32,
                has_block_sizes=block_sizes is not None,
                num_splits=self.kv_splits,
                allow_empty_block_nums=self._op.allow_empty_block_nums
                and self.block_nums_desc is not None,
            )
            kernel(
                q_kernel,
                k_kernel,
                v_kernel,
                o_kernel,
                lse_kernel,
                index_kernel,
                nums_kernel,
                sizes_kernel,
                split_kernel,
                cutlass.Float32(self.softmax_scale),
                stream,
            )
        elif self.compute_capability_family == 120:
            from .csrc.fwd.sm120_blk64.bsa_fwd_sm120 import (
                BlockSparseAttnForwardSm120Blk64,
            )

            def select(tensor: Any, modes: tuple[int, ...]) -> Any:
                return cute.make_tensor(
                    tensor.iterator, cute.select(tensor.layout, mode=modes)
                )

            q_kernel = select(q, (2, 3, 1, 0))
            k_kernel = select(k, (2, 3, 1, 0))
            v_kernel = select(v, (3, 2, 1, 0))
            o_kernel = select(kernel_output, (2, 3, 1, 0))
            lse_kernel = select(kernel_lse, (2, 1, 0))
            index_kernel = select(block_index, (3, 2, 1, 0))
            nums_kernel = (
                index_kernel if block_nums is None else select(block_nums, (2, 1, 0))
            )
            block_sizes_mode = 0
            sizes_kernel = nums_kernel
            if block_sizes is not None:
                block_sizes_mode = self.block_sizes_desc.ndim
                sizes_kernel = (
                    block_sizes
                    if block_sizes_mode == 1
                    else select(block_sizes, tuple(reversed(range(block_sizes_mode))))
                )
            kernel = BlockSparseAttnForwardSm120Blk64(
                gqa_ratio=self._op.gqa_ratio,
                head_dim=self._op.head_dim,
                value_dim=self._op.value_dim,
                blocksparse_blocksize_q=64,
                blocksparse_blocksize_k=64,
                dtype=q.element_type,
                acc_dtype=cutlass.Float32,
                has_block_sizes=block_sizes is not None,
                has_block_nums=self.block_nums_desc is not None,
                block_sizes_mode=block_sizes_mode,
            )
            kernel(
                q_kernel,
                k_kernel,
                v_kernel,
                o_kernel,
                lse_kernel,
                index_kernel,
                nums_kernel,
                cutlass.Int32(
                    0 if self.block_nums_desc is not None else self.block_sparse_num
                ),
                sizes_kernel,
                cutlass.Float32(self.softmax_scale),
                stream,
            )
        elif self.sparse_block_size == 64:
            from .csrc.fwd.sm100_blk64.bsa_fwd_sm100 import (
                BlockSparseAttnForwardSm100Blk64,
            )

            use_clc = _resolve_sm100_blk64_use_clc(
                kv_splits=self.kv_splits,
                requested=self.use_clc,
                batch=self._op.batch,
                heads=self._op.num_q_heads,
                seqlen_q=self._op.seqlen_q,
                block_sparse_num=self.block_sparse_num,
                has_variable_block_nums=self.block_nums_desc is not None,
            )
            kernel = BlockSparseAttnForwardSm100Blk64(
                self._op.head_dim,
                self._op.value_dim,
                qhead_per_kvhead=1,
                pack_gqa=False,
                m_block_size=64,
                n_block_size=256,
                sparse_block_size=64,
                is_persistent=use_clc,
                use_clc_scheduler=use_clc,
                allow_empty_block_nums=self._op.allow_empty_block_nums
                and self.block_nums_desc is not None,
                has_block_sizes=block_sizes is not None,
                num_splits=self.kv_splits,
                use_int64_kv_strides=(
                    _sm100_blk64_uses_int64_kv_strides(self.k_desc)
                    or _sm100_blk64_uses_int64_kv_strides(self.v_desc)
                ),
            )
            kernel(
                q,
                k,
                v,
                kernel_output,
                kernel_lse,
                cutlass.Float32(self.softmax_scale),
                block_index,
                block_sizes,
                cutlass.Int32(
                    0 if self.block_nums_desc is not None else self.block_sparse_num
                ),
                block_nums,
                split_offsets,
                stream,
            )
        else:
            from .csrc.fwd.sm100_blk128.bsa_fwd_sm100 import (
                BlockSparseAttnForwardSm100Blk128,
            )

            kernel = BlockSparseAttnForwardSm100Blk128(
                self._op.head_dim,
                self._op.value_dim,
                qhead_per_kvhead=self._op.gqa_ratio,
                pack_gqa=self._op.pack_gqa,
                allow_empty_block_nums=self._op.allow_empty_block_nums
                and self.block_nums_desc is not None,
                has_block_sizes=block_sizes is not None,
            )
            kernel(
                q,
                k,
                v,
                output,
                lse,
                cutlass.Float32(self.softmax_scale),
                block_index,
                block_sizes,
                cutlass.Int32(self.block_sparse_num),
                block_nums,
                stream,
            )

        if self.kv_splits > 1:
            from .csrc.fwd.sm100_blk64.bsa_fwd_combine import (
                BlockSparseAttnForwardCombine,
            )

            split_heads = self.kv_splits * self._op.num_q_heads
            o_partial = cute.make_tensor(
                kernel_output.iterator,
                cute.make_layout(
                    (
                        self.kv_splits,
                        self._op.batch,
                        self._op.seqlen_q,
                        self._op.num_q_heads,
                        self._op.value_dim,
                    ),
                    stride=(
                        self._op.num_q_heads * self._op.seqlen_q * self._op.value_dim,
                        self._op.seqlen_q * split_heads * self._op.value_dim,
                        self._op.value_dim,
                        self._op.seqlen_q * self._op.value_dim,
                        1,
                    ),
                ),
            )
            lse_partial = cute.make_tensor(
                kernel_lse.iterator,
                cute.make_layout(
                    (
                        self.kv_splits,
                        self._op.batch,
                        self._op.seqlen_q,
                        self._op.num_q_heads,
                    ),
                    stride=(
                        self._op.num_q_heads * self._op.seqlen_q,
                        self._op.seqlen_q * split_heads,
                        1,
                        self._op.seqlen_q,
                    ),
                ),
            )
            output_bshd = cute.make_tensor(
                output.iterator, cute.select(output.layout, mode=(0, 2, 1, 3))
            )
            lse_bsh = cute.make_tensor(
                lse.iterator, cute.select(lse.layout, mode=(0, 2, 1))
            )
            combine = BlockSparseAttnForwardCombine(
                dtype=output.element_type,
                head_dim=self._op.value_dim,
                tile_m=16,
                k_block_size=64,
                log_max_splits=(self.kv_splits - 1).bit_length(),
                num_threads=128,
                stages=4,
            )
            combine(
                o_partial,
                lse_partial,
                output_bshd,
                lse_bsh,
                None,
                None,
                None,
                None,
                None,
                stream,
            )


class BlockSparseAttentionBackward(JaxApiBase):
    """JAX callable specialized from BSA backward tensor metadata."""

    def __init__(
        self,
        sample_do: Any,
        sample_q: Any,
        sample_k: Any,
        sample_v: Any,
        sample_o: Any,
        sample_lse: Any,
        sample_q2k_block_index: Any,
        *,
        sample_block_sizes: Any | None = None,
        sample_q2k_block_nums: Any | None = None,
        sample_dq: Any | None = None,
        sample_dk: Any | None = None,
        sample_dv: Any | None = None,
        block_sparse_num: int | None = None,
        bucket_size_blocks: int | None = None,
        sparse_block_size: int | None = None,
        softmax_scale: float | None = None,
        layout: str = "bhsd",
        target_compute_capability: int | None = None,
    ) -> None:
        self.layout = layout
        self.data_mode = _data_mode(layout)
        self.compute_capability = self._resolve_compute_capability(
            target_compute_capability,
            SUPPORTED_BACKWARD_COMPUTE_CAPABILITIES,
            "BlockSparseAttentionBackward",
        )
        self.compute_capability_family = compute_capability_family(
            self.compute_capability
        )
        self.do_desc = self._to_tensor_desc(sample_do, "sample_do", mode=self.data_mode)
        self.q_desc = self._to_tensor_desc(sample_q, "sample_q", mode=self.data_mode)
        self.k_desc = self._to_tensor_desc(sample_k, "sample_k", mode=self.data_mode)
        self.v_desc = self._to_tensor_desc(sample_v, "sample_v", mode=self.data_mode)
        self.o_desc = self._to_tensor_desc(sample_o, "sample_o", mode=self.data_mode)
        self.lse_desc = self._to_tensor_desc(sample_lse, "sample_lse")
        self.block_index_desc = self._to_tensor_desc(
            sample_q2k_block_index, "sample_q2k_block_index"
        )
        self.block_sizes_desc = (
            None
            if sample_block_sizes is None
            else self._to_tensor_desc(sample_block_sizes, "sample_block_sizes")
        )
        self.block_nums_desc = (
            None
            if sample_q2k_block_nums is None
            else self._to_tensor_desc(sample_q2k_block_nums, "sample_q2k_block_nums")
        )

        def output_desc(
            sample: Any | None, source: JaxTensorDesc, name: str
        ) -> JaxTensorDesc:
            if sample is not None:
                return self._to_tensor_desc(sample, name, mode=self.data_mode)
            return self._to_tensor_desc(
                _shape_sample(source.shape, source.dtype, self.data_mode),
                name,
                mode=self.data_mode,
            )

        self.dq_desc = output_desc(sample_dq, self.q_desc, "sample_dq")
        self.dk_desc = output_desc(sample_dk, self.k_desc, "sample_dk")
        self.dv_desc = output_desc(sample_dv, self.v_desc, "sample_dv")
        self.sparse_block_size = (
            _default_backward_sparse_block_size(self.compute_capability)
            if sparse_block_size is None
            else sparse_block_size
        )
        capacity = (
            self.block_index_desc.shape[3] if self.block_index_desc.ndim == 4 else 0
        )
        self.block_sparse_num = (
            capacity if block_sparse_num is None else block_sparse_num
        )
        head_dim = self.q_desc.shape[3] if self.q_desc.ndim == 4 else 1
        self.softmax_scale = (
            head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
        )
        num_q_blocks = (
            ceil_div(self.q_desc.shape[2], self.sparse_block_size)
            if self.q_desc.ndim == 4
            else 0
        )
        self.bucket_size_blocks = (
            _default_bucket_size(
                self.compute_capability,
                self.sparse_block_size,
                num_q_blocks,
                self.q_desc.shape[1],
            )
            if bucket_size_blocks is None
            else bucket_size_blocks
        )
        self._op = BlockSparseAttentionBackwardOp(
            dout=self.do_desc,
            q=self.q_desc,
            k=self.k_desc,
            v=self.v_desc,
            output=self.o_desc,
            lse=self.lse_desc,
            block_index=self.block_index_desc,
            dq=self.dq_desc,
            dk=self.dk_desc,
            dv=self.dv_desc,
            block_sizes=self.block_sizes_desc,
            block_nums=self.block_nums_desc,
            block_sparse_num=self.block_sparse_num,
            sparse_block_size=self.sparse_block_size,
            softmax_scale=self.softmax_scale,
            bucket_size_blocks=self.bucket_size_blocks,
            target_compute_capability=self.compute_capability,
        )
        self.check_support()

    def check_support(self) -> bool:
        return self._op.check_support()

    def _workspace_descs(self) -> tuple[tuple[str, JaxTensorDesc], ...]:
        int_source = self.block_index_desc
        float_source = self.lse_desc
        b, h = self._op.batch, self._op.num_heads
        g, nk = self._op.num_q_groups, self._op.num_kv_blocks
        workspaces: list[tuple[str, JaxTensorDesc]] = [
            (
                "counts",
                int_source.compact_like(
                    cudnn_dtype=data_type.INT32,
                    shape=(b, h, g, nk),
                    name="counts_workspace",
                    init_value=0,
                ),
            ),
            (
                "local_offsets",
                int_source.compact_like(
                    cudnn_dtype=data_type.INT32,
                    shape=(b, h, g, nk + 1),
                    name="local_offsets_workspace",
                ),
            ),
            (
                "group_totals",
                int_source.compact_like(
                    cudnn_dtype=data_type.INT32,
                    shape=(b, h, g),
                    name="group_totals_workspace",
                ),
            ),
            (
                "bucket_offsets",
                int_source.compact_like(
                    cudnn_dtype=data_type.INT32,
                    shape=(b, h, g, nk + 1),
                    name="bucket_offsets_workspace",
                ),
            ),
            (
                "cursors",
                int_source.compact_like(
                    cudnn_dtype=data_type.INT32,
                    shape=(b, h, g, nk),
                    name="cursors_workspace",
                ),
            ),
            (
                "bucket_indices",
                int_source.compact_like(
                    cudnn_dtype=data_type.INT32,
                    shape=(b, h, self._op.max_edges),
                    name="bucket_indices_workspace",
                ),
            ),
        ]
        if self.block_nums_desc is None:
            workspaces.append(
                (
                    "block_nums_placeholder",
                    int_source.compact_like(
                        cudnn_dtype=data_type.INT32,
                        shape=(b, h, self._op.num_q_blocks),
                        name="block_nums_placeholder",
                        init_value=self.block_sparse_num,
                    ),
                )
            )
        if self.block_sizes_desc is None:
            workspaces.append(
                (
                    "block_sizes_placeholder",
                    int_source.compact_like(
                        cudnn_dtype=data_type.INT32,
                        shape=(b, nk),
                        name="block_sizes_placeholder",
                        init_value=0,
                    ),
                )
            )

        if self.compute_capability_family == 100 and self.sparse_block_size == 128:
            q_rounded = self._op.num_q_blocks * 128
            k_rounded = self._op.num_kv_blocks * 128
            d_rounded = _round_up(self._op.head_dim, 32)
            workspaces.extend(
                (
                    (
                        "dpsum",
                        float_source.compact_like(
                            cudnn_dtype=data_type.FLOAT,
                            shape=(b, h, q_rounded),
                            name="dpsum_workspace",
                        ),
                    ),
                    (
                        "lse_log2",
                        float_source.compact_like(
                            cudnn_dtype=data_type.FLOAT,
                            shape=(b, h, q_rounded),
                            name="lse_log2_workspace",
                        ),
                    ),
                    (
                        "dq_accum",
                        float_source.compact_like(
                            cudnn_dtype=data_type.FLOAT,
                            shape=(b, h, q_rounded * d_rounded),
                            name="dq_accum_workspace",
                        ),
                    ),
                )
            )
            if self._op.num_q_groups > 1:
                workspaces.extend(
                    (
                        (
                            "dk_accum",
                            float_source.compact_like(
                                cudnn_dtype=data_type.FLOAT,
                                shape=(b, h, k_rounded * d_rounded),
                                name="dk_accum_workspace",
                                init_value=0.0,
                            ),
                        ),
                        (
                            "dv_accum",
                            float_source.compact_like(
                                cudnn_dtype=data_type.FLOAT,
                                shape=(b, h, k_rounded * d_rounded),
                                name="dv_accum_workspace",
                                init_value=0.0,
                            ),
                        ),
                    )
                )
        else:
            q_round_to = 64 if self.compute_capability_family == 90 else 8
            k_round_to = 64 if self.compute_capability_family == 90 else 8
            d_round_to = 32 if self.compute_capability_family == 90 else 8
            q_rounded = _round_up(self._op.seqlen_q, q_round_to)
            k_rounded = _round_up(self._op.seqlen_k, k_round_to)
            d_rounded = _round_up(self._op.head_dim, d_round_to)
            elems_per_bh = (
                2 * q_rounded + q_rounded * d_rounded + 2 * k_rounded * d_rounded
            )
            workspaces.append(
                (
                    "bwd_workspace",
                    float_source.compact_like(
                        cudnn_dtype=data_type.FLOAT,
                        shape=(b, h, elems_per_bh),
                        name="bwd_workspace",
                        init_value=0.0,
                    ),
                )
            )
        return tuple(workspaces)

    def __call__(
        self,
        do: Any,
        q: Any,
        k: Any,
        v: Any,
        o: Any,
        lse: Any,
        q2k_block_index: Any,
        block_sizes: Any | None = None,
        q2k_block_nums: Any | None = None,
    ) -> TupleDict:
        self.check_support()
        for value, desc in (
            (do, self.do_desc),
            (q, self.q_desc),
            (k, self.k_desc),
            (v, self.v_desc),
            (o, self.o_desc),
        ):
            self._check_tensor_signature(value, desc, mode=self.data_mode)
        self._check_tensor_signature(lse, self.lse_desc)
        self._check_tensor_signature(q2k_block_index, self.block_index_desc)
        _optional_signature(block_sizes, self.block_sizes_desc, "block_sizes")
        _optional_signature(q2k_block_nums, self.block_nums_desc, "q2k_block_nums")

        inputs: list[Any] = [do, q, k, v, o, lse, q2k_block_index]
        bindings: list[tuple[JaxTensorDesc, tuple[int, ...] | None]] = [
            (self.do_desc, self.data_mode),
            (self.q_desc, self.data_mode),
            (self.k_desc, self.data_mode),
            (self.v_desc, self.data_mode),
            (self.o_desc, self.data_mode),
            (self.lse_desc, None),
            (self.block_index_desc, None),
        ]
        if self.block_sizes_desc is not None:
            inputs.append(block_sizes)
            bindings.append((self.block_sizes_desc, None))
        if self.block_nums_desc is not None:
            inputs.append(q2k_block_nums)
            bindings.append((self.block_nums_desc, None))
        self._backward_input_count = len(inputs)
        workspace_items = self._workspace_descs()
        self._backward_workspace_names = tuple(name for name, _ in workspace_items)
        workspace_descs = tuple(desc for _, desc in workspace_items)

        dq, dk, dv = self._call_kernel(
            tuple(inputs),
            launch=self._launch,
            output_descs=(self.dq_desc, self.dk_desc, self.dv_desc),
            workspace_descs=workspace_descs,
            input_spec=tuple(
                self._to_tensor_spec(desc, mode=mode) for desc, mode in bindings
            ),
            output_spec=(
                self._to_tensor_spec(self.dq_desc, mode=self.data_mode),
                self._to_tensor_spec(self.dk_desc, mode=self.data_mode),
                self._to_tensor_spec(self.dv_desc, mode=self.data_mode),
            ),
            compile_options=compile_options_for_target(self.compute_capability),
        )
        return TupleDict(dq_tensor=dq, dk_tensor=dk, dv_tensor=dv)

    def _launch(self, stream: Any, *arguments: Any) -> None:
        import cutlass
        import cutlass.cute as cute

        inputs = list(arguments[: self._backward_input_count])
        dq, dk, dv = arguments[
            self._backward_input_count : self._backward_input_count + 3
        ]
        workspaces = dict(
            zip(
                self._backward_workspace_names,
                arguments[self._backward_input_count + 3 :],
            )
        )
        do, q, k, v, o, lse, block_index = inputs[:7]
        cursor = 7
        block_sizes = None
        if self.block_sizes_desc is not None:
            block_sizes = inputs[cursor]
            cursor += 1
        block_nums = None
        if self.block_nums_desc is not None:
            block_nums = inputs[cursor]
            cursor += 1
        if cursor != len(inputs):
            raise RuntimeError("Unexpected BSA backward kernel inputs")
        if block_nums is None:
            block_nums_kernel = workspaces["block_nums_placeholder"]
        else:
            block_nums_kernel = block_nums

        from .csrc.bwd.bucketed_k2q_csr import BucketedK2QCsrUniversal

        builder = BucketedK2QCsrUniversal(
            self.block_sparse_num,
            self.bucket_size_blocks,
            self.block_nums_desc is not None,
            self.block_index_desc.shape[3],
        )
        builder(
            workspaces["counts"],
            workspaces["local_offsets"],
            workspaces["group_totals"],
            workspaces["bucket_offsets"],
            workspaces["cursors"],
            workspaces["bucket_indices"],
            block_index,
            block_nums_kernel,
            stream,
        )

        if block_sizes is None:
            variable_block_sizes = workspaces["block_sizes_placeholder"]
        elif self.block_sizes_desc.ndim == 1:
            variable_block_sizes = cute.make_tensor(
                block_sizes.iterator,
                cute.make_layout(
                    (self._op.batch, self._op.num_kv_blocks),
                    stride=(0, block_sizes.stride[0]),
                ),
            )
        else:
            variable_block_sizes = block_sizes

        if self.compute_capability_family == 90:
            from .csrc.bwd.sm90_blk64.bsa_bwd_sm90 import (
                BlockSparseAttnBackwardSm90Blk64,
            )

            kernel = BlockSparseAttnBackwardSm90Blk64(
                q.element_type, self._op.head_dim, self._op.head_dim
            )
            kernel(
                (
                    cutlass.Int32(self._op.seqlen_q),
                    cutlass.Int32(self._op.seqlen_k),
                    cutlass.Int32(self._op.head_dim),
                    (cutlass.Int32(self._op.num_heads), cutlass.Int32(self._op.batch)),
                ),
                do,
                o,
                q,
                k,
                v,
                lse,
                dq,
                dk,
                dv,
                workspaces["bucket_offsets"],
                workspaces["bucket_indices"],
                variable_block_sizes if self.block_sizes_desc is not None else None,
                workspaces["bwd_workspace"],
                cutlass.Float32(self.softmax_scale),
                stream,
            )
            return

        if self.sparse_block_size == 64:
            from .csrc.bwd.sm100_blk64.bsa_bwd_sm100 import (
                BlockSparseAttnBackwardSm100Blk64,
            )

            kernel = BlockSparseAttnBackwardSm100Blk64(
                sparse_block_size=64,
                has_block_sizes=self.block_sizes_desc is not None,
            )
            kernel(
                (
                    cutlass.Int32(self._op.seqlen_q),
                    cutlass.Int32(self._op.seqlen_k),
                    cutlass.Int32(self._op.head_dim),
                    (cutlass.Int32(self._op.num_heads), cutlass.Int32(self._op.batch)),
                ),
                do,
                o,
                q,
                k,
                v,
                lse,
                dq,
                dk,
                dv,
                workspaces["bucket_offsets"],
                workspaces["bucket_indices"],
                variable_block_sizes,
                workspaces["bwd_workspace"],
                cutlass.Float32(self.softmax_scale),
                stream,
            )
            return

        from .csrc.bwd.bsa_bwd_postprocess import BlockSparseAttnBackwardPostprocess
        from .csrc.bwd.bsa_bwd_preprocess import BlockSparseAttnBackwardPreprocess
        from .csrc.bwd.sm100_blk128.bsa_bwd_sm100 import (
            BsaK2qCsrTensors,
            BlockSparseAttnBackwardSm100Blk128,
        )

        def bhsd_to_bshd(tensor: Any) -> Any:
            return cute.make_tensor(
                tensor.iterator, cute.select(tensor.layout, mode=(0, 2, 1, 3))
            )

        do_bshd, q_bshd, k_bshd, v_bshd, o_bshd = [
            bhsd_to_bshd(tensor) for tensor in (do, q, k, v, o)
        ]
        dq_bshd, dk_bshd, dv_bshd = [bhsd_to_bshd(tensor) for tensor in (dq, dk, dv)]
        preprocess = BlockSparseAttnBackwardPreprocess(
            q.element_type,
            self._op.head_dim,
            self._op.head_dim,
            128,
        )
        preprocess(
            o_bshd,
            do_bshd,
            workspaces["dpsum"],
            lse,
            workspaces["lse_log2"],
            workspaces["dq_accum"],
            None,
            None,
            None,
            stream,
        )
        use_dkv_postprocess = self._op.num_q_groups > 1
        core = BlockSparseAttnBackwardSm100Blk128(
            self._op.head_dim,
            force_dkv_postprocess=use_dkv_postprocess,
        )
        core(
            q_bshd,
            k_bshd,
            v_bshd,
            do_bshd,
            workspaces["lse_log2"],
            workspaces["dpsum"],
            workspaces["dq_accum"],
            workspaces["dk_accum"] if use_dkv_postprocess else dk_bshd,
            workspaces["dv_accum"] if use_dkv_postprocess else dv_bshd,
            cutlass.Float32(self.softmax_scale),
            BsaK2qCsrTensors(
                workspaces["bucket_offsets"],
                workspaces["bucket_indices"],
            ),
            stream,
        )
        postprocess = BlockSparseAttnBackwardPostprocess(
            q.element_type,
            self._op.head_dim,
            self.compute_capability,
            128,
            False,
        )
        postprocess(
            workspaces["dq_accum"],
            dq_bshd,
            cutlass.Float32(self.softmax_scale),
            None,
            None,
            stream,
        )
        if use_dkv_postprocess:
            postprocess(
                workspaces["dk_accum"],
                dk_bshd,
                cutlass.Float32(self.softmax_scale),
                None,
                None,
                stream,
            )
            postprocess(
                workspaces["dv_accum"],
                dv_bshd,
                cutlass.Float32(1.0),
                None,
                None,
                stream,
            )


@partial(
    jax.jit,
    static_argnames=(
        "block_sparse_num",
        "sparse_block_size",
        "allow_empty_block_nums",
        "softmax_scale",
        "pack_gqa",
        "layout",
        "kv_splits",
        "use_clc",
        "target_compute_capability",
    ),
)
def block_sparse_attention_forward(
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    q2k_block_index: Any,
    block_sparse_num: int | None = None,
    block_sizes: Any | None = None,
    q2k_block_nums: Any | None = None,
    *,
    sparse_block_size: int | None = None,
    allow_empty_block_nums: bool = False,
    softmax_scale: float | None = None,
    pack_gqa: bool | None = None,
    layout: str = "bhsd",
    kv_splits: int | str = 1,
    use_clc: bool | None = None,
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Run non-causal block-sparse attention from JAX arrays."""

    def sample(value: Any | None) -> Any | None:
        return None if value is None else jax.ShapeDtypeStruct(value.shape, value.dtype)

    return BlockSparseAttentionForward(
        sample(q_tensor),
        sample(k_tensor),
        sample(v_tensor),
        sample(q2k_block_index),
        sample_block_sizes=sample(block_sizes),
        sample_q2k_block_nums=sample(q2k_block_nums),
        block_sparse_num=block_sparse_num,
        sparse_block_size=sparse_block_size,
        allow_empty_block_nums=allow_empty_block_nums,
        softmax_scale=softmax_scale,
        pack_gqa=pack_gqa,
        layout=layout,
        kv_splits=kv_splits,
        use_clc=use_clc,
        target_compute_capability=target_compute_capability,
    )(
        q_tensor,
        k_tensor,
        v_tensor,
        q2k_block_index,
        block_sizes,
        q2k_block_nums,
    )


@partial(
    jax.jit,
    static_argnames=(
        "block_sparse_num",
        "bucket_size_blocks",
        "sparse_block_size",
        "softmax_scale",
        "layout",
        "target_compute_capability",
    ),
)
def block_sparse_attention_backward(
    do_tensor: Any,
    q_tensor: Any,
    k_tensor: Any,
    v_tensor: Any,
    o_tensor: Any,
    lse_tensor: Any,
    q2k_block_index: Any,
    block_sparse_num: int | None = None,
    block_sizes: Any | None = None,
    q2k_block_nums: Any | None = None,
    *,
    softmax_scale: float | None = None,
    bucket_size_blocks: int | None = None,
    sparse_block_size: int | None = None,
    layout: str = "bhsd",
    target_compute_capability: int | None = None,
) -> TupleDict:
    """Compute explicit block-sparse attention gradients from JAX arrays."""

    def sample(value: Any | None) -> Any | None:
        return None if value is None else jax.ShapeDtypeStruct(value.shape, value.dtype)

    return BlockSparseAttentionBackward(
        sample(do_tensor),
        sample(q_tensor),
        sample(k_tensor),
        sample(v_tensor),
        sample(o_tensor),
        sample(lse_tensor),
        sample(q2k_block_index),
        sample_block_sizes=sample(block_sizes),
        sample_q2k_block_nums=sample(q2k_block_nums),
        block_sparse_num=block_sparse_num,
        bucket_size_blocks=bucket_size_blocks,
        sparse_block_size=sparse_block_size,
        softmax_scale=softmax_scale,
        layout=layout,
        target_compute_capability=target_compute_capability,
    )(
        do_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        q2k_block_index,
        block_sizes,
        q2k_block_nums,
    )


__all__ = [
    "BlockSparseAttentionBackward",
    "BlockSparseAttentionForward",
    "block_sparse_attention_backward",
    "block_sparse_attention_forward",
]
