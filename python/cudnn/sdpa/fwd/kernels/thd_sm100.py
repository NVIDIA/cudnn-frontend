# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver

TENSOR_MAP_QWORDS = 128 // 8


@cute.kernel
def build_o_descs_kernel(
    o_tensor: cute.Tensor,
    base_o_desc: cutlass.GridConstant[tmap.TensorMap],
    o_desc_words: cute.Tensor,
    seq_kv_lens_t: cute.Tensor,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    d_v: cutlass.Int32,
) -> None:
    if nvvm.elect_sync():
        o_ptr = o_tensor.iterator.raw_ptr()
        desc_base = o_desc_words.iterator.raw_ptr()
        src_words = Pointer(base_o_desc.get_ptr(), dtype=cutlass.Int64)
        cu = cutlass.make_array_view(seq_kv_lens_t)
        cuq0 = n_batch
        row_elems = n_qh * d_v
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
            for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
                (dptr + i).store((src_words + i).load())
            cu_q_b = cutlass.Int32(cu[cuq0 + b])
            s_i = cutlass.Int32(cu[cuq0 + b + cutlass.Int32(1)]) - cu_q_b
            row_base = o_ptr + cu_q_b * row_elems
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_ADDRESS,
                dptr,
                new_value=row_base.toint(cutlass.Int64),
            )
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_DIM,
                dptr,
                new_value=s_i,
                ord=2,
            )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU,
            from_proxy=nvvm.Proxy.GENERIC,
            to_proxy=nvvm.Proxy.TENSORMAP,
        )
