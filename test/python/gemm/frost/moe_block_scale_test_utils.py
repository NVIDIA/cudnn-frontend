# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MoE grouped block-scale matmul graph builder, shared by the MoE test entries and independent of them."""

import cudnn

from gemm_test_utils import ceil_div as _ceil_div

# combo -> (block_size, data dtype, SF dtype).
_COMBOS = {
    "nvfp4": (16, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E4M3),
    "mxfp4": (32, cudnn.data_type.FP4_E2M1, cudnn.data_type.FP8_E8M0),
    "mxfp8": (32, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E8M0),
}


def _build_graph(
    E,
    S,
    N,
    K,
    num_groups,
    combo="nvfp4",
    offset_dt=cudnn.data_type.INT32,
    quant=False,
    output_dt=cudnn.data_type.BFLOAT16,
    reduction_mode=None,
    reduction_dims=None,
    reduction_stride=None,
    reduction_dt=cudnn.data_type.FLOAT,
    reduction_compute_dt=None,
    quant_out_dt=cudnn.data_type.FP8_E4M3,
    quant_scale_dt=cudnn.data_type.FP8_E8M0,
    quant_scale_reorder=False,
    quant_scale_dim=None,
    quant_block_size=32,
    quant_axis=None,
    quant_group_offset=False,
    weight_major="k",
    a_dt_override=None,
    b_dt_override=None,
    dequant_a=True,
    dequant_b=True,
    epilogue_relu=False,
    output_major="n",
    offset_multiple=1,
    scatter_top_k=None,
    combine_top_k=None,
    gather_rows=None,
    scale_k=None,
    negative_scale=(False, False),
):
    block_size, default_dt, sf_dt = _COMBOS[combo]
    a_dt = default_dt if a_dt_override is None else a_dt_override
    b_dt = default_dt if b_dt_override is None else b_dt_override
    sf_k = K // block_size if scale_k is None else scale_k
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    w = g.tensor(name="weight", dim=[E, K, N], stride=[K * N, 1, K] if weight_major == "k" else [K * N, N, 1], data_type=b_dt)
    sf_pitch = _ceil_div(sf_k, 16) * 16 if gather_rows is not None else sf_k
    SFA = g.tensor(
        name="SFA",
        dim=[1, S, sf_k],
        stride=[S * sf_pitch, sf_pitch, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.NONE if gather_rows is not None else cudnn.tensor_reordering.F8_128x4,
    )
    SFB = g.tensor(
        name="SFB",
        dim=[E, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[num_groups + 1, 1, 1],
        stride=[1, 1, 1],
        data_type=offset_dt,
    )
    fto.set_alignment_value(offset_multiple)
    # ``negative_scale`` = (token side, weight side); None leaves is_negative_scale unset.
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size], is_negative_scale=negative_scale[0] or None) if dequant_a else tok
    w_d = g.block_scale_dequantize(input=w, descale=SFB, block_size=[block_size, 1], is_negative_scale=negative_scale[1] or None) if dequant_b else w
    routing = {}
    mode = cudnn.moe_grouped_matmul_mode.NONE
    if gather_rows is not None:
        mode = cudnn.moe_grouped_matmul_mode.GATHER
        routing["token_index"] = g.tensor(name="token_index", dim=[1, gather_rows, 1], stride=[gather_rows, 1, 1], data_type=cudnn.data_type.INT32)
    if scatter_top_k is not None or combine_top_k is not None:
        mode = cudnn.moe_grouped_matmul_mode.COMBINE if combine_top_k is not None else cudnn.moe_grouped_matmul_mode.SCATTER
        routing = {
            "token_index": g.tensor(name="token_index", dim=[1, S, 1], stride=[S, 1, 1], data_type=cudnn.data_type.INT32),
            "token_ks": g.tensor(name="token_ks", dim=[1, S, 1], stride=[S, 1, 1], data_type=cudnn.data_type.INT32),
            "top_k": combine_top_k if combine_top_k is not None else scatter_top_k,
        }
        if combine_top_k is not None:
            count = S // combine_top_k
            routing["top_k_scores"] = g.tensor(name="scores", dim=[1, count, combine_top_k], stride=[S, combine_top_k, 1], data_type=cudnn.data_type.FLOAT)
    out = g.moe_grouped_matmul(
        tok_d,
        w_d,
        fto,
        mode=mode,
        compute_data_type=cudnn.data_type.FLOAT,
        name="moe",
        **routing,
    )
    if epilogue_relu:
        out = g.relu(input=out, name="relu")
    if reduction_mode is not None:
        red_kwargs = {}
        if reduction_compute_dt is not None:
            red_kwargs["compute_data_type"] = reduction_compute_dt
        R = g.reduction(input=out, mode=reduction_mode, name="red", **red_kwargs)
        assert reduction_dims is not None
        stride = reduction_stride
        if stride is None:
            stride = (
                reduction_dims[1] * reduction_dims[2],
                reduction_dims[2],
                1,
            )
        R.set_dim(list(reduction_dims)).set_stride(list(stride))
        R.set_output(True).set_data_type(reduction_dt)
    if quant:
        quant_kwargs = {"input": out, "block_size": quant_block_size, "name": "q"}
        if quant_axis is not None:
            quant_kwargs["axis"] = quant_axis
        if quant_group_offset:
            quant_kwargs["group_offset"] = fto
        q, q_scale = g.block_scale_quantize(**quant_kwargs)
        q.set_data_type(quant_out_dt).set_output(True)
        if quant_scale_dim is not None:
            q_scale.set_dim(list(quant_scale_dim)).set_stride([quant_scale_dim[1] * quant_scale_dim[2], quant_scale_dim[2], 1])
        q_scale.set_data_type(quant_scale_dt).set_output(True)
        if quant_scale_reorder:
            q_scale.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
        return g
    out.set_data_type(output_dt).set_output(True)
    if output_major == "m":
        ldm = _ceil_div(gather_rows if gather_rows is not None else S, 8) * 8  # Align BF16 column strides independently of S.
        out.set_stride([ldm * N, 1, ldm])
    return g
