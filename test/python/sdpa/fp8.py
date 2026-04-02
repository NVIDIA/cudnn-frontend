import cudnn
import pytest
import torch
import math
from enum import IntEnum
from looseversion import LooseVersion

from .fp8_ref import (
    compute_ref,
    compute_ref_backward,
)
from .helpers import (
    get_fp8_scale_factor,
    get_fp8_descale_factor,
    convert_to_cudnn_type,
    create_sparse_int_tensor,
    print_tensor_stats,
    exact_equal,
    prefix_sum,
    convert_packed_to_uniform,
    convert_uniform_to_packed,
    time_execution,
    profile_execution,
)

# fmt: off

class GraphFwdUid(IntEnum):
    q = 0
    k = 1
    v = 2
    q_descale = 5
    k_descale = 6
    v_descale = 7
    s_scale = 9
    s_descale = 8
    o_scale = 10
    o = 3
    stats = 4
    s_amax = 11
    o_amax = 12
    kv_seq_len = 13
    q_seq_len = 14
    k_block_table = 15
    v_block_table = 16
    q_ragged_offset = 17
    k_ragged_offset = 18
    v_ragged_offset = 19
    o_ragged_offset = 20
    stats_ragged_offset = 21
    sink_token = 22

class GraphBwdUid(IntEnum):
    q = 100
    k = 101
    v = 102
    o = 103
    dO = 104
    stats = 105
    q_descale = 106
    k_descale = 107
    v_descale = 108
    o_descale = 109
    dO_descale = 110
    s_descale = 111
    dP_descale = 112
    s_scale = 113
    dQ_scale = 114
    dK_scale = 115
    dV_scale = 116
    dP_scale = 117
    dQ = 118
    dK = 119
    dV = 120
    dQ_amax = 121
    dK_amax = 122
    dV_amax = 123
    dP_amax = 124
    q_ragged_offset = 125
    k_ragged_offset = 126
    v_ragged_offset = 127
    o_ragged_offset = 128
    stats_ragged_offset = 129
    dO_ragged_offset = 130
    kv_seq_len = 131
    q_seq_len = 132
    sink_token = 133
    dSink_token = 134

def generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size, is_ragged=False, generate_stats=True, left_bound=None, right_bound=None, diag_align=None, with_sink_token=False):
    graph_fwd = cudnn.pygraph(io_data_type=cudnn_itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    use_padding_mask = None
    kv_seq_len = None
    q_seq_len = None
    k_block_table = None
    v_block_table = None

    # BSHD stride order: (s*h*d, d, h*d, 1)
    stride_q = (s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1)
    stride_k = (s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)
    stride_v = (s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1)

    if block_size == 0:
        q = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=stride_q, data_type=cudnn_itype)
        k = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(b, h_k, s_kv, d_qk), stride=stride_k, data_type=cudnn_itype)
        v = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(b, h_v, s_kv, d_vo), stride=stride_v, data_type=cudnn_itype)
    else:
        table_size = math.ceil(s_kv / block_size)
        num_blocks = table_size * b

        q = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=stride_q, data_type=cudnn_itype)
        k = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(num_blocks, h_k, block_size, d_qk), stride=(block_size * h_k * d_qk, block_size * d_qk, d_qk, 1), data_type=cudnn_itype)
        v = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(num_blocks, h_v, block_size, d_vo), stride=(block_size * h_v * d_vo, block_size * d_vo, d_vo, 1), data_type=cudnn_itype)

        use_padding_mask = True
        kv_seq_len = graph_fwd.tensor(uid=GraphFwdUid.kv_seq_len, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        q_seq_len = graph_fwd.tensor(uid=GraphFwdUid.q_seq_len, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        k_block_table = graph_fwd.tensor(uid=GraphFwdUid.k_block_table, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)
        v_block_table = graph_fwd.tensor(uid=GraphFwdUid.v_block_table, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)

    if is_ragged:
        use_padding_mask = True
        q_seq_len = graph_fwd.tensor(uid=GraphFwdUid.q_seq_len, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        kv_seq_len = graph_fwd.tensor(uid=GraphFwdUid.kv_seq_len, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)

        q_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.q_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        k_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.k_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        v_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.v_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        o_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.o_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        stats_ragged_offset = graph_fwd.tensor(uid=int(GraphFwdUid.stats_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64) if generate_stats else None
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)

    q_descale = graph_fwd.tensor(uid=GraphFwdUid.q_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    k_descale = graph_fwd.tensor(uid=GraphFwdUid.k_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    v_descale = graph_fwd.tensor(uid=GraphFwdUid.v_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_scale = graph_fwd.tensor(uid=GraphFwdUid.s_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_descale = graph_fwd.tensor(uid=GraphFwdUid.s_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    o_scale = graph_fwd.tensor(uid=GraphFwdUid.o_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    sink_token = None
    if with_sink_token:
        sink_token = graph_fwd.tensor(uid=GraphFwdUid.sink_token, dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    sdpa_kwargs = dict(
        q=q, k=k, v=v,
        descale_q=q_descale, descale_k=k_descale, descale_v=v_descale,
        scale_s=s_scale, descale_s=s_descale, scale_o=o_scale,
        generate_stats=generate_stats, attn_scale=attn_scale, use_causal_mask=False,
        use_padding_mask=use_padding_mask, seq_len_kv=kv_seq_len, seq_len_q=q_seq_len,
        paged_attention_k_table=k_block_table, paged_attention_v_table=v_block_table,
        paged_attention_max_seq_len_kv=s_kv,
        left_bound=left_bound, right_bound=right_bound,
        sink_token=sink_token,
    )
    # Only pass diagonal_alignment if it's not None (pybind11 doesn't accept None for enum types)
    if diag_align is not None:
        sdpa_kwargs['diagonal_alignment'] = diag_align
    o, stats, amax_s, amax_o = graph_fwd.sdpa_fp8(**sdpa_kwargs)

    stride_o = (s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1)
    o.set_uid(GraphFwdUid.o).set_output(True).set_dim((b, h_q, s_qo, d_vo)).set_stride(stride_o).set_data_type(cudnn_otype)
    if is_ragged:
        o.set_ragged_offset(o_ragged_offset)

    if generate_stats:
        stats_stride = (s_qo * h_q, 1, h_q, 1) if is_ragged else (s_qo * h_q, s_qo, 1, 1)
        stats.set_uid(GraphFwdUid.stats).set_output(True).set_dim((b, h_q, s_qo, 1)).set_stride(stats_stride).set_data_type(cudnn.data_type.FLOAT)
        if is_ragged:
            stats.set_ragged_offset(stats_ragged_offset)

    amax_s.set_uid(GraphFwdUid.s_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_uid(GraphFwdUid.o_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_fwd

def generate_graph_bwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, deterministic, is_ragged=False, left_bound=None, right_bound=None, diag_align=None, with_sink_token=False):
    graph_bwd = cudnn.pygraph(io_data_type=cudnn_itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    stride_q = (s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1)
    stride_k = (s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)
    stride_v = (s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1)
    stride_o = (s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1)
    stats_stride = (s_qo * h_q, 1, h_q, 1) if is_ragged else (s_qo * h_q, s_qo, 1, 1)

    q = graph_bwd.tensor(uid=GraphBwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=stride_q, data_type=cudnn_itype)
    k = graph_bwd.tensor(uid=GraphBwdUid.k, dim=(b, h_k, s_kv, d_qk), stride=stride_k, data_type=cudnn_itype)
    v = graph_bwd.tensor(uid=GraphBwdUid.v, dim=(b, h_v, s_kv, d_vo), stride=stride_v, data_type=cudnn_itype)
    o = graph_bwd.tensor(uid=GraphBwdUid.o, dim=(b, h_q, s_qo, d_vo), stride=stride_o, data_type=cudnn_otype)
    dO = graph_bwd.tensor(uid=GraphBwdUid.dO, dim=(b, h_q, s_qo, d_vo), stride=stride_o, data_type=cudnn_itype)
    stats = graph_bwd.tensor(uid=GraphBwdUid.stats, dim=(b, h_q, s_qo, 1), stride=stats_stride, data_type=cudnn.data_type.FLOAT)

    use_padding_mask = False
    seq_len_q = None
    seq_len_kv = None

    if is_ragged:
        use_padding_mask = True
        seq_len_q = graph_bwd.tensor(uid=int(GraphBwdUid.q_seq_len), dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        seq_len_kv = graph_bwd.tensor(uid=int(GraphBwdUid.kv_seq_len), dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)

        q_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.q_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        k_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.k_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        v_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.v_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        o_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.o_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        stats_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.stats_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        dO_ragged_offset = graph_bwd.tensor(uid=int(GraphBwdUid.dO_ragged_offset), dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)
        o.set_ragged_offset(o_ragged_offset)
        stats.set_ragged_offset(stats_ragged_offset)
        dO.set_ragged_offset(dO_ragged_offset)

    q_descale = graph_bwd.tensor(uid=GraphBwdUid.q_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    k_descale = graph_bwd.tensor(uid=GraphBwdUid.k_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    v_descale = graph_bwd.tensor(uid=GraphBwdUid.v_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    o_descale = graph_bwd.tensor(uid=GraphBwdUid.o_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dO_descale = graph_bwd.tensor(uid=GraphBwdUid.dO_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_descale = graph_bwd.tensor(uid=GraphBwdUid.s_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dP_descale = graph_bwd.tensor(uid=GraphBwdUid.dP_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    s_scale = graph_bwd.tensor(uid=GraphBwdUid.s_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dQ_scale = graph_bwd.tensor(uid=GraphBwdUid.dQ_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dK_scale = graph_bwd.tensor(uid=GraphBwdUid.dK_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dV_scale = graph_bwd.tensor(uid=GraphBwdUid.dV_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dP_scale = graph_bwd.tensor(uid=GraphBwdUid.dP_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    sink_token = None
    dSink_token = None
    if with_sink_token:
        sink_token = graph_bwd.tensor(uid=GraphBwdUid.sink_token, dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
        dSink_token = graph_bwd.tensor(uid=GraphBwdUid.dSink_token, dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP = graph_bwd.sdpa_fp8_backward(
        q=q, k=k, v=v, o=o, dO=dO, stats=stats,
        descale_q=q_descale, descale_k=k_descale, descale_v=v_descale,
        descale_o=o_descale, descale_dO=dO_descale, descale_s=s_descale, descale_dP=dP_descale,
        scale_s=s_scale, scale_dQ=dQ_scale, scale_dK=dK_scale, scale_dV=dV_scale, scale_dP=dP_scale,
        attn_scale=attn_scale, use_padding_mask=use_padding_mask,
        diagonal_alignment=diag_align if diag_align is not None else cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=left_bound,
        right_bound=right_bound,
        use_deterministic_algorithm=deterministic,
        seq_len_q=seq_len_q, seq_len_kv=seq_len_kv,
        sink_token=sink_token,
        dSink_token=dSink_token,
    )

    dQ.set_uid(GraphBwdUid.dQ).set_output(True).set_dim((b, h_q, s_qo, d_qk)).set_stride(stride_q).set_data_type(cudnn_otype)
    dK.set_uid(GraphBwdUid.dK).set_output(True).set_dim((b, h_k, s_kv, d_qk)).set_stride(stride_k).set_data_type(cudnn_otype)
    dV.set_uid(GraphBwdUid.dV).set_output(True).set_dim((b, h_v, s_kv, d_vo)).set_stride(stride_v).set_data_type(cudnn_otype)

    if is_ragged:
        dQ.set_ragged_offset(q_ragged_offset)
        dK.set_ragged_offset(k_ragged_offset)
        dV.set_ragged_offset(v_ragged_offset)

    amax_dQ.set_uid(GraphBwdUid.dQ_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dK.set_uid(GraphBwdUid.dK_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dV.set_uid(GraphBwdUid.dV_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dP.set_uid(GraphBwdUid.dP_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    if with_sink_token:
        dSink_token.set_uid(GraphBwdUid.dSink_token).set_output(True).set_dim((1, h_q, 1, 1)).set_stride((h_q, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_bwd

def create_paged_container_and_block_table(tensor, block_size):
    B, H, S, D = tensor.shape
    blocks_per_batch = math.ceil(S / block_size)

    padding_seq = blocks_per_batch * block_size - S
    if padding_seq > 0:
        zeros = torch.zeros(B, H, padding_seq, D, device="cuda", dtype=tensor.dtype)
        cat_tensor = torch.cat((tensor, zeros), dim=2)
    else:
        cat_tensor = tensor

    container = torch.cat(cat_tensor.chunk(blocks_per_batch, dim=2), dim=0)

    table_size = math.ceil(S / block_size)
    block_table_temp = torch.linspace(0, B * table_size - 1, B * table_size, device="cuda", dtype=torch.int32).reshape(table_size, 1, B, 1)
    block_table_temp = torch.transpose(block_table_temp, 0, 2)

    block_table = (torch.zeros(blocks_per_batch * B, device="cuda", dtype=torch.int32).as_strided((B, 1, blocks_per_batch, 1), (blocks_per_batch, blocks_per_batch, 1, 1)))
    block_table.copy_(block_table_temp)

    return (container, block_table)

def exec_sdpa_fp8(cfg, request, cudnn_handle):
    if request.config.option.dryrun:
        pytest.skip("dryrun")

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.14.0":
        pytest.skip("SDPA FP8 requires cuDNN 9.14.0 or higher")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("SDPA FP8 requires Hopper or higher")

    torch_itype = cfg.data_type
    torch_otype = cfg.output_type if hasattr(cfg, 'output_type') and cfg.output_type else cfg.data_type
    cudnn_itype = convert_to_cudnn_type(torch_itype)
    cudnn_otype = convert_to_cudnn_type(torch_otype)

    b = cfg.batches
    h_q, h_k, h_v = cfg.h_q, cfg.h_k, cfg.h_v
    s_qo, s_kv = cfg.s_q, cfg.s_kv
    d_qk, d_vo = cfg.d_qk, cfg.d_v
    block_size = cfg.block_size if cfg.is_paged else 0
    deterministic = cfg.is_determin if hasattr(cfg, 'is_determin') else False
    is_ragged = cfg.is_ragged if hasattr(cfg, 'is_ragged') else False
    left_bound = cfg.left_bound if hasattr(cfg, 'left_bound') else None
    right_bound = cfg.right_bound if hasattr(cfg, 'right_bound') else None
    diag_align = cfg.diag_align if hasattr(cfg, 'diag_align') else None
    with_sink_token = cfg.with_sink_token if hasattr(cfg, 'with_sink_token') else False

    attn_scale = 0.125

    is_paged = block_size > 0

    seq_len_q_list = cfg.seq_len_q if hasattr(cfg, 'seq_len_q') and cfg.seq_len_q else []
    seq_len_kv_list = cfg.seq_len_kv if hasattr(cfg, 'seq_len_kv') and cfg.seq_len_kv else []

    if is_ragged:
        seq_len_q_gpu = torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda").view(-1, 1, 1, 1)
        seq_len_kv_gpu = torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda").view(-1, 1, 1, 1)
        max_t_q = max(64, ((seq_len_q_gpu.sum().item() + 63) // 64) * 64)
        max_t_kv = max(64, ((seq_len_kv_gpu.sum().item() + 63) // 64) * 64)

        q_ragged_offset_gpu = (prefix_sum(seq_len_q_gpu) * h_q * d_qk).to(torch.int64)
        k_ragged_offset_gpu = (prefix_sum(seq_len_kv_gpu) * h_k * d_qk).to(torch.int64)
        v_ragged_offset_gpu = (prefix_sum(seq_len_kv_gpu) * h_v * d_vo).to(torch.int64)
        o_ragged_offset_gpu = (prefix_sum(seq_len_q_gpu) * h_q * d_vo).to(torch.int64)
        stats_ragged_offset_gpu = (prefix_sum(seq_len_q_gpu) * h_q * 1).to(torch.int64)

    # Build forward graph (always needed)
    try:
        graph_fwd = generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size, is_ragged=is_ragged, left_bound=left_bound, right_bound=right_bound, diag_align=diag_align, with_sink_token=with_sink_token)
        graph_fwd.validate()
        graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_fwd.check_support()
        graph_fwd.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"unsupported forward graph: {e}")
    except Exception as e:
        pytest.fail(f"Error building forward graph: {e}")

    rng_data = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)

    # Use sparse small integers for better low-precision testing
    q_gen = create_sparse_int_tensor((b, s_qo, h_q, d_qk), torch.float, rng_data)
    k_gen = create_sparse_int_tensor((b, s_kv, h_k, d_qk), torch.float, rng_data)
    v_gen = create_sparse_int_tensor((b, s_kv, h_v, d_vo), torch.float, rng_data)

    q_amax = q_gen.abs().max().item()
    k_amax = k_gen.abs().max().item()
    v_amax = v_gen.abs().max().item()
    s_amax = 1.0

    q_fp8 = (q_gen * get_fp8_scale_factor(q_amax, torch_itype)).to(torch_itype)
    k_fp8 = (k_gen * get_fp8_scale_factor(k_amax, torch_itype)).to(torch_itype)
    v_fp8 = (v_gen * get_fp8_scale_factor(v_amax, torch_itype)).to(torch_itype)

    q_descale_gpu = torch.tensor([get_fp8_descale_factor(q_amax, torch_itype)], dtype=torch.float, device="cuda")
    k_descale_gpu = torch.tensor([get_fp8_descale_factor(k_amax, torch_itype)], dtype=torch.float, device="cuda")
    v_descale_gpu = torch.tensor([get_fp8_descale_factor(v_amax, torch_itype)], dtype=torch.float, device="cuda")
    s_scale_gpu = torch.tensor([get_fp8_scale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")
    s_descale_gpu = torch.tensor([get_fp8_descale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")

    # Create sink_token tensor if needed
    sink_token_gpu = None
    if with_sink_token:
        sink_token_gpu = torch.randn((1, h_q, 1, 1), dtype=torch.float, device="cuda", generator=rng_data) * 0.5

    # Compute forward reference (also computes o_amax internally)
    if is_ragged:
        seq_len_q_ref = torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda")
        seq_len_kv_ref = torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda")
        padding = (seq_len_q_ref, seq_len_kv_ref)
    else:
        padding = None

    o_ref, stats_ref, o_amax = compute_ref(q_fp8, k_fp8, v_fp8, attn_scale=attn_scale,
                                            q_descale=q_descale_gpu, k_descale=k_descale_gpu, v_descale=v_descale_gpu,
                                            s_scale=s_scale_gpu, s_descale=s_descale_gpu, torch_itype=torch_itype,
                                            torch_otype=torch_otype, padding=padding,
                                            left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
                                            sink_token=sink_token_gpu)

    o_scale_gpu = torch.tensor([get_fp8_scale_factor(o_amax, torch_otype)], dtype=torch.float, device="cuda")

    # Prepare GPU input tensors (pack for ragged, page for paged)
    q_gpu = q_fp8
    k_gpu = k_fp8
    v_gpu = v_fp8

    if is_ragged:
        q_gpu = convert_uniform_to_packed(torch.einsum("bshd->bhsd", q_fp8), torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda"), max_t_q)
        k_gpu = convert_uniform_to_packed(torch.einsum("bshd->bhsd", k_fp8), torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda"), max_t_kv)
        v_gpu = convert_uniform_to_packed(torch.einsum("bshd->bhsd", v_fp8), torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda"), max_t_kv)

    if is_paged:
        k_gpu_bhsd = torch.einsum('bshd->bhsd', k_fp8).contiguous()
        v_gpu_bhsd = torch.einsum('bshd->bhsd', v_fp8).contiguous()
        container_k_gpu, k_block_table_gpu = create_paged_container_and_block_table(k_gpu_bhsd, block_size)
        container_v_gpu, v_block_table_gpu = create_paged_container_and_block_table(v_gpu_bhsd, block_size)

    # Allocate forward output tensors
    if is_ragged:
        o_gpu = torch.full((max_t_q, h_q, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        stats_gpu = torch.full((max_t_q, h_q, 1), float('nan'), dtype=torch.float, device="cuda")
    else:
        o_gpu = torch.full((b, s_qo, h_q, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        stats_gpu = torch.full((b, h_q, s_qo, 1), float('nan'), dtype=torch.float, device="cuda")

    s_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
    o_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")

    variant_pack = {
        int(GraphFwdUid.q): q_gpu,
        int(GraphFwdUid.k): k_gpu,
        int(GraphFwdUid.v): v_gpu,
        int(GraphFwdUid.q_descale): q_descale_gpu,
        int(GraphFwdUid.k_descale): k_descale_gpu,
        int(GraphFwdUid.v_descale): v_descale_gpu,
        int(GraphFwdUid.s_descale): s_descale_gpu,
        int(GraphFwdUid.s_scale): s_scale_gpu,
        int(GraphFwdUid.o_scale): o_scale_gpu,
        int(GraphFwdUid.o): o_gpu,
        int(GraphFwdUid.stats): stats_gpu,
        int(GraphFwdUid.s_amax): s_amax_gpu,
        int(GraphFwdUid.o_amax): o_amax_gpu,
    }

    if is_paged:
        variant_pack[int(GraphFwdUid.k)] = container_k_gpu
        variant_pack[int(GraphFwdUid.v)] = container_v_gpu
        variant_pack[int(GraphFwdUid.kv_seq_len)] = torch.full((b, 1, 1, 1), s_kv, device="cuda", dtype=torch.int32)
        variant_pack[int(GraphFwdUid.q_seq_len)] = torch.full((b, 1, 1, 1), s_qo, device="cuda", dtype=torch.int32)
        variant_pack[int(GraphFwdUid.k_block_table)] = k_block_table_gpu
        variant_pack[int(GraphFwdUid.v_block_table)] = v_block_table_gpu

    if is_ragged:
        variant_pack[int(GraphFwdUid.q_seq_len)] = torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda").view(-1, 1, 1, 1)
        variant_pack[int(GraphFwdUid.kv_seq_len)] = torch.tensor(seq_len_kv_list, dtype=torch.int32, device="cuda").view(-1, 1, 1, 1)
        variant_pack[int(GraphFwdUid.q_ragged_offset)] = q_ragged_offset_gpu
        variant_pack[int(GraphFwdUid.k_ragged_offset)] = k_ragged_offset_gpu
        variant_pack[int(GraphFwdUid.v_ragged_offset)] = v_ragged_offset_gpu
        variant_pack[int(GraphFwdUid.o_ragged_offset)] = o_ragged_offset_gpu
        variant_pack[int(GraphFwdUid.stats_ragged_offset)] = stats_ragged_offset_gpu

    if with_sink_token:
        variant_pack[int(GraphFwdUid.sink_token)] = sink_token_gpu

    workspace = torch.empty(graph_fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    if request.config.getoption("--perf"):
        times_ms = time_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
        print(f"@@@@ FP8 Fwd graph_fwd.execute avg_time_ms={times_ms.mean().item():.3f}")
        profile_execution(graph_fwd.execute, variant_pack, workspace, cudnn_handle)
    graph_fwd.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # Compare forward output
    if is_ragged:
        o_ref_comp = convert_uniform_to_packed(torch.einsum("bshd->bhsd", o_ref), seq_len_q_ref, max_t_q)
    else:
        o_ref_comp = o_ref

    o_gpu_float = o_gpu.detach().float() * get_fp8_descale_factor(o_amax, torch_otype)
    o_ref_float = o_ref_comp.detach().float() * get_fp8_descale_factor(o_amax, torch_otype)

    if is_ragged:
        t_idx = sum(seq_len_q_list)
        o_gpu_float[t_idx:] = 0
        o_ref_float[t_idx:] = 0

    atol, rtol = 0.08, 0.2
    torch.testing.assert_close(o_gpu_float, o_ref_float, atol=atol, rtol=rtol)

    # Backward pass
    if not cfg.is_infer:
        dO_gen = create_sparse_int_tensor((b, s_qo, h_q, d_vo), torch.float, rng_data)
        dO_amax = dO_gen.abs().max().item()
        dO_fp8 = (dO_gen * get_fp8_scale_factor(dO_amax, torch_itype)).to(torch_itype)

        o_descale_gpu = torch.tensor([get_fp8_descale_factor(o_amax, torch_otype)], dtype=torch.float, device="cuda")
        dO_descale_gpu = torch.tensor([get_fp8_descale_factor(dO_amax, torch_itype)], dtype=torch.float, device="cuda")

        # Get unpacked BSHD references for backward
        if is_ragged:
            q_ref_bwd = torch.einsum("bhsd->bshd", convert_packed_to_uniform(q_gpu, seq_len_q_ref, s_qo))
            k_ref_bwd = torch.einsum("bhsd->bshd", convert_packed_to_uniform(k_gpu, seq_len_kv_ref, s_kv))
            v_ref_bwd = torch.einsum("bhsd->bshd", convert_packed_to_uniform(v_gpu, seq_len_kv_ref, s_kv))
            o_ref_bwd = torch.einsum("bhsd->bshd", convert_packed_to_uniform(o_gpu, seq_len_q_ref, s_qo))
            dO_ref_bwd = dO_fp8
        else:
            q_ref_bwd = q_gpu
            k_ref_bwd = k_gpu
            v_ref_bwd = v_gpu
            o_ref_bwd = o_gpu
            dO_ref_bwd = dO_fp8

        padding_bwd = (seq_len_q_ref, seq_len_kv_ref) if is_ragged else None
        dQ_ref, dK_ref, dV_ref, dSink_token_ref, dP_amax, dQ_amax, dK_amax, dV_amax = compute_ref_backward(
            q_ref_bwd, k_ref_bwd, v_ref_bwd, o_ref_bwd, dO_ref_bwd, attn_scale=attn_scale,
            q_descale=q_descale_gpu, k_descale=k_descale_gpu, v_descale=v_descale_gpu,
            s_scale=s_scale_gpu, s_descale=s_descale_gpu, torch_itype=torch_itype,
            o_descale=o_descale_gpu, dO_descale=dO_descale_gpu,
            torch_otype=torch_otype, padding=padding_bwd,
            left_bound=left_bound, right_bound=right_bound, diag_align=diag_align,
            sink_token=sink_token_gpu
        )

        dP_descale_gpu = torch.tensor([get_fp8_descale_factor(dP_amax, torch_itype)], dtype=torch.float, device="cuda")
        dQ_scale_gpu = torch.tensor([get_fp8_scale_factor(dQ_amax, torch_otype)], dtype=torch.float, device="cuda")
        dK_scale_gpu = torch.tensor([get_fp8_scale_factor(dK_amax, torch_otype)], dtype=torch.float, device="cuda")
        dV_scale_gpu = torch.tensor([get_fp8_scale_factor(dV_amax, torch_otype)], dtype=torch.float, device="cuda")
        dP_scale_gpu = torch.tensor([get_fp8_scale_factor(dP_amax, torch_otype)], dtype=torch.float, device="cuda")

        try:
            graph_bwd = generate_graph_bwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, deterministic, is_ragged=is_ragged, left_bound=left_bound, right_bound=right_bound, diag_align=diag_align, with_sink_token=with_sink_token)
            graph_bwd.validate()
            graph_bwd.build_operation_graph()
            graph_bwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph_bwd.check_support()
            graph_bwd.build_plans()
        except cudnn.cudnnGraphNotSupportedError as e:
            pytest.skip(f"unsupported backward graph: {e}")
        except Exception as e:
            pytest.fail(f"Error building backward graph: {e}")

        if is_ragged:
            dO_gpu = convert_uniform_to_packed(torch.einsum("bshd->bhsd", dO_fp8), torch.tensor(seq_len_q_list, dtype=torch.int32, device="cuda"), max_t_q)
        else:
            dO_gpu = dO_fp8

        if is_ragged:
            dQ_gpu = torch.full((max_t_q, h_q, d_qk), float('nan'), dtype=torch_otype, device="cuda")
            dK_gpu = torch.full((max_t_kv, h_k, d_qk), float('nan'), dtype=torch_otype, device="cuda")
            dV_gpu = torch.full((max_t_kv, h_v, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        else:
            dQ_gpu = torch.full((b, s_qo, h_q, d_qk), float('nan'), dtype=torch_otype, device="cuda")
            dK_gpu = torch.full((b, s_kv, h_k, d_qk), float('nan'), dtype=torch_otype, device="cuda")
            dV_gpu = torch.full((b, s_kv, h_v, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        dQ_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dK_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dV_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dP_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dSink_token_gpu = None
        if with_sink_token:
            dSink_token_gpu = torch.full((1, h_q, 1, 1), float('nan'), dtype=torch.float, device="cuda")

        variant_pack_bwd = {
            int(GraphBwdUid.q): q_gpu, int(GraphBwdUid.k): k_gpu, int(GraphBwdUid.v): v_gpu,
            int(GraphBwdUid.o): o_gpu, int(GraphBwdUid.dO): dO_gpu, int(GraphBwdUid.stats): stats_gpu,
            int(GraphBwdUid.q_descale): q_descale_gpu, int(GraphBwdUid.k_descale): k_descale_gpu,
            int(GraphBwdUid.v_descale): v_descale_gpu, int(GraphBwdUid.o_descale): o_descale_gpu,
            int(GraphBwdUid.dO_descale): dO_descale_gpu, int(GraphBwdUid.s_descale): s_descale_gpu,
            int(GraphBwdUid.s_scale): s_scale_gpu, int(GraphBwdUid.dP_descale): dP_descale_gpu,
            int(GraphBwdUid.dP_scale): dP_scale_gpu, int(GraphBwdUid.dQ_scale): dQ_scale_gpu,
            int(GraphBwdUid.dK_scale): dK_scale_gpu, int(GraphBwdUid.dV_scale): dV_scale_gpu,
            int(GraphBwdUid.dQ): dQ_gpu, int(GraphBwdUid.dK): dK_gpu, int(GraphBwdUid.dV): dV_gpu,
            int(GraphBwdUid.dQ_amax): dQ_amax_gpu, int(GraphBwdUid.dK_amax): dK_amax_gpu,
            int(GraphBwdUid.dV_amax): dV_amax_gpu, int(GraphBwdUid.dP_amax): dP_amax_gpu,
        }

        if is_ragged:
            variant_pack_bwd[int(GraphBwdUid.q_seq_len)] = seq_len_q_gpu
            variant_pack_bwd[int(GraphBwdUid.kv_seq_len)] = seq_len_kv_gpu
            variant_pack_bwd[int(GraphBwdUid.q_ragged_offset)] = q_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.k_ragged_offset)] = k_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.v_ragged_offset)] = v_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.o_ragged_offset)] = o_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.stats_ragged_offset)] = stats_ragged_offset_gpu
            variant_pack_bwd[int(GraphBwdUid.dO_ragged_offset)] = o_ragged_offset_gpu

        if with_sink_token:
            variant_pack_bwd[int(GraphBwdUid.sink_token)] = sink_token_gpu
            variant_pack_bwd[int(GraphBwdUid.dSink_token)] = dSink_token_gpu

        workspace_bwd = torch.empty(graph_bwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
        if request.config.getoption("--perf"):
            times_ms = time_execution(graph_bwd.execute, variant_pack_bwd, workspace_bwd, cudnn_handle)
            print(f"@@@@ FP8 Bwd graph.execute avg_time_ms={times_ms.mean().item():.3f}")
            profile_execution(graph_bwd.execute, variant_pack_bwd, workspace_bwd, cudnn_handle)
        graph_bwd.execute(variant_pack_bwd, workspace_bwd, handle=cudnn_handle)
        torch.cuda.synchronize()

        if deterministic:
            dQ_gpu_rerun = dQ_gpu.clone().detach()
            dK_gpu_rerun = dK_gpu.clone().detach()
            dV_gpu_rerun = dV_gpu.clone().detach()

            dQ_gpu = torch.fill_(dQ_gpu, float("nan"))
            dK_gpu = torch.fill_(dK_gpu, float("nan"))
            dV_gpu = torch.fill_(dV_gpu, float("nan"))
            torch.cuda.synchronize()
            graph_bwd.execute(variant_pack_bwd, workspace_bwd, handle=cudnn_handle)
            torch.cuda.synchronize()

            determin_err_count = 0
            determin_err_count += exact_equal(dQ_gpu, dQ_gpu_rerun, tag="dQ_determin", disp_elems=10)
            determin_err_count += exact_equal(dK_gpu, dK_gpu_rerun, tag="dK_determin", disp_elems=10)
            determin_err_count += exact_equal(dV_gpu, dV_gpu_rerun, tag="dV_determin", disp_elems=10)

            if determin_err_count != 0:
                print("@@@@ Overall result: FAILED, determinism check failed - outputs differ between runs.")
                pytest.fail("determinism check failed", pytrace=False)
            print("@@@@ Determinism check: PASSED, dQ, dK, dV bitwise match between runs.")

        if is_ragged:
            dQ_ref = convert_uniform_to_packed(torch.einsum("bshd->bhsd", dQ_ref), seq_len_q_ref, max_t_q)
            dK_ref = convert_uniform_to_packed(torch.einsum("bshd->bhsd", dK_ref), seq_len_kv_ref, max_t_kv)
            dV_ref = convert_uniform_to_packed(torch.einsum("bshd->bhsd", dV_ref), seq_len_kv_ref, max_t_kv)

        dQ_out = dQ_gpu.detach().float() * get_fp8_descale_factor(dQ_amax, torch_otype)
        dK_out = dK_gpu.detach().float() * get_fp8_descale_factor(dK_amax, torch_otype)
        dV_out = dV_gpu.detach().float() * get_fp8_descale_factor(dV_amax, torch_otype)

        dQ_ref_float = dQ_ref.detach().float() * get_fp8_descale_factor(dQ_amax, torch_otype)
        dK_ref_float = dK_ref.detach().float() * get_fp8_descale_factor(dK_amax, torch_otype)
        dV_ref_float = dV_ref.detach().float() * get_fp8_descale_factor(dV_amax, torch_otype)

        if is_ragged:
            t_idx_q = sum(seq_len_q_list)
            dQ_out[t_idx_q:] = 0
            dQ_ref_float[t_idx_q:] = 0
            t_idx_kv = sum(seq_len_kv_list)
            dK_out[t_idx_kv:] = 0
            dK_ref_float[t_idx_kv:] = 0
            dV_out[t_idx_kv:] = 0
            dV_ref_float[t_idx_kv:] = 0

        atol, rtol = 0.04, 0.2
        torch.testing.assert_close(dQ_out, dQ_ref_float, atol=atol, rtol=rtol)
        torch.testing.assert_close(dK_out, dK_ref_float, atol=atol, rtol=rtol)
        torch.testing.assert_close(dV_out, dV_ref_float, atol=atol, rtol=rtol)

        if with_sink_token:
            torch.testing.assert_close(dSink_token_gpu, dSink_token_ref, atol=0.02, rtol=0.2)

    # Print hash and stats for determinism verification
    print_tensor_stats(o_gpu, tag="o_gpu")
    print_tensor_stats(s_amax_gpu, tag="s_amax_gpu")
    print_tensor_stats(o_amax_gpu, tag="o_amax_gpu")
    if not cfg.is_infer:
        print_tensor_stats(stats_gpu, tag="stats_gpu")
        print_tensor_stats(dQ_gpu, tag="dQ_gpu")
        print_tensor_stats(dK_gpu, tag="dK_gpu")
        print_tensor_stats(dV_gpu, tag="dV_gpu")
        print_tensor_stats(dQ_amax_gpu, tag="dQ_amax_gpu")
        print_tensor_stats(dK_amax_gpu, tag="dK_amax_gpu")
        print_tensor_stats(dV_amax_gpu, tag="dV_amax_gpu")
        print_tensor_stats(dP_amax_gpu, tag="dP_amax_gpu")
        if with_sink_token:
            print_tensor_stats(dSink_token_gpu, tag="dSink_token_gpu")
