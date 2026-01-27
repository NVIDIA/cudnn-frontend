import cudnn
import pytest
import torch
import math
from enum import IntEnum
from looseversion import LooseVersion

from .fp8_ref import compute_ref
from .helpers import get_fp8_scale_factor, get_fp8_descale_factor, convert_to_cudnn_type

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

def generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size):
    graph_fwd = cudnn.pygraph(io_data_type=cudnn_itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    use_padding_mask = None
    kv_seq_len = None
    q_seq_len = None
    k_block_table = None
    v_block_table = None

    if block_size == 0:
        q = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=(s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1), data_type=cudnn_itype)
        k = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(b, h_k, s_kv, d_qk), stride=(s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1), data_type=cudnn_itype)
        v = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(b, h_v, s_kv, d_vo), stride=(s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1), data_type=cudnn_itype)
    else:
        table_size = math.ceil(s_kv / block_size)
        num_blocks = table_size * b

        q = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=(s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1), data_type=cudnn_itype)
        k = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(num_blocks, h_k, block_size, d_qk), stride=(block_size * h_k * d_qk, block_size * d_qk, d_qk, 1), data_type=cudnn_itype)
        v = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(num_blocks, h_v, block_size, d_vo), stride=(block_size * h_v * d_vo, block_size * d_vo, d_vo, 1), data_type=cudnn_itype)

        use_padding_mask = True
        kv_seq_len = graph_fwd.tensor(uid=GraphFwdUid.kv_seq_len, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        q_seq_len = graph_fwd.tensor(uid=GraphFwdUid.q_seq_len, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        k_block_table = graph_fwd.tensor(uid=GraphFwdUid.k_block_table, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)
        v_block_table = graph_fwd.tensor(uid=GraphFwdUid.v_block_table, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)

    q_descale = graph_fwd.tensor(uid=GraphFwdUid.q_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    k_descale = graph_fwd.tensor(uid=GraphFwdUid.k_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    v_descale = graph_fwd.tensor(uid=GraphFwdUid.v_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_scale = graph_fwd.tensor(uid=GraphFwdUid.s_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_descale = graph_fwd.tensor(uid=GraphFwdUid.s_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    o_scale = graph_fwd.tensor(uid=GraphFwdUid.o_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    o, stats, amax_s, amax_o = graph_fwd.sdpa_fp8(
        q=q, k=k, v=v,
        descale_q=q_descale, descale_k=k_descale, descale_v=v_descale,
        scale_s=s_scale, descale_s=s_descale, scale_o=o_scale,
        generate_stats=True, attn_scale=attn_scale, use_causal_mask=False,
        use_padding_mask=use_padding_mask, seq_len_kv=kv_seq_len, seq_len_q=q_seq_len,
        paged_attention_k_table=k_block_table, paged_attention_v_table=v_block_table,
        paged_attention_max_seq_len_kv=s_kv,
    )

    o.set_uid(GraphFwdUid.o).set_output(True).set_dim((b, h_q, s_qo, d_vo)).set_stride((s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1)).set_data_type(cudnn_otype)
    stats.set_uid(GraphFwdUid.stats).set_output(True).set_dim((b, h_q, s_qo, 1)).set_stride((s_qo * h_q, s_qo, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_s.set_uid(GraphFwdUid.s_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_uid(GraphFwdUid.o_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_fwd

def generate_graph_bwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, deterministic):
    graph_bwd = cudnn.pygraph(io_data_type=cudnn_itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    q = graph_bwd.tensor(uid=GraphBwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=(s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1), data_type=cudnn_itype)
    k = graph_bwd.tensor(uid=GraphBwdUid.k, dim=(b, h_k, s_kv, d_qk), stride=(s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1), data_type=cudnn_itype)
    v = graph_bwd.tensor(uid=GraphBwdUid.v, dim=(b, h_v, s_kv, d_vo), stride=(s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1), data_type=cudnn_itype)
    o = graph_bwd.tensor(uid=GraphBwdUid.o, dim=(b, h_q, s_qo, d_vo), stride=(s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1), data_type=cudnn_otype)
    dO = graph_bwd.tensor(uid=GraphBwdUid.dO, dim=(b, h_q, s_qo, d_vo), stride=(s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1), data_type=cudnn_itype)
    stats = graph_bwd.tensor(uid=GraphBwdUid.stats, dim=(b, h_q, s_qo, 1), stride=(s_qo * h_q, s_qo, 1, 1), data_type=cudnn.data_type.FLOAT)

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

    dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP = graph_bwd.sdpa_fp8_backward(
        q=q, k=k, v=v, o=o, dO=dO, stats=stats,
        descale_q=q_descale, descale_k=k_descale, descale_v=v_descale,
        descale_o=o_descale, descale_dO=dO_descale, descale_s=s_descale, descale_dP=dP_descale,
        scale_s=s_scale, scale_dQ=dQ_scale, scale_dK=dK_scale, scale_dV=dV_scale, scale_dP=dP_scale,
        attn_scale=attn_scale, use_padding_mask=False, use_deterministic_algorithm=deterministic,
    )

    dQ.set_uid(GraphBwdUid.dQ).set_output(True).set_dim((b, h_q, s_qo, d_qk)).set_stride((s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1)).set_data_type(cudnn_itype)
    dK.set_uid(GraphBwdUid.dK).set_output(True).set_dim((b, h_k, s_kv, d_qk)).set_stride((s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)).set_data_type(cudnn_itype)
    dV.set_uid(GraphBwdUid.dV).set_output(True).set_dim((b, h_v, s_kv, d_vo)).set_stride((s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1)).set_data_type(cudnn_itype)

    amax_dQ.set_uid(GraphBwdUid.dQ_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dK.set_uid(GraphBwdUid.dK_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dV.set_uid(GraphBwdUid.dV_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dP.set_uid(GraphBwdUid.dP_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

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
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SDPA FP8 requires Blackwell or higher")

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

    attn_scale = 0.125

    is_paged = block_size > 0

    try:
        if cfg.is_infer:
            graph = generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size)
        else:
            graph = generate_graph_bwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, deterministic)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"unsupported graph: {e}")
    except Exception as e:
        pytest.fail(f"Error building graph: {e}")

    rng_data = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)

    q_gen = torch.clamp(torch.randn(b, s_qo, h_q, d_qk, dtype=torch.float, device="cuda", generator=rng_data), min=-2.0, max=2.0)
    k_gen = torch.clamp(torch.randn(b, s_kv, h_k, d_qk, dtype=torch.float, device="cuda", generator=rng_data), min=-2.0, max=2.0)
    v_gen = torch.clamp(torch.randn(b, s_kv, h_v, d_vo, dtype=torch.float, device="cuda", generator=rng_data), min=-2.0, max=2.0)

    q_amax = q_gen.abs().max().item()
    k_amax = k_gen.abs().max().item()
    v_amax = v_gen.abs().max().item()
    s_amax, o_amax = compute_ref(q_gen, k_gen, v_gen, attn_scale, return_type="amax")

    q_gpu = (q_gen * get_fp8_scale_factor(q_amax, torch_itype)).to(torch_itype)
    k_gpu = (k_gen * get_fp8_scale_factor(k_amax, torch_itype)).to(torch_itype)
    v_gpu = (v_gen * get_fp8_scale_factor(v_amax, torch_itype)).to(torch_itype)

    if cfg.is_infer:
        if is_paged:
            k_gpu_bhsd = torch.einsum('bshd->bhsd', k_gpu).contiguous()
            v_gpu_bhsd = torch.einsum('bshd->bhsd', v_gpu).contiguous()
            container_k_gpu, k_block_table_gpu = create_paged_container_and_block_table(k_gpu_bhsd, block_size)
            container_v_gpu, v_block_table_gpu = create_paged_container_and_block_table(v_gpu_bhsd, block_size)

        kv_seq_len_gpu = torch.full((b, 1, 1, 1), s_kv, device="cuda", dtype=torch.int32)
        q_seq_len_gpu = torch.full((b, 1, 1, 1), s_qo, device="cuda", dtype=torch.int32)
        o_gpu = torch.full((b, s_qo, h_q, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        stats_gpu = torch.full((b, h_q, s_qo, 1), float('nan'), dtype=torch.float, device="cuda")

        q_descale_gpu = torch.tensor([get_fp8_descale_factor(q_amax, torch_itype)], dtype=torch.float, device="cuda")
        k_descale_gpu = torch.tensor([get_fp8_descale_factor(k_amax, torch_itype)], dtype=torch.float, device="cuda")
        v_descale_gpu = torch.tensor([get_fp8_descale_factor(v_amax, torch_itype)], dtype=torch.float, device="cuda")
        s_scale_gpu = torch.tensor([get_fp8_scale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")
        s_descale_gpu = torch.tensor([get_fp8_descale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")
        o_scale_gpu = torch.tensor([get_fp8_scale_factor(o_amax, torch_otype)], dtype=torch.float, device="cuda")

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
            variant_pack[int(GraphFwdUid.kv_seq_len)] = kv_seq_len_gpu
            variant_pack[int(GraphFwdUid.q_seq_len)] = q_seq_len_gpu
            variant_pack[int(GraphFwdUid.k_block_table)] = k_block_table_gpu
            variant_pack[int(GraphFwdUid.v_block_table)] = v_block_table_gpu

        workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
        graph.execute(variant_pack, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()

        q_ref = q_gpu.detach().float() * get_fp8_descale_factor(q_amax, torch_itype)
        k_ref = k_gpu.detach().float() * get_fp8_descale_factor(k_amax, torch_itype)
        v_ref = v_gpu.detach().float() * get_fp8_descale_factor(v_amax, torch_itype)
        o_ref = compute_ref(q_ref, k_ref, v_ref, attn_scale=attn_scale)

        o_gpu_comp = o_gpu.detach().float() * get_fp8_descale_factor(o_amax, torch_otype)

        atol, rtol = 0.08, 0.2
        if torch_itype == torch.float8_e5m2:
            atol, rtol = 0.16, 0.4

        torch.testing.assert_close(o_gpu_comp, o_ref, atol=atol, rtol=rtol)

    else:
        dO_gen = torch.clamp(torch.randn(b, s_qo, h_q, d_vo, dtype=torch.float, device="cuda", generator=rng_data), min=-2.0, max=2.0)
        dO_amax = dO_gen.abs().max().item()

        q_gpu = q_gen.to(torch_itype)
        k_gpu = k_gen.to(torch_itype)
        v_gpu = v_gen.to(torch_itype)

        graph_fwd = generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, 0)
        graph_fwd.validate(); graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_fwd.check_support(); graph_fwd.build_plans()

        o_gpu = torch.full((b, s_qo, h_q, d_vo), float('nan'), dtype=torch_otype, device="cuda")
        stats_gpu = torch.full((b, h_q, s_qo, 1), float('nan'), dtype=torch.float, device="cuda")
        dO_gpu = dO_gen.to(torch_itype)

        q_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        k_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        v_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        s_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        s_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        o_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        s_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        o_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")

        variant_pack_fwd = {
            int(GraphFwdUid.q): q_gpu, int(GraphFwdUid.k): k_gpu, int(GraphFwdUid.v): v_gpu,
            int(GraphFwdUid.q_descale): q_descale_gpu, int(GraphFwdUid.k_descale): k_descale_gpu,
            int(GraphFwdUid.v_descale): v_descale_gpu, int(GraphFwdUid.s_descale): s_descale_gpu,
            int(GraphFwdUid.s_scale): s_scale_gpu, int(GraphFwdUid.o_scale): o_scale_gpu,
            int(GraphFwdUid.o): o_gpu, int(GraphFwdUid.stats): stats_gpu,
            int(GraphFwdUid.s_amax): s_amax_gpu, int(GraphFwdUid.o_amax): o_amax_gpu,
        }

        workspace_fwd = torch.empty(graph_fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
        graph_fwd.execute(variant_pack_fwd, workspace_fwd, handle=cudnn_handle)
        torch.cuda.synchronize()

        o_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        dO_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        dP_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        dQ_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        dK_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        dV_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
        dP_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")

        dQ_gpu = torch.full((b, s_qo, h_q, d_qk), float('nan'), dtype=torch_itype, device="cuda")
        dK_gpu = torch.full((b, s_kv, h_k, d_qk), float('nan'), dtype=torch_itype, device="cuda")
        dV_gpu = torch.full((b, s_kv, h_v, d_vo), float('nan'), dtype=torch_itype, device="cuda")
        dQ_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dK_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dV_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
        dP_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")

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

        workspace_bwd = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
        graph.execute(variant_pack_bwd, workspace_bwd, handle=cudnn_handle)
        torch.cuda.synchronize()

        q_ref = q_gpu.detach().float()
        k_ref = k_gpu.detach().float()
        v_ref = v_gpu.detach().float()

        q_ref.requires_grad_(True)
        k_ref.requires_grad_(True)
        v_ref.requires_grad_(True)
        o_tmp = compute_ref(q_ref, k_ref, v_ref, attn_scale=attn_scale)
        dQ_ref, dK_ref, dV_ref = torch.autograd.grad(outputs=o_tmp, inputs=[q_ref, k_ref, v_ref], grad_outputs=dO_gen)

        dQ_out = dQ_gpu.detach().float()
        dK_out = dK_gpu.detach().float()
        dV_out = dV_gpu.detach().float()

        atol, rtol = 0.16, 0.2
        torch.testing.assert_close(dQ_out, dQ_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(dK_out, dK_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(dV_out, dV_ref, atol=atol, rtol=rtol)
