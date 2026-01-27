import cudnn
import pytest
import torch
from enum import IntEnum
from looseversion import LooseVersion

from .fp16_ref import compute_ref
from .helpers import (
    convert_to_cudnn_type,
    exact_equal,
    approx_equal,
    alloc_tensor,
    prefix_sum,
    convert_packed_to_uniform,
    convert_uniform_to_packed,
    create_container_and_page_table,
    time_execution,
    profile_execution,
)

# fmt: off

class TensorUid(IntEnum):
    q = 0
    k = 1
    v = 2
    o = 3
    stats = 4
    bias = 5
    dQ = 6
    dK = 7
    dV = 8
    dO = 9
    dBias = 10
    seq_len_q = 11
    seq_len_kv = 12
    q_ragged_offset = 13
    k_ragged_offset = 14
    v_ragged_offset = 15
    o_ragged_offset = 16
    stats_ragged_offset = 17
    seed = 18
    offset = 19
    rng_dump = 20
    block_mask = 21
    container_k = 22
    container_v = 23
    page_table_k = 24
    page_table_v = 25
    workspace = 26


def validate_config(cfg):
    if not all((x > 0 and type(x) == int) for x in (cfg.batches, cfg.d_qk, cfg.d_v, cfg.s_q, cfg.s_kv, cfg.h_q, cfg.h_k, cfg.h_v)):
       assert False, "tensor dimensions must be integer and positive"

    assert cfg.shape_q == (cfg.batches, cfg.h_q, cfg.s_q, cfg.d_qk), f"wrong shape_q={cfg.shape_q}"
    assert cfg.shape_k == (cfg.batches, cfg.h_k, cfg.s_kv, cfg.d_qk), f"wrong shape_k={cfg.shape_k}"
    assert cfg.shape_v == (cfg.batches, cfg.h_v, cfg.s_kv, cfg.d_v), f"wrong shape_v={cfg.shape_v}"
    assert cfg.shape_o == (cfg.batches, cfg.h_q, cfg.s_q, cfg.d_v), f"wrong shape_o={cfg.shape_o}"

    if cfg.is_train:
        assert cfg.is_paged == False and cfg.block_size == None, "paged attention not allowed in backward pass"

    if cfg.is_ragged:
        assert cfg.is_padding == True, "is_ragged=True and is_padding=False not allowed"

    assert isinstance(cfg.seq_len_q, (list, tuple)), "input 'seq_len_q' must be list or tuple"
    if cfg.is_padding:
        assert len(cfg.seq_len_q) == cfg.batches, f"wrong 'seq_len_q' length"
    else:
        assert len(cfg.seq_len_q) == 0, f"wrong 'seq_len_q' length, expecting 0"

    assert isinstance(cfg.seq_len_kv, (list, tuple)), "input 'seq_len_kv' must be list or tuple"
    if cfg.is_padding:
        assert len(cfg.seq_len_kv) == cfg.batches, f"wrong 'seq_len_kv' length, expecting {cfg.batches}"
    else:
        assert len(cfg.seq_len_kv) == 0, f"wrong 'seq_len_kv' length, expecting 0"

    assert all(x >= 0 and type(x) == int for x in cfg.seq_len_q), f"wrong seq_len_q={cfg.seq_len_q}"
    assert all(x >= 0 and type(x) == int for x in cfg.seq_len_kv), f"wrong seq_len_kv={cfg.seq_len_kv}"

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.10.0":
        print("@@@@ Overall result: WAIVED, test_mhas_v2.py supports cudnn 9.10.0 or higher.")
        pytest.skip("test_mhas_v2.py requires cudnn 9.10.0 or higher")

    if cudnn_version < "9.13.1" and cfg.implementation == cudnn.attention_implementation.UNIFIED:
        print("@@@@ Overall result: WAIVED, unified SDPA implementation requires cudnn 9.13.1 or higher.")
        pytest.skip("unified SDPA implementation requires cudnn 9.13.1 or higher")

    if cfg.s_q == cfg.s_kv == 1:
        print("@@@@ Overall result: WAIVED, skipping known issue of s_q == s_kv == 1.")
        pytest.skip("skipping known issue of s_q == s_kv == 1")


def allocate_tensors(cfg, rng_data_gen):
    allocs = {}
    max_t_q = max(64, ((sum(cfg.seq_len_q) + 63) // 64) * 64) if cfg.is_ragged else None
    max_t_kv = max(64, ((sum(cfg.seq_len_kv) + 63) // 64) * 64) if cfg.is_ragged else None

    if cfg.is_ragged:
        allocs[TensorUid.q] = alloc_tensor((max_t_q, cfg.h_q, cfg.d_qk), cfg.data_type, rng=rng_data_gen, mean=-0.5, std=1.0)
        allocs[TensorUid.k] = alloc_tensor((max_t_kv, cfg.h_k, cfg.d_qk), cfg.data_type, rng=rng_data_gen, mean=-0.5, std=1.0)
        allocs[TensorUid.v] = alloc_tensor((max_t_kv, cfg.h_v, cfg.d_v), cfg.data_type, rng=rng_data_gen, mean=-0.5, std=1.0)
        allocs[TensorUid.o] = alloc_tensor((max_t_q, cfg.h_q, cfg.d_v), cfg.data_type)
        allocs[TensorUid.stats] = alloc_tensor((max_t_q, cfg.h_q, 1), torch.float32) if cfg.is_train else (None, None, None)
        if cfg.is_train:
            allocs[TensorUid.dQ] = alloc_tensor((max_t_q, cfg.h_q, cfg.d_qk), cfg.data_type)
            allocs[TensorUid.dK] = alloc_tensor((max_t_kv, cfg.h_k, cfg.d_qk), cfg.data_type)
            allocs[TensorUid.dV] = alloc_tensor((max_t_kv, cfg.h_v, cfg.d_v), cfg.data_type)
            allocs[TensorUid.dO] = alloc_tensor((max_t_q, cfg.h_q, cfg.d_v), cfg.data_type, rng=rng_data_gen, mean=0.0, std=0.1)
    else:
        allocs[TensorUid.q] = alloc_tensor(cfg.shape_q, cfg.data_type, strides=cfg.stride_q, rng=rng_data_gen, mean=-0.5, std=1.0)
        allocs[TensorUid.k] = alloc_tensor(cfg.shape_k, cfg.data_type, strides=cfg.stride_k, rng=rng_data_gen, mean=-0.5, std=1.0)
        allocs[TensorUid.v] = alloc_tensor(cfg.shape_v, cfg.data_type, strides=cfg.stride_v, rng=rng_data_gen, mean=-0.5, std=1.0)
        allocs[TensorUid.o] = alloc_tensor(cfg.shape_o, cfg.data_type, strides=cfg.stride_o)
        allocs[TensorUid.stats] = alloc_tensor((cfg.batches, cfg.h_q, cfg.s_q, 1), torch.float32) if cfg.is_train else (None, None, None)
        if cfg.is_train:
            allocs[TensorUid.dQ] = alloc_tensor(cfg.shape_q, cfg.data_type, strides=cfg.stride_q)
            allocs[TensorUid.dK] = alloc_tensor(cfg.shape_k, cfg.data_type, strides=cfg.stride_k)
            allocs[TensorUid.dV] = alloc_tensor(cfg.shape_v, cfg.data_type, strides=cfg.stride_v)
            allocs[TensorUid.dO] = alloc_tensor(cfg.shape_o, cfg.data_type, strides=cfg.stride_o, rng=rng_data_gen, mean=0.0, std=0.1)

    seq_len_q_gpu = torch.tensor(cfg.seq_len_q, dtype=torch.int32, device="cuda").view(-1, 1, 1, 1) if len(cfg.seq_len_q) > 0 else None
    seq_len_kv_gpu = torch.tensor(cfg.seq_len_kv, dtype=torch.int32, device="cuda").view(-1, 1, 1, 1) if len(cfg.seq_len_kv) > 0 else None
    allocs[TensorUid.seq_len_q] = (seq_len_q_gpu, None, None)
    allocs[TensorUid.seq_len_kv] = (seq_len_kv_gpu, None, None)

    if cfg.is_ragged:
        allocs[TensorUid.q_ragged_offset] = ((prefix_sum(seq_len_q_gpu) * cfg.h_q * cfg.d_qk).to(torch.int64), None, None)
        allocs[TensorUid.k_ragged_offset] = ((prefix_sum(seq_len_kv_gpu) * cfg.h_k * cfg.d_qk).to(torch.int64), None, None)
        allocs[TensorUid.v_ragged_offset] = ((prefix_sum(seq_len_kv_gpu) * cfg.h_v * cfg.d_v).to(torch.int64), None, None)
        allocs[TensorUid.o_ragged_offset] = ((prefix_sum(seq_len_q_gpu) * cfg.h_q * cfg.d_v).to(torch.int64), None, None)
        allocs[TensorUid.stats_ragged_offset] = ((prefix_sum(seq_len_q_gpu) * cfg.h_q * 1).to(torch.int64), None, None)

    if cfg.is_bias:
        allocs[TensorUid.bias] = alloc_tensor((1, cfg.h_q, cfg.s_q, cfg.s_kv), cfg.data_type, rng=rng_data_gen, mean=0.0, std=1.0)
    if cfg.is_train and cfg.is_bias:
        allocs[TensorUid.dBias] = alloc_tensor((1, cfg.h_q, cfg.s_q, cfg.s_kv), cfg.data_type)

    if cfg.is_block_mask:
        TILE_M, TILE_N = 128, 128
        block_mask_gpu = torch.randint(0, 256, (cfg.batches, cfg.h_q, (cfg.s_q + TILE_M - 1) // TILE_M, ((cfg.s_kv + TILE_N - 1) // TILE_N + 7) // 8), dtype=torch.uint8, device="cuda")
        allocs[TensorUid.block_mask] = (block_mask_gpu, None, None)

    if cfg.is_dropout:
        allocs[TensorUid.seed] = (torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda"), None, None)
        allocs[TensorUid.offset] = (torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda"), None, None)
        allocs[TensorUid.rng_dump] = (torch.zeros((cfg.batches, cfg.h_q, cfg.s_q, cfg.s_kv), dtype=torch.float32, device="cuda"), None, None)

    if cfg.is_paged:
        container_k, page_table_k = create_container_and_page_table(allocs[TensorUid.k][0], cfg.block_size)
        container_v, page_table_v = create_container_and_page_table(allocs[TensorUid.v][0], cfg.block_size)
        allocs[TensorUid.container_k] = (container_k, None, None)
        allocs[TensorUid.container_v] = (container_v, None, None)
        allocs[TensorUid.page_table_k] = (page_table_k, None, None)
        allocs[TensorUid.page_table_v] = (page_table_v, None, None)

    tensors = {uid: alloc[0] for uid, alloc in allocs.items()}
    return allocs, tensors, max_t_q, max_t_kv


def create_forward_graph(cfg, tensors, cudnn_handle):
    cudnn_dtype = convert_to_cudnn_type(cfg.data_type)
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)
    graph = cudnn.pygraph(
        io_data_type=cudnn_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    q = graph.tensor(uid=int(TensorUid.q), dim=cfg.shape_q, stride=cfg.stride_q, data_type=cudnn_dtype)
    k = graph.tensor(uid=int(TensorUid.k), dim=cfg.shape_k, stride=cfg.stride_k, data_type=cudnn_dtype)
    v = graph.tensor(uid=int(TensorUid.v), dim=cfg.shape_v, stride=cfg.stride_v, data_type=cudnn_dtype)

    page_table_k = page_table_v = paged_attention_max_seq_len_kv = None
    if cfg.is_paged:
        container_k_gpu = tensors.get(TensorUid.container_k)
        container_v_gpu = tensors.get(TensorUid.container_v)
        page_table_k_gpu = tensors.get(TensorUid.page_table_k)
        page_table_v_gpu = tensors.get(TensorUid.page_table_v)
        k = graph.tensor(uid=int(TensorUid.container_k), dim=container_k_gpu.size(), stride=container_k_gpu.stride(), data_type=cudnn_dtype)
        v = graph.tensor(uid=int(TensorUid.container_v), dim=container_v_gpu.size(), stride=container_v_gpu.stride(), data_type=cudnn_dtype)
        page_table_k = graph.tensor(uid=int(TensorUid.page_table_k), dim=page_table_k_gpu.size(), stride=page_table_k_gpu.stride(), data_type=cudnn.data_type.INT32)
        page_table_v = graph.tensor(uid=int(TensorUid.page_table_v), dim=page_table_v_gpu.size(), stride=page_table_v_gpu.stride(), data_type=cudnn.data_type.INT32)
        paged_attention_max_seq_len_kv = cfg.s_kv

    bias = graph.tensor(uid=int(TensorUid.bias), dim=(1, cfg.h_q, cfg.s_q, cfg.s_kv), stride=(cfg.h_q * cfg.s_q * cfg.s_kv, cfg.s_q * cfg.s_kv, cfg.s_kv, 1), data_type=cudnn_dtype) if cfg.is_bias else None

    TILE_M, TILE_N = 128, 128
    block_mask_dim = (cfg.batches, cfg.h_q, (cfg.s_q + TILE_M - 1) // TILE_M, ((cfg.s_kv + TILE_N - 1) // TILE_N + 7) // 8)
    block_mask = graph.tensor(uid=int(TensorUid.block_mask), dim=block_mask_dim, stride=(block_mask_dim[1]*block_mask_dim[2]*block_mask_dim[3], block_mask_dim[2]*block_mask_dim[3], block_mask_dim[3], 1), data_type=cudnn.data_type.UINT8) if cfg.is_block_mask else None

    seq_len_q = graph.tensor(uid=int(TensorUid.seq_len_q), dim=(cfg.batches, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32) if cfg.is_padding else None
    seq_len_kv = graph.tensor(uid=int(TensorUid.seq_len_kv), dim=(cfg.batches, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32) if cfg.is_padding else None

    seed = offset = dropout_tuple = rng_dump = None
    if cfg.is_dropout:
        seed = graph.tensor(uid=int(TensorUid.seed), dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        offset = graph.tensor(uid=int(TensorUid.offset), dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        dropout_tuple = (cfg.dropout_prob, seed, offset)
        rng_dump = graph.tensor(uid=int(TensorUid.rng_dump), dim=(cfg.batches, cfg.h_q, cfg.s_q, cfg.s_kv), stride=(cfg.h_q * cfg.s_q * cfg.s_kv, cfg.s_q * cfg.s_kv, cfg.s_kv, 1), data_type=cudnn.data_type.FLOAT)

    q_ragged_offset = graph.tensor(uid=int(TensorUid.q_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64) if cfg.is_ragged else None
    k_ragged_offset = graph.tensor(uid=int(TensorUid.k_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64) if cfg.is_ragged else None
    v_ragged_offset = graph.tensor(uid=int(TensorUid.v_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64) if cfg.is_ragged else None
    o_ragged_offset = graph.tensor(uid=int(TensorUid.o_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64) if cfg.is_ragged else None
    stats_ragged_offset = graph.tensor(uid=int(TensorUid.stats_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64) if cfg.is_ragged and cfg.is_train else None

    if cfg.is_ragged:
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)

    attn_scale = 0.125

    o, stats = graph.sdpa(
        name="sdpa_forward",
        q=q, k=k, v=v,
        generate_stats=cfg.is_train,
        attn_scale=attn_scale,
        bias=bias,
        block_mask=block_mask,
        use_alibi_mask=cfg.is_alibi,
        use_padding_mask=cfg.is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        diagonal_band_left_bound=cfg.left_bound,
        diagonal_band_right_bound=cfg.right_bound,
        diagonal_alignment=cfg.diag_align,
        dropout=dropout_tuple,
        rng_dump=rng_dump,
        paged_attention_k_table=page_table_k,
        paged_attention_v_table=page_table_v,
        paged_attention_max_seq_len_kv=paged_attention_max_seq_len_kv,
        implementation=cfg.implementation,
    )

    o.set_uid(int(TensorUid.o)).set_output(True).set_dim(cfg.shape_o).set_stride(cfg.stride_o)
    if cfg.is_ragged:
        o.set_ragged_offset(o_ragged_offset)

    if cfg.is_train:
        dim_stats = (cfg.batches, cfg.h_q, cfg.s_q, 1)
        stride_stats = (cfg.s_q * cfg.h_q, 1, cfg.h_q, 1) if cfg.is_ragged else (cfg.h_q * cfg.s_q, cfg.s_q, 1, 1)
        stats.set_uid(int(TensorUid.stats)).set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(dim_stats).set_stride(stride_stats)
        if cfg.is_ragged:
            stats.set_ragged_offset(stats_ragged_offset)

    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"@@@@ Overall result: WAIVED, not supported forward graph. {e}")
        pytest.skip("not supported forward graph")
    except Exception as e:
        print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception during forward graph build. {e}")
        pytest.fail("unexpected exception during forward graph build", pytrace=False)

    variant_pack = {
        int(TensorUid.q): tensors.get(TensorUid.q),
        int(TensorUid.container_k) if cfg.is_paged else int(TensorUid.k): tensors.get(TensorUid.container_k) if cfg.is_paged else tensors.get(TensorUid.k),
        int(TensorUid.container_v) if cfg.is_paged else int(TensorUid.v): tensors.get(TensorUid.container_v) if cfg.is_paged else tensors.get(TensorUid.v),
        int(TensorUid.bias): tensors.get(TensorUid.bias),
        int(TensorUid.block_mask): tensors.get(TensorUid.block_mask),
        int(TensorUid.seq_len_q): tensors.get(TensorUid.seq_len_q),
        int(TensorUid.seq_len_kv): tensors.get(TensorUid.seq_len_kv),
        int(TensorUid.q_ragged_offset): tensors.get(TensorUid.q_ragged_offset),
        int(TensorUid.k_ragged_offset): tensors.get(TensorUid.k_ragged_offset),
        int(TensorUid.v_ragged_offset): tensors.get(TensorUid.v_ragged_offset),
        int(TensorUid.o_ragged_offset): tensors.get(TensorUid.o_ragged_offset),
        int(TensorUid.stats_ragged_offset): tensors.get(TensorUid.stats_ragged_offset),
        int(TensorUid.o): tensors.get(TensorUid.o),
        int(TensorUid.stats): tensors.get(TensorUid.stats),
        int(TensorUid.page_table_k): tensors.get(TensorUid.page_table_k),
        int(TensorUid.page_table_v): tensors.get(TensorUid.page_table_v),
        int(TensorUid.seed): tensors.get(TensorUid.seed),
        int(TensorUid.offset): tensors.get(TensorUid.offset),
        int(TensorUid.rng_dump): tensors.get(TensorUid.rng_dump),
    }
    variant_pack = {k: v for k, v in variant_pack.items() if v is not None}

    return graph, variant_pack


def create_backward_graph(cfg, tensors, cudnn_handle, max_t_q, max_t_kv):
    cudnn_dtype = convert_to_cudnn_type(cfg.data_type)
    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)
    sm_version = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]

    graph = cudnn.pygraph(
        io_data_type=cudnn_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
        sm_version=sm_version
    )

    dim_stats = (cfg.batches, cfg.h_q, cfg.s_q, 1)
    stride_stats = (cfg.s_q * cfg.h_q, 1, cfg.h_q, 1) if cfg.is_ragged else (cfg.h_q * cfg.s_q, cfg.s_q, 1, 1)

    q = graph.tensor(uid=int(TensorUid.q), dim=cfg.shape_q, stride=cfg.stride_q, data_type=cudnn_dtype)
    k = graph.tensor(uid=int(TensorUid.k), dim=cfg.shape_k, stride=cfg.stride_k, data_type=cudnn_dtype)
    v = graph.tensor(uid=int(TensorUid.v), dim=cfg.shape_v, stride=cfg.stride_v, data_type=cudnn_dtype)
    o = graph.tensor(uid=int(TensorUid.o), dim=cfg.shape_o, stride=cfg.stride_o, data_type=cudnn_dtype)
    dO = graph.tensor(uid=int(TensorUid.dO), dim=cfg.shape_o, stride=cfg.stride_o, data_type=cudnn_dtype)
    stats = graph.tensor(uid=int(TensorUid.stats), dim=dim_stats, stride=stride_stats, data_type=cudnn.data_type.FLOAT)

    bias_dim = (1, cfg.h_q, cfg.s_q, cfg.s_kv)
    bias_stride = (cfg.h_q * cfg.s_q * cfg.s_kv, cfg.s_q * cfg.s_kv, cfg.s_kv, 1)
    bias = graph.tensor(uid=int(TensorUid.bias), dim=bias_dim, stride=bias_stride, data_type=cudnn_dtype) if cfg.is_bias else None
    dBias = graph.tensor(uid=int(TensorUid.dBias), dim=bias_dim, stride=bias_stride, data_type=cudnn_dtype) if cfg.is_bias else None

    seq_len_q = graph.tensor(uid=int(TensorUid.seq_len_q), dim=(cfg.batches, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32) if cfg.is_padding else None
    seq_len_kv = graph.tensor(uid=int(TensorUid.seq_len_kv), dim=(cfg.batches, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32) if cfg.is_padding else None

    seed = offset = dropout_tuple = None
    if cfg.is_dropout:
        seed = graph.tensor(uid=int(TensorUid.seed), dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        offset = graph.tensor(uid=int(TensorUid.offset), dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        dropout_tuple = (cfg.dropout_prob, seed, offset)

    attn_scale = 0.125

    dQ, dK, dV = graph.sdpa_backward(
        name="sdpa_backward",
        q=q, k=k, v=v, o=o, dO=dO, stats=stats,
        attn_scale=attn_scale,
        bias=bias,
        dBias=dBias,
        use_alibi_mask=cfg.is_alibi,
        use_padding_mask=cfg.is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        max_total_seq_len_q=max_t_q,
        max_total_seq_len_kv=max_t_kv,
        diagonal_band_left_bound=cfg.left_bound,
        diagonal_band_right_bound=cfg.right_bound,
        diagonal_alignment=cfg.diag_align,
        dropout=dropout_tuple,
        use_deterministic_algorithm=cfg.is_determin,
    )

    dQ.set_uid(int(TensorUid.dQ)).set_output(True).set_dim(cfg.shape_q).set_stride(cfg.stride_q)
    dK.set_uid(int(TensorUid.dK)).set_output(True).set_dim(cfg.shape_k).set_stride(cfg.stride_k)
    dV.set_uid(int(TensorUid.dV)).set_output(True).set_dim(cfg.shape_v).set_stride(cfg.stride_v)

    if cfg.is_ragged:
        q_ragged_offset = graph.tensor(uid=int(TensorUid.q_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        k_ragged_offset = graph.tensor(uid=int(TensorUid.k_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        v_ragged_offset = graph.tensor(uid=int(TensorUid.v_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        o_ragged_offset = graph.tensor(uid=int(TensorUid.o_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        stats_ragged_offset = graph.tensor(uid=int(TensorUid.stats_ragged_offset), dim=(cfg.batches + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64)
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)
        o.set_ragged_offset(o_ragged_offset)
        stats.set_ragged_offset(stats_ragged_offset)
        dQ.set_ragged_offset(q_ragged_offset)
        dK.set_ragged_offset(k_ragged_offset)
        dV.set_ragged_offset(v_ragged_offset)
        dO.set_ragged_offset(o_ragged_offset)

    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"@@@@ Overall result: WAIVED, not supported backward graph. {e}")
        pytest.skip("not supported backward graph")
    except Exception as e:
        print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception during backward graph build. {e}")
        pytest.fail("unexpected exception during backward graph build", pytrace=False)

    variant_pack = {
        int(TensorUid.q): tensors.get(TensorUid.q),
        int(TensorUid.k): tensors.get(TensorUid.k),
        int(TensorUid.v): tensors.get(TensorUid.v),
        int(TensorUid.o): tensors.get(TensorUid.o),
        int(TensorUid.stats): tensors.get(TensorUid.stats),
        int(TensorUid.dQ): tensors.get(TensorUid.dQ),
        int(TensorUid.dK): tensors.get(TensorUid.dK),
        int(TensorUid.dV): tensors.get(TensorUid.dV),
        int(TensorUid.dO): tensors.get(TensorUid.dO),
        int(TensorUid.bias): tensors.get(TensorUid.bias),
        int(TensorUid.dBias): tensors.get(TensorUid.dBias),
        int(TensorUid.seq_len_q): tensors.get(TensorUid.seq_len_q),
        int(TensorUid.seq_len_kv): tensors.get(TensorUid.seq_len_kv),
        int(TensorUid.q_ragged_offset): tensors.get(TensorUid.q_ragged_offset),
        int(TensorUid.k_ragged_offset): tensors.get(TensorUid.k_ragged_offset),
        int(TensorUid.v_ragged_offset): tensors.get(TensorUid.v_ragged_offset),
        int(TensorUid.o_ragged_offset): tensors.get(TensorUid.o_ragged_offset),
        int(TensorUid.stats_ragged_offset): tensors.get(TensorUid.stats_ragged_offset),
        int(TensorUid.seed): tensors.get(TensorUid.seed),
        int(TensorUid.offset): tensors.get(TensorUid.offset),
    }
    variant_pack = {k: v for k, v in variant_pack.items() if v is not None}

    return graph, variant_pack


def check_deterministic(cfg, tensors, allocs, bwd_graph, bwd_pack, cudnn_handle, request):
    if not cfg.is_determin:
        return
    
    dQ_gpu = tensors.get(TensorUid.dQ)
    dK_gpu = tensors.get(TensorUid.dK)
    dV_gpu = tensors.get(TensorUid.dV)
    workspace = allocs[TensorUid.workspace]

    dQ_gpu_rerun = dQ_gpu.clone().detach()
    dK_gpu_rerun = dK_gpu.clone().detach()
    dV_gpu_rerun = dV_gpu.clone().detach()

    torch.fill_(dQ_gpu, float("nan"))
    torch.fill_(dK_gpu, float("nan"))
    torch.fill_(dV_gpu, float("nan"))
    bwd_graph.execute(bwd_pack, workspace[0], cudnn_handle)
    torch.cuda.synchronize()

    determin_err_count = 0
    determin_err_count += exact_equal(dQ_gpu, dQ_gpu_rerun, tag="dQ_determin", disp_elems=request.config.getoption("--diffs"))
    determin_err_count += exact_equal(dK_gpu, dK_gpu_rerun, tag="dK_determin", disp_elems=request.config.getoption("--diffs"))
    determin_err_count += exact_equal(dV_gpu, dV_gpu_rerun, tag="dV_determin", disp_elems=request.config.getoption("--diffs"))

    if determin_err_count != 0:
        print("@@@@ Overall result: FAILED, determinism check failed - outputs differ between runs.")
        pytest.fail("determinism check failed", pytrace=False)
    print("@@@@ Determinism check: PASSED, dQ, dK, dV bitwise match between runs.")


def execute_graph(graph, variant_pack, allocs, tensors, cudnn_handle, request, label="Graph"):
    workspace = alloc_tensor(graph.get_workspace_size(), torch.uint8)
    allocs[TensorUid.workspace] = workspace
    tensors[TensorUid.workspace] = workspace[0]

    if request.config.getoption("--perf"):
        times_ms = time_execution(graph.execute, variant_pack, workspace[0], cudnn_handle)
        print(f"@@@@ {label} graph.execute avg_time_ms={times_ms.mean().item():.3f}")
        profile_execution(graph.execute, variant_pack, workspace[0], cudnn_handle)

    graph.execute(variant_pack, workspace[0], cudnn_handle)
    torch.cuda.synchronize()

    if workspace[1] is not None and not torch.all(workspace[1]==-1).item():
        print(f"@@@@ Overall result: FAILED, {label} workspace overwritten outside its boundaries.")
        print(workspace[1])
        pytest.fail(f"{label} workspace overwritten outside boundaries", pytrace=False)


def compute_and_compare_reference(cfg, allocs, tensors, diffs):
    cudnn_version = LooseVersion(cudnn.backend_version_string())

    q_gpu = tensors.get(TensorUid.q)
    k_gpu = tensors.get(TensorUid.k)
    v_gpu = tensors.get(TensorUid.v)
    dO_gpu = tensors.get(TensorUid.dO)
    seq_len_q_gpu = tensors.get(TensorUid.seq_len_q)
    seq_len_kv_gpu = tensors.get(TensorUid.seq_len_kv)
    block_mask_gpu = tensors.get(TensorUid.block_mask)
    bias_gpu = tensors.get(TensorUid.bias)
    rng_dump_gpu = tensors.get(TensorUid.rng_dump)

    q_ref = q_gpu.detach().float()
    k_ref = k_gpu.detach().float()
    v_ref = v_gpu.detach().float()
    dO_ref = dO_gpu.detach().float() if dO_gpu is not None else None
    seq_len_q_ref = seq_len_q_gpu.flatten().detach() if seq_len_q_gpu is not None else None
    seq_len_kv_ref = seq_len_kv_gpu.flatten().detach() if seq_len_kv_gpu is not None else None
    block_mask_ref = block_mask_gpu.detach() if block_mask_gpu is not None else None
    bias_ref = bias_gpu.detach().float() if bias_gpu is not None else None
    rng_dump_ref = rng_dump_gpu.detach().float() if rng_dump_gpu is not None else None

    if cfg.is_train:
        q_ref.requires_grad_()
        k_ref.requires_grad_()
        v_ref.requires_grad_()
    if cfg.is_train and cfg.is_bias:
        bias_ref.requires_grad_()

    if cfg.is_ragged:
        q_ref = convert_packed_to_uniform(q_ref, seq_len_q_ref, cfg.s_q)
        k_ref = convert_packed_to_uniform(k_ref, seq_len_kv_ref, cfg.s_kv)
        v_ref = convert_packed_to_uniform(v_ref, seq_len_kv_ref, cfg.s_kv)
    if cfg.is_ragged and cfg.is_train:
        dO_ref = convert_packed_to_uniform(dO_ref, seq_len_q_ref, cfg.s_q)

    max_t_q = max(64, ((seq_len_q_ref.sum().item() + 63) // 64) * 64) if cfg.is_ragged else None
    max_t_kv = max(64, ((seq_len_kv_ref.sum().item() + 63) // 64) * 64) if cfg.is_ragged else None

    attn_scale = 0.125

    ret = compute_ref(
        q_ref, k_ref, v_ref,
        attn_scale=attn_scale,
        bias=bias_ref,
        block_mask=block_mask_ref,
        is_alibi=cfg.is_alibi,
        padding=(seq_len_q_ref, seq_len_kv_ref) if cfg.is_padding else None,
        left_bound=cfg.left_bound,
        right_bound=cfg.right_bound,
        diag_align=cfg.diag_align,
        dropout_prob=cfg.dropout_prob,
        dropout_mask=rng_dump_ref,
        generate_stats=cfg.is_train,
    )

    o_ref, stats_ref = ret if cfg.is_train else (ret, None)

    o_gpu = tensors.get(TensorUid.o)
    stats_gpu = tensors.get(TensorUid.stats)

    if cfg.is_padding and not cfg.is_ragged:
        for i, m in enumerate(seq_len_q_ref):
            o_ref[i, :, m:, :] = 0
            o_gpu[i, :, m:, :] = 0
            if cfg.is_train:
                if cudnn_version < "9.14.0":
                    stats_ref[i, :, m:, :] = 0
                    stats_gpu[i, :, m:, :] = 0
                else:
                    stats_ref[i, :, m:, :] = -float("inf")

    if cfg.is_train:
        inputs_ref = [q_ref, k_ref, v_ref, bias_ref] if cfg.is_bias else [q_ref, k_ref, v_ref]
        grads = torch.autograd.grad(outputs=o_ref, inputs=inputs_ref, grad_outputs=dO_ref)
        dQ_ref = grads[0]
        dK_ref = grads[1]
        dV_ref = grads[2]
        dBias_ref = grads[3] if cfg.is_bias else None

    if cfg.is_train and cfg.is_padding:
        for i, (m, n) in enumerate(zip(seq_len_q_ref, seq_len_kv_ref)):
            dQ_ref[i, :, m:, :] = 0
            dK_ref[i, :, n:, :] = 0
            dV_ref[i, :, n:, :] = 0

    if cfg.is_ragged:
        o_ref = convert_uniform_to_packed(o_ref, seq_len_q_ref, max_t_q)
    if cfg.is_train and cfg.is_ragged:
        dQ_ref = convert_uniform_to_packed(dQ_ref, seq_len_q_ref, max_t_q)
        dK_ref = convert_uniform_to_packed(dK_ref, seq_len_kv_ref, max_t_kv)
        dV_ref = convert_uniform_to_packed(dV_ref, seq_len_kv_ref, max_t_kv)
        stats_ref = convert_uniform_to_packed(stats_ref, seq_len_q_ref, max_t_q)

    err_count = 0
    err_count += approx_equal(allocs[TensorUid.o], o_ref, atol=2e-2, rtol=2e-2, tag="o", disp_elems=diffs)
    if cfg.is_train:
        dkv_atol = 2e-2 if cfg.data_type == torch.float16 else 7e-2
        err_count += approx_equal(allocs[TensorUid.stats], stats_ref, atol=2e-2, rtol=2e-2, tag="stats", disp_elems=diffs)
        err_count += approx_equal(allocs[TensorUid.dQ], dQ_ref, atol=2e-2, rtol=2e-2, tag="dQ", disp_elems=diffs)
        err_count += approx_equal(allocs[TensorUid.dK], dK_ref, atol=dkv_atol, rtol=2e-2, tag="dK", disp_elems=diffs)
        err_count += approx_equal(allocs[TensorUid.dV], dV_ref, atol=dkv_atol, rtol=2e-2, tag="dV", disp_elems=diffs)
    if cfg.is_train and cfg.is_bias:
        err_count += approx_equal(allocs[TensorUid.dBias], dBias_ref, atol=2e-2, rtol=2e-2, tag="dBias", disp_elems=diffs)

    if err_count != 0:
        print("@@@@ Overall result: FAILED, disallowed mismatches")
        pytest.fail("disallowed mismatches", pytrace=False)
    else:
        print("@@@@ Overall result: PASSED, everything looks good!")


def cleanup_tensors(allocs):
    for uid in list(allocs.keys()):
        entry = allocs.get(uid)
        if entry is not None and entry[0] is not None:
            del allocs[uid]
    torch.cuda.empty_cache()


def exec_sdpa(cfg, request, cudnn_handle):
    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    validate_config(cfg)

    rng_data_gen = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)
    allocs, tensors, max_t_q, max_t_kv = allocate_tensors(cfg, rng_data_gen)

    fwd_graph, fwd_pack = create_forward_graph(cfg, tensors, cudnn_handle)
    bwd_graph, bwd_pack = create_backward_graph(cfg, tensors, cudnn_handle, max_t_q, max_t_kv) if cfg.is_train else (None, None)

    execute_graph(fwd_graph, fwd_pack, allocs, tensors, cudnn_handle, request, label="Forward")

    if cfg.is_train:
        execute_graph(bwd_graph, bwd_pack, allocs, tensors, cudnn_handle, request, label="Backward")
        check_deterministic(cfg, tensors, allocs, bwd_graph, bwd_pack, cudnn_handle, request)

    compute_and_compare_reference(cfg, allocs, tensors, request.config.getoption("--diffs"))
    cleanup_tensors(allocs)
