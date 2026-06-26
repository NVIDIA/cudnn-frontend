import math

import cudnn
import pytest
import torch
from looseversion import LooseVersion

from sdpa.fp16 import exec_sdpa
from sdpa.fp16_ref import compute_ref, compute_ref_backward
from sdpa.random_config import ExecConfig
from test_utils import torch_fork_set_rng


def require_cudnn(cudnn_handle):
    if cudnn_handle is None:
        pytest.skip("cuDNN backend not available")
    if LooseVersion(cudnn.backend_version_string()) < "9.10.0":
        pytest.skip("SDPA edge cases require cuDNN 9.10.0 or higher")


def make_bshd_tensor(shape, dtype, fill=None):
    b, h, s, d = shape
    strides = (s * h * d, d, h * d, 1)
    if fill is None:
        storage = torch.randn(b * s * h * d, dtype=dtype, device="cuda")
    else:
        storage = torch.full((b * s * h * d,), fill, dtype=dtype, device="cuda")
    return torch.as_strided(storage, shape, strides)


def build_graph(graph, label):
    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"unsupported {label} graph: {e}")
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)


def assert_no_nan_or_inf(name, tensor):
    assert torch.isfinite(tensor.float()).all(), f"{name} has NaN/Inf"


def run_sdpa_fwd_bwd(
    cudnn_handle,
    q_gpu,
    k_gpu,
    v_gpu,
    seq_len_q_gpu=None,
    seq_len_kv_gpu=None,
    right_bound=None,
    diag_align=None,
):
    cudnn.set_stream(handle=cudnn_handle, stream=torch.cuda.current_stream().cuda_stream)

    b, h_q, s_q, d_qk = q_gpu.shape
    _, h_k, s_kv, _ = k_gpu.shape
    _, h_v, _, d_v = v_gpu.shape
    attn_scale = 1.0 / math.sqrt(d_qk)

    o_gpu = torch.empty_like(q_gpu) if d_qk == d_v else make_bshd_tensor((b, h_q, s_q, d_v), q_gpu.dtype)
    stats_gpu = torch.empty((b, h_q, s_q, 1), dtype=torch.float32, device="cuda")

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)
    seq_len_q = graph.tensor_like(seq_len_q_gpu) if seq_len_q_gpu is not None else None
    seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if seq_len_kv_gpu is not None else None

    sdpa_kwargs = {
        "name": "sdpa_forward",
        "q": q,
        "k": k,
        "v": v,
        "generate_stats": True,
        "attn_scale": attn_scale,
        "use_causal_mask": False,
        "use_padding_mask": seq_len_q is not None,
        "seq_len_q": seq_len_q,
        "seq_len_kv": seq_len_kv,
    }
    if right_bound is not None:
        sdpa_kwargs["diagonal_band_right_bound"] = right_bound
    if diag_align is not None:
        sdpa_kwargs["diagonal_alignment"] = diag_align
    o, stats = graph.sdpa(**sdpa_kwargs)
    o.set_output(True).set_dim(o_gpu.size()).set_stride(o_gpu.stride())
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(stats_gpu.size()).set_stride(stats_gpu.stride())

    build_graph(graph, "forward")

    fwd_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        stats: stats_gpu,
        seq_len_q: seq_len_q_gpu,
        seq_len_kv: seq_len_kv_gpu,
    }
    fwd_pack = {k: v for k, v in fwd_pack.items() if k is not None}
    graph.execute(fwd_pack, torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda"), cudnn_handle)
    torch.cuda.synchronize()

    dO_gpu = make_bshd_tensor(o_gpu.shape, o_gpu.dtype)
    dQ_gpu = torch.empty_like(q_gpu)
    dK_gpu = torch.empty_like(k_gpu)
    dV_gpu = torch.empty_like(v_gpu)

    bwd_graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    q_bwd = bwd_graph.tensor_like(q_gpu)
    k_bwd = bwd_graph.tensor_like(k_gpu)
    v_bwd = bwd_graph.tensor_like(v_gpu)
    o_bwd = bwd_graph.tensor_like(o_gpu)
    dO_bwd = bwd_graph.tensor_like(dO_gpu)
    stats_bwd = bwd_graph.tensor_like(stats_gpu)
    seq_len_q_bwd = bwd_graph.tensor_like(seq_len_q_gpu) if seq_len_q_gpu is not None else None
    seq_len_kv_bwd = bwd_graph.tensor_like(seq_len_kv_gpu) if seq_len_kv_gpu is not None else None

    bwd_kwargs = {
        "name": "sdpa_backward",
        "q": q_bwd,
        "k": k_bwd,
        "v": v_bwd,
        "o": o_bwd,
        "dO": dO_bwd,
        "stats": stats_bwd,
        "attn_scale": attn_scale,
        "use_causal_mask": False,
        "use_padding_mask": seq_len_q_bwd is not None,
        "seq_len_q": seq_len_q_bwd,
        "seq_len_kv": seq_len_kv_bwd,
    }
    if right_bound is not None:
        bwd_kwargs["diagonal_band_right_bound"] = right_bound
    if diag_align is not None:
        bwd_kwargs["diagonal_alignment"] = diag_align
    dQ, dK, dV = bwd_graph.sdpa_backward(**bwd_kwargs)
    dQ.set_output(True).set_dim(q_gpu.size()).set_stride(q_gpu.stride())
    dK.set_output(True).set_dim(k_gpu.size()).set_stride(k_gpu.stride())
    dV.set_output(True).set_dim(v_gpu.size()).set_stride(v_gpu.stride())

    build_graph(bwd_graph, "backward")

    bwd_pack = {
        q_bwd: q_gpu,
        k_bwd: k_gpu,
        v_bwd: v_gpu,
        o_bwd: o_gpu,
        dO_bwd: dO_gpu,
        stats_bwd: stats_gpu,
        dQ: dQ_gpu,
        dK: dK_gpu,
        dV: dV_gpu,
        seq_len_q_bwd: seq_len_q_gpu,
        seq_len_kv_bwd: seq_len_kv_gpu,
    }
    bwd_pack = {k: v for k, v in bwd_pack.items() if k is not None}
    bwd_graph.execute(
        bwd_pack,
        torch.empty(bwd_graph.get_workspace_size(), dtype=torch.uint8, device="cuda"),
        cudnn_handle,
    )
    torch.cuda.synchronize()

    diag_align_ref = diag_align if diag_align is not None else cudnn.diagonal_alignment.TOP_LEFT
    padding_ref = None
    if seq_len_q_gpu is not None:
        padding_ref = (seq_len_q_gpu.flatten().detach(), seq_len_kv_gpu.flatten().detach())
    o_ref, stats_ref, _, _ = compute_ref(
        q_gpu.detach(),
        k_gpu.detach(),
        v_gpu.detach(),
        attn_scale=attn_scale,
        padding=padding_ref,
        right_bound=right_bound,
        diag_align=diag_align_ref,
        torch_type=q_gpu.dtype,
    )
    dQ_ref, dK_ref, dV_ref, _, _ = compute_ref_backward(
        q_gpu.detach(),
        k_gpu.detach(),
        v_gpu.detach(),
        o_ref,
        dO_gpu.detach(),
        attn_scale=attn_scale,
        padding=padding_ref,
        right_bound=right_bound,
        diag_align=diag_align_ref,
        torch_type=q_gpu.dtype,
    )

    assert_no_nan_or_inf("o", o_gpu)
    assert_no_nan_or_inf("dQ", dQ_gpu)
    assert_no_nan_or_inf("dK", dK_gpu)
    assert_no_nan_or_inf("dV", dV_gpu)
    assert not torch.isnan(stats_gpu).any(), "stats has NaN"

    torch.testing.assert_close(o_gpu.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(stats_gpu, stats_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dQ_gpu.float(), dQ_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dK_gpu.float(), dK_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dV_gpu.float(), dV_ref.float(), atol=2e-2, rtol=2e-2)

    return o_gpu, stats_gpu, dQ_gpu, dK_gpu, dV_gpu


def make_edge_config(seq_len_q, seq_len_kv):
    cfg = ExecConfig(
        data_type=torch.float16,
        rng_data_seed=1234,
        rng_geom_seed=5678,
        is_alibi=False,
        is_infer=False,
        is_paged=False,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=False,
        is_ragged=True,
        is_dropout=False,
        is_determin=False,
        batches=len(seq_len_q),
        d_qk=64,
        d_v=64,
        s_q=128,
        s_kv=128,
        h_q=3,
        h_k=3,
        h_v=3,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=None,
        right_bound=None,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        implementation=cudnn.attention_implementation.AUTO,
    )
    cfg.fill_derived_fields()
    return cfg


@pytest.mark.L0
@pytest.mark.parametrize(
    ("seq_len_q", "seq_len_kv"),
    [
        ([0, 128, 0, 128], [128, 128, 128, 128]),
        ([128, 128, 128, 128], [0, 128, 0, 128]),
        ([0, 128, 0, 128], [0, 128, 0, 128]),
    ],
    ids=["half_zero_q", "half_zero_kv", "half_zero_qkv"],
)
def test_thd_zero_seqlen(seq_len_q, seq_len_kv, request, cudnn_handle):
    require_cudnn(cudnn_handle)
    exec_sdpa(make_edge_config(seq_len_q, seq_len_kv), request, cudnn_handle)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_row_all_inf(cudnn_handle):
    require_cudnn(cudnn_handle)

    b, h, s, d = 2, 3, 128, 64
    q_gpu = make_bshd_tensor((b, h, s, d), torch.float16)
    k_gpu = make_bshd_tensor((b, h, s, d), torch.float16)
    v_gpu = make_bshd_tensor((b, h, s, d), torch.float16)
    seq_len_q_gpu = torch.tensor([128, 128], dtype=torch.int32, device="cuda")
    seq_len_kv_gpu = torch.tensor([64, 64], dtype=torch.int32, device="cuda")

    _, stats_gpu, *_ = run_sdpa_fwd_bwd(
        cudnn_handle,
        q_gpu,
        k_gpu,
        v_gpu,
        seq_len_q_gpu=seq_len_q_gpu,
        seq_len_kv_gpu=seq_len_kv_gpu,
        right_bound=0,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT,
    )

    assert torch.isneginf(stats_gpu[:, :, :64, :]).all()
    assert torch.isfinite(stats_gpu[:, :, 64:, :]).all()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_col_all_inf(cudnn_handle):
    require_cudnn(cudnn_handle)

    b, h, s_q, s_kv, d = 2, 3, 128, 768, 64
    q_gpu = make_bshd_tensor((b, h, s_q, d), torch.float16)
    k_gpu = make_bshd_tensor((b, h, s_kv, d), torch.float16)
    v_gpu = make_bshd_tensor((b, h, s_kv, d), torch.float16)

    _, stats_gpu, _, dK_gpu, dV_gpu = run_sdpa_fwd_bwd(
        cudnn_handle,
        q_gpu,
        k_gpu,
        v_gpu,
        right_bound=0,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
    )

    assert torch.isfinite(stats_gpu).all()
    torch.testing.assert_close(dK_gpu[:, :, s_q:, :].float(), torch.zeros_like(dK_gpu[:, :, s_q:, :].float()))
    torch.testing.assert_close(dV_gpu[:, :, s_q:, :].float(), torch.zeros_like(dV_gpu[:, :, s_q:, :].float()))


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_slightly_negative_row_max(cudnn_handle):
    require_cudnn(cudnn_handle)

    b, h, s, d = 2, 3, 128, 64
    q_gpu = make_bshd_tensor((b, h, s, d), torch.float16, fill=0.25)
    k_gpu = make_bshd_tensor((b, h, s, d), torch.float16, fill=-0.03125)
    v_gpu = make_bshd_tensor((b, h, s, d), torch.float16)

    scores = torch.einsum("bhqd,bhkd->bhqk", q_gpu.float(), k_gpu.float()) * (1.0 / math.sqrt(d))
    row_max = scores.max(dim=-1).values
    assert (row_max < 0).all()
    assert (row_max > -0.1).all()

    run_sdpa_fwd_bwd(cudnn_handle, q_gpu, k_gpu, v_gpu)
