"""Test cuDNN grouped query attention support surface and tieout with PyTorch implementation"""

import functools

import cudnn
import torch
import torch.nn.functional as F
from torch.autograd.function import Function


def check_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> tuple[int, int, float]:
    """Similar to torch.testing.assert_close, but counts the percentage of mismatches instead of
    raising an exception.
    """
    # find the positions where the actual and expected values are close
    close_mask = torch.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    # compute the percentage of close positions
    num_el = actual.numel()
    close_cnt = close_mask.detach().sum().cpu().item()
    # compute the max diff
    max_diff = (actual - expected).detach().abs().max().cpu().item()
    return close_cnt, num_el, max_diff


def report_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> str:
    """reports the percentage of mismatches like torch.testing.assert_close"""
    close_cnt, num_el, max_diff = check_close(actual, expected, atol, rtol)
    # print the results
    result = f"{100 * close_cnt / num_el:.1f}% close at atol={atol} rtol={rtol}, max diff={max_diff}"
    return result


@functools.lru_cache(maxsize=None)
def get_cudnn_gqa_fwd(batch_size, seq_len, heads_q, heads_kv, dim, dtype):
    """As the replacement for PyTorch GQA function"""
    print(f"fwd graph: bs={batch_size}, seq_len={seq_len}, heads_q={heads_q}, heads_kv={heads_kv}, dim={dim}, dtype: {dtype}")
    attn_scale = float(dim) ** -0.5
    q_dim = (batch_size, heads_q, seq_len, dim)
    q_stride = (dim * seq_len * heads_q, dim, dim * heads_q, 1)
    kv_dim = (batch_size, heads_kv, seq_len, dim)
    kv_stride = (dim * seq_len * heads_kv, dim, dim * heads_kv, 1)
    with cudnn.Graph(
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        inputs=["q", "k", "v"],
        outputs=["out", "stats"],
        handle="auto",
    ) as graph:
        q_gpu = graph.tensor(name="q", dim=q_dim, stride=q_stride)
        k_gpu = graph.tensor(name="k", dim=kv_dim, stride=kv_stride)
        v_gpu = graph.tensor(name="v", dim=kv_dim, stride=kv_stride)
        out, stats = graph.sdpa(
            q=q_gpu,
            k=k_gpu,
            v=v_gpu,
            attn_scale=attn_scale,
            is_inference=False,
            use_causal_mask=True,
        )
        # set output, inv_var must be float32 tensor
        out.set_output(True).set_dim(q_dim).set_stride(q_stride).set_name("out")
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_name("stats")
    return graph


@functools.lru_cache(maxsize=None)
def get_cudnn_gqa_bwd(batch_size, seq_len, heads_q, heads_kv, dim, dtype):
    """As the replacement for PyTorch GQA function in backward pass"""
    print(f"bwd graph: bs={batch_size}, seq_len={seq_len}, heads_q={heads_q}, heads_kv={heads_kv}, dim={dim}, dtype: {dtype}")
    attn_scale = float(dim) ** -0.5
    q_dim = (batch_size, heads_q, seq_len, dim)
    q_stride = (dim * seq_len * heads_q, dim, dim * heads_q, 1)
    kv_dim = (batch_size, heads_kv, seq_len, dim)
    kv_stride = (dim * seq_len * heads_kv, dim, dim * heads_kv, 1)
    stats_dim = (batch_size, heads_q, seq_len, 1)
    stats_stride = (heads_q * seq_len, seq_len, 1, 1)
    dO_stride = (dim * seq_len * heads_q, dim * seq_len, dim, 1)
    with cudnn.Graph(
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        inputs=["q", "k", "v", "o", "dO", "stats"],
        outputs=["dQ", "dK", "dV"],
        handle="auto",
    ) as graph:
        q_gpu = graph.tensor(name="q", dim=q_dim, stride=q_stride)
        k_gpu = graph.tensor(name="k", dim=kv_dim, stride=kv_stride)
        v_gpu = graph.tensor(name="v", dim=kv_dim, stride=kv_stride)
        o_gpu = graph.tensor(name="o", dim=q_dim, stride=q_stride)
        dO_gpu = graph.tensor(name="dO", dim=q_dim, stride=dO_stride)
        stats_gpu = graph.tensor(
            name="stats",
            dim=stats_dim,
            stride=stats_stride,
            data_type=cudnn.data_type.FLOAT,
        )
        dQ, dK, dV = graph.sdpa_backward(
            q=q_gpu,
            k=k_gpu,
            v=v_gpu,
            o=o_gpu,
            dO=dO_gpu,
            stats=stats_gpu,
            attn_scale=attn_scale,
            use_causal_mask=True,
        )
        # set output, inv_var must be float32 tensor
        dQ.set_output(True).set_dim(q_dim).set_stride(q_stride).set_name("dQ")
        dK.set_output(True).set_dim(kv_dim).set_stride(kv_stride).set_name("dK")
        dV.set_output(True).set_dim(kv_dim).set_stride(kv_stride).set_name("dV")
    return graph


class CudnnGQA(Function):
    @staticmethod
    def forward(q, k, v):
        """GQA function: o = softmax(qk^T/sqrt(d)) @ v

        q, k, v are 4D tensors of shape (batch_size, num_heads, seq_length, head_dim), with different head dimensions
        """
        bq, hq, sq, dq = q.shape
        bk, hk, sk, dk = k.shape
        bv, hv, sv, dv = v.shape
        assert hq % hk == 0, "H_q must be a multiple of H_kv (GQA/MQA constraint)"
        assert hv == hk, "H_v must be equal to H_kv"
        assert dq == dk == dv, "All head dimensions must be equal"
        assert bq == bk == bv, "All batch sizes must be equal"
        assert q.dtype == k.dtype == v.dtype, "All input tensors must have the same dtype"
        assert q.stride() == (
            sq * dq * hq,
            dq,
            dq * hq,
            1,
        ), "q.stride() != (s*d*h, d, d*h, 1)"
        assert k.stride() == (
            sk * dk * hk,
            dk,
            dk * hk,
            1,
        ), "k.stride() != (s*d*h, d, d*h, 1)"
        assert v.stride() == (
            sv * dv * hv,
            dv,
            dv * hv,
            1,
        ), "v.stride() != (s*d*h, d, d*h, 1)"
        print(f"CudnnGQA: q: {q.shape}, {q.stride()}; k: {k.shape}, {k.stride()}; v: {v.shape}, {v.stride()}, dtype: {q.dtype}")
        graph = get_cudnn_gqa_fwd(bq, sq, hq, hk, dq, q.dtype)
        o, stats = graph(q, k, v)
        return o, stats

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v = inputs
        o, stats = output
        ctx.save_for_backward(q, k, v, o, stats)

    @staticmethod
    def backward(ctx, dO, _dstats):
        q, k, v, o, stats = ctx.saved_tensors
        print(
            "bwd:",
            "q:",
            q.shape,
            "k:",
            k.shape,
            "v:",
            v.shape,
            "o:",
            o.shape,
            "stats:",
            stats.shape,
            "dO:",
            dO.shape,
        )
        print(
            "bwd:",
            "q:",
            q.stride(),
            "k:",
            k.stride(),
            "v:",
            v.stride(),
            "o:",
            o.stride(),
            "stats:",
            stats.stride(),
            "dO:",
            dO.stride(),
        )
        dq = dk = dv = None
        # collect shapes and check for consistency
        bq, hq, sq, dq = q.shape
        bk, hk, sk, dk = k.shape
        bv, hv, sv, dv = v.shape
        bo, ho, so, do = o.shape
        bs, hs, ss, ds = stats.shape
        bdO, hdO, sdO, ddO = dO.shape
        assert bq == bk == bv == bo == bs == bdO, "All batch sizes must be equal"
        assert sq == so == ss == sdO, "Output and stats sequence lengths must match query sequence length"
        assert hk == hv, "H_kv must be equal to H_kv"
        assert sk == sv, "K and V sequence lengths must match"
        assert hq == ho == hs == hdO, "Output and stats num heads must match num query heads"
        assert ds == 1, "stats.shape[-1] != 1"
        assert dq == dk == dv == do == ddO, "All head dimensions must be equal"
        assert q.stride() == (
            sq * dq * hq,
            dq,
            dq * hq,
            1,
        ), "q.stride() != (s*d*h, d, d*h, 1)"
        assert k.stride() == (
            sk * dk * hk,
            dk,
            dk * hk,
            1,
        ), "k.stride() != (s*d*h, d, d*h, 1)"
        assert v.stride() == (
            sv * dv * hv,
            dv,
            dv * hv,
            1,
        ), "v.stride() != (s*d*h, d, d*h, 1)"
        assert o.stride() == (
            so * do * ho,
            do,
            do * ho,
            1,
        ), "o.stride() != (s*d*h, d, d*h, 1)"
        assert stats.stride() == (ss * hs, ss, 1, 1), "stats.stride() != (s*h, s, 1, 1)"
        assert dO.stride() == (
            sdO * ddO * hdO,
            sdO * ddO,
            ddO,
            1,
        ), "dO.stride() != (s*d*h, s*d, d, 1)"
        assert q.dtype == k.dtype == v.dtype == o.dtype == dO.dtype, "All input/output tensors must have the same dtype"
        # cuDNN compute all grads at once
        graph = get_cudnn_gqa_bwd(bq, sq, hq, hk, dq, q.dtype)
        dQ, dK, dV = graph(q, k, v, o, dO, stats)
        return dQ, dK, dV


def compare_gqa(dtype):
    device = torch.device("cuda")
    torch.set_default_device(device)
    B, S, Hq, Hkv, D = 3, 10, 32, 8, 128
    native_query = torch.randn(B, S, Hq, D, requires_grad=True, dtype=dtype)
    native_key = torch.randn(B, S, Hkv, D, requires_grad=True, dtype=dtype)
    native_value = torch.randn(B, S, Hkv, D, requires_grad=True, dtype=dtype)
    cudnn_query = native_query.detach().clone().requires_grad_(True)
    cudnn_key = native_key.detach().clone().requires_grad_(True)
    cudnn_value = native_value.detach().clone().requires_grad_(True)

    native_query = native_query.transpose(1, 2)  # in shape (B, Hq, S, D)
    native_key = native_key.transpose(1, 2)  # in shape (B, Hkv, S, D)
    native_value = native_value.transpose(1, 2)  # in shape (B, Hkv, S, D)
    cudnn_query = cudnn_query.transpose(1, 2)  # in shape (B, Hq, S, D)
    cudnn_key = cudnn_key.transpose(1, 2)  # in shape (B, Hkv, S, D)
    cudnn_value = cudnn_value.transpose(1, 2)  # in shape (B, Hkv, S, D)

    native_output = torch.nn.functional.scaled_dot_product_attention(
        native_query,
        native_key,
        native_value,
        dropout_p=0.0,
        is_causal=True,
        enable_gqa=True,
    )
    cudnn_output, _ = CudnnGQA.apply(cudnn_query, cudnn_key, cudnn_value)

    tensors = {
        "native_output": native_output.detach().clone(),
        "cudnn_output": cudnn_output.detach().clone(),
    }

    # preserve gradients
    native_query.retain_grad()
    native_key.retain_grad()
    native_value.retain_grad()
    cudnn_query.retain_grad()
    cudnn_key.retain_grad()
    cudnn_value.retain_grad()

    # backward pass
    o_gt = torch.randn_like(native_output)
    native_loss = F.mse_loss(native_output, o_gt)
    cudnn_loss = F.mse_loss(cudnn_output, o_gt)
    native_loss.backward()
    cudnn_loss.backward()

    tensors.update(
        {
            "native_query_grad": native_query.grad.detach().clone(),
            "cudnn_query_grad": cudnn_query.grad.detach().clone(),
            "native_key_grad": native_key.grad.detach().clone(),
            "cudnn_key_grad": cudnn_key.grad.detach().clone(),
            "native_value_grad": native_value.grad.detach().clone(),
            "cudnn_value_grad": cudnn_value.grad.detach().clone(),
        }
    )
    return tensors


def test_gqa_bfloat16():
    """Test GQA with bfloat16 data type for PyTest"""
    tensors = compare_gqa(torch.bfloat16)
    assert tensors["native_output"].shape == tensors["cudnn_output"].shape
    assert tensors["native_output"].stride() == tensors["cudnn_output"].stride()
    assert tensors["native_output"].dtype == tensors["cudnn_output"].dtype
    assert tensors["native_query_grad"].shape == tensors["cudnn_query_grad"].shape
    assert tensors["native_query_grad"].stride() == tensors["cudnn_query_grad"].stride()
    assert tensors["native_query_grad"].dtype == tensors["cudnn_query_grad"].dtype
    assert tensors["native_key_grad"].shape == tensors["cudnn_key_grad"].shape
    assert tensors["native_key_grad"].stride() == tensors["cudnn_key_grad"].stride()
    assert tensors["native_key_grad"].dtype == tensors["cudnn_key_grad"].dtype
    assert tensors["native_value_grad"].shape == tensors["cudnn_value_grad"].shape
    assert tensors["native_value_grad"].stride() == tensors["cudnn_value_grad"].stride()
    assert tensors["native_value_grad"].dtype == tensors["cudnn_value_grad"].dtype
    close_cnt, num_el, max_diff = check_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_query_grad"], tensors["cudnn_query_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_key_grad"], tensors["cudnn_key_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_value_grad"], tensors["cudnn_value_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3


def test_gqa_float16():
    """Test GQA with float16 data type for PyTest"""
    tensors = compare_gqa(torch.float16)
    assert tensors["native_output"].shape == tensors["cudnn_output"].shape
    assert tensors["native_output"].stride() == tensors["cudnn_output"].stride()
    assert tensors["native_output"].dtype == tensors["cudnn_output"].dtype
    assert tensors["native_query_grad"].shape == tensors["cudnn_query_grad"].shape
    assert tensors["native_query_grad"].stride() == tensors["cudnn_query_grad"].stride()
    assert tensors["native_query_grad"].dtype == tensors["cudnn_query_grad"].dtype
    assert tensors["native_key_grad"].shape == tensors["cudnn_key_grad"].shape
    assert tensors["native_key_grad"].stride() == tensors["cudnn_key_grad"].stride()
    assert tensors["native_key_grad"].dtype == tensors["cudnn_key_grad"].dtype
    assert tensors["native_value_grad"].shape == tensors["cudnn_value_grad"].shape
    assert tensors["native_value_grad"].stride() == tensors["cudnn_value_grad"].stride()
    assert tensors["native_value_grad"].dtype == tensors["cudnn_value_grad"].dtype
    close_cnt, num_el, max_diff = check_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_query_grad"], tensors["cudnn_query_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_key_grad"], tensors["cudnn_key_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3
    close_cnt, num_el, max_diff = check_close(tensors["native_value_grad"], tensors["cudnn_value_grad"], atol=2e-3, rtol=2e-3)
    assert close_cnt / num_el > 0.95
    assert max_diff < 5e-3


def test_gqa_float32():
    """Test GQA with float32 data type for PyTest"""
    import pytest

    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        _ = compare_gqa(torch.float32)


if __name__ == "__main__":
    print(f"cuDNN version: {cudnn.backend_version()}")
    print()
    print("=" * 10, "bfloat16", "=" * 10)
    tensors = compare_gqa(torch.bfloat16)
    print()
    print("native output:", tensors["native_output"].shape, tensors["native_output"].stride())
    print("cudnn output:", tensors["cudnn_output"].shape, tensors["cudnn_output"].stride())
    print("output closeness:", report_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3))
    print()
    print(
        "native query grad:",
        tensors["native_query_grad"].shape,
        tensors["native_query_grad"].stride(),
        tensors["native_query_grad"].dtype,
    )
    print(
        "cudnn query grad:",
        tensors["cudnn_query_grad"].shape,
        tensors["cudnn_query_grad"].stride(),
        tensors["cudnn_query_grad"].dtype,
    )
    print(
        "query grad closeness:",
        report_close(tensors["native_query_grad"], tensors["cudnn_query_grad"], atol=2e-3, rtol=2e-3),
    )
    print("key grad closeness:", report_close(tensors["native_key_grad"], tensors["cudnn_key_grad"], atol=2e-3, rtol=2e-3))
    print(
        "value grad closeness:",
        report_close(tensors["native_value_grad"], tensors["cudnn_value_grad"], atol=2e-3, rtol=2e-3),
    )
    print()

    print("=" * 10, "float16", "=" * 10)
    tensors = compare_gqa(torch.float16)
    print()
    print("native output:", tensors["native_output"].shape, tensors["native_output"].stride())
    print("cudnn output:", tensors["cudnn_output"].shape, tensors["cudnn_output"].stride())
    print("output closeness:", report_close(tensors["native_output"], tensors["cudnn_output"], atol=2e-3, rtol=2e-3))
    print()
    print(
        "native query grad:",
        tensors["native_query_grad"].shape,
        tensors["native_query_grad"].stride(),
        tensors["native_query_grad"].dtype,
    )
    print(
        "cudnn query grad:",
        tensors["cudnn_query_grad"].shape,
        tensors["cudnn_query_grad"].stride(),
        tensors["cudnn_query_grad"].dtype,
    )
    print(
        "query grad closeness:",
        report_close(tensors["native_query_grad"], tensors["cudnn_query_grad"], atol=2e-3, rtol=2e-3),
    )
    print("key grad closeness:", report_close(tensors["native_key_grad"], tensors["cudnn_key_grad"], atol=2e-3, rtol=2e-3))
    print(
        "value grad closeness:",
        report_close(tensors["native_value_grad"], tensors["cudnn_value_grad"], atol=2e-3, rtol=2e-3),
    )
    print()

    print("=" * 10, "float32 (expected to fail)", "=" * 10)
    tensors = compare_gqa(torch.float32)
