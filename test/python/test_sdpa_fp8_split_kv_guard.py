"""
Build-only tests for the cuDNN 9.24 FP8/MXFP8 SDPA split-KV guard.

cuDNN 9.24 can select a split-KV prefill kernel for FP8/MXFP8 forward graphs
whose combine step encodes the result as FP16 regardless of the requested
output type, corrupting BF16 outputs. The frontend rejects the exposed
combination (BF16 output, s_q > 1, s_kv >= 4096, no diagonal band bounds,
non-paged/ragged/dropout) on cuDNN 9.24; the backend fix ships in 9.25.0.

These tests only build graphs (no execution), so they need no quantization
helpers and run on any Blackwell machine.
"""

import math

import pytest
import torch

import cudnn


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_mxfp8_graph(s_q, s_kv, use_causal_mask=False):
    b, h, d, block = 1, 3, 128, 32
    d_scale_pad = ceil_div(ceil_div(d, block), 4) * 4
    s_q_pad = ceil_div(s_q, 128) * 128
    s_kv_pad = ceil_div(s_kv, 128) * 128
    s_kv_scale_pad = ceil_div(ceil_div(s_kv, block), 4) * 4
    d_pad = ceil_div(d, 128) * 128

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    def fp8_tensor(s):
        return graph.tensor(
            dim=(b, h, s, d),
            stride=(h * s * d, s * d, d, 1),
            data_type=cudnn.data_type.FP8_E4M3,
        )

    def sf_qk_tensor(s_pad):
        return graph.tensor(
            dim=(b, h, s_pad, d_scale_pad),
            stride=(h * s_pad * d_scale_pad, s_pad * d_scale_pad, d_scale_pad, 1),
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

    sf_v = graph.tensor(
        dim=(b, h, s_kv_scale_pad, d_pad),
        stride=(h * s_kv_scale_pad * d_pad, s_kv_scale_pad * d_pad, d_pad, 1),
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )

    o, stats, amax_o = graph.sdpa_mxfp8(
        q=fp8_tensor(s_q),
        k=fp8_tensor(s_kv),
        v=fp8_tensor(s_kv),
        descale_q=sf_qk_tensor(s_q_pad),
        descale_k=sf_qk_tensor(s_kv_pad),
        descale_v=sf_v,
        attn_scale=1.0 / math.sqrt(d),
        use_causal_mask=use_causal_mask,
        generate_stats=True,
    )
    o.set_output(True).set_dim((b, h, s_q, d)).set_stride(
        (h * s_q * d, s_q * d, d, 1)
    ).set_data_type(cudnn.data_type.BFLOAT16)
    stats.set_output(True).set_dim((b, h, s_q, 1)).set_stride(
        (h * s_q, s_q, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride(
        (1, 1, 1, 1)
    ).set_data_type(cudnn.data_type.FLOAT)
    return graph


def skip_unless_mxfp8_capable():
    if cudnn.backend_version() < 92100:
        pytest.skip("MXFP8 SDPA requires cuDNN 9.21.0 or higher")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("MXFP8 SDPA requires Blackwell")


def affected_backend():
    return 92400 <= cudnn.backend_version() < 92500


@pytest.mark.L0
def test_sdpa_mxfp8_bf16_long_kv_gated_on_cudnn_924(cudnn_handle):
    """The exposed combination (BF16 out, s_q > 1, long s_kv, no band bounds)
    must be rejected on cuDNN 9.24 and accepted elsewhere."""
    skip_unless_mxfp8_capable()
    graph = build_mxfp8_graph(s_q=1024, s_kv=8192)
    if affected_backend():
        with pytest.raises(cudnn.cudnnGraphNotSupportedError, match="split-KV"):
            graph.validate()
    else:
        graph.validate()


@pytest.mark.L0
def test_sdpa_mxfp8_bf16_short_kv_allowed_on_cudnn_924(cudnn_handle):
    """Short s_kv never selects the split-KV prefill kernel; the guard must not
    reject it."""
    skip_unless_mxfp8_capable()
    graph = build_mxfp8_graph(s_q=1024, s_kv=512)
    graph.validate()


@pytest.mark.L0
def test_sdpa_mxfp8_bf16_causal_long_kv_allowed_on_cudnn_924(cudnn_handle):
    """Diagonal-band (causal) graphs never select the split-KV prefill kernel;
    the guard must not reject them."""
    skip_unless_mxfp8_capable()
    graph = build_mxfp8_graph(s_q=1024, s_kv=8192, use_causal_mask=True)
    graph.validate()
