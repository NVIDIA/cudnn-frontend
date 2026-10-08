# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for https://github.com/NVIDIA/cudnn-frontend/issues/1009.

The SM90 FP8 SDPA forward kernel hangs when a padding mask is combined with a
left diagonal band bound and some batch has seq_len_q well above seq_len_kv
(query rows whose whole window lies past that batch's keys). The per-batch
lengths are runtime data, so ``validate_sdpa_support_surface`` declines the
whole padding + left-bound combination on Hopper. The graphs here pin
``sm_version`` so the check runs on any GPU; only ``validate()`` is exercised.
"""

import cudnn
import pytest

pytestmark = pytest.mark.L0

_MATCH = "left diagonal band bound is not supported on Hopper"


def _validate_fp8_fwd(*, sm_version, use_padding_mask, left_bound, right_bound):
    b, h, s, d = 2, 4, 512, 64
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        sm_version=sm_version,
    )
    dim = (b, h, s, d)
    stride = (h * s * d, d, h * d, 1)
    q = graph.tensor(name="q", dim=dim, stride=stride, data_type=cudnn.data_type.FP8_E4M3)
    k = graph.tensor(name="k", dim=dim, stride=stride, data_type=cudnn.data_type.FP8_E4M3)
    v = graph.tensor(name="v", dim=dim, stride=stride, data_type=cudnn.data_type.FP8_E4M3)

    def scalar(name):
        return graph.tensor(name=name, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    seq_len_q = seq_len_kv = None
    if use_padding_mask:
        seq_len_q = graph.tensor(name="seq_len_q", dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        seq_len_kv = graph.tensor(name="seq_len_kv", dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)

    o, stats, _amax_s, amax_o = graph.sdpa_fp8(
        q=q,
        k=k,
        v=v,
        descale_q=scalar("descale_q"),
        descale_k=scalar("descale_k"),
        descale_v=scalar("descale_v"),
        descale_s=scalar("descale_s"),
        scale_s=scalar("scale_s"),
        scale_o=scalar("scale_o"),
        generate_stats=True,
        attn_scale=1.0 / (d**0.5),
        use_padding_mask=use_padding_mask,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        left_bound=left_bound,
        right_bound=right_bound,
    )
    o.set_output(True).set_dim(dim).set_stride(stride).set_data_type(cudnn.data_type.FP8_E4M3)
    stats.set_output(True).set_dim((b, h, s, 1)).set_stride((h * s, s, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    graph.validate()


@pytest.mark.parametrize("right_bound", [0, 128], ids=["causal_left", "band"])
def test_hopper_rejects_fp8_padding_with_left_bound(right_bound):
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match=_MATCH):
        _validate_fp8_fwd(sm_version=90, use_padding_mask=True, left_bound=128, right_bound=right_bound)


def _assert_not_our_rejection(fn):
    try:
        fn()
    except cudnn.cudnnGraphNotSupportedError as e:
        assert _MATCH not in str(e), f"wrongly rejected by the #1009 guard: {e}"


def test_hopper_accepts_fp8_padding_without_left_bound():
    _assert_not_our_rejection(lambda: _validate_fp8_fwd(sm_version=90, use_padding_mask=True, left_bound=None, right_bound=0))


def test_hopper_accepts_fp8_left_bound_without_padding():
    _assert_not_our_rejection(lambda: _validate_fp8_fwd(sm_version=90, use_padding_mask=False, left_bound=128, right_bound=0))


def test_blackwell_accepts_fp8_padding_with_left_bound():
    _assert_not_our_rejection(lambda: _validate_fp8_fwd(sm_version=100, use_padding_mask=True, left_bound=128, right_bound=0))
