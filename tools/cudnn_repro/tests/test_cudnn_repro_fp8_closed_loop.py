# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from .test_cudnn_repro_closed_loop import _assert_reproducer_json_matches_target


def _target_test_id(target):
    return target.split("::", 1)[-1]


def _targets():
    raw = os.environ.get("CUDNN_REPRO_FP8_TARGETS")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [
        "test/python/sdpa/graph/test_mhas_v2.py::test_sdpa_fp8_fwd_L0[test1]",
        "test/python/sdpa/graph/test_mhas_v2.py::test_sdpa_fp8_fwd_paged_L0[test1]",
        "test/python/sdpa/graph/test_mhas_v2.py::test_sdpa_fp8_bwd_L0[test1]",
    ]


@pytest.mark.parametrize("target", _targets(), ids=_target_test_id)
def test_fp8_reproducer_json_matches(tmp_path, target):
    _assert_reproducer_json_matches_target(tmp_path, target)


def test_ragged_fp8_cumulative_replay(tmp_path):
    cfg = dict(
        data_type="torch.float8_e4m3fn",
        output_type="torch.float8_e4m3fn",
        implementation="cudnn.attention_implementation.UNIFIED",
        rng_data_seed=1234,
        is_infer=True,
        fwd_stats=False,
        is_ragged=True,
        is_padding=True,
        is_cu_seq_len=True,
        batches=2,
        h_q=4,
        h_k=2,
        h_v=2,
        d_qk=64,
        d_v=64,
        s_q=64,
        s_kv=128,
        seq_len_q=[32, 64],
        seq_len_kv=[73, 96],
    )
    _assert_reproducer_json_matches_target(tmp_path, "test/python/sdpa/graph/test_mhas_v2.py::test_repro", "--repro", repr(cfg))
