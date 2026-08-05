# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudnn
import pytest
import torch

from sdpa.fp16 import TensorUid, exec_sdpa
from sdpa.random_config import ExecConfig

ZERO_SEQLEN_CASES = [
    ([0, 128, 0, 128], [128, 128, 128, 128]),
    ([128, 128, 128, 128], [0, 128, 0, 128]),
    ([0, 128, 0, 128], [0, 128, 0, 128]),
]
ZERO_SEQLEN_IDS = ["half_zero_q", "half_zero_kv", "half_zero_qkv"]


def make_edge_config(
    *,
    data_type=torch.float16,
    batches=2,
    s_q=128,
    s_kv=128,
    seq_len_q=(),
    seq_len_kv=(),
    is_ragged=False,
    is_infer=False,
    is_cu_seq_len=False,
    diag_align=cudnn.diagonal_alignment.TOP_LEFT,
    right_bound=None,
):
    cfg = ExecConfig(
        data_type=data_type,
        rng_data_seed=1234,
        rng_geom_seed=5678,
        is_alibi=False,
        is_infer=is_infer,
        is_paged=False,
        is_bias=False,
        is_block_mask=False,
        is_padding=len(seq_len_q) > 0,
        is_cu_seq_len=is_cu_seq_len,
        is_ragged=is_ragged,
        is_dropout=False,
        is_determin=False,
        batches=batches,
        d_qk=64,
        d_v=64,
        s_q=s_q,
        s_kv=s_kv,
        h_q=3,
        h_k=3,
        h_v=3,
        diag_align=diag_align,
        left_bound=None,
        right_bound=right_bound,
        seq_len_q=list(seq_len_q),
        seq_len_kv=list(seq_len_kv),
        implementation=cudnn.attention_implementation.AUTO,
    )
    cfg.fill_derived_fields()
    return cfg


def check_row_all_inf(tensors):
    stats = tensors[TensorUid.stats]
    assert torch.isneginf(stats[:, :, :64, :]).all()
    assert torch.isfinite(stats[:, :, 64:, :]).all()


def check_col_all_inf(tensors):
    stats = tensors[TensorUid.stats]
    dK = tensors[TensorUid.dK]
    dV = tensors[TensorUid.dV]
    assert torch.isfinite(stats).all()
    assert torch.count_nonzero(dK[:, :, 128:, :]) == 0
    assert torch.count_nonzero(dV[:, :, 128:, :]) == 0


def initialize_negative_qk(tensors, rng):
    q = tensors[TensorUid.q]
    k = tensors[TensorUid.k]
    q.uniform_(0.125, 0.25, generator=rng)
    k.uniform_(-0.03125, -0.015625, generator=rng)

    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * 0.125
    row_max = scores.max(dim=-1).values
    assert (row_max < 0).all()
    assert (row_max > -0.1).all()


def check_negative_results(tensors):
    for name, uid in (
        ("o", TensorUid.o),
        ("stats", TensorUid.stats),
        ("dQ", TensorUid.dQ),
        ("dK", TensorUid.dK),
        ("dV", TensorUid.dV),
    ):
        assert torch.isfinite(tensors[uid].float()).all(), f"{name} has NaN/Inf"
    assert torch.count_nonzero(tensors[TensorUid.dQ]) > 0


@pytest.mark.L0
@pytest.mark.skipif(
    cudnn.backend_version() < 92500,
    reason="zero sequence length SDPA requires cuDNN >= 9.25.0",
)
@pytest.mark.parametrize(
    ("seq_len_q", "seq_len_kv"),
    ZERO_SEQLEN_CASES,
    ids=ZERO_SEQLEN_IDS,
)
def test_thd_zero_seqlen(seq_len_q, seq_len_kv, request, cudnn_handle):
    cfg = make_edge_config(
        batches=4,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        is_ragged=True,
    )
    exec_sdpa(cfg, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.skipif(
    cudnn.backend_version() < 92500,
    reason="zero sequence length SDPA requires cuDNN >= 9.25.0",
)
@pytest.mark.parametrize(
    ("seq_len_q", "seq_len_kv"),
    ZERO_SEQLEN_CASES,
    ids=ZERO_SEQLEN_IDS,
)
def test_cu_seqlen_zero_seqlen(seq_len_q, seq_len_kv, request, cudnn_handle):
    cfg = make_edge_config(
        batches=4,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        is_ragged=True,
        is_infer=True,
        is_cu_seq_len=True,
    )
    exec_sdpa(cfg, request, cudnn_handle)


@pytest.mark.L0
def test_sdpa_row_all_inf(request, cudnn_handle):
    cfg = make_edge_config(
        seq_len_q=[128, 128],
        seq_len_kv=[64, 64],
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT,
        right_bound=0,
    )
    exec_sdpa(cfg, request, cudnn_handle, tensor_checker=check_row_all_inf)


@pytest.mark.L0
def test_sdpa_col_all_inf(request, cudnn_handle):
    cfg = make_edge_config(s_kv=768, right_bound=0)
    exec_sdpa(cfg, request, cudnn_handle, tensor_checker=check_col_all_inf)


@pytest.mark.L0
@pytest.mark.parametrize(
    "data_type", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_sdpa_slightly_negative_row_max(data_type, request, cudnn_handle):
    cfg = make_edge_config(data_type=data_type)
    exec_sdpa(
        cfg,
        request,
        cudnn_handle,
        tensor_initializer=initialize_negative_qk,
        tensor_checker=check_negative_results,
    )
