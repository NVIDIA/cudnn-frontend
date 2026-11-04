# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed query/output buffers attend dense paged KV pools in both harnesses."""

import random

import cudnn
import pytest
import torch

from sdpa import fp8
from sdpa.fp16 import exec_sdpa
from sdpa.graph.suites.common import build_config
from sdpa.graph.suites.generation.test_generation_f16 import PAGED as F16_PAGED
from sdpa.graph.suites.generation.test_generation_fp8 import PAGED as FP8_PAGED
from sdpa.graph.suites.registry import _model_post
from sdpa.random_config import ExecConfig


@pytest.mark.L0
@pytest.mark.parametrize("block_size", [0, 64], ids=["packed-kv", "paged-kv"])
@pytest.mark.parametrize("multiplier", [False, True])
@pytest.mark.parametrize("cu_seq_len", [False, True])
def test_fp8_ragged_graph_offsets(block_size, multiplier, cu_seq_len):
    graph = fp8.generate_graph_fwd(
        cudnn.data_type.FP8_E4M3,
        cudnn.data_type.HALF,
        3,
        4,
        2,
        2,
        128,
        256,
        64,
        64,
        0.125,
        block_size,
        is_ragged=True,
        is_cu_seq_len=cu_seq_len,
        with_ragged_offset_multiplier=multiplier,
    )
    tensors = graph._tensor_by_uid
    for uid in (fp8.GraphFwdUid.q, fp8.GraphFwdUid.o, fp8.GraphFwdUid.stats):
        assert tensors[uid].ragged_offset is not None
    for uid in (fp8.GraphFwdUid.k, fp8.GraphFwdUid.v):
        assert (tensors[uid].ragged_offset is not None) == (block_size == 0)
        assert tensors[uid].ragged_offset_multiplier == (64 if multiplier and block_size == 0 else 1)
    for uid in (fp8.GraphFwdUid.k_ragged_offset, fp8.GraphFwdUid.v_ragged_offset):
        assert (uid in tensors) == (block_size == 0)
    for uid in (fp8.GraphFwdUid.q_seq_len, fp8.GraphFwdUid.kv_seq_len):
        assert (uid in tensors) == (not cu_seq_len)


@pytest.mark.L0
@pytest.mark.parametrize("spec", [F16_PAGED, FP8_PAGED], ids=lambda spec: spec.name)
def test_paged_generation_draws_packed_queries(spec):
    configs = []
    for seed in spec.seeds():
        cfg, rng = build_config(spec, seed)
        spec.post(cfg, rng, None)
        configs.append(cfg)
    assert all(cfg.is_paged for cfg in configs)
    assert any(cfg.is_ragged for cfg in configs), "paged generation must exercise packed Q/O"
    assert any(not cfg.is_ragged for cfg in configs), "retain padded Q/O coverage"
    if spec.exec_kind == "fp8":
        assert all(cfg.d_qk <= 128 and cfg.d_v <= 128 for cfg in configs if cfg.is_ragged)


@pytest.mark.L0
def test_model_generation_can_page_packed_queries():
    cfg = ExecConfig(is_ragged=True)
    _model_post("generation")(cfg, random.Random(1), None)
    assert cfg.is_paged, "model paging must also cover packed Q/O"


@pytest.mark.L0
@pytest.mark.parametrize("dtype,dim", [(torch.float16, 64), (torch.bfloat16, 128), (torch.float8_e4m3fn, 64), (torch.float8_e5m2, 128)])
@pytest.mark.parametrize("layout", ["ragged", "cu_ragged", "padded"])
def test_paged_ragged_mixed_lengths(dtype, dim, layout, request, cudnn_handle):
    """Non-page-aligned KV lengths and unequal packed Q lengths include a one-token sequence.

    Both shared harnesses compare O and packed Stats with their independent
    references. Unsupported backends retain the shared harness's skip behavior.
    """
    is_fp8 = dtype.itemsize == 1
    if is_fp8 and (torch.cuda.get_device_capability()[0] < 9 or cudnn.backend_version() < 91400):
        pytest.skip("FP8 SDPA requires Hopper or newer and cuDNN 9.14+")
    if layout == "cu_ragged" and cudnn.backend_version() < (92500 if is_fp8 else 92400):
        pytest.skip("cumulative sequence lengths require cuDNN 9.25+ for FP8 or 9.24+ for f16")
    cfg = ExecConfig(
        data_type=dtype,
        output_type=torch.float16 if is_fp8 else None,
        rng_data_seed=966,
        rng_geom_seed=966,
        is_infer=True,
        is_paged=True,
        is_ragged=layout != "padded",
        is_cu_seq_len=layout == "cu_ragged",
        is_padding=True,
        is_alibi=False,
        is_bias=False,
        is_block_mask=False,
        is_dropout=False,
        is_determin=False,
        batches=3,
        h_q=4,
        h_k=2,
        h_v=2,
        s_q=128,
        s_kv=256,
        d_qk=dim,
        d_v=dim,
        block_size=64,
        seq_len_q=[65, 17, 1],
        seq_len_kv=[193, 33, 1],
        total_q=192 if layout != "padded" else None,
        fwd_stats=True,
        with_ragged_token_gap=False,
        with_ragged_head_gap=False,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        implementation=cudnn.attention_implementation.AUTO,
    )
    cfg.fill_derived_fields()
    if request.config.option.dryrun:
        pytest.skip("dryrun")
    original_stream = cudnn.get_stream(cudnn_handle)
    try:
        (fp8.exec_sdpa_fp8 if is_fp8 else exec_sdpa)(cfg, request, cudnn_handle)
    finally:
        cudnn.set_stream(handle=cudnn_handle, stream=original_stream)
