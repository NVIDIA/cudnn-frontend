# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 gradient amax must exclude padded sequence rows (#2122)."""

import cudnn
import pytest
import torch

from sdpa.mxfp8 import exec_sdpa_mxfp8
from sdpa.random_config import ExecConfig


@pytest.mark.L0
@pytest.mark.parametrize("sequence", [2816, 2818], ids=["aligned", "tail"])
def test_mxfp8_bwd_amax_live_rows(sequence, request, cudnn_handle):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 SDPA requires Blackwell (SM100+)")

    cfg = ExecConfig(
        data_type=torch.float8_e4m3fn,
        output_type=torch.bfloat16,
        rng_geom_seed=1407202494215190570,
        rng_data_seed=217522697,
        is_infer=False,
        is_mxfp8=True,
        is_determin=True,
        is_paged=False,
        is_padding=False,
        is_ragged=False,
        is_cu_seq_len=False,
        is_bias=False,
        is_alibi=False,
        is_block_mask=False,
        is_dropout=False,
        batches=3,
        h_q=6,
        h_k=6,
        h_v=6,
        s_q=sequence,
        s_kv=sequence,
        d_qk=192,
        d_v=128,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT,
        rescale_threshold=0.0,
    )
    cfg.fill_derived_fields()
    print(cfg.to_repro_cmd(request.module.__file__), flush=True)
    exec_sdpa_mxfp8(cfg, request, cudnn_handle)
