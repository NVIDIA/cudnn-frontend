# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The randomized fp16 harness must reproduce every input from its data seed."""

import pytest
import torch

from sdpa.fp16 import TensorUid, allocate_tensors
from sdpa.random_config import ExecConfig


@pytest.mark.L0
def test_block_mask_uses_the_per_test_data_generator():
    cfg = ExecConfig(
        batches=1,
        h_q=2,
        h_k=2,
        h_v=2,
        s_q=256,
        s_kv=1024,
        d_qk=16,
        d_v=16,
        data_type=torch.bfloat16,
        rng_geom_seed=17,
        rng_data_seed=42,
        is_block_mask=True,
        is_infer=True,
    )
    cfg.fill_derived_fields()
    samples = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for global_seed in (1, 2):
            torch.cuda.manual_seed(global_seed)
            _, tensors, _, _ = allocate_tensors(cfg, torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed))
            samples.append({uid: tensors[uid] for uid in (TensorUid.q, TensorUid.k, TensorUid.v, TensorUid.block_mask)})
    for uid in samples[0]:
        assert torch.equal(samples[0][uid], samples[1][uid]), f"{uid.name} depends on global RNG instead of the per-test data seed"
