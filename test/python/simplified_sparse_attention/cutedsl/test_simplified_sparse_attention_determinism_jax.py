# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bytewise JAX replay, packed-batch changes, and versioned goldens."""

import pytest
import torch

jax = pytest.importorskip("jax")

from determinism_utils import assert_bitwise, assert_golden, assert_replays
from ssa_reference import check_reference, pack, sample
from test_simplified_sparse_attention import supported_gpu
from test_simplified_sparse_attention_determinism import check_bytes
from test_simplified_sparse_attention_jax import execute

pytestmark = [pytest.mark.L0, pytest.mark.usefixtures("supported_gpu")]


def checked_run(inputs, backward, topk):
    actual = execute(inputs, topk=topk, backward=backward)
    check_reference(inputs, actual, backward, topk)
    return actual


@pytest.mark.parametrize("backward", [False, True], ids=["bf16-fprop", "bf16-bprop"])
@pytest.mark.parametrize("heads,topk", [(16, 512), (32, 1024), (96, 2048)], ids=["h16-k512", "h32-k1024", "h96-k2048"])
def test_replay(backward, heads, topk):
    inputs = pack([sample(65, heads), sample(3, heads, 9), sample(17, heads, 31)])
    check_bytes(lambda: assert_replays(lambda: checked_run(inputs, backward, topk)), backward)


@pytest.mark.parametrize("backward", [False, True], ids=["bf16-fprop", "bf16-bprop"])
def test_scheduling_boundary(backward):
    rows = torch.cuda.get_device_properties(0).multi_processor_count * 2 + 1
    inputs = pack([sample(rows)])
    check_bytes(lambda: assert_replays(lambda: checked_run(inputs, backward, 512)), backward)


@pytest.mark.parametrize("backward", [False, True], ids=["bf16-fprop", "bf16-bprop"])
@pytest.mark.parametrize("heads,topk", [(16, 512), (32, 1024), (96, 2048)], ids=["h16-k512", "h32-k1024", "h96-k2048"])
def test_batch_invariance(backward, heads, topk):
    target, neighbor = sample(65, heads), sample(17, heads, 31)
    alone = checked_run(pack([target]), backward, topk)
    for layout, slot in (([target, neighbor], 0), ([neighbor, target], 1), ([target, target], 1), ([sample(17, heads, 99), target], 1)):
        inputs = pack(layout)
        offset = sum(s[0][0].shape[0] for s in layout[:slot])
        stop = offset + 65
        inputs[1].query_valid[:offset] = False
        inputs[1].query_valid[stop:] = False
        inputs[2][:offset] = 0
        inputs[2][stop:] = 0
        actual = checked_run(inputs, backward, topk)
        actual = {name: x if x.ndim == 0 else x[offset:stop] for name, x in actual.items()}
        check_bytes(
            lambda: assert_bitwise(actual, alone, f"packed slot {slot}"),
            backward,
            packed_loss=(heads, topk) in ((32, 1024), (96, 2048)) or (heads, topk, backward) == (16, 512, False),
        )


@pytest.mark.parametrize("backward", [False, True], ids=["bf16-fprop", "bf16-bprop"])
@pytest.mark.parametrize("heads,topk", [(16, 512), (32, 1024), (96, 2048)], ids=["h16-k512", "h32-k1024", "h96-k2048"])
def test_golden(backward, heads, topk):
    inputs = pack([sample(17, heads)])
    actual = check_bytes(lambda: assert_replays(lambda: checked_run(inputs, backward, topk)), backward)
    if backward and torch.cuda.get_device_capability() == (10, 0):
        pytest.xfail("Backward golden awaits stable atomic gradient replay; record after replay is fixed")
    tensors = dict(zip(("q", "k", "v", "q_idx", "k_idx"), inputs[0]))
    tensors.update(
        upstream=inputs[2], starts=inputs[1].key_starts, ends=inputs[1].key_ends, valid=inputs[1].query_valid, denominator=inputs[1].loss_denominator
    )
    from cutlass.base_dsl.version_info import CUDA_VERSION

    name = f"ssa-jax{jax.__version__}-cu{CUDA_VERSION.major}.{CUDA_VERSION.minor}-h{heads}-k{topk}-{'bprop' if backward else 'fprop'}"
    assert_golden(name, tensors, actual)
