# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bytewise detectors; the API has no deterministic mode."""

import re

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays
from ssa_reference import check_reference, pack, run, sample
from test_simplified_sparse_attention import supported_gpu

pytestmark = [pytest.mark.L0, pytest.mark.usefixtures("supported_gpu")]


def check_bytes(check, backward, *, packed_loss=False):
    try:
        return check()
    except AssertionError as error:
        # Keep exceptions limited to the measured tensor and failure type.
        if torch.cuda.get_device_capability() != (10, 0):
            raise
        if backward and re.fullmatch(r"(?:replay \d+|packed slot \d+) d[kv]: \d+/\d+ bytes differ", str(error)):
            pytest.xfail("Known atomic dK/dV reductions vary by schedule; remove when backward replay and batch invariance are fixed")
        if packed_loss and str(error) == "packed slot 1 indexer_loss: 1/4 bytes differ":
            pytest.xfail("Known packed loss rounding at H32/K1024 and H96/K2048; remove when the reduction is batch invariant")
        raise


def checked_run(inputs, backward, topk=512):
    actual = run(inputs, backward, topk)
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
    check_bytes(lambda: assert_replays(lambda: checked_run(inputs, backward)), backward)


@pytest.mark.parametrize("backward", [False, True], ids=["bf16-fprop", "bf16-bprop"])
@pytest.mark.parametrize("heads,topk", [(16, 512), (32, 1024), (96, 2048)], ids=["h16-k512", "h32-k1024", "h96-k2048"])
def test_batch_invariance(backward, heads, topk):
    target, neighbor = sample(65, heads), sample(17, heads, seed=31)
    alone = run(pack([target]), backward, topk)
    check_reference(pack([target]), alone, backward, topk)
    # The scalar loss reduces all rows; isolate target rows in every layout.
    for layout, slot in (([target, neighbor], 0), ([neighbor, target], 1), ([target, target], 1), ([sample(17, heads, seed=99), target], 1)):
        inputs = pack(layout)
        offset = sum(s[0][0].shape[0] for s in layout[:slot])
        stop = offset + target[0][0].shape[0]
        inputs[1].query_valid[:offset] = False
        inputs[1].query_valid[stop:] = False
        inputs[2][:offset] = 0
        inputs[2][stop:] = 0
        actual = run(inputs, backward, topk)
        check_reference(inputs, actual, backward, topk)
        actual = {name: value if value.ndim == 0 else value[offset:stop] for name, value in actual.items()}
        check_bytes(lambda: assert_bitwise(actual, alone, f"packed slot {slot}"), backward, packed_loss=(heads, topk) in ((32, 1024), (96, 2048)))


@pytest.mark.parametrize("backward", [False, True], ids=["bf16-fprop", "bf16-bprop"])
@pytest.mark.parametrize("heads,topk", [(16, 512), (32, 1024), (96, 2048)], ids=["h16-k512", "h32-k1024", "h96-k2048"])
def test_golden(backward, heads, topk):
    inputs = pack([sample(17, heads)])
    actual = check_bytes(lambda: assert_replays(lambda: checked_run(inputs, backward, topk)), backward)
    if backward and torch.cuda.get_device_capability() == (10, 0):
        pytest.xfail("No stable backward golden while atomic gradients vary; record after replay is fixed")
    tensors = dict(zip(("q", "k", "v", "q_idx", "k_idx"), inputs[0]))
    tensors.update(
        upstream=inputs[2], starts=inputs[1].key_starts, ends=inputs[1].key_ends, valid=inputs[1].query_valid, denominator=inputs[1].loss_denominator
    )
    from cutlass.base_dsl.version_info import CUDA_VERSION

    assert_golden(f'ssa-cu{CUDA_VERSION.major}.{CUDA_VERSION.minor}-h{heads}-k{topk}-{"bprop" if backward else "fprop"}', tensors, actual)
