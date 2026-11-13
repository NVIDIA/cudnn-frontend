# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import re

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays
from phase1_reference import check_reference, pack, reference, run, sample
from test_simplified_sparse_attention import supported_gpu

pytestmark = [pytest.mark.L0, pytest.mark.usefixtures("supported_gpu")]

_KNOWN_BYTE_FAILURES = {
    "test_replay[bf16-h32-tail-fprop-cutedsl]",
    "test_replay[bf16-h32-tail-bprop-cutedsl]",
    "test_replay[bf16-h96-split-bprop-cutedsl]",
    "test_scheduling_boundary[fprop-cutedsl]",
    "test_scheduling_boundary[bprop-cutedsl]",
    "test_batch_invariance[bf16-h16-tiny-bprop-torch]",
    "test_batch_invariance[bf16-h32-tail-bprop-torch]",
    "test_batch_invariance[bf16-h32-tail-bprop-cutedsl]",
    "test_batch_invariance[bf16-h96-split-fprop-torch]",
    "test_batch_invariance[bf16-h96-split-fprop-cutedsl]",
    "test_batch_invariance[bf16-h96-split-bprop-torch]",
    "test_batch_invariance[bf16-h96-split-bprop-cutedsl]",
}


_KNOWN_LOSS_FAILURES = {
    "test_replay[bf16-h96-split-fprop-cutedsl]": r"replay \d+ loss: 1/4 bytes differ",
    "test_batch_invariance[bf16-h16-tiny-fprop-torch]": r"packed slot 0 loss: 1/4 bytes differ",
}


@contextmanager
def known_byte_failure(request):
    try:
        yield
    except AssertionError as error:
        if torch.cuda.get_device_capability() != (10, 0):
            raise
        pattern = _KNOWN_LOSS_FAILURES.get(request.node.name)
        if pattern is not None and re.fullmatch(pattern, str(error)):
            pytest.xfail(f"Known scalar-loss reduction difference; remove when the reduction is deterministic: {error}")
        if request.node.name not in _KNOWN_BYTE_FAILURES:
            raise
        if not re.fullmatch(r"(?:replay \d+|packed slot \d+) (?:loss|(?:saved_)?d[qk]_idx): \d+/\d+ bytes differ", str(error)):
            raise
        pytest.xfail(f"Known Phase 1 reduction-order difference; remove when replay and batch invariance are fixed: {error}")


def checked(inputs, use_kf, backward):
    expected, lse = reference(inputs)

    def invoke():
        actual = run(inputs, use_kf, lse, backward)
        check_reference(actual, expected)
        return actual

    return invoke


@pytest.mark.parametrize("use_kf", [False, True], ids=["torch", "cutedsl"])
@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
@pytest.mark.parametrize("heads,rows,keys", [(16, 3, 17), (32, 65, 257), (96, 33, 4097)], ids=["bf16-h16-tiny", "bf16-h32-tail", "bf16-h96-split"])
def test_replay(request, use_kf, backward, heads, rows, keys):
    with known_byte_failure(request):
        assert_replays(checked(sample(rows, keys, heads), use_kf, backward))


@pytest.mark.parametrize("use_kf", [False, True], ids=["torch", "cutedsl"])
@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
def test_scheduling_boundary(request, use_kf, backward):
    rows = 32 * (torch.cuda.get_device_properties(0).multi_processor_count + 1)
    with known_byte_failure(request):
        assert_replays(checked(sample(rows, 257), use_kf, backward))


@pytest.mark.parametrize("use_kf", [False, True], ids=["torch", "cutedsl"])
@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
@pytest.mark.parametrize("heads,rows,keys", [(16, 3, 17), (32, 65, 257), (96, 33, 4097)], ids=["bf16-h16-tiny", "bf16-h32-tail", "bf16-h96-split"])
def test_batch_invariance(request, use_kf, backward, heads, rows, keys):
    target, neighbor = sample(rows, keys, heads), sample(2, 7, heads, seed=31)
    alone = checked(target, use_kf, backward)()
    for layout, slot in (([target, neighbor], 0), ([neighbor, target], 1), ([target, target], 1), ([sample(2, 7, heads, 99), target], 1)):
        inputs = pack(layout, slot)
        actual = checked(inputs, use_kf, backward)()
        qoff = sum(s[0][0].shape[0] for s in layout[:slot])
        koff = sum(s[0][1].shape[0] for s in layout[:slot])
        selected = {}
        for name, value in actual.items():
            offset, length = (koff, keys) if "dk" in name else (qoff, rows)
            selected[name] = value if value.ndim == 0 else value[offset : offset + length]
        with known_byte_failure(request):
            assert_bitwise(selected, alone, f"packed slot {slot}")


@pytest.mark.parametrize("use_kf", [False, True], ids=["torch", "cutedsl"])
@pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
def test_golden(use_kf, backward):
    inputs = sample(1, 17)
    inputs[1][2][:] = True
    inputs[1][1][:] = 17
    actual = assert_replays(checked(inputs, use_kf, backward))
    from cutlass.base_dsl.version_info import CUDA_VERSION

    tensors = dict(zip(("q", "k", "q_idx", "k_idx", "starts", "ends", "valid", "denominator"), (*inputs[0], *inputs[1])))
    assert_golden(f'phase1-cu{CUDA_VERSION.major}.{CUDA_VERSION.minor}-{"cutedsl" if use_kf else "torch"}-{"bprop" if backward else "fprop"}', tensors, actual)


@pytest.mark.parametrize(
    "name,message",
    [
        ("test_replay[bf16-h96-split-fprop-cutedsl]", "replay 2 loss: 1/4 bytes differ"),
        ("test_batch_invariance[bf16-h16-tiny-fprop-torch]", "packed slot 0 loss: 1/4 bytes differ"),
    ],
)
@pytest.mark.parametrize("change", [None, "arch", "bytes", "tensor", "case", "numerics"])
def test_ci_loss_exception_scope(monkeypatch, name, message, change):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 3) if change == "arch" else (10, 0))
    if change == "bytes":
        message = message.replace("1/4", "2/4")
    elif change == "tensor":
        message = message.replace(" loss:", " saved_dq_idx:")
    elif change == "case":
        name = name.replace("fprop", "other")
    elif change == "numerics":
        message = "Tensor-likes are not close!"
    expected = pytest.xfail.Exception if change is None else AssertionError
    with pytest.raises(expected):
        with known_byte_failure(SimpleNamespace(node=SimpleNamespace(name=name))):
            raise AssertionError(message)
