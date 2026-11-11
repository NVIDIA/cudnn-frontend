# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Kernel-package dispatch and local-autograd contracts (CPU mock kernels)."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import torch

from cudnn.simplified_sparse_attention.cutedsl import backend as cute_backend

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    ("query_heads", "topk", "module_name", "entry_name"),
    [
        pytest.param(16, 512, "attention_teacher_forward_hq16", "run", id="h16-k512"),
        pytest.param(16, 1024, "sparse_attention_large", "run", id="h16-k1024"),
        pytest.param(16, 2048, "sparse_attention_large", "run", id="h16-k2048"),
        pytest.param(32, 512, "attention_teacher_forward_hq32", "run", id="h32-k512"),
        pytest.param(32, 1024, "sparse_attention_large_h32_lookahead", "run", id="h32-k1024"),
        pytest.param(32, 2048, "sparse_attention_large", "run", id="h32-k2048"),
        pytest.param(96, 512, "attention_teacher_forward_hq96", "run", id="h96-k512"),
        pytest.param(96, 1024, "sparse_attention_h96_big", "run_group", id="h96-k1024"),
        pytest.param(96, 2048, "sparse_attention_h96_big", "run_group", id="h96-k2048"),
    ],
)
def test_attention_forward_route_matrix(monkeypatch, query_heads, topk, module_name, entry_name):
    imported = []
    module = SimpleNamespace(run=object(), run_group=object())

    monkeypatch.setattr(cute_backend, "_KERNEL_MODULES", {})
    monkeypatch.setattr(cute_backend, "_validate_production_environment", lambda: None)

    def fake_import(name):
        imported.append(name)
        return module

    monkeypatch.setattr(cute_backend.importlib, "import_module", fake_import)

    entry = cute_backend._attention_forward_entry(query_heads=query_heads, topk=topk)

    assert imported == [f"{cute_backend.__package__}.kernels.{module_name}"]
    assert entry is getattr(module, entry_name)


@pytest.mark.parametrize(("query_heads", "topk"), [(64, 512), (16, 256), (96, 4096)])
def test_attention_forward_route_matrix_fails_closed(monkeypatch, query_heads, topk):
    monkeypatch.setattr(cute_backend, "_KERNEL_MODULES", {})
    monkeypatch.setattr(cute_backend, "_validate_production_environment", lambda: None)

    def fail_import(name):
        pytest.fail(f"unsupported route unexpectedly imported {name}")

    monkeypatch.setattr(cute_backend.importlib, "import_module", fail_import)

    with pytest.raises(NotImplementedError, match=f"Hq={query_heads}, TopK={topk}"):
        cute_backend._kernel_module("attention_teacher_forward", query_heads=query_heads, topk=topk)


@pytest.mark.parametrize(
    ("topk", "expected_lengths"),
    [pytest.param(1024, [1024, 600], id="k1024"), pytest.param(2048, [1500, 600], id="k2048")],
)
@pytest.mark.parametrize("defer_kv", [False, True])
def test_sparse_forward_allocates_and_clamps_with_runtime_topk(monkeypatch, topk, expected_lengths, defer_kv):
    query = torch.empty((2, 16, 256), dtype=torch.bfloat16)
    key = torch.empty((1600, 1, 256), dtype=torch.bfloat16)
    value = torch.empty_like(key)
    q_idx = torch.empty((2, 128), dtype=torch.bfloat16)
    k_idx = torch.empty((1600, 128), dtype=torch.bfloat16)
    layout = cute_backend.RowMetadata(
        key_starts=torch.tensor([0, 0], dtype=torch.int32),
        key_ends=torch.tensor([1500, 600], dtype=torch.int32),
        query_valid=torch.tensor([True, True]),
        loss_denominator=torch.tensor([2.0]),
        triangular_scores=False,
        mask_invalid_rows=False,
    )
    seen = {}
    execution_order = []
    ready_event = object() if defer_kv else None

    def wait_event(event):
        assert event is ready_event
        execution_order.append("wait")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: SimpleNamespace(wait_event=wait_event))

    def fake_indexer_run(q_idx_arg, k_idx_arg, indices, predict, starts, ends, **kwargs):
        execution_order.append("indexer")
        seen["indexer"] = {
            "q_idx": q_idx_arg,
            "k_idx": k_idx_arg,
            "indices_shape": indices.shape,
            "predict_shape": predict.shape,
            "starts": starts,
            "ends": ends,
            "kwargs": kwargs,
        }
        indices.zero_()
        predict.zero_()

    def fake_attention_run(
        query_arg,
        key_arg,
        value_arg,
        indices,
        lengths,
        output,
        lse,
        teacher,
        softmax_max,
        softmax_inv_sum,
    ):
        execution_order.append("attention")
        seen["attention"] = (
            query_arg,
            key_arg,
            value_arg,
            indices.shape,
            lengths.clone(),
            teacher.shape,
        )
        output.zero_()
        lse.zero_()
        teacher.zero_()
        softmax_max.zero_()
        softmax_inv_sum.fill_(1.0)

    monkeypatch.setattr(cute_backend, "_validate_inputs", lambda *_args: None)
    monkeypatch.setattr(cute_backend, "_nvtx_range", lambda _name: nullcontext())
    monkeypatch.setattr(
        cute_backend,
        "_kernel_module",
        lambda name, **_kwargs: (SimpleNamespace(run=fake_indexer_run) if name == "indexer_forward_topk" else pytest.fail(f"unexpected kernel module {name}")),
    )

    runtime_topk = topk

    def fake_attention_entry(*, query_heads, topk):
        assert query_heads == 16
        assert topk == runtime_topk
        return fake_attention_run

    monkeypatch.setattr(cute_backend, "_attention_forward_entry", fake_attention_entry)

    result = cute_backend._run_sparse_forward(query, key, value, q_idx, k_idx, layout, topk, kv_ready_event=ready_event)

    expected_order = ["indexer", "wait", "attention"] if defer_kv else ["indexer", "attention"]
    assert execution_order == expected_order

    expected = torch.tensor(expected_lengths, dtype=torch.int32)
    assert result[5].shape == (2, topk)
    assert result[2].shape == (2, topk)
    torch.testing.assert_close(result[6], expected)
    torch.testing.assert_close(result[7], expected)
    assert seen["indexer"]["q_idx"] is q_idx
    assert seen["indexer"]["k_idx"] is k_idx
    assert seen["indexer"]["indices_shape"] == torch.Size((2, topk))
    assert seen["indexer"]["predict_shape"] == torch.Size((2, topk))
    assert seen["indexer"]["starts"] is result[9].key_starts
    assert seen["indexer"]["ends"] is result[9].key_ends
    assert seen["attention"][3] == torch.Size((2, topk))
    torch.testing.assert_close(seen["attention"][4], expected)
    assert seen["attention"][5] == torch.Size((2, topk))


@pytest.mark.parametrize("loss_coeff", [None, 0, 0.03, 0.1, 0.2])
@pytest.mark.parametrize("denominator", [1.0, 2.0])
@pytest.mark.parametrize("defer_kv", [False, True])
def test_loss_coefficient_scales_only_indexer_objective_and_gradients(monkeypatch, loss_coeff, denominator, defer_kv):
    topk = 1024
    query = torch.zeros((2, 16, 256), dtype=torch.bfloat16, requires_grad=True)
    key = torch.zeros((3, 1, 256), dtype=torch.bfloat16, requires_grad=True)
    value = torch.zeros_like(key, requires_grad=True)
    q_idx = torch.zeros((2, 128), dtype=torch.bfloat16, requires_grad=True)
    k_idx = torch.zeros((3, 128), dtype=torch.bfloat16, requires_grad=True)
    layout = cute_backend.RowMetadata(
        key_starts=torch.zeros(2, dtype=torch.int32),
        key_ends=torch.tensor([2, 3], dtype=torch.int32),
        query_valid=torch.ones(2, dtype=torch.bool),
        loss_denominator=torch.tensor([denominator]),
        triangular_scores=False,
        mask_invalid_rows=False,
    )
    ready_event = object() if defer_kv else None

    def fake_sparse_forward(
        query_arg,
        key_arg,
        value_arg,
        q_idx_arg,
        k_idx_arg,
        layout_arg,
        topk_arg,
        kv_ready_event=None,
    ):
        assert kv_ready_event is ready_event
        assert q_idx_arg is q_idx
        assert k_idx_arg is k_idx
        assert topk_arg == topk
        return (
            torch.zeros_like(query_arg),
            torch.zeros((2, 16), dtype=torch.float32),
            torch.zeros((2, topk), dtype=torch.float32),
            torch.zeros((2, 16), dtype=torch.float32),
            torch.ones((2, 16), dtype=torch.float32),
            torch.zeros((2, topk), dtype=torch.int32),
            torch.tensor([2, 3], dtype=torch.int32),
            torch.tensor([2, 3], dtype=torch.int32),
            torch.zeros((2, topk), dtype=torch.float32),
            layout_arg,
        )

    def fake_aux_run(
        _q_idx,
        _k_idx,
        _indices,
        _lengths,
        _predict,
        _teacher,
        indexer_loss,
        unit_dq_idx,
        unit_dk_idx,
        loss_denominator,
        *,
        loss_coeff,
        **_kwargs,
    ):
        assert isinstance(loss_coeff, float)
        scale = loss_coeff / float(loss_denominator.item())
        indexer_loss.fill_(2.0 * scale)
        unit_dq_idx.fill_(3.0 * scale)
        unit_dk_idx.fill_(5.0 * scale)

    def fake_attention_backward(*args):
        dq, dk, dv = args[-3:]
        dq.fill_(2.0)
        dk.fill_(3.0)
        dv.fill_(4.0)

    monkeypatch.setattr(cute_backend, "_run_sparse_forward", fake_sparse_forward)
    monkeypatch.setattr(cute_backend, "_nvtx_range", lambda _name: nullcontext())
    monkeypatch.setattr(
        cute_backend,
        "_kernel_module",
        lambda name, **_kwargs: (
            SimpleNamespace(run=fake_aux_run)
            if name == "indexer_aux_grad_precompute"
            else SimpleNamespace(run=fake_attention_backward) if name == "sparse_attention_backward" else pytest.fail(f"unexpected kernel module {name}")
        ),
    )

    kwargs = {} if loss_coeff is None else {"loss_coeff": loss_coeff}
    if defer_kv:
        kwargs["kv_ready_event"] = ready_event
    output, indexer_loss = cute_backend.sparse_attention(query, key, value, q_idx, k_idx, layout, topk, **kwargs)
    (output.sum() + 7.0 * indexer_loss).backward()

    scale = (0.1 if loss_coeff is None else loss_coeff) / denominator
    torch.testing.assert_close(indexer_loss, torch.tensor(2.0 * scale))
    torch.testing.assert_close(output, torch.zeros_like(query))
    torch.testing.assert_close(query.grad, torch.full_like(query, 2.0))
    torch.testing.assert_close(key.grad, torch.full_like(key, 3.0))
    torch.testing.assert_close(value.grad, torch.full_like(value, 4.0))
    # Saved gradients are BF16, so their rounding precedes upstream scaling.
    torch.testing.assert_close(q_idx.grad, torch.full_like(q_idx, 3.0 * scale) * 7.0)
    torch.testing.assert_close(k_idx.grad, torch.full_like(k_idx, 5.0 * scale) * 7.0)


def _debug_check_forward(monkeypatch, q_idx, topk_rows):
    """Run _run_sparse_forward with SSA_DEBUG_CHECKS on and fake kernels; return the kernels run."""
    query = torch.zeros((3, 16, 256), dtype=torch.bfloat16)
    key = torch.zeros((8, 1, 256), dtype=torch.bfloat16)
    layout = cute_backend.RowMetadata(
        key_starts=torch.tensor([0, 0, 4], dtype=torch.int32),
        key_ends=torch.tensor([4, 4, 8], dtype=torch.int32),
        query_valid=torch.tensor([True, True, False]),
        loss_denominator=torch.tensor([2.0]),
        triangular_scores=False,
        mask_invalid_rows=True,
    )
    ran = []

    def fake_indexer_run(_q_idx, _k_idx, indices, predict, *_args, **_kwargs):
        ran.append("indexer")
        for row, value in enumerate(topk_rows):
            indices[row] = value
        predict.zero_()

    def fake_attention_run(*args):
        ran.append("attention")
        for tensor in args[5:]:
            tensor.zero_()

    monkeypatch.setattr(cute_backend, "_DEBUG_CHECKS", True)
    monkeypatch.setattr(cute_backend, "_validate_inputs", lambda *_args: None)
    monkeypatch.setattr(cute_backend, "_nvtx_range", lambda _name: nullcontext())
    monkeypatch.setattr(cute_backend, "_kernel_module", lambda _name, **_kwargs: SimpleNamespace(run=fake_indexer_run))
    monkeypatch.setattr(cute_backend, "_attention_forward_entry", lambda **_kwargs: fake_attention_run)
    cute_backend._run_sparse_forward(
        query,
        key,
        torch.zeros_like(key),
        q_idx,
        torch.ones((8, 128), dtype=torch.bfloat16),
        layout,
        512,
    )
    return ran


def test_debug_checks_accept_valid_call(monkeypatch):
    # Row 1 pads with end-1; the padding row is masked to the zero sentinel.
    ran = _debug_check_forward(monkeypatch, torch.ones((3, 128), dtype=torch.bfloat16), (2, 3, -5))
    assert ran == ["indexer", "attention"]


def test_debug_checks_reject_nonfinite_indexer_inputs_before_topk(monkeypatch):
    q_idx = torch.ones((3, 128), dtype=torch.bfloat16)
    q_idx[1, 7] = torch.nan
    with pytest.raises(RuntimeError, match=r"1/3 q_idx rows contain NaN/Inf \(first row 1\)"):
        _debug_check_forward(monkeypatch, q_idx, (2, 3, 0))


def test_debug_checks_reject_out_of_bounds_ids_before_gather(monkeypatch):
    with pytest.raises(RuntimeError, match=r"outside the key bounds in 1 rows \(first row 1"):
        _debug_check_forward(monkeypatch, torch.ones((3, 128), dtype=torch.bfloat16), (2, 7, 0))
