# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the model-facing causal-conv facade."""

import ast
import asyncio
import functools
import importlib.util
import inspect
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.L0


@functools.lru_cache(maxsize=1)
def _ops_module():
    source = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "ops" / "causal_conv1d.py"
    spec = importlib.util.spec_from_file_location("_causal_conv1d_semantic_contract", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signature_matches_consumed_semantics_and_hides_backend_lifecycle():
    signature = inspect.signature(_ops_module().causal_conv1d)

    assert tuple(signature.parameters) == (
        "x",
        "weight",
        "bias",
        "activation",
        "seq_idx",
        "cu_seqlens",
        "initial_states",
        "return_final_states",
        "final_states_out",
    )
    for name in (
        "seq_idx",
        "cu_seqlens",
        "initial_states",
        "return_final_states",
        "final_states_out",
    ):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    for forbidden in (
        "wrapper",
        "sm100",
        "schedule",
        "workspace",
        "output",
        "current_stream",
        "plan",
        "compile",
    ):
        assert forbidden not in signature.parameters


def test_top_level_cudnn_does_not_export_experimental_backend_types():
    source = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "__init__.py"
    tree = ast.parse(source.read_text())
    lazy_imports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_LAZY_OPTIONAL_IMPORTS" for target in node.targets)
    )
    for name in (
        "CausalConv1dBulkAutogradPrototype",
        "CausalConv1dBulkBwdPrototype",
        "CausalConv1dBulkFwdSm100",
        "compile_causal_conv1d_bulk_bwd_prototype",
        "causal_conv1d_bulk_fwd_wrapper_sm100",
    ):
        assert name not in lazy_imports


def test_ops_surface_adds_only_the_semantic_causal_conv_name():
    source = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "ops" / "__init__.py"
    tree = ast.parse(source.read_text())
    lazy_exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_LAZY_EXPORTS" for target in node.targets)
    )

    assert lazy_exports["causal_conv1d"] == (".causal_conv1d", "causal_conv1d")
    assert "_get_causal_conv1d_last_route" not in lazy_exports
    assert not any(token in name.lower() for name in lazy_exports for token in ("bulk", "wrapper", "sm100", "prototype"))


def test_primary_docs_do_not_promote_legacy_or_backend_api_names():
    docs = Path(__file__).resolve().parents[4] / "docs" / "fe-oss-apis" / "causal_conv1d.md"
    text = docs.read_text()

    assert "`cudnn.ops.causal_conv1d`" in text
    for forbidden in (
        "causal_conv1d_nwh",
        "causal_conv1d_bulk",
        "CausalConv1dBulk",
        "wrapper_sm100",
    ):
        assert forbidden not in text


def test_dense_native_route_is_bdt_public_and_btd_zero_copy(monkeypatch):
    module = _ops_module()
    backing = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    x = backing.transpose(1, 2)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    observed = {}

    monkeypatch.setattr(module, "_can_route_causal_conv1d_bulk", lambda *args: True)

    def run(x_btd, native_weight, bias, cu_seqlens):
        observed.update(x_btd=x_btd, weight=native_weight, bias=bias)
        assert cu_seqlens is None
        return torch.empty_like(x_btd)

    monkeypatch.setattr(module, "_run_causal_conv1d_bulk_backend", run)
    output = module.causal_conv1d(x, weight, activation="silu")

    assert output.shape == x.shape
    assert output.stride() == x.stride()
    assert observed["x_btd"].is_contiguous()
    assert observed["x_btd"].data_ptr() == x.data_ptr()
    assert observed["weight"] is weight
    assert module._get_causal_conv1d_last_route() == "native-inference"


def test_generic_backend_layout_adapter_preserves_channel_last_input_format():
    module = _ops_module()
    backing = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    public_x = backing.transpose(1, 2)
    contiguous_output = torch.randn_like(public_x.contiguous())

    output = module._match_causal_conv1d_output_layout(contiguous_output, public_x)

    assert output.shape == public_x.shape
    assert output.stride() == public_x.stride()
    torch.testing.assert_close(output, contiguous_output)


def test_mathematical_w_minus_one_state_returns_an_ordinary_tuple(monkeypatch):
    module = _ops_module()
    backing = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    x = backing.transpose(1, 2)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    initial_states = torch.randn(2, 8, 3, dtype=torch.bfloat16)
    final_states_out = torch.empty_like(initial_states)
    semantic_output = torch.empty_like(x)
    observed = {}

    def run(
        native_x,
        native_weight,
        bias,
        activation,
        seq_idx,
        cu_seqlens,
        native_initial,
        return_final,
        native_final,
    ):
        observed.update(
            x=native_x,
            weight=native_weight,
            bias=bias,
            activation=activation,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            initial_states=native_initial,
            return_final_states=return_final,
            final_states_out=native_final,
        )
        return semantic_output, native_final

    monkeypatch.setattr(module, "_run_causal_conv1d_sequence_backend", run)
    result = module.causal_conv1d(
        x,
        weight,
        initial_states=initial_states,
        return_final_states=True,
        final_states_out=final_states_out,
        activation="silu",
    )

    assert type(result) is tuple
    output, final_states = result
    assert output.shape == x.shape
    assert final_states is final_states_out
    assert observed["x"] is x
    assert observed["initial_states"] is initial_states


def test_full_width_state_adapter_uses_private_storage_and_preserves_gradient():
    module = _ops_module()
    initial_states = torch.randn(2, 8, 3, dtype=torch.bfloat16, requires_grad=True)

    full_width = module._to_causal_conv1d_full_width_state(initial_states)
    assert full_width.shape == (2, 8, 4)
    torch.testing.assert_close(full_width[..., 0], torch.zeros_like(full_width[..., 0]))
    torch.testing.assert_close(full_width[..., 1:], initial_states)

    public_final = module._from_causal_conv1d_full_width_state(full_width * 2, None)
    assert public_final.shape == initial_states.shape
    assert public_final.untyped_storage().nbytes() == public_final.numel() * public_final.element_size()
    assert public_final.data_ptr() != full_width.data_ptr()
    torch.testing.assert_close(public_final, initial_states * 2)

    public_final.float().sum().backward()
    torch.testing.assert_close(initial_states.grad, torch.full_like(initial_states, 2))


@pytest.mark.parametrize("packed", [False, True])
def test_sequence_backend_adapts_dense_and_packed_state_with_final_states_out_gradient(monkeypatch, packed):
    module = _ops_module()
    batch = 1 if packed else 2
    backing = torch.randn(batch, 5, 8, dtype=torch.bfloat16)
    x = backing.transpose(1, 2)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32) if packed else None
    state_rows = 2
    initial_states = torch.randn(state_rows, 8, 3, dtype=torch.bfloat16, requires_grad=True)
    final_states_out = torch.empty(state_rows, 3, 8, dtype=torch.bfloat16).transpose(1, 2)
    observed = {}

    monkeypatch.setattr(module, "_can_route_causal_conv1d_bulk", lambda *args: True)

    def run(
        x_btd,
        native_weight,
        bias,
        native_cu_seqlens,
        *,
        initial_state,
        output_final_state,
    ):
        observed.update(
            x_btd=x_btd,
            weight=native_weight,
            cu_seqlens=native_cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
        return x_btd.clone(), initial_state * 2

    monkeypatch.setattr(module, "_run_causal_conv1d_bulk_backend", run)
    output, final_states = module.causal_conv1d(
        x,
        weight,
        activation="silu",
        cu_seqlens=cu_seqlens,
        initial_states=initial_states,
        return_final_states=True,
        final_states_out=final_states_out,
    )

    assert output.shape == x.shape
    assert output.stride() == x.stride()
    assert final_states is final_states_out
    assert observed["cu_seqlens"] is cu_seqlens
    assert observed["output_final_state"] is True
    assert observed["initial_state"].shape == (state_rows, 8, 4)
    torch.testing.assert_close(
        observed["initial_state"][..., 0],
        torch.zeros_like(observed["initial_state"][..., 0]),
    )
    torch.testing.assert_close(observed["initial_state"][..., 1:], initial_states)
    torch.testing.assert_close(final_states, initial_states * 2)

    final_states.float().sum().backward()
    torch.testing.assert_close(initial_states.grad, torch.full_like(initial_states, 2))
    assert module._get_causal_conv1d_last_route() == "native-autograd"


def test_bulk_backend_training_dispatch_includes_internal_state(monkeypatch):
    module = _ops_module()
    x_btd = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    initial_state = torch.randn(1, 8, 4, dtype=torch.bfloat16, requires_grad=True)
    observed = {}

    class Backend:
        def __call__(self, x, native_weight, cu_seqlens, **kwargs):
            observed.update(x=x, weight=native_weight, cu_seqlens=cu_seqlens, **kwargs)
            return {
                "output_tensor": x.clone(),
                "final_state_tensor": kwargs["initial_state"].clone(),
            }

    def get_backend(x, native_weight, bias, cu_seqlens, native_initial, output_final_state, deterministic):
        observed.update(cache_initial=native_initial, cache_final=output_final_state, deterministic=deterministic)
        return Backend()

    monkeypatch.setattr(module, "_get_causal_conv1d_training_backend", get_backend)
    output, final_state = module._run_causal_conv1d_bulk_backend(
        x_btd,
        weight,
        None,
        None,
        initial_state=initial_state,
        output_final_state=True,
    )

    assert output.shape == x_btd.shape
    assert final_state.shape == initial_state.shape
    assert observed["cache_initial"] is initial_state
    assert observed["initial_state"] is initial_state
    assert observed["cache_final"] is True
    assert observed["output_final_state"] is True


def test_bulk_backend_inference_dispatch_includes_internal_state(monkeypatch):
    module = _ops_module()
    x_btd = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    initial_state = torch.randn(1, 8, 4, dtype=torch.bfloat16)
    observed = {}

    def forward(x, native_weight, **kwargs):
        observed.update(x=x, weight=native_weight, **kwargs)
        return {
            "output_tensor": x.clone(),
            "final_state_tensor": kwargs["initial_state_tensor"].clone(),
        }

    fake_api = SimpleNamespace(causal_conv1d_bulk_fwd_wrapper_sm100=forward)
    monkeypatch.setitem(sys.modules, "cudnn.causal_conv1d_bulk_sm100.api", fake_api)
    with torch.no_grad():
        output, final_state = module._run_causal_conv1d_bulk_backend(
            x_btd,
            weight,
            None,
            None,
            initial_state=initial_state,
            output_final_state=True,
        )

    assert output.shape == x_btd.shape
    assert final_state.shape == initial_state.shape
    assert observed["initial_state_tensor"] is initial_state
    assert observed["output_final_state"] is True


def test_full_width_backend_cache_is_not_accepted_as_public_state():
    module = _ops_module()
    backing = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    x = backing.transpose(1, 2)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    full_width_cache = torch.randn(1, 8, 4, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"initial_states must have shape \(1, 8, 3\)"):
        module.causal_conv1d(x, weight, initial_states=full_width_cache, activation="silu")


@pytest.mark.parametrize("with_seq_idx", [False, True])
def test_dense_and_seq_idx_states_use_batch_rows(with_seq_idx):
    module = _ops_module()
    backing = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    x = backing.transpose(1, 2)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    state = torch.randn(2, 8, 3, dtype=torch.bfloat16)
    seq_idx = torch.zeros(2, 5, dtype=torch.int32) if with_seq_idx else None

    result = module._validate_causal_conv1d_sequence_contract(
        x,
        weight,
        None,
        seq_idx,
        None,
        state,
        False,
        None,
    )

    assert result is None
    with pytest.raises(ValueError, match=r"initial_states must have shape \(2, 8, 3\)"):
        module._validate_causal_conv1d_sequence_contract(
            x,
            weight,
            None,
            seq_idx,
            None,
            torch.randn(1, 8, 3, dtype=torch.bfloat16),
            False,
            None,
        )


def test_packed_states_use_number_of_sequences_not_storage_batch():
    module = _ops_module()
    backing = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    x = backing.transpose(1, 2)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    initial_states = torch.randn(2, 8, 3, dtype=torch.bfloat16)
    final_states_out = torch.empty_like(initial_states)

    result = module._validate_causal_conv1d_sequence_contract(
        x,
        weight,
        None,
        None,
        cu_seqlens,
        initial_states,
        True,
        final_states_out,
    )

    assert result is None
    with pytest.raises(ValueError, match=r"initial_states must have shape \(2, 8, 3\)"):
        module._validate_causal_conv1d_sequence_contract(
            x,
            weight,
            None,
            None,
            cu_seqlens,
            torch.randn(1, 8, 3, dtype=torch.bfloat16),
            True,
            final_states_out,
        )


def test_exact_shape_training_backend_is_private_and_reused(monkeypatch):
    module = _ops_module()
    module._CAUSAL_CONV1D_TRAINING_CACHE.clear()
    x_btd = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    created = []
    backend = object()

    monkeypatch.setattr(module, "_causal_conv1d_training_key", lambda *args: ("same-shape",))
    monkeypatch.setattr(
        module,
        "_compile_causal_conv1d_training_backend",
        lambda *args: created.append(args) or backend,
    )

    assert module._get_causal_conv1d_training_backend(x_btd, weight, None, None) is backend
    assert module._get_causal_conv1d_training_backend(x_btd, weight, None, None) is backend
    assert len(created) == 1


def test_training_cache_keys_every_routed_plan_field_but_not_packed_values(monkeypatch):
    module = _ops_module()
    properties = SimpleNamespace(major=10, minor=0, multi_processor_count=148)
    monkeypatch.setattr(module.torch.cuda, "get_device_properties", lambda device: properties)

    x_btd = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    bias = torch.randn(8, dtype=torch.bfloat16)
    packed_a = torch.tensor([0, 2, 5], dtype=torch.int32)
    packed_b = torch.tensor([0, 1, 5], dtype=torch.int32)
    packed_more_sequences = torch.tensor([0, 1, 3, 5], dtype=torch.int32)
    initial_state = torch.randn(1, 8, 4, dtype=torch.bfloat16)

    dense_key = module._causal_conv1d_training_key(x_btd, weight, None, None)
    biased_key = module._causal_conv1d_training_key(x_btd, weight, bias, None)
    packed_a_key = module._causal_conv1d_training_key(x_btd, weight, None, packed_a)
    packed_b_key = module._causal_conv1d_training_key(x_btd, weight, None, packed_b)
    packed_more_sequences_key = module._causal_conv1d_training_key(x_btd, weight, None, packed_more_sequences)
    initial_state_key = module._causal_conv1d_training_key(x_btd, weight, None, None, initial_state, False)
    final_state_key = module._causal_conv1d_training_key(x_btd, weight, None, None, initial_state, True)
    strided_weight = torch.empty(4, 8, dtype=torch.bfloat16).transpose(0, 1)
    strided_weight_key = module._causal_conv1d_training_key(x_btd, strided_weight, None, None)

    assert dense_key != biased_key
    assert dense_key != packed_a_key
    assert packed_a_key == packed_b_key
    assert packed_a_key != packed_more_sequences_key
    assert dense_key != initial_state_key
    assert initial_state_key != final_state_key
    assert dense_key != strided_weight_key


def test_unmapped_state_and_seq_idx_modes_decline_explicitly():
    module = _ops_module()
    backing = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    x = backing.transpose(1, 2)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    seq_idx = torch.zeros(1, 5, dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="does not yet implement"):
        module.causal_conv1d(x, weight, seq_idx=seq_idx, activation="silu")


def test_cu_seqlens_packed_route_returns_a_tensor_and_is_mutually_exclusive(
    monkeypatch,
):
    module = _ops_module()
    backing = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    x = backing.transpose(1, 2)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    output_btd = torch.empty_like(backing)
    observed = {}

    monkeypatch.setattr(module, "_can_route_causal_conv1d_bulk", lambda *args: True)

    def run(x_btd, native_weight, bias, native_cu_seqlens):
        observed.update(x_btd=x_btd, weight=native_weight, cu_seqlens=native_cu_seqlens)
        return output_btd

    monkeypatch.setattr(module, "_run_causal_conv1d_bulk_backend", run)
    output = module.causal_conv1d(x, weight, None, "silu", cu_seqlens=cu_seqlens)

    assert type(output) is torch.Tensor
    assert output.shape == x.shape
    assert observed["x_btd"].data_ptr() == x.data_ptr()
    assert observed["cu_seqlens"] is cu_seqlens
    assert module._get_causal_conv1d_last_route() == "native-inference"

    with pytest.raises(ValueError, match="mutually exclusive"):
        module.causal_conv1d(
            x,
            weight,
            None,
            "silu",
            seq_idx=torch.zeros(1, 5, dtype=torch.int32),
            cu_seqlens=cu_seqlens,
        )


def test_generic_route_is_recorded_only_after_success(monkeypatch):
    module = _ops_module()
    x = torch.randn(2, 8, 5)
    weight = torch.randn(8, 3)
    expected = torch.empty_like(x)

    monkeypatch.setattr(module, "_can_route_causal_conv1d_bulk", lambda *args: False)
    monkeypatch.setattr(
        module.torch.ops.cudnn,
        "causal_conv1d_fwd_primitive",
        lambda *args: expected,
    )

    assert module.causal_conv1d(x, weight) is expected
    assert module._get_causal_conv1d_last_route() == "generic-cudnn"

    with pytest.raises(NotImplementedError, match="activation must"):
        module.causal_conv1d(x, weight, activation="relu")
    assert module._get_causal_conv1d_last_route() is None


def test_mixed_fp32_weight_declines_explicitly_when_native_layout_is_unavailable(
    monkeypatch,
):
    module = _ops_module()
    x = torch.randn(1, 8, 5, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.float32)

    monkeypatch.setattr(module, "_can_route_causal_conv1d_bulk", lambda *args: False)
    with pytest.raises(NotImplementedError, match="BF16 activation with FP32 weight"):
        module.causal_conv1d(x, weight, activation="silu")
    assert module._get_causal_conv1d_last_route() is None


def test_last_route_diagnostic_stays_out_of_torch_compile_graph(monkeypatch):
    module = _ops_module()
    x = torch.randn(2, 8, 5)
    weight = torch.randn(8, 3)

    monkeypatch.setattr(
        module.torch.ops.cudnn,
        "causal_conv1d_fwd_primitive",
        lambda public_x, *args: public_x.clone(),
    )
    module._record_causal_conv1d_route("native-inference")

    compiled = torch.compile(module.causal_conv1d, backend="eager", fullgraph=True)
    output = compiled(x, weight)

    torch.testing.assert_close(output, x)
    # Compiled calls neither graph-break on ContextVar nor replace the most
    # recent eager route with a trace-time observation.
    assert module._get_causal_conv1d_last_route() == "native-inference"


def test_last_route_diagnostic_is_thread_and_async_context_local():
    module = _ops_module()
    module._record_causal_conv1d_route("generic-cudnn")
    barrier = threading.Barrier(2)

    def thread_worker(route):
        assert module._get_causal_conv1d_last_route() is None
        module._record_causal_conv1d_route(route)
        barrier.wait()
        return module._get_causal_conv1d_last_route()

    routes = ("native-inference", "native-autograd")
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert tuple(pool.map(thread_worker, routes)) == routes
    assert module._get_causal_conv1d_last_route() == "generic-cudnn"

    async def async_worker(route):
        module._record_causal_conv1d_route(route)
        await asyncio.sleep(0)
        return module._get_causal_conv1d_last_route()

    async def run_async_workers():
        return await asyncio.gather(*(async_worker(route) for route in routes))

    assert tuple(asyncio.run(run_async_workers())) == routes
    assert module._get_causal_conv1d_last_route() == "generic-cudnn"


@pytest.mark.parametrize("alias", ["x", "weight", "initial_states"])
def test_final_states_out_must_not_share_memory_with_inputs(alias):
    module = _ops_module()
    backing = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    x = backing.transpose(1, 2)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    initial_states = torch.randn(2, 8, 3, dtype=torch.bfloat16)
    if alias == "x":
        final_states_out = x[..., :3]
    elif alias == "weight":
        final_states_out = weight.as_strided((2, 8, 3), (0, 4, 1))
    else:
        final_states_out = initial_states

    with pytest.raises(ValueError, match=f"final_states_out must not share memory with {alias}"):
        module.causal_conv1d(
            x,
            weight,
            activation="silu",
            initial_states=initial_states,
            return_final_states=True,
            final_states_out=final_states_out,
        )


def test_shared_memory_check_uses_addressed_byte_spans_not_storage_identity():
    module = _ops_module()
    storage = torch.empty(2 * 2 * 8 * 3, dtype=torch.bfloat16)
    first = storage[: 2 * 8 * 3].view(2, 8, 3)
    second = storage[2 * 8 * 3 :].view(2, 8, 3)
    shifted = storage[1 : 2 * 8 * 3 + 1].view(2, 8, 3)

    assert not module._tensors_share_memory(first, second)
    assert module._tensors_share_memory(first, shifted)
    assert module._tensors_share_memory(first, first.transpose(1, 2))


def test_training_dispatch_keys_and_forwards_torch_deterministic_mode(monkeypatch):
    module = _ops_module()
    properties = SimpleNamespace(major=10, minor=0, multi_processor_count=148)
    monkeypatch.setattr(module.torch.cuda, "get_device_properties", lambda device: properties)
    x_btd = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16, requires_grad=True)

    default_key = module._causal_conv1d_training_key(x_btd, weight, None, None)
    deterministic_key = module._causal_conv1d_training_key(x_btd, weight, None, None, None, False, True)
    assert default_key != deterministic_key

    observed = []

    def get_backend(x, native_weight, bias, cu_seqlens, initial_state, output_final_state, deterministic):
        observed.append(deterministic)
        return lambda *args, **kwargs: x.clone()

    monkeypatch.setattr(module, "_get_causal_conv1d_training_backend", get_backend)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        module._run_causal_conv1d_bulk_backend(x_btd, weight, None, None)
        torch.use_deterministic_algorithms(True, warn_only=True)
        module._run_causal_conv1d_bulk_backend(x_btd, weight, None, None)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
    assert observed == [False, True]
