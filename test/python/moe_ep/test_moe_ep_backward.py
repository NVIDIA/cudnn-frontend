# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fixed-resource MoE EP backward and training-graph contracts."""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import os
import threading
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cudnn
import pytest
import torch
import torch.multiprocessing as mp

from cudnn.moe_ep import (
    MoeEp,
    MoeEpExecutionLane,
    MoeEpTrainingResources,
    MoeEpTrainingSlot,
    MoeEpTrainingWgradOperands,
)
from cudnn.moe_ep._validation import validate_training_weights
from cudnn.moe_ep._megamoe_backend.mxfp8._adapter import (
    _typed_k_major_view,
)
from cudnn.moe_ep._megamoe_backend._workspace import (
    BufferRegion,
    WorkspaceRequirements,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._training_resources import (
    Mxfp8TrainingResourceOwner,
    _build_training_abi_facts,
    _harmonize_symmetric_regions,
    _verify_training_abi_across_ranks,
    build_training_workspace_requirements,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._training_stage import (
    Mxfp8TrainingStager,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._fingerprint import (
    canonical_json_sha256,
)
from cudnn.moe_ep._megamoe_backend.mxfp8._training_weights import (
    Mxfp8TrainingWeightBindings,
)
from cudnn.moe_ep._tuning import MoeEpTuningConfig
from moe_ep.moe_ep_reference import (
    MoeEpReference,
)
from moe_ep.moe_ep_distributed_workers import (
    _distributed_backward_reference_worker,
    _distributed_subgroup_backward_reference_worker,
)
from moe_ep.moe_ep_test_support import (
    _assert_fixed_training_drop_overflow_result,
    _assert_fixed_training_matches_reference,
    _assert_training_graph_tails_are_reset,
    _assert_training_weight_sources_changed,
    _capture_fixed_training_batch,
    _copy_training_weight_sources_,
    _dense_wgrads_from_operands,
    _fixed_training_case,
    _fixed_training_drop_overflow_case,
    _fixed_training_drop_overflow_reference,
    _fixed_training_reference,
    _fixed_training_weights,
    _grad_output,
    _prefill_training_graph_sentinels,
    _require_distributed_sm107,
    _run_fixed_training_batch,
    _sm107_device,
    _training_public_pointers,
    _training_source_pointers,
    _training_weight_source_pointers,
    _training_weight_source_values,
    _TrainingResourceContractOwner,
    _training_abi_prepared,
    _training_config,
    _training_contract_resources,
    _training_inputs,
    _training_prepared_pair,
    _training_staging_tensors,
    _training_weight_defect,
    _training_weights,
    make_forward_inputs,
)

# L0 contracts


@pytest.mark.L0
@pytest.mark.parametrize(
    "case",
    [
        pytest.param("backward-regions", id="backward-regions"),
        pytest.param("slot-lane-layout", id="slot-lane-layout"),
    ],
)
def test_training_workspace_layout_contract(case):
    if case == "backward-regions":
        requirements = WorkspaceRequirements.for_mxfp8(
            _training_config(),
            kernel_local_workspace_bytes=64,
            kernel_shared_workspace_bytes=128,
            backward_fc1_preact_bytes=1024,
            backward_dprob_bytes=32,
            backward_aux_data_bytes=512,
            backward_aux_scale_bytes=256,
        )
        expected = (
            ("symmetric", "backward_dprob", 32, None),
            ("local", "backward_fc1_preact", 1024, 128),
            ("local", "backward_aux_data", 512, None),
            ("local", "backward_aux_scale", 256, None),
        )
    else:
        config = _training_config()
        forward, backward = _training_prepared_pair(config)
        requirements = build_training_workspace_requirements(
            config,
            forward,
            backward,
            slot_count=2,
            lane_count=1,
        )
        expected = (
            ("symmetric", "lane.0.forward.symmetric.kernel_shared_workspace", None, None),
            ("symmetric", "lane.0.backward.symmetric.kernel_shared_workspace", None, None),
            ("symmetric", "slot.0.backward.symmetric.backward_dprob", None, None),
            ("symmetric", "slot.1.backward.symmetric.backward_dprob", None, None),
            ("local", "slot.0.persistent.local.fc1_preact", None, None),
            ("local", "slot.1.persistent.local.fc1_preact", None, None),
        )

    regions = {
        "symmetric": {region.name: region for region in requirements.symmetric_regions},
        "local": {region.name: region for region in requirements.local_regions},
    }
    for storage, name, nbytes, alignment in expected:
        region = regions[storage][name]
        if nbytes is not None:
            assert region.nbytes == nbytes
        if alignment is not None:
            assert region.alignment == alignment

    if case == "slot-lane-layout":
        assert tuple(region.name for region in requirements.symmetric_regions if region.name.startswith("slot.0.")) == (
            "slot.0.forward.symmetric.output_data",
            "slot.0.backward.symmetric.backward_dprob",
            "slot.0.backward.symmetric.output_data",
            "slot.0.persistent.symmetric.routing_topk_weights",
        )


@pytest.mark.L0
def test_training_workspace_harmonizes_each_symmetric_region(monkeypatch):
    requirements = WorkspaceRequirements(
        max_tokens_per_rank=1,
        symmetric_regions=(
            BufferRegion("first", 1),
            BufferRegion("second", 257),
        ),
        local_regions=(BufferRegion("local", 1),),
    )
    runtime = SimpleNamespace(world_size=2, group=object())

    def all_reduce(tensor, *, op, group):
        assert group is runtime.group
        if tensor.numel() == 2 and op == torch.distributed.ReduceOp.MAX:
            tensor.copy_(torch.tensor([257, 257], dtype=torch.int64))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    harmonized = _harmonize_symmetric_regions(
        requirements,
        runtime,
        torch.device("cpu"),
    )

    assert tuple(region.nbytes for region in harmonized.symmetric_regions) == (257, 257)
    assert harmonized.local_regions == requirements.local_regions


@pytest.mark.L0
def test_training_abi_fingerprint_is_stable_and_structural():
    config = _training_config(ep_size=2, ep_global_ranks=(0, 1))
    forward = _training_abi_prepared("forward")
    backward = _training_abi_prepared("backward")
    weights = _training_weights()
    requirements = WorkspaceRequirements(
        max_tokens_per_rank=4,
        symmetric_regions=(BufferRegion("symmetric", 256),),
        local_regions=(BufferRegion("local", 128),),
    )
    first = _build_training_abi_facts(
        config,
        forward,
        backward,
        weights,
        requirements,
        slot_count=2,
        lane_count=1,
        source_tree_digest="source",
    )
    second = _build_training_abi_facts(
        config,
        forward,
        backward,
        weights,
        requirements,
        slot_count=2,
        lane_count=1,
        source_tree_digest="source",
    )
    changed = _build_training_abi_facts(
        config,
        forward,
        backward,
        weights,
        requirements,
        slot_count=2,
        lane_count=2,
        source_tree_digest="source",
    )

    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    assert canonical_json_sha256(first) != canonical_json_sha256(changed)


@pytest.mark.L0
def test_training_abi_handshake_rejects_rank_mismatch(monkeypatch):
    runtime = SimpleNamespace(world_size=2, group=object())

    def all_reduce(tensor, *, op, group):
        assert group is runtime.group
        if op == torch.distributed.ReduceOp.MAX:
            tensor.add_(1)

    def all_gather_object(output, value, *, group):
        assert group is runtime.group
        output[:] = [value, "different"]

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        all_gather_object,
    )
    with pytest.raises(RuntimeError, match="ABI differs"):
        _verify_training_abi_across_ranks(
            {"schema_version": 1},
            runtime,
            torch.device("cpu"),
        )


@pytest.mark.L0
def test_training_resource_views_share_lane_scratch_but_not_slot_state():
    config = _training_config()
    forward, backward = _training_prepared_pair(config)

    class Runtime:
        device = torch.device("cpu")
        rank = 0
        world_size = 1
        nvshmem_enabled = False
        closed = False

        def ensure_open(self):
            assert not self.closed

        def close(self):
            self.closed = True

    runtime = Runtime()
    runtime_manager = SimpleNamespace(acquire=lambda actual_config, actual_device: runtime)
    weights = _training_weights()
    owner = Mxfp8TrainingResourceOwner(
        config,
        torch.device("cpu"),
        forward,
        backward,
        weights,
        slot_count=2,
        lane_count=1,
        runtime_manager=runtime_manager,
    )
    try:
        first = owner.views(slot=0, lane=0, token_count=4)
        second = owner.views(slot=1, lane=0, token_count=4)
        assert first.forward.workspace.local["kernel_local_workspace"].data_ptr() == second.forward.workspace.local["kernel_local_workspace"].data_ptr()
        assert first.slot.fc1_preact.data_ptr() != second.slot.fc1_preact.data_ptr()
        assert first.slot.dprob.data_ptr() != second.slot.dprob.data_ptr()
        assert first.forward_expert_size_snapshot is not None
        assert first.forward_expert_size_snapshot.data_ptr() == (second.forward_expert_size_snapshot.data_ptr())
    finally:
        owner.close()

    assert runtime.closed


@pytest.mark.L0
def test_training_sources_track_adapter_grad_y2_and_dfc2_contracts():
    from cudnn.moe_ep._megamoe_backend.mxfp8 import (
        _backward_compile,
        _compile,
    )

    forward_source = inspect.getsource(_compile.prepare_kernel)
    backward_source = inspect.getsource(_backward_compile.prepare_backward_kernel)
    runtime_source = inspect.getsource(_backward_compile.build_backward_runtime_kwargs)
    dglu_source = _DGLU.read_text(encoding="utf-8")
    dfc2_source = _DGLU_EPILOGUE.read_text(encoding="utf-8")

    assert "gate_up_clamp=config.gate_up_clamp" in backward_source
    assert "dfc2_recompute=dfc2_recompute" in backward_source
    assert "enable_grad_y2_col_quant=enable_grad_y2_col_quant" in backward_source
    assert '"fc1_preact":' in runtime_source
    assert '"dprob":' in runtime_source
    assert "generate_c=config.generate_c" in forward_source
    for contract in (
        "enable_grad_y2_col_quant",
        "num_ctas_grad_y2_col_quant",
        "grad_y2_sizes_region",
        "_snapshot_grad_y2_expert_sizes",
        "grad_y2_col_quant",
        "grad_y2: cute.Tensor",
        "grad_y2_sf: cute.Tensor",
    ):
        assert contract in dglu_source
    assert dglu_source.index("self._snapshot_grad_y2_expert_sizes(tidx)") < dglu_source.index("self.token_comm.reset_tail()")
    assert dglu_source.index("self._topk_reduce(") < dglu_source.index("self.grad_y2_col_quant(")
    assert "def _stg_col_sf_atom_value(" in dfc2_source
    assert "feature_atom = feature // cutlass.Int32(128)" in dfc2_source
    assert "feature_lane * cutlass.Int32(16)" in dfc2_source
    assert "feature_bank * cutlass.Int32(4)" in dfc2_source
    assert "real_sf[feature_atom, token_atom, atom_byte]" in dfc2_source
    assert "def tma_store_dfc2_outputs(" in dfc2_source
    assert dfc2_source.count("self._stg_col_sf_atom_value(") >= 2


# WGrad operand contracts


@pytest.mark.L1
@pytest.mark.parametrize(
    ("field", "defect"),
    [
        pytest.param(field, "logical_shape", id=f"{field}-logical-shape")
        for field in (
            "forward_fc1",
            "forward_fc2",
            "backward_w2_transpose",
            "backward_w1_transpose",
        )
    ]
    + [
        pytest.param("forward_fc1", defect, id=f"forward_fc1-{defect}")
        for defect in (
            "plain_tensor",
            "axis",
            "format",
            "data_noncontiguous",
            "scale_noncontiguous",
        )
    ],
)
def test_validate_training_weights_rejects_targeted_defects(field, defect):
    invalid, error_type, message = _training_weight_defect(
        _training_weights(),
        field,
        defect,
    )
    with pytest.raises(error_type) as exc_info:
        validate_training_weights(_training_config(), invalid)
    assert str(exc_info.value) == message


@pytest.mark.L1
def test_validate_training_weights_rejects_cross_field_device_mismatch():
    invalid, error_type, message = _training_weight_defect(
        _training_weights(),
        "backward_w1_transpose",
        "device",
    )
    with pytest.raises(error_type) as exc_info:
        validate_training_weights(_training_config(), invalid)
    assert str(exc_info.value) == message


@pytest.mark.L1
def test_validate_training_weights_accepts_complete_fixed_weight_set():
    assert validate_training_weights(
        _training_config(),
        _training_weights(),
    ) == torch.device("cpu")


def _operator(**overrides) -> MoeEp:
    values = {
        "num_experts": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "top_k": 2,
        "max_tokens_per_rank": 4,
    }
    values.update(overrides)
    return MoeEp(
        **values,
    )


def _install_contract_backend(
    monkeypatch,
    *,
    weights=None,
    slot_count=1,
    lane_count=1,
):
    import cudnn.moe_ep._backend as backend_seam
    import cudnn.moe_ep.api as api_module

    weights = weights or SimpleNamespace(mock_training_weights=True)
    state = SimpleNamespace(
        backends=[],
        validate=Mock(return_value=torch.device("cpu")),
    )

    def create_backend(config, device):
        del config, device
        owner = _TrainingResourceContractOwner(
            slot_count=slot_count,
            lane_count=lane_count,
        )
        backend = SimpleNamespace(
            owner=owner,
            prepare_training_resources=Mock(return_value=owner),
            close=Mock(),
        )
        state.backends.append(backend)
        return backend

    monkeypatch.setattr(api_module, "validate_training_weights", state.validate)
    monkeypatch.setattr(backend_seam, "validate_config", lambda config: None)
    monkeypatch.setattr(backend_seam, "create_backend", create_backend)
    return weights, state


@pytest.mark.L0
def test_k_major_workspace_view_matches_upstream_token_major_abi():
    storage = torch.arange(12, dtype=torch.uint8)
    view = _typed_k_major_view(storage, torch.uint8, (3, 4))

    assert view.shape == (3, 4)
    assert view.stride() == (1, 3)
    assert torch.equal(view, storage.reshape(4, 3).transpose(0, 1))


@pytest.mark.L0
def test_only_fixed_training_wgrad_types_are_public():
    expected = [f"fc{layer}_{part}" for layer in (1, 2) for part in ("a", "sfa", "b", "sfb")]
    expected += ["expert_offsets", "valid_route_counts"]
    assert [field.name for field in fields(MoeEpTrainingWgradOperands)] == expected
    assert not hasattr(cudnn, "MoeEpWgradForwardStash")
    assert not hasattr(cudnn, "MoeEpWgradOperands")


@pytest.mark.L0
def test_prepare_training_resources_binds_weights_and_slot_lanes(monkeypatch):
    weights = _training_weights()
    _, state = _install_contract_backend(
        monkeypatch,
        weights=weights,
        slot_count=2,
    )
    operator = _operator()
    resources = operator.prepare_training_resources(
        weights,
        slot_count=2,
        lane_count=1,
    )

    assert isinstance(resources, MoeEpTrainingResources)
    assert all(isinstance(slot, MoeEpTrainingSlot) for slot in resources.slots)
    assert isinstance(resources.lanes[0], MoeEpExecutionLane)
    resources.refresh_weights()
    owner = state.backends[0].owner
    assert owner.refresh_calls == 1
    operator.close()
    assert resources.closed
    assert owner.close_calls == 1


@pytest.mark.L0
def test_prepare_training_resources_rejects_plain_weights():
    with _operator() as operator:
        with pytest.raises(TypeError, match="MoeEpTrainingWeights"):
            operator.prepare_training_resources(_training_inputs()[1])


@pytest.mark.L0
def test_error_mode_requires_async_assert_before_prepare(monkeypatch):
    from cudnn.moe_ep.api import _validate_training_assert_capability

    monkeypatch.setattr(torch, "_assert_async", None)
    config = SimpleNamespace(drop_on_overflow=False, ep_size=1)
    with pytest.raises(RuntimeError, match="callable torch._assert_async"):
        _validate_training_assert_capability(config)

    _validate_training_assert_capability(SimpleNamespace(drop_on_overflow=True, ep_size=1))


@pytest.mark.L0
def test_distributed_error_mode_requires_nccl(monkeypatch):
    from cudnn.moe_ep.api import _validate_training_assert_capability

    monkeypatch.setattr(torch, "_assert_async", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")
    config = SimpleNamespace(
        drop_on_overflow=False,
        ep_size=2,
        ep_group=object(),
    )
    with pytest.raises(NotImplementedError, match="NCCL"):
        _validate_training_assert_capability(config)


@pytest.mark.L0
def test_training_weight_refresh_keeps_destination_addresses_stable():
    weights = _training_weights()
    bindings = Mxfp8TrainingWeightBindings(weights)
    bindings.refresh()
    tensors = (
        bindings.forward.fc1_weight,
        bindings.forward.fc1_weight_sf,
        bindings.forward.fc2_weight,
        bindings.forward.fc2_weight_sf,
        bindings.backward.fc1_weight,
        bindings.backward.fc1_weight_sf,
        bindings.backward.fc2_weight,
        bindings.backward.fc2_weight_sf,
    )
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    snapshots = tuple(tensor.clone() for tensor in tensors)

    weights.forward_fc1.data.view(torch.uint8).bitwise_xor_(1)
    bindings.refresh()

    assert tuple(tensor.data_ptr() for tensor in tensors) == pointers
    assert not torch.equal(bindings.forward.fc1_weight, snapshots[0])
    for tensor in tensors:
        assert tensor.is_contiguous() or tensor.stride(1) == 1


@pytest.mark.L0
def test_reference_wgrad_math_remains_in_test_tree():
    torch.manual_seed(20260821)
    tokens, hidden, intermediate = 3, 32, 32
    topk_idx = torch.tensor([[0, 2], [2, 0], [0, 2]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.25], [0.0, 0.75], [1.0, 0.125]])
    activation, fc1_weight, fc2_weight, grad_output = (
        torch.randn(shape) / 8
        for shape in (
            (tokens, hidden),
            (3, hidden, 2 * intermediate),
            (3, intermediate, hidden),
            (tokens, hidden),
        )
    )
    reference = MoeEpReference(
        num_experts=3,
        hidden_size=hidden,
        intermediate_size=intermediate,
        top_k=2,
        max_tokens_per_rank=tokens,
        generate_c=True,
        backward_wgrad_mode="operands",
        token_padding_size=256,
    )

    _, fc1_c, metadata, stash = reference(activation, fc1_weight, fc2_weight, topk_idx, topk_weights)
    _, _, operands = reference.backward(
        grad_output,
        fc1_weight,
        fc2_weight,
        topk_idx,
        topk_weights,
        fc1_c,
        metadata,
        wgrad_forward_stash=stash,
    )
    dw1, dw2 = operands.dense_wgrads()

    assert operands.valid_route_counts.tolist() == [3, 0, 3]
    assert dw1.shape == (3, hidden, 2 * intermediate)
    assert dw2.shape == (3, intermediate, hidden)
    assert dw1[1].eq(0).all()
    assert dw2[1].eq(0).all()


# Source contracts


_ROOT = Path(__file__).resolve().parents[3]
_CUTEDSL = _ROOT / "python/cudnn/moe_ep/_megamoe_backend/cutedsl_src/kernel_src/rubin" / "training/mega"
_DGLU = _CUTEDSL / "bwd_dglu/dglu_mxfp8_mega_moe_kernel.py"
_DGLU_EPILOGUE = _CUTEDSL / "bwd_dglu/dglu_mxfp8_fc12_epilogue.py"


# L1 fail-fast training-resource contracts


@pytest.mark.L1
@pytest.mark.parametrize(
    ("field", "value"),
    [(field, value) for field in ("slot_count", "lane_count") for value in (0, True, 1.5)],
)
def test_prepare_training_resources_rejects_invalid_counts_before_backend(
    monkeypatch,
    field,
    value,
):
    import cudnn.moe_ep._backend as backend_seam
    import cudnn.moe_ep.api as api_module

    def unexpected_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("invalid counts must fail before weight/backend work")

    monkeypatch.setattr(
        api_module,
        "validate_training_weights",
        unexpected_call,
    )
    monkeypatch.setattr(backend_seam, "create_backend", unexpected_call)
    counts = {"slot_count": 1, "lane_count": 1, field: value}

    with _operator() as operator, pytest.raises(
        ValueError,
        match=rf"{field} must be a positive integer",
    ):
        operator.prepare_training_resources(
            SimpleNamespace(mock_training_weights=True),
            **counts,
        )


@pytest.mark.L1
def test_prepare_training_resources_rejects_duplicate_open_resources(monkeypatch):
    weights, state = _install_contract_backend(monkeypatch)

    with _operator() as operator:
        resources = operator.prepare_training_resources(
            weights,
            slot_count=1,
            lane_count=1,
        )
        with pytest.raises(RuntimeError, match="already exist"):
            operator.prepare_training_resources(
                weights,
                slot_count=1,
                lane_count=1,
            )

    assert resources.closed
    backend = state.backends[0]
    state.validate.assert_called_once()
    backend.prepare_training_resources.assert_called_once_with(
        weights,
        slot_count=1,
        lane_count=1,
    )
    assert (backend.close.call_count, backend.owner.close_calls) == (1, 1)


@pytest.mark.L1
def test_closed_training_resources_require_a_new_operator(monkeypatch):
    weights, state = _install_contract_backend(monkeypatch)

    old_operator = _operator()
    old_resources = old_operator.prepare_training_resources(
        weights,
        slot_count=1,
        lane_count=1,
    )
    old_resources.close()
    with pytest.raises(
        RuntimeError,
        match="create a new MoeEp instance",
    ):
        old_operator.prepare_training_resources(
            weights,
            slot_count=1,
            lane_count=1,
        )
    old_operator.close()

    with _operator() as new_operator:
        new_resources = new_operator.prepare_training_resources(
            weights,
            slot_count=1,
            lane_count=1,
        )
        assert not new_resources.closed

    assert len(state.backends) == 2
    assert state.validate.call_count == 2
    assert all(backend.prepare_training_resources.call_count == 1 for backend in state.backends)


@pytest.mark.L1
def test_training_prepare_and_backend_close_reject_capture(monkeypatch):
    from cudnn.moe_ep._megamoe_backend.mxfp8._backend import Mxfp8Backend

    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )

    owner = object.__new__(Mxfp8TrainingResourceOwner)
    owner._lock = threading.RLock()
    owner._closed = False
    owner._runtime = None
    owner._workspace = None
    with pytest.raises(
        RuntimeError,
        match="must be prepared before CUDA graph capture",
    ):
        owner.prepare()

    backend = object.__new__(Mxfp8Backend)
    backend._lock = threading.RLock()
    backend._closed = False
    backend.device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    with pytest.raises(RuntimeError, match="cannot be closed during"):
        backend.close()


@pytest.mark.L1
def test_training_resources_reject_foreign_and_forged_slot_lane_bindings():
    resources, owner = _training_contract_resources()
    foreign, _ = _training_contract_resources()
    slot = resources.slots[0]
    lane = resources.lanes[0]
    activation = torch.empty((0, 128), dtype=torch.bfloat16)
    routing = (
        torch.empty((0, 2), dtype=torch.int32),
        torch.empty((0, 2), dtype=torch.float32),
    )
    checks = (
        ("training slot does not belong", resources.forward, (foreign.slots[0], lane, activation, *routing)),
        ("training slot does not belong", resources.backward, (MoeEpTrainingSlot(99, slot._resource_token), lane, activation.float())),
        ("execution lane does not belong", resources.forward, (slot, foreign.lanes[0], activation, *routing)),
        ("execution lane does not belong", resources.backward, (slot, MoeEpExecutionLane(99, lane._resource_token), activation.float())),
    )
    for message, call, args in checks:
        with pytest.raises(ValueError, match=message):
            call(*args)

    assert owner.views_calls == 0


@pytest.mark.L1
def test_training_resources_reject_invalid_overflow_finalization():
    resources, _ = _training_contract_resources()
    foreign, _ = _training_contract_resources()
    slot = resources.slots[0]
    lane = resources.lanes[0]

    with pytest.raises(ValueError, match="at least one slot"):
        resources.finalize_overflow((), lane)
    with pytest.raises(ValueError, match="slots must be unique"):
        resources.finalize_overflow((slot, slot), lane)
    with pytest.raises(ValueError, match="overflow slot does not belong"):
        resources.finalize_overflow((foreign.slots[0],), lane)
    with pytest.raises(ValueError, match="overflow execution lane does not belong"):
        resources.finalize_overflow((slot,), foreign.lanes[0])


@pytest.mark.L1
def test_training_resources_reject_calls_after_close_and_close_is_idempotent():
    resources, owner = _training_contract_resources()
    slot = resources.slots[0]
    lane = resources.lanes[0]
    activation = torch.empty((0, 128), dtype=torch.bfloat16)
    routing = (
        torch.empty((0, 2), dtype=torch.int32),
        torch.empty((0, 2), dtype=torch.float32),
    )

    resources.close()
    resources.close()

    assert resources.closed
    assert owner.close_calls == 1
    calls = (
        resources.refresh_weights,
        lambda: resources.forward(slot, lane, activation, *routing),
        lambda: resources.backward(slot, lane, activation.float()),
        lambda: resources.finalize_overflow((slot,), lane),
    )
    for call in calls:
        with pytest.raises(RuntimeError, match="resources are closed"):
            call()
    assert owner.refresh_calls == 0
    assert owner.views_calls == 0


@pytest.mark.L1
@pytest.mark.parametrize(
    ("mismatch_reduce", "message"),
    [
        (2, "region counts differ"),
        (4, "names, order, or alignments differ"),
    ],
)
def test_harmonize_symmetric_regions_rejects_collective_metadata_mismatch(
    monkeypatch,
    mismatch_reduce,
    message,
):
    requirements = WorkspaceRequirements(
        max_tokens_per_rank=1,
        symmetric_regions=(
            BufferRegion("first", 64, alignment=128),
            BufferRegion("second", 128, alignment=256),
        ),
        local_regions=(),
    )
    runtime = SimpleNamespace(world_size=2, group=object())
    reduce_calls = []

    def all_reduce(tensor, *, op, group):
        assert group is runtime.group
        reduce_calls.append(op)
        if len(reduce_calls) == mismatch_reduce:
            tensor.add_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    with pytest.raises(RuntimeError, match=message):
        _harmonize_symmetric_regions(
            requirements,
            runtime,
            torch.device("cpu"),
        )

    assert len(reduce_calls) == mismatch_reduce


_STAGER_FAILURES = {
    "source-shape": (lambda t: t.update(source=t["source"][:, :-1].contiguous()), ValueError, r"source must have shape \(T, 128\)"),
    "route-shape": (lambda t: t.update(topk_idx=t["topk_idx"][:, :-1].contiguous()), ValueError, "topk_idx shape mismatch"),
    "weight-shape": (lambda t: t.update(topk_weights=t["topk_weights"][:, :-1].contiguous()), ValueError, "topk_weights shape mismatch"),
    "route-dtype": (lambda t: t.update(topk_idx=t["topk_idx"].to(torch.int64)), TypeError, "contiguous Int32"),
    "route-contiguity": (lambda t: t.update(topk_idx=t["topk_idx"].t().contiguous().t()), TypeError, "contiguous Int32"),
    "weight-dtype": (lambda t: t.update(topk_weights=t["topk_weights"].to(torch.bfloat16)), TypeError, "contiguous FP32"),
    "weight-contiguity": (lambda t: t.update(topk_weights=t["topk_weights"].t().contiguous().t()), TypeError, "contiguous FP32"),
    "capacity": (lambda t: t.update(**{name: value[:4] for name, value in t.items() if name.startswith("output")}), ValueError, "token count 5 exceeds capacity 4"),
    "device": (lambda t: t.update(source=torch.empty_like(t["source"], device="meta")), ValueError, "must share one device"),
}


@pytest.mark.L1
@pytest.mark.parametrize(
    ("mutator", "error_type", "message"),
    [
        pytest.param(*case, id=name)
        for name, case in _STAGER_FAILURES.items()
    ],
)
def test_training_stager_rejects_invalid_inputs(mutator, error_type, message):
    tensors = _training_staging_tensors()
    mutator(tensors)
    with pytest.raises(error_type, match=message):
        Mxfp8TrainingStager(hidden=128, top_k=2)._validate(**tensors)


# L1 training graph


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    (
        "input_kind",
        "combine_format",
        "gate_up_clamp",
        "top_k",
        "tuning",
        "all_dropped",
    ),
    [
        pytest.param("fixed", "bf16", None, 2, MoeEpTuningConfig(), False, id="bf16-unclamped"),
        pytest.param("fixed", "mxfp8", 0.5, 2, MoeEpTuningConfig(), False, id="mxfp8-clamp-0.5"),
        pytest.param("routed", "bf16", None, 1, MoeEpTuningConfig(), False, id="topk1-default-tuning"),
        pytest.param(
            "routed",
            "bf16",
            None,
            2,
            MoeEpTuningConfig(
                token_back_mode="reuse_dispatch_warps",
                epi_flag_batch=(2, 2),
                token_in_flag_batch=2,
                group_hint=128,
            ),
            False,
            id="topk2-nondefault-tuning",
        ),
        pytest.param("routed", "bf16", None, 2, MoeEpTuningConfig(), True, id="topk2-all-dropped"),
    ],
)
def test_fixed_training_resources_ep1_matches_independent_reference(
    input_kind,
    combine_format,
    gate_up_clamp,
    top_k,
    tuning,
    all_dropped,
):
    device = _sm107_device()
    if input_kind == "fixed":
        args, grad_output = _fixed_training_case(device)
        max_recv_size = 1
    else:
        base_args = make_forward_inputs(device)
        args = (
            base_args[0].dequantize(torch.bfloat16),
            base_args[1],
            base_args[2],
            base_args[3][:, :top_k].contiguous(),
            base_args[4][:, :top_k].float().contiguous(),
        )
        if all_dropped:
            args[3].fill_(-1)
            args[4].zero_()
        grad_output = _grad_output(device, args[0].shape[0], seed=20260830)
        max_recv_size = args[0].shape[0] * top_k
    expected = _fixed_training_reference(
        args,
        grad_output,
        combine_format=combine_format,
        gate_up_clamp=gate_up_clamp,
        tuning=tuning,
    )

    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=top_k,
        max_tokens_per_rank=args[0].shape[0],
        max_recv_size_per_rank=max_recv_size,
        drop_on_overflow=True,
        combine_format=combine_format,
        gate_up_clamp=gate_up_clamp,
        tuning=tuning,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(args),
            slot_count=1,
            lane_count=1,
        )
        slot = resources.slots[0]
        lane = resources.lanes[0]
        actual = _run_fixed_training_batch(
            resources,
            lane,
            ((slot, args, grad_output),),
        )[0]
        torch.cuda.synchronize(device)

        assert actual.overflow.eq(0).all()
        _assert_fixed_training_matches_reference(
            (actual.y, actual.dx, actual.dprob, actual.wgrads),
            expected,
            args[3],
        )

        if all_dropped:
            actual_dw1, actual_dw2 = _dense_wgrads_from_operands(actual.wgrads)
            expected_dw1, expected_dw2 = expected[3].dense_wgrads()
            zero_tensors = (
                actual.y,
                expected[0],
                actual.dx,
                expected[1],
                actual.dprob,
                expected[2],
                actual_dw1,
                expected_dw1,
                actual_dw2,
                expected_dw2,
            )
            assert all(tensor.eq(0).all() for tensor in zero_tensors)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    ("world_size", "combine_format", "gate_up_clamp"),
    [
        pytest.param(2, "bf16", None, id="ep2-bf16"),
        pytest.param(2, "mxfp8", None, id="ep2-mxfp8"),
        pytest.param(4, "bf16", None, id="ep4-bf16"),
        pytest.param(4, "mxfp8", None, id="ep4-mxfp8"),
        pytest.param(2, "bf16", 0.5, id="ep2-bf16-clamp-0.5"),
    ],
)
def test_fixed_training_resources_multi_gpu_matches_independent_reference(
    world_size,
    combine_format,
    gate_up_clamp,
    tmp_path,
):
    _require_distributed_sm107(world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    clamp_id = "none" if gate_up_clamp is None else str(gate_up_clamp)
    init_file = tmp_path / f"backward_ep{world_size}_{combine_format}_clamp_{clamp_id}.init"
    mp.spawn(
        _distributed_backward_reference_worker,
        args=(
            world_size,
            str(init_file),
            combine_format,
            gate_up_clamp,
        ),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_noncontiguous_ep2_fixed_training_matches_independent_reference(
    tmp_path,
):
    global_world_size = 4
    _require_distributed_sm107(global_world_size)
    os.environ.setdefault("NVIDIA_IMEX_CHANNELS", "0")
    init_file = tmp_path / "backward_two_noncontiguous_ep2.init"
    mp.spawn(
        _distributed_subgroup_backward_reference_worker,
        args=(global_world_size, str(init_file)),
        nprocs=global_world_size,
        join=True,
    )


@pytest.mark.L1
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            SimpleNamespace(
                combine_format="bf16",
                drop_on_overflow=True,
                max_recv_size=1,
                replay_count=20,
            ),
            id="bf16-drop",
        ),
        pytest.param(
            SimpleNamespace(
                combine_format="mxfp8",
                drop_on_overflow=True,
                max_recv_size=1,
                replay_count=20,
            ),
            id="mxfp8-drop",
        ),
        pytest.param(
            SimpleNamespace(
                combine_format="bf16",
                drop_on_overflow=False,
                max_recv_size=2,
                replay_count=2,
            ),
            id="bf16-error-no-overflow",
        ),
    ],
)
def test_fixed_training_resources_ep1_cuda_graph_replay(case):
    device = _sm107_device()
    if case.drop_on_overflow:
        args0, grad0 = _fixed_training_case(device)
        topk_idx1 = args0[3].clone()
        topk_idx1[0, 0] = 1
        inputs = (
            (args0, grad0),
            (
                (
                    args0[0].clone(),
                    args0[1],
                    args0[2],
                    topk_idx1,
                    args0[4].clone(),
                ),
                grad0.clone(),
            ),
        )
    else:
        inputs = (_fixed_training_drop_overflow_case(device),)
    references = tuple(
        _fixed_training_reference(
            args,
            grad_output,
            combine_format=case.combine_format,
            gate_up_clamp=None,
        )
        for args, grad_output in inputs
    )

    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=inputs[0][0][0].shape[0],
        max_recv_size_per_rank=case.max_recv_size,
        drop_on_overflow=case.drop_on_overflow,
        combine_format=case.combine_format,
        token_padding_size=128,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(inputs[0][0]),
            slot_count=len(inputs),
            lane_count=1,
        )
        lane = resources.lanes[0]
        batch = tuple((slot, args, grad_output) for slot, (args, grad_output) in zip(resources.slots, inputs))

        def assert_batch(actuals):
            for actual, (args, _), reference in zip(
                actuals,
                inputs,
                references,
            ):
                assert actual.overflow.shape == (1,)
                assert actual.overflow.dtype == torch.int32
                assert actual.overflow.eq(0).all()
                _assert_fixed_training_matches_reference(
                    (actual.y, actual.dx, actual.dprob, actual.wgrads),
                    reference,
                    args[3],
                )

        eager_actuals = _run_fixed_training_batch(resources, lane, batch)
        torch.cuda.synchronize(device)
        assert_batch(eager_actuals)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        captured = _capture_fixed_training_batch(
            resources,
            lane,
            batch,
            stream,
        )
        # In error mode, each replay executes the captured torch._assert_async
        # with a false overflow condition; stable public nodes prove reuse.
        for _ in range(case.replay_count):
            captured.graph.replay()
            torch.cuda.synchronize(device)
            assert captured.public_pointers == tuple(_training_public_pointers(actual) for actual in captured.actuals)
            assert_batch(captured.actuals)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_fixed_training_resources_ep1_two_shape_cuda_graph_contract():
    device = _sm107_device()
    args, grad_large = _fixed_training_case(device)
    max_tokens = int(args[0].shape[0])
    small_tokens = max_tokens - 2
    assert 0 < small_tokens < max_tokens

    large = SimpleNamespace(
        name="large",
        activation=args[0],
        topk_idx=args[3],
        topk_weights=args[4],
        grad_output=grad_large,
    )
    small = SimpleNamespace(
        name="small",
        activation=args[0][:small_tokens].clone(),
        topk_idx=args[3][:small_tokens].clone(),
        topk_weights=args[4][:small_tokens].clone(),
        grad_output=grad_large[:small_tokens].clone(),
    )
    assert all(getattr(large, name).data_ptr() != getattr(small, name).data_ptr() for name in ("activation", "topk_idx", "topk_weights", "grad_output"))

    weights = _fixed_training_weights(args)
    weight_source_pointers = _training_weight_source_pointers(weights)

    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=max_tokens,
        max_recv_size_per_rank=1,
        drop_on_overflow=True,
    ) as op:
        resources = op.prepare_training_resources(
            weights,
            slot_count=1,
            lane_count=1,
        )
        slot = resources.slots[0]
        lane = resources.lanes[0]

        def case_args(case):
            return (
                case.activation,
                weights.forward_fc1,
                weights.forward_fc2,
                case.topk_idx,
                case.topk_weights,
            )

        def independent_reference(case):
            return _fixed_training_reference(
                case_args(case),
                case.grad_output,
                combine_format="bf16",
                gate_up_clamp=None,
            )

        def warmup(case) -> None:
            actual = _run_fixed_training_batch(
                resources,
                lane,
                ((slot, case_args(case), case.grad_output),),
            )[0]
            torch.cuda.synchronize(device)
            assert actual.overflow.eq(0).all(), f"{case.name} warmup overflowed"
            _assert_fixed_training_matches_reference(
                (actual.y, actual.dx, actual.dprob, actual.wgrads),
                independent_reference(case),
                case.topk_idx,
            )

        # Compile each static token-count specialization on the same resources,
        # slot, and lane before either capture.
        warmup(large)
        warmup(small)

        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))

        def capture(case):
            # The shared sequence records refresh so replay observes in-place
            # updates to all four bound source packs.
            captured = _capture_fixed_training_batch(
                resources,
                lane,
                ((slot, case_args(case), case.grad_output),),
                capture_stream,
            )
            return SimpleNamespace(
                case=case,
                graph=captured.graph,
                actual=captured.actuals[0],
                public_pointers=captured.public_pointers[0],
                source_pointers=_training_source_pointers(case),
            )

        large_graph = capture(large)
        small_graph = capture(small)
        slot_views = resources._owner.views(
            slot=slot.index,
            lane=lane.index,
            token_count=max_tokens,
        ).slot

        def replay_and_check(captured):
            _prefill_training_graph_sentinels(slot_views, captured.actual)
            captured.graph.replay()
            torch.cuda.synchronize(device)

            assert captured.actual.overflow.eq(0).all()
            assert _training_public_pointers(captured.actual) == captured.public_pointers
            assert _training_source_pointers(captured.case) == captured.source_pointers
            assert _training_weight_source_pointers(weights) == weight_source_pointers
            _assert_fixed_training_matches_reference(
                (
                    captured.actual.y,
                    captured.actual.dx,
                    captured.actual.dprob,
                    captured.actual.wgrads,
                ),
                independent_reference(captured.case),
                captured.case.topk_idx,
            )
            # The dense-dW check above decodes every expert segment and rejects
            # nonzero expert padding, nonzero data capacity tails, or
            # non-neutral scale tails left by the sentinels.
            _assert_training_graph_tails_are_reset(
                slot_views,
                captured.actual,
                token_count=int(captured.case.activation.shape[0]),
                capacity=max_tokens,
            )
            return captured.actual.y.clone()

        # The two graphs alias one persistent slot. Each replay must therefore
        # fully replace the other shape's routing, gradients, and WGrad state.
        for captured in (large_graph, small_graph, large_graph):
            replay_and_check(captured)

        small_source_pointers = _training_source_pointers(small)
        small.activation.mul_(-0.5)
        small.topk_idx.fill_(-1)
        small.topk_idx[0, 0] = 1
        small.topk_weights.zero_()
        small.topk_weights[0, 0] = 0.625
        small.grad_output.mul_(-0.75)
        assert _training_source_pointers(small) == small_source_pointers

        for captured in (small_graph, large_graph, small_graph):
            replay_and_check(captured)

        old_large_y = replay_and_check(large_graph)
        old_weight_values = _training_weight_source_values(weights)
        generator = torch.Generator(device=device).manual_seed(20260829)
        new_fc1 = (
            torch.randn(
                weights.forward_fc1.logical_shape,
                generator=generator,
                device=device,
            )
            / 16
        )
        new_fc2 = (
            torch.randn(
                weights.forward_fc2.logical_shape,
                generator=generator,
                device=device,
            )
            / 16
        )
        replacement = _fixed_training_weights(
            (
                large.activation,
                new_fc1,
                new_fc2,
                large.topk_idx,
                large.topk_weights,
            )
        )
        _copy_training_weight_sources_(weights, replacement)

        assert _training_weight_source_pointers(weights) == weight_source_pointers
        _assert_training_weight_sources_changed(weights, old_weight_values)

        new_large_y = replay_and_check(large_graph)
        assert not torch.equal(new_large_y, old_large_y)


@pytest.mark.L1
@pytest.mark.gpu_exclusive
def test_fixed_training_resources_ep1_drop_overflow_boundary_and_graph_transitions():
    device = _sm107_device()
    args, grad_output = _fixed_training_drop_overflow_case(device)
    assert args[0].shape[0] == 1
    assert args[3].detach().cpu().tolist() == [[0, 1]]

    references = {
        expected_overflow: _fixed_training_drop_overflow_reference(
            args,
            grad_output,
            drop_expert1=bool(expected_overflow),
        )
        for expected_overflow in (0, 1)
    }

    def assert_result(actual, expected_overflow):
        expected, reference_topk_idx = references[expected_overflow]
        _assert_fixed_training_drop_overflow_result(
            actual,
            expected,
            reference_topk_idx,
            expected_overflow=expected_overflow,
        )

    # The graph warmup below covers maxrecv=1 overflow; exercise the exact
    # non-overflow boundary separately here.
    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=1,
        max_recv_size_per_rank=2,
        drop_on_overflow=True,
        token_padding_size=128,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(args),
            slot_count=1,
            lane_count=1,
        )
        slot = resources.slots[0]
        lane = resources.lanes[0]
        actual = _run_fixed_training_batch(
            resources,
            lane,
            ((slot, args, grad_output),),
        )[0]
        torch.cuda.synchronize(device)

        assert_result(actual, 0)
        assert args[3][0, 1].eq(1)
        assert actual.wgrads.valid_route_counts.detach().cpu().tolist() == [1, 1]
        assert actual.wgrads.expert_offsets.detach().cpu().tolist() == [128, 256]

    overflow_routing = args[3].clone()
    expert0_only_routing = overflow_routing.clone()
    expert0_only_routing[0, 1] = -1
    routing_pointer = args[3].data_ptr()

    with MoeEp(
        num_experts=2,
        hidden_size=128,
        intermediate_size=256,
        top_k=2,
        max_tokens_per_rank=1,
        max_recv_size_per_rank=1,
        drop_on_overflow=True,
        token_padding_size=128,
    ) as op:
        resources = op.prepare_training_resources(
            _fixed_training_weights(args),
            slot_count=1,
            lane_count=1,
        )
        slot = resources.slots[0]
        lane = resources.lanes[0]
        batch = ((slot, args, grad_output),)

        # Compile the fixed T=1 specialization and validate overflow eagerly
        # before capturing the same forward/backward/finalize sequence.
        warmup = _run_fixed_training_batch(resources, lane, batch)[0]
        torch.cuda.synchronize(device)
        assert_result(warmup, 1)

        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        captured = _capture_fixed_training_batch(
            resources,
            lane,
            batch,
            capture_stream,
        )
        graph_actual = captured.actuals[0]

        for routing, expected_overflow in (
            (overflow_routing, 1),
            (expert0_only_routing, 0),
            (overflow_routing, 1),
        ):
            args[3].copy_(routing)
            assert args[3].data_ptr() == routing_pointer
            expected = _fixed_training_drop_overflow_reference(
                args,
                grad_output,
                drop_expert1=bool(expected_overflow),
            )
            captured.graph.replay()
            torch.cuda.synchronize(device)
            _assert_fixed_training_drop_overflow_result(
                graph_actual,
                *expected,
                expected_overflow=expected_overflow,
            )
            assert captured.public_pointers[0] == _training_public_pointers(graph_actual)
