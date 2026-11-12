# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native JAX shared-expert decode: numerics, buffer reuse, and row invariance."""

import pytest

jax = pytest.importorskip("jax", minversion="0.9.1")
torch = pytest.importorskip("torch")

from cudnn.ops import shared_expert_decode_forward_jax
from determinism_utils import assert_bitwise, assert_replays, snapshot
from moe.frost.shared_expert_utils import GEOMETRIES, _gid, assert_in_band, make_operands, reference

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture
def device():
    target = next((d for d in jax.local_devices() if d.platform == "gpu" and str(getattr(d, "compute_capability", "")).startswith("10.")), None)
    if target is None:
        pytest.skip("Shared-expert decode requires an SM10x GPU")
    with jax.default_device(target), torch.cuda.device(target.local_hardware_id):
        yield torch.device("cuda", target.local_hardware_id)


def execute(run, operands):
    with torch.cuda.device(operands["x"].device):
        values = tuple(jax.dlpack.from_dlpack(operands[name]) for name in ("x", "w_gate_up", "w_down", "w_gate"))
        result = jax.block_until_ready(run(*values))
        assert type(result) is tuple and len(result) == 1
        return {"y": torch.from_dlpack(result[0])}


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=_gid)
@pytest.mark.parametrize("tokens", [1, 17, 24])
def test_eager_and_jit_with_changed_inputs(device, geometry, tokens):
    run = jax.jit(shared_expert_decode_forward_jax)
    retained = None
    for iteration in range(2):
        operands = make_operands(tokens, device=device, seed=72 + iteration, geometry=geometry)
        before = snapshot(operands)
        if iteration:
            with jax.no_tracing(True):
                actual = execute(run, operands)
        else:
            actual = execute(run, operands)
            assert_bitwise(snapshot(execute(shared_expert_decode_forward_jax, operands)), snapshot(actual), "eager vs JIT")
        assert actual["y"].shape == (tokens, geometry.HIDDEN)
        assert actual["y"].dtype == torch.bfloat16
        assert_in_band(actual["y"], reference(operands), f"{_gid(geometry)} T{tokens}")
        assert_bitwise(snapshot(operands), before, "input mutation")
        if retained is not None:
            assert_bitwise(snapshot(retained[0]), retained[1], "previous result mutation")
        retained = actual, snapshot(actual)


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=_gid)
def test_replay_and_batch_invariance(device, geometry):
    operands = make_operands(24, device=device, seed=81, geometry=geometry)
    run = jax.jit(shared_expert_decode_forward_jax)
    batch = assert_replays(lambda: execute(run, operands))
    target = operands["x"][16:17]
    alone = snapshot(execute(run, dict(operands, x=target)))
    assert_bitwise(alone, {"y": batch["y"][16:17]}, "alone vs 24 rows")

    base_x = torch.cat((operands["x"][:15], target))
    base = snapshot(execute(run, dict(operands, x=base_x)))
    inserted = torch.cat((base_x[:7], operands["x"][23:24], base_x[7:]))
    grown = snapshot(execute(run, dict(operands, x=inserted)))
    assert_bitwise({"y": torch.cat((grown["y"][:7], grown["y"][8:]))}, base, "16 to 17 rows")

    reversed_batch = snapshot(execute(run, dict(operands, x=operands["x"].flip(0))))
    assert_bitwise({"y": reversed_batch["y"].flip(0)}, batch, "reordered rows")
    duplicated = snapshot(execute(run, dict(operands, x=torch.cat((target, target)))))
    assert_bitwise({"y": duplicated["y"][:1]}, alone, "first duplicate")
    assert_bitwise({"y": duplicated["y"][1:]}, alone, "second duplicate")

    neighbors = make_operands(23, device=device, seed=82, geometry=geometry)["x"]
    mixed = snapshot(execute(run, dict(operands, x=torch.cat((neighbors[:7], target, neighbors[7:])))))
    assert_bitwise({"y": mixed["y"][7:8]}, alone, "changed neighbors")


@pytest.mark.parametrize("invalid", ["rows", "dtype", "weight_shape"])
def test_invalid_operands_decline_before_launch(device, invalid):
    operands = make_operands(25 if invalid == "rows" else 1, device=device, geometry=GEOMETRIES[0])
    if invalid == "dtype":
        operands["x"] = operands["x"].float()
    elif invalid == "weight_shape":
        operands["w_down"] = operands["w_down"][:, :-1].contiguous()
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        execute(jax.jit(shared_expert_decode_forward_jax), operands)
