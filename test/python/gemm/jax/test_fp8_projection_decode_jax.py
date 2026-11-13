# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native JAX FP8 decode projection: numerics, buffer reuse, and row invariance."""

from functools import partial

import pytest

jax = pytest.importorskip("jax", minversion="0.9.1")
torch = pytest.importorskip("torch")

from cudnn.ops import fp8_projection_decode_forward_jax
from determinism_utils import assert_bitwise, assert_replays, snapshot
from gemm.frost.projection_decode.fp8_projection_decode_utils import CLASSES, cid, error_over_bound, integer_operands, make_operands, reference

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


@pytest.fixture
def device():
    from cudnn.frost.buffers import cutedsl_requirement_error

    requirement = cutedsl_requirement_error("JAX FP8 projection decode tests", floor=(4, 9, 0))
    if requirement:
        pytest.skip(requirement)
    target = next((d for d in jax.local_devices() if d.platform == "gpu" and str(getattr(d, "compute_capability", "")).startswith("10.")), None)
    if target is None:
        pytest.skip("FP8 projection decode requires an SM10x GPU")
    with jax.default_device(target), torch.cuda.device(target.local_hardware_id):
        yield torch.device("cuda", target.local_hardware_id)


def execute(run, operands, *, scalar_scale=False):
    with torch.cuda.device(operands["weight"].device):
        scale = operands["scale"].reshape(()) if scalar_scale else operands["scale"]
        values = tuple(jax.dlpack.from_dlpack(value) for value in (operands["weight"], operands["activation"], scale))
        result = jax.block_until_ready(run(*values))
        assert type(result) is tuple and len(result) == 1
        return {"out": torch.from_dlpack(result[0])}


def in_bound(actual, operands, numerics):
    ratio, nonfinite = error_over_bound(actual["out"], operands, numerics.SPLIT_K)
    assert nonfinite == 0
    assert ratio <= 1.0, f"{cid(numerics)}: numerical error is {ratio} times the existing bound"


@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
@pytest.mark.parametrize("rows", [1, 9, 17, 32])
def test_eager_and_jit_with_changed_inputs(device, numerics, rows):
    operation = partial(fp8_projection_decode_forward_jax, split_k=numerics.SPLIT_K)
    run = jax.jit(operation)
    retained = None
    for iteration in range(2):
        operands = make_operands(numerics.GEOM, rows, device, seed=91 + iteration, kind="wide")
        before = snapshot(operands)
        if iteration:
            with jax.no_tracing(True):
                actual = execute(run, operands)
        else:
            actual = execute(run, operands)
            assert_bitwise(snapshot(execute(operation, operands)), snapshot(actual), "eager vs JIT")
        assert actual["out"].shape == (rows, numerics.GEOM.OUT_FEATURES)
        assert actual["out"].dtype == torch.bfloat16
        in_bound(actual, operands, numerics)
        assert_bitwise(snapshot(operands), before, "input mutation")
        if retained is not None:
            assert_bitwise(snapshot(retained[0]), retained[1], "previous result mutation")
        retained = actual, snapshot(actual)


@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_exact_integer_projection_and_scalar_scale(device, numerics):
    operands = integer_operands(numerics.GEOM, 3, device, seed=102)
    run = jax.jit(partial(fp8_projection_decode_forward_jax, split_k=numerics.SPLIT_K))
    actual = snapshot(execute(run, operands, scalar_scale=True))
    expected, _ = reference(operands)
    assert_bitwise(actual, {"out": expected.bfloat16()}, "exact-integer projection")


@pytest.mark.parametrize("numerics", CLASSES, ids=cid)
def test_replay_and_batch_invariance(device, numerics):
    operands = make_operands(numerics.GEOM, 32, device, seed=112, kind="wide")
    run = jax.jit(partial(fp8_projection_decode_forward_jax, split_k=numerics.SPLIT_K))
    batch = assert_replays(lambda: execute(run, operands))
    in_bound(batch, operands, numerics)
    target = operands["activation"][7:8]
    alone = snapshot(execute(run, dict(operands, activation=target)))
    assert_bitwise(alone, {"out": batch["out"][7:8]}, "alone vs 32 rows")

    for rows in (8, 16, 31):
        activation = operands["activation"][:rows]
        base = snapshot(execute(run, dict(operands, activation=activation)))
        inserted = torch.cat((activation[:3], target, activation[3:]))
        grown = snapshot(execute(run, dict(operands, activation=inserted)))
        assert_bitwise({"out": torch.cat((grown["out"][:3], grown["out"][4:]))}, base, f"{rows} to {rows + 1} rows")
        assert_bitwise({"out": grown["out"][3:4]}, alone, "inserted duplicate")

    reverse = torch.arange(31, -1, -1, device=device)
    reversed_batch = snapshot(execute(run, dict(operands, activation=operands["activation"][reverse].contiguous())))
    assert_bitwise({"out": reversed_batch["out"].flip(0)}, batch, "reordered rows")


@pytest.mark.parametrize("invalid", ["rows", "dtype", "scale_shape", "split_k"])
def test_invalid_operands_decline_before_launch(device, invalid):
    numerics = CLASSES[0]
    operands = make_operands(numerics.GEOM, 33 if invalid == "rows" else 1, device)
    if invalid == "dtype":
        operands["activation"] = operands["activation"].bfloat16()
    elif invalid == "scale_shape":
        operands["scale"] = operands["scale"].repeat(2)
    split_k = 3 if invalid == "split_k" else numerics.SPLIT_K
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        execute(jax.jit(partial(fp8_projection_decode_forward_jax, split_k=split_k)), operands)
