# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw-byte replay and sequence isolation for JAX recurrent training calls."""

from functools import lru_cache

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax", minversion="0.9.1")
import jax.numpy as jnp

import cudnn
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from test_linear_attention_jax import OPERATIONS, assert_close, gradients, inputs, invoke, reference, torch_array, upstream

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]
TOKEN_NAMES = {"q", "k", "v", "g", "beta", "w", "z"}


def samples(kind, lengths, *, dtype=jnp.bfloat16, fused=True, heads=None, seed=19):
    result = []
    for index, length in enumerate(lengths):
        values, _, options = inputs(kind, dtype=dtype, bounds=(0, length), fused=fused, heads=heads, seed=seed + index)
        if result:
            for name in values.keys() - TOKEN_NAMES - {"initial_state"}:
                values[name] = result[0][0][name]
        result.append((values, *upstream(values, seed=seed + index + 10)))
    return result, options


def pack(selected):
    values = {
        name: jnp.concatenate([sample[0][name] for sample in selected]) if name in TOKEN_NAMES or name == "initial_state" else selected[0][0][name]
        for name in selected[0][0]
    }
    bounds = np.cumsum([0] + [sample[0]["q"].shape[0] for sample in selected], dtype=np.int32)
    return values, jnp.asarray(bounds), jnp.concatenate([sample[1] for sample in selected]), jnp.concatenate([sample[2] for sample in selected])


@lru_cache(maxsize=None)
def runner(kind, static_options):
    options = dict(static_options)
    backward = getattr(cudnn.ops, OPERATIONS[kind] + "_backward_jax")

    @jax.jit
    def run(values, cu, do, ds):
        out, state, residual = invoke(kind, values, cu, explicit=True, **options)
        grads = gradients(kind, backward(residual, do, d_final_state=ds), values)
        return {"out": out, "final_state": state, **{"d" + name: grad for name, grad in grads.items()}}

    return run


def execute(kind, packed, options):
    fresh = jax.tree.map(lambda value: value.copy(), packed)
    result = runner(kind, tuple(sorted(options.items())))(*fresh)
    return snapshot({name: torch_array(value) for name, value in result.items()})


def check_reference(kind, packed, options, actual):
    values, cu, do, ds = packed
    expected, grads = reference(kind, values, np.asarray(cu), do, ds, **options)
    expected.update({"d" + name: value for name, value in grads.items()})
    for name, want in expected.items():
        tolerance = 0.06 if name == "dinitial_state" else 0.04 if name.startswith("d") else 0.02
        assert_close(actual[name].float().numpy(), want, tolerance)


def tensor_inputs(packed):
    values, cu, do, ds = packed
    return {name: torch_array(value) for name, value in {**values, "cu": cu, "do": do, "d_final_state": ds}.items()}


@pytest.mark.parametrize("kind", OPERATIONS)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bf16-fprop-bprop", "fp16-fprop-bprop"])
def test_replay_and_reference(kind, dtype, cuda_device):
    selected, options = samples(kind, (129, 0, 17), dtype=dtype)
    packed = pack(selected)
    before = snapshot(tensor_inputs(packed))
    with torch.cuda.device(cuda_device.local_hardware_id):
        actual = assert_replays(lambda: execute(kind, packed, options), repeats=4)
    check_reference(kind, packed, options, actual)
    assert_bitwise(tensor_inputs(packed), before, "inputs changed")
    assert_bitwise({"state": actual["final_state"][1:2]}, {"state": torch_array(selected[1][0]["initial_state"])}, "empty sequence")
    assert_bitwise({"ds": actual["dinitial_state"][1:2]}, {"ds": torch_array(selected[1][2])}, "empty sequence backward")


def target_result(kind, result, selected, slot):
    offset = sum(sample[0]["q"].shape[0] for sample in selected[:slot])
    length = selected[slot][0]["q"].shape[0]
    n = 3 if kind == "gdp" else 1
    target = {}
    for name, value in result.items():
        if name in ("final_state", "dinitial_state"):
            target[name] = value[slot : slot + 1]
        elif name == "out" or name.startswith("d") and name[1:] in TOKEN_NAMES:
            factor = n if name in ("dk", "dv", "dbeta") else 1
            target[name] = value[offset * factor : (offset + length) * factor]
        else:
            target[name] = value
    return target


@pytest.mark.parametrize("kind", OPERATIONS)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bf16-fprop-bprop", "fp16-fprop-bprop"])
def test_batch_invariance(kind, dtype):
    selected, options = samples(kind, (65, 0, 17, 17), dtype=dtype)
    target, empty, neighbor, changed = selected
    packed_target = pack([target])
    values, cu, do, ds = packed_target
    _, expected_grads = reference(kind, values, np.asarray(cu), do, ds, **options)
    shared = ("da", "dd", "ddt_bias") if kind == "mamba2" else ("da_log", "ddt_bias")
    expected = {name: value for name, value in execute(kind, packed_target, options).items() if name not in shared}
    for layout, slot in (([target, empty, neighbor], 0), ([neighbor, target, empty], 1), ([target, target], 1), ([changed, empty, target], 2)):
        # Parameter gradients sum over sequences. Isolate the target's fixed loss.
        isolated = [(values, do if i == slot else jnp.zeros_like(do), ds if i == slot else jnp.zeros_like(ds)) for i, (values, do, ds) in enumerate(layout)]
        actual = target_result(kind, execute(kind, pack(isolated), options), isolated, slot)
        sequence_results = {name: value for name, value in actual.items() if name not in shared}
        assert_bitwise(sequence_results, expected, f"target slot {slot}, lengths {[x[0]['q'].shape[0] for x in layout]}")
        # Shared reductions repeat for fixed shapes; repacking can change their summation order.
        for name in shared:
            assert_close(actual[name].float().numpy(), expected_grads[name[1:]], 0.04 if dtype == jnp.bfloat16 else 0.03)


@pytest.mark.parametrize("kind", OPERATIONS, ids=lambda kind: kind + "-bf16-fprop-bprop")
def test_scheduling_boundary(kind, cuda_device):
    heads = cuda_device.core_count // 4 + 1
    selected, options = samples(kind, (129, 17), heads=(heads, heads, heads), fused=False)
    packed = pack(selected)
    with torch.cuda.device(cuda_device.local_hardware_id):
        assert_replays(lambda: execute(kind, packed, options), repeats=4)


@pytest.mark.parametrize("kind", OPERATIONS, ids=lambda kind: kind + "-bf16-fprop-bprop")
def test_golden(kind, cuda_device):
    selected, options = samples(kind, (17, 3))
    packed = pack(selected)
    with torch.cuda.device(cuda_device.local_hardware_id):
        actual = assert_replays(lambda: execute(kind, packed, options), repeats=3)
        check_reference(kind, packed, options, actual)
        from cutlass.base_dsl.version_info import CUDA_VERSION

        name = f"{kind}-jax{jax.__version__}-cu{CUDA_VERSION.major}.{CUDA_VERSION.minor}-fprop-bprop"
        assert_golden(name, tensor_inputs(packed), actual)
