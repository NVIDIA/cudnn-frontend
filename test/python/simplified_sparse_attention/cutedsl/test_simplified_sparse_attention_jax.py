# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public JAX forward and all five reverse-mode gradients."""

from functools import lru_cache
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cudnn.ops import simplified_sparse_attention_jax
from ssa_reference import check_reference, pack, sample
from test_simplified_sparse_attention import supported_gpu

pytestmark = [pytest.mark.L0, pytest.mark.usefixtures("supported_gpu")]


def test_public_jax_export():
    from cudnn.simplified_sparse_attention import simplified_sparse_attention_jax as package_export

    assert package_export is simplified_sparse_attention_jax


class _TorchDLPack:
    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, **kwargs):
        # JAX can change the current device before requesting the capsule.
        with torch.cuda.device(self.tensor.device):
            return self.tensor.__dlpack__(**kwargs)


def jax_inputs(inputs):
    tensors, metadata, upstream = inputs
    caller = torch.cuda.current_device()

    def convert(t):
        return jax.dlpack.from_dlpack(_TorchDLPack(t))

    try:
        values = tuple(convert(t) for t in tensors)
        bounds = tuple(convert(t) for t in (metadata.key_starts, metadata.key_ends, metadata.query_valid, metadata.loss_denominator))
        return values, bounds, convert(upstream)
    finally:
        torch.cuda.set_device(caller)


def torch_outputs(result, device):
    caller = torch.cuda.current_device()
    try:
        outputs = []
        for x in result:
            torch.cuda.set_device(device)
            outputs.append(torch.from_dlpack(x))
        return tuple(outputs)
    finally:
        torch.cuda.set_device(caller)


@lru_cache(maxsize=None)
def runner(topk=512, coefficient=0.1, triangular=False, backward=True):
    @jax.jit
    def train(*arrays):
        tensors, metadata, upstream = arrays[:5], arrays[5:9], arrays[9]

        def operation(*values):
            return simplified_sparse_attention_jax(*values, *metadata, topk=topk, loss_coeff=coefficient, triangular_scores=triangular)

        if not backward:
            return operation(*tensors)
        (output, loss), pullback = jax.vjp(operation, *tensors)
        gradients = pullback((upstream, jnp.ones_like(loss)))
        return output, loss, *gradients

    return train


def execute(inputs, topk=512, coefficient=0.1, triangular=False, backward=True):
    caller = torch.cuda.current_device()
    try:
        tensors, metadata, upstream = jax_inputs(inputs)
        result = jax.block_until_ready(runner(topk, coefficient, triangular, backward)(*tensors, *metadata, upstream))
        return dict(zip(("output", "indexer_loss", "dq", "dk", "dv", "dq_idx", "dk_idx"), torch_outputs(result, inputs[0][0].device)))
    finally:
        torch.cuda.set_device(caller)


@pytest.mark.parametrize("heads", [16, 32, 96], ids=lambda n: f"bf16-h{n}")
@pytest.mark.parametrize("topk", [512, 1024, 2048], ids=lambda n: f"k{n}")
def test_jit_forward_backward(heads, topk):
    inputs = pack([sample(65, heads), sample(17, heads, seed=31)])
    before = [t.clone() for t in (*inputs[0], inputs[2])]
    actual = execute(inputs, topk)
    check_reference(inputs, actual, True, topk)
    for a, b in zip((*inputs[0], inputs[2]), before):
        assert torch.equal(a, b)


def test_selected_support():
    tensors, upstream = sample(1025)
    tensors[0], tensors[3] = tensors[0][:65], tensors[3][:65]
    _, metadata, _ = pack([sample(65)])
    metadata.key_ends.fill_(1025)
    inputs = tensors, metadata, upstream[:65]
    check_reference(inputs, execute(inputs), True)


def test_triangular_and_zero_loss():
    inputs = pack([sample(17)])
    check_reference(inputs, execute(inputs, triangular=True), True)
    zero = execute(inputs, coefficient=0)
    for name in ("indexer_loss", "dq_idx", "dk_idx"):
        assert torch.count_nonzero(zero[name]) == 0


@pytest.mark.parametrize("change", ["dtype", "heads", "topk", "bounds", "denominator", "empty"])
def test_invalid_inputs(change):
    tensors, metadata, _ = jax_inputs(pack([sample(8)]))
    tensors, metadata, topk = list(tensors), list(metadata), 512
    if change == "dtype":
        tensors[0] = tensors[0].astype(jnp.float32)
    elif change == "heads":
        tensors[0] = tensors[0][:, :8]
    elif change == "topk":
        topk = 32
    elif change == "bounds":
        metadata[0] = metadata[0][:-1]
    elif change == "denominator":
        metadata[3] = metadata[3][0]
    else:
        tensors[0] = tensors[0][:0]
    with pytest.raises((ValueError, TypeError)):
        jax.jit(lambda *x: simplified_sparse_attention_jax(*x, *metadata, topk=topk))(*tensors)


@pytest.mark.parametrize("captured", [False, True])
def test_mixed_architecture_rejected_before_launch(monkeypatch, captured):
    from types import SimpleNamespace
    from cudnn.simplified_sparse_attention import jax_api

    tensors, metadata, _ = jax_inputs(pack([sample(8)]))
    monkeypatch.setattr(jax, "local_devices", lambda **kw: [SimpleNamespace(compute_capability=cc) for cc in ("10.0", "10.3")])

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid placement reached the kernel")

    monkeypatch.setattr(jax_api, "_attention", forbidden)
    with pytest.raises(ValueError, match="only SM100 or only SM103"):
        if captured:
            jax.jit(lambda: simplified_sparse_attention_jax(*tensors, *metadata))()
        else:
            jax.jit(lambda *x: simplified_sparse_attention_jax(*x, *metadata))(*tensors)


def test_captured_jit_uses_native_buffers(monkeypatch):
    inputs = pack([sample(19)])
    tensors, metadata, _ = jax_inputs(inputs)

    def forbidden(*args, **kwargs):
        raise AssertionError("JAX execution allocated through PyTorch")

    with monkeypatch.context() as patch:
        for name in ("empty", "zeros", "empty_like", "zeros_like"):
            patch.setattr(torch, name, forbidden)
        result = jax.block_until_ready(jax.jit(lambda: simplified_sparse_attention_jax(*tensors, *metadata))())
    actual = dict(zip(("output", "indexer_loss"), torch_outputs(result, inputs[0][0].device)))
    check_reference(inputs, actual, False)


def test_dense_workspace_index_limit():
    shapes = ((65536, 16, 256), (65536, 1, 256), (65536, 1, 256), (65536, 128), (65536, 128), (65536,), (65536,), (65536,), (1,))
    dtypes = (jnp.bfloat16,) * 5 + (jnp.int32, jnp.int32, jnp.bool_, jnp.float32)
    inputs = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in zip(shapes, dtypes)]
    with pytest.raises(ValueError, match="Dense score workspace exceeds"):
        jax.eval_shape(lambda *x: simplified_sparse_attention_jax(*x, triangular_scores=True), *inputs)


@pytest.mark.parametrize("fail", [False, True])
def test_dlpack_device_context(monkeypatch, fail):
    inputs = pack([sample(3)])
    current = [7]
    seen = []

    def set_device(device):
        current[0] = torch.device(device).index if isinstance(device, torch.device) else device

    def exchange(device):
        previous = current[0]
        set_device(device)
        return previous

    def export(t, **kwargs):
        assert current[0] == t.device.index
        seen.append(t)
        if fail:
            raise BufferError("export failed")
        return t

    def convert(value):
        # Model JAX stream setup changing the current CUDA device.
        current[0] = 3
        return value.__dlpack__()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch.cuda, "_exchange_device", exchange)
    monkeypatch.setattr(torch.cuda, "_maybe_exchange_device", exchange)
    monkeypatch.setattr(torch.Tensor, "__dlpack__", export)
    monkeypatch.setattr(jax.dlpack, "from_dlpack", convert)
    if fail:
        with pytest.raises(BufferError, match="export failed"):
            jax_inputs(inputs)
    else:
        values, bounds, upstream = jax_inputs(inputs)
        assert len(seen) == 10
        assert current[0] == 7

        def reverse(t):
            assert current[0] == inputs[0][0].device.index
            current[0] = 3
            return t

        monkeypatch.setattr(torch, "from_dlpack", reverse)
        torch_outputs((*values, *bounds, upstream), inputs[0][0].device)
    assert current[0] == 7


@pytest.mark.parametrize("fail", [False, True])
def test_execute_restores_device(monkeypatch, fail):
    import sys

    module = sys.modules[__name__]
    inputs = pack([sample(3)])
    current = [7]

    def invoke(*args):
        current[0] = 3
        if fail:
            raise RuntimeError("execution failed")
        return ()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: current.__setitem__(0, device))
    monkeypatch.setattr(module, "jax_inputs", lambda inputs: ((), (), None))
    monkeypatch.setattr(module, "runner", lambda *args: invoke)
    monkeypatch.setattr(module, "torch_outputs", lambda *args: ())
    if fail:
        with pytest.raises(RuntimeError, match="execution failed"):
            execute(inputs)
    else:
        assert execute(inputs) == {}
    assert current[0] == 7


def test_dlpack_with_another_gpu_current():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    with torch.cuda.device(0):
        inputs = pack([sample(17)])
    with torch.cuda.device(1):
        actual = execute(inputs)
        assert torch.cuda.current_device() == 1
    with torch.cuda.device(0):
        check_reference(inputs, actual, True)
