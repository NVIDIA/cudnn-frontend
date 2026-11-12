# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native JAX scan bindings against the independent recurrent FP64 references."""

import os
import subprocess
import sys
from functools import partial

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax", minversion="0.9.1")
import jax.numpy as jnp

import cudnn
from linear_attention.reference_gdn import gdn_reference, gdp_reference
from linear_attention.reference_gdn2 import gdn2_reference
from linear_attention.reference_mamba2 import mamba2_reference

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

OPERATIONS = {"gdn": "gated_delta_net", "gdn2": "gated_delta_net_v2", "gdp": "gated_delta_product", "mamba2": "mamba2"}
DELTA_GRADIENTS = dict(q="dq", k="dk", v="dv", g="dg", beta="dbeta", w="dw", initial_state="d_initial_state", a_log="d_a_log", dt_bias="d_dt_bias")
MAMBA_GRADIENTS = dict(q="dc", k="db", v="dx", g="ddt", a="da", initial_state="d_initial_state", d="dd", dt_bias="d_dt_bias", z="dz")


def inputs(kind, *, dtype=jnp.bfloat16, bounds=(0, 17, 17, 35), fused=False, seed=3, heads=None, value_dim=64):
    rng = np.random.default_rng(seed)
    total, sequences = bounds[-1], len(bounds) - 1
    hq, hk, hv = heads or ((1, 1, 2) if kind == "mamba2" or fused else (2, 1, 1))
    ho, dk, dv = max(hq, hv), 128 if kind == "mamba2" or fused else 64, value_dim
    n = 3 if kind == "gdp" else 1

    def normal(shape, dtype=dtype, scale=0.15, offset=0):
        return jnp.asarray(rng.normal(offset, scale, shape), dtype)

    q, k, v = normal((total, hq, dk)), normal((total * n, hk, dk)), normal((total * n, hv, dv))
    k = (k.astype(jnp.float32) / jnp.linalg.norm(k.astype(jnp.float32), axis=-1, keepdims=True)).astype(dtype)
    shape = (total, ho, dk) if kind == "gdn2" else (total, ho)
    values = dict(q=q, k=k, v=v, g=normal(shape, jnp.float32) if fused else jnp.asarray(-rng.uniform(0.01, 0.1, shape), jnp.float32))
    values["initial_state"] = normal((sequences, ho, dv, dk), dtype if fused else jnp.float32, 0.03)
    options = dict(batch_invariant=True, checkpoint_every_n_tokens={"gdn": 64, "gdn2": 16, "gdp": 64, "mamba2": 128}[kind] if fused else 0)
    if kind == "mamba2":
        values.update(g=normal(shape, jnp.float32, 0.5, -2), a=-jnp.ones((ho,), jnp.float32))
        if fused:
            values.update(dt_bias=normal((ho,), jnp.float32), d=normal((ho, dv), jnp.float32), z=normal((total, ho, dv)))
            options.update(dt_min=0.08, dt_max=0.16)
    else:
        beta_shape = (total, ho, dk) if kind == "gdn2" else (total * n, ho)
        beta_dtype = dtype if kind == "gdn2" else jnp.float32
        values["beta"] = normal(beta_shape, beta_dtype) if fused else jnp.asarray(rng.uniform(0.2, 0.8, beta_shape), beta_dtype)
        if kind == "gdn2":
            values["w"] = jnp.asarray(rng.uniform(0.2, 0.8, (total, ho, dv)), dtype)
        if kind == "gdp":
            options["num_householder"] = n
        if fused:
            bias_shape = (ho, dk) if kind == "gdn2" else (ho,)
            values.update(a_log=normal((ho,), jnp.float32, offset=-2), dt_bias=normal(bias_shape, jnp.float32))
            options.update(safe_gate=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True, use_qk_l2norm_in_kernel=True)
            if kind == "gdn2":
                options["gate_lower_bound"] = -0.1
    return values, jnp.asarray(bounds, jnp.int32), options


def invoke(kind, values, cu, *, explicit=False, output_final_state=True, **options):
    name = OPERATIONS[kind]
    fn = getattr(cudnn.ops, name + "_forward_jax") if explicit else getattr(cudnn.jax, name)
    kwargs = dict(initial_state=values.get("initial_state"), output_final_state=output_final_state, **options)
    if kind == "mamba2":
        return fn(
            values["v"], values["g"], values["a"], values["k"], values["q"], cu, D=values.get("d"), dt_bias=values.get("dt_bias"), z=values.get("z"), **kwargs
        )
    names = ("q", "k", "v", "g", "beta", "w") if kind == "gdn2" else ("q", "k", "v", "g", "beta")
    return fn(*(values[name] for name in names), cu, a_log=values.get("a_log"), dt_bias=values.get("dt_bias"), **kwargs)


def gradients(kind, result, values):
    fields = MAMBA_GRADIENTS if kind == "mamba2" else DELTA_GRADIENTS
    assert all(getattr(result, field) is None for name, field in fields.items() if name not in values)
    return {name: getattr(result, fields[name]) for name in values}


def torch_array(value):
    host = np.asarray(value)
    if host.dtype.name == "bfloat16":
        return torch.from_numpy(host.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.from_numpy(host.copy())


def reference(kind, values, bounds, do, ds, **options):
    leaves = {name: torch_array(value).double().requires_grad_() for name, value in values.items()}
    q, k, v, g = (leaves[name].unsqueeze(0) for name in ("q", "k", "v", "g"))
    if options.get("use_qk_l2norm_in_kernel", False):
        q, k = (x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt() for x in (q, k))
    kwargs = dict(initial_state=leaves.get("initial_state"), cu_seqlens=torch.tensor(bounds, dtype=torch.int32))
    if kind == "mamba2":
        out, state = mamba2_reference(
            q,
            k,
            v,
            g,
            A=leaves["a"],
            D=leaves.get("d"),
            dt_bias=leaves.get("dt_bias"),
            z=leaves["z"].unsqueeze(0) if "z" in leaves else None,
            scale=1.0,
            dt_softplus=options.get("dt_softplus", True),
            dt_min=options.get("dt_min", 0.0),
            dt_max=options.get("dt_max", float("inf")),
            **kwargs,
        )
    else:
        beta = leaves["beta"].unsqueeze(0)
        if options.get("safe_gate", False):
            a, bias = leaves.get("a_log"), leaves.get("dt_bias")
            if kind == "gdn2":
                g = options.get("gate_lower_bound", -5.0) * torch.sigmoid(a.exp()[:, None] * (g + bias))
            else:
                g = -a.exp() * torch.nn.functional.softplus(g + bias)
        if options.get("gate_domain") == "linear":
            g = g.clamp_min(1e-10).log()
        if options.get("use_beta_sigmoid_in_kernel", False):
            beta = beta.sigmoid() * (2 if options.get("allow_neg_eigval", False) else 1)
        kwargs["scale"] = options.get("scale")
        if kind == "gdn2":
            out, state = gdn2_reference(q, k, v, g, beta, leaves["w"].unsqueeze(0), **kwargs)
        elif kind == "gdp":
            out, state = gdp_reference(q, k, v, g, beta, num_householder=options["num_householder"], **kwargs)
        else:
            out, state = gdn_reference(q, k, v, g, beta, **kwargs)
    seed_out, seed_state = torch_array(do).double().unsqueeze(0), torch_array(ds).double()
    grads = torch.autograd.grad((out, state), tuple(leaves.values()), (seed_out, seed_state))
    return {"out": out[0], "final_state": state}, dict(zip(leaves, grads))


def assert_close(got, want, tolerance):
    got = np.asarray(got, dtype=np.float64)
    want = want.detach().numpy() if isinstance(want, torch.Tensor) else np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape
    assert np.all(np.isfinite(got))
    error = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
    assert error < tolerance, f"relative L2 error {error:g} >= {tolerance:g}"


def upstream(values, seed=8):
    rng = np.random.default_rng(seed)
    state = values["initial_state"]
    shape = (values["q"].shape[0], state.shape[1], values["v"].shape[-1])
    return jnp.asarray(rng.normal(size=shape), values["q"].dtype), jnp.asarray(rng.normal(size=state.shape), state.dtype)


@pytest.mark.parametrize("kind", OPERATIONS)
@pytest.mark.parametrize(
    "dtype,fused,chain",
    [(jnp.bfloat16, False, False), (jnp.float16, True, False), (jnp.bfloat16, True, True)],
    ids=["bf16-packed", "fp16-fused-checkpoint", "bf16-default-chain"],
)
def test_forward_and_jitted_gradients(kind, dtype, fused, chain):
    bounds = (0, 1537) if chain else (0, 17, 17, 146 if fused else 35)
    values, cu, options = inputs(
        kind, dtype=dtype, bounds=bounds, fused=fused, heads=(1, 1, 1) if chain else None, value_dim=128 if chain and kind != "mamba2" else 64
    )
    options["batch_invariant"] = not chain
    empty_sequences = np.flatnonzero(np.diff(bounds) == 0)
    do, ds = upstream(values)
    run = jax.jit(partial(invoke, kind, **options))
    result = run(values, cu)
    assert isinstance(result, tuple) and len(result) == 2
    expected, expected_grads = reference(kind, values, bounds, do, ds, **options)
    for name, got in zip(("out", "final_state"), result):
        assert_close(got, expected[name], 0.02 if dtype == jnp.bfloat16 else 0.01)
    for index in empty_sequences:
        np.testing.assert_array_equal(result[1][index], values["initial_state"][index])

    def loss(args):
        out, state = run(args, cu)
        return jnp.sum(out.astype(jnp.float32) * do) + jnp.sum(state.astype(jnp.float32) * ds)

    actual = jax.jit(jax.grad(loss))(values)
    for name, got in actual.items():
        assert got.dtype == values[name].dtype
        assert_close(got, expected_grads[name], 0.06 if name == "initial_state" else 0.04 if dtype == jnp.bfloat16 else 0.03)
    for index in empty_sequences:
        np.testing.assert_array_equal(actual["initial_state"][index], ds[index])
    _, _, residual = jax.jit(partial(invoke, kind, explicit=True, **options))(values, cu)
    backward = getattr(cudnn.ops, OPERATIONS[kind] + "_backward_jax")
    explicit = gradients(kind, jax.jit(lambda r, do, ds: backward(r, do, d_final_state=ds))(residual, do, ds), values)
    for name in actual:
        np.testing.assert_array_equal(explicit[name], actual[name])
    changed = {name: value * 0.7 for name, value in values.items()}
    changed_expected, _ = reference(kind, changed, bounds, do, ds, **options)
    for name, got in zip(("out", "final_state"), run(changed, cu)):
        assert_close(got, changed_expected[name], 0.02 if dtype == jnp.bfloat16 else 0.01)


@pytest.mark.parametrize("kind", OPERATIONS)
def test_optional_state_and_output_only_gradient(kind):
    values, cu, options = inputs(kind)
    state_shape = values.pop("initial_state").shape
    run = jax.jit(partial(invoke, kind, output_final_state=False, **options))
    out, state = run(values, cu)
    assert state is None
    actual = jax.jit(jax.grad(lambda args: run(args, cu)[0].astype(jnp.float32).sum()))(values)
    out2, _, residual = invoke(kind, values, cu, explicit=True, output_final_state=False, **options)
    backward = getattr(cudnn.ops, OPERATIONS[kind] + "_backward_jax")
    explicit = gradients(kind, jax.jit(backward)(residual, jnp.ones_like(out2)), values)
    for name in actual:
        np.testing.assert_array_equal(explicit[name], actual[name])
    expected, expected_grads = reference(kind, values, (0, 17, 17, 35), jnp.ones_like(out), jnp.zeros(state_shape, jnp.float32), **options)
    assert_close(out, expected["out"], 0.02)
    for name in actual:
        assert_close(actual[name], expected_grads[name], 0.04)


@pytest.mark.parametrize("kind", OPERATIONS)
def test_state_only_gradient(kind):
    values, cu, options = inputs(kind, fused=True)
    do, ds = upstream(values)
    run = jax.jit(partial(invoke, kind, **options))
    actual = jax.jit(jax.grad(lambda args: jnp.sum(run(args, cu)[1].astype(jnp.float32) * ds)))(values)
    _, expected = reference(kind, values, (0, 17, 17, 35), jnp.zeros_like(do), ds, **options)
    for name, got in actual.items():
        assert got.dtype == values[name].dtype
        assert_close(got, expected[name], 0.06 if name == "initial_state" else 0.04)
    for name in ("q", "d", "z") if kind == "mamba2" else ("q",):
        np.testing.assert_array_equal(actual[name], jnp.zeros_like(values[name]))


@pytest.mark.parametrize("kind", ["gdn", "gdn2", "gdp"])
def test_linear_gate_domain(kind):
    values, cu, options = inputs(kind)
    values["g"] = jnp.exp(values["g"])
    do, ds = upstream(values)
    options["gate_domain"] = "linear"
    run = jax.jit(partial(invoke, kind, **options))
    out, state = run(values, cu)
    expected, expected_grads = reference(kind, values, (0, 17, 17, 35), do, ds, **options)
    assert_close(out, expected["out"], 0.02)
    assert_close(state, expected["final_state"], 0.02)
    actual = jax.jit(jax.grad(lambda args: sum(jnp.sum(x.astype(jnp.float32) * dy) for x, dy in zip(run(args, cu), (do, ds)))))(values)
    for name in actual:
        assert_close(actual[name], expected_grads[name], 0.06 if name == "initial_state" else 0.04)


@pytest.mark.parametrize(
    "kind,invalid,error,message",
    [
        pytest.param("gdn", "offsets", ValueError, "cu_seqlens must be int32 or int64", id="floating-offsets"),
        pytest.param("gdn", "checkpoint", NotImplementedError, "checkpoint_every_n_tokens.*multiple of 64", id="unaligned-checkpoint"),
        pytest.param("gdn2", "write_gate", ValueError, "w must have shape", id="gdn2-write-gate-shape"),
        pytest.param("gdp", "householder", ValueError, "num_householder must be positive", id="zero-householder-count"),
        pytest.param("mamba2", "skip", ValueError, "D must have shape", id="mamba2-skip-shape"),
        pytest.param("gdn", "cpu", NotImplementedError, "requires a CUDA GPU; got cpu", id="cpu-inputs"),
    ],
)
def test_reject_unsupported_inputs(kind, invalid, error, message):
    values, cu, options = inputs(kind)
    match invalid:
        case "offsets":
            cu = cu.astype(jnp.float32)
        case "checkpoint":
            options["checkpoint_every_n_tokens"] = 65
        case "write_gate":
            values["w"] = values["w"][..., :-1]
        case "householder":
            options["num_householder"] = 0
        case "skip":
            values["d"] = jnp.ones((values["v"].shape[1], values["v"].shape[-1] + 1), jnp.float32)
        case "cpu":
            cpu = jax.devices("cpu")[0]
            values, cu = jax.tree.map(lambda value: jax.device_put(value, cpu), (values, cu))
    with pytest.raises(error, match=message):
        invoke(kind, values, cu, explicit=True, **options)


def test_independent_process_and_nondefault_device(monkeypatch, cuda_device):
    values, cu, options = inputs("gdn")
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    run = jax.jit(partial(invoke, "gdn", **options))
    run(values, cu)[0].block_until_ready()
    devices = [device for device in jax.local_devices() if device.platform == "gpu" and str(getattr(device, "compute_capability", "")) in ("10.0", "10.3")]
    if len(devices) < 2:
        pytest.skip("requires two supported GPUs for nondefault-device placement")
    target = next(device for device in devices if device != cuda_device)
    changed = jax.tree.map(lambda x: jax.device_put(x * 0.7, target), values)
    other_cu = jax.device_put(cu, target)
    with jax.default_device(target):
        out, state = run(changed, other_cu)
        grads = jax.jit(jax.grad(lambda args: run(args, other_cu)[0].astype(jnp.float32).sum()))(changed)
    assert out.devices() == state.devices() == {target}
    assert all(value.devices() == {target} for value in grads.values())
    do, ds = upstream(changed)
    expected, _ = reference("gdn", changed, (0, 17, 17, 35), do, ds, **options)
    assert_close(out, expected["out"], 0.02)


def test_execution_without_torch():
    script = r"""
import faulthandler
faulthandler.dump_traceback_later(540, exit=True)
import sys
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("torch import attempted")
sys.meta_path.insert(0, NoTorch())
import jax
import jax.numpy as jnp
import cudnn
device = next(d for d in jax.local_devices() if d.platform == "gpu" and str(d.compute_capability) in ("10.0", "10.3"))
with jax.default_device(device):
    for name in ("gated_delta_net", "gated_delta_net_v2", "gated_delta_product", "mamba2"):
        op = getattr(cudnn.jax, name)
        q = jnp.full((17, 1, 128), 0.01, jnp.bfloat16)
        v = jnp.full((17, 1, 64), 0.02, jnp.bfloat16)
        cu = jnp.asarray([0, 17], jnp.int32)
        g = jnp.full((17, 1), -0.1, jnp.float32)
        beta = jnp.full((17, 1), 0.5, jnp.float32)
        def loss(q):
            if name == "mamba2":
                out, _ = op(v, g, -jnp.ones((1,), jnp.float32), q, q, cu)
            elif name == "gated_delta_net_v2":
                out, _ = op(q, q, v, jnp.broadcast_to(g[..., None], q.shape), jnp.full_like(q, 0.5), jnp.full_like(v, 0.5), cu)
            else:
                options = {"num_householder": 1} if name == "gated_delta_product" else {}
                out, _ = op(q, q, v, g, beta, cu, **options)
            return out.astype(jnp.float32).sum()
        value, grad = jax.jit(jax.value_and_grad(loss))(q)
        jax.block_until_ready((value, grad))
        assert bool(jnp.isfinite(value)) and bool(jnp.all(jnp.isfinite(grad)))
assert "torch" not in sys.modules
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=600, env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"Torch-free execution timed out:\n{error.stdout}\n{error.stderr}")
    assert result.returncode == 0, result.stdout + result.stderr
