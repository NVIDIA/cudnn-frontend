# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from functools import partial

import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("JAX KDA tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

jax = pytest.importorskip("jax", minversion="0.9.1")
import jax.numpy as jnp
import numpy as np

from cudnn.jax import (
    kimi_delta_attention as kda,
    kimi_delta_attention_fwd as fwd,
    kimi_delta_attention_bwd as bwd,
)

pytestmark = [pytest.mark.L0, pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]


def inputs(
    dk=64,
    dv=64,
    hq=1,
    hk=1,
    hv=1,
    dtype=jnp.bfloat16,
    bounds=(0, 17, 17, 35),
    gates=False,
    state_dtype=jnp.float32,
):
    rng = np.random.default_rng(3)
    t, ho, n = bounds[-1], max(hq, hv), len(bounds) - 1

    def normal(shape, dtype=dtype, scale=0.15):
        return jnp.asarray(rng.normal(0, scale, shape), dtype)

    q, k, v = normal((t, hq, dk)), normal((t, hk, dk)), normal((t, hv, dv))
    k = (k.astype(jnp.float32) / jnp.linalg.norm(k.astype(jnp.float32), axis=-1, keepdims=True)).astype(dtype)
    g = normal((t, ho, dk), jnp.float32) if gates else jnp.asarray(-rng.uniform(0.01, 0.1, (t, ho, dk)), jnp.float32)
    beta = normal((t, ho), jnp.float32) if gates else jnp.asarray(rng.uniform(0.1, 0.9, (t, ho)), jnp.float32)
    state = normal((n, ho, dv, dk), state_dtype, 0.03)
    a, dt = (normal((ho,), jnp.float32), normal((ho, dk), jnp.float32)) if gates else (None, None)
    return (q, k, v, g, beta, state, a, dt), jnp.asarray(bounds, jnp.int32)


def reference(
    args,
    bounds,
    *,
    safe_gate=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    use_qk_l2norm_in_kernel=False,
    gate_lower_bound=None,
    gate_domain="log",
    scale=None,
    **unused,
):
    q, k, v, g, beta, state, a, dt = jax.tree.map(lambda x: x.astype(jnp.float32), args)
    ho, dk, dv = g.shape[1], q.shape[-1], v.shape[-1]
    if use_qk_l2norm_in_kernel:
        q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-12)
        k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-12)
    q, k, v = (jnp.repeat(x, ho // x.shape[1], axis=1) for x in (q, k, v))
    q = q * (dk**-0.5 if scale is None else scale)
    if safe_gate:
        g = (-5.0 if gate_lower_bound is None else gate_lower_bound) * jax.nn.sigmoid(
            (jnp.exp(a)[:, None] if a is not None else 1) * (g + (dt if dt is not None else 0))
        )
    if use_beta_sigmoid_in_kernel:
        beta = jax.nn.sigmoid(beta) * (2 if allow_neg_eigval else 1)

    def step(s, xs):
        qt, kt, vt, gt, bt = xs
        s = s * (gt if gate_domain == "linear" else jnp.exp(gt))[:, None, :]
        delta = vt - jnp.einsum("hvk,hk->hv", s, kt)
        s = s + bt[:, None, None] * delta[:, :, None] * kt[:, None, :]
        return s, jnp.einsum("hvk,hk->hv", s, qt)

    outs, finals = [], []
    for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
        s = state[i] if state is not None else jnp.zeros((ho, dv, dk), jnp.float32)
        s, out = jax.lax.scan(step, s, tuple(x[start:end] for x in (q, k, v, g, beta)))
        outs.append(out)
        finals.append(s)
    return jnp.concatenate(outs), jnp.stack(finals)


def run(args, cu, **options):
    q, k, v, g, beta, state, a, dt = args
    return kda(
        q,
        k,
        v,
        g,
        beta,
        cu,
        initial_state=state,
        a_log=a,
        dt_bias=dt,
        output_final_state=True,
        **options,
    )


@pytest.fixture
def built_plans(monkeypatch):
    from cudnn.linear_attention import jax_api
    from cudnn.linear_attention.frost import kda_engine

    build = kda_engine.build_kda
    plans = []

    def record(graph):
        plans.append(build(graph))
        return plans[-1]

    jax_api.build_call.cache_clear()
    monkeypatch.setattr(kda_engine, "build_kda", record)
    return plans


def assert_close(actual, expected, tol=0.04):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert np.all(np.isfinite(actual))
    ratio = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-10)
    assert ratio < tol, f"relative L2 error {ratio} >= {tol}"


@pytest.mark.parametrize(
    "dims,heads,dtype,checkpoint,gates,invariant",
    [
        ((64, 64), (1, 1, 1), jnp.bfloat16, 0, False, True),
        ((64, 128), (2, 1, 1), jnp.float16, 16, False, False),
        ((128, 64), (1, 2, 2), jnp.bfloat16, 0, True, False),
        ((128, 128), (1, 1, 1), jnp.bfloat16, 16, True, True),
    ],
)
def test_forward_and_jitted_gradients(dims, heads, dtype, checkpoint, gates, invariant):
    bounds = (0, 17, 17, 35)
    args, cu = inputs(*dims, *heads, dtype, bounds, gates)
    options = dict(
        checkpoint_every_n_tokens=checkpoint,
        safe_gate=gates,
        use_beta_sigmoid_in_kernel=gates,
        allow_neg_eigval=gates,
        use_qk_l2norm_in_kernel=gates,
        batch_invariant=invariant,
    )
    invoke = jax.jit(partial(run, **options))
    o, fs = invoke(args, cu)
    ro, rfs = reference(args, bounds, **options)
    assert_close(o, ro, 0.02)
    assert_close(fs, rfs, 0.02)
    np.testing.assert_array_equal(fs[1], args[5][1])
    rng = np.random.default_rng(8)
    do = jnp.asarray(rng.normal(size=o.shape), dtype)
    ds = jnp.asarray(rng.normal(size=fs.shape), jnp.float32)

    def loss(args, reference_mode=False):
        o, s = reference(args, bounds, **options) if reference_mode else invoke(args, cu)
        return jnp.sum(o.astype(jnp.float32) * do) + jnp.sum(s.astype(jnp.float32) * ds)

    actual = jax.jit(jax.grad(loss))(args)
    expected = jax.grad(partial(loss, reference_mode=True))(args)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert_close(got, want, 0.06)
    np.testing.assert_array_equal(actual[5][1], ds[1])
    _, _, residual = jax.jit(
        lambda args: fwd(
            *args[:5],
            cu,
            initial_state=args[5],
            a_log=args[6],
            dt_bias=args[7],
            output_final_state=True,
            **options,
        )
    )(args)
    explicit = jax.jit(lambda r, do, ds: bwd(r, do, d_final_state=ds))(residual, do, ds)
    for got, want in zip(jax.tree.leaves(explicit), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(got, want)
    if checkpoint:
        valid = sum((end - start + 15) // 16 for start, end in zip(bounds, bounds[1:]))
        np.testing.assert_array_equal(residual.checkpoints[valid:], 0)


@pytest.mark.parametrize("checkpoint", [0, 16])
@pytest.mark.parametrize(
    "dims,heads,bounds,gates,state_dtype,with_state",
    [
        ((64, 64), (1, 1, 1), (0, 256), False, jnp.float32, True),
        ((64, 128), (2, 1, 1), (0, 0, 129, 768), True, jnp.bfloat16, True),
        ((128, 64), (1, 2, 2), (0, 129, 129, 768), False, jnp.float32, False),
    ],
)
def test_long_sequence_forward_and_gradients(checkpoint, dims, heads, bounds, gates, state_dtype, with_state, built_plans):
    args, cu = inputs(*dims, *heads, bounds=bounds, gates=gates, state_dtype=state_dtype)
    if not with_state:
        args = (*args[:5], None, *args[6:])
    if not gates:
        args = (*args[:3], jnp.full_like(args[3], -1e-4), *args[4:])
    options = dict(
        checkpoint_every_n_tokens=checkpoint,
        safe_gate=gates,
        gate_lower_bound=-0.01 if gates else None,
        use_beta_sigmoid_in_kernel=gates,
        use_qk_l2norm_in_kernel=gates,
    )
    actual = jax.jit(partial(run, **options))(args, cu)
    expected = reference(args, bounds, **options)
    for got, want in zip(actual, expected):
        assert_close(got, want, 0.02)

    def loss(args, reference_mode=False):
        o, state = reference(args, bounds, **options) if reference_mode else run(args, cu, **options)
        return o.astype(jnp.float32).sum() + state.astype(jnp.float32).sum()

    actual_grads = jax.jit(jax.grad(loss))(args)
    expected_grads = jax.jit(jax.grad(partial(loss, reference_mode=True)))(args)
    for got, want in zip(jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads)):
        assert_close(got, want, 0.06)
    assert {plan.node.node_type.name for plan in built_plans} == {"KDA", "KDA_BWD"}
    assert all(plan.chain for plan in built_plans), "JAX must preserve Frost's automatic piece-chain selection"


@pytest.mark.parametrize("total", [35, 768])
def test_repeated_calls_and_dynamic_boundaries(total):
    args, cu = inputs(bounds=(0, 16, total))
    invoke = jax.jit(lambda args, cu: run(args, cu))
    first = invoke(args, cu)
    changed = tuple(x * 0.7 if x is not None else None for x in args)
    bounds = (0, 9, total)
    second = invoke(changed, jnp.asarray(bounds, jnp.int32))
    for got, want in zip(second, reference(changed, bounds)):
        assert_close(got, want, 0.02)
    again = invoke(args, cu)
    for old, new in zip(first, again):
        np.testing.assert_array_equal(old, new)
    assert invoke._cache_size() == 1


def test_no_state_and_state_only_loss():
    args, cu = inputs(bounds=(0, 35))
    no_state = (*args[:5], None, None, None)
    o, fs = run(no_state, cu)
    for got, want in zip((o, fs), reference(no_state, (0, 35))):
        assert_close(got, want, 0.02)
    got = jax.jit(jax.grad(lambda args: run(args, cu)[1].sum()))(args)
    want = jax.grad(lambda args: reference(args, (0, 35))[1].sum())(args)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
        assert_close(a, b, 0.06)
    plain = jax.jit(lambda q: kda(q, *args[1:5], cu))(args[0])
    assert plain[1] is None


def test_torch_free_execution():
    script = """
import sys
class RejectTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("torch import attempted: " + fullname)
sys.meta_path.insert(0, RejectTorch())
import jax
import jax.numpy as jnp
from cudnn.jax import kimi_delta_attention as kda
jax.config.update("jax_enable_x64", True)
q = jnp.ones((80, 1, 64), jnp.bfloat16) * 0.05
b = jnp.ones((80, 1), jnp.float32) * 0.5
for domain, dtype, cadence in (("log", jnp.int32, 0), ("linear", jnp.int64, 64)):
    g = jnp.full(q.shape, -0.1 if domain == "log" else 0.9, jnp.float32)
    cu = jnp.array([0, 80], dtype)
    f = jax.jit(jax.value_and_grad(lambda q: kda(
        q, q, q, g, b, cu, gate_domain=domain, checkpoint_every_n_tokens=cadence
    )[0].astype(jnp.float32).sum()))
    jax.block_until_ready(f(q))
assert "torch" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(checkpoint_every_n_tokens=17),
        dict(checkpoint_every_n_tokens=-16),
        dict(checkpoint_every_n_tokens=2**31),
        dict(checkpoint_every_n_tokens=16.5),
        dict(gate_domain="invalid"),
        dict(gate_domain="linear", safe_gate=True),
        dict(allow_neg_eigval=True),
        dict(gate_lower_bound=-6),
    ],
)
def test_reject_unsupported_options(options):
    args, cu = inputs()
    with pytest.raises((ValueError, NotImplementedError)):
        run(args, cu, **options)


@pytest.mark.parametrize(
    "gate_dtype,beta_dtype,state_dtype,parameter",
    [
        (jnp.bfloat16, jnp.bfloat16, jnp.bfloat16, "a"),
        (jnp.float16, jnp.bfloat16, jnp.float32, "dt"),
        (jnp.float32, jnp.float32, jnp.bfloat16, "none"),
    ],
)
def test_optional_gate_parameters_and_dtypes(gate_dtype, beta_dtype, state_dtype, parameter):
    args, cu = inputs(bounds=(0, 17), gates=True, state_dtype=state_dtype)
    args = (
        *args[:3],
        args[3].astype(gate_dtype),
        args[4].astype(beta_dtype),
        args[5],
        args[6].astype(jnp.bfloat16) if parameter == "a" else None,
        args[7].astype(jnp.float16) if parameter == "dt" else None,
    )
    opts = dict(safe_gate=True, gate_lower_bound=-2.0, use_beta_sigmoid_in_kernel=True)

    def loss(args, ref=False):
        o, s = reference(args, (0, 17), **opts) if ref else run(args, cu, **opts)
        return jnp.sum(o.astype(jnp.float32)) + jnp.sum(s.astype(jnp.float32))

    actual = jax.jit(jax.grad(loss))(args)
    expected = jax.grad(partial(loss, ref=True))(args)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert_close(a, b, 0.06)


def test_recurrent_scan_gradients():
    args, cu = inputs(bounds=(0, 32))
    segments = tuple(x.reshape((2, 16, *x.shape[1:])) for x in args[:5])
    boundaries = jnp.asarray([0, 16], jnp.int32)

    def chained(initial_state, q):
        def step(state, chunk):
            out, final = kda(
                *chunk,
                boundaries,
                initial_state=state,
                output_final_state=True,
                batch_invariant=True,
            )
            return final, out

        state, out = jax.lax.scan(step, initial_state, (q, *segments[1:]))
        return out.astype(jnp.float32).sum() + state.sum()

    def whole(initial_state, q):
        out, state = reference((q.reshape(args[0].shape), *args[1:5], initial_state, None, None), (0, 32))
        return out.sum() + state.sum()

    got = jax.jit(jax.value_and_grad(chained, argnums=(0, 1)))(args[5], segments[0])
    expected = jax.value_and_grad(whole, argnums=(0, 1))(args[5], segments[0])
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        assert_close(a, b, 0.06)


@pytest.mark.parametrize("total", [35, 768])
def test_command_buffer_replay_and_concurrent_dispatch(total):
    from concurrent.futures import ThreadPoolExecutor

    args, cu = inputs(bounds=(0, total))
    invoke = jax.jit(
        partial(run, checkpoint_every_n_tokens=16),
        compiler_options={"xla_gpu_enable_command_buffer": "CUSTOM_CALL", "xla_gpu_graph_min_graph_size": 1},
    )
    jax.block_until_ready(invoke(args, cu))
    variants = [tuple(x * factor if x is not None else None for x in args) for factor in (0.5, 0.8, 1.0, 1.2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke, a, cu) for a in variants]
        for a, future in zip(variants, futures):
            for got, want in zip(future.result(), reference(a, (0, total))):
                assert_close(got, want, 0.02)


def test_forward_ad_rejected():
    args, cu = inputs(bounds=(0, 17))
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jvp(lambda q: kda(q, *args[1:5], cu)[0], (args[0],), (jnp.ones_like(args[0]),))


def test_shape_inference():
    args, cu = inputs()
    abstract = jax.eval_shape(lambda args, cu: fwd(*args[:5], cu, output_final_state=True, checkpoint_every_n_tokens=16), args, cu)
    output, state, residual = abstract
    assert output.shape == (35, 1, 64)
    assert output.dtype == jnp.bfloat16
    assert state.shape == (3, 1, 64, 64)
    assert state.dtype == jnp.float32
    assert residual.checkpoints.shape == (5, 1, 64, 64)
    assert residual.checkpoints.dtype == jnp.bfloat16


def test_helper_types_not_exported():
    import cudnn.jax

    for name in ("KdaResidual", "KdaGradients"):
        assert name not in cudnn.jax.__all__
        assert not hasattr(cudnn.jax, name)


def test_bad_metadata():
    args, cu = inputs()
    with pytest.raises(ValueError, match="beta must have shape"):
        kda(*args[:4], jnp.ones((35,), jnp.float32), cu)
    with pytest.raises(ValueError, match="cu_seqlens must be int32"):
        kda(*args[:5], cu.astype(jnp.float32))
    with pytest.raises(ValueError, match="matching float16 or bfloat16"):
        kda(args[0].astype(jnp.float32), *args[1:5], cu)
    _, _, residual = fwd(*args[:5], cu)
    with pytest.raises(ValueError, match="doutput"):
        bwd(residual, jnp.ones((35, 1, 64), jnp.float32))
    with pytest.raises(ValueError, match="output_final_state"):
        bwd(residual, jnp.ones((35, 1, 64), jnp.bfloat16), d_final_state=jnp.ones((3, 1, 64, 64), jnp.float32))


@pytest.mark.parametrize("schedule", ["uncut", "warmup", "chain"])
@pytest.mark.parametrize(
    "domain,offset_dtype,checkpoint,safe_gate,gate_dtype",
    [
        ("log", jnp.int64, 0, False, jnp.float32),
        ("linear", jnp.int32, 16, False, jnp.bfloat16),
        ("log", jnp.int32, 48, True, jnp.float32),
        ("linear", jnp.int64, 64, False, jnp.float16),
        ("linear", jnp.int32, 32, False, jnp.float32),
    ],
    ids=["int64", "linear", "coarse_safe", "combined", "linear_fp32"],
)
def test_extended_options_forward_backward(schedule, domain, offset_dtype, checkpoint, safe_gate, gate_dtype, built_plans):
    bounds = dict(uncut=(0, 81, 81, 177), warmup=(0, 49, 49, 97), chain=(0, 129, 129, 3073))[schedule]
    dtype = jnp.float16 if domain == "linear" and offset_dtype == jnp.int64 else jnp.bfloat16
    args, _ = inputs(dv=128 if schedule == "chain" else 64, dtype=dtype, bounds=bounds, gates=safe_gate)
    if domain == "linear":
        args = (*args[:3], jnp.exp(8 * args[3]).astype(gate_dtype), *args[4:])
    if safe_gate and schedule == "warmup":
        args = (*args[:5], None, *args[6:])
    options = dict(checkpoint_every_n_tokens=checkpoint, batch_invariant=schedule == "uncut", safe_gate=safe_gate)
    if domain == "linear":
        options["gate_domain"] = domain

    with jax.enable_x64():
        cu = jnp.asarray(bounds, offset_dtype)
        assert cu.dtype == offset_dtype
        actual = run(args, cu, **options)
        expected = reference(args, bounds, **options)
        for got, want in zip(actual, expected):
            assert_close(got, want, 0.02)
        rng = np.random.default_rng(19)
        do = jnp.asarray(rng.normal(size=actual[0].shape), dtype)
        ds = jnp.asarray(rng.normal(size=actual[1].shape), jnp.float32)

        def loss(args, cu, ref=False):
            o, state = reference(args, bounds, **options) if ref else run(args, cu, **options)
            return jnp.sum(o.astype(jnp.float32) * do) + jnp.sum(state.astype(jnp.float32) * ds)

        differentiated = jax.jit(jax.grad(loss))
        actual_grads = differentiated(args, cu)
        expected_grads = jax.jit(jax.grad(partial(loss, ref=True)))(args, cu)
        for got, want in zip(jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads)):
            assert_close(got, want, 0.06)
        out, state, residual = jax.jit(
            lambda args, cu: fwd(*args[:5], cu, initial_state=args[5], a_log=args[6], dt_bias=args[7], output_final_state=True, **options)
        )(args, cu)
        assert residual.primals[5].dtype == offset_dtype
        explicit = jax.jit(lambda r: bwd(r, do, d_final_state=ds))(residual)
        for got, want in zip(jax.tree.leaves(explicit), jax.tree.leaves(actual_grads)):
            np.testing.assert_array_equal(got, want)
        for got, want in zip((out, state), actual):
            np.testing.assert_array_equal(got, want)
        if checkpoint:
            assert residual.checkpoints.shape == (bounds[-1] // checkpoint + len(bounds) - 1, 1, args[2].shape[-1], 64)
            valid = sum((end - start + checkpoint - 1) // checkpoint for start, end in zip(bounds, bounds[1:]))
            np.testing.assert_array_equal(residual.checkpoints[valid:], 0)
        changed = tuple(x * 0.9 if x is not None else None for x in args)
        repeated = differentiated(changed, cu)
        expected_repeated = jax.jit(jax.grad(partial(loss, ref=True)))(changed, cu)
        for got, want in zip(jax.tree.leaves(repeated), jax.tree.leaves(expected_repeated)):
            assert_close(got, want, 0.06)
        assert differentiated._cache_size() == 1
    assert {plan.node.node_type.name for plan in built_plans} == {"KDA", "KDA_BWD"}
    for plan in built_plans:
        assert plan.chain == (schedule == "chain")
        assert plan.split == (schedule == "warmup")


@pytest.mark.parametrize("prep", [False, True], ids=["direct", "prep"])
@pytest.mark.parametrize("checkpoint", [0, 48])
def test_value_split_forward_and_gradients(prep, checkpoint, built_plans):
    from cudnn.frost.device import multiprocessor_count

    heads = 1 if prep else multiprocessor_count(0) // 2
    span = 128 * max(1, checkpoint // 16)
    bounds = (0, span, span, 3 * span + 1) if prep else (0, span + 1)
    args, cu = inputs(dv=128, hq=heads, hk=heads, hv=heads, dtype=jnp.bfloat16 if prep else jnp.float16, bounds=bounds, gates=prep)
    options = dict(checkpoint_every_n_tokens=checkpoint, safe_gate=prep, use_qk_l2norm_in_kernel=prep, use_beta_sigmoid_in_kernel=prep)
    actual = jax.jit(partial(run, **options))(args, cu)
    expected = reference(args, bounds, **options)
    for got, want in zip(actual, expected):
        assert_close(got, want, 0.02)
    rng = np.random.default_rng(23)
    do = jnp.asarray(rng.normal(size=actual[0].shape), args[0].dtype)
    ds = jnp.asarray(rng.normal(size=actual[1].shape), jnp.float32)

    def loss(args, ref=False):
        out, state = reference(args, bounds, **options) if ref else run(args, cu, **options)
        return jnp.sum(out.astype(jnp.float32) * do) + jnp.sum(state.astype(jnp.float32) * ds)

    actual_grads = jax.jit(jax.grad(loss))(args)
    expected_grads = jax.jit(jax.grad(partial(loss, ref=True)))(args)
    for got, want in zip(jax.tree.leaves(actual_grads), jax.tree.leaves(expected_grads)):
        assert_close(got, want, 0.06)
    assert {plan.node.node_type.name for plan in built_plans} == {"KDA", "KDA_BWD"}
    for plan in built_plans:
        if plan.node.node_type.name == "KDA":
            assert plan.dv_split and not plan.chain and not plan.split
            assert plan.prep == prep
            assert plan.tiles_per_head == 2
