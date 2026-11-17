# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare identical inputs and cotangents through both JAX SDPA routes."""

from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.L0


def test_missing_ffi_header(monkeypatch, tmp_path):
    from cudnn._cake import loader
    from cudnn.sdpa import jax_runtime

    library = tmp_path / "lib" / "libcudnn.so.9"
    library.parent.mkdir()
    library.touch()
    monkeypatch.setenv("CUDNN_PATH", str(tmp_path))
    monkeypatch.setattr(loader, "_toolkit", lambda: ("/cuda/bin/nvcc", SimpleNamespace(full="test")))
    monkeypatch.setattr(loader, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(jax.ffi, "include_dir", lambda: str(tmp_path / "jax_include"))
    with mock.patch.object(jax_runtime.subprocess, "run") as compile_bridge:
        with pytest.raises(ImportError, match="Missing XLA FFI header") as error:
            jax_runtime._library.__wrapped__()
        compile_bridge.assert_not_called()
    assert str(tmp_path / "jax_include" / "xla/ffi/api/ffi.h") in str(error.value)
    assert "Install compatible JAX and jaxlib" in str(error.value)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("feature", ["plain", "causal", "gqa", "mqa", "cross", "bias", "mask", "padding", "window", "unbatched"])
def test_routes(monkeypatch, dtype, feature):
    rng = np.random.default_rng(42)
    shape = (2, 64, 4, 64)
    kvshape = (2, 96 if feature == "cross" else 64, 1 if feature == "mqa" else 2 if feature == "gqa" else 4, 64)
    q, k, v = [jnp.asarray(rng.normal(size=s) * 0.2, dtype) for s in (shape, kvshape, kvshape)]
    kwargs = dict(implementation="cudnn", return_residual=True, scale=0.125)
    extra = None
    if feature in ("causal", "window"):
        kwargs["is_causal"] = True
    if feature == "window":
        kwargs["local_window_size"] = (15, 0)
    if feature == "padding":
        kwargs.update(query_seq_lengths=jnp.array([57, 64], jnp.int32), key_value_seq_lengths=jnp.array([61, 64], jnp.int32))
    if feature == "mask":
        kwargs["mask"] = jnp.asarray(rng.uniform(size=(1, 4, 64, 64)) > 0.2)
    if feature == "bias":
        extra = jnp.asarray(rng.normal(size=(1, 4, 64, 64)) * 0.1, dtype)
    if feature == "unbatched":
        q, k, v = q[0], k[0], v[0]
    cotangent = jnp.asarray(rng.normal(size=q.shape), dtype)

    def evaluate(q, k, v, bias):
        def attention(q, k, v, bias):
            return jax.nn.dot_product_attention(q, k, v, bias=bias, **kwargs)

        (out, lse), pullback = jax.vjp(attention, q, k, v, bias)
        gradients = pullback((cotangent, jnp.zeros_like(lse)))
        return out, lse, gradients

    results = []
    for setting in (None, "0", "1"):
        if setting is None:
            monkeypatch.delenv("JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND", raising=False)
        else:
            monkeypatch.setenv("JAX_CUDNN_SDPA_USE_PYTHON_FRONTEND", setting)
        jax.clear_caches()
        result = jax.jit(evaluate)(q, k, v, extra)
        results.append(jax.device_get(result))
    for result in results[1:]:
        for actual, expected in zip(jax.tree.leaves(result), jax.tree.leaves(results[0])):
            np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=0.015, rtol=0.03)

    kwargs["implementation"] = "xla"
    jax.clear_caches()
    reference = jax.device_get(jax.jit(evaluate)(q, k, v, extra))
    for actual, expected in zip(jax.tree.leaves((results[-1][0], results[-1][2])), jax.tree.leaves((reference[0], reference[2]))):
        np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), atol=0.015, rtol=0.03)


def test_native_lowering():
    import cudnn

    q = jax.ShapeDtypeStruct((1, 64, 2, 64), jnp.bfloat16)
    text = jax.jit(cudnn.sdpa_jax).lower(q, q, q).as_text()
    assert "cudnn_sdpa_" in text
    assert "callback" not in text.lower()


def test_deterministic_replay():
    import cudnn

    rng = np.random.default_rng(71)
    args = tuple(jnp.asarray(rng.normal(size=(2, 128, 4, 64)) * 0.1, jnp.bfloat16) for _ in range(3))
    cotangent = jnp.asarray(rng.normal(size=args[0].shape), jnp.bfloat16)

    def evaluate(q, k, v, cotangent):
        o, pullback = jax.vjp(lambda q, k, v: cudnn.sdpa_jax(q, k, v, deterministic=True), q, k, v)
        return o, pullback(cotangent)

    run = jax.jit(evaluate)
    expected = jax.device_get(run(*args, cotangent))
    for _ in range(3):
        actual = jax.device_get(run(*(jnp.copy(x) for x in args), cotangent))
        for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            assert np.isfinite(a.astype(np.float32)).all()
            assert a.tobytes() == b.tobytes()

    for index in range(2):
        isolated = jax.device_get(run(*(x[index : index + 1] for x in args), cotangent[index : index + 1]))
        for a, b in zip(jax.tree.leaves(isolated), jax.tree.leaves(expected)):
            assert a.tobytes() == b[index : index + 1].tobytes()
