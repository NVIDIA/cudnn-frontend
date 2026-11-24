# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from cudnn.frost.buffers import cutedsl_requirement_error

pytest.importorskip("cutlass")
requirement_error = cutedsl_requirement_error("DSA JAX tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from cudnn.ops import dsa

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    from cudnn.frost.buffers import cutedsl_arch_requirement_error

    device = jax.local_devices()[0]
    capability = str(getattr(device, "compute_capability", ""))
    if device.platform != "gpu" or not capability.startswith("10."):
        pytest.skip("DSA JAX ops require the first visible GPU to be SM100-family")
    arch_error = cutedsl_arch_requirement_error(tuple(int(x) for x in capability.split(".")))
    if arch_error:
        pytest.skip(arch_error)


def indexer_scores(q, k, weights, sm_scale):
    dots = jnp.einsum("bshd,bkd->bshk", q.astype(jnp.float32), k.astype(jnp.float32))
    return jnp.einsum("bshk,bsh->bsk", jax.nn.relu(dots), weights.astype(jnp.float32) * sm_scale)


def kl_score_gradient(target, predict, valid, grad_scale):
    """Clipped-log KL gradient w.r.t. the scores, as the kernels form it (dsa_reference.ref_indexer_backward)."""
    eps = math.exp(-100.0)
    g = -jnp.maximum(target, eps) * (predict >= eps) * grad_scale
    return jnp.where(valid, g - predict * g.sum(-1, keepdims=True), 0.0)


def assert_close_gradients(actual, expected):
    for got, ref in zip(actual, expected):
        got, ref = np.asarray(got, np.float32).ravel(), np.asarray(ref, np.float32).ravel()
        cosine = got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref))
        rms_rel = np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref**2))
        assert cosine >= 0.999 and rms_rel <= 0.02, (cosine, rms_rel)


def test_indexer_jax_matches_native_math():
    key = jax.random.key(4)
    q = jax.random.normal(key, (1, 16, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (1, 4, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (1, 16, 64), jnp.bfloat16)
    scale = 128**-0.5
    got = dsa.indexer_forward_jax(q, k, weights, sm_scale=scale)
    ref = indexer_scores(q, k[:, :, 0], weights, scale)
    valid = jnp.isfinite(got)
    np.testing.assert_allclose(np.asarray(got[valid]), np.asarray(ref[valid]), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("lengths", [(21, 33), (3, 33)])
def test_indexer_jax_thd_matches_independent_sequences(lengths):
    compressed = [n // 4 for n in lengths]
    # The last prefix pair repeats: an empty sequence.
    cu_q = jnp.array([0, lengths[0], sum(lengths), sum(lengths)], jnp.int32)
    cu_k = jnp.array([0, compressed[0], sum(compressed), sum(compressed)], jnp.int32)
    # One extra NaN row of capacity in each tensor must not be read.
    key = jax.random.key(12)
    q = jax.random.normal(key, (sum(lengths) + 1, 64, 128), jnp.bfloat16).at[-1].set(jnp.nan)
    k = jax.random.normal(jax.random.fold_in(key, 1), (sum(compressed) + 1, 1, 128), jnp.bfloat16).at[-1].set(jnp.nan)
    w = jax.random.normal(jax.random.fold_in(key, 2), q.shape[:2], jnp.bfloat16).at[-1].set(jnp.nan)
    max_k = -(-max(lengths) // 4)
    scale = 128**-0.5
    got, lse = jax.jit(
        lambda x, y, z: dsa.indexer_forward_jax(
            x, y, z, sm_scale=scale, return_lse=True, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max(lengths), max_seqlen_k=max_k
        )
    )(q, k, w)
    assert got.shape == (q.shape[0], -(-max_k // 4) * 4) and lse.shape == (q.shape[0],)

    expected = np.full(got.shape, -np.inf, np.float32)
    qs = ks = 0
    for nq, nk in zip(lengths, compressed):
        scores = indexer_scores(q[None, qs : qs + nq], k[None, ks : ks + nk, 0], w[None, qs : qs + nq], scale)[0]
        causal = 4 * (jnp.arange(nk)[None] + 1) <= jnp.arange(nq)[:, None] + 1
        expected[qs : qs + nq, :nk] = np.asarray(jnp.where(causal, scores, -jnp.inf))
        qs, ks = qs + nq, ks + nk
    valid = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(np.asarray(got)), valid)
    np.testing.assert_allclose(np.asarray(got)[valid], expected[valid], atol=2e-5, rtol=2e-5)
    rows = slice(0, sum(lengths))
    np.testing.assert_allclose(np.asarray(lse)[rows], np.asarray(jax.nn.logsumexp(expected[rows], axis=-1)), atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("dtype, num_cols", [(jnp.float32, 1003), (jnp.bfloat16, 1001)])
def test_indexer_top_k_jax_matches_lax_top_k(dtype, num_cols):
    top_k, rows = 64, 8
    scores = jax.random.normal(jax.random.key(6), (rows, num_cols), jnp.float32).astype(dtype)
    seq_lens = jnp.array([num_cols, num_cols - 1, 700, 65, top_k, 40, 1, 0], jnp.int32)
    got = np.asarray(jax.jit(lambda s, n: dsa.top_k_forward_jax(s, n, top_k=top_k))(scores, seq_lens))
    for row, length in enumerate(np.asarray(seq_lens)):
        if length <= top_k:
            expected = np.concatenate([np.arange(length), np.full(top_k - length, -1)])
            np.testing.assert_array_equal(got[row], expected)
        else:
            _, expected = jax.lax.top_k(scores[row, :length], top_k)
            np.testing.assert_array_equal(np.sort(got[row]), np.sort(np.asarray(expected)))


def test_dense_indexer_backward_jax_matches_reference():
    batch, seqlen_q, seqlen_k, ratio, sm_scale = 1, 256, 64, 4, 128**-0.5
    key = jax.random.key(7)
    q = jax.random.normal(key, (batch, seqlen_q, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (batch, seqlen_k, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (batch, seqlen_q, 64), jnp.bfloat16)
    valid = jnp.arange(seqlen_k)[None, None] < (jnp.arange(seqlen_q)[None, :, None] + 1) // ratio
    attn_score = jnp.where(valid, jax.random.uniform(jax.random.fold_in(key, 3), (batch, seqlen_q, seqlen_k)), 0.0)
    attn_l1norm = attn_score.sum(-1)
    grad_loss = jnp.array([1.5], jnp.float32)

    index_score, index_lse = dsa.indexer_forward_jax(q, k, weights, ratio=ratio, sm_scale=sm_scale, return_lse=True)
    got = dsa.dense_indexer_backward_jax(q, k, weights, attn_score, attn_l1norm, index_score, index_lse, grad_loss, ratio=ratio, sm_scale=sm_scale)

    scores, vjp = jax.vjp(lambda x, y, w: indexer_scores(x, y[:, :, 0], w, sm_scale), q, k, weights)
    predict = jnp.where(valid, jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), axis=-1), 0.0)
    target = attn_score / jnp.maximum(attn_l1norm, 1e-10)[..., None]
    expected = vjp(kl_score_gradient(target, predict, valid, grad_loss[0] / (batch * seqlen_q)))
    assert_close_gradients(got, (expected[0], expected[2], expected[1][:, :, 0]))


@pytest.mark.parametrize("batch,clip_log,seqlen_q", [(1, True, 128), (1, False, 32), (2, False, 32)])
def test_indexer_backward_jax_matches_reference_and_keeps_inputs(batch, clip_log, seqlen_q):
    seqlen_k, topk = 1024, 512
    sm_scale = 128**-0.5 if clip_log else 0.17
    key = jax.random.key(8)
    q = jax.random.normal(key, (batch, seqlen_q, 64, 128), jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (batch, seqlen_k, 1, 128), jnp.bfloat16)
    weights = jax.random.normal(jax.random.fold_in(key, 2), (batch, seqlen_q, 64), jnp.bfloat16)
    indices = jax.vmap(lambda r: jax.random.permutation(r, seqlen_k)[:topk])(jax.random.split(jax.random.fold_in(key, 3), batch * seqlen_q)).reshape(
        batch, seqlen_q, topk
    )
    valid = jnp.arange(topk)[None, None] < jnp.arange(seqlen_q)[None, :, None] * 4
    indices = jnp.where(valid, indices, -1).astype(jnp.int32)
    grad_loss = jnp.array([0.5], jnp.float32)

    def selected_scores(x, y, w):
        gathered = jnp.take_along_axis(y[:, None, :, 0], jnp.maximum(indices, 0)[..., None], axis=2)
        dots = jnp.einsum("bshd,bstd->bsht", x.astype(jnp.float32), gathered.astype(jnp.float32))
        return jnp.einsum("bsht,bsh->bst", jax.nn.relu(dots), w.astype(jnp.float32) * sm_scale)

    scores, vjp = jax.vjp(selected_scores, q, k, weights)
    index_score = jnp.where(valid, jax.nn.softmax(jnp.where(valid, scores, -jnp.inf), axis=-1), 0.0)
    attn_score = jnp.where(valid, jax.nn.softmax(jax.random.normal(jax.random.fold_in(key, 4), scores.shape), axis=-1), 0.0)
    attn_score = attn_score / jnp.maximum(attn_score.sum(-1, keepdims=True), 1e-8) * 0.7
    attn_score_before = np.asarray(attn_score)

    got = jax.jit(lambda *xs: dsa.indexer_backward_jax(*xs, sm_scale=sm_scale, clip_log=clip_log))(q, k, weights, attn_score, index_score, indices, grad_loss)

    np.testing.assert_array_equal(np.asarray(attn_score), attn_score_before)
    if clip_log:
        score_grad = kl_score_gradient(attn_score, index_score, valid, grad_loss[0] / (batch * seqlen_q))
    else:
        score_grad = (index_score * attn_score.sum(-1, keepdims=True) - attn_score) * grad_loss[0] / (batch * seqlen_q)
    expected = vjp(score_grad)
    assert_close_gradients(got, (expected[0], expected[2], expected[1][:, :, 0]))


@pytest.mark.parametrize(
    "q_lengths,k_lengths",
    [((3, 0, 5, 2, 0), (2, 3, 4, 0, 0)), ((17, 0, 33, 2), (5, 0, 9, 0)), ((0, 0), (0, 0))],
)
@pytest.mark.parametrize("clip_log", [False, True])
def test_indexer_backward_jax_thd_matches_independent_kl(q_lengths, k_lengths, clip_log):
    # The oracle computes selected logits directly in JAX; no native indexer forward is involved.
    cu_q_np = np.cumsum((0,) + q_lengths, dtype=np.int32)
    cu_k_np = np.cumsum((0,) + k_lengths, dtype=np.int32)
    tq, tk, scale, upstream = int(cu_q_np[-1]) + 3, int(cu_k_np[-1]) + 2, 0.13, 0.7
    cu_q, cu_k = jnp.asarray(cu_q_np), jnp.asarray(cu_k_np)
    key = jax.random.key(21)
    q = jax.random.normal(key, (tq, 64, 128), jnp.bfloat16).at[cu_q_np[-1] :].set(jnp.nan)
    k = jax.random.normal(jax.random.fold_in(key, 1), (tk, 1, 128), jnp.bfloat16).at[cu_k_np[-1] :].set(jnp.nan)
    weights = (jax.random.normal(jax.random.fold_in(key, 2), (tq, 64), jnp.bfloat16) * 0.125).at[cu_q_np[-1] :].set(jnp.nan)
    ids_np = np.full((tq, 512), -1, np.int32)
    for b, (nq, nk) in enumerate(zip(q_lengths, k_lengths)):
        for row in range(int(cu_q_np[b]), int(cu_q_np[b + 1])):
            # First row stays all-invalid; final slot is positive but outside this segment.
            if row != cu_q_np[b]:
                ids_np[row, :nk] = np.arange(nk, dtype=np.int32)[::-1]
            ids_np[row, -1] = nk
    # Capacity rows carry apparently valid IDs and poisoned probabilities; prefixes must override them.
    ids_np[cu_q_np[-1] :, 0] = 0
    ids = jnp.asarray(ids_np)
    target_raw = jax.random.uniform(jax.random.fold_in(key, 3), (tq, 512)) + 0.1
    target = jnp.full((tq, 512), jnp.nan, jnp.float32)
    predict = jnp.full((tq, 512), jnp.nan, jnp.float32)

    def local_logits(x, y, w, b):
        qs, qe, ks, ke = int(cu_q_np[b]), int(cu_q_np[b + 1]), int(cu_k_np[b]), int(cu_k_np[b + 1])
        local_ids = ids[qs:qe]
        valid = (local_ids >= 0) & (local_ids < ke - ks)
        active = jnp.any(valid, -1)
        x = jnp.where(active[:, None, None], x[qs:qe], 0).astype(jnp.float32)
        w = jnp.where(active[:, None], w[qs:qe], 0).astype(jnp.float32)
        gathered = y[ks:ke, 0][jnp.clip(local_ids, 0, ke - ks - 1)].astype(jnp.float32)
        logits = jnp.einsum("qht,qh->qt", jax.nn.relu(jnp.einsum("qhd,qtd->qht", x, gathered)), w * scale)
        finite_logits = jnp.where(active[:, None], jnp.where(valid, logits, -jnp.inf), 0)
        return finite_logits, valid, active

    for b, (nq, nk) in enumerate(zip(q_lengths, k_lengths)):
        if nq == 0 or nk == 0:
            continue
        qs, qe = int(cu_q_np[b]), int(cu_q_np[b + 1])
        logits, valid, active = local_logits(q, k, weights, b)
        mass = jnp.where(jnp.arange(nq) % 2, 0.37, 1.4)[:, None]
        t = jnp.where(valid, target_raw[qs:qe], 0)
        t = t / jnp.maximum(t.sum(-1, keepdims=True), 1e-8) * mass
        target = target.at[qs:qe].set(jnp.where(valid, t, jnp.nan))
        predict = predict.at[qs:qe].set(jnp.where(valid & active[:, None], jax.nn.softmax(logits), jnp.nan))

    def reference_loss(x, y, w):
        result = jnp.zeros((), jnp.float32)
        for b, (nq, nk) in enumerate(zip(q_lengths, k_lengths)):
            if nq == 0 or nk == 0:
                continue
            qs, qe = int(cu_q_np[b]), int(cu_q_np[b + 1])
            logits, valid, _ = local_logits(x, y, w, b)
            t = jnp.where(valid, target[qs:qe], 0)
            if clip_log:
                # Matches the default clipped-log objective, independently differentiated.
                p = jnp.where(valid, jax.nn.softmax(logits), 1.0)
                term = -jnp.maximum(t, math.exp(-100)) * jnp.log(jnp.maximum(p, math.exp(-100)))
            else:
                term = -t * jnp.where(valid, jax.nn.log_softmax(logits), 0.0)
            result += jnp.where(valid, term, 0).sum()
        return result * (upstream / tq)

    inputs = (q, k, weights, target, predict, ids, cu_q, cu_k)
    before = [np.asarray(x).copy() for x in inputs]
    backward = jax.jit(
        lambda x, y, w, t, p, i, cq, ck: dsa.indexer_backward_jax(
            x,
            y,
            w,
            t,
            p,
            i,
            jnp.asarray(upstream, jnp.float32),
            sm_scale=scale,
            clip_log=clip_log,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
            max_seqlen_q=max(max(q_lengths), 1),
            max_seqlen_k=max(max(k_lengths), 1),
        )
    )
    got = backward(*inputs)
    dq_ref, dk_ref, dw_ref = jax.grad(reference_loss, argnums=(0, 1, 2))(q, k, weights)
    expected = (dq_ref, dw_ref, dk_ref[:, 0])
    assert [x.shape for x in got] == [q.shape, weights.shape, (tk, 128)]
    assert [x.dtype for x in got] == [jnp.bfloat16, jnp.bfloat16, jnp.float32]
    for actual, ref in zip(got, expected):
        a, r = np.asarray(actual, np.float32), np.asarray(ref, np.float32)
        assert np.isfinite(a).all()
        if np.any(r):
            assert_close_gradients((actual,), (ref,))
        else:
            np.testing.assert_array_equal(a, np.zeros_like(a))
    for actual, original in zip(inputs, before):
        # BF16 NaNs are deliberately retained; byte equality also checks their payloads.
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint8), original.view(np.uint8))
    np.testing.assert_array_equal(np.asarray(got[0][cu_q_np[-1] :]), 0)
    np.testing.assert_array_equal(np.asarray(got[1][cu_q_np[-1] :]), 0)
    np.testing.assert_array_equal(np.asarray(got[2][cu_k_np[-1] :]), 0)
    for b, (nq, nk) in enumerate(zip(q_lengths, k_lengths)):
        if nq:
            row = int(cu_q_np[b])
            np.testing.assert_array_equal(np.asarray(got[0][row]), 0)
            np.testing.assert_array_equal(np.asarray(got[1][row]), 0)
        if nq and not nk:
            np.testing.assert_array_equal(np.asarray(got[0][cu_q_np[b] : cu_q_np[b + 1]]), 0)
        if not nq and nk:
            np.testing.assert_array_equal(np.asarray(got[2][cu_k_np[b] : cu_k_np[b + 1]]), 0)

    # dQ/dW have no cross-CTA reduction. FP32 dK uses atomics and has no bitwise replay guarantee.
    replay = backward(*(jnp.array(x, copy=True) for x in inputs))
    for first, second in zip(got[:2], replay[:2]):
        np.testing.assert_array_equal(np.asarray(first).view(np.uint8), np.asarray(second).view(np.uint8))
    np.testing.assert_allclose(np.asarray(replay[2]), np.asarray(got[2]), rtol=1e-5, atol=1e-6)


def test_sparse_attention_jax_forward_and_backward():
    key = jax.random.key(5)
    q = jax.random.normal(key, (1, 64, 512), jnp.bfloat16) / jnp.sqrt(512)
    kv = jax.random.normal(jax.random.fold_in(key, 1), (640, 512), jnp.bfloat16)
    indices = jnp.arange(640, dtype=jnp.int32)[None]
    lengths = jnp.array([640], jnp.int32)
    sinks = jnp.zeros((64,), jnp.float32)
    got, _, lse, _ = dsa.sparse_attention_forward_jax(q, kv, indices, lengths, sinks, softmax_scale=1.0)
    logits = jnp.einsum("hd,kd->hk", q[0].astype(jnp.float32), kv.astype(jnp.float32))
    maximum = jnp.maximum(jnp.max(logits, axis=-1), sinks)
    probabilities = jnp.exp(logits - maximum[:, None])
    ref = jnp.einsum("hk,kd->hd", probabilities, kv.astype(jnp.float32))
    ref /= (jnp.sum(probabilities, axis=-1) + jnp.exp(sinks - maximum))[:, None]
    np.testing.assert_allclose(np.asarray(got[0], np.float32), np.asarray(ref), atol=1e-2, rtol=2e-2)

    gradients = dsa.sparse_attention_backward_jax(q, kv, got, jnp.ones_like(got), lse, sinks, indices, lengths, softmax_scale=1.0)

    def reference_loss(x, y, s):
        reference_logits = jnp.einsum("thd,kd->thk", x.astype(jnp.float32), y.astype(jnp.float32))
        reference_max = jnp.maximum(jnp.max(reference_logits, axis=-1), s)
        exponentials = jnp.exp(reference_logits - reference_max[..., None])
        denominator = jnp.sum(exponentials, axis=-1) + jnp.exp(s - reference_max)
        output = jnp.einsum("thk,kd->thd", exponentials, y.astype(jnp.float32)) / denominator[..., None]
        return output.sum()

    reference_gradients = jax.grad(reference_loss, argnums=(0, 1, 2))(q, kv, sinks)
    np.testing.assert_allclose(np.asarray(gradients[0], np.float32), np.asarray(reference_gradients[0], np.float32), atol=4e-2, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(gradients[1], np.float32), np.asarray(reference_gradients[1], np.float32), atol=3e-3, rtol=2e-2)
    np.testing.assert_allclose(np.asarray(gradients[2]), np.asarray(reference_gradients[2]), atol=2e-5, rtol=2e-2)


def test_indexer_backward_jax_unclipped_underflow_and_replay():
    q = jnp.ones((1, 1, 64, 128), jnp.bfloat16)
    k = jnp.broadcast_to(jnp.arange(1, 5, dtype=jnp.bfloat16)[None, :, None, None], (1, 4, 1, 128))
    weights = jnp.ones((1, 1, 64), jnp.bfloat16)
    ids = jnp.full((1, 1, 512), -1, jnp.int32).at[..., :2].set(jnp.array([0, 1], jnp.int32))
    teacher = jnp.zeros((1, 1, 512), jnp.float32).at[..., :2].set(0.35)
    # Stable softmax([-200, 0]) underflows the first probability but not its KL gradient.
    probabilities = jnp.zeros_like(teacher).at[..., 1].set(1.0)
    upstream = jnp.array([1.7], jnp.float32)
    original = [np.asarray(x).copy() for x in (q, k, weights, teacher, probabilities, ids, upstream)]
    native = jax.jit(lambda clip, *xs: dsa.indexer_backward_jax(*xs, sm_scale=0.25, clip_log=clip), static_argnums=0)
    clipped = native(True, q, k, weights, teacher, probabilities, ids, upstream)
    default = dsa.indexer_backward_jax(q, k, weights, teacher, probabilities, ids, upstream, sm_scale=0.25)
    raw = native(False, q, k, weights, teacher, probabilities, ids, upstream)
    replay = native(False, q, k, weights, teacher, probabilities, ids, upstream)
    for x, y in zip(clipped, default):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    for x, y in zip(raw, replay):
        np.testing.assert_array_equal(np.asarray(x).view(np.uint8), np.asarray(y).view(np.uint8))
    score_grad = probabilities * teacher.sum(-1, keepdims=True) - teacher
    _, vjp = jax.vjp(lambda x, y, w: indexer_scores(x, y[:, :, 0], w, 0.25), q, k, weights)
    full_grad = jnp.zeros((1, 1, 4), jnp.float32).at[..., :2].set(score_grad[..., :2] * upstream[0])
    expected = vjp(full_grad)
    assert_close_gradients(raw, (expected[0], expected[2], expected[1][:, :, 0]))
    assert float(jnp.max(jnp.abs(raw[0].astype(jnp.float32)))) > 0
    assert float(jnp.max(jnp.abs(clipped[0].astype(jnp.float32)))) == 0
    for before, after in zip(original, (q, k, weights, teacher, probabilities, ids, upstream)):
        np.testing.assert_array_equal(before, np.asarray(after))
