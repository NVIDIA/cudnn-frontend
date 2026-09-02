# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic perf-share / support-gap harness for the ``benchmark/e2e`` models.

Each ``<Model>/run_model.py`` builds its model (and wires ``cudnn.fla`` /any other
cuDNN swap), then calls :func:`profile_and_report`. The harness times a fwd+bwd
training step and profiles the CUDA kernels, bucketing self-time by *category*
(linear-attn / full-attn / gemm / norm / misc) and by *backend* (cuDNN / cuBLAS /
torch) so you can see what fraction of a step already runs on cuDNN and what still
falls to cuBLAS/torch — i.e. which op is the next one worth owning.
"""

import collections

import torch

# Kernel-name substrings -> category (the op the kernel implements, NOT the backend
# that runs it). First match wins, so order matters. Backend names like "frost" /
# "cutile" belong in backend() below, not here -- FROST serves the MLP dgrad too, so
# bucketing it as linear-attention would miscount the headline category shares.
_CATEGORIES = (
    ("linear_attn", ("gdn", "delta", "chunk_gated", "wy_fast", "solve", "cumsum", "l2norm", "kda")),
    ("full_attn", ("flash", "fmha", "sdpa", "scaled_dot", "mha", "_attention")),
    ("gemm", ("gemm", "cutlass", "ampere", "sm100_tst", "nvjet", "cublas", "matmul", "wgrad", "dgrad", "tensorop")),
    ("norm", ("rmsnorm", "layernorm", "layer_norm", "rms_norm", "norm")),
    (
        "misc",
        (
            "elementwise",
            "vectorized",
            "silu",
            "swiglu",
            "sigmoid",
            "softplus",
            "add",
            "mul",
            "cast",
            "copy",
            "index",
            "embedding",
            "cross_entropy",
            "softmax",
            "fill",
            "reduce",
            "cat",
        ),
    ),
)


def pick_sm100():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        if 100 <= p.major * 10 + p.minor < 120:  # SM100-family (Blackwell); the fused engine is not on SM120
            return torch.device(f"cuda:{i}")
    raise SystemExit("no SM100-family (Blackwell) device; the fused SwiGLU-MLP engine requires one")


def categorize(name):
    n = name.lower()
    for tag, keys in _CATEGORIES:
        if any(x in n for x in keys):
            return tag
    return "other"


def backend(name):
    n = name.lower()
    if "cudnn" in n or "gdn" in n or "kda" in n or "fort_native" in n or "frost" in n or "cutile" in n:
        return "cuDNN"
    if "nvjet" in n or "cublas" in n or ("cutlass" in n and "gdn" not in n):
        return "cuBLAS"
    return "torch"


def _default_step(model, ids):
    """A fwd+bwd training step for an HF-style ``*ForCausalLM`` model."""
    model(input_ids=ids, labels=ids).loss.backward()


def profile_and_report(model, ids, *, step=_default_step, warmup=3, iters=10, extra_path=None):
    """Warm up, time the best fwd+bwd step (wall), then profile one step's kernels.

    ``extra_path`` is an optional ``() -> str`` printed as an op-path diagnostic
    (e.g. which route ``cudnn.fla`` took). Wall and kernel self-time are reported
    separately — their difference is only an approximate host/overhead gap (they
    come from different iterations), so it is shown only when positive.
    """
    if ids.device.type != "cuda":
        raise ValueError(f"profile_and_report requires CUDA input ids, got {ids.device}")
    device = ids.device
    with torch.cuda.device(device):
        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            step(model, ids)
        torch.cuda.synchronize(device)
    if extra_path is not None:
        print(f"op path: {extra_path()}")

    best = float("inf")
    for _ in range(iters):
        with torch.cuda.device(device):
            model.zero_grad(set_to_none=True)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            step(model, ids)
            e.record()
            torch.cuda.synchronize(device)
            best = min(best, s.elapsed_time(e))

    with torch.cuda.device(device):
        model.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            step(model, ids)
            torch.cuda.synchronize(device)

    cat, be, total = collections.defaultdict(float), collections.defaultdict(float), 0.0
    for ev in prof.key_averages():
        t = ev.self_device_time_total
        if t <= 0:
            continue
        cat[categorize(ev.key)] += t
        be[backend(ev.key)] += t
        total += t

    kernel_ms = total / 1e3
    print("\nfwd+bwd training step (eager):")
    print(f"  wall (min over {iters}):   {best:.3f} ms")
    print(f"  GPU kernel self-time:      {kernel_ms:.3f} ms")
    if best > kernel_ms:  # different iterations -> only an approximate host/overhead gap
        print(f"  approx host/overhead gap:  {best - kernel_ms:.3f} ms ({100 * (best - kernel_ms) / best:.0f}% of wall)")

    print(f"\n{'category':12} {'ms':>9} {'share':>7}")
    print("-" * 30)
    for c in ("linear_attn", "full_attn", "gemm", "norm", "misc", "other"):
        if cat[c] > 0:
            print(f"{c:12} {cat[c] / 1e3:9.3f} {100 * cat[c] / total:6.1f}%")
    print(f"\n{'backend':12} {'ms':>9} {'share':>7}")
    print("-" * 30)
    for b in ("cuDNN", "cuBLAS", "torch"):
        if be[b] > 0:
            print(f"{b:12} {be[b] / 1e3:9.3f} {100 * be[b] / total:6.1f}%")
    print(f"\ncuDNN-owned share of GPU kernel time: {100 * be['cuDNN'] / total:.1f}%")
    return {"wall_ms": best, "kernel_ms": kernel_ms, "category": dict(cat), "backend": dict(be)}
