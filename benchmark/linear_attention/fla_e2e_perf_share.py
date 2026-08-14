# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end perf-share / support-gap analysis for a hybrid linear-attention LM.

Builds a Qwen3-Next-style hybrid Gated DeltaNet language model (flash-linear-attention's
model, mostly linear-attention layers + a few full-attention layers + SwiGLU MLP),
calls ``cudnn.fla.accelerate_fla()`` so the linear-attention op runs on cuDNN, does a
fwd+bwd training step, and profiles the CUDA time by category (linear-attn / full-attn /
gemm / norm / misc) and by backend (cuDNN / cuBLAS / torch) so you can see what fraction
of a step already runs on cuDNN and what still falls to torch/cuBLAS.

Requires ``flash-linear-attention`` and a cuDNN build with the linear-attention engines
(SM100). Full-attention layers use torch SDPA (which dispatches to cuDNN on SM100) as a
stand-in for flash-attn, so no flash-attn install is needed.

    python benchmark/linear_attention/fla_e2e_perf_share.py --layers 12 --hidden 1024 --seq 2048
"""

import argparse
import collections

import torch
import torch.nn.functional as F


def _wire_sdpa_attention():
    """FLA's full-attention layer hard-requires flash-attn; substitute torch SDPA
    (which dispatches to cuDNN's fused attention on SM100)."""
    import fla.layers.attn as fla_attn

    def _sdpa_flash(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **kw):
        qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))  # [B,L,H,D] -> [B,H,L,D]
        o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal, scale=softmax_scale, dropout_p=dropout_p)
        return o.transpose(1, 2)

    fla_attn.flash_attn_func = _sdpa_flash


_wire_sdpa_attention()

from fla.models.gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetConfig
import cudnn.fla as cfla
from cudnn.fla.gated_delta_rule import last_path as gdn_last_path


def pick_sm100():
    for i in range(torch.cuda.device_count()):
        if torch.cuda.get_device_properties(i).major >= 10:
            return torch.device(f"cuda:{i}")
    raise SystemExit("no SM100 device")


def build_model(dev, layers, hidden, attn_every, vocab):
    heads = hidden // 128
    attn_layers = [i for i in range(layers) if (i + 1) % attn_every == 0]
    cfg = GatedDeltaNetConfig(
        hidden_size=hidden,
        expand_v=1.0,
        head_dim=128,
        num_heads=heads,
        num_v_heads=heads,
        use_gate=True,
        use_short_conv=False,
        num_hidden_layers=layers,
        attn={"layers": attn_layers, "num_heads": heads, "num_kv_heads": heads},
        hidden_ratio=4,
        vocab_size=vocab,
        max_position_embeddings=8192,
        fuse_cross_entropy=True,
    )
    model = GatedDeltaNetForCausalLM(cfg).to(dev).to(torch.bfloat16).train()
    return model, attn_layers


def categorize(name):
    n = name.lower()
    groups = (
        ("linear_attn", ("gdn", "delta", "chunk_gated", "wy_fast", "solve", "cumsum", "l2norm", "kda", "frost", "cutile")),
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
    for tag, keys in groups:
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


def run_step(model, ids):
    model(input_ids=ids, labels=ids).loss.backward()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--attn_every", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--accelerate", type=int, default=1)
    ap.add_argument("--inspect", action="store_true", help="print model structure + GEMM sites and exit")
    args = ap.parse_args()

    dev = pick_sm100()
    torch.manual_seed(0)
    model, attn_layers = build_model(dev, args.layers, args.hidden, args.attn_every, args.vocab)
    print(f"device {torch.cuda.get_device_properties(dev).name}")
    print(
        f"model: {args.layers} layers (attn at {attn_layers}), hidden={args.hidden}, head_dim=128, "
        f"seq={args.seq}, bs={args.bs}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M"
    )
    if args.inspect:
        print("\n=== model structure ===")
        print(model)
        print("\n=== nn.Linear (GEMM) sites — module : [out, in] ===")
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f"  {name:55} [{m.out_features}, {m.in_features}]")
        return

    if args.accelerate:
        cfla.accelerate_fla(verbose=True)

    ids = torch.randint(0, args.vocab, (args.bs, args.seq), device=dev)
    for _ in range(3):
        model.zero_grad(set_to_none=True)
        run_step(model, ids)
    torch.cuda.synchronize()
    print(f"linear-attn op path: {gdn_last_path()}")

    best = float("inf")
    for _ in range(10):
        model.zero_grad(set_to_none=True)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        run_step(model, ids)
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))

    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        run_step(model, ids)
        torch.cuda.synchronize()

    cat, be, total = collections.defaultdict(float), collections.defaultdict(float), 0.0
    for ev in prof.key_averages():
        t = ev.self_device_time_total
        if t <= 0:
            continue
        cat[categorize(ev.key)] += t
        be[backend(ev.key)] += t
        total += t

    print(f"\nfull training step (fwd+bwd, eager): {best:.3f} ms")
    print(f"  GPU kernel self-time: {total/1e3:.3f} ms   host/overhead gap: {best - total/1e3:.3f} ms ({100*(best - total/1e3)/best:.0f}% of wall)")
    print(f"\n{'category':12} {'ms':>9} {'share':>7}")
    print("-" * 30)
    for c in ("linear_attn", "full_attn", "gemm", "norm", "misc", "other"):
        if cat[c] > 0:
            print(f"{c:12} {cat[c]/1e3:9.3f} {100*cat[c]/total:6.1f}%")
    print(f"\n{'backend':12} {'ms':>9} {'share':>7}")
    print("-" * 30)
    for b in ("cuDNN", "cuBLAS", "torch"):
        if be[b] > 0:
            print(f"{b:12} {be[b]/1e3:9.3f} {100*be[b]/total:6.1f}%")
    print(f"\ncuDNN-owned share of GPU kernel time: {100*be['cuDNN']/total:.1f}%")


if __name__ == "__main__":
    main()
