# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-block perf share for any published Qwen3.5 / Qwen3.6 configuration.

Times one linear-attention block, one full-attention block, and the non-per-layer head,
then weights them by the model's real layer counts. Nothing larger than a single block is
ever resident, so the 397B-A17B configuration is as cheap to measure as the 0.8B one.

    python benchmark/e2e/Qwen3.5/run_blocks.py --model Qwen3.5-27B --seq 16384

Attention head_dim is set explicitly rather than derived from hidden_size, so the published
(num_attention_heads, head_dim) pair is reproduced exactly; six of the eight configurations
have ``num_attention_heads * 256 > hidden_size`` and cannot be expressed otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _blocks import report, time_block, time_step  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import ALIASES, MODELS, get  # noqa: E402


class FullAttentionBlock(nn.Module):
    """GQA self-attention with an explicit head_dim, plus its SwiGLU MLP and two norms."""

    def __init__(self, cfg):
        super().__init__()
        h, d = cfg["hidden_size"], cfg["head_dim"]
        self.nh, self.nkv, self.d = cfg["num_attention_heads"], cfg["num_key_value_heads"], d
        self.norm1, self.norm2 = nn.RMSNorm(h), nn.RMSNorm(h)
        self.q = nn.Linear(h, self.nh * d, bias=False)
        self.k = nn.Linear(h, self.nkv * d, bias=False)
        self.v = nn.Linear(h, self.nkv * d, bias=False)
        self.o = nn.Linear(self.nh * d, h, bias=False)
        i = cfg["intermediate_size"]
        self.gate, self.up, self.down = nn.Linear(h, i, bias=False), nn.Linear(h, i, bias=False), nn.Linear(i, h, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        y = self.norm1(x)
        q = self.q(y).view(B, T, self.nh, self.d).transpose(1, 2)
        k = self.k(y).view(B, T, self.nkv, self.d).transpose(1, 2)
        v = self.v(y).view(B, T, self.nkv, self.d).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.nh != self.nkv)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, self.nh * self.d))
        y = self.norm2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class LinearAttentionBlock(nn.Module):
    """FLA Gated DeltaNet (with its short convolution) plus its SwiGLU MLP and two norms."""

    def __init__(self, cfg):
        super().__init__()
        from fla.layers.gated_deltanet import GatedDeltaNet

        h = cfg["hidden_size"]
        self.norm1, self.norm2 = nn.RMSNorm(h), nn.RMSNorm(h)
        self.mixer = GatedDeltaNet(
            mode="chunk",
            hidden_size=h,
            head_dim=cfg["linear_key_head_dim"],
            num_heads=cfg["linear_num_key_heads"],
            num_v_heads=cfg["linear_num_value_heads"],
            expand_v=cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"] / (cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]),
            use_gate=True,
            use_short_conv=True,
            conv_size=cfg["linear_conv_kernel_dim"],
        )
        i = cfg["intermediate_size"]
        self.gate, self.up, self.down = nn.Linear(h, i, bias=False), nn.Linear(h, i, bias=False), nn.Linear(i, h, bias=False)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))[0]
        y = self.norm2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class Head(nn.Module):
    """Everything that is not per-layer: embedding, final norm, LM head, loss.

    Timed once and weighted once, at the model's real vocabulary. Scaling the vocabulary
    down in proportion to a reduced layer count keeps the head's FLOP *ratio* but hides how
    the ratio moves with model size: at seq 16384 this head is 24% of a composed Qwen3.5-0.8B
    step and 3% of a Qwen3.5-27B one.
    """

    def __init__(self, cfg):
        super().__init__()
        h, v = cfg["hidden_size"], cfg["vocab_size"]
        self.embed = nn.Embedding(v, h)
        self.norm = nn.RMSNorm(h)
        self.lm_head = nn.Linear(h, v, bias=False)

    def forward(self, ids):
        logits = self.lm_head(self.norm(self.embed(ids))).float()
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), ids.view(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.5-27B", choices=sorted(MODELS) + sorted(ALIASES))
    ap.add_argument("--seq", type=int, default=16384, help="tokens per microbatch; for a packed THD workload use its effective length")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--forwards-per-backward", type=int, default=1, help="GRPO-style recipes run several; 1 keeps a plain training step")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    cfg = get(args.model)
    if cfg["kind"] == "moe":
        raise SystemExit(
            f"{cfg['name']} is a mixture-of-experts configuration and its MoE block is not implemented yet. "
            "A block driven by a synthetic uniform router would report grouped-GEMM shapes that do not occur "
            "in practice, and routing imbalance is what sets MoE throughput, so no number is better than a wrong one."
        )

    dev = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(dev)  # autograd's engine thread runs cuBLAS; give it a primary context
    p = torch.cuda.get_device_properties(dev)
    print(f"# {p.name} sm{p.major}{p.minor}  {cfg['name']}  seq={args.seq} bs={args.bs}")
    print(f"# hidden={cfg['hidden_size']} layers={cfg['num_hidden_layers']} ({cfg['num_linear_layers']} linear / {cfg['num_full_attention_layers']} full-attn)")
    print(
        f"# conv_dim={cfg['conv_dim']} (K={cfg['linear_conv_kernel_dim']})  attn={cfg['num_attention_heads']}q/{cfg['num_key_value_heads']}kv x {cfg['head_dim']}  vocab={cfg['vocab_size']}"
    )

    def residual_stream():
        return torch.randn(args.bs, args.seq, cfg["hidden_size"], device=dev, dtype=torch.bfloat16, requires_grad=True)

    measured = {}
    for label, block_cls in (("linear_attn (GDN+conv+MLP)", LinearAttentionBlock), ("full_attn (SDPA+MLP)", FullAttentionBlock)):
        block = block_cls(cfg).to(dev).to(torch.bfloat16).train()
        fwd, fwd_bwd = time_block(block, residual_stream, iters=args.iters)
        measured[label] = (fwd, fwd_bwd)
        del block
        torch.cuda.empty_cache()

    head = Head(cfg).to(dev).to(torch.bfloat16).train()
    ids = torch.randint(0, cfg["vocab_size"], (args.bs, args.seq), device=dev)

    def head_step():
        head.zero_grad(set_to_none=True)
        head(ids).backward()

    head_fwd_bwd = time_step(head_step, iters=args.iters)
    del head
    torch.cuda.empty_cache()

    # a recipe with several forward passes per backward -- GRPO runs three -- is a weight,
    # not a different model: cost = f * fwd + (fwd_bwd - fwd)
    f = args.forwards_per_backward

    def weighted(fwd, fwd_bwd):
        return f * fwd + (fwd_bwd - fwd)

    blocks = {
        "linear_attn (GDN+conv+MLP)": (cfg["num_linear_layers"], weighted(*measured["linear_attn (GDN+conv+MLP)"])),
        "full_attn (SDPA+MLP)": (cfg["num_full_attention_layers"], weighted(*measured["full_attn (SDPA+MLP)"])),
    }
    head_ms = head_fwd_bwd
    print(f"\n  per block, fwd / fwd+bwd ms:")
    for label, (fwd, fwd_bwd) in measured.items():
        print(f"    {label:28} {fwd:8.3f} {fwd_bwd:9.3f}")
    print(f"    {'head (fwd+bwd)':28} {'':8} {head_fwd_bwd:9.3f}")
    if f != 1:
        print(f"  weighting {f} forward passes per backward")

    report(cfg["name"], blocks, head_ms)


if __name__ == "__main__":
    main()
