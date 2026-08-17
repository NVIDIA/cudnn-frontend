# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Next-style hybrid Gated DeltaNet LM — e2e perf-share / support-gap.

Builds flash-linear-attention's Gated DeltaNet model (mostly linear-attention
layers + a few full-attention layers + a SwiGLU MLP) and profiles one fwd+bwd
training step by category and backend, to show what fraction of a real step cuDNN
owns and which un-owned op is next.

Two cuDNN swaps, each behind a flag:
  --accelerate_mlp   route the SwiGLU MLP through cudnn.gemm.ops.swiglu_mlp (PR #609).
                     The MLP GEMMs are ~70% of a step at real dims, so this is the
                     dominant block; the forward SwiGLU fusion wins (1.05-1.20x) and the
                     backward fuses the dgrad GEMM + dSwiGLU into one FROST kernel
                     (~1.25x vs torch), so fwd+bwd is a training-step win.
  --accelerate_attn  route linear attention through cudnn.fla (PR #596), if installed.
                     Only ~6% of a step, so small e2e effect; kept for completeness.

Full-attention layers use torch SDPA (which dispatches to cuDNN's fused attention on
SM100) as a flash-attn stand-in, so no flash-attn install is needed. Requires an SM100
(Blackwell) device.

    python benchmark/e2e/Qwen3-Next/run_model.py --accelerate_mlp 1 --layers 12 --hidden 1024
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _perfshare import pick_sm100, profile_and_report  # noqa: E402


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

from fla.models.gated_deltanet import GatedDeltaNetForCausalLM, GatedDeltaNetConfig  # noqa: E402


def _accelerate_mlp():
    """Route FLA's GatedMLP through cudnn.gemm.ops.swiglu_mlp (bias-free swish MLP:
    (silu(x@Wg^T) * (x@Wu^T)) @ Wd^T, exactly what the op computes)."""
    from cudnn.gemm.ops import swiglu_mlp
    import fla.modules.mlp as fla_mlp

    def _fwd(self, x, **kwargs):
        return swiglu_mlp(x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight)

    fla_mlp.GatedMLP.forward = _fwd
    print("[e2e] MLP -> cudnn.gemm.ops.swiglu_mlp (PR #609)")


def _accelerate_attn():
    """Route linear attention through cudnn.fla (PR #596) if the package is present."""
    try:
        import cudnn.fla as cfla
    except ImportError:
        print("[e2e] cudnn.fla not installed (PR #596); linear attention stays on FLA")
        return None
    cfla.accelerate_fla(verbose=True)
    from cudnn.fla.gated_delta_rule import last_path

    return last_path


def build_model(dev, layers, hidden, attn_every, vocab):
    if attn_every < 1:
        raise ValueError("attn_every must be >= 1")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--attn_every", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--accelerate_mlp", type=int, default=1, help="route the SwiGLU MLP through cuDNN (PR #609)")
    ap.add_argument("--accelerate_attn", type=int, default=0, help="route linear attention through cudnn.fla (PR #596)")
    ap.add_argument("--inspect", action="store_true", help="print model structure + GEMM sites and exit")
    args = ap.parse_args()

    dev = pick_sm100()
    torch.manual_seed(0)
    model, attn_layers = build_model(dev, args.layers, args.hidden, args.attn_every, args.vocab)
    print(f"device {torch.cuda.get_device_properties(dev).name}")
    print(
        f"model: {args.layers} layers (attn at {attn_layers}), hidden={args.hidden}, head_dim=128, "
        f"seq={args.seq}, bs={args.bs}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M"
    )
    if args.inspect:
        print("\n=== model structure ===")
        print(model)
        print("\n=== nn.Linear (GEMM) sites — module : [out, in] ===")
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f"  {name:55} [{m.out_features}, {m.in_features}]")
        return

    attn_path = _accelerate_attn() if args.accelerate_attn else None
    if args.accelerate_mlp:
        _accelerate_mlp()

    ids = torch.randint(0, args.vocab, (args.bs, args.seq), device=dev)
    extra = (lambda: f"linear-attn op path: {attn_path()}") if attn_path is not None else None
    profile_and_report(model, ids, extra_path=extra)


if __name__ == "__main__":
    main()
