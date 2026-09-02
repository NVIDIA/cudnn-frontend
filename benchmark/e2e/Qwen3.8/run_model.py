# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-27B layer-shape hybrid Gated DeltaNet LM — e2e perf-share.

Builds flash-linear-attention's Gated DeltaNet model with the real Qwen3.8-27B
hidden, MLP, and GVA dimensions.  The default keeps one representative four-layer
period (three GDN + one full-attention layer) and a reduced vocabulary, so it
preserves the repeated layer work without allocating the full 64-layer model.
Qwen3.5/3.6-27B have the same kernel-relevant layer dimensions and are available
as alias presets. Models without a published, reproducible config are not guessed.

Three independent backend choices:
  --accelerate_mlp   route the SwiGLU MLP through cudnn.gemm.ops.swiglu_mlp (PR #609).
                     The forward fuses the two FC1 GEMMs with SwiGLU and the
                     backward fuses the dgrad GEMM + dSwiGLU into FROST.
  --accelerate_attn  route linear attention through cudnn.fla (PR #596), if installed.
                     This is measured independently from the MLP swap.
  --full_attn_backend
                     choose vanilla torch SDPA or the FE public cuDNN-backend
                     SDPA op for the common full-attention layer.

Full-attention layers use a Torch-SDPA-compatible flash-attn stand-in, so no
flash-attn install is needed.  The default calls the cuDNN Frontend op directly,
bypassing the tested Torch 2.13 dispatch guard that does not select cuDNN for
head-dim 256; ``--full_attn_backend torch`` restores the vanilla dispatcher for
an exact A/B.  The cuDNN arm requires cuDNN backend >= 9.23; after FE #682 the
public d256 op is backend-graph-only.  Requires an SM100 (Blackwell) device.

    python benchmark/e2e/Qwen3.8/run_model.py --accelerate_mlp 1 --accelerate_attn 1
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _perfshare import pick_sm100, profile_and_report  # noqa: E402


def _wire_sdpa_attention():
    """Build switchable Torch/cuDNN SDPA adapters for FLA full attention.

    FE #682 removed the legacy standalone d256 OSS/CuteDSL stacks, so the public
    op below is backend-graph-only.  cuDNN 9.23 is the first backend release with
    d256 forward and backward support.  Both adapters receive the same
    projected/rotated Q/K/V and return the same packed BLHD layout.  Thus
    QKV/RoPE/O remain common code and are included in module and model timings;
    only the SDPA core changes across this axis.
    """
    import cudnn
    import cudnn.experimental.ops.sdpa as cudnn_sdpa_module
    import fla.layers.attn as fla_attn
    import torch.nn.functional as F

    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise RuntimeError(
            "disable CUDNN_FRONTEND_ENABLE_FROST_ENGINES for this benchmark: "
            "the SwiGLU backward uses direct FROST JIT, while the global opt-in "
            "would also replace the intended cuDNN-backend d256 SDPA plan"
        )

    backend_floor = 92300
    backend_version = cudnn.backend_version()
    if backend_version < backend_floor:
        raise RuntimeError(f"d256 SDPA requires cuDNN backend >= {backend_floor}; got {backend_version}")
    cudnn_sdpa = cudnn_sdpa_module.scaled_dot_product_attention

    def _prepare(q, k, v, window_size):
        qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))  # [B,L,H,D] -> [B,H,L,D]
        if kt.shape[1] != vt.shape[1] or qt.shape[1] % kt.shape[1] != 0:
            raise ValueError(f"invalid GQA heads: Q/K/V={qt.shape[1]}/{kt.shape[1]}/{vt.shape[1]}")
        if not isinstance(window_size, tuple) or len(window_size) != 2:
            raise ValueError(f"expected flash-attn window_size tuple, got {window_size!r}")
        return qt, kt, vt

    def _torch_sdpa_flash(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        **kw,
    ):
        qt, kt, vt = _prepare(q, k, v, window_size)
        if window_size != (-1, -1):
            raise NotImplementedError("the vanilla torch full-attention A/B supports only the full causal window")
        o = F.scaled_dot_product_attention(
            qt,
            kt,
            vt,
            is_causal=causal,
            scale=softmax_scale,
            dropout_p=dropout_p,
            enable_gqa=qt.shape[1] != kt.shape[1],
        )
        return o.transpose(1, 2).contiguous()

    def _cudnn_sdpa_flash(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        **kw,
    ):
        qt, kt, vt = _prepare(q, k, v, window_size)
        o = cudnn_sdpa(
            qt,
            kt,
            vt,
            is_causal=causal,
            scale=softmax_scale,
            dropout_p=dropout_p,
            enable_gqa=qt.shape[1] != kt.shape[1],
            left_bound=window_size[0],
            right_bound=window_size[1],
        )
        # flash-attn's adapter contract is packed [B,L,H,D].  The direct cuDNN
        # wrapper returns packed BHSD, so normalize once before FLA flattens H*D.
        return o.transpose(1, 2).contiguous()

    adapters = {"torch": _torch_sdpa_flash, "cudnn": _cudnn_sdpa_flash}

    def select(name):
        try:
            fla_attn.flash_attn_func = adapters[name]
        except KeyError:
            raise ValueError(f"unknown full-attention backend {name!r}; choices={tuple(adapters)}") from None
        return name

    select("cudnn")
    return select


_set_full_attention_backend = _wire_sdpa_attention()

from fla.models.gated_deltanet import (
    GatedDeltaNetForCausalLM,
    GatedDeltaNetConfig,
)  # noqa: E402

_QWEN_DENSE_27B = {
    "layers": 4,
    "hidden": 5120,
    "intermediate": 17408,
    "linear_heads": 16,
    "linear_v_heads": 48,
    "linear_head_dim": 128,
    "attn_heads": 20,
    "attn_kv_heads": 4,
    "attn_every": 4,
    # 64 -> 4 layers is a 16x depth reduction; scale the fixed LM head by the
    # same factor (248320 / 16) to retain its approximate FLOP share.
    "vocab": 15520,
    "short_conv": True,
}

MODEL_PRESETS = {
    # Qwen3.5/3.6/3.8-27B layer dimensions. FLA's full-attention stand-in requires
    # hidden_size % num_heads == 0, so it uses 20x256 rather than the model's
    # gated 24x256 projection. The target MLP and GDN dimensions are exact.
    "qwen3.8-27b": dict(_QWEN_DENSE_27B),
    "qwen3.6-27b": dict(_QWEN_DENSE_27B),
    "qwen3.5-27b": dict(_QWEN_DENSE_27B),
    # Fast smoke/debug preset; never use it for a model-level performance claim.
    "small": {
        "layers": 12,
        "hidden": 1024,
        "intermediate": 2816,
        "linear_heads": 8,
        "linear_v_heads": 8,
        "linear_head_dim": 128,
        "attn_heads": 8,
        "attn_kv_heads": 8,
        "attn_every": 4,
        "vocab": 8192,
        "short_conv": False,
    },
}

# Numerical policy is deliberately data, not another implementation bit. A
# future FP8/FP4 leaf gets its own runner/record rather than branching throughout
# this BF16 benchmark.
NUMERICAL_RECIPE = {
    "id": "conservative-bf16-v1",
    "parameter_dtype": "bfloat16",
    "activation_dtype": "bfloat16",
    "scope": "forward_backward_no_optimizer",
    "anchor": {
        "project": "NVIDIA-NeMo/Megatron-Bridge",
        "commit": "2e77041c194d106beb7462e226d7ca06b33ea63f",
        "path": "src/megatron/bridge/training/mixed_precision.py",
        "symbol": "bf16_mixed",
        "aligned_scope": "single-GPU BF16 parameter/activation policy only; no distributed or optimizer claim",
    },
    "alignment": "upstream_anchored_subset",
}


def _accelerate_mlp():
    """Opt FLA's supported GatedMLP instances into the production cuDNN shim."""
    import cudnn.fla as cfla

    cfla.accelerate_fla(verbose=True, targets="gated_mlp")
    return cfla.mlp_last_path


def _accelerate_attn():
    """Route linear attention through cudnn.fla (PR #596) if the package is present."""
    try:
        import cudnn.fla as cfla
    except ImportError:
        print("[e2e] cudnn.fla not installed (PR #596); linear attention stays on FLA")
        return None
    cfla.accelerate_fla(verbose=True, targets="gated_delta_rule")
    from cudnn.fla.gated_delta_rule import last_path

    return last_path


def build_model(
    dev,
    *,
    layers,
    hidden,
    intermediate,
    linear_heads,
    linear_v_heads,
    linear_head_dim,
    attn_heads,
    attn_kv_heads,
    attn_every,
    vocab,
    short_conv,
):
    if attn_every < 1:
        raise ValueError("attn_every must be >= 1")
    attn_layers = [i for i in range(layers) if (i + 1) % attn_every == 0]
    cfg = GatedDeltaNetConfig(
        hidden_size=hidden,
        expand_v=1.0,
        head_dim=linear_head_dim,
        num_heads=linear_heads,
        num_v_heads=linear_v_heads,
        use_gate=True,
        use_short_conv=short_conv,
        num_hidden_layers=layers,
        attn={
            "layers": attn_layers,
            "num_heads": attn_heads,
            "num_kv_heads": attn_kv_heads,
        },
        intermediate_size=intermediate,
        vocab_size=vocab,
        max_position_embeddings=8192,
        fuse_cross_entropy=True,
    )
    model = GatedDeltaNetForCausalLM(cfg).to(dev).to(torch.bfloat16).train()
    return model, attn_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=MODEL_PRESETS, default="qwen3.8-27b")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--intermediate", type=int)
    ap.add_argument("--linear_heads", type=int)
    ap.add_argument("--linear_v_heads", type=int)
    ap.add_argument("--linear_head_dim", type=int)
    ap.add_argument("--attn_heads", type=int)
    ap.add_argument("--attn_kv_heads", type=int)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--attn_every", type=int)
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--short_conv", type=int, choices=(0, 1))
    ap.add_argument(
        "--accelerate_mlp",
        type=int,
        default=1,
        help="route the SwiGLU MLP through the cudnn.fla shim backed by PR #609",
    )
    ap.add_argument(
        "--accelerate_attn",
        type=int,
        default=0,
        help="route linear attention through cudnn.fla (PR #596)",
    )
    ap.add_argument(
        "--full_attn_backend",
        choices=("torch", "cudnn"),
        default="cudnn",
        help="full-attention SDPA core; QKV/O projections are common",
    )
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="print model structure + GEMM sites and exit",
    )
    args = ap.parse_args()
    _set_full_attention_backend(args.full_attn_backend)

    shape = MODEL_PRESETS[args.preset].copy()
    for name in shape:
        override = getattr(args, name)
        if override is not None:
            shape[name] = bool(override) if name == "short_conv" else override

    dev = pick_sm100()
    torch.manual_seed(0)
    model, attn_layers = build_model(dev, **shape)
    print(f"device {torch.cuda.get_device_properties(dev).name}")
    print(
        f"model: preset={args.preset}, {shape['layers']} layers (attn at {attn_layers}), "
        f"hidden={shape['hidden']}, intermediate={shape['intermediate']}, "
        f"GDN={shape['linear_heads']}qk/{shape['linear_v_heads']}v x{shape['linear_head_dim']}, "
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
    mlp_path = _accelerate_mlp() if args.accelerate_mlp else None

    ids = torch.randint(0, shape["vocab"], (args.bs, args.seq), device=dev)

    def extra():
        paths = [f"full-attn SDPA: {args.full_attn_backend}"]
        if attn_path is not None:
            paths.append(f"linear-attn op path: {attn_path()}")
        if mlp_path is not None:
            paths.append(f"MLP op path: {mlp_path()}")
        return ", ".join(paths)

    profile_and_report(model, ids, extra_path=extra)


if __name__ == "__main__":
    main()
