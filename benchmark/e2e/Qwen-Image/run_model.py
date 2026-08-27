#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image transformer-shape proxy and switchable BF16 joint attention.

The proxy instantiates Diffusers' real ``QwenImageTransformer2DModel`` class
with the published hidden/head/FFN dimensions.  It reduces only the repeated
60-block depth and uses random weights/precomputed text embeddings, so this is
a transformer-backbone performance proxy rather than image-quality or complete
pipeline inference.

Qwen-Image's joint Q/K/V order is ``[text, image]``.  Its mask only rejects
padded text *key columns*; all valid text and all image columns are visible to
every query.  Canonical single-prompt inference trims text padding and uses no
mask.  For a batch with right-padded text, the cuDNN adapter temporarily
permutes Q/K/V to ``[image, text]`` so padding becomes a suffix representable by
``seq_len_kv``.  Full non-causal attention is permutation-equivariant, and the
output is permuted back before Diffusers splits the two streams.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

OFFICIAL_MODEL = {
    "id": "Qwen/Qwen-Image",
    "revision": "75e0b4be04f60ec59a75f475837eced720f823b6",
    "config": "transformer/config.json",
    "config_url": ("https://huggingface.co/Qwen/Qwen-Image/blob/" "75e0b4be04f60ec59a75f475837eced720f823b6/transformer/config.json"),
}
DIFFUSERS_ANCHOR = {
    "project": "huggingface/diffusers",
    "commit": "2f7e0154a9db246e95c9ede43edba7db5b130805",
    "path": "src/diffusers/models/transformers/transformer_qwenimage.py",
    "source_sha256": "cea921e2dd8bba5fcd86ae99054d7320b2773cc653c04c218983d54e737e2cc5",
    "url": (
        "https://github.com/huggingface/diffusers/blob/"
        "2f7e0154a9db246e95c9ede43edba7db5b130805/"
        "src/diffusers/models/transformers/transformer_qwenimage.py"
    ),
}
NUMERICAL_RECIPE = {
    "id": "qwen-image-conservative-bf16-v1",
    "parameter_dtype": "bfloat16",
    "activation_dtype": "bfloat16",
    "internal_math": "upstream Diffusers defaults, including explicit FP32 RoPE",
    "scope": "inference_transformer_forward",
    "anchor": None,
    "alignment": "local_baseline",
}
PUBLISHED_SHAPE = {
    "full_layers": 60,
    "hidden": 3072,
    "heads": 24,
    "head_dim": 128,
    "ffn": 12288,
    "joint_attention_dim": 3584,
    "in_channels": 64,
    "out_channels": 16,
    "patch_size": 2,
}
MODE_DEFAULTS = {
    "smoke": {"layers": 1, "image_tokens": 256, "text_tokens": 64},
    "formal": {"layers": 4, "image_tokens": 4096, "text_tokens": 512},
}


def resolve_shape(mode, *, layers=None, image_tokens=None, text_tokens=None, bs=1):
    if mode not in MODE_DEFAULTS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {tuple(MODE_DEFAULTS)}")
    shape = dict(MODE_DEFAULTS[mode])
    for name, value in (("layers", layers), ("image_tokens", image_tokens), ("text_tokens", text_tokens)):
        if value is not None:
            shape[name] = value
    shape["bs"] = bs
    if any(not isinstance(shape[name], int) or shape[name] <= 0 for name in ("layers", "image_tokens", "text_tokens", "bs")):
        raise ValueError(f"all resolved dimensions must be positive integers, got {shape}")
    image_side = math.isqrt(shape["image_tokens"])
    if image_side * image_side != shape["image_tokens"]:
        raise ValueError("image_tokens must be a perfect square for Qwen-Image RoPE")
    shape["image_side"] = image_side
    shape["joint_tokens"] = shape["image_tokens"] + shape["text_tokens"]
    shape.update({name: PUBLISHED_SHAPE[name] for name in ("hidden", "heads", "head_dim", "ffn", "joint_attention_dim")})
    return shape


def is_right_padded(mask):
    """Return whether every row is a True prefix followed by only False values."""
    if mask.ndim != 2 or mask.dtype is not __import__("torch").bool:
        return False
    lengths = mask.sum(dim=-1)
    expected = __import__("torch").arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    return bool(__import__("torch").equal(mask, expected))


def load_runtime():
    import cudnn
    import cudnn.experimental.ops.sdpa as cudnn_sdpa_module
    import diffusers
    import diffusers.models.transformers.transformer_qwenimage as qwen_module
    import torch
    import torch.nn.functional as F

    return torch, F, cudnn, cudnn_sdpa_module, diffusers, qwen_module


def install_joint_attention_dispatch(qwen_module, *, text_tokens, counters=None, torch_probe=None):
    """Install Torch/cuDNN dispatchers and return ``(select, restore)``.

    Diffusers keeps projection, QK norm, RoPE, stream split, and output
    projections around this function.  The treatment therefore changes only
    the joint SDPA core (plus the exact padding-layout adapter when needed).
    """
    import cudnn.experimental.ops.sdpa as cudnn_sdpa_module
    import torch
    import torch.nn.functional as F

    if counters is None:
        counters = {}
    for name in ("torch_reference", "torch_flash", "cudnn"):
        counters.setdefault(name, 0)
    if torch_probe is None:
        torch_probe = {}
    original = qwen_module.dispatch_attention_fn
    cudnn_sdpa = cudnn_sdpa_module.scaled_dot_product_attention

    def _validate(q, k, v, attn_mask, dropout_p, is_causal, parallel_config):
        if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
            raise ValueError(f"expected equal BLHD Q/K/V, got {q.shape}, {k.shape}, {v.shape}")
        if q.shape[1] <= text_tokens:
            raise ValueError(f"joint sequence {q.shape[1]} must exceed text_tokens={text_tokens}")
        if dropout_p != 0.0 or is_causal:
            raise NotImplementedError("Qwen-Image proxy requires non-causal attention with dropout_p=0")
        if parallel_config is not None:
            raise NotImplementedError("context-parallel attention is outside this single-GPU proxy")
        if attn_mask is not None and tuple(attn_mask.shape) != (q.shape[0], 1, 1, q.shape[1]):
            raise ValueError(f"expected broadcast joint key mask [B,1,1,S], got {attn_mask.shape}")

    def _torch_inputs(q, k, v):
        return tuple(tensor.transpose(1, 2) for tensor in (q, k, v))

    def torch_reference_dispatch(
        q,
        k,
        v,
        *,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        backend=None,
        parallel_config=None,
        **kwargs,
    ):
        del backend, kwargs
        _validate(q, k, v, attn_mask, dropout_p, is_causal, parallel_config)
        counters["torch_reference"] += 1
        qt, kt, vt = _torch_inputs(q, k, v)
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        return out.transpose(1, 2)

    def torch_flash_dispatch(
        q,
        k,
        v,
        *,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        backend=None,
        parallel_config=None,
        **kwargs,
    ):
        del backend, kwargs
        _validate(q, k, v, attn_mask, dropout_p, is_causal, parallel_config)
        if attn_mask is not None:
            raise NotImplementedError("the timed PyTorch FlashAttention treatment is dense; use torch_reference for mask parity")
        counters["torch_flash"] += 1
        qt, kt, vt = _torch_inputs(q, k, v)
        names = {int(value): name for name, value in torch.nn.attention.SDPBackend.__members__.items()}
        if not torch_probe:
            natural_choice = int(torch._fused_sdp_choice(qt, kt, vt, None, dropout_p, is_causal, scale=scale, enable_gqa=False))
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                forced_choice = int(torch._fused_sdp_choice(qt, kt, vt, None, dropout_p, is_causal, scale=scale, enable_gqa=False))
            torch_probe.update(
                {
                    "natural_choice": natural_choice,
                    "natural_choice_name": names.get(natural_choice, f"unknown:{natural_choice}"),
                    "forced_choice": forced_choice,
                    "forced_choice_name": names.get(forced_choice, f"unknown:{forced_choice}"),
                    "shape": list(qt.shape),
                }
            )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(qt, kt, vt, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        return out.transpose(1, 2)

    def cudnn_dispatch(
        q,
        k,
        v,
        *,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        backend=None,
        parallel_config=None,
        **kwargs,
    ):
        del backend, kwargs
        _validate(q, k, v, attn_mask, dropout_p, is_causal, parallel_config)
        counters["cudnn"] += 1
        image_tokens = q.shape[1] - text_tokens
        seq_len_q = seq_len_kv = None
        reordered = False
        if attn_mask is not None:
            if attn_mask.dtype is not torch.bool:
                raise NotImplementedError("the Qwen-Image cuDNN adapter accepts only the canonical boolean key mask")
            valid = attn_mask[:, 0, 0]
            text_valid = valid[:, :text_tokens]
            if not bool(valid[:, text_tokens:].all()) or not is_right_padded(text_valid):
                raise NotImplementedError("cuDNN Qwen-Image adapter supports only right-padded text and all-valid image tokens")
            if not bool(text_valid.all()):
                # [text, pad, image] has an interior hole. Joint full attention
                # permits a common token permutation, so [image, text, pad]
                # makes the same mask representable as a valid prefix.
                q, k, v = (torch.cat([tensor[:, text_tokens:], tensor[:, :text_tokens]], dim=1) for tensor in (q, k, v))
                batch = q.shape[0]
                seq_len_q = torch.full((batch, 1, 1, 1), q.shape[1], dtype=torch.int32, device=q.device)
                seq_len_kv = (image_tokens + text_valid.sum(dim=-1, dtype=torch.int32)).reshape(batch, 1, 1, 1)
                reordered = True
        qt, kt, vt = (tensor.transpose(1, 2) for tensor in (q, k, v))
        out = cudnn_sdpa(
            qt,
            kt,
            vt,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
        ).transpose(1, 2)
        if reordered:
            out = torch.cat([out[:, image_tokens:], out[:, :image_tokens]], dim=1)
        return out

    dispatchers = {
        "torch_reference": torch_reference_dispatch,
        "torch_flash": torch_flash_dispatch,
        "cudnn": cudnn_dispatch,
    }

    def select(name):
        if name not in dispatchers:
            raise ValueError(f"unknown joint-attention backend {name!r}")
        qwen_module.dispatch_attention_fn = dispatchers[name]

    def restore():
        if qwen_module.dispatch_attention_fn in dispatchers.values():
            qwen_module.dispatch_attention_fn = original

    return select, restore, counters, torch_probe


def build_model(torch, qwen_module, device, *, layers):
    torch.manual_seed(0)
    with torch.cuda.device(device):
        model = qwen_module.QwenImageTransformer2DModel(
            patch_size=PUBLISHED_SHAPE["patch_size"],
            in_channels=PUBLISHED_SHAPE["in_channels"],
            out_channels=PUBLISHED_SHAPE["out_channels"],
            num_layers=layers,
            attention_head_dim=PUBLISHED_SHAPE["head_dim"],
            num_attention_heads=PUBLISHED_SHAPE["heads"],
            joint_attention_dim=PUBLISHED_SHAPE["joint_attention_dim"],
        )
        model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return model


def make_inputs(torch, device, shape, *, valid_text_lengths=None):
    generator = torch.Generator(device=device).manual_seed(1234)
    image = torch.randn(shape["bs"], shape["image_tokens"], PUBLISHED_SHAPE["in_channels"], device=device, dtype=torch.bfloat16, generator=generator)
    text = torch.randn(shape["bs"], shape["text_tokens"], PUBLISHED_SHAPE["joint_attention_dim"], device=device, dtype=torch.bfloat16, generator=generator)
    timestep = torch.full((shape["bs"],), 0.5, device=device, dtype=torch.bfloat16)
    img_shapes = [[(1, shape["image_side"], shape["image_side"])] for _ in range(shape["bs"])]
    mask = None
    if valid_text_lengths is not None:
        lengths = torch.as_tensor(valid_text_lengths, dtype=torch.int64, device=device)
        if tuple(lengths.shape) != (shape["bs"],) or bool((lengths < 0).any()) or bool((lengths > shape["text_tokens"]).any()):
            raise ValueError(f"invalid valid_text_lengths={valid_text_lengths}")
        mask = torch.arange(shape["text_tokens"], device=device).unsqueeze(0) < lengths.unsqueeze(1)
        if bool(mask.all()):
            mask = None
    return {
        "hidden_states": image,
        "encoder_hidden_states": text,
        "encoder_hidden_states_mask": mask,
        "timestep": timestep,
        "img_shapes": img_shapes,
        "return_dict": False,
    }


def forward(model, inputs):
    return model(**inputs)[0]


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODE_DEFAULTS, default="smoke")
    parser.add_argument("--backend", choices=("torch_reference", "torch_flash", "cudnn"), default="cudnn")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--image-tokens", type=int)
    parser.add_argument("--text-tokens", type=int)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    shape = resolve_shape(args.mode, layers=args.layers, image_tokens=args.image_tokens, text_tokens=args.text_tokens)
    if args.inspect:
        print(json.dumps({"shape": shape, "model": OFFICIAL_MODEL, "diffusers": DIFFUSERS_ANCHOR, "recipe": NUMERICAL_RECIPE}, indent=2))
        return
    torch, _, cudnn, _, _, qwen_module = load_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    if properties.major != 10:
        raise RuntimeError(f"Qwen-Image proxy requires SM100, got {properties.name}")
    if cudnn.backend_version() < 92100:
        raise RuntimeError(f"joint d128 SDPA requires a current cuDNN backend, got {cudnn.backend_version()}")
    model = build_model(torch, qwen_module, device, layers=shape["layers"])
    inputs = make_inputs(torch, device, shape)
    select, restore, counters, probe = install_joint_attention_dispatch(qwen_module, text_tokens=shape["text_tokens"])
    try:
        select(args.backend)
        samples = []
        with torch.inference_mode():
            for _ in range(2):
                forward(model, inputs)
            torch.cuda.synchronize(device)
            for _ in range(10):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                output = forward(model, inputs)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("non-finite output")
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "p50_ms": statistics.median(samples),
                    "shape": shape,
                    "calls": counters,
                    "torch_probe": probe,
                },
                sort_keys=True,
            )
        )
    finally:
        restore()


if __name__ == "__main__":
    _main()
