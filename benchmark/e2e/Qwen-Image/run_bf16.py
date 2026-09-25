#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Balanced BF16 Qwen-Image joint-attention A/B benchmark.

This is one numerical-recipe leaf: conservative BF16. It compares explicitly
forced PyTorch FlashAttention with the FE public cuDNN backend graph while keeping Diffusers'
Q/K/V projections, QK norm, RoPE, output projections, AdaLN, GELU FFNs, and all
other transformer work identical.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

THIS_DIR = Path(__file__).resolve().parent
E2E_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]
MODEL_PATH = THIS_DIR / "run_model.py"
FACTORIAL_PATH = E2E_DIR / "_factorial.py"
sys.path.insert(0, str(E2E_DIR))

from _factorial import config_fingerprint, paired_stats, percentile, williams_orders  # noqa: E402

PROTOCOL_DEFAULTS = {
    "smoke": {"warmup": 1, "rounds": 8, "repeats": 1},
    "formal": {"warmup": 3, "rounds": 40, "repeats": 3},
}
VARIANTS = (("0", "torch_flash"), ("1", "cudnn"))


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path):
    path = Path(path).resolve()
    try:
        display = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display = str(path)
    return {"path": display, "sha256": _sha256(path)}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git_provenance():
    def run(*arguments):
        return subprocess.run(arguments, cwd=REPO_ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current") or "detached",
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=PROTOCOL_DEFAULTS, default="smoke")
    parser.add_argument("--layers", type=int)
    parser.add_argument("--image-tokens", type=int)
    parser.add_argument("--text-tokens", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "qwen-image-bf16-results")
    parser.add_argument("--tag", default="")
    parser.add_argument("--compare", type=Path)
    return parser.parse_args()


def _resolve_protocol(args):
    protocol = dict(PROTOCOL_DEFAULTS[args.mode])
    for name in protocol:
        value = getattr(args, name)
        if value is not None:
            protocol[name] = value
    if any(not isinstance(value, int) or value <= 0 for value in protocol.values()):
        raise ValueError(f"protocol values must be positive integers, got {protocol}")
    if protocol["rounds"] % 2:
        raise ValueError("rounds must be a multiple of 2 for the complete two-treatment Williams design")
    return protocol


def _pick_device(torch, mode):
    candidates = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        candidates.append(f"cuda:{index}={properties.name}/{properties.multi_processor_count}SM")
        if properties.major == 10 and (mode != "formal" or (properties.name == "NVIDIA B200" and properties.multi_processor_count == 148)):
            return torch.device(f"cuda:{index}"), properties
    requirement = "a full 148-SM NVIDIA B200" if mode == "formal" else "an SM100-family GPU"
    raise RuntimeError(f"{mode} mode requires {requirement}; visible: {', '.join(candidates)}")


def _rel_l2(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def _focused_padding_check(torch, qwen_module, model_api, device):
    text_tokens, image_tokens, heads, head_dim, batch = 64, 128, 4, 128, 2
    generator = torch.Generator(device=device).manual_seed(2026)
    shape = (batch, text_tokens + image_tokens, heads, head_dim)
    q, k, v = (torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(3))
    text_mask = torch.arange(text_tokens, device=device).unsqueeze(0) < torch.tensor([64, 37], device=device).unsqueeze(1)
    mask = torch.cat([text_mask, torch.ones(batch, image_tokens, dtype=torch.bool, device=device)], dim=1)[:, None, None]
    select, restore, _, _ = model_api.install_joint_attention_dispatch(qwen_module, text_tokens=text_tokens)
    try:
        with torch.inference_mode():
            select("torch_reference")
            expected = qwen_module.dispatch_attention_fn(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            select("cudnn")
            actual = qwen_module.dispatch_attention_fn(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        torch.cuda.synchronize(device)
    finally:
        restore()
    rel = _rel_l2(actual, expected)
    if not math.isfinite(rel) or rel > 0.01:
        raise RuntimeError(f"right-padded joint-attention adapter mismatch: rel_l2={rel}")
    return {
        "shape": list(shape),
        "text_valid_lengths": [64, 37],
        "rel_l2": rel,
        "role": "official [text,pad,image] boolean mask vs permuted [image,text,pad] cuDNN seq_len path",
    }


def _load_comparison(path):
    def reject(value):
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def _compare(current, previous):
    current_fp = current["config"]["comparability_fingerprint"]["sha256"]
    previous_fp = previous["config"]["comparability_fingerprint"]["sha256"]
    if current_fp != previous_fp:
        raise ValueError(f"comparison fingerprint mismatch: current={current_fp}, previous={previous_fp}")
    arms = {}
    for bits in ("0", "1"):
        new = float(current["summary"][bits]["p50_ms"])
        old = float(previous["summary"][bits]["p50_ms"])
        if not all(math.isfinite(value) and value > 0 for value in (new, old)):
            raise ValueError("comparison p50 values must be finite and positive")
        arms[bits] = {"previous_p50_ms": old, "current_p50_ms": new, "change_percent": (new / old - 1.0) * 100.0}
    return {
        "paired_across_runs": False,
        "arms": arms,
        "previous_within_run_ratio": previous["comparison_within_run"]["paired_ratio_p50"],
        "current_within_run_ratio": current["comparison_within_run"]["paired_ratio_p50"],
    }


def _render_markdown(metadata, raw_name, raw_hash):
    config = metadata["config"]
    paired = metadata["comparison_within_run"]
    smoke = config["mode"] == "smoke"
    lines = [
        "# Qwen-Image BF16 joint-attention benchmark",
        "",
        f"Generated: `{metadata['completed_utc']}`  ",
        f"Mode: `{config['mode']}`  ",
        f"Comparability fingerprint: `{config['comparability_fingerprint']['sha256']}`  ",
        f"Build/provenance fingerprint: `{config['build_fingerprint']['sha256']}`  ",
        f"Raw JSON: [`{raw_name}`]({raw_name}) (`sha256:{raw_hash}`)",
        "",
    ]
    if smoke:
        lines += [
            "## Smoke validation",
            "",
            "**Validation only; these reduced-token timings are not a performance headline.**",
            "",
        ]
    else:
        ratio = paired["paired_ratio_p50"]
        lines += [
            "## Result",
            "",
            f"Direct FE/cuDNN is `{ratio:.5f}x` the paired forced-PyTorch-Flash elapsed time "
            f"({(1 - ratio) * 100:.2f}% lower, `{1 / ratio:.3f}x` speedup; {paired['wins']}/{paired['batches']} wins).",
            "",
        ]
    lines += [
        "| transformer forward (SDPA treatment) | p10 | p50 | p90 | paired ratio vs PyTorch Flash |",
        "|---|---:|---:|---:|---:|",
    ]
    for bits, label in VARIANTS:
        value = metadata["summary"][bits]
        ratio = 1.0 if bits == "0" else value["paired_ratio_p50"]
        lines.append(f"| {label} | {value['p10_ms']:.3f} ms | {value['p50_ms']:.3f} ms | {value['p90_ms']:.3f} ms | {ratio:.5f} |")
    shape = config["shape"]
    lines += [
        "",
        "## Scope and gates",
        "",
        f"- Shape: B={shape['bs']}, image/text/joint tokens={shape['image_tokens']}/{shape['text_tokens']}/{shape['joint_tokens']}, "
        f"H={shape['hidden']}, heads={shape['heads']}x{shape['head_dim']}, FFN={shape['ffn']}, repeated layers={shape['layers']}/60.",
        f"- Recipe: `{config['numerical_recipe']['id']}` ({config['numerical_recipe']['parameter_dtype']} parameters and activations).",
        f"- Workload: `{config['workload']}`. One conditional transformer forward; no text encoder, VAE, scheduler, checkpoint weights, or full denoising loop.",
        f"- PyTorch route: natural `{metadata['route']['torch_probe']['natural_choice_name']}`, timed treatment forced "
        f"`{metadata['route']['torch_probe']['forced_choice_name']}`; calls Flash/cuDNN: "
        f"`{metadata['route']['calls']['torch_flash']}/{metadata['route']['calls']['cudnn']}`.",
        f"- Full-model output relative L2: `{metadata['correctness']['model_output_rel_l2']:.6g}`; "
        f"right-padded B=2 mask adapter relative L2: `{metadata['correctness']['padding_adapter']['rel_l2']:.6g}`.",
        "- Joint mask semantics: every query sees valid text and every image token; only padded text key columns are rejected.",
        "",
        "## Anchors",
        "",
        f"- Model: [`{config['model_anchor']['id']}@{config['model_anchor']['revision']}`]({config['model_anchor']['config_url']})",
        f"- Implementation: [`diffusers@{config['diffusers_anchor']['commit']}`]({config['diffusers_anchor']['url']})",
        "",
        "This is a random-weight, depth-reduced transformer-shape proxy, not image-quality evidence or complete Qwen-Image pipeline throughput.",
        "",
    ]
    return "\n".join(lines)


def main():
    started_utc = _utc_now()
    args = _parse_args()
    protocol = _resolve_protocol(args)
    model_api = _load_module("qwen_image_model", MODEL_PATH)
    shape = model_api.resolve_shape(args.mode, layers=args.layers, image_tokens=args.image_tokens, text_tokens=args.text_tokens)
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0").lower() in ("1", "true", "yes", "on"):
        raise RuntimeError("disable global FROST engines: the FE arm is defined as the cuDNN backend graph")
    torch, _, cudnn, sdpamod, diffusers, qwen_module = model_api.load_runtime()
    loaded_diffusers_source = _source_record(qwen_module.__file__)
    expected_diffusers_sha = model_api.DIFFUSERS_ANCHOR["source_sha256"]
    if loaded_diffusers_source["sha256"] != expected_diffusers_sha:
        raise RuntimeError(
            "loaded Diffusers Qwen-Image source does not match the pinned implementation: "
            f"got {loaded_diffusers_source['sha256']}, expected {expected_diffusers_sha}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device, properties = _pick_device(torch, args.mode)
    with torch.cuda.device(device):
        if any(hasattr(sdpamod, name) for name in ("sdpa_fwd_d256", "sdpa_bwd_d256")):
            raise RuntimeError("loaded FE SDPA predates backend-only #682")
        padding_check = _focused_padding_check(torch, qwen_module, model_api, device)
        model = model_api.build_model(torch, qwen_module, device, layers=shape["layers"])
        inputs = model_api.make_inputs(torch, device, shape)
        calls = {"torch_flash": 0, "cudnn": 0}
        torch_probe = {}
        select, restore, calls, torch_probe = model_api.install_joint_attention_dispatch(
            qwen_module, text_tokens=shape["text_tokens"], counters=calls, torch_probe=torch_probe
        )

        def step(backend):
            select(backend)
            with torch.inference_mode():
                return model_api.forward(model, inputs)

        try:
            for bits, backend in VARIANTS:
                before = dict(calls)
                for _ in range(protocol["warmup"]):
                    step(backend)
                torch.cuda.synchronize(device)
                delta = calls[backend] - before[backend]
                expected = protocol["warmup"] * shape["layers"]
                other = "cudnn" if backend == "torch_flash" else "torch_flash"
                if delta != expected or calls[other] != before[other]:
                    raise RuntimeError(f"{backend} warm route mismatch: delta={delta}, expected={expected}, calls={calls}, before={before}")

            expected_output = step("torch_flash")
            actual_output = step("cudnn")
            torch.cuda.synchronize(device)
            if not bool(torch.isfinite(expected_output).all()) or not bool(torch.isfinite(actual_output).all()):
                raise RuntimeError("non-finite model output")
            model_rel_l2 = _rel_l2(actual_output, expected_output)
            if not math.isfinite(model_rel_l2) or model_rel_l2 > 0.02:
                raise RuntimeError(f"model output mismatch: rel_l2={model_rel_l2}")

            orders = williams_orders(2)
            raw = {bits: [] for bits, _ in VARIANTS}
            batches = {bits: [] for bits, _ in VARIANTS}
            timing_started = time.time()
            for batch in range(protocol["rounds"]):
                for variant_index in orders[batch % len(orders)]:
                    bits, backend = VARIANTS[variant_index]
                    samples = []
                    for _ in range(protocol["repeats"]):
                        select(backend)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        with torch.inference_mode():
                            output = model_api.forward(model, inputs)
                        end.record()
                        end.synchronize()
                        elapsed = start.elapsed_time(end)
                        if not math.isfinite(elapsed) or elapsed <= 0:
                            raise RuntimeError(f"invalid timing {elapsed}")
                        samples.append(elapsed)
                    raw[bits].append(samples)
                    batches[bits].append(statistics.median(samples))
                print(f"BATCH {batch + 1}/{protocol['rounds']} elapsed_s={time.time() - timing_started:.1f}", flush=True)

            paired = paired_stats(batches["1"], batches["0"])
            summary = {}
            for bits, _ in VARIANTS:
                values = batches[bits]
                result = {
                    "p10_ms": percentile(values, 0.1),
                    "p50_ms": percentile(values, 0.5),
                    "p90_ms": percentile(values, 0.9),
                    "mean_ms": statistics.mean(values),
                    "batches": len(values),
                }
                arm_paired = paired_stats(values, batches["0"])
                result.update(arm_paired)
                summary[bits] = result

            calls_per_backend = protocol["warmup"] + 1 + protocol["rounds"] * protocol["repeats"]
            expected_calls = calls_per_backend * shape["layers"]
            if calls != {"torch_reference": 0, "torch_flash": expected_calls, "cudnn": expected_calls}:
                raise RuntimeError(f"final attention route mismatch: calls={calls}, expected_each={expected_calls}")
            if torch_probe.get("forced_choice_name") != "FLASH_ATTENTION":
                raise RuntimeError(f"forced PyTorch FlashAttention treatment route changed: {torch_probe}")
        finally:
            restore()

    sources = {
        "runner": _source_record(Path(__file__)),
        "model_adapter": _source_record(MODEL_PATH),
        "statistics": _source_record(FACTORIAL_PATH),
        "diffusers_qwen_image": loaded_diffusers_source,
        "cudnn_sdpa": _source_record(sdpamod.__file__),
    }
    config = {
        "schema_version": 1,
        "mode": args.mode,
        "timing_role": "validation_only" if args.mode == "smoke" else "formal_performance",
        "performance_claim_eligible": args.mode == "formal",
        "device": properties.name,
        "device_id": str(device),
        "sm_arch": f"sm_{properties.major}{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "unknown",
        "cudnn_frontend": cudnn.__version__,
        "cudnn_backend": cudnn.backend_version(),
        "diffusers": getattr(diffusers, "__version__", "unknown"),
        "shape": shape,
        "protocol": protocol,
        "workload": "single_conditional_transformer_forward_no_checkpoint",
        "weights": "random_init_seed_0",
        "inputs": "random_latents_and_precomputed_text_embeddings_seed_1234",
        "numerical_claim_eligible": False,
        "numerical_recipe": dict(model_api.NUMERICAL_RECIPE),
        "model_anchor": dict(model_api.OFFICIAL_MODEL),
        "diffusers_anchor": dict(model_api.DIFFUSERS_ANCHOR),
        "variants": [{"bits": bits, "backend": backend} for bits, backend in VARIANTS],
        "williams_orders": orders,
    }
    comparable = {
        key: config[key]
        for key in (
            "schema_version",
            "mode",
            "timing_role",
            "performance_claim_eligible",
            "device",
            "sm_arch",
            "sm_count",
            "python",
            "torch",
            "torch_cuda",
            "cudnn_backend",
            "diffusers",
            "shape",
            "protocol",
            "workload",
            "weights",
            "inputs",
            "numerical_recipe",
            "model_anchor",
            "diffusers_anchor",
            "variants",
            "williams_orders",
        )
    }
    build = {"schema_version": 1, "git": _git_provenance(), "sources": {name: value["sha256"] for name, value in sorted(sources.items())}}
    config["comparability_fingerprint"] = {"inputs": comparable, "sha256": config_fingerprint(comparable)}
    config["build_fingerprint"] = {"inputs": build, "sha256": config_fingerprint(build)}
    metadata = {
        "schema_version": 1,
        "started_utc": started_utc,
        "completed_utc": _utc_now(),
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "config": config,
        "correctness": {"model_output_rel_l2": model_rel_l2, "padding_adapter": padding_check},
        "summary": summary,
        "comparison_within_run": paired,
        "batch_medians_ms": batches,
        "raw_ms": raw,
        "route": {"calls": calls, "expected_calls_each": expected_calls, "torch_probe": torch_probe},
        "provenance": {"git": build["git"], "sources": sources},
    }
    if args.compare is not None:
        if args.mode != "formal":
            raise ValueError("--compare is formal-only")
        metadata["comparison_across_runs"] = _compare(metadata, _load_comparison(args.compare))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"-{args.tag}" if args.tag else ""
    raw_path = args.output_dir / f"qwen-image-bf16-{args.mode}-{stamp}{suffix}.json"
    report_path = raw_path.with_suffix(".md")
    if raw_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact {raw_path} or {report_path}")
    metadata["artifacts"] = {"raw_json": str(raw_path), "markdown": str(report_path)}
    raw_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    raw_hash = _sha256(raw_path)
    report_path.write_text(_render_markdown(metadata, raw_path.name, raw_hash), encoding="utf-8")
    print("RESULT " + json.dumps({"torch_flash_p50_ms": summary["0"]["p50_ms"], "cudnn_p50_ms": summary["1"]["p50_ms"], **paired}, sort_keys=True))
    print(f"RAW_JSON {raw_path} sha256={raw_hash}")
    print(f"MARKDOWN {report_path} sha256={_sha256(report_path)}")


if __name__ == "__main__":
    main()
