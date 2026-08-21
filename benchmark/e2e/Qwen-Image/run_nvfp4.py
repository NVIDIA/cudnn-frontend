#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Balanced Qwen-Image BF16/cuDNN/ModelOpt-NVFP4 three-arm benchmark.

The low-precision arm aligns its Linear placement and NVFP4/max policy with
ModelOpt 0.46.0's Qwen-Image recipe: all fourteen Linear roles are NVFP4 while
the attention core stays BF16 (``quantize_mha=False``).  Explicit BF16,
synthetic-calibration, and depth-reduction overrides make this performance
evidence, not an exact ModelOpt state or image-quality claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
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
ADAPTER_PATH = THIS_DIR / "modelopt_nvfp4.py"
FACTORIAL_PATH = E2E_DIR / "_factorial.py"
sys.path.insert(0, str(E2E_DIR))

from _factorial import config_fingerprint, paired_stats, percentile  # noqa: E402

PROTOCOL_DEFAULTS = {
    "smoke": {"warmup": 1, "rounds": 6, "repeats": 1},
    "formal": {"warmup": 3, "rounds": 42, "repeats": 3},
}

ARMS = (
    {
        "id": "A",
        "name": "bf16_off",
        "linears": "Torch BF16",
        "mlp": "pinned Diffusers Torch GELU FFN",
        "attention": "forced PyTorch Flash BF16",
        "attention_route": "torch_flash",
    },
    {
        "id": "B",
        "name": "bf16_cudnn",
        "linears": "Torch BF16 outside FFN",
        "mlp": "cudnn.gemm.ops.gelu_mlp BF16",
        "attention": "cuDNN BF16",
        "attention_route": "cudnn",
    },
    {
        "id": "C",
        "name": "modelopt046_nvfp4_cudnn",
        "linears": "cuDNN FROST NVFP4, all 14 roles",
        "mlp": "NVFP4 FC1 + BF16 bias/GELU + fused NVFP4 hidden requant + NVFP4 FC2",
        "attention": "cuDNN BF16 (ModelOpt quantize_mha=False)",
        "attention_route": "cudnn",
    },
)


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
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "qwen-image-nvfp4-results")
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
    if protocol["rounds"] % 6:
        raise ValueError("rounds must be a multiple of 6 for the balanced three-treatment design")
    return protocol


def _pick_device(torch, mode):
    candidates = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        candidates.append(f"cuda:{index}={properties.name}/{properties.multi_processor_count}SM")
        if (properties.major, properties.minor) == (10, 0) and (
            mode != "formal" or (properties.name == "NVIDIA B200" and properties.multi_processor_count == 148)
        ):
            return torch.device(f"cuda:{index}"), properties
    requirement = "a full 148-SM NVIDIA B200" if mode == "formal" else "an SM100 GPU"
    raise RuntimeError(f"{mode} mode requires {requirement}; visible: {', '.join(candidates)}")


def _rel_l2(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def _scale_counter_tree(value, factor):
    return {key: _scale_counter_tree(item, factor) if isinstance(item, dict) else int(item) * factor for key, item in value.items()}


def _add_counter_trees(*values):
    if not values:
        return {}
    keys = set(values[0])
    if any(set(value) != keys for value in values[1:]):
        raise ValueError("counter trees have different keys")
    result = {}
    for key in values[0]:
        items = [value[key] for value in values]
        result[key] = _add_counter_trees(*items) if isinstance(items[0], dict) else sum(int(item) for item in items)
    return result


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
    for arm in ARMS:
        key = arm["id"]
        new = float(current["summary"][key]["p50_ms"])
        old = float(previous["summary"][key]["p50_ms"])
        if not all(math.isfinite(value) and value > 0 for value in (new, old)):
            raise ValueError("comparison p50 values must be finite and positive")
        arms[key] = {
            "previous_p50_ms": old,
            "current_p50_ms": new,
            "change_ms": new - old,
            "change_percent": (new / old - 1.0) * 100.0,
        }
    return {"paired_across_runs": False, "arms": arms}


def _format_elapsed_effect(ratio):
    """Describe a paired elapsed-time ratio without calling regressions wins."""
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"elapsed-time ratio must be finite and positive, got {ratio!r}")
    if ratio < 1:
        return f"{1 / ratio:.3f}x speedup ({(1 - ratio) * 100:.2f}% lower elapsed time)"
    if ratio > 1:
        return f"{ratio:.3f}x slower ({(ratio - 1) * 100:.2f}% higher elapsed time)"
    return "1.000x (no elapsed-time change)"


def _render_markdown(metadata, raw_name, raw_hash):
    config = metadata["config"]
    comparisons = metadata["comparisons"]
    smoke = config["mode"] == "smoke"
    lines = [
        "# Qwen-Image ModelOpt 0.46 NVFP4/max benchmark",
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
        ba = comparisons["B_vs_A"]
        cb = comparisons["C_vs_B"]
        ca = comparisons["C_vs_A"]
        lines += [
            "## Result",
            "",
            f"- BF16 cuDNN effect (B/A): {_format_elapsed_effect(ba['paired_ratio_p50'])}.",
            f"- ModelOpt NVFP4 increment (C/B): {_format_elapsed_effect(cb['paired_ratio_p50'])}.",
            f"- Total cuDNN platform impact (C/A): {_format_elapsed_effect(ca['paired_ratio_p50'])}.",
            "",
        ]
    lines += [
        "| arm | linears | GELU FFN | attention | p10 | p50 | p90 | paired ratio vs A |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        value = metadata["summary"][arm["id"]]
        lines.append(
            f"| `{arm['id']}` | {arm['linears']} | {arm['mlp']} | {arm['attention']} | "
            f"{value['p10_ms']:.3f} ms | {value['p50_ms']:.3f} ms | {value['p90_ms']:.3f} ms | "
            f"{value['paired_ratio_p50']:.5f} |"
        )
    if not smoke:
        lines += [
            "",
            "## Paired contrasts",
            "",
            "| contrast | meaning | paired ratio (p50) | elapsed-time effect | delta (p50) | wins |",
            "|---|---|---:|---:|---:|---:|",
        ]
        labels = {
            "B_vs_A": "fixed-precision cuDNN MLP + attention effect",
            "C_vs_B": "NVFP4 recipe increment; precision and expanded cuDNN Linear coverage",
            "C_vs_A": "total cuDNN platform impact",
        }
        for key, label in labels.items():
            value = comparisons[key]
            lines.append(
                f"| {key.replace('_vs_', '/')} | {label} | {value['paired_ratio_p50']:.5f} | "
                f"{_format_elapsed_effect(value['paired_ratio_p50'])} | {value['paired_delta_p50_ms']:+.3f} ms | "
                f"{value['wins']}/{value['batches']} |"
            )
    shape = config["shape"]
    recipe = config["numerical_recipe"]
    route = metadata["route"]
    correctness = metadata["correctness"]["model_output_rel_l2_vs_A"]
    contract_gate = metadata["correctness"].get("low_precision_contract_gate")
    lines += [
        "",
        "## Scope and gates",
        "",
        f"- Shape: B={shape['bs']}, image/text/joint tokens={shape['image_tokens']}/{shape['text_tokens']}/{shape['joint_tokens']}, "
        f"H={shape['hidden']}, heads={shape['heads']}x{shape['head_dim']}, FFN={shape['ffn']}, repeated layers={shape['layers']}/60.",
        f"- Proxy block mapping: `{config['representative_full_blocks']}` within ModelOpt's quantized full-model range 2..57.",
        f"- Recipe placement/format anchor: `{recipe['id']}` at ModelOpt `{recipe['release']}@{recipe['commit']}`; "
        "all fourteen Linear roles use NVFP4 E2M1/block-16 with E4M3 block scales.",
        f"- Proxy overrides: `{json.dumps(recipe['proxy_overrides'], sort_keys=True)}`. This is not an exact upstream dtype, calibration-state, or workload reproduction.",
        "- Attention remains BF16 in C. ModelOpt 0.46 `quantize_mha` defaults to false; this result is not MXFP8 or per-tensor FP8 attention.",
        "- Calibration is one deterministic synthetic BF16 max pass and is frozen before timing. Random weights/inputs make this quality-ineligible.",
        "- B/A isolates the existing BF16 cuDNN treatments. C/B is intentionally not a kernel-only contrast: it changes precision and moves all "
        "fourteen block Linear roles onto cuDNN FROST.",
        "- A turns cuDNN off only for the measured FFN/attention treatments; unrelated framework operators are unchanged.",
        f"- PyTorch SDPA probe: natural `{route['torch_probe']['natural_choice_name']}`, A forced `{route['torch_probe']['forced_choice_name']}`.",
        f"- Model output relative L2 vs A: `{json.dumps(correctness, sort_keys=True)}`. C is recorded as a finite sanity signal, not a quality gate.",
        f"- C per-forward activation quantizations: logical `{route['expected_C_per_forward']['activation_quant_logical']}`, "
        f"physical `{route['expected_C_per_forward']['activation_quant_physical']}` "
        f"(`{route['expected_C_per_forward']['activation_quant_standalone']}` standalone + "
        f"`{route['expected_C_per_forward']['activation_quant_fused']}` fused).",
        "",
        "## Provenance",
        "",
        "| source | path | sha256 |",
        "|---|---|---|",
    ]
    if contract_gate is not None:
        lines.insert(
            lines.index("## Provenance") - 1,
            f"- Low-precision implementation contract gate: `{contract_gate['status']}` across "
            f"`{contract_gate['contracts_checked']}` exact plan shapes/epilogues; this is a kernel-contract gate, not an image-quality claim.",
        )
    for name, source in sorted(metadata["provenance"]["sources"].items()):
        lines.append(f"| {name} | `{source['path']}` | `{source['sha256']}` |")
    comparison = metadata.get("comparison_across_runs")
    if comparison is not None:
        lines += [
            "",
            "## Cross-run comparison",
            "",
            "**Not paired across runs.** Each row compares independent p50 estimates.",
            "",
            "| arm | previous p50 | current p50 | non-paired change | change |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, value in sorted(comparison["arms"].items()):
            lines.append(
                f"| `{arm}` | {value['previous_p50_ms']:.3f} ms | {value['current_p50_ms']:.3f} ms | "
                f"{value['change_ms']:+.3f} ms | {value['change_percent']:+.2f}% |"
            )
    lines += [
        "",
        "This is a random-weight, depth-reduced transformer-backbone proxy: no text encoder, VAE, scheduler, denoising loop, or image-quality claim.",
        "",
    ]
    return "\n".join(lines)


def main():
    started_utc = _utc_now()
    args = _parse_args()
    protocol = _resolve_protocol(args)
    model_api = _load_module("qwen_image_model_nvfp4", MODEL_PATH)
    lowp = _load_module("qwen_image_modelopt_nvfp4", ADAPTER_PATH)
    shape = model_api.resolve_shape(args.mode, layers=args.layers, image_tokens=args.image_tokens, text_tokens=args.text_tokens)
    representative = lowp.representative_middle_blocks(shape["layers"])
    orders = lowp.three_arm_orders()

    # The low-precision adapter calls FROST directly.  Keeping the global engine
    # opt-in disabled prevents arm B's public BF16 GELU op from acquiring a
    # different plan population than the existing BF16 benchmark.
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0").lower() in ("1", "true", "yes", "on"):
        raise RuntimeError("disable global FROST engines; this runner invokes only its private NVFP4 FROST plans directly")

    torch, _, cudnn, sdpamod, diffusers, qwen_module = model_api.load_runtime()
    diffusers_attention_module = importlib.import_module("diffusers.models.attention")
    diffusers_activations_module = importlib.import_module("diffusers.models.activations")
    loaded_diffusers_source = _source_record(qwen_module.__file__)
    if loaded_diffusers_source["sha256"] != model_api.DIFFUSERS_ANCHOR["source_sha256"]:
        raise RuntimeError(
            "loaded Diffusers Qwen-Image source does not match the pin: "
            f"got {loaded_diffusers_source['sha256']}, expected {model_api.DIFFUSERS_ANCHOR['source_sha256']}"
        )
    loaded_supporting_sources = {
        "attention": _source_record(diffusers_attention_module.__file__),
        "activations": _source_record(diffusers_activations_module.__file__),
    }
    for name, source in loaded_supporting_sources.items():
        expected = model_api.DIFFUSERS_ANCHOR["supporting_sources"][name]["source_sha256"]
        if source["sha256"] != expected:
            raise RuntimeError(f"loaded Diffusers {name} source does not match the pin: got {source['sha256']}, expected {expected}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError("this PyTorch build has no packed float4_e2m1fn_x2 dtype")

    device, properties = _pick_device(torch, args.mode)
    with torch.cuda.device(device):
        if any(hasattr(sdpamod, name) for name in ("sdpa_fwd_d256", "sdpa_bwd_d256")):
            raise RuntimeError("loaded FE SDPA predates backend-only #682")
        padding_check = None
        # Use the same focused mask coverage as the BF16 leaf without importing
        # or changing it: dense formal timings remain mask-free.
        text_tokens, image_tokens, heads, head_dim, batch = 64, 128, 4, 128, 2
        generator = torch.Generator(device=device).manual_seed(2026)
        qkv_shape = (batch, text_tokens + image_tokens, heads, head_dim)
        q, k, v = (torch.randn(qkv_shape, device=device, dtype=torch.bfloat16, generator=generator) for _ in range(3))
        text_mask = torch.arange(text_tokens, device=device).unsqueeze(0) < torch.tensor([64, 37], device=device).unsqueeze(1)
        mask = torch.cat([text_mask, torch.ones(batch, image_tokens, dtype=torch.bool, device=device)], dim=1)[:, None, None]
        pad_select, pad_restore, _, _ = model_api.install_joint_attention_dispatch(qwen_module, text_tokens=text_tokens)
        try:
            with torch.inference_mode():
                pad_select("torch_reference")
                expected = qwen_module.dispatch_attention_fn(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
                pad_select("cudnn")
                actual = qwen_module.dispatch_attention_fn(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            torch.cuda.synchronize(device)
        finally:
            pad_restore()
        padding_rel_l2 = _rel_l2(actual, expected)
        if not math.isfinite(padding_rel_l2) or padding_rel_l2 > 0.01:
            raise RuntimeError(f"right-padded joint-attention adapter mismatch: rel_l2={padding_rel_l2}")
        padding_check = {"shape": list(qkv_shape), "text_valid_lengths": [64, 37], "rel_l2": padding_rel_l2}

        model = model_api.build_model(torch, qwen_module, device, layers=shape["layers"])
        inputs = model_api.make_inputs(torch, device, shape)
        calibration = lowp.collect_max_calibration(model, shape, lambda: model_api.forward(model, inputs))
        torch.cuda.synchronize(device)

        attention_calls = {"torch_flash": 0, "cudnn": 0}
        torch_probe = {}
        select_attention, restore_attention, attention_calls, torch_probe = model_api.install_joint_attention_dispatch(
            qwen_module, text_tokens=shape["text_tokens"], counters=attention_calls, torch_probe=torch_probe
        )
        adapter = lowp.install_modelopt_nvfp4_dispatch(model, shape, calibration)
        numerical_gate = adapter.run_focused_numerical_gate()

        arm_by_id = {arm["id"]: arm for arm in ARMS}

        def configure(arm_id):
            adapter.select(arm_id)
            select_attention(arm_by_id[arm_id]["attention_route"])

        def invoke(arm_id):
            with adapter.forward_scope(arm_id):
                with torch.inference_mode():
                    return model_api.forward(model, inputs)

        try:
            for arm in ARMS:
                arm_id = arm["id"]
                configure(arm_id)
                adapter_before = adapter.snapshot()
                attention_before = dict(attention_calls)
                for _ in range(protocol["warmup"]):
                    invoke(arm_id)
                torch.cuda.synchronize(device)
                adapter_delta = lowp.counter_delta(adapter.snapshot(), adapter_before)
                expected_adapter = _scale_counter_tree(lowp.expected_route_delta(arm_id, shape["layers"]), protocol["warmup"])
                if adapter_delta != expected_adapter:
                    raise RuntimeError(f"{arm_id} warm adapter route mismatch: delta={adapter_delta}, expected={expected_adapter}")
                attention_delta = {name: attention_calls[name] - attention_before[name] for name in attention_calls}
                expected_attention = protocol["warmup"] * shape["layers"]
                expected_attention_delta = {
                    "torch_reference": 0,
                    "torch_flash": expected_attention if arm["attention_route"] == "torch_flash" else 0,
                    "cudnn": expected_attention if arm["attention_route"] == "cudnn" else 0,
                }
                if attention_delta != expected_attention_delta:
                    raise RuntimeError(f"{arm_id} warm attention route mismatch: delta={attention_delta}, expected={expected_attention_delta}")

            outputs = {}
            for arm in ARMS:
                configure(arm["id"])
                outputs[arm["id"]] = invoke(arm["id"])
                torch.cuda.synchronize(device)
                if not bool(torch.isfinite(outputs[arm["id"]]).all()):
                    raise RuntimeError(f"non-finite model output in arm {arm['id']}")
            model_rel_l2 = {arm["id"]: _rel_l2(outputs[arm["id"]], outputs["A"]) for arm in ARMS}
            if model_rel_l2["B"] > 0.02:
                raise RuntimeError(f"BF16 cuDNN arm B mismatch: rel_l2={model_rel_l2['B']}")

            raw = {arm["id"]: [] for arm in ARMS}
            batches = {arm["id"]: [] for arm in ARMS}
            timing_started = time.time()
            for batch_index in range(protocol["rounds"]):
                for arm_index in orders[batch_index % len(orders)]:
                    arm_id = ARMS[arm_index]["id"]
                    samples = []
                    for _ in range(protocol["repeats"]):
                        configure(arm_id)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        output = invoke(arm_id)
                        end.record()
                        end.synchronize()
                        elapsed = start.elapsed_time(end)
                        if not math.isfinite(elapsed) or elapsed <= 0:
                            raise RuntimeError(f"invalid timing {elapsed}")
                        samples.append(elapsed)
                    raw[arm_id].append(samples)
                    batches[arm_id].append(statistics.median(samples))
                print(f"BATCH {batch_index + 1}/{protocol['rounds']} elapsed_s={time.time() - timing_started:.1f}", flush=True)

            summary = {}
            for arm in ARMS:
                arm_id = arm["id"]
                values = batches[arm_id]
                summary[arm_id] = {
                    "p10_ms": percentile(values, 0.1),
                    "p50_ms": percentile(values, 0.5),
                    "p90_ms": percentile(values, 0.9),
                    "mean_ms": statistics.mean(values),
                    "batches": len(values),
                    **paired_stats(values, batches["A"]),
                }
            comparisons = {
                "B_vs_A": paired_stats(batches["B"], batches["A"]),
                "C_vs_B": paired_stats(batches["C"], batches["B"]),
                "C_vs_A": paired_stats(batches["C"], batches["A"]),
            }

            calls_per_arm = protocol["warmup"] + 1 + protocol["rounds"] * protocol["repeats"]
            expected_adapter_totals = _add_counter_trees(
                *[_scale_counter_tree(lowp.expected_route_delta(arm["id"], shape["layers"]), calls_per_arm) for arm in ARMS]
            )
            expected_adapter_totals["weight_pack_calls"] = 14 * shape["layers"]
            expected_adapter_totals["plan_build_calls"] = 7
            if adapter.snapshot() != expected_adapter_totals:
                raise RuntimeError(f"final adapter route mismatch: got={adapter.snapshot()}, expected={expected_adapter_totals}")
            expected_attention_calls = {
                "torch_reference": 0,
                "torch_flash": calls_per_arm * shape["layers"],
                "cudnn": 2 * calls_per_arm * shape["layers"],
            }
            if attention_calls != expected_attention_calls:
                raise RuntimeError(f"final attention route mismatch: got={attention_calls}, expected={expected_attention_calls}")
            if torch_probe.get("forced_choice_name") != "FLASH_ATTENTION":
                raise RuntimeError(f"forced PyTorch FlashAttention treatment route changed: {torch_probe}")
        finally:
            adapter.restore()
            restore_attention()

    quant_module = importlib.import_module("cudnn.gemm.ops._nvfp4_quantize")
    gelu_module = importlib.import_module("cudnn.gemm.ops._gelu_mlp")
    frost_module_names = (
        "compiler",
        "dtypes",
        "epilogue_codegen",
        "fusion_ir",
        "graph_analyzer",
        "kernel_registry",
        "recipe",
        "tile_config",
    )
    frost_modules = {name: importlib.import_module(f"cudnn.gemm.frost.{name}") for name in frost_module_names}
    quant_csrc = Path(quant_module.__file__).with_name("csrc")
    sources = {
        "runner": _source_record(Path(__file__)),
        "model_adapter": _source_record(MODEL_PATH),
        "modelopt_adapter": _source_record(ADAPTER_PATH),
        "statistics": _source_record(FACTORIAL_PATH),
        "diffusers_qwen_image": loaded_diffusers_source,
        "diffusers_attention": loaded_supporting_sources["attention"],
        "diffusers_activations": loaded_supporting_sources["activations"],
        "cudnn_sdpa": _source_record(sdpamod.__file__),
        "cudnn_gelu_mlp": _source_record(gelu_module.__file__),
        "cudnn_nvfp4_quantize": _source_record(quant_module.__file__),
        "cudnn_nvfp4_quantize_cuda": _source_record(quant_csrc / "nvfp4_quantize_sm100.cu"),
        "cudnn_nvfp4_quantize_header": _source_record(quant_csrc / "nvfp4_smooth_quantize_sm100.cuh"),
    }
    sources.update({f"cudnn_frost_{name}": _source_record(module.__file__) for name, module in frost_modules.items()})
    for index, plan in enumerate(adapter.plan_provenance()):
        sources[f"cudnn_frost_generated_plan_{index:02d}"] = _source_record(plan["generated_path"])
    config = {
        "schema_version": 1,
        "mode": args.mode,
        "timing_role": "validation_only" if args.mode == "smoke" else "formal_performance",
        "performance_claim_eligible": args.mode == "formal",
        "numerical_claim_eligible": False,
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
        "representative_full_blocks": representative,
        "protocol": protocol,
        "workload": "single_conditional_transformer_forward_no_checkpoint",
        "weights": "random_init_seed_0; bf16 originals retained; NVFP4 weights prepacked before timing",
        "inputs": "random_latents_and_precomputed_text_embeddings_seed_1234",
        "numerical_recipe": lowp.MODELOPT_RECIPE,
        "model_anchor": dict(model_api.OFFICIAL_MODEL),
        "diffusers_anchor": dict(model_api.DIFFUSERS_ANCHOR),
        "arms": [{key: value for key, value in arm.items() if key != "attention_route"} for arm in ARMS],
        "balanced_orders": orders,
    }
    comparable_keys = (
        "schema_version",
        "mode",
        "timing_role",
        "performance_claim_eligible",
        "numerical_claim_eligible",
        "device",
        "sm_arch",
        "sm_count",
        "python",
        "torch",
        "torch_cuda",
        "cudnn_backend",
        "diffusers",
        "shape",
        "representative_full_blocks",
        "protocol",
        "workload",
        "weights",
        "inputs",
        "numerical_recipe",
        "model_anchor",
        "diffusers_anchor",
        "arms",
        "balanced_orders",
    )
    comparable = {key: config[key] for key in comparable_keys}
    build = {"schema_version": 1, "git": _git_provenance(), "sources": {name: value["sha256"] for name, value in sorted(sources.items())}}
    config["comparability_fingerprint"] = {"inputs": comparable, "sha256": config_fingerprint(comparable)}
    config["build_fingerprint"] = {"inputs": build, "sha256": config_fingerprint(build)}
    metadata = {
        "schema_version": 1,
        "started_utc": started_utc,
        "completed_utc": _utc_now(),
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "config": config,
        "correctness": {
            "model_output_rel_l2_vs_A": model_rel_l2,
            "low_precision_quality_gate": False,
            "low_precision_contract_gate": numerical_gate,
            "padding_adapter": padding_check,
        },
        "summary": summary,
        "comparisons": comparisons,
        "batch_medians_ms": batches,
        "raw_ms": raw,
        "route": {
            "attention_calls": attention_calls,
            "adapter_calls": adapter.snapshot(),
            "expected_C_per_forward": lowp.expected_route_delta("C", shape["layers"]),
            "torch_probe": torch_probe,
        },
        "quantization": adapter.metadata(),
        "provenance": {"git": build["git"], "sources": sources},
    }
    if args.compare is not None:
        if args.mode != "formal":
            raise ValueError("--compare is formal-only")
        metadata["comparison_across_runs"] = _compare(metadata, _load_comparison(args.compare))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"-{args.tag}" if args.tag else ""
    raw_path = args.output_dir / f"qwen-image-modelopt-nvfp4-{args.mode}-{stamp}{suffix}.json"
    report_path = raw_path.with_suffix(".md")
    if raw_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact {raw_path} or {report_path}")
    metadata["artifacts"] = {"raw_json": str(raw_path), "markdown": str(report_path)}
    raw_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    raw_hash = _sha256(raw_path)
    report_path.write_text(_render_markdown(metadata, raw_path.name, raw_hash), encoding="utf-8")
    print(
        "RESULT "
        + json.dumps(
            {
                "A_p50_ms": summary["A"]["p50_ms"],
                "B_p50_ms": summary["B"]["p50_ms"],
                "C_p50_ms": summary["C"]["p50_ms"],
                "B_over_A": comparisons["B_vs_A"]["paired_ratio_p50"],
                "C_over_B": comparisons["C_vs_B"]["paired_ratio_p50"],
                "C_over_A": comparisons["C_vs_A"]["paired_ratio_p50"],
            },
            sort_keys=True,
        )
    )
    print(f"RAW_JSON {raw_path} sha256={raw_hash}")
    print(f"MARKDOWN {report_path} sha256={_sha256(report_path)}")


if __name__ == "__main__":
    main()
