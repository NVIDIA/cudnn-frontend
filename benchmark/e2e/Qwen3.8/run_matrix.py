#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Balanced 2^3 Qwen3.8 proxy benchmark for GDN, SwiGLU, and SDPA.

The three bits in every treatment are GDN / MLP / full-attention SDPA:

* 0: stock FLA GDN, stock FLA MLP, or vanilla Torch SDPA
* 1: cuDNN FLA shim, cuDNN SwiGLU MLP, or FE's cuDNN-backend SDPA

All eight arms run in a Williams design.  Compilation, warmup, route checks,
and aggregate BF16 correctness checks are outside CUDA-event timing.  Every
successful run writes raw JSON plus a generated Markdown summary with source
provenance and a deterministic configuration fingerprint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

torch = None  # Imported after argparse so ``--help`` works without GPU dependencies.

THIS_DIR = Path(__file__).resolve().parent
E2E_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2]
RUN_MODEL = THIS_DIR / "run_model.py"
FACTORIAL_MODULE = E2E_DIR / "_factorial.py"
sys.path.insert(0, str(E2E_DIR))

from _factorial import (  # noqa: E402
    compare_results,
    config_fingerprint,
    factorial_main_effects,
    paired_stats,
    percentile,
    render_markdown,
    shapley_savings,
    williams_orders,
)


@dataclass(frozen=True)
class Variant:
    bits: str
    label: str
    gdn: bool
    mlp: bool
    attn: bool

    @property
    def name(self):
        return f"{self.bits}_{self.label}"


# Integer index equals binary G/M/A bits: G=4, M=2, A=1.
VARIANTS = (
    Variant("000", "baseline", False, False, False),
    Variant("001", "attn", False, False, True),
    Variant("010", "mlp", False, True, False),
    Variant("011", "mlp_attn", False, True, True),
    Variant("100", "gdn", True, False, False),
    Variant("101", "gdn_attn", True, False, True),
    Variant("110", "gdn_mlp", True, True, False),
    Variant("111", "all", True, True, True),
)
BY_BITS = {variant.bits: variant for variant in VARIANTS}

PAIR_BITS = (
    ("gdn_on_baseline", "100", "000"),
    ("gdn_on_attn", "101", "001"),
    ("gdn_on_mlp", "110", "010"),
    ("gdn_on_mlp_attn", "111", "011"),
    ("mlp_on_baseline", "010", "000"),
    ("mlp_on_attn", "011", "001"),
    ("mlp_on_gdn", "110", "100"),
    ("mlp_on_gdn_attn", "111", "101"),
    ("attn_on_baseline", "001", "000"),
    ("attn_on_mlp", "011", "010"),
    ("attn_on_gdn", "101", "100"),
    ("attn_on_gdn_mlp", "111", "110"),
    ("all_vs_baseline", "111", "000"),
)

MODE_DEFAULTS = {
    # Same kernel dimensions and four-layer 3:1 hybrid period as formal mode;
    # only token work and the fixed embedding/LM-head allocation are reduced.
    "smoke": {"seq": 128, "bs": 1, "vocab": 1024, "warmup": 1, "rounds": 8, "repeats": 1},
    "formal": {"seq": 2048, "bs": 4, "vocab": None, "warmup": 3, "rounds": 40, "repeats": 3},
}
DEFAULT_PRESET = "qwen3.8-27b"

ROUTE_CONTRACT = {
    "torch_full_attention": {
        "can_use_flash": True,
        "can_use_cudnn": False,
        "flash_sdp_enabled": True,
        "required_grad_fn": "ScaledDotProductFlashAttentionBackward0",
    },
    "cudnn_full_attention": {
        "adapter": "FE direct cuDNN-backend d256 SDPA",
        "route": "backend graph only after FE #682",
        "minimum_cudnn_backend": 92300,
    },
    "gdn": {
        "baseline": "stock FLA",
        "accelerated": "cudnn.fla native gated_delta_net",
        "successful_native_calls_per_step": "one per GDN layer when enabled, zero when disabled",
    },
    "mlp": {
        "baseline": "stock FLA GatedMLP",
        "accelerated": "cudnn.gemm.ops.swiglu_mlp",
        "backward": "FROST dgrad+dSwiGLU",
        "pointwise_fallback_calls": 0,
    },
}

MEASUREMENT_CONTRACT = {
    "timed_region": "model forward + fused cross entropy + backward",
    "outside_timed_region": ["backend selection", "zero_grad", "JIT/autotune", "warmup", "correctness"],
    "per_arm_batch_value": "median of CUDA-event repeats",
    "schedule": "eight-treatment Williams design",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _source_record(path):
    path = Path(path).resolve()
    return {"path": _display_path(path), "sha256": _sha256(path)}


def _strict_json_load(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value!r} is not allowed")

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot load comparison artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"comparison artifact {path} must contain a JSON object")
    return payload


def _git_provenance():
    def run(*arguments):
        completed = subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("git", "rev-parse", "HEAD")
        dirty = bool(run("git", "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", "unknown"
    return {"commit": commit, "dirty": dirty}


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_run_model():
    spec = importlib.util.spec_from_file_location("qwen38_run_model_matrix", RUN_MODEL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling runner {RUN_MODEL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grad_fn_names(tensor, limit=64):
    pending = [tensor.grad_fn]
    seen = set()
    names = []
    while pending and len(seen) < limit:
        node = pending.pop(0)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        names.append(type(node).__name__)
        pending.extend(next_node for next_node, _ in node.next_functions)
    return names


def _validate_torch_baseline_route(probe):
    contract = ROUTE_CONTRACT["torch_full_attention"]
    failures = []
    for name in ("can_use_flash", "can_use_cudnn", "flash_sdp_enabled"):
        if probe.get(name) is not contract[name]:
            failures.append(f"{name}={probe.get(name)!r}, expected {contract[name]!r}")
    required_grad_fn = contract["required_grad_fn"]
    if required_grad_fn not in probe.get("grad_fn_chain", ()):
        failures.append(f"grad_fn_chain lacks {required_grad_fn}")
    if failures:
        raise RuntimeError("Torch baseline route contract failed: " + "; ".join(failures))


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=MODE_DEFAULTS,
        default="smoke",
        help="smoke keeps Qwen kernel dimensions but reduces tokens; formal requires a full 148-SM B200",
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--linear-heads", type=int)
    parser.add_argument("--linear-v-heads", type=int)
    parser.add_argument("--linear-head-dim", type=int)
    parser.add_argument("--attn-heads", type=int)
    parser.add_argument("--attn-kv-heads", type=int)
    parser.add_argument("--attn-every", type=int)
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--short-conv", type=int, choices=(0, 1))
    parser.add_argument("--seq", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("qwen3.8-factorial-results"),
        help="directory for timestamped artifacts when explicit paths are omitted",
    )
    parser.add_argument("--artifact-name", help="filename stem for default artifact paths")
    parser.add_argument("--raw-json", type=Path, help="exact raw JSON path")
    parser.add_argument("--markdown", type=Path, help="exact generated Markdown path")
    parser.add_argument(
        "--compare",
        type=Path,
        help="previous formal JSON artifact; requires an identical comparability fingerprint",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace explicitly colliding artifact paths")
    args = parser.parse_args()

    defaults = MODE_DEFAULTS[args.mode]
    for name in ("seq", "bs", "warmup", "rounds", "repeats"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])
    if args.vocab is None and defaults["vocab"] is not None:
        args.vocab = defaults["vocab"]

    if args.rounds <= 0 or args.rounds % len(VARIANTS):
        raise ValueError(f"--rounds must be a positive multiple of {len(VARIANTS)}")
    if args.repeats <= 0 or args.warmup < 1:
        raise ValueError("--repeats must be positive and --warmup must be at least one")
    if args.seq <= 0 or args.bs <= 0:
        raise ValueError("--seq and --bs must be positive")
    if args.artifact_name and Path(args.artifact_name).name != args.artifact_name:
        raise ValueError("--artifact-name must be a filename stem, not a path")
    if args.compare is not None and args.mode != "formal":
        raise ValueError("--compare is formal-only; smoke timings are validation diagnostics, not performance trends")
    return args


def _resolve_shape(args, qwen):
    if args.preset not in qwen.MODEL_PRESETS:
        raise ValueError(f"unknown preset {args.preset!r}; choices={tuple(qwen.MODEL_PRESETS)}")
    shape = qwen.MODEL_PRESETS[args.preset].copy()
    for argument_name, shape_name in (
        ("layers", "layers"),
        ("hidden", "hidden"),
        ("intermediate", "intermediate"),
        ("linear_heads", "linear_heads"),
        ("linear_v_heads", "linear_v_heads"),
        ("linear_head_dim", "linear_head_dim"),
        ("attn_heads", "attn_heads"),
        ("attn_kv_heads", "attn_kv_heads"),
        ("attn_every", "attn_every"),
        ("vocab", "vocab"),
        ("short_conv", "short_conv"),
    ):
        value = getattr(args, argument_name)
        if value is not None:
            shape[shape_name] = bool(value) if shape_name == "short_conv" else value
    return shape


def _mode_overrides(args, qwen, shape):
    expected_shape = qwen.MODEL_PRESETS[args.preset].copy()
    default_vocab = MODE_DEFAULTS[args.mode]["vocab"]
    if default_vocab is not None:
        expected_shape["vocab"] = default_vocab

    overrides = {}
    if args.preset != DEFAULT_PRESET:
        overrides["preset"] = {"mode_default": DEFAULT_PRESET, "actual": args.preset}
    shape_changes = {name: {"mode_default": expected_shape[name], "actual": value} for name, value in shape.items() if value != expected_shape[name]}
    if shape_changes:
        overrides["shape"] = shape_changes
    measurement_changes = {
        name: {"mode_default": MODE_DEFAULTS[args.mode][name], "actual": getattr(args, name)}
        for name in ("seq", "bs", "warmup", "rounds", "repeats")
        if getattr(args, name) != MODE_DEFAULTS[args.mode][name]
    }
    if measurement_changes:
        overrides["measurement"] = measurement_changes
    return overrides


def _prepare_artifacts(args, fingerprint):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.artifact_name or f"qwen38_{args.mode}_{stamp}_{fingerprint[:12]}"
    raw_path = args.raw_json or args.output_dir / f"{stem}.json"
    markdown_path = args.markdown or args.output_dir / f"{stem}.md"
    raw_path = Path(raw_path)
    markdown_path = Path(markdown_path)
    if raw_path.resolve() == markdown_path.resolve():
        raise ValueError("raw JSON and Markdown paths must be different")
    if not args.overwrite:
        collisions = [str(path) for path in (raw_path, markdown_path) if path.exists()]
        if collisions:
            raise FileExistsError(f"artifact path already exists; pass --overwrite to replace: {collisions}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    return raw_path, markdown_path


def _serializable_args(args):
    return {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}


def _pick_device(mode):
    visible = []
    smoke_candidate = None
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        visible.append(f"cuda:{index}={properties.name}/{properties.multi_processor_count}SM")
        if properties.name == "NVIDIA B200" and properties.multi_processor_count == 148:
            if mode == "formal":
                return torch.device(f"cuda:{index}")
            smoke_candidate = smoke_candidate or torch.device(f"cuda:{index}")
        elif 100 <= properties.major * 10 + properties.minor < 120 and smoke_candidate is None:
            smoke_candidate = torch.device(f"cuda:{index}")
    if mode == "smoke" and smoke_candidate is not None:
        return smoke_candidate
    requirement = "a full 148-SM NVIDIA B200" if mode == "formal" else "an SM100-family Blackwell GPU"
    raise RuntimeError(f"{mode} mode requires {requirement}; visible devices: " + ", ".join(visible))


def _run_experiment(args, qwen, device, properties, orders, started_utc):
    import cudnn
    from cudnn import _env as cudnn_env
    import cudnn.experimental.ops.sdpa as sdpamod
    import cudnn.fla as cfla
    import fla
    import fla.layers.attn as fla_attn
    import fla.modules.mlp as fla_mlp

    if not hasattr(qwen, "_set_full_attention_backend"):
        raise RuntimeError(f"{RUN_MODEL} lacks _set_full_attention_backend")
    backend_floor = 92300
    if cudnn.backend_version() < backend_floor:
        raise RuntimeError("d256 FE arm requires cuDNN backend " f">= {backend_floor}; got {cudnn.backend_version()}")
    if any(hasattr(sdpamod, name) for name in ("sdpa_fwd_d256", "sdpa_bwd_d256")):
        raise RuntimeError("loaded FE SDPA module predates #682 and still exposes the legacy standalone d256 stacks")

    from cudnn.gemm.ops import swiglu_mlp as public_swiglu_mlp

    opmod = sys.modules[public_swiglu_mlp.__module__]
    gdnmod = importlib.import_module("cudnn.fla.gated_delta_rule")
    real_native_gdn = gdnmod.gated_delta_net
    native_gdn_module = sys.modules[real_native_gdn.__module__]
    frost_compiler_module = importlib.import_module("cudnn.gemm.frost.compiler")
    frost_tile_config_module = importlib.import_module("cudnn.gemm.frost.tile_config")
    frost_kernel_registry_module = importlib.import_module("cudnn.gemm.frost.kernel_registry")
    gated_mlp_adapter_module = importlib.import_module("cudnn.fla.gated_mlp")
    shape = _resolve_shape(args, qwen)
    mode_overrides = _mode_overrides(args, qwen, shape)

    torch.manual_seed(0)
    model, attn_layers = qwen.build_model(device, **shape)
    if not attn_layers:
        raise RuntimeError("the three-axis benchmark requires at least one full-attention layer")
    linear_layers = [index for index in range(shape["layers"]) if index not in attn_layers]
    if not linear_layers:
        raise RuntimeError("the three-axis benchmark requires at least one GDN layer")
    ids = torch.randint(0, shape["vocab"], (args.bs, args.seq), device=device)

    qwen._set_full_attention_backend("torch")
    torch_attn_adapter = fla_attn.flash_attn_func
    qwen._set_full_attention_backend("cudnn")
    cudnn_attn_adapter = fla_attn.flash_attn_func
    if torch_attn_adapter is cudnn_attn_adapter:
        raise RuntimeError("Torch and FE attention selectors resolved to the same adapter")

    attn_calls = {"torch": 0, "cudnn": 0}
    torch_sdpa_probe = {}

    def counted_torch_attn(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        **kwargs,
    ):
        attn_calls["torch"] += 1
        if dropout_p != 0.0:
            raise RuntimeError("the balanced backend A/B requires full-attention dropout_p=0")
        if not torch_sdpa_probe:
            qt, kt, vt = (tensor.transpose(1, 2) for tensor in (q, k, v))
            enable_gqa = qt.shape[1] != kt.shape[1]
            params = torch.backends.cuda.SDPAParams(
                qt,
                kt,
                vt,
                None,
                float(dropout_p),
                bool(causal),
                enable_gqa,
            )
            torch_sdpa_probe.update(
                {
                    "can_use_cudnn": torch.backends.cuda.can_use_cudnn_attention(params),
                    "can_use_flash": torch.backends.cuda.can_use_flash_attention(params),
                    "can_use_efficient": torch.backends.cuda.can_use_efficient_attention(params),
                    "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
                    "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                    "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
                    "shape_q": tuple(qt.shape),
                    "shape_k": tuple(kt.shape),
                    "shape_v": tuple(vt.shape),
                    "enable_gqa": enable_gqa,
                }
            )
            if torch_sdpa_probe["can_use_cudnn"]:
                raise RuntimeError("vanilla Torch now admits cuDNN attention for this d256 GQA shape; " "the intended dispatcher-gap axis has collapsed")
        output = torch_attn_adapter(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs,
        )
        if "grad_fn_chain" not in torch_sdpa_probe:
            torch_sdpa_probe["grad_fn_chain"] = _grad_fn_names(output)
        return output

    def counted_cudnn_attn(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        **kwargs,
    ):
        attn_calls["cudnn"] += 1
        if dropout_p != 0.0:
            raise RuntimeError("the balanced backend A/B requires full-attention dropout_p=0")
        return cudnn_attn_adapter(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            **kwargs,
        )

    counted_attn_adapters = {"torch": counted_torch_attn, "cudnn": counted_cudnn_attn}

    frost_calls = 0
    pointwise_calls = 0
    native_gdn_calls = 0
    real_frost = opmod._frost_dswiglu
    real_pointwise = opmod._dswiglu

    def counted_frost(*arguments, **kwargs):
        nonlocal frost_calls
        frost_calls += 1
        return real_frost(*arguments, **kwargs)

    def counted_pointwise(*arguments, **kwargs):
        nonlocal pointwise_calls
        pointwise_calls += 1
        return real_pointwise(*arguments, **kwargs)

    def counted_native_gdn(*arguments, **kwargs):
        nonlocal native_gdn_calls
        result = real_native_gdn(*arguments, **kwargs)
        # Count only a successful native call. A pre-native decline or an
        # exception followed by FLA fallback must leave a deficit.
        native_gdn_calls += 1
        return result

    opmod._frost_dswiglu = counted_frost
    opmod._dswiglu = counted_pointwise
    gdnmod.gated_delta_net = counted_native_gdn

    def select(variant):
        cfla.restore_fla(targets=("gated_delta_rule", "gated_mlp"))
        selected_fla_targets = []
        if variant.gdn:
            selected_fla_targets.append("gated_delta_rule")
        if variant.mlp:
            selected_fla_targets.append("gated_mlp")
        if selected_fla_targets:
            cfla.accelerate_fla(verbose=False, targets=selected_fla_targets)
        attention_backend = "cudnn" if variant.attn else "torch"
        qwen._set_full_attention_backend(attention_backend)
        # Keep run_model's selector authoritative, then add symmetric telemetry.
        fla_attn.flash_attn_func = counted_attn_adapters[attention_backend]

    def step(variant):
        select(variant)
        model.zero_grad(set_to_none=True)
        output = model(input_ids=ids, labels=ids)
        output.loss.backward()
        return output

    frost_template_dir = Path(frost_compiler_module.__file__).resolve().parent / "kernel_templates"
    sources = {
        "factorial_runner": _source_record(Path(__file__)),
        "factorial_statistics": _source_record(FACTORIAL_MODULE),
        "qwen_run_model": _source_record(RUN_MODEL),
        "swiglu_mlp": _source_record(opmod.__file__),
        "frost_dswiglu": _source_record(opmod.__file__),
        "frost_compiler": _source_record(frost_compiler_module.__file__),
        "frost_tile_config": _source_record(frost_tile_config_module.__file__),
        "frost_kernel_registry": _source_record(frost_kernel_registry_module.__file__),
        "frost_kernel_template_1ctamma": _source_record(frost_template_dir / "sm100_matmul_1ctamma.py"),
        "frost_kernel_template_2ctamma": _source_record(frost_template_dir / "sm100_matmul_2ctamma.py"),
        "frost_mainloop_template_1ctamma": _source_record(frost_template_dir / "sm100_matmul_mainloop_1ctamma.py"),
        "frost_mainloop_template_2ctamma": _source_record(frost_template_dir / "sm100_matmul_mainloop_2ctamma.py"),
        "cudnn_fla": _source_record(cfla.__file__),
        "cudnn_fla_gdn_adapter": _source_record(gdnmod.__file__),
        "cudnn_fla_gated_mlp_adapter": _source_record(gated_mlp_adapter_module.__file__),
        "native_gdn": _source_record(native_gdn_module.__file__),
        "sdpa": _source_record(sdpamod.__file__),
        "cudnn_init": _source_record(cudnn.__file__),
        "fla_init": _source_record(fla.__file__),
        "fla_mlp": _source_record(fla_mlp.__file__),
        "fla_attn": _source_record(fla_attn.__file__),
    }
    provenance = {
        "git": _git_provenance(),
        "sources": sources,
        "d256_route_contract": "FE public backend graph only; legacy standalone stacks absent after #682",
    }
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    config = {
        "schema_version": 3,
        "mode": args.mode,
        "timing_role": "validation_only" if args.mode == "smoke" else "formal_performance",
        "performance_claim_eligible": args.mode == "formal",
        "device": properties.name,
        "device_id": f"cuda:{device_index}",
        "sm_arch": f"sm_{properties.major}{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "unknown",
        "cuda_driver": cudnn_env.driver_version(),
        "cuda_runtime": cudnn_env.runtime_version(),
        "cudnn_frontend": cudnn.__version__,
        "cudnn_backend": cudnn.backend_version(),
        "fla": getattr(fla, "__version__", "unknown"),
        "preset": args.preset,
        "seq": args.seq,
        "bs": args.bs,
        "resolved_shape": shape,
        "attn_layers": attn_layers,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "warmup": args.warmup,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "mode_overrides": mode_overrides,
        "variants": [asdict(variant) | {"name": variant.name} for variant in VARIANTS],
        "williams_orders": orders,
        "route_contract": ROUTE_CONTRACT,
        "measurement_contract": MEASUREMENT_CONTRACT,
        "numerical_recipe": dict(qwen.NUMERICAL_RECIPE),
        "provenance": provenance,
    }
    comparability_inputs = {
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
            "cuda_driver",
            "cuda_runtime",
            "cudnn_backend",
            "fla",
            "preset",
            "seq",
            "bs",
            "resolved_shape",
            "attn_layers",
            "params",
            "warmup",
            "rounds",
            "repeats",
            "mode_overrides",
            "variants",
            "williams_orders",
            "route_contract",
            "measurement_contract",
            "numerical_recipe",
        )
    }
    build_inputs = {
        "schema_version": 1,
        "git": provenance["git"],
        "cudnn_frontend": config["cudnn_frontend"],
        "source_sha256": {name: source["sha256"] for name, source in sorted(sources.items())},
    }
    config["comparability_fingerprint"] = {
        "inputs": comparability_inputs,
        "sha256": config_fingerprint(comparability_inputs),
    }
    config["build_fingerprint"] = {
        "inputs": build_inputs,
        "sha256": config_fingerprint(build_inputs),
    }
    previous_metadata = _strict_json_load(args.compare) if args.compare is not None else None
    if previous_metadata is not None:
        try:
            previous_fingerprint = previous_metadata["config"]["comparability_fingerprint"]["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"comparison artifact {args.compare} lacks a comparability fingerprint") from error
        current_fingerprint = config["comparability_fingerprint"]["sha256"]
        if previous_fingerprint != current_fingerprint:
            raise ValueError("comparison artifact is not comparable: " f"current={current_fingerprint}, previous={previous_fingerprint}")
    raw_path, markdown_path = _prepare_artifacts(args, config["build_fingerprint"]["sha256"])
    print("CONFIG " + json.dumps(config, sort_keys=True, allow_nan=False))

    # Warm every combination outside timing. Per-arm deltas prove that selecting
    # one axis does not accidentally mutate another.
    for variant in VARIANTS:
        before_attn = dict(attn_calls)
        before_frost = frost_calls
        before_pointwise = pointwise_calls
        before_native_gdn = native_gdn_calls
        for _ in range(args.warmup):
            step(variant)
        torch.cuda.synchronize(device)
        attention_backend = "cudnn" if variant.attn else "torch"
        expected_attn = args.warmup * len(attn_layers)
        for backend in ("torch", "cudnn"):
            observed = attn_calls[backend] - before_attn[backend]
            expected = expected_attn if backend == attention_backend else 0
            if observed != expected:
                raise RuntimeError(f"{variant.name}: {backend} attention calls {observed}, expected {expected}")
        expected_frost = args.warmup * shape["layers"] if variant.mlp else 0
        if frost_calls - before_frost != expected_frost:
            raise RuntimeError(f"{variant.name}: FROST calls {frost_calls - before_frost}, expected {expected_frost}")
        if pointwise_calls != before_pointwise:
            raise RuntimeError(f"{variant.name}: unexpected pointwise dSwiGLU fallback")
        expected_native_gdn = args.warmup * len(linear_layers) if variant.gdn else 0
        observed_native_gdn = native_gdn_calls - before_native_gdn
        if observed_native_gdn != expected_native_gdn:
            raise RuntimeError(f"{variant.name}: successful native GDN calls {observed_native_gdn}, expected {expected_native_gdn}")
        gdn_path = gdnmod.last_path() if variant.gdn else "off"
        if variant.gdn and gdn_path != "native":
            raise RuntimeError(f"{variant.name}: expected native GDN, got {gdn_path}")
        mlp_path = cfla.mlp_last_path() if variant.mlp else "off"
        if cfla.is_accelerated("gated_mlp") != variant.mlp:
            raise RuntimeError(f"{variant.name}: gated_mlp registry state does not match the selected arm")
        if variant.mlp and mlp_path != "native":
            raise RuntimeError(f"{variant.name}: expected native MLP, got {mlp_path}")
        print(
            f"WARM {variant.name} gdn_path={gdn_path} mlp_path={mlp_path} native_gdn_delta={observed_native_gdn} "
            f"full_attn={attention_backend} frost_delta={frost_calls - before_frost}"
        )

    if not torch_sdpa_probe:
        raise RuntimeError("Torch SDPA route probe did not run")
    _validate_torch_baseline_route(torch_sdpa_probe)
    print("TORCH_SDPA_PROBE " + json.dumps(torch_sdpa_probe, sort_keys=True))

    named_parameters = dict(model.named_parameters())
    linear_layer = linear_layers[0]
    full_attn_layer = attn_layers[0]
    grad_names = (
        f"model.layers.{linear_layer}.mlp.gate_proj.weight",
        f"model.layers.{linear_layer}.mlp.down_proj.weight",
        f"model.layers.{linear_layer}.attn.q_proj.weight",
        f"model.layers.{linear_layer}.attn.v_proj.weight",
        f"model.layers.{full_attn_layer}.attn.q_proj.weight",
        f"model.layers.{full_attn_layer}.attn.v_proj.weight",
        f"model.layers.{full_attn_layer}.attn.o_proj.weight",
    )
    downstream_full_attn_grad_names = frozenset(grad_names[-3:])
    missing_grad_names = [name for name in grad_names if name not in named_parameters]
    if missing_grad_names:
        raise RuntimeError(f"model parameter names changed; missing correctness samples {missing_grad_names}")

    correctness = {}
    reference = None
    for variant in VARIANTS:
        torch.manual_seed(1234)
        output = step(variant)
        loss_value = float(output.loss.detach())
        grad_values = {name: named_parameters[name].grad.detach().reshape(-1)[: 1 << 20].float().cpu() for name in grad_names}
        current = {
            "loss": loss_value,
            "grads": grad_values,
        }
        torch.cuda.synchronize(device)
        if not math.isfinite(loss_value):
            raise RuntimeError(f"{variant.name}: non-finite correctness loss {loss_value}")
        nonfinite_grads = [name for name, value in grad_values.items() if not bool(torch.isfinite(value).all())]
        if nonfinite_grads:
            raise RuntimeError(f"{variant.name}: non-finite correctness gradients {nonfinite_grads}")
        if reference is None:
            reference = current
        rel_loss = abs(current["loss"] - reference["loss"]) / max(abs(reference["loss"]), 1e-12)
        if not math.isfinite(rel_loss):
            raise RuntimeError(f"{variant.name}: non-finite relative loss {rel_loss}")
        rel_grads = {}
        for name in grad_names:
            actual, expected = current["grads"][name], reference["grads"][name]
            rel_grads[name] = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
            if not math.isfinite(rel_grads[name]):
                raise RuntimeError(f"{variant.name}: non-finite relative gradient for {name}: {rel_grads[name]}")
        # This multi-layer BF16 comparison is a composition diagnostic. Focused
        # op/full-block tests remain the primary numerical gates.
        grad_tolerances = {name: (0.07 if variant.gdn and name in downstream_full_attn_grad_names else 0.05 if variant.gdn else 0.03) for name in grad_names}
        if rel_loss > 0.02 or any(value > grad_tolerances[name] for name, value in rel_grads.items()):
            raise RuntimeError(
                f"{variant.name}: correctness outside aggregate BF16 tolerance: " f"loss={rel_loss}, grads={rel_grads}, grad_tolerances={grad_tolerances}"
            )
        correctness[variant.bits] = {
            "rel_loss": rel_loss,
            "rel_grads": rel_grads,
            "grad_tolerances": grad_tolerances,
        }
        print(
            f"CORRECT {variant.name} rel_loss={rel_loss:.6g} "
            f"grad_tolerances={json.dumps(grad_tolerances, sort_keys=True)} "
            f"rel_grads={json.dumps(rel_grads, sort_keys=True)}"
        )
    del reference

    raw = {variant.bits: [] for variant in VARIANTS}
    batches = {variant.bits: [] for variant in VARIANTS}
    timing_started = time.time()
    for batch in range(args.rounds):
        for variant_index in orders[batch % len(orders)]:
            variant = VARIANTS[variant_index]
            samples = []
            for _ in range(args.repeats):
                select(variant)
                model.zero_grad(set_to_none=True)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = model(input_ids=ids, labels=ids)
                output.loss.backward()
                end.record()
                end.synchronize()
                elapsed_ms = start.elapsed_time(end)
                if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
                    raise RuntimeError(f"{variant.name}: invalid CUDA-event timing {elapsed_ms}")
                samples.append(elapsed_ms)
            raw[variant.bits].append(samples)
            batches[variant.bits].append(statistics.median(samples))
        print(f"BATCH {batch + 1}/{args.rounds} elapsed_s={time.time() - timing_started:.1f}", flush=True)

    baseline = batches["000"]
    summary = {}
    for variant in VARIANTS:
        values = batches[variant.bits]
        result = {
            "p10_ms": percentile(values, 0.1),
            "p50_ms": percentile(values, 0.5),
            "p90_ms": percentile(values, 0.9),
            "mean_ms": statistics.mean(values),
            "batches": len(values),
        }
        result.update(paired_stats(values, baseline))
        result["wins_vs_baseline"] = result.pop("wins")
        summary[variant.bits] = result
        print(f"RESULT {variant.name} " + json.dumps(result, sort_keys=True))

    comparisons = {}
    for label, numerator_bits, denominator_bits in PAIR_BITS:
        result = paired_stats(batches[numerator_bits], batches[denominator_bits])
        result.update(
            {
                "numerator": BY_BITS[numerator_bits].name,
                "denominator": BY_BITS[denominator_bits].name,
            }
        )
        comparisons[label] = result
        print("COMPARISON " + label + " " + json.dumps(result, sort_keys=True))

    batch_times_by_mask = {mask: batches[f"{mask:03b}"] for mask in range(8)}
    main_effects = factorial_main_effects(batch_times_by_mask)
    shapley = shapley_savings(batch_times_by_mask)
    print("MAIN_EFFECTS " + json.dumps(main_effects, sort_keys=True))
    print(
        "SHAPLEY "
        + json.dumps(
            {
                "saving_ms": shapley["saving_ms"],
                "combined_saving_ms": shapley["combined_saving_ms"],
            },
            sort_keys=True,
        )
    )

    calls_per_variant = args.warmup + 1 + args.rounds * args.repeats
    expected_frost_calls = sum(variant.mlp for variant in VARIANTS) * calls_per_variant * shape["layers"]
    expected_attn_calls = sum(not variant.attn for variant in VARIANTS) * calls_per_variant * len(attn_layers)
    expected_native_gdn_calls = sum(variant.gdn for variant in VARIANTS) * calls_per_variant * len(linear_layers)
    if frost_calls != expected_frost_calls or pointwise_calls != 0:
        raise RuntimeError(f"final dSwiGLU route mismatch: frost={frost_calls}/{expected_frost_calls}, " f"pointwise={pointwise_calls}/0")
    if attn_calls != {"torch": expected_attn_calls, "cudnn": expected_attn_calls}:
        raise RuntimeError(f"final full-attention route mismatch: calls={attn_calls}, expected_each={expected_attn_calls}")
    if native_gdn_calls != expected_native_gdn_calls:
        raise RuntimeError(f"final native GDN route mismatch: calls={native_gdn_calls}/{expected_native_gdn_calls}")
    if gdnmod.last_path() != "native":
        raise RuntimeError(f"unexpected final GDN route: {gdnmod.last_path()}")

    route = {
        "frost_calls": frost_calls,
        "expected_frost_calls": expected_frost_calls,
        "pointwise_calls": pointwise_calls,
        "native_gdn_calls": native_gdn_calls,
        "expected_native_gdn_calls": expected_native_gdn_calls,
        "last_gdn_path": gdnmod.last_path(),
        "last_mlp_path": cfla.mlp_last_path(),
        "uses_public_gated_mlp_adapter": True,
        "full_attention_calls": attn_calls,
        "expected_full_attention_calls_each": expected_attn_calls,
        "torch_sdpa_probe": torch_sdpa_probe,
        "torch_baseline_route_contract": ROUTE_CONTRACT["torch_full_attention"],
        "torch_baseline_route_contract_passed": True,
    }
    completed_utc = _utc_now()
    metadata = {
        "schema_version": 3,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "arguments": _serializable_args(args),
        "config": config,
        "correctness_role": ("multi-layer BF16 composition diagnostic; focused op and full-block " "parity tests are the primary correctness gates"),
        "correctness": correctness,
        "summary": summary,
        "comparisons": comparisons,
        "main_effects": main_effects,
        "shapley": shapley,
        "batch_medians_ms": batches,
        "raw_ms": raw,
        "route": route,
        "artifacts": {"raw_json": str(raw_path), "markdown": str(markdown_path)},
    }
    if previous_metadata is not None:
        comparison = compare_results(metadata, previous_metadata)
        comparison["previous_artifact"] = _source_record(args.compare)
        comparison["current_build_fingerprint_sha256"] = config["build_fingerprint"]["sha256"]
        comparison["previous_build_fingerprint_sha256"] = previous_metadata.get("config", {}).get("build_fingerprint", {}).get("sha256", "unknown")
        metadata["comparison"] = comparison
    raw_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    raw_hash = _sha256(raw_path)
    raw_link = os.path.relpath(raw_path.resolve(), markdown_path.resolve().parent)
    report = render_markdown(metadata, raw_json_link=raw_link, raw_json_sha256=raw_hash)
    markdown_path.write_text(report, encoding="utf-8")
    print("ROUTE " + json.dumps(route, sort_keys=True, allow_nan=False))
    print(f"RAW_JSON {raw_path} sha256={raw_hash}")
    print(f"MARKDOWN {markdown_path} sha256={_sha256(markdown_path)}")


def main():
    args = _parse_args()
    # Keep ``--help`` and static inspection usable in a CPU-only environment;
    # the actual benchmark still requires the normal Torch/CUDA dependencies.
    global torch
    torch = importlib.import_module("torch")
    orders = williams_orders()
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "0").lower() in ("1", "true", "yes", "on"):
        raise RuntimeError("global FROST opt-in must be disabled: FE d256 must use the cuDNN backend " "while the MLP calls its direct FROST dSwiGLU kernel")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = _pick_device(args.mode)
    with torch.cuda.device(device):
        # Load under the selected device context. After FE #682 the public d256
        # operator is backend-graph-only; the sibling exposes the canonical
        # Torch-versus-FE selector used by this matrix.
        qwen = _load_run_model()
        properties = torch.cuda.get_device_properties(device)
        if args.mode == "formal" and (properties.name != "NVIDIA B200" or properties.multi_processor_count != 148):
            raise RuntimeError("formal mode requires a full 148-SM NVIDIA B200, got " f"{properties.name}, {properties.multi_processor_count} SMs on {device}")
        _run_experiment(args, qwen, device, properties, orders, _utc_now())


if __name__ == "__main__":
    main()
