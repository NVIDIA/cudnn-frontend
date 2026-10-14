# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Attention inference benchmark runner.

Expands an InferenceBenchmarkConfig into (model, phase, shape, backend) cases,
runs each in a subprocess (clean CUDA context, independent failures), collects
CSV results, and generates one chart per phase (context / generation).

Usage:
    python -m benchmark.attention_inference.runner --config llama
    python -m benchmark.attention_inference.runner --config kimi_k3 --dry-run
    python -m benchmark.attention_inference.runner --config deepseek_v4 --backend flashinfer
"""

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .config_types import BenchmarkResult, InferenceBenchmarkConfig, ModelPreset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# The backends whose generation phase reads a paged KV cache, and so sweep ``config.page_sizes``; the cudnn graph paths and fa4
# attend over dense K / V and take one case per shape at ``config.page_size``.
PAGED_BACKENDS = ("flashinfer", "flash_mla", "b12x")


def _device_cc() -> tuple:
    try:
        import torch

        return torch.cuda.get_device_capability()
    except Exception:
        return (0, 0)


def backend_supported(backend: str, model: ModelPreset, phase: str, cc: tuple) -> Optional[str]:
    """Return a skip reason if this (backend, model, phase) is known-unsupported, else None.

    Only *structural* impossibilities are skipped here (wrong arch, kernels
    that do not exist). Anything a backend might plausibly run is attempted so
    genuine NOT_SUPPORTED errors are recorded in the CSV instead of silently
    dropped.
    """
    if backend == "b12x":
        if cc[0] != 12:
            return "b12x targets SM12x only"
    if backend == "flash_mla":
        if cc[0] not in (9, 10):
            return "flash_mla targets SM90/SM100 only"
        if model.kind != "mla_absorbed":
            return "flash_mla only serves absorbed-MLA shapes"
        if phase != "generation":
            return "flash_mla benchmarked for generation only"
    if backend == "flash_attention_4":
        if cc[0] not in (9, 10, 11):
            return "fa4 supports cc 9.x/10.x/11.x"
        if model.kind == "mla_absorbed":
            return "fa4 dense path has no shared-KV MLA contract"
    if model.kind == "mla_absorbed" and model.head_dim_qk != model.head_dim_vo and phase == "context":
        # True latent absorption (e.g. Kimi K3 576/512) only exists at decode;
        # prefill runs unabsorbed — that's the training suite's config.
        # Shared-K=V MQA with equal dims (DeepSeek-V4) prefills as-is and runs.
        return "context phase for latent-absorbed MLA runs unabsorbed (see training suite prefill)"
    return None


class InferenceBenchmarkRunner:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @staticmethod
    def page_sizes_for(config: InferenceBenchmarkConfig, backend: str, model: Optional[ModelPreset] = None) -> List[int]:
        """Generation pages, deduplicated after applying the model's backend layout constraints.

        Without a model, return the requested backend axis. Case expansion supplies
        the model so subprocess arguments, logs and result rows use effective sizes.
        """
        pages = config.page_sizes if backend in PAGED_BACKENDS and config.page_sizes else [config.page_size]
        if backend == "b12x" and model is not None:
            # Match setup_b12x: sliding-window GQA uses contiguous K/V; d576 MLA
            # honors the requested page; the remaining paged paths require >=64.
            if model.kind == "gqa" and model.sliding_window_size:
                return [config.page_size]
            if not (model.kind == "mla_absorbed" and model.head_dim_qk == 576):
                pages = [max(int(p), 64) for p in pages]
        return list(dict.fromkeys(int(p) for p in pages))

    def expand_config(self, config: InferenceBenchmarkConfig) -> Iterator[Dict[str, Any]]:
        for model in config.models:
            for backend in config.backends:
                for data_type in config.data_types:
                    for s_q, s_kv in [(s, s) for s in config.context_seqlens] + list(config.context_chunked_shapes):
                        yield {
                            "config": config,
                            "model": model,
                            "backend": backend,
                            "data_type": data_type,
                            "kv_cache_dtype": data_type,
                            "phase": "context",
                            "batch_size": config.context_batch_size,
                            "q_tokens": s_q,
                            "kv_len": s_kv,
                            "page_size": config.page_size,
                        }
                    for q_tokens, kv_len in config.generation_shapes:
                        for batch in config.generation_batch_sizes:
                            for kv_dtype in config.kv_cache_dtypes:
                                for page_size in self.page_sizes_for(config, backend, model):
                                    yield {
                                        "config": config,
                                        "model": model,
                                        "backend": backend,
                                        "data_type": data_type,
                                        "kv_cache_dtype": kv_dtype,
                                        "phase": "generation",
                                        "batch_size": batch,
                                        "q_tokens": q_tokens,
                                        "kv_len": kv_len,
                                        "page_size": page_size,
                                    }

    def run_single(self, case: Dict[str, Any], cc: tuple) -> BenchmarkResult:
        config: InferenceBenchmarkConfig = case["config"]
        model: ModelPreset = case["model"]
        base = dict(
            config_name=config.name,
            model_name=model.name,
            phase=case["phase"],
            backend=case["backend"],
            data_type=case["data_type"],
            kv_cache_dtype=case["kv_cache_dtype"],
            batch_size=case["batch_size"],
            q_tokens=case["q_tokens"],
            kv_len=case["kv_len"],
            num_q_heads=model.num_q_heads,
            num_kv_heads=model.num_kv_heads,
            head_dim_qk=model.head_dim_qk,
            head_dim_vo=model.head_dim_vo,
            kind=model.kind,
            sliding_window_size=model.sliding_window_size,
            page_size=case.get("page_size", config.page_size),
            num_iterations=config.num_iterations,
        )

        skip = backend_supported(case["backend"], model, case["phase"], cc)
        if skip:
            return BenchmarkResult(**base, time_ms=float("inf"), tflops=0.0, gbps=0.0, sol_pct=None, success=False, error_message=f"SKIPPED: {skip}")

        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_single_attention.py")
        cmd = [
            sys.executable,
            script,
            "--phase",
            case["phase"],
            "--backend",
            case["backend"],
            "--batch_size",
            str(case["batch_size"]),
            "--q_tokens",
            str(case["q_tokens"]),
            "--kv_len",
            str(case["kv_len"]),
            "--num_q_heads",
            str(model.num_q_heads),
            "--num_kv_heads",
            str(model.num_kv_heads),
            "--head_dim_qk",
            str(model.head_dim_qk),
            "--head_dim_vo",
            str(model.head_dim_vo),
            "--kind",
            model.kind,
            "--page_size",
            str(case.get("page_size", config.page_size)),
            "--data_type",
            case["data_type"],
            "--kv_cache_dtype",
            case["kv_cache_dtype"],
            "--num_iterations",
            str(config.num_iterations),
            "--num_warmup_iterations",
            str(config.num_warmup_iterations),
        ]
        if case["phase"] == "context" and config.context_causal:
            cmd.append("--causal")
        if model.sliding_window_size:
            cmd += ["--sliding_window_size", str(model.sliding_window_size)]
        if model.sm_scale is not None:
            cmd += ["--sm_scale", str(model.sm_scale)]
        if model.has_sink:
            cmd.append("--has_sink")

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT,")), None)
            if proc.returncode != 0 or line is None:
                tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-12:])
                return BenchmarkResult(**base, time_ms=float("inf"), tflops=0.0, gbps=0.0, sol_pct=None, success=False, error_message=tail[-1500:])
            _, ms, tflops, gbps, sol, gpu, detail = line.split(",", 6)
            return BenchmarkResult(
                **base,
                time_ms=float(ms),
                tflops=float(tflops),
                gbps=float(gbps),
                sol_pct=float(sol) if sol else None,
                success=True,
                gpu_name=gpu,
                backend_detail=detail,
            )
        except subprocess.TimeoutExpired:
            return BenchmarkResult(**base, time_ms=float("inf"), tflops=0.0, gbps=0.0, sol_pct=None, success=False, error_message="timeout")

    def run_config(self, config, filter_backend=None, filter_model=None, filter_phase=None) -> List[BenchmarkResult]:
        cc = _device_cc()
        cases = list(self.expand_config(config))
        if filter_backend:
            cases = [c for c in cases if c["backend"] == filter_backend]
        if filter_model:
            cases = [c for c in cases if filter_model in c["model"].name]
        if filter_phase:
            cases = [c for c in cases if c["phase"] == filter_phase]
        logger.info(f"Running {len(cases)} cases from config '{config.name}' (cc {cc[0]}.{cc[1]})")

        results = []
        for i, case in enumerate(cases, 1):
            m = case["model"]
            kvd = "" if case["kv_cache_dtype"] == case["data_type"] else f" | kv={case['kv_cache_dtype']}"
            pg = f" | page={case['page_size']}" if case["phase"] == "generation" and case["backend"] in PAGED_BACKENDS else ""
            logger.info(
                f"[{i}/{len(cases)}] {m.name} | {case['phase']} | b={case['batch_size']} "
                f"q={case['q_tokens']} kv={case['kv_len']} | {case['backend']} | {case['data_type']}{kvd}{pg}"
            )
            r = self.run_single(case, cc)
            results.append(r)
            if r.success:
                sol = f", {r.sol_pct:.0f}% mem-SOL" if r.sol_pct else ""
                metric = f"{r.tflops:.0f} TFLOPS" if r.phase == "context" else f"{r.gbps:.0f} GB/s{sol}"
                logger.info(f"  -> {r.time_ms:.3f} ms ({metric}) [{r.backend_detail}]")
            else:
                logger.warning(f"  -> FAILED: {(r.error_message or '').splitlines()[-1][:140]}")
        return results

    def results_to_dataframe(self, results):
        import pandas as pd

        return pd.DataFrame([asdict(r) for r in results])

    def save_csv(self, results, config, output: Optional[Path] = None) -> Path:
        df = self.results_to_dataframe(results)
        if output is None:
            outdir = Path(config.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            output = outdir / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output, index=False)
        logger.info(f"Saved {len(df)} rows to {output}")
        return output


def main():
    parser = argparse.ArgumentParser(description="Attention inference benchmark runner")
    parser.add_argument("--config", required=False, help="config module name under configs/")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", dest="filter_backend")
    parser.add_argument("--filter", dest="filter_model")
    parser.add_argument("--phase", dest="filter_phase", choices=["context", "generation"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, help="override config.output_dir (CSV + charts land here)")
    parser.add_argument("--no-chart", action="store_true")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()

    from .configs import list_configs, load_config

    if args.list_configs:
        print("Available configurations:")
        for name in list_configs():
            print(f"  {name}")
        return
    if not args.config:
        parser.error("--config is required")

    config = load_config(args.config)
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    if args.filter_backend and args.filter_backend not in config.backends:
        # a backend the config does not list (a paged-KV reference backend on a cudnn-only config) runs ALONE over the
        # config's models, shapes and page sizes; a listed one is filtered as before
        logger.info(f"backend '{args.filter_backend}' is not in config '{config.name}' ({config.backends}); running it alone over the config's axes")
        config.backends = [args.filter_backend]
    runner = InferenceBenchmarkRunner()

    if args.dry_run:
        cases = list(runner.expand_config(config))
        if args.filter_backend:
            cases = [c for c in cases if c["backend"] == args.filter_backend]
        if args.filter_model:
            cases = [c for c in cases if args.filter_model in c["model"].name]
        if args.filter_phase:
            cases = [c for c in cases if c["phase"] == args.filter_phase]
        print(f"Would run {len(cases)} cases from '{config.name}':")
        for i, c in enumerate(cases, 1):
            pg = f" | page={c['page_size']}" if c["phase"] == "generation" and c["backend"] in PAGED_BACKENDS else ""
            print(f"  [{i}] {c['model'].name} | {c['phase']} | b={c['batch_size']} q={c['q_tokens']} kv={c['kv_len']} | {c['backend']} | {c['data_type']}{pg}")
        return

    results = runner.run_config(config, args.filter_backend, args.filter_model, args.filter_phase)
    if not results:
        print("No results", file=sys.stderr)
        sys.exit(1)
    runner.save_csv(results, config, args.output)

    if not args.no_chart:
        try:
            from .charts import generate_charts

            for path in generate_charts(runner.results_to_dataframe(results), config):
                logger.info(f"Chart: {path}")
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")


if __name__ == "__main__":
    main()
