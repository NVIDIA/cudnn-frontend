# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DSA sparse attention benchmark runner.

Expands a DsaBenchmarkConfig into (model, seqlen, backend, dtype, pass) cases,
runs each in a subprocess (clean CUDA context and CuTe DSL compile state,
independent failures), collects CSV results, and generates a fwd/bwd chart.

Usage:
    python -m benchmark.dsa.runner --config deepseek_v4
    python -m benchmark.dsa.runner --config deepseek_v4 --dry-run
    python -m benchmark.dsa.runner --config deepseek_v4 --filter flash --pass bwd
"""

import argparse
import itertools
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .config_types import BenchmarkResult, DsaBenchmarkConfig, ModelPreset

logger = logging.getLogger(__name__)

UNSUPPORTED_EXIT_CODE = 3  # keep in sync with benchmark_single_dsa.py


def log_environment_info():
    try:
        import torch

        logger.info(f"torch.__version__ = '{torch.__version__}'")
        logger.info(f"torch.version.cuda = '{torch.version.cuda}'")
        if torch.cuda.is_available():
            logger.info(f"torch.cuda.get_device_name() = '{torch.cuda.get_device_name()}'")
            logger.info(f"torch.cuda.get_device_capability() = {torch.cuda.get_device_capability()}")
    except ImportError:
        logger.warning("torch not available")
    try:
        import cudnn

        logger.info(f"cuDNN Frontend Version: cudnn.__version__ = '{cudnn.__version__}'")
    except ImportError:
        logger.warning("cudnn not available")


class DsaBenchmarkRunner:
    """Runs DSA benchmarks from configurations with cartesian product expansion."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="[%(levelname)s] %(message)s",
            stream=sys.stderr,
        )

    def expand_config(self, config: DsaBenchmarkConfig) -> Iterator[Dict[str, Any]]:
        passes = ["fwd", "bwd"] if config.profile_pass == "both" else [config.profile_pass]
        for model, (q_seqlen, kv_seqlen), backend, data_type, profile_pass in itertools.product(
            config.models, config.seqlens, config.backends, config.data_types, passes
        ):
            # deterministic_bwd is a backward-only knob: fwd emits one row.
            det_values = [False] if profile_pass == "fwd" else list(config.deterministic_bwd)
            for det_bwd in det_values:
                yield {
                    "config": config,
                    "model": model,
                    "q_seqlen": q_seqlen,
                    "kv_seqlen": kv_seqlen,
                    "backend": backend,
                    "data_type": data_type,
                    "profile_pass": profile_pass,
                    "deterministic_bwd": det_bwd,
                }

    @staticmethod
    def _case_label(case: Dict[str, Any]) -> str:
        m: ModelPreset = case["model"]
        det = " | det" if case["deterministic_bwd"] else ""
        return (
            f"{m.name} | seq={case['q_seqlen']}x{case['kv_seqlen']} | H{m.num_q_heads} d={m.head_dim_qk}/{m.head_dim_vo} "
            f"K={m.topk} | {case['backend']} | {case['data_type']} | {case['profile_pass']}{det}"
        )

    def run_single(self, case: Dict[str, Any]) -> BenchmarkResult:
        config: DsaBenchmarkConfig = case["config"]
        model: ModelPreset = case["model"]
        base = dict(
            config_name=config.name,
            model_name=model.name,
            backend=case["backend"],
            data_type=case["data_type"],
            q_seqlen=case["q_seqlen"],
            kv_seqlen=case["kv_seqlen"],
            num_q_heads=model.num_q_heads,
            head_dim_qk=model.head_dim_qk,
            head_dim_vo=model.head_dim_vo,
            topk=model.topk,
            indexer_topk=model.indexer_topk,
            has_sink=model.has_sink,
            use_topk_length=config.use_topk_length,
            profile_pass=case["profile_pass"],
            deterministic_bwd=case["deterministic_bwd"],
            num_iterations=config.num_iterations,
        )
        failed = dict(time_ms=float("inf"), tflops=0.0, success=False)

        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_single_dsa.py")
        cmd = [
            sys.executable,
            script,
            "--profile_pass",
            case["profile_pass"],
            "--backend",
            case["backend"],
            "--q_seqlen",
            str(case["q_seqlen"]),
            "--kv_seqlen",
            str(case["kv_seqlen"]),
            "--num_q_heads",
            str(model.num_q_heads),
            "--head_dim_qk",
            str(model.head_dim_qk),
            "--head_dim_vo",
            str(model.head_dim_vo),
            "--topk",
            str(model.topk),
            "--indexer_topk",
            str(model.indexer_topk),
            "--data_type",
            case["data_type"],
            "--num_iterations",
            str(config.num_iterations),
            "--num_warmup_iterations",
            str(config.num_warmup_iterations),
        ]
        if model.has_sink:
            cmd.append("--has_sink")
        if config.use_topk_length:
            cmd.append("--use_topk_length")
        if case["deterministic_bwd"]:
            cmd.append("--deterministic")

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            return BenchmarkResult(**base, **failed, error_message="timeout")

        if proc.returncode == UNSUPPORTED_EXIT_CODE:
            line = next((l for l in proc.stdout.splitlines() if l.startswith("UNSUPPORTED,")), "UNSUPPORTED,")
            return BenchmarkResult(**base, **failed, skipped=True, error_message=line.split(",", 1)[1])

        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT,")), None)
        if proc.returncode != 0 or line is None:
            tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-12:])
            message = tail[-1500:].strip() or f"subprocess exited with code {proc.returncode} and produced no output"
            return BenchmarkResult(**base, **failed, error_message=message)

        _, ms, tflops, peak, gpu, cudnn_version, detail = line.split(",", 6)
        return BenchmarkResult(
            **base,
            time_ms=float(ms),
            tflops=float(tflops),
            success=True,
            gpu_name=gpu,
            cudnn_version=cudnn_version,
            peak_mma_tflops=float(peak) if peak else None,
            backend_detail=detail,
        )

    @staticmethod
    def filter_cases(cases, filter_model=None, filter_backend=None, filter_dtype=None, filter_pass=None):
        if filter_model:
            cases = [c for c in cases if filter_model in c["model"].name]
        if filter_backend:
            cases = [c for c in cases if c["backend"] == filter_backend]
        if filter_dtype:
            cases = [c for c in cases if c["data_type"] == filter_dtype]
        if filter_pass:
            cases = [c for c in cases if c["profile_pass"] == filter_pass]
        return cases

    def run_config(self, config: DsaBenchmarkConfig, **filters) -> List[BenchmarkResult]:
        log_environment_info()
        cases = self.filter_cases(list(self.expand_config(config)), **filters)
        if not cases:
            logger.warning("No benchmark cases to run after applying filters")
            return []
        logger.info(f"Running {len(cases)} benchmark cases from config '{config.name}'")

        results = []
        for i, case in enumerate(cases, 1):
            logger.info(f"[{i}/{len(cases)}] {self._case_label(case)}")
            r = self.run_single(case)
            results.append(r)
            if r.success:
                sol = f", {r.tflops / r.peak_mma_tflops * 100:.1f}% SOL" if r.peak_mma_tflops else ""
                logger.info(f"  -> {r.profile_pass}: {r.time_ms:.3f}ms ({r.tflops:.0f} TFLOPS{sol}) [{r.backend_detail}]")
            elif r.skipped:
                logger.info(f"  -> SKIPPED (unsupported): {r.error_message}")
            else:
                logger.warning(f"  -> FAILED: {(r.error_message or '').splitlines()[-1][:200]}")
        return results

    def results_to_dataframe(self, results: List[BenchmarkResult]):
        import pandas as pd

        return pd.DataFrame([asdict(r) for r in results])

    def save_csv(self, results, config: DsaBenchmarkConfig, output_path: Optional[Path] = None) -> Path:
        df = self.results_to_dataframe(results)
        if output_path is None:
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, float_format="%.3f")
        logger.info(f"Results saved to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run DSA sparse attention benchmarks from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmark.dsa.runner --config deepseek_v4
    python -m benchmark.dsa.runner --config deepseek_v4 --dry-run
    python -m benchmark.dsa.runner --config deepseek_v4 --filter flash
    python -m benchmark.dsa.runner --config deepseek_v4 --pass fwd
    python -m benchmark.dsa.runner --config deepseek_v4 --output-dir benchmark/dsa/results/deepseek_v4/b200
        """,
    )
    parser.add_argument("--config", help="Config name (a Python file in configs/)")
    parser.add_argument("--dry-run", action="store_true", help="Print benchmark cases without executing")
    parser.add_argument("--filter", dest="filter_model", help="Filter by model name (substring match)")
    parser.add_argument("--backend", dest="filter_backend", help="Filter by backend (exact match)")
    parser.add_argument("--dtype", dest="filter_dtype", help="Filter by data type (exact match)")
    parser.add_argument("--pass", dest="filter_pass", choices=["fwd", "bwd"], help="Filter by pass")
    parser.add_argument("--output", type=Path, help="Output path for CSV (default: <output_dir>/<config>_<timestamp>.csv)")
    parser.add_argument("--output-dir", type=Path, help="Override config.output_dir (CSV + chart land here)")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--list-configs", action="store_true", help="List available configurations and exit")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    from .configs import list_configs, load_config

    if args.list_configs:
        print("Available configurations:")
        for name in list_configs():
            print(f"  {name}")
        return
    if not args.config:
        parser.error("--config is required")

    try:
        config = load_config(args.config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if args.output_dir:
        config.output_dir = str(args.output_dir)

    runner = DsaBenchmarkRunner(verbose=not args.quiet)
    filters = dict(
        filter_model=args.filter_model,
        filter_backend=args.filter_backend,
        filter_dtype=args.filter_dtype,
        filter_pass=args.filter_pass,
    )

    if args.dry_run:
        cases = runner.filter_cases(list(runner.expand_config(config)), **filters)
        print(f"Would run {len(cases)} benchmark cases from config '{config.name}':")
        print()
        for i, case in enumerate(cases, 1):
            print(f"  [{i}] {runner._case_label(case)}")
        return

    results = runner.run_config(config, **filters)
    if not results:
        print("No results to save", file=sys.stderr)
        sys.exit(1)

    csv_path = runner.save_csv(results, config, args.output)

    if not args.no_chart:
        try:
            from .charts import generate_charts

            for path in generate_charts(runner.results_to_dataframe(results), config):
                print(f"Chart saved to {path}")
        except ImportError as e:
            logger.warning(f"Could not generate chart (missing dependency): {e}")
        except Exception as e:
            logger.warning(f"Could not generate chart: {e}")

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
