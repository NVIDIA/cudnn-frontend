# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DSA sparse attention benchmark runner.

Expands a DsaBenchmarkConfig into cases over (model, shape, backend, dtype,
phase) and the appended axes (component, kv_dtype, paged, indexer_mode, the
generation grid), decides in-process which cases no kernel serves
(:func:`support.decide_support`; those rows are written as skipped without a
subprocess), runs every other case in a subprocess (clean CUDA context and
CuTe DSL compile state, independent failures), collects CSV results, and
generates the legacy fwd/bwd chart plus one extended-axes chart.

Legacy configs (no appended axes) expand to exactly the same cases, labels
and order as before; the first 25 CSV columns are unchanged.

Usage:
    python -m benchmark.dsa.runner --config deepseek_v4
    python -m benchmark.dsa.runner --config deepseek_v4 --dry-run
    python -m benchmark.dsa.runner --config deepseek_v4 --filter flash --pass bwd
    python -m benchmark.dsa.runner --config deepseek_v41 --component indexer --phase generation
    python -m benchmark.dsa.runner --config deepseek_v41 --kv-dtype mxfp8 --flat
"""

import argparse
import itertools
import logging
import os
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import support
from .config_types import BenchmarkResult, DsaBenchmarkConfig, ModelPreset

logger = logging.getLogger(__name__)

UNSUPPORTED_EXIT_CODE = 3  # keep in sync with benchmark_single_dsa.py

# Optional `key=<number>` tokens the worker appends to the detail field; the first two land in
# their own columns, bytes_gathered becomes gather_gbps (bytes / time) on success rows.
_DETAIL_FLOAT_TOKENS = ("time_step1_ms", "time_step2_ms")
_DETAIL_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])(time_step1_ms|time_step2_ms|bytes_gathered)=([0-9][0-9.eE+-]*)")

_UNPROBED = object()  # sentinel: device facts not queried yet


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


def device_capability() -> Optional[Tuple[int, int]]:
    """Compute capability of the current CUDA device, or None without torch / CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability()
        return int(major), int(minor)
    except Exception:
        return None


def parse_detail_tokens(detail: Optional[str]) -> Dict[str, float]:
    """``{token: value}`` for the numeric ``time_step1_ms`` / ``time_step2_ms`` / ``bytes_gathered`` tokens in a detail string."""
    found: Dict[str, float] = {}
    for key, value in _DETAIL_TOKEN_RE.findall(detail or ""):
        try:
            found.setdefault(key, float(value))
        except ValueError:
            continue
    return found


class DsaBenchmarkRunner:
    """Runs DSA benchmarks from configurations with cartesian product expansion.

    ``cc`` and ``dsl_target_error`` are probed lazily on the first
    :meth:`run_single` (a dry run touches no GPU); tests pass them explicitly.
    """

    def __init__(self, verbose: bool = True, cc=_UNPROBED, dsl_target_error=_UNPROBED):
        self.verbose = verbose
        self._cc = cc
        self._dsl_target_error = dsl_target_error
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="[%(levelname)s] %(message)s",
            stream=sys.stderr,
        )

    @property
    def cc(self) -> Optional[Tuple[int, int]]:
        if self._cc is _UNPROBED:
            self._cc = device_capability()
        return self._cc

    @property
    def dsl_target_error(self) -> Optional[str]:
        if self._dsl_target_error is _UNPROBED:
            self._dsl_target_error = support.dsl_target_error(self.cc)  # once per run; the worker repeats the same probe
        return self._dsl_target_error

    # ------------------------------------------------------------------ enumeration

    @staticmethod
    def _expand_axes(config: DsaBenchmarkConfig, model: ModelPreset, phase: str, outer: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Inner loops shared by the shape sweep and the generation grid: component, kv_dtype, paged, indexer_mode, deterministic_bwd."""
        for component in config.components:
            modes = list(model.indexer_modes) if component == "indexer" else [None]
            for kv_dtype in config.kv_dtypes:
                for pg in config.paged:
                    for mode in modes:
                        # deterministic_bwd is a backward-only knob: forward phases emit one row.
                        det_values = list(config.deterministic_bwd) if phase == "bprop" else [False]
                        for det_bwd in det_values:
                            yield {
                                **outer,
                                "config": config,
                                "model": model,
                                "profile_pass": support.PROFILE_PASS_FOR_PHASE[phase],
                                "deterministic_bwd": det_bwd,
                                "component": component,
                                "phase": phase,
                                "kv_dtype": kv_dtype,
                                "paged": bool(pg),
                                "indexer_mode": mode,
                            }

    def expand_config(self, config: DsaBenchmarkConfig) -> Iterator[Dict[str, Any]]:
        """Cases in a legacy-stable order.

        The shape sweep keeps today's loop shape, ``product(models, seqlens,
        backends, data_types)`` with the phase innermost (context then bprop,
        exactly the former fwd/bwd order), then fans out the appended axes.
        The generation grid follows: ``model x generation_kv_lens x
        generation_batch_sizes x sorted({1, mtp_rows})`` with the same inner
        loops; its ``q_seqlen`` / ``kv_seqlen`` are per request.
        """
        phases = support.phases_for_config(config)
        shape_phases = [p for p in phases if p != "generation"]
        for model, (q_seqlen, kv_seqlen), backend, data_type in itertools.product(config.models, config.seqlens, config.backends, config.data_types):
            for phase in shape_phases:
                outer = dict(q_seqlen=q_seqlen, kv_seqlen=kv_seqlen, backend=backend, data_type=data_type, batch_size=1)
                yield from self._expand_axes(config, model, phase, outer)
        if "generation" in phases:
            for model in config.models:
                for kv_len in config.generation_kv_lens:
                    for batch_size in config.generation_batch_sizes:
                        for q_tokens in sorted({1, int(model.mtp_rows)}):
                            for backend, data_type in itertools.product(config.backends, config.data_types):
                                outer = dict(q_seqlen=q_tokens, kv_seqlen=kv_len, backend=backend, data_type=data_type, batch_size=batch_size)
                                yield from self._expand_axes(config, model, "generation", outer)

    @staticmethod
    def _case_label(case: Dict[str, Any]) -> str:
        """Today's label, plus a tag per appended axis that is off its legacy default."""
        m: ModelPreset = case["model"]
        det = " | det" if case["deterministic_bwd"] else ""
        label = (
            f"{m.name} | seq={case['q_seqlen']}x{case['kv_seqlen']} | H{m.num_q_heads} d={m.head_dim_qk}/{m.head_dim_vo} "
            f"K={m.topk} | {case['backend']} | {case['data_type']} | {case['profile_pass']}{det}"
        )
        if case.get("component", "core") != "core":
            label += f" | {case['component']}"
        if case.get("phase") == "generation":
            label += f" | gen B={case.get('batch_size', 1)} q={case['q_seqlen']}"
        if case.get("kv_dtype", "bfloat16") != "bfloat16":
            label += f" | {case['kv_dtype']}"
        if case.get("paged", False):
            label += " | paged"
        if case.get("indexer_mode") not in (None, "dense"):
            label += f" | {case['indexer_mode']}"
        return label

    # ------------------------------------------------------------------ execution

    @staticmethod
    def worker_command(case: Dict[str, Any]) -> List[str]:
        """argv of the worker subprocess for ``case`` (legacy flags first, appended axes after)."""
        config: DsaBenchmarkConfig = case["config"]
        model: ModelPreset = case["model"]
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
        # Appended axes (worker defaults = legacy behaviour).
        cmd += [
            "--component",
            case.get("component", "core"),
            "--phase",
            case.get("phase") or {"fwd": "context", "bwd": "bprop"}[case["profile_pass"]],
            "--kv_dtype",
            case.get("kv_dtype", "bfloat16"),
            "--page_size",
            str(config.page_size),
            "--batch_size",
            str(case.get("batch_size", 1)),
            "--num_kv_heads",
            str(model.num_kv_heads),
            "--kv_layout",
            model.kv_layout,
            "--index_granularity",
            str(model.index_granularity),
            "--indexer_heads",
            str(model.indexer_heads),
            "--indexer_pool_ratio",
            str(model.indexer_pool_ratio),
            "--mtp_rows",
            str(model.mtp_rows),
        ]
        if case.get("paged", False):
            cmd.append("--paged")
        if case.get("indexer_mode") is not None:
            cmd += ["--indexer_mode", case["indexer_mode"]]
        return cmd

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
        base.update(support.static_columns(case))
        failed = dict(time_ms=float("inf"), tflops=0.0, success=False)

        # Rows no kernel serves are decided here, in our own wording, and cost no subprocess.
        reason = support.decide_support(case, self.cc, self.dsl_target_error)
        if reason:
            return BenchmarkResult(**base, **failed, skipped=True, error_message=reason)

        cmd = self.worker_command(case)
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
        time_ms = float(ms)
        tokens = parse_detail_tokens(detail)
        steps = {key: tokens.get(key) for key in _DETAIL_FLOAT_TOKENS}
        bytes_gathered = tokens.get("bytes_gathered")
        gather_gbps = bytes_gathered / (time_ms * 1e6) if bytes_gathered and time_ms > 0 else None
        return BenchmarkResult(
            **base,
            time_ms=time_ms,
            tflops=float(tflops),
            success=True,
            gpu_name=gpu,
            cudnn_version=cudnn_version,
            peak_mma_tflops=float(peak) if peak else None,
            backend_detail=detail,
            gather_gbps=gather_gbps,
            **steps,
        )

    # ------------------------------------------------------------------ filters / driver

    @staticmethod
    def filter_cases(
        cases,
        filter_model=None,
        filter_backend=None,
        filter_dtype=None,
        filter_pass=None,
        filter_component=None,
        filter_phase=None,
        filter_kv_dtype=None,
        filter_paged=None,
    ):
        if filter_model:
            cases = [c for c in cases if filter_model in c["model"].name]
        if filter_backend:
            cases = [c for c in cases if c["backend"] == filter_backend]
        if filter_dtype:
            cases = [c for c in cases if c["data_type"] == filter_dtype]
        if filter_pass:
            cases = [c for c in cases if c["profile_pass"] == filter_pass]
        if filter_component:
            cases = [c for c in cases if c.get("component", "core") == filter_component]
        if filter_phase:
            cases = [c for c in cases if c.get("phase") == filter_phase]
        if filter_kv_dtype:
            cases = [c for c in cases if c.get("kv_dtype", "bfloat16") == filter_kv_dtype]
        if filter_paged is not None:
            cases = [c for c in cases if bool(c.get("paged", False)) == bool(filter_paged)]
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
                gbps = f", {r.gather_gbps:.0f} GB/s gathered" if r.gather_gbps else ""
                logger.info(f"  -> {r.profile_pass}: {r.time_ms:.3f}ms ({r.tflops:.0f} TFLOPS{sol}{gbps}) [{r.backend_detail}]")
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
    python -m benchmark.dsa.runner --config deepseek_v41 --component indexer --phase generation --dry-run
    python -m benchmark.dsa.runner --config glm53 --kv-dtype mxfp8 --flat
        """,
    )
    parser.add_argument("--config", help="Config name (a Python file in configs/)")
    parser.add_argument("--dry-run", action="store_true", help="Print benchmark cases without executing")
    parser.add_argument("--filter", dest="filter_model", help="Filter by model name (substring match)")
    parser.add_argument("--backend", dest="filter_backend", help="Filter by backend (exact match)")
    parser.add_argument("--dtype", dest="filter_dtype", help="Filter by data type (exact match)")
    parser.add_argument("--pass", dest="filter_pass", choices=["fwd", "bwd"], help="Filter by pass")
    parser.add_argument("--component", dest="filter_component", choices=["core", "indexer"], help="Filter by component")
    parser.add_argument("--phase", dest="filter_phase", choices=list(support.PHASES), help="Filter by phase")
    parser.add_argument("--kv-dtype", dest="filter_kv_dtype", help="Filter by KV / index-key storage format (exact match)")
    paged_group = parser.add_mutually_exclusive_group()
    paged_group.add_argument("--paged", dest="filter_paged", action="store_const", const=True, help="Only paged-pool cases")
    paged_group.add_argument("--flat", dest="filter_paged", action="store_const", const=False, help="Only flat-pool cases")
    parser.add_argument("--output", type=Path, help="Output path for CSV (default: <output_dir>/<config>_<timestamp>.csv)")
    parser.add_argument("--output-dir", type=Path, help="Override config.output_dir (CSV + charts land here)")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--list-configs", action="store_true", help="List available configurations and exit")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.set_defaults(filter_paged=None)
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
        filter_component=args.filter_component,
        filter_phase=args.filter_phase,
        filter_kv_dtype=args.filter_kv_dtype,
        filter_paged=args.filter_paged,
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
