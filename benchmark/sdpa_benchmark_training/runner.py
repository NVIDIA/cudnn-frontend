"""
Benchmark runner with configuration expansion, execution, and result collection.

This module provides the BenchmarkRunner class for running SDPA benchmarks
from configuration files, and a CLI entry point.

Usage:
    # Run from command line
    python -m benchmark.sdpa_benchmark_training.runner --config mlperf
    python -m benchmark.sdpa_benchmark_training.runner --config mlperf --dry-run

    # Import and use programmatically
    from benchmark.sdpa_benchmark_training.runner import BenchmarkRunner
    from benchmark.sdpa_benchmark_training.configs import load_config

    config = load_config("mlperf")
    runner = BenchmarkRunner()
    results = runner.run_config(config)
    runner.save_csv(results, config)
"""

import itertools
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional

from .config_types import BenchmarkConfig, BenchmarkResult, ModelPreset

logger = logging.getLogger(__name__)


def log_environment_info():
    """Log environment information (torch, CUDA, cuDNN, flash_attn versions)."""
    try:
        import torch

        logger.info(f"torch.__version__ = '{torch.__version__}'")
        logger.info(f"torch.version.cuda = '{torch.version.cuda}'")
        logger.info(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
            logger.info(f"torch.cuda.current_device() = {torch.cuda.current_device()}")
            logger.info(f"torch.cuda.get_device_name(torch.cuda.current_device()) = '{torch.cuda.get_device_name(torch.cuda.current_device())}'")
        logger.info(f"torch.backends.cudnn.enabled = {torch.backends.cudnn.enabled}")
    except ImportError:
        logger.warning("torch not available")

    try:
        import cudnn

        logger.info(f"cuDNN Backend Version: cudnn.backend_version() = {cudnn.backend_version()}")
        logger.info(f"cuDNN Frontend Version: cudnn.__version__ = '{cudnn.__version__}'")
    except ImportError:
        logger.warning("cudnn not available")

    try:
        import flash_attn

        logger.info(f"flash_attn.__version__ = '{flash_attn.__version__}'")
    except ImportError:
        pass  # flash_attn is optional


class BenchmarkRunner:
    """
    Runs benchmarks from configurations with cartesian product expansion.

    The runner takes a BenchmarkConfig and expands it into individual benchmark
    cases via cartesian product of all configuration dimensions. Each case is
    then executed and results are collected.

    Attributes:
        verbose: Whether to print progress information

    Example:
        runner = BenchmarkRunner(verbose=True)
        config = load_config("mlperf")

        # Dry run to see what would be executed
        for case in runner.expand_config(config):
            print(case)

        # Actually run the benchmarks
        results = runner.run_config(config)
        runner.save_csv(results, config)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the runner.

        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on verbosity setting."""
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="[%(levelname)s] %(message)s",
            stream=sys.stderr,
        )

    def expand_config(self, config: BenchmarkConfig) -> Iterator[Dict[str, Any]]:
        """
        Expand a BenchmarkConfig into individual benchmark cases.

        Performs cartesian product expansion over:
            models x seqlens x backends x data_types x attn_masks x deterministic_bwd

        Args:
            config: BenchmarkConfig to expand

        Yields:
            Dict containing all parameters for a single benchmark run
        """
        for model, (q_seqlen, kv_seqlen), backend, data_type, attn_mask, det_bwd in itertools.product(
            config.models,
            config.seqlens,
            config.backends,
            config.data_types,
            config.attn_masks,
            config.deterministic_bwd,
        ):
            # Skip deterministic mode for forward-only runs
            if det_bwd and config.profile_pass == "fwd":
                continue

            yield {
                "config_name": config.name,
                "model": model,
                "q_seqlen": q_seqlen,
                "kv_seqlen": kv_seqlen,
                "backend": backend,
                "data_type": data_type,
                "attn_mask": attn_mask,
                "profile_pass": config.profile_pass,
                "batch_size": config.batch_size,
                "num_iterations": config.num_iterations,
                "num_warmup_iterations": config.num_warmup_iterations,
                "skip_ref": config.skip_ref,
                "deterministic_bwd": det_bwd,
            }

    def run_single(self, case: Dict[str, Any]) -> BenchmarkResult:
        """
        Run a single benchmark case.

        Calls the run_benchmark() function from benchmark_single_sdpa.py
        and wraps the result in a BenchmarkResult.

        Args:
            case: Dict containing benchmark parameters (from expand_config)

        Returns:
            BenchmarkResult with timing data or error information
        """
        model: ModelPreset = case["model"]

        try:
            # Import here to avoid circular imports and allow the module to be
            # used even if torch/cudnn aren't installed (for dry-run mode)
            from .benchmark_single_sdpa import run_benchmark

            result = run_benchmark(
                batch_size=case["batch_size"],
                q_seqlen=case["q_seqlen"],
                kv_seqlen=case["kv_seqlen"],
                num_q_heads=model.num_q_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim_qk=model.head_dim_qk,
                head_dim_vo=model.head_dim_vo,
                data_type=case["data_type"],
                backend=case["backend"],
                attn_mask=case["attn_mask"],
                profile_pass=case["profile_pass"],
                num_iterations=case["num_iterations"],
                num_warmup_iterations=case["num_warmup_iterations"],
                skip_ref=case["skip_ref"],
                deterministic_bwd=case["deterministic_bwd"],
            )

            return BenchmarkResult(
                config_name=case["config_name"],
                model_name=model.name,
                backend=case["backend"],
                data_type=case["data_type"],
                attn_mask=case["attn_mask"],
                batch_size=case["batch_size"],
                q_seqlen=case["q_seqlen"],
                kv_seqlen=case["kv_seqlen"],
                num_q_heads=model.num_q_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim_qk=model.head_dim_qk,
                head_dim_vo=model.head_dim_vo,
                profile_pass=case["profile_pass"],
                deterministic_bwd=case["deterministic_bwd"],
                fwd_time_ms=result["fwd_time_ms"],
                bwd_time_ms=result["bwd_time_ms"],
                fwd_tflops=result["fwd_tflops"],
                bwd_tflops=result["bwd_tflops"],
                max_diff=result["max_diff"],
                num_iterations=case["num_iterations"],
                success=True,
                gpu_name=result.get("gpu_name"),
                cudnn_version=result.get("cudnn_version"),
                cudnn_backend_version=result.get("cudnn_backend_version"),
            )

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return BenchmarkResult(
                config_name=case["config_name"],
                model_name=model.name,
                backend=case["backend"],
                data_type=case["data_type"],
                attn_mask=case["attn_mask"],
                batch_size=case["batch_size"],
                q_seqlen=case["q_seqlen"],
                kv_seqlen=case["kv_seqlen"],
                num_q_heads=model.num_q_heads,
                num_kv_heads=model.num_kv_heads,
                head_dim_qk=model.head_dim_qk,
                head_dim_vo=model.head_dim_vo,
                profile_pass=case["profile_pass"],
                deterministic_bwd=case["deterministic_bwd"],
                fwd_time_ms=float("inf"),
                bwd_time_ms=float("inf"),
                fwd_tflops=0.0,
                bwd_tflops=0.0,
                max_diff=0.0,
                num_iterations=case["num_iterations"],
                success=False,
                error_message=str(e),
            )

    def run_config(
        self,
        config: BenchmarkConfig,
        filter_model: Optional[str] = None,
        filter_backend: Optional[str] = None,
        filter_dtype: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """
        Run all benchmarks from a configuration.

        Args:
            config: BenchmarkConfig to run
            filter_model: Optional model name filter (substring match)
            filter_backend: Optional backend filter (exact match)
            filter_dtype: Optional data type filter (exact match)

        Returns:
            List of BenchmarkResult for all executed cases
        """
        # Log environment info at the start
        log_environment_info()
        logger.info("")  # Blank line for readability

        results = []
        cases = list(self.expand_config(config))

        # Apply filters
        if filter_model:
            cases = [c for c in cases if filter_model in c["model"].name]
        if filter_backend:
            cases = [c for c in cases if c["backend"] == filter_backend]
        if filter_dtype:
            cases = [c for c in cases if c["data_type"] == filter_dtype]

        if not cases:
            logger.warning("No benchmark cases to run after applying filters")
            return results

        logger.info(f"Running {len(cases)} benchmark cases from config '{config.name}'")

        for i, case in enumerate(cases, 1):
            model = case["model"]
            det_str = "det" if case["deterministic_bwd"] else "non-det"
            logger.info(
                f"[{i}/{len(cases)}] {model.name} | "
                f"seq={case['q_seqlen']}x{case['kv_seqlen']} | "
                f"{case['backend']} | {case['data_type']} | "
                f"{case['attn_mask']} | {det_str}"
            )

            result = self.run_single(case)
            results.append(result)

            if result.success:
                fwd_info = f"fwd: {result.fwd_time_ms:.3f}ms ({result.fwd_tflops:.0f} TFLOPS)"
                bwd_info = f"bwd: {result.bwd_time_ms:.3f}ms ({result.bwd_tflops:.0f} TFLOPS)"
                logger.info(f"  -> {fwd_info}, {bwd_info}")
            else:
                logger.warning(f"  -> FAILED: {result.error_message}")

        return results

    def results_to_dataframe(self, results: List[BenchmarkResult]):
        """
        Convert results to a pandas DataFrame.

        Args:
            results: List of BenchmarkResult

        Returns:
            pandas DataFrame with all result fields as columns
        """
        import pandas as pd

        return pd.DataFrame([asdict(r) for r in results])

    def save_csv(
        self,
        results: List[BenchmarkResult],
        config: BenchmarkConfig,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Save results to a CSV file.

        Args:
            results: List of BenchmarkResult
            config: BenchmarkConfig (used for default filename)
            output_path: Optional explicit output path

        Returns:
            Path to the saved CSV file
        """
        import pandas as pd

        df = self.results_to_dataframe(results)

        if output_path is None:
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{config.name}_{timestamp}.csv"

        df.to_csv(output_path, index=False, float_format="%.3f")
        logger.info(f"Results saved to {output_path}")

        return output_path


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SDPA benchmarks from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all benchmarks from mlperf config
    python -m benchmark.sdpa_benchmark_training.runner --config mlperf

    # Dry run (show what would be executed)
    python -m benchmark.sdpa_benchmark_training.runner --config mlperf --dry-run

    # Filter by model name
    python -m benchmark.sdpa_benchmark_training.runner --config mlperf --filter llama3.1

    # Filter by backend
    python -m benchmark.sdpa_benchmark_training.runner --config mlperf --backend cudnn

    # Skip chart generation
    python -m benchmark.sdpa_benchmark_training.runner --config mlperf --no-chart
        """,
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Config name (e.g., 'mlperf'). Must be a Python file in configs/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print benchmark cases without executing",
    )
    parser.add_argument(
        "--filter",
        dest="filter_model",
        help="Filter by model name (substring match)",
    )
    parser.add_argument(
        "--backend",
        dest="filter_backend",
        help="Filter by backend (exact match)",
    )
    parser.add_argument(
        "--dtype",
        dest="filter_dtype",
        help="Filter by data type (exact match)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for CSV (default: artifacts/<config>_<timestamp>.csv)",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip chart generation",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and exit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Handle --list-configs
    if args.list_configs:
        from .configs import list_configs

        configs = list_configs()
        print("Available configurations:")
        for name in configs:
            print(f"  {name}")
        return

    # Load config
    from .configs import load_config

    try:
        config = load_config(args.config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    runner = BenchmarkRunner(verbose=not args.quiet)

    # Dry run mode
    if args.dry_run:
        cases = list(runner.expand_config(config))

        # Apply filters for display
        if args.filter_model:
            cases = [c for c in cases if args.filter_model in c["model"].name]
        if args.filter_backend:
            cases = [c for c in cases if c["backend"] == args.filter_backend]
        if args.filter_dtype:
            cases = [c for c in cases if c["data_type"] == args.filter_dtype]

        print(f"Would run {len(cases)} benchmark cases from config '{config.name}':")
        print()
        for i, case in enumerate(cases, 1):
            model = case["model"]
            det_str = "det" if case["deterministic_bwd"] else "non-det"
            print(
                f"  [{i}] {model.name} | "
                f"seq={case['q_seqlen']}x{case['kv_seqlen']} | "
                f"{case['backend']} | {case['data_type']} | "
                f"{case['attn_mask']} | {det_str}"
            )
        return

    # Run benchmarks
    results = runner.run_config(
        config,
        filter_model=args.filter_model,
        filter_backend=args.filter_backend,
        filter_dtype=args.filter_dtype,
    )

    if not results:
        print("No results to save", file=sys.stderr)
        sys.exit(1)

    # Save CSV
    csv_path = runner.save_csv(results, config, args.output)

    # Generate charts (separate chart per mask type for clarity)
    if not args.no_chart:
        try:
            from .charts import generate_charts_by_mask

            df = runner.results_to_dataframe(results)
            chart_paths = generate_charts_by_mask(df, config)
            for path in chart_paths:
                print(f"Chart saved to {path}")
        except ImportError as e:
            logger.warning(f"Could not generate chart (missing dependency): {e}")
        except Exception as e:
            logger.warning(f"Could not generate chart: {e}")

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
