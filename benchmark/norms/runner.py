"""
Benchmark runner with configuration expansion, execution, and result collection.

Usage:
    python -m benchmark.norms.runner --config llama3_8b
    python -m benchmark.norms.runner --config llama3_8b --dry-run
"""

import itertools
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional

from .config_types import NormBenchmarkConfig, NormBenchmarkResult, NormPreset

logger = logging.getLogger(__name__)


def log_environment_info():
    """Log environment information."""
    try:
        import torch

        logger.info(f"torch.__version__ = '{torch.__version__}'")
        logger.info(f"torch.version.cuda = '{torch.version.cuda}'")
        logger.info(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"torch.cuda.get_device_name() = '{torch.cuda.get_device_name(torch.cuda.current_device())}'")
    except ImportError:
        logger.warning("torch not available")

    try:
        import cudnn

        logger.info(f"cuDNN Backend Version: {cudnn.backend_version()}")
        logger.info(f"cuDNN Frontend Version: '{cudnn.__version__}'")
    except ImportError:
        logger.warning("cudnn not available")


class NormBenchmarkRunner:
    """
    Runs norm benchmarks from configurations with cartesian product expansion.

    The runner takes a NormBenchmarkConfig and expands it into individual
    benchmark cases via cartesian product of all configuration dimensions.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._setup_logging()

    def _setup_logging(self):
        level = logging.INFO if self.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="[%(levelname)s] %(message)s",
            stream=sys.stderr,
        )

    def expand_config(self, config: NormBenchmarkConfig) -> Iterator[Dict[str, Any]]:
        """
        Expand a NormBenchmarkConfig into individual benchmark cases.

        Cartesian product: norms x backends x data_types
        """
        for norm, backend, data_type in itertools.product(
            config.norms,
            config.backends,
            config.data_types,
        ):
            yield {
                "config_name": config.name,
                "norm": norm,
                "backend": backend,
                "data_type": data_type,
                "profile_pass": config.profile_pass,
                "num_iterations": config.num_iterations,
                "num_warmup_iterations": config.num_warmup_iterations,
            }

    def run_single(self, case: Dict[str, Any]) -> NormBenchmarkResult:
        """Run a single benchmark case."""
        norm: NormPreset = case["norm"]

        try:
            from .benchmark_single_norm import run_benchmark

            result = run_benchmark(
                norm_type=norm.norm_type,
                N=norm.N,
                C=norm.C,
                epsilon=norm.epsilon,
                has_bias=norm.has_bias,
                data_type=case["data_type"],
                backend=case["backend"],
                profile_pass=case["profile_pass"],
                num_iterations=case["num_iterations"],
                num_warmup_iterations=case["num_warmup_iterations"],
            )

            return NormBenchmarkResult(
                config_name=case["config_name"],
                norm_name=norm.name,
                norm_type=norm.norm_type,
                backend=case["backend"],
                data_type=case["data_type"],
                N=norm.N,
                C=norm.C,
                epsilon=norm.epsilon,
                has_bias=norm.has_bias,
                profile_pass=case["profile_pass"],
                fwd_time_ms=result["fwd_time_ms"],
                bwd_time_ms=result["bwd_time_ms"],
                fwd_gbps=result["fwd_gbps"],
                bwd_gbps=result["bwd_gbps"],
                num_iterations=case["num_iterations"],
                success=True,
                gpu_name=result.get("gpu_name"),
                cudnn_version=result.get("cudnn_version"),
                cudnn_backend_version=result.get("cudnn_backend_version"),
            )

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return NormBenchmarkResult(
                config_name=case["config_name"],
                norm_name=norm.name,
                norm_type=norm.norm_type,
                backend=case["backend"],
                data_type=case["data_type"],
                N=norm.N,
                C=norm.C,
                epsilon=norm.epsilon,
                has_bias=norm.has_bias,
                profile_pass=case["profile_pass"],
                fwd_time_ms=float("inf"),
                bwd_time_ms=float("inf"),
                fwd_gbps=0.0,
                bwd_gbps=0.0,
                num_iterations=case["num_iterations"],
                success=False,
                error_message=str(e),
            )

    def run_config(
        self,
        config: NormBenchmarkConfig,
        filter_norm: Optional[str] = None,
        filter_backend: Optional[str] = None,
        filter_dtype: Optional[str] = None,
    ) -> List[NormBenchmarkResult]:
        """Run all benchmarks from a configuration."""
        log_environment_info()
        logger.info("")

        results = []
        cases = list(self.expand_config(config))

        if filter_norm:
            cases = [c for c in cases if filter_norm in c["norm"].name]
        if filter_backend:
            cases = [c for c in cases if c["backend"] == filter_backend]
        if filter_dtype:
            cases = [c for c in cases if c["data_type"] == filter_dtype]

        if not cases:
            logger.warning("No benchmark cases to run after applying filters")
            return results

        logger.info(f"Running {len(cases)} benchmark cases from config '{config.name}'")

        for i, case in enumerate(cases, 1):
            norm = case["norm"]
            logger.info(f"[{i}/{len(cases)}] {norm.name} | " f"{norm.norm_type} | N={norm.N} C={norm.C} | " f"{case['backend']} | {case['data_type']}")

            result = self.run_single(case)
            results.append(result)

            if result.success:
                fwd_info = f"fwd: {result.fwd_time_ms:.3f}ms ({result.fwd_gbps:.1f} GB/s)"
                bwd_info = f"bwd: {result.bwd_time_ms:.3f}ms ({result.bwd_gbps:.1f} GB/s)"
                logger.info(f"  -> {fwd_info}, {bwd_info}")
            else:
                logger.warning(f"  -> FAILED: {result.error_message}")

        return results

    def results_to_dataframe(self, results: List[NormBenchmarkResult]):
        import pandas as pd

        return pd.DataFrame([asdict(r) for r in results])

    def save_csv(
        self,
        results: List[NormBenchmarkResult],
        config: NormBenchmarkConfig,
        output_path: Optional[Path] = None,
    ) -> Path:
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
        description="Run norm benchmarks from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmark.norms.runner --config llama3_8b
    python -m benchmark.norms.runner --config llama3_8b --dry-run
    python -m benchmark.norms.runner --config all_models --backend cudnn
    python -m benchmark.norms.runner --list-configs
        """,
    )

    parser.add_argument("--config", required=True, help="Config name (e.g., 'llama3_8b')")
    parser.add_argument("--dry-run", action="store_true", help="Print benchmark cases without executing")
    parser.add_argument("--filter", dest="filter_norm", help="Filter by norm name (substring match)")
    parser.add_argument("--backend", dest="filter_backend", help="Filter by backend (exact match)")
    parser.add_argument("--dtype", dest="filter_dtype", help="Filter by data type (exact match)")
    parser.add_argument("--output", type=Path, help="Output path for CSV")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--list-configs", action="store_true", help="List available configurations and exit")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    if args.list_configs:
        from .configs import list_configs

        configs = list_configs()
        print("Available configurations:")
        for name in configs:
            print(f"  {name}")
        return

    from .configs import load_config

    try:
        config = load_config(args.config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    runner = NormBenchmarkRunner(verbose=not args.quiet)

    if args.dry_run:
        cases = list(runner.expand_config(config))

        if args.filter_norm:
            cases = [c for c in cases if args.filter_norm in c["norm"].name]
        if args.filter_backend:
            cases = [c for c in cases if c["backend"] == args.filter_backend]
        if args.filter_dtype:
            cases = [c for c in cases if c["data_type"] == args.filter_dtype]

        print(f"Would run {len(cases)} benchmark cases from config '{config.name}':")
        print()
        for i, case in enumerate(cases, 1):
            norm = case["norm"]
            print(f"  [{i}] {norm.name} | {norm.norm_type} | " f"N={norm.N} C={norm.C} | " f"{case['backend']} | {case['data_type']}")
        return

    results = runner.run_config(
        config,
        filter_norm=args.filter_norm,
        filter_backend=args.filter_backend,
        filter_dtype=args.filter_dtype,
    )

    if not results:
        print("No results to save", file=sys.stderr)
        sys.exit(1)

    csv_path = runner.save_csv(results, config, args.output)

    if not args.no_chart:
        try:
            from .charts import generate_norm_charts

            df = runner.results_to_dataframe(results)
            chart_paths = generate_norm_charts(df, config)
            for path in chart_paths:
                print(f"Chart saved to {path}")
        except ImportError as e:
            logger.warning(f"Could not generate chart (missing dependency): {e}")
        except Exception as e:
            logger.warning(f"Could not generate chart: {e}")

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
