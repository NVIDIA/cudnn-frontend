"""Peak-vs-peak SDPA bench for the autoregressive DiT shape.

Sweeps FAv4 ``num_splits`` and reports the best, paired against cuDNN
(whatever the linked libcudnn provides; this script is intended to run
against a cuDNN build that has split-K so the comparison is split-KV
on both sides).

CSV schema matches the rest of ``benchmark.sdpa_benchmark_training``
plus an extra ``num_splits`` column so the per-seqlen winners are
visible.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

import torch
import cudnn

from benchmark.sdpa_benchmark_training.benchmark_single_sdpa import run_benchmark

H = 9
D = 128
S_KV = 62208
S_Q_LIST = [985, 1024, 2048, 4096, 8192]
DTYPES_CUDNN = ["bfloat16", "fp8", "mxfp8"]
FA4_SPLITS_SWEEP = [1, 2, 4, 8, 16, 32]
WARMUP = 5
ITERS = 30


def run(backend, dtype, s_q, fa4_num_splits=None):
    return run_benchmark(
        batch_size=1,
        q_seqlen=s_q,
        kv_seqlen=S_KV,
        num_q_heads=H,
        num_kv_heads=H,
        head_dim=D,
        data_type=dtype,
        backend=backend,
        attn_mask="no_mask",
        profile_pass="fwd",
        num_iterations=ITERS,
        num_warmup_iterations=WARMUP,
        skip_ref=True,
        deterministic_bwd=False,
        sliding_window_size=None,
        fa4_num_splits=fa4_num_splits,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output CSV path")
    args = p.parse_args()

    rows = []
    gpu_name = torch.cuda.get_device_name(0)
    cudnn_be = cudnn.backend_version()
    cudnn_fe = cudnn.__version__
    log.info(f"gpu={gpu_name}  cudnn-frontend={cudnn_fe}  backend={cudnn_be}")

    for s_q in S_Q_LIST:
        # cuDNN on all dtypes
        for dt in DTYPES_CUDNN:
            try:
                r = run("cudnn", dt, s_q)
                log.info(f"cudnn {dt:>8} s_q={s_q:>5}: {r['time_ms']:.3f} ms  {r['tflops']:.1f} TF")
                rows.append(
                    dict(
                        backend="cudnn",
                        data_type=dt,
                        q_seqlen=s_q,
                        kv_seqlen=S_KV,
                        num_splits=0,
                        time_ms=r["time_ms"],
                        tflops=r["tflops"],
                        gpu_name=gpu_name,
                        cudnn_backend_version=cudnn_be,
                    )
                )
            except Exception as e:
                log.info(f"cudnn {dt} s_q={s_q} FAILED: {e}")

        # FAv4 BF16, sweep num_splits
        best = None
        for ks in FA4_SPLITS_SWEEP:
            try:
                r = run("flash_attention_4", "bfloat16", s_q, fa4_num_splits=ks)
                log.info(f"  fa4  bfloat16 s_q={s_q:>5} num_splits={ks:>2}: {r['time_ms']:.3f} ms  {r['tflops']:.1f} TF")
                if best is None or r["time_ms"] < best["time_ms"]:
                    best = dict(r)
                    best["num_splits"] = ks
            except Exception as e:
                log.info(f"  fa4  bfloat16 s_q={s_q} num_splits={ks} FAILED: {e}")
        if best is not None:
            log.info(f"fa4  bfloat16 s_q={s_q:>5} BEST num_splits={best['num_splits']:>2}: {best['time_ms']:.3f} ms  {best['tflops']:.1f} TF")
            rows.append(
                dict(
                    backend="flash_attention_4",
                    data_type="bfloat16",
                    q_seqlen=s_q,
                    kv_seqlen=S_KV,
                    num_splits=best["num_splits"],
                    time_ms=best["time_ms"],
                    tflops=best["tflops"],
                    gpu_name=gpu_name,
                    cudnn_backend_version=cudnn_be,
                )
            )

    fields = ["backend", "data_type", "q_seqlen", "kv_seqlen", "num_splits", "time_ms", "tflops", "gpu_name", "cudnn_backend_version"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
