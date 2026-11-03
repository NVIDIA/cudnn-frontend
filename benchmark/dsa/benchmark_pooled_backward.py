# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pooled DSA backward against reference formulations.

Indexer side: the cuDNN chain (pooled keys, predict and teacher recompute,
pool target, indexer backward, unpool) against the naive dense-score plus
dense full-teacher reference backward.
Attention side: compact expansion plus the SM100 sparse attention backward
against the argsort compaction route plus the same kernel.

Usage:
    python -m benchmark.dsa.benchmark_pooled_backward --seqlens 4096 8192 16384 32768 --output pooled_backward.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test" / "python"))

from deepseek_sparse_attention.cutedsl import pooled_training_reference as R  # noqa: E402
from cudnn.deepseek_sparse_attention.pooled import backward as PB  # noqa: E402
from cudnn.deepseek_sparse_attention.score_recompute.api import sparse_attn_score_recompute_wrapper, sparse_indexer_score_recompute_wrapper  # noqa: E402
from cudnn.deepseek_sparse_attention.sparse_attention_backward.api import sparse_attention_backward_wrapper  # noqa: E402
from cudnn.deepseek_sparse_attention.sparse_attention_forward.api import sparse_attention_forward_wrapper  # noqa: E402


class Timer:
    def __init__(self, warmup, repeats, flush_mb=256):
        self.warmup, self.repeats = warmup, repeats
        self.flush = torch.empty(flush_mb * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def __call__(self, fn):
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(self.repeats):
            self.flush.fill_(1)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        return statistics.median(times)


def build_case(seqlen, k_pools, pool_size, seed):
    c = R.make_case(1, seqlen, torch.device("cuda"), pool_size=pool_size, topk_tokens=k_pools * pool_size, seed=seed)
    index_q, weights, keys, gates, ape = c["index_q"], c["weights"], c["keys"], c["gates"], c["ape"]
    q_attn, k_attn, positions = c["q_attn"], c["k_attn"], c["positions"]
    softmax_scale = q_attn.shape[-1] ** -0.5
    pooled = PB.pooled_keys_forward(keys, gates, ape, pool_size)["pooled"]
    scores = R.pooled_dense_scores(index_q, weights, pooled, 1.0)
    pool_ids = R.select_pools(scores, R.eligible_pools(positions, pooled.shape[1], pool_size), k_pools)
    del scores
    expanded = PB.pooled_expand_compact(pool_ids, positions, pool_size, seqlen_k=seqlen, global_ids=True)
    token_ids, topk_length = expanded["token_ids"], expanded["topk_length"]
    s, h, d = q_attn.shape[1:]
    fwd = sparse_attention_forward_wrapper(
        q_attn.reshape(s, h, d), k_attn.reshape(-1, d), token_ids.reshape(s, -1), topk_length=topk_length.reshape(s), softmax_scale=softmax_scale
    )
    out, lse = fwd["out"].reshape(1, s, h, -1), fwd["lse"].reshape(1, s, h).float()
    dout = torch.randn_like(out)
    return dict(
        index_q=index_q,
        weights=weights,
        keys=keys,
        gates=gates,
        ape=ape,
        q_attn=q_attn,
        k_attn=k_attn,
        positions=positions,
        pooled=pooled,
        pool_ids=pool_ids,
        token_ids=token_ids,
        topk_length=topk_length,
        out=out,
        lse=lse,
        dout=dout,
        softmax_scale=softmax_scale,
        pool_size=pool_size,
        k_pools=k_pools,
    )


def cudnn_indexer_chain(c, timer, grad_loss):
    pool_size = c["pool_size"]
    s = c["index_q"].shape[1]
    stages = {}

    def pooled():
        return PB.pooled_keys_forward(c["keys"], c["gates"], c["ape"], pool_size)["pooled"]

    def predict():
        return sparse_indexer_score_recompute_wrapper(c["index_q"], c["pooled"], c["weights"], c["pool_ids"])["predict"]

    def teacher():
        return sparse_attn_score_recompute_wrapper(
            c["q_attn"], c["k_attn"], c["lse"], c["token_ids"], c["softmax_scale"], topk_length=c["topk_length"], topk_indices_global=True
        )["target"]

    stages["pooled_keys_us"] = timer(pooled)
    stages["predict_recompute_us"] = timer(predict)
    stages["teacher_recompute_us"] = timer(teacher)
    tt = teacher()
    stages["pool_target_us"] = timer(lambda: PB.pooled_indexer_target(tt, c["pool_ids"], pool_size)["target"])
    target = PB.pooled_indexer_target(tt, c["pool_ids"], pool_size)["target"]

    def teacher_fused():
        return PB.pooled_teacher_recompute(c["q_attn"], c["k_attn"], c["lse"], c["token_ids"], c["topk_length"], c["k_pools"], pool_size, c["softmax_scale"])[
            "target"
        ]

    stages["teacher_fused_us"] = timer(teacher_fused)
    fused = teacher_fused()
    stages["teacher_fused_rel_diff"] = ((fused - target).norm() / target.norm()).item()
    pred = predict()
    d_pooled = torch.empty_like(c["pooled"], dtype=torch.float32)
    work = target.clone()

    def indexer_bwd():
        work.copy_(target)
        return PB.indexer_backward_wrapper(
            c["index_q"], c["weights"], c["pooled"], work, pred, c["pool_ids"], grad_loss, sm_scale=1.0, loss_coeff=1.0, d_index_k=d_pooled
        )

    stages["indexer_backward_us"] = timer(indexer_bwd)
    stages["unpool_backward_us"] = timer(lambda: PB.unpool_backward(c["keys"], c["gates"], c["ape"], d_pooled, pool_size))

    def full():
        work.copy_(target)
        return PB.pooled_indexer_backward(c["index_q"], c["weights"], c["keys"], c["gates"], c["ape"], c["pool_ids"], work, grad_loss, pool_size, 1.0, 1.0)

    stages["chain_total_us"] = timer(full)

    def select():
        return PB.pooled_indexer_select(c["index_q"], c["pooled"], c["weights"], c["k_pools"], pool_size)

    stages["indexer_select_fwd_us"] = timer(select)
    sel = select()
    stages["select_predict_vs_recompute_rel_diff"] = (
        (sel["predict"] - sparse_indexer_score_recompute_wrapper(c["index_q"], c["pooled"], c["weights"], sel["pool_ids"])["predict"]).norm() / pred.norm()
    ).item()

    def full_predict_from_forward():
        work.copy_(target)
        return PB.pooled_indexer_backward(
            c["index_q"], c["weights"], c["keys"], c["gates"], c["ape"], c["pool_ids"], work, grad_loss, pool_size, 1.0, 1.0, predict=pred
        )

    stages["chain_predict_from_forward_us"] = timer(full_predict_from_forward)
    stages["chain_with_teacher_us"] = stages["chain_total_us"] + stages["teacher_recompute_us"] + stages["pool_target_us"]
    stages["chain_with_fused_teacher_us"] = stages["chain_total_us"] + stages["teacher_fused_us"]
    return stages


def naive_indexer_baseline(c, timer):
    pool_size = c["pool_size"]

    def run():
        return R.naive_pooled_indexer_backward(
            c["index_q"],
            c["weights"],
            c["keys"],
            c["gates"],
            c["ape"],
            c["q_attn"],
            c["k_attn"],
            c["token_ids"],
            c["pool_ids"],
            c["positions"],
            1.0,
            c["softmax_scale"],
            1.0,
            1.0,
            pool_size,
        )

    try:
        return {"naive_total_us": timer(run)}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"naive_total_us": None}


def attention_side(c, timer):
    pool_size = c["pool_size"]
    s, h, d = c["q_attn"].shape[1:]
    q, kv, out, dout, lse = (c[k] for k in ("q_attn", "k_attn", "out", "dout", "lse"))
    sink = torch.full((h,), float("-inf"), dtype=torch.float32, device="cuda")
    res = {}
    res["expand_compact_us"] = timer(lambda: PB.pooled_expand_compact(c["pool_ids"], c["positions"], pool_size, seqlen_k=s, global_ids=True))
    # open-source route: inference-layout ids, argsort compaction, then the same kernel
    inference_ids = R.expand_pools_inference(c["pool_ids"], c["positions"], pool_size).reshape(s, -1)
    res["argsort_compaction_us"] = timer(lambda: R.argsort_compact_indices(inference_ids))
    idx_c, len_c = R.argsort_compact_indices(inference_ids)

    def kernel_open_source():
        return sparse_attention_backward_wrapper(
            q.reshape(s, h, d),
            kv.reshape(-1, d),
            out.reshape(s, h, -1),
            dout.reshape(s, h, -1),
            lse.reshape(s, h),
            sink,
            idx_c,
            softmax_scale=c["softmax_scale"],
            topk_length=len_c,
        )

    def kernel_pooled():
        return PB.pooled_sparse_attention_backward(q, kv, out, dout, lse, c["pool_ids"], c["positions"], pool_size, softmax_scale=c["softmax_scale"])

    res["attn_bwd_open_source_kernel_us"] = timer(kernel_open_source)
    res["attn_bwd_open_source_total_us"] = res["attn_bwd_open_source_kernel_us"] + res["argsort_compaction_us"]
    res["attn_bwd_pooled_total_us"] = timer(kernel_pooled)

    def kernel_pooled_with_teacher():
        return PB.pooled_sparse_attention_backward_with_teacher(
            q,
            kv,
            out,
            dout,
            lse,
            c["pool_ids"],
            c["positions"],
            c["k_pools"],
            pool_size,
            softmax_scale=c["softmax_scale"],
            token_ids=c["token_ids"],
            topk_length=c["topk_length"],
        )

    res["attn_bwd_with_teacher_us"] = timer(kernel_pooled_with_teacher)
    fused_t = kernel_pooled_with_teacher()["target"]
    teacher_sep = PB.pooled_teacher_recompute(
        c["q_attn"], c["k_attn"], c["lse"], c["token_ids"], c["topk_length"], c["k_pools"], pool_size, c["softmax_scale"]
    )["target"]
    res["teacher_in_bwd_rel_diff"] = ((fused_t - teacher_sep).norm() / teacher_sep.norm()).item()
    # agreement of the two routes
    a = kernel_open_source()
    b = kernel_pooled()
    res["dq_rel_diff"] = ((a["dq"].float() - b["dq"].reshape(s, h, d).float()).norm() / a["dq"].float().norm()).item()
    res["dkv_rel_diff"] = ((a["dkv"].float() - b["dkv"].reshape(-1, d).float()).norm() / a["dkv"].float().norm()).item()
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--k-pools", type=int, default=512)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--naive-max-seqlen", type=int, default=16384, help="Skip the naive baseline above this length")
    parser.add_argument("--output", type=Path, default=Path("pooled_backward.json"))
    args = parser.parse_args()
    timer = Timer(args.warmup, args.repeats)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    results = {"device": torch.cuda.get_device_name(), "cases": []}
    for seqlen in args.seqlens:
        t0 = time.time()
        c = build_case(seqlen, args.k_pools, args.pool_size, seed=seqlen)
        row = {"seqlen": seqlen, "k_pools": args.k_pools, "pool_size": args.pool_size}
        row.update(cudnn_indexer_chain(c, timer, grad_loss))
        row.update(naive_indexer_baseline(c, timer) if seqlen <= args.naive_max_seqlen else {"naive_total_us": None})
        row.update(attention_side(c, timer))
        row["setup_s"] = round(time.time() - t0, 1)
        results["cases"].append(row)
        print(json.dumps(row), flush=True)
        del c
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
