# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-by-stage cost attribution for the gated attention block's torch baseline.

This is an ATTRIBUTION harness, not an A/B: it answers "where does the time go
and how does that shift with S", which is what decides the block's partitioning.
It deliberately does not compare two implementations, so it needs no control
pair -- when that day comes, use the protocol in the module docstring of
``gated_block_reference.py`` instead of extending this.

The number it exists to produce: the S at which the attention overtakes the two
projections. Below it the gate GEMM is the dominant term and has a softmax
shadow to hide in; above it the SDPA is ~99% of the layer and the whole fusion
thesis changes shape.

Usage::

    python benchmark_baseline.py [--seq 1024,2048,4096,8192,16384] [--batch 1] [--iters 50]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time

# cudnn BEFORE torch: torch bundles its own libcudnn, and whichever loads first
# wins the sublibrary search. Importing torch first made the graph path die with
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED against a 9.24 that the LD_LIBRARY_PATH
# 9.27 was supposed to shadow.
import cudnn  # noqa: F401  (import order is the point)
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gated_block_reference import (  # noqa: E402
    GEOMETRY_D4096_H32_KV2_D256,
    RefGeometry,
    apply_partial_rope,
    build_rope_tables,
    rms_norm,
    split_qkvg,
)


def _time_ms(fn, *, iters: int, warmup: int) -> float:
    """Median of ``iters`` timed launches, in ms. Warm the artifact first."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def analytic_flops(geom: RefGeometry, b: int, s: int) -> dict:
    """Per-layer FLOPs, so measured time can be read as achieved fraction.

    Attention counts both BMMs over the causal half; the projections are dense.
    """
    causal = 0.5 if geom.is_causal else 1.0
    attn = 2.0 * 2.0 * b * s * s * geom.h_q * geom.d_head * causal
    qkvg = 2.0 * b * s * geom.n_qkvg * geom.d_model
    o_proj = 2.0 * b * s * geom.d_model * geom.h_q * geom.d_head
    return {"attention": attn, "qkv_gate_proj": qkvg, "out_proj": o_proj, "total": attn + qkvg + o_proj}


# bf16/f16 dense MMA throughput per SM per clock. cc10 (Blackwell/Rubin) is
# 8192; the A100 is 2048 (312 TFLOPS / 108 SMs / 1.41 GHz). FP8 doubles cc10's
# figure -- a mixed-dtype chart needs one peak line PER DTYPE or every fp8 SOL
# is wrong by 2x.
_FLOPS_PER_CLK_PER_SM = {(10, 0): 8192, (10, 3): 8192, (10, 7): 8192, (9, 0): 2048, (8, 0): 2048}


class _SmClockSampler:
    """Poll the SM clock at ~1 kHz for the measurement window.

    The SOL denominator has to be the clock the kernel ACTUALLY ran at: a
    datasheet boost is not reachable and NVML's max-clock query is unreliable on
    some datacenter SKUs, so a sampled max is the honest figure. Without pynvml
    this reports nothing and the caller prints "n/a" -- loudly wrong beats
    quietly wrong.
    """

    def __init__(self):
        self._samples, self._stop, self._thread, self._nvml, self._handle = [], threading.Event(), None, None, None

    def start(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            # Index NVML by the VISIBLE device, not torch's index: on a masked
            # box those differ and you would sample an idle GPU's clock.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            phys = int(visible.split(",")[torch.cuda.current_device()]) if visible else torch.cuda.current_device()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(phys)
            self._nvml = pynvml
        except Exception:
            self._nvml = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self._samples.append(self._nvml.nvmlDeviceGetClockInfo(self._handle, self._nvml.NVML_CLOCK_SM))
            except Exception:
                break
            time.sleep(0.001)

    def stop_mhz(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        return max(self._samples) if self._samples else None


def _sol_text(gbs: float, ceiling: float) -> str:
    """Fraction-of-ceiling, or a loud complaint when the ratio is impossible.

    Above 100% the KERNEL is not the finding — the probe is. Say so instead of
    printing a plausible-looking number (`frost-tile-dsl.md` § 11).
    """
    if not ceiling:
        return "SOL n/a (no ceiling measured)"
    pct = 100.0 * gbs / ceiling
    if pct > 100.0:
        return f"{pct:.0f}% -> ABOVE THE MEASURED CEILING ({ceiling:.0f} GB/s), SO THE CEILING IS WRONG, NOT THE KERNEL"
    return f"{pct:.1f}% of the measured {ceiling:.0f} GB/s ceiling"


def mma_peak_tflops(clock_mhz) -> float:
    """Dense bf16 MMA peak at the sampled clock, or 0.0 when unknown."""
    if clock_mhz is None or not torch.cuda.is_available():
        return 0.0
    cc = torch.cuda.get_device_capability()
    per_clk = _FLOPS_PER_CLK_PER_SM.get(cc)
    if per_clk is None:
        return 0.0
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return per_clk * sms * (clock_mhz * 1e6) / 1e12


def measure_bandwidth_ceiling(*, iters: int = 20, warmup: int = 5) -> float:
    """Best achievable HBM bandwidth in GB/s, MEASURED — the honest denominator.

    A datasheet peak is the wrong denominator for a kernel: it is not reachable
    by any code. But a single naive probe is worse than wrong — it is
    CONFIDENTLY wrong, and it fails in the flattering direction. A first version
    of this timed one 1 GiB ``uint8`` ``copy_`` and reported 6797 GB/s on Rubin;
    the block's own gate kernel then measured **145% of it**, which is not a
    result, it is a proof that the denominator was too low.

    So: sweep sizes, race two different copies, and take the BEST — a ceiling is
    a maximum, and any probe that a real kernel beats was not one.

      * ``Tensor.copy_`` over uint8 (what the naive version used), and
      * a widened ``float4``-aligned copy, which is what a tuned streaming
        kernel actually issues.

    If a measured kernel still exceeds what this returns, do not report the
    ratio — the probe is still the thing that is wrong.
    """
    best = 0.0
    for nbytes in (1 << 28, 1 << 30, 1 << 31):
        try:
            src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            dst = torch.empty_like(src)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        for probe in (
            lambda: dst.copy_(src),
            # float4-wide: 16 B per lane, the width a streaming kernel uses.
            lambda: dst.view(torch.float32).copy_(src.view(torch.float32)),
        ):
            try:
                ms = _time_ms(probe, iters=iters, warmup=warmup)
            except Exception:
                continue
            best = max(best, 2.0 * nbytes / (ms * 1e-3) / 1e9)
        del src, dst
        torch.cuda.empty_cache()
    return best


def _make_frost_norm_rope_stage(geom: RefGeometry, b: int, s: int, dtype, want_rstd: bool, *, rows_per_group: int, threads_per_cta: int):
    """Stages (2)+(3) of the block, compiled — the fused FROST kernel."""
    from cudnn.gated_attention_block import GatedAttentionBlockGeometry
    from cudnn.gated_attention_block.api import _QkNormRope

    block_geom = GatedAttentionBlockGeometry(
        d_model=geom.d_model,
        h_q=geom.h_q,
        h_kv=geom.h_kv,
        d_head=geom.d_head,
        rope_dim=geom.rope_dim,
        qk_norm_eps=geom.qk_norm_eps,
        attn_scale=geom.attn_scale,
        is_causal=geom.is_causal,
    )
    block_geom.validate()
    stage = _QkNormRope(block_geom, batch=b, seq_len=s, dtype=dtype, want_rstd=want_rstd, rows_per_group=rows_per_group, threads_per_cta=threads_per_cta)
    stage.check_support()
    stage.compile()
    return stage


def _make_frost_sdpa_stage(geom: RefGeometry, b: int, s: int, dtype, device):
    """Stage (4) of the block, compiled — the FROST SM107 (256, 256) f16 kernel.

    Timing this next to torch's SDPA is the point of ``--frost-sdpa``: the
    baseline's sdpa-vs-projections crossover is a statement about torch until
    this row exists beside it.
    """
    from cudnn.gated_attention_block import GatedAttentionBlockGeometry
    from cudnn.gated_attention_block.api import _Sdpa

    block_geom = GatedAttentionBlockGeometry(
        d_model=geom.d_model,
        h_q=geom.h_q,
        h_kv=geom.h_kv,
        d_head=geom.d_head,
        rope_dim=geom.rope_dim,
        qk_norm_eps=geom.qk_norm_eps,
        attn_scale=geom.attn_scale,
        is_causal=geom.is_causal,
    )
    block_geom.validate()
    stage = _Sdpa(block_geom, batch=b, seq_len=s, dtype=dtype, device=device, want_lse=False)
    stage.check_support()
    stage.compile()
    return stage


def run_shape(
    geom: RefGeometry,
    b: int,
    s: int,
    *,
    iters: int,
    warmup: int,
    dtype=torch.bfloat16,
    frost_sdpa: bool = False,
    frost_norm_rope: bool = False,
    frost_proj: bool = False,
    frost_gate: bool = False,
    nr_rows_per_group: int = 4,
    nr_threads: int = 256,
    nr_want_rstd: bool = True,
) -> dict:
    dev = "cuda"
    d = geom.d_head
    h = torch.randn(b, s, geom.d_model, device=dev, dtype=dtype)
    w_qkvg = (torch.randn(geom.n_qkvg, geom.d_model, device=dev, dtype=torch.float32) * 0.02).to(dtype)
    w_o = (torch.randn(geom.d_model, geom.h_q * d, device=dev, dtype=torch.float32) * 0.02).to(dtype)
    w_qn = torch.ones(d, device=dev, dtype=dtype)
    w_kn = torch.ones(d, device=dev, dtype=dtype)
    cos, sin = build_rope_tables(s, geom.rope_dim, base=geom.rope_base, batch=b, device=dev, dtype=dtype)

    # Materialize each stage's input once, outside the timed region, so a stage
    # is timed on its own and not on its predecessor's allocation.
    proj = F.linear(h, w_qkvg)
    q0, gate, k0, v = split_qkvg(proj, geom)
    qn, _ = rms_norm(q0, w_qn, geom.qk_norm_eps)
    kn, _ = rms_norm(k0, w_kn, geom.qk_norm_eps)
    q = apply_partial_rope(qn, cos, sin, geom.rope_dim).transpose(1, 2).contiguous()
    k = apply_partial_rope(kn, cos, sin, geom.rope_dim).transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    o = F.scaled_dot_product_attention(q, k, vt, is_causal=geom.is_causal, scale=geom.scale, enable_gqa=True).transpose(1, 2).contiguous()
    o_gated = o * torch.sigmoid(gate)

    def stage_proj():
        F.linear(h, w_qkvg)

    def stage_norm_rope():
        a, _ = rms_norm(q0, w_qn, geom.qk_norm_eps)
        bb, _ = rms_norm(k0, w_kn, geom.qk_norm_eps)
        apply_partial_rope(a, cos, sin, geom.rope_dim)
        apply_partial_rope(bb, cos, sin, geom.rope_dim)

    def stage_sdpa():
        F.scaled_dot_product_attention(q, k, vt, is_causal=geom.is_causal, scale=geom.scale, enable_gqa=True)

    def stage_gate():
        o * torch.sigmoid(gate)

    def stage_out_proj():
        F.linear(o_gated.reshape(b, s, geom.h_q * d), w_o)

    stages = {
        "1 qkv_gate_proj": stage_proj,
        "2+3 qk_norm+rope": stage_norm_rope,
        "4 sdpa": stage_sdpa,
        "5 sigmoid_gate": stage_gate,
        "6 out_proj": stage_out_proj,
    }
    timings = {name: _time_ms(fn, iters=iters, warmup=warmup) for name, fn in stages.items()}
    timings["TOTAL (sum of stages)"] = sum(timings.values())

    extra = {}
    if frost_gate:
        from cudnn.gated_attention_block import GatedAttentionBlockGeometry
        from cudnn.gated_attention_block.api import _SigmoidGate, _VCompaction

        bg = GatedAttentionBlockGeometry(
            d_model=geom.d_model, h_q=geom.h_q, h_kv=geom.h_kv, d_head=geom.d_head, rope_dim=geom.rope_dim, is_causal=geom.is_causal
        )
        t_tok = b * s
        o_flat = o.reshape(t_tok, geom.h_q, d).contiguous()
        gate_flat = gate.reshape(t_tok, geom.h_q, d).contiguous()
        # OUT-OF-PLACE for the measurement, even though the block gates IN
        # PLACE. Identical DRAM traffic (2 reads + 1 write either way), but
        # in-place makes the kernel eat its own output: o *= sigmoid(gate) with
        # sigmoid < 1 drives o to zero within ~40 iterations, and zero pages
        # COMPRESS -- which is how the first version of this measured 145% of
        # the HBM ceiling. A benchmark whose data degenerates is measuring the
        # degeneracy.
        o_dst = torch.empty_like(o_flat)
        v_flat = v.reshape(t_tok, geom.h_kv, d).contiguous()
        v_dst = torch.empty_like(v_flat)
        for st, args in (
            (_SigmoidGate(bg, batch=b, seq_len=s, dtype=dtype), (o_flat, gate_flat, o_dst)),
            (_VCompaction(bg, batch=b, seq_len=s, dtype=dtype), (v_flat, v_dst)),
        ):
            st.check_support()
            st.compile()
            extra[f"5' {st.name} (FROST)"] = _time_ms(lambda st=st, a=args: st.execute(*a), iters=iters, warmup=warmup)
            extra[f"__ebytes_{st.name}"] = float(st.moved_bytes())
        del o_flat, gate_flat, o_dst, v_flat, v_dst
        torch.cuda.empty_cache()

    if frost_proj:
        from cudnn.gated_attention_block.api import _out_projection, _qkv_gate_projection

        for factory, src in ((_qkv_gate_projection, h), (_out_projection, o_gated.reshape(b, s, geom.h_q * d))):
            st = factory(geom, batch=b, seq_len=s, dtype=dtype)
            st.check_support()
            st.compile()
            wt = torch.empty(st.n, st.k, device=src.device, dtype=dtype)
            dst = torch.empty(st.m, st.n, device=src.device, dtype=dtype)
            ws = torch.empty(st.workspace_bytes(), dtype=torch.uint8, device=src.device)
            extra[f"{'1' if st.name == 'qkv_gate_proj' else '6'}' {st.name} (FROST)"] = _time_ms(
                lambda st=st, src=src, wt=wt, dst=dst, ws=ws: st.execute(src, wt, dst, ws), iters=iters, warmup=warmup
            )
            extra[f"__flops_{st.name}"] = float(st.flops())
            del wt, dst, ws
        torch.cuda.empty_cache()

    if frost_norm_rope:
        # BSHD-compact [B, S, H, D], the layout stage (1) produces.
        q_b = q0.contiguous()
        k_b = k0.contiguous()
        rstd_q = torch.empty(b, s, geom.h_q, device=q_b.device, dtype=torch.float32) if nr_want_rstd else None
        rstd_k = torch.empty(b, s, geom.h_kv, device=q_b.device, dtype=torch.float32) if nr_want_rstd else None
        st = _make_frost_norm_rope_stage(geom, b, s, dtype, want_rstd=nr_want_rstd, rows_per_group=nr_rows_per_group, threads_per_cta=nr_threads)
        extra["2+3' norm+rope (FROST)"] = _time_ms(
            lambda: st.execute(q_b, k_b, w_qn, w_kn, cos, sin, rstd_q=rstd_q, rstd_k=rstd_k),
            iters=iters,
            warmup=warmup,
        )
        extra["__norm_rope_bytes"] = float(st.moved_bytes())
        del q_b, k_b, rstd_q, rstd_k

    frost = None
    if frost_sdpa:
        # BSHD-compact, the layout stage (1) produces and stage (4) consumes.
        q_b = q.transpose(1, 2).contiguous()
        k_b = k.transpose(1, 2).contiguous()
        v_b = vt.transpose(1, 2).contiguous()
        o_b = torch.empty_like(q_b)
        stage = _make_frost_sdpa_stage(geom, b, s, dtype, q_b.device)
        ws = torch.empty(max(stage.scratch_workspace_bytes(), 1), dtype=torch.uint8, device=q_b.device)
        frost = _time_ms(lambda: stage.execute(q_b, k_b, v_b, o_b, workspace=ws), iters=iters, warmup=warmup)
        del q_b, k_b, v_b, o_b, ws
    if frost is not None:
        timings["4' sdpa (FROST sm107)"] = frost
    timings.update(extra)

    del proj, q0, gate, k0, v, qn, kn, q, k, vt, o, o_gated
    torch.cuda.empty_cache()
    return timings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="1024,2048,4096,8192,16384")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-causal", action="store_true")
    ap.add_argument("--frost-gate", action="store_true", help="also time stage (5) and the V compaction, with HBM SOL")
    ap.add_argument("--frost-proj", action="store_true", help="also time stages (1) and (6) on the shipped FROST GEMM and report MMA SOL")
    ap.add_argument("--nr-rows-per-group", type=int, default=4, help="rows each lane group processes -- the memory-level-parallelism knob")
    ap.add_argument("--nr-threads", type=int, default=256)
    ap.add_argument("--nr-no-rstd", action="store_true", help="drop the backward's rstd outputs -- the inference-only shape of stages (2)+(3)")
    ap.add_argument("--frost-norm-rope", action="store_true", help="also time the block's stages (2)+(3) -- the fused FROST kernel -- and report its HBM SOL")
    ap.add_argument("--frost-sdpa", action="store_true", help="also time the block's stage (4) -- the FROST SM107 kernel -- beside torch's SDPA")
    args = ap.parse_args()

    geom = GEOMETRY_D4096_H32_KV2_D256
    if args.no_causal:
        geom = RefGeometry(**{**geom.__dict__, "is_causal": False})

    print(f"device: {torch.cuda.get_device_name()}  cc={torch.cuda.get_device_capability()}  torch={torch.__version__}")
    print(
        f"geometry: d_model={geom.d_model} h_q={geom.h_q} h_kv={geom.h_kv} d={geom.d_head} rope={geom.rope_dim} "
        f"N_qkvg={geom.n_qkvg} causal={geom.is_causal}"
    )
    print(f"protocol: median of {args.iters} launches after {args.warmup} warmups, cuda events, artifacts built outside the timed region")
    sampler = _SmClockSampler()
    sampler.start()
    ceiling = measure_bandwidth_ceiling() if (args.frost_norm_rope or args.frost_gate) else 0.0
    if ceiling:
        print(f"measured HBM ceiling: {ceiling:.0f} GB/s (1 GiB device-to-device copy, read+write)")
    print()

    clock_mhz = None
    peak = 0.0

    for s in [int(x) for x in args.seq.split(",")]:
        try:
            t = run_shape(
                geom,
                args.batch,
                s,
                iters=args.iters,
                warmup=args.warmup,
                frost_sdpa=args.frost_sdpa,
                frost_norm_rope=args.frost_norm_rope,
                nr_rows_per_group=args.nr_rows_per_group,
                nr_threads=args.nr_threads,
                nr_want_rstd=not args.nr_no_rstd,
                frost_proj=args.frost_proj,
                frost_gate=args.frost_gate,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"S={s}: OOM, skipped")
            torch.cuda.empty_cache()
            continue
        if clock_mhz is None:
            clock_mhz = sampler.stop_mhz()
            peak = mma_peak_tflops(clock_mhz)
        fl = analytic_flops(geom, args.batch, s)
        nr_bytes = t.pop("__norm_rope_bytes", None)
        proj_flops = {name[len("__flops_") :]: t.pop(name) for name in [k for k in t if k.startswith("__flops_")]}
        e_bytes = {name[len("__ebytes_") :]: t.pop(name) for name in [k for k in t if k.startswith("__ebytes_")]}
        total = t["TOTAL (sum of stages)"]
        print(f"--- B={args.batch} S={s} ---")
        for name, ms in t.items():
            share = 100.0 * ms / total
            tag = "  (not in TOTAL)" if "'" in name else ""
            print(f"  {name:<24} {ms:8.3f} ms  {share:5.1f}%{tag}")
        for pname, pflops in proj_flops.items():
            key = f"{'1' if pname == 'qkv_gate_proj' else '6'}' {pname} (FROST)"
            torch_key = "1 qkv_gate_proj" if pname == "qkv_gate_proj" else "6 out_proj"
            ms = t[key]
            tf = pflops / (ms * 1e-3) / 1e12
            sol = f"{100.0 * tf / peak:.1f}% of MMA peak ({peak:.0f} TFLOP/s @ {clock_mhz} MHz)" if peak else "SOL n/a (no clock sample -- pip install pynvml)"
            print(f"  FROST {pname:<14}: {t[torch_key] / ms:.2f}x vs cuBLAS   {tf:.0f} TFLOP/s   -> {sol}")
        for ename, eb in e_bytes.items():
            ems = t[f"5' {ename} (FROST)"]
            gbs = eb / (ems * 1e-3) / 1e9
            sol = _sol_text(gbs, ceiling)
            vs = f"{t['5 sigmoid_gate'] / ems:.2f}x vs torch   " if ename == "sigmoid_gate" else " " * 17
            print(f"  FROST {ename:<14}: {vs}{gbs:.0f} GB/s over {eb / 1e6:.0f} MB   -> {sol}")
        nr_ms = t.get("2+3' norm+rope (FROST)")
        if nr_ms is not None and nr_bytes:
            gbs = nr_bytes / (nr_ms * 1e-3) / 1e9
            sol = _sol_text(gbs, ceiling)
            print(f"  FROST norm+rope          : {t['2+3 qk_norm+rope'] / nr_ms:.2f}x vs torch   {gbs:.0f} GB/s over {nr_bytes / 1e6:.0f} MB   -> {sol}")
        frost_ms = t.get("4' sdpa (FROST sm107)")
        if frost_ms is not None:
            attn_flops = analytic_flops(geom, args.batch, s)["attention"]
            print(
                f"  FROST vs torch SDPA      : {t['4 sdpa'] / frost_ms:.2f}x   "
                f"({attn_flops / (frost_ms * 1e-3) / 1e12:.0f} vs {attn_flops / (t['4 sdpa'] * 1e-3) / 1e12:.0f} TFLOP/s on attention FLOPs)"
            )
        proj_ms = t["1 qkv_gate_proj"] + t["6 out_proj"]
        print(f"  attention vs projections : sdpa {t['4 sdpa']:.3f} ms  vs  proj {proj_ms:.3f} ms  -> ratio {t['4 sdpa'] / proj_ms:.2f}x")
        if frost_ms is not None:
            print(f"  ... with the FROST SDPA  : sdpa {frost_ms:.3f} ms  vs  proj {proj_ms:.3f} ms  -> ratio {frost_ms / proj_ms:.2f}x")
        print(
            f"  analytic FLOP share      : attention {100.0 * fl['attention'] / fl['total']:.1f}%   projections {100.0 * (fl['total'] - fl['attention']) / fl['total']:.1f}%"
        )
        print(f"  achieved (sum of stages) : {fl['total'] / (total * 1e-3) / 1e12:.1f} TFLOP/s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
