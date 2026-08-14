"""Autotune the warp LN/RMS forward per shape: sweep (wn, persist-cap), record
GB/s, report the fastest config. Data feeds a persist-cap / wn heuristic.

    python autotune_warp_fwd.py            # sweep config shapes, write CSV
"""
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
D = os.path.join(_REPO, "python", "cudnn")
stub = types.ModuleType("cudnn"); stub.__path__ = [D]; stub.pygraph = type("pygraph", (), {}); sys.modules["cudnn"] = stub
sys.path.insert(0, _REPO)

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from cudnn.norm.config_sm100 import NormVariant as NV, TemplateParams, rowwise_spec, warp_cfg_candidates
from cudnn.norm.dtypes import torch_dtype_to_str
from cudnn.norm.fprop.kernels import layernorm_warp_sm100 as lw
from benchmark.norms.configs.all_models import CONFIG

NSM = torch.cuda.get_device_properties(0).multi_processor_count
l2 = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)


def dev_ms(fn, it=30, wu=12):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        l2.zero_()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            with record_function("op"): fn()
            torch.cuda.synchronize()
        ka = p.key_averages(); ev = [i for i in ka if i.key == "op"]
        ts.append((ev[0].device_time if ev else sum(i.device_time for i in ka if i.device_time > 0)) / 1000)
    return float(np.median(ts))


# cuDNN reference GB/s (from the latest sweep; ~10-20% run variance)
CUDNN = {"llama3-8b": 5779, "llama3-70b": 5075, "llama31-405b": 5246, "llama4-e16": 5520,
         "gpt3-175b": 3398, "mixtral-8x7b": 5420, "mixtral-8x22b": 5283, "nemotronh-56b": 5466,
         "deepseek-v3-2048x4096": 3497, "deepseek-v3-131072x128": 3227, "deepseek-v3-8192x128": 992,
         "qwen3-235b": 5538, "qwen3-30b-4096x2048": 4155, "qwen3-30b-131072x128": 3219,
         "qwen3-30b-16384x128": 1515}
CAP_MULTS = [2, 3, 4, 6, 8, 0]  # 0 = no cap (one tile/CTA)


def main():
    rows = []
    seen = set()
    print(f"SM count = {NSM}\n")
    print(f"{'model':>22} {'N':>7} {'C':>6} | {'best cfg (wn,ldgs,cap/nsm)':>26} {'frost':>6} {'cuDNN':>6} {'ratio':>5}")
    for p in CONFIG.norms:
        if p.name in seen:
            continue
        seen.add(p.name)
        variant = NV.RMS_NORM if p.norm_type == "rms_norm" else NV.LAYER_NORM
        dt = torch.bfloat16
        io = torch_dtype_to_str(dt)
        params = TemplateParams(variant=variant, io_dtype=io, has_beta=p.has_bias)
        x = torch.randn(p.N, p.C, device="cuda", dtype=dt)
        g = torch.randn(p.C, device="cuda", dtype=dt)
        b = torch.randn(p.C, device="cuda", dtype=dt) if p.has_bias else None
        spec = rowwise_spec(variant, x.shape, normalized_shape=[p.C])
        x2d = x.reshape(spec.R, spec.M)
        cands = warp_cfg_candidates(params, p.C)
        if not cands:
            continue
        best = (0.0, None)
        for wcfg in cands:
            tpr, wn, intra, ldgs, rpc, bt, V = wcfg
            full = (spec.R + rpc - 1) // rpc
            for m in CAP_MULTS:
                cap = full if m == 0 else min(full, NSM * m)
                lw._CTAS_CAP = cap
                try:
                    lw.forward(spec, x2d, g, b, eps=1e-5, wcfg=wcfg, params=params)  # compile/warm
                    ms = dev_ms(lambda: lw.forward(spec, x2d, g, b, eps=1e-5, wcfg=wcfg, params=params))
                except Exception:
                    continue
                gb = 2 * p.N * p.C * 2 / (ms * 1e-3) / 1e9
                rows.append((p.name, p.N, p.C, wn, ldgs, tpr, rpc, m, cap, round(gb)))
                if gb > best[0]:
                    best = (gb, (wn, ldgs, m))
        lw._CTAS_CAP = 0
        gb, (wn, ldgs, m) = best
        cud = CUDNN.get(p.name, 0)
        capstr = "full" if m == 0 else f"{m}x"
        print(f"{p.name:>22} {p.N:>7} {p.C:>6} | wn={wn} ldgs={ldgs:<2} cap={capstr:<4} {'':>10} {gb:>6.0f} {cud:>6} {gb/cud if cud else 0:>5.2f}")

    # write full data
    out = os.path.join(_HERE, "results", "warp_fwd_autotune.csv")
    with open(out, "w") as f:
        f.write("model,N,C,wn,ldgs,tpr,rpc,cap_mult,cap,gbps\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nwrote {out} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
