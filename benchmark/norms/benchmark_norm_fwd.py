# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BatchNorm / InstanceNorm / GroupNorm FORWARD: bandwidth utilisation vs the
achievable-copy ceiling, and vs PyTorch.

Metric: GB/s counting the MINIMAL traffic a forward must move --
``2 * N*C*H*W * elem_bytes`` (read x, write y); mean/rstd are O(N*G). The ceiling
is what a pure copy achieves on the same tensor, so ``util`` is the fraction of
achievable bandwidth. A norm forward cannot overlap its read and write phases the
way a copy does, so ~0.85-0.92 is the practical maximum (see BN_FWD_NOTES.md).

Also reports whether the output preserved a channels-last input, since a forward
that silently transposes forces the backward onto the transposing path too.

  --variant {bn,in,gn} --layout {nchw,nhwc} --n N --groups G
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_CUDNN = os.path.abspath(os.path.join(_HERE, "..", "..", "python", "cudnn"))
if "cudnn" not in sys.modules:
    import enum as _enum

    _stub = types.ModuleType("cudnn")
    _stub.__path__ = [_REPO_CUDNN]
    _stub.pygraph = type("pygraph", (), {})
    # cudnn.frost.buffers imports this at module scope (the CuTe DSL version gate
    # lives behind it); it is only called into from a method, so a placeholder does.
    _stub._pybind_module = types.ModuleType("cudnn._pybind_module")

    class _DT(_enum.Enum):
        HALF = 1
        BFLOAT16 = 2
        FLOAT = 3
        NOT_SET = 0

    _stub.data_type = _DT
    sys.modules["cudnn"] = _stub

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from cudnn.norm import NormVariant, norm_fprop  # noqa: E402

RN50 = [(64, 56, 56), (256, 56, 56), (128, 28, 28), (256, 28, 28), (512, 14, 14), (512, 7, 7), (2048, 7, 7)]
_l2 = None


def dev_ms(fn, iters=30, warmup=10):
    global _l2
    if _l2 is None:
        _l2 = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.int8)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        _l2.zero_()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            with record_function("op"):
                fn()
            torch.cuda.synchronize()
        ka = p.key_averages()
        ev = [i for i in ka if i.key == "op"]
        us = ev[0].device_time if ev else sum(i.device_time for i in ka if i.device_time > 0)
        ts.append(us / 1000.0)
    return float(statistics.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["bn", "in", "gn"], default="bn")
    ap.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--groups", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    NV = {"bn": NormVariant.BATCH_NORM, "in": NormVariant.INSTANCE_NORM, "gn": NormVariant.GROUP_NORM}[a.variant]
    print(f"# {a.variant} FWD {a.layout} {a.dtype} N={a.n}" + (f" G={a.groups}" if a.variant == "gn" else ""))
    hdr = f"{'shape':<17}{'ceil GB/s':>10}{'fwd GB/s':>10}{'util':>7}{'torch':>9}{'spd':>8}{'keeps':>7}"
    print(hdr)
    print("-" * len(hdr))
    utils = []
    for C, H, W in RN50:
        G = min(a.groups, C)
        x = torch.randn(a.n, C, H, W, device="cuda", dtype=dt)
        if a.layout == "nhwc":
            x = x.to(memory_format=torch.channels_last)
        g = torch.randn(C, device="cuda", dtype=dt)
        b = torch.randn(C, device="cuda", dtype=dt)
        nb = a.n * C * H * W * x.element_size()
        yc = torch.empty_like(x)
        ceil = 2 * nb / (dev_ms(lambda: yc.copy_(x)) * 1e-3) / 1e9
        kw = dict(eps=1e-5)
        if a.variant == "gn":
            kw["num_groups"] = G
        if a.variant == "bn":
            kw["training"] = True
        try:
            y, _, _ = norm_fprop(NV, x, g, b, **kw)
            torch.cuda.synchronize()
            keeps = bool(y.is_contiguous(memory_format=torch.channels_last)) if a.layout == "nhwc" else True
            t = dev_ms(lambda: norm_fprop(NV, x, g, b, **kw))
            gb = 2 * nb / (t * 1e-3) / 1e9
        except Exception as e:
            print(f"C={C:<5}{H}x{W:<8} ERR {type(e).__name__}: {str(e)[:40]}")
            continue
        if a.variant == "gn":
            ref = lambda: torch.nn.functional.group_norm(x, G, g, b, 1e-5)  # noqa: E731
        elif a.variant == "in":
            ref = lambda: torch.nn.functional.instance_norm(x, weight=g, bias=b, use_input_stats=True, eps=1e-5)  # noqa: E731
        else:
            ref = lambda: torch.nn.functional.batch_norm(x, None, None, g, b, True, 0.1, 1e-5)  # noqa: E731
        tgb = 2 * nb / (dev_ms(ref) * 1e-3) / 1e9
        utils.append(gb / ceil)
        print(f"C={C:<5}{H}x{W:<8}{ceil:>10.0f}{gb:>10.0f}{gb/ceil:>7.2f}{tgb:>9.0f}{gb/tgb:>7.2f}x{str(keeps):>7}", flush=True)
    if utils:
        print("-" * len(hdr))
        print(f"geomean util: {math.exp(sum(math.log(u) for u in utils)/len(utils)):.2f}")


main()
