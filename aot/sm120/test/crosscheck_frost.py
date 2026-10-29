#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise cross-check of libcudnn_frontend_aot_sm120.so against the in-process Python FROST engine.

For each case the AOT plan picks a knob set from its selection table (cudnn_fe_aot_sm120_sdpa_query); the Python
engine (cudnn.sdpa.fwd.api_dsl.SdpaFwdDslSm120) is then built with exactly those knobs and both run the same inputs.
The outputs must be identical bit for bit: the AOT kernels are the same templates, compiled ahead of time with dynamic
extents and strides instead of per-plan in the process.

    PYTHONPATH=<repo>/python python aot/sm120/test/crosscheck_frost.py --lib <dir>/lib/libcudnn_frontend_aot_sm120.so

Needs torch, the CuTeDSL wheel and an sm_120 GPU; the AOT library itself is driven through its C ABI (ctypes).
"""

import argparse
import ctypes
import math
import sys

import torch

I32, I64 = ctypes.c_int32, ctypes.c_int64


class Desc(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32)] + [
        (n, I32) for n in ("head_dim", "q_dtype", "kv_dtype", "o_dtype", "mask", "window_left", "has_sinks", "gqa_ratio")
    ]


class Knobs(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32)] + [(n, I32) for n in ("pack_gqa", "sched_policy", "split_kv", "tile_m", "tile_n")]


class Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("b", I32),
        ("h_q", I32),
        ("h_kv", I32),
        ("s_q", I32),
        ("s_kv", I32),
        ("q", ctypes.c_void_p),
        ("k", ctypes.c_void_p),
        ("v", ctypes.c_void_p),
        ("o", ctypes.c_void_p),
        ("q_stride", I64 * 3),
        ("k_stride", I64 * 3),
        ("v_stride", I64 * 3),
        ("o_stride", I64 * 3),
        ("scale", ctypes.c_float),
        ("sinks", ctypes.c_void_p),
        ("mask", ctypes.c_void_p),
        ("mask_stride", I64 * 2),
    ]


def load(path):
    lib = ctypes.CDLL(path)
    lib.cudnn_fe_aot_sm120_sdpa_plan_create.argtypes = [ctypes.POINTER(Desc), ctypes.POINTER(Knobs), ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    lib.cudnn_fe_aot_sm120_sdpa_workspace_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(Args), ctypes.POINTER(ctypes.c_size_t)]
    lib.cudnn_fe_aot_sm120_sdpa_execute.argtypes = [ctypes.c_void_p, ctypes.POINTER(Args), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    lib.cudnn_fe_aot_sm120_sdpa_query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Args), ctypes.POINTER(Knobs), ctypes.POINTER(ctypes.c_char_p)]
    lib.cudnn_fe_aot_sm120_sdpa_plan_destroy.argtypes = [ctypes.c_void_p]
    lib.cudnn_fe_aot_sm120_last_error.restype = ctypes.c_char_p
    lib.cudnn_fe_aot_sm120_build_info.restype = ctypes.c_char_p
    return lib


def check(lib, st, what):
    if st:
        raise RuntimeError(f"{what}: status {st}: {lib.cudnn_fe_aot_sm120_last_error().decode()}")


def case(lib, dtype, b, hq, hkv, sq, skv, d, causal, sink=False, window_left=None, layout="bshd"):
    g = torch.Generator(device="cuda").manual_seed(sq * 131 + skv)
    if layout == "bshd":
        q = torch.randn(b, sq, hq, d, device="cuda", dtype=dtype, generator=g)
        k = torch.randn(b, skv, hkv, d, device="cuda", dtype=dtype, generator=g)
        v = torch.randn(b, skv, hkv, d, device="cuda", dtype=dtype, generator=g)
    else:  # Q in BHSD storage, K / V as a slice of a larger KV cache, viewed as (B, S, H, D)
        q = torch.randn(b, hq, sq, d, device="cuda", dtype=dtype, generator=g).transpose(1, 2)
        k = torch.randn(b, skv + 64, hkv, d, device="cuda", dtype=dtype, generator=g)[:, :skv]
        v = torch.randn(b, skv + 64, hkv, d, device="cuda", dtype=dtype, generator=g)[:, :skv]
    sinks = torch.randn(hq, device="cuda", dtype=torch.float32, generator=g) if sink else None
    scale = 1.0 / math.sqrt(d)
    o_aot, o_fe = torch.empty(b, sq, hq, d, device="cuda", dtype=dtype), torch.empty(b, sq, hq, d, device="cuda", dtype=dtype)

    dt = 1 if dtype == torch.bfloat16 else 0
    desc = Desc(ctypes.sizeof(Desc), d, dt, dt, dt, int(causal), -1 if window_left is None else window_left, int(sink), hq // hkv)
    plan = ctypes.c_void_p()
    check(lib, lib.cudnn_fe_aot_sm120_sdpa_plan_create(ctypes.byref(desc), None, -1, ctypes.byref(plan)), "plan_create")
    a = Args()
    a.struct_size = ctypes.sizeof(Args)
    a.b, a.h_q, a.h_kv, a.s_q, a.s_kv = b, hq, hkv, sq, skv
    a.q, a.k, a.v, a.o = q.data_ptr(), k.data_ptr(), v.data_ptr(), o_aot.data_ptr()
    for name, t in (("q_stride", q), ("k_stride", k), ("v_stride", v), ("o_stride", o_aot)):
        getattr(a, name)[:] = [t.stride(i) * t.element_size() for i in range(3)]
    a.scale = scale
    a.sinks = sinks.data_ptr() if sink else None
    kn, kname = Knobs(), ctypes.c_char_p()
    check(lib, lib.cudnn_fe_aot_sm120_sdpa_query(plan, ctypes.byref(a), ctypes.byref(kn), ctypes.byref(kname)), "query")
    ws_n = ctypes.c_size_t()
    check(lib, lib.cudnn_fe_aot_sm120_sdpa_workspace_size(plan, ctypes.byref(a), ctypes.byref(ws_n)), "workspace_size")
    ws = torch.empty(max(ws_n.value, 256), dtype=torch.uint8, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    check(lib, lib.cudnn_fe_aot_sm120_sdpa_execute(plan, ctypes.byref(a), ws.data_ptr(), ws_n.value, stream), "execute")
    lib.cudnn_fe_aot_sm120_sdpa_plan_destroy(plan)

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120

    t = lambda x: x.transpose(1, 2)  # noqa: E731  (B, S, H, D) storage -> logical (B, H, S, D)
    api = SdpaFwdDslSm120(
        t(q),
        t(k),
        t(v),
        t(o_fe),
        None,
        is_causal=causal,
        causal_bottom_right=causal,
        window_size_left=window_left,
        scale_softmax=scale,
        has_sink=sink,
        sched_policy=kn.sched_policy,
        tile_m=kn.tile_m,
        tile_n=kn.tile_n,
        split_kv=kn.split_kv,
        pack_gqa=bool(kn.pack_gqa),
    )
    api.check_support()
    api.compile()
    fe_ws = torch.empty(max(api.scratch_workspace_bytes(), 256), dtype=torch.uint8, device="cuda")
    api.execute(t(q), t(k), t(v), t(o_fe), sinks=sinks, workspace=fe_ws)
    torch.cuda.synchronize()
    same = torch.equal(o_aot, o_fe)
    diff = (o_aot.float() - o_fe.float()).abs().max().item()
    knobs = (kn.pack_gqa, kn.sched_policy, kn.split_kv, kn.tile_m, kn.tile_n)
    print(
        f"{'PASS' if same else 'FAIL'} {str(dtype)[6:]:8s} b={b} hq={hq} hkv={hkv} sq={sq:5d} skv={skv:5d} d={d} causal={int(causal)} "
        f"sink={int(sink)} wl={window_left} {layout}: knobs={knobs} bitwise={'yes' if same else 'no'} max|diff|={diff:.3e}",
        flush=True,
    )
    return same


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    args = ap.parse_args()
    lib = load(args.lib)
    print(f"{torch.cuda.get_device_name()}; {lib.cudnn_fe_aot_sm120_build_info().decode()}", flush=True)
    f16, bf16 = torch.float16, torch.bfloat16
    cases = [
        (f16, 1, 16, 8, 1000, 1000, 256, True),
        (bf16, 2, 8, 4, 333, 777, 256, True),
        (f16, 1, 32, 8, 257, 257, 128, True),
        (bf16, 2, 8, 2, 9, 3000, 128, True),
        (f16, 1, 32, 4, 3, 2048, 64, True),
        (f16, 1, 16, 8, 600, 600, 64, False),
        (f16, 1, 24, 4, 1, 8192, 128, True),  # G = 6: PackGQA with tile_q % G != 0
        (bf16, 2, 24, 4, 37, 1500, 256, True),
        (f16, 1, 32, 8, 1, 16384, 128, True),
        (f16, 4, 32, 8, 1, 4096, 128, False),
    ]
    results = [case(lib, *c) for c in cases]
    results.append(case(lib, f16, 1, 32, 8, 7, 3000, 128, True, layout="cache"))
    results.append(case(lib, bf16, 1, 64, 8, 300, 300, 64, True, sink=True))
    results.append(case(lib, f16, 1, 64, 8, 1, 6000, 64, True, sink=True))  # split + sink (PR #1731)
    results.append(case(lib, f16, 1, 64, 8, 300, 300, 64, True, sink=True, window_left=127))
    results.append(case(lib, bf16, 1, 64, 8, 1, 6000, 64, True, sink=True, window_left=127))
    print(f"SUMMARY: {sum(results)}/{len(results)} bitwise identical")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
