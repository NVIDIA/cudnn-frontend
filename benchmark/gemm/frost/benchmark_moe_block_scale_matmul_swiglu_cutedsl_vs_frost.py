# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Head-to-head: cuteDSL grouped-GEMM GLU vs FROST MoE grouped block-scale
dual-matmul + SwiGLU, on the same problem, same timing methodology and output
contract.

cuteDSL runs ONE grouped GEMM of width 2N whose gate/up halves are 32-column
interleaved (cudnn.GroupedGemmGluSm100; on an SM 10.7 part this picks the Rubin
kernel).  FROST runs TWO parallel grouped matmuls of width N sharing the token
operand, fused by a pointwise SwiGLU epilogue.  Same FLOPs, different
decomposition.

The block-scaled GLU backend always materializes the pre-activation C
(valid_m x 2N) alongside D -- generate_c is dropped on that path -- so the FROST
graph taps both grouped-matmul outputs and writes the same bytes.  The per-side
byte accounting is printed below the header.

The FROST graph mirrors the cuteDSL epilogue op for op: the per-expert alpha
multiplies BOTH accumulators before the activation and the taps carry the
alpha-scaled pre-activation, exactly as the Rubin kernel does (alpha -> store_c
-> swiglu_act).  So both sides compute silu(alpha*gate) * (alpha*up) from a
per-group alpha of the same shape, not two differently-parenthesised functions.

``--output-mode fp8-rowcol`` (MXFP8 only) additionally gives both sides the
same dual-quant output contract: one row-wise and one column-wise FP8 E4M3 D,
each with a block-32 E8M0 scale in F8_128x4 layout.  cuteDSL calls those
D/D_col and SFD_row/SFD_col; FROST expresses the same work as two terminal
block_scale_quantize nodes.  ``norm_const=1`` makes the scale conventions
identical.

    python benchmark/gemm/frost/benchmark_moe_block_scale_matmul_swiglu_cutedsl_vs_frost.py \
        --shape 12,1024,3072,7168 --combo mxfp8 --output-mode fp8-rowcol
"""

from __future__ import annotations

import argparse

import sys
import time

import torch

import benchmark_moe_block_scale_matmul_swiglu as fb
import benchmark_utils as bu

# combo -> (sf_vec_size, ab dtype, sf dtype, d dtype)
_COMBOS = {
    "nvfp4": (16, torch.float4_e2m1fn_x2, torch.float8_e4m3fn, torch.bfloat16),
    "mxfp4": (32, torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, torch.bfloat16),
    "mxfp8": (32, torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.bfloat16),
}

_DEV = "cuda"
_OUTPUT_MODES = ("bf16", "fp8-rowcol")
_QUANT_BLOCK = 32


def _ceil_div(a, b):
    return (a + b - 1) // b


def _permuted(l, mode0, mode1, dtype, *, mode0_major=False):
    shape = (l, mode1, mode0) if mode0_major else (l, mode0, mode1)
    order = (2, 1, 0) if mode0_major else (1, 2, 0)
    if dtype is torch.float4_e2m1fn_x2:
        packed = torch.randint(0, 256, (*shape[:-1], shape[-1] // 2), dtype=torch.uint8, device=_DEV)
        return packed.view(dtype).permute(order)
    return torch.empty(shape, dtype=torch.float32, device=_DEV).uniform_(-2, 2).permute(order).to(dtype)


def _make_sf(l, mn, k, sf_vec_size, dtype):
    sf_k = _ceil_div(k, sf_vec_size)
    shape = (l, _ceil_div(mn, 128), _ceil_div(sf_k, 4), 32, 4, 4)
    buf = torch.empty(shape, dtype=torch.float32, device=_DEV).uniform_(1, 3).to(torch.int8).to(torch.float32)
    return buf.permute(3, 4, 1, 5, 2, 0).to(dtype)


def _cutedsl_set(E, M, N, K, combo, output_mode="bf16", want_amax=False):
    sf_vec, ab_dt, sf_dt, default_d_dt = _COMBOS[combo]
    d_dt = torch.float8_e4m3fn if output_mode == "fp8-rowcol" else default_d_dt
    n = 2 * N
    valid_m = E * M
    offsets = torch.tensor([(i + 1) * M for i in range(E)], dtype=torch.int32, device=_DEV)
    s = {
        "a": _permuted(1, valid_m, K, ab_dt),
        "b": _permuted(E, n, K, ab_dt),
        "sfa": _make_sf(1, valid_m, K, sf_vec, sf_dt),
        "sfb": _make_sf(E, n, K, sf_vec, sf_dt),
        "offsets": offsets,
        "alpha": torch.ones(E, dtype=torch.float32, device=_DEV),
        "c": _permuted(1, valid_m, n, torch.bfloat16),
        "d": _permuted(1, valid_m, N, d_dt),
        "d_col": _permuted(1, valid_m, N, d_dt),
        "amax": None,
        "sfd_row": None,
        "sfd_col": None,
        "norm_const": None,
    }
    if output_mode == "fp8-rowcol":
        # GroupedGemmGluSm100's two D tensors are both physically N-major.  The
        # row/col names describe the reduction axis used to make their scales,
        # not their GMEM layout.  These are precisely the two F8_128x4 layouts
        # used by FROST's row- and M-axis block_scale_quantize outputs below.
        s["sfd_row"] = _make_sf(1, valid_m, N, _QUANT_BLOCK, sf_dt)
        s["sfd_col"] = _make_sf(1, N, valid_m, _QUANT_BLOCK, sf_dt)
        s["norm_const"] = torch.ones(1, dtype=torch.float32, device=_DEV)
    if d_dt in (torch.bfloat16, torch.float16) and want_amax:
        s["amax"] = torch.full((E, 1), float("-inf"), dtype=torch.float32, device=_DEV)
    return s


def _cutedsl_bytes(E, M, N, K, combo, output_mode):
    s = _cutedsl_set(E, M, N, K, combo, output_mode)
    total = sum(t.untyped_storage().nbytes() for t in s.values() if torch.is_tensor(t))
    del s
    torch.cuda.empty_cache()
    return total


def _cutedsl_pool(E, M, N, K, combo, output_mode, nbuf):
    return [_cutedsl_set(E, M, N, K, combo, output_mode) for _ in range(max(1, nbuf))]


def _build_cutedsl(s, combo, mma_tiler, cluster, dynamic_sched, vector_f32):
    from cudnn import GroupedGemmGluSm100

    sf_vec, _ab, _sf, _d = _COMBOS[combo]
    api = GroupedGemmGluSm100(
        sample_a=s["a"],
        sample_c=s["c"],
        sample_d=s["d"],
        sample_sfa=s["sfa"],
        sample_padded_offsets=s["offsets"],
        sample_alpha=s["alpha"],
        sample_d_col=s["d_col"],
        sample_b=s["b"],
        sample_sfb=s["sfb"],
        sample_sfd_row=s["sfd_row"],
        sample_sfd_col=s["sfd_col"],
        sample_amax=s["amax"],
        sample_norm_const=s["norm_const"],
        sample_prob=None,
        acc_dtype=torch.float32,
        mma_tiler_mn=mma_tiler,
        cluster_shape_mn=cluster,
        sf_vec_size=sf_vec,
        m_aligned=256,
        vector_f32=vector_f32,
        act_func="swiglu",
        use_dynamic_sched=dynamic_sched,
    )
    if not api.check_support():
        raise ValueError("check_support() returned False")
    api.compile()
    return api


def _cutedsl_launch(api, s, stream):
    api.execute(
        a_tensor=s["a"],
        c_tensor=s["c"],
        d_tensor=s["d"],
        sfa_tensor=s["sfa"],
        padded_offsets=s["offsets"],
        alpha_tensor=s["alpha"],
        b_tensor=s["b"],
        sfb_tensor=s["sfb"],
        d_col_tensor=s["d_col"],
        sfd_row_tensor=s["sfd_row"],
        sfd_col_tensor=s["sfd_col"],
        amax_tensor=s["amax"],
        norm_const_tensor=s["norm_const"],
        current_stream=stream,
    )


def _frost_graph(S, N, K, E, combo, output_mode="bf16", alignment=1):
    import cudnn
    from types import SimpleNamespace

    block_size, a_dt, sf_dt = fb._COMBOS[combo]
    sf_k = K // block_size
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    tok = g.tensor(name="token", dim=[1, S, K], stride=[S * K, K, 1], data_type=a_dt)
    w0 = g.tensor(name="weight0", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
    w1 = g.tensor(name="weight1", dim=[E, K, N], stride=[K * N, 1, K], data_type=a_dt)
    SFA = g.tensor(
        name="SFA",
        dim=[1, S, sf_k],
        stride=[S * sf_k, sf_k, 1],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB0 = g.tensor(
        name="SFB0",
        dim=[E, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    SFB1 = g.tensor(
        name="SFB1",
        dim=[E, sf_k, N],
        stride=[sf_k * N, 1, sf_k],
        data_type=sf_dt,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    fto = g.tensor(
        name="first_token_offset",
        dim=[E, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
        alignment_value=alignment,
    )
    alpha = g.tensor(name="alpha", dim=[E, 1, 1], stride=[1, 1, 1], data_type=cudnn.data_type.FLOAT)
    tok_d = g.block_scale_dequantize(input=tok, descale=SFA, block_size=[1, block_size])
    w0_d = g.block_scale_dequantize(input=w0, descale=SFB0, block_size=[block_size, 1])
    w1_d = g.block_scale_dequantize(input=w1, descale=SFB1, block_size=[block_size, 1])
    c0 = g.moe_grouped_matmul(tok_d, w0_d, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe0")
    c1 = g.moe_grouped_matmul(tok_d, w1_d, fto, mode=cudnn.moe_grouped_matmul_mode.NONE, compute_data_type=cudnn.data_type.FLOAT, name="moe1")
    a0 = g.mul(a=c0, b=alpha, name="alpha0")
    a1 = g.mul(a=c1, b=alpha, name="alpha1")
    s0 = g.swish(input=a0, name="silu0")
    dq = g.mul(a=s0, b=a1, name="mul0")
    outputs = []
    if output_mode == "fp8-rowcol":
        qrow, sfrow = g.block_scale_quantize(input=dq, block_size=_QUANT_BLOCK, axis=-1, name="qrow")
        qrow.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
        sfrow_n = _ceil_div(N // _QUANT_BLOCK, 4) * 4
        sfrow.set_dim([1, _ceil_div(S, 128) * 128, sfrow_n]).set_stride([(_ceil_div(S, 128) * 128) * sfrow_n, sfrow_n, 1])
        sfrow.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
        sfrow.set_reordering_type(cudnn.tensor_reordering.F8_128x4)

        qcol, sfcol = g.block_scale_quantize(
            input=dq,
            block_size=_QUANT_BLOCK,
            axis=1,
            group_offset=fto,
            name="qcol",
        )
        qcol.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)
        sfcol_m = _ceil_div(S // _QUANT_BLOCK, 4) * 4
        sfcol_n = _ceil_div(N, 128) * 128
        sfcol.set_dim([1, sfcol_n, sfcol_m]).set_stride([sfcol_n * sfcol_m, sfcol_m, 1])
        sfcol.set_output(True).set_data_type(cudnn.data_type.FP8_E8M0)
        sfcol.set_reordering_type(cudnn.tensor_reordering.F8_128x4)
        outputs.extend([qrow, qcol])
    else:
        dq.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        outputs.append(dq)
    for src_c, nm in ((a0, "tapC0"), (a1, "tapC1")):
        t = g.identity(input=src_c, name=nm)
        t.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        outputs.append(t)
    if output_mode == "fp8-rowcol":
        # Binding order is dense outputs in set_output order, then quant scales.
        outputs.extend([sfrow, sfcol])
    return g, SimpleNamespace(
        first_token_offset=fto,
        a_operands=[tok],
        b_operands=[w0, w1],
        sfa_operands=[SFA],
        sfb_operands=[SFB0, SFB1],
        outputs=outputs,
        aux=[alpha],
    )


def _frost_spec_map(combo, output_mode, alignment=1):
    from cudnn.gemm.frost.graph_analyzer import analyze
    from cudnn.gemm.frost.kernel_registry import candidates as registry_candidates

    chain = analyze(_frost_graph(1024, 256, 512, 2, combo, output_mode, alignment)[0])
    m = {}
    for t, cfg in registry_candidates(chain):
        if cfg.pipeline != "sm100" or cfg.mma_tile_m != 128:
            continue
        m[cfg.name] = (cfg, cfg.cta_group)
    return m


def _frost_set(S, N, K, E, combo, output_mode):
    base = fb._mkdata(S, N, K, E, combo)
    alpha = torch.ones(E, 1, 1, dtype=torch.float32, device=_DEV)
    taps = [torch.empty(1, S, N, dtype=torch.bfloat16, device=_DEV) for _ in range(2)]
    if output_mode == "fp8-rowcol":
        qrow = torch.empty(1, S, N, dtype=torch.float8_e4m3fn, device=_DEV)
        qcol = torch.empty_like(qrow)
        sfrow = torch.empty(
            1,
            _ceil_div(S, 128) * 128,
            _ceil_div(N // _QUANT_BLOCK, 4) * 4,
            dtype=torch.float8_e8m0fnu,
            device=_DEV,
        )
        sfcol = torch.empty(
            1,
            _ceil_div(N, 128) * 128,
            _ceil_div(S // _QUANT_BLOCK, 4) * 4,
            dtype=torch.float8_e8m0fnu,
            device=_DEV,
        )
        outputs = [qrow, qcol, *taps, sfrow, sfcol]
    else:
        d = torch.empty(1, S, N, dtype=torch.bfloat16, device=_DEV)
        outputs = [d, *taps]
    # Drop benchmark_moe_block_scale_matmul_swiglu's own D/scalar slots: this
    # comparison supplies its own output contract and per-expert alpha.
    return (base[:6], alpha, outputs)


def _frost_launch(plan, h, item, offsets, stream):
    inputs, alpha, outputs = item
    plan(fb._vp_moe_bs_mg(h, fb._gemm_pairs(inputs), offsets, outputs, alpha), stream=stream)


def _cutedsl_all_tiles():
    """Every (mma_tiler, cluster) the block-scaled GLU backend accepts: N is
    pinned to 256, use_2cta_instrs is derived from tiler M, and
    cluster_tiler_m = cluster_m // (2 if 2cta) * tiler_m must be 128 or 256."""
    out = []
    for tm in (128, 256):
        two_cta = tm == 256
        for cm in (1, 2, 4):
            if two_cta and cm % 2:
                continue
            if (cm // (2 if two_cta else 1)) * tm not in (128, 256):
                continue
            for cn in (1, 2, 4):
                if cm * cn <= 16:
                    out.append(((tm, 256), (cm, cn)))
    return out


def _parse_tiles(spec):
    if spec.strip().lower() == "all":
        return _cutedsl_all_tiles()
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        tile, _, clus = item.partition(":")
        tm, _, tn = tile.partition("x")
        mma = (int(tm), int(tn))
        if clus:
            cm, _, cn = clus.partition("x")
            out.append((mma, (int(cm), int(cn))))
        else:
            out.append((mma, None))
    return out


def _fmt_cutedsl_label(mma, cluster, dyn, vecf32):
    c = "auto" if cluster is None else f"{cluster[0]}x{cluster[1]}"
    return f"cutedsl {mma[0]}x{mma[1]} cluster{c}{' dynsched' if dyn else ''}{' vecf32' if vecf32 else ''}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="12,4096,3072,7168", help="G,M,N,K — G groups x M tokens, N = SwiGLU OUTPUT width (gemm width is 2N)")
    p.add_argument("--combo", choices=tuple(_COMBOS), default="nvfp4")
    p.add_argument(
        "--output-mode",
        choices=_OUTPUT_MODES,
        default="bf16",
        help="shared output contract: bf16, or dual row/col block-32 FP8+E8M0 quantization (MXFP8 only)",
    )
    p.add_argument("--sides", choices=("both", "cutedsl", "frost"), default="both")
    p.add_argument("--cutedsl-tiles", default="all", help="'all' (every legal tiler x cluster) or an MxN[:cmxcn] list")
    p.add_argument("--dynamic-sched", choices=("off", "on", "both"), default="both")
    p.add_argument("--vector-f32", choices=("off", "on", "both"), default="both", help="cuteDSL packed f32x2 epilogue math")
    p.add_argument("--frost-store", choices=("tma", "stg", "both"), default="both", help="FROST epilogue store mode to sweep (default: both)")
    p.add_argument(
        "--fto-alignment",
        type=int,
        default=1,
        help="first_token_offset `alignment_value`: promise every routed-group start is a "
        "multiple of it. Lets FROST address A / SFA / D through the ORIGINAL TMA descriptors "
        "instead of rewriting them per group, for every config whose cluster tile M (and, "
        "block-scale, 128) divides it. 1 (default) = no promise. The bench checks its own "
        "offsets against it; the kernel does not.",
    )
    bu.add_sweep_args(p, nsys=False)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA, skipping.")
        return 1

    parts = [int(x) for x in args.shape.split(",")]
    if len(parts) != 4:
        sys.exit("--shape must be G,M,N,K")
    E, M, N, K = parts
    if M % 256:
        sys.exit(f"M={M} must be a multiple of 256 (cuteDSL pads each group to m_aligned=256)")
    if args.output_mode == "fp8-rowcol" and args.combo != "mxfp8":
        sys.exit("--output-mode fp8-rowcol requires --combo mxfp8: GroupedGemmGluSm100 rejects low-precision D for FP4 A/B")
    if args.output_mode == "fp8-rowcol" and (N % _QUANT_BLOCK or (E * M) % _QUANT_BLOCK):
        sys.exit(f"--output-mode fp8-rowcol requires N and S=E*M divisible by {_QUANT_BLOCK}")
    S = E * M
    n_gemm = 2 * N

    from cuda.bindings import driver as cuda_drv

    # One stream for both sides and for the timing events: cuteDSL takes a
    # CUstream, FROST's _as_custream takes the raw handle. Leaving FROST on its
    # stream=None default hard-wires it to the legacy default stream, which only
    # matches torch's current stream by coincidence.
    stream_handle = torch.cuda.current_stream().cuda_stream
    stream = cuda_drv.CUstream(stream_handle)

    flops = 2 * S * n_gemm * K
    major, minor = torch.cuda.get_device_capability()
    print(f"\n=== MoE grouped SwiGLU ({args.combo}, {args.output_mode})  " f"E={E} x M={M} (S={S})  N_out={N} (gemm N={n_gemm})  K={K} ===")
    print(f"  {torch.cuda.get_device_name(0)}  sm_{major}{minor}   ~{flops / 1e9:.1f} GFLOP")
    print(f"  [timing: {args.timing}, warmup={args.warmup}, iters={args.iters}]")
    if args.fto_alignment > 1:
        print(f"  [fto alignment_value: {args.fto_alignment} — FROST may take the global-descriptor path]")

    elem = 0.5 if _COMBOS[args.combo][1] is torch.float4_e2m1fn_x2 else 1.0
    read_bytes = int(S * K * elem + E * n_gemm * K * elem)
    if args.output_mode == "fp8-rowcol":
        d_bytes = 2 * S * N
        sf_bytes = (_ceil_div(S, 128) * 128) * (_ceil_div(N // _QUANT_BLOCK, 4) * 4)
        sf_bytes += (_ceil_div(N, 128) * 128) * (_ceil_div(S // _QUANT_BLOCK, 4) * 4)
    else:
        d_bytes = S * N * 2
        sf_bytes = 0
    c_bytes = S * n_gemm * 2
    print(
        f"  compulsory reads ~{read_bytes / 1e6:.0f} MB   "
        f"D write ~{d_bytes / 1e6:.0f} MB" + (f"   quant-SF write ~{sf_bytes / 1e6:.1f} MB" if sf_bytes else "")
    )
    print(f"  both sides also write the pre-activation C ~{c_bytes / 1e6:.0f} MB " f"(cuteDSL: one {n_gemm}-wide buffer; FROST: two {N}-wide taps)")
    print()

    results = []

    if args.sides in ("both", "cutedsl"):
        per_set = _cutedsl_bytes(E, M, N, K, args.combo, args.output_mode)
        nbuf = bu.resolve_nbuf(args.rotate_buffers, per_set)
        print(f"  -- cuteDSL GroupedGemmGluSm100 --")
        bu.report_pool(nbuf, per_set)
        wset = _cutedsl_set(E, M, N, K, args.combo, args.output_mode)
        pool = _cutedsl_pool(E, M, N, K, args.combo, args.output_mode, nbuf)
        modes = {"off": [False], "on": [True], "both": [False, True]}
        dyn_modes = modes[args.dynamic_sched]
        vec_modes = modes[args.vector_f32]
        sweep = [(mma, cluster, dyn, vecf32) for mma, cluster in _parse_tiles(args.cutedsl_tiles) for dyn in dyn_modes for vecf32 in vec_modes]
        for mma, cluster, dyn, vecf32 in sweep:
            label = _fmt_cutedsl_label(mma, cluster, dyn, vecf32)
            if args.stream:
                print(f"  > {label} ...", flush=True)
            t0 = time.time()
            try:
                api = _build_cutedsl(wset, args.combo, mma, cluster, dyn, vecf32)
            except (ValueError, NotImplementedError, RuntimeError) as e:
                print(f"  {label:52s} UNSUPPORTED: {type(e).__name__}: {str(e)[:60]}")
                continue
            try:
                _cutedsl_launch(api, wset, stream)
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(f"  {label:52s} LAUNCH FAIL: {type(e).__name__}: {str(e)[:60]}")
                continue
            jit = time.time() - t0
            ms = bu.time_ms(
                bu.rotating(lambda s, _a=api: _cutedsl_launch(_a, s, stream), pool),
                lambda _a=api: _cutedsl_launch(_a, wset, stream),
                warmup=args.warmup,
                iters=args.iters,
                timing=args.timing,
            )
            tf = flops / (ms * 1e-3) / 1e12
            results.append((label, ms, tf))
            print(f"  {label:52s} {tf:8.2f} TFLOP/s  {ms:8.3f} ms   [jit {jit:5.1f}s]", flush=True)
        del wset, pool
        torch.cuda.empty_cache()
        print()

    if args.sides in ("both", "frost"):
        print("  -- FROST dual grouped block-scale matmul + SwiGLU + matched outputs + C taps --")
        probe = _frost_set(S, N, K, E, args.combo, args.output_mode)
        per_set = bu.set_bytes(probe[0]) + bu.set_bytes(probe[1:2]) + bu.set_bytes(probe[2])
        del probe
        torch.cuda.empty_cache()
        nbuf = bu.resolve_nbuf(args.rotate_buffers, per_set)
        bu.report_pool(nbuf, per_set)
        from cudnn.gemm.frost import compiler as C
        from cudnn.gemm.frost.compiler import jit_from_cudnn_graph

        store_modes = {"tma": ["tma"], "stg": ["stg"], "both": ["tma", "stg"]}[args.frost_store]
        spec_map = _frost_spec_map(args.combo, args.output_mode, args.fto_alignment)
        offsets = bu.group_offsets(S, E)
        bad = [int(o) for o in offsets.tolist() if int(o) % args.fto_alignment]
        if bad:
            sys.exit(
                f"--fto-alignment {args.fto_alignment} is not true of the offsets this bench builds "
                f"({bad[:4]}...): the promise is unchecked at runtime and would miscompute."
            )
        wset = _frost_set(S, N, K, E, args.combo, args.output_mode)
        pool = [wset] + [_frost_set(S, N, K, E, args.combo, args.output_mode) for _ in range(max(0, nbuf - 1))]
        for cname in bu.select_configs(args.configs, spec_map):
            spec = bu.spec_for(cname, spec_map)
            if spec is None:
                print(f"  {cname:52s} UNKNOWN config")
                continue
            cfg, cta_group = spec
            for store in store_modes:
                label = f"frost[{store}] {cname}"
                if args.stream:
                    print(f"  > {label} ...", flush=True)
                t0 = time.time()
                try:
                    g, h = _frost_graph(S, N, K, E, args.combo, args.output_mode, args.fto_alignment)
                    with C.force_stg_epi(store == "stg"):
                        plan = jit_from_cudnn_graph(g, config=cfg)
                except (NotImplementedError, ValueError) as e:
                    print(f"  {label:52s} UNSUPPORTED: {type(e).__name__}: {str(e)[:60]}")
                    continue
                try:
                    _frost_launch(plan, h, wset, offsets, stream_handle)
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    print(f"  {label:52s} LAUNCH FAIL: {type(e).__name__}: {str(e)[:60]}")
                    continue
                jit = time.time() - t0
                ms = bu.time_ms(
                    bu.rotating(lambda s, _p=plan, _h=h: _frost_launch(_p, _h, s, offsets, stream_handle), pool),
                    lambda _p=plan, _h=h: _frost_launch(_p, _h, wset, offsets, stream_handle),
                    warmup=args.warmup,
                    iters=args.iters,
                    timing=args.timing,
                )
                tf = flops / (ms * 1e-3) / 1e12
                results.append((label, ms, tf))
                print(f"  {label:52s} {tf:8.2f} TFLOP/s  {ms:8.3f} ms   [jit {jit:5.1f}s]", flush=True)
        del wset, pool
        torch.cuda.empty_cache()
        print()

    if not results:
        print("  nothing ran.")
        return 1

    best_cute = min((r for r in results if r[0].startswith("cutedsl")), key=lambda r: r[1], default=None)
    best_frost = min((r for r in results if r[0].startswith("frost")), key=lambda r: r[1], default=None)
    print("  === summary ===")
    for tag, r in (("cuteDSL", best_cute), ("FROST", best_frost)):
        if r is None:
            print(f"  best {tag:8s} —")
        else:
            print(f"  best {tag:8s} {r[2]:8.2f} TFLOP/s  {r[1]:8.3f} ms   {r[0]}")
    if best_cute and best_frost:
        print(f"\n  FROST / cuteDSL = {best_cute[1] / best_frost[1]:.3f}x " f"({'FROST faster' if best_frost[1] < best_cute[1] else 'cuteDSL faster'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
