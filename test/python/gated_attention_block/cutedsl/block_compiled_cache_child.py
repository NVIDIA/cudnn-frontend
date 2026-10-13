# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The child process of ``test_block_compiled_cache.py``: compile and run EVERY block kernel compile site once (each arm a
separate compiled object), and print one JSON line ``{"sites": {<site>: {"stats": <compiled_cache.stats() over that site's
compile>, "digest": sha256(<every output byte>)}}}``.  The parent runs it twice under one ``CUDNN_FRONTEND_COMPILED_CACHE``
root: the first process compiles and exports every kernel, the second reloads every one of them and must write the same
bytes.  Inputs come from seeded generators, so the two processes see identical operands.  Standalone:

    CUDNN_FRONTEND_COMPILED_CACHE=<dir> python block_compiled_cache_child.py
"""

import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # gated_block_reference
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))  # test/python: sdpa.mxfp8_quant

import torch  # noqa: E402
import cudnn  # noqa: E402
from cudnn.frost import compiled_cache as cc  # noqa: E402

print("cudnn.__file__ =", cudnn.__file__, file=sys.stderr, flush=True)

FP8 = torch.float8_e4m3fn
BLOCK = 32  # the MXFP8 scale block


def _bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _digest(*tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(_bytes(t))
    return h.hexdigest()


def _gen(seed: int):
    return torch.Generator(device="cuda").manual_seed(seed)


def _randn(g, *shape, scale=1.0, dtype=torch.bfloat16):
    return (torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32) * scale).to(dtype)


def _tables(g, t: int, rope_dim: int):
    """``cos`` / ``sin`` ``[T, ROPE]`` bf16 with duplicated halves (what the kernels take)."""
    ang = torch.randn(t, rope_dim // 2, generator=g, device="cuda", dtype=torch.float32)
    emb = torch.cat((ang, ang), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def _stream():
    return torch.cuda.current_stream().cuda_stream


# ---- the sites ----------------------------------------------------------------------------------------------------------------


def elementwise(has_gate: bool):
    from cudnn.gated_attention_block.kernels.elementwise import compile_elementwise_gate, run_elementwise_gate

    t, h, d = 37, 8, 256
    g = _gen(1)
    src, gate = _randn(g, t, h, d), _randn(g, t, h, d, scale=3.0)
    dst = torch.zeros_like(src)
    r = compile_elementwise_gate(dtype=torch.bfloat16, h=h, d=d, has_gate=has_gate)
    run_elementwise_gate(r, src, gate if has_gate else None, dst, stream=_stream())
    return (dst,)


def qk_norm_rope_tma(arm: str):
    from cudnn.gated_attention_block.kernels.qk_norm_rope_tma import compile_qk_norm_rope_tma, run_qk_norm_rope_tma
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import sf_bytes

    t, h_q, h_kv, d, rope_dim = 37, 8, 2, 256, 64
    g = _gen(2)
    q, k, w_q, w_k = _randn(g, t, h_q, d), _randn(g, t, h_kv, d), _randn(g, d), _randn(g, d)
    cos, sin = _tables(g, t, rope_dim)
    kw = dict(dtype=torch.bfloat16, h_q=h_q, h_kv=h_kv, d=d, rope_dim=rope_dim, eps=1e-6, tile_rows=8)
    st = _stream()
    if arm == "norm":
        r = compile_qk_norm_rope_tma(want_rstd=True, **kw)
        q_out, k_out = torch.zeros_like(q), torch.zeros_like(k)
        rstd_q = torch.zeros((t, h_q), device="cuda", dtype=torch.float32)
        rstd_k = torch.zeros((t, h_kv), device="cuda", dtype=torch.float32)
        run_qk_norm_rope_tma(r, q, k, q_out, k_out, w_q, w_k, cos, sin, rstd_q, rstd_k, stream=st)
        return (q_out, k_out, rstd_q, rstd_k)
    if arm == "rope_only":
        r = compile_qk_norm_rope_tma(want_rstd=False, apply_norm=False, **kw)
        q_out, k_out = torch.zeros_like(q), torch.zeros_like(k)
        run_qk_norm_rope_tma(r, q, k, q_out, k_out, None, None, cos, sin, stream=st)
        return (q_out, k_out)
    if arm == "fp8":
        r = compile_qk_norm_rope_tma(want_rstd=False, fp8_out=True, **kw)
        q8 = torch.zeros((t, h_q, d), dtype=torch.uint8, device="cuda").view(FP8)
        k8 = torch.zeros((t, h_kv, d), dtype=torch.uint8, device="cuda").view(FP8)
        sq, sk = torch.tensor([7.0], device="cuda"), torch.tensor([0.25], device="cuda")
        run_qk_norm_rope_tma(r, q, k, None, None, w_q, w_k, cos, sin, stream=st, q8=q8, k8=k8, scale_q=sq, scale_k=sk)
        return (q8, k8)
    if arm == "mx":
        r = compile_qk_norm_rope_tma(want_rstd=False, mx_out=True, **kw)
        pay = lambda h: torch.zeros((t, h, d), dtype=torch.uint8, device="cuda").view(FP8)  # noqa: E731
        blob = lambda h: torch.zeros((sf_bytes(1, h, t, d),), dtype=torch.uint8, device="cuda")  # noqa: E731
        o = dict(q8=pay(h_q), sf_q=blob(h_q), q_T8=pay(h_q), sf_q_T=blob(h_q), k8=pay(h_kv), sf_k=blob(h_kv), k_T8=pay(h_kv), sf_k_T=blob(h_kv))
        run_qk_norm_rope_tma(r, q, k, None, None, w_q, w_k, cos, sin, stream=st, batch=1, seq_len=t, **o)
        return tuple(o[n] for n in ("q8", "sf_q", "q_T8", "sf_q_T", "k8", "sf_k", "k_T8", "sf_k_T"))
    raise ValueError(arm)


def _fork_params(**kw):
    from cudnn.gated_attention_block.kernels.proj_gemm import NormRopeFusionParams

    return NormRopeFusionParams(d_head=256, rope_dim=64, h_q=4, h_kv=2, eps=1e-6, **kw)


def fork_bf16():
    from cudnn.gated_attention_block.kernels.proj_gemm import build_fused_proj_gemm, run_fused_proj_gemm

    m, k = 256, 1024
    p = _fork_params()
    plan = build_fused_proj_gemm(p)
    g = _gen(3)
    h, w = _randn(g, m, k, scale=0.5), _randn(g, p.n_qkvg, k, scale=0.02)
    wq, wk = (1.0 + 0.1 * torch.randn(p.d_head, generator=g, device="cuda")).to(torch.bfloat16), (
        1.0 + 0.1 * torch.randn(p.d_head, generator=g, device="cuda")
    ).to(torch.bfloat16)
    cos, sin = _tables(g, m, p.rope_dim)
    out = torch.zeros((m, p.n_qkvg), device="cuda", dtype=torch.bfloat16)
    run_fused_proj_gemm(plan, h, w, out, wq, wk, cos, sin, None, None, stream=_stream())
    return (out,)


def fork_fp8():
    from cudnn.gated_attention_block.kernels.proj_gemm import build_fused_proj_gemm, run_fused_proj_gemm_fp8

    m, k = 256, 1024
    p = _fork_params(quant_fp8=True)
    plan = build_fused_proj_gemm(p)
    g = _gen(4)
    a8 = _randn(g, m, k, scale=0.5, dtype=torch.float32).clamp(-448.0, 448.0).to(FP8)
    w8 = _randn(g, p.n_qkvg, k, scale=0.02, dtype=torch.float32).clamp(-448.0, 448.0).to(FP8)
    wq, wk = (1.0 + 0.1 * torch.randn(p.d_head, generator=g, device="cuda")).to(torch.bfloat16), (
        1.0 + 0.1 * torch.randn(p.d_head, generator=g, device="cuda")
    ).to(torch.bfloat16)
    cos, sin = _tables(g, m, p.rope_dim)
    e4m3 = lambda width: torch.zeros((m, width), dtype=torch.uint8, device="cuda").view(FP8)  # noqa: E731
    q8, k8, v8 = e4m3(p.n_q), e4m3(p.n_kv), e4m3(p.n_kv)
    gate16 = torch.zeros((m, p.n_gate), dtype=torch.bfloat16, device="cuda")
    qscal = torch.tensor([1.0, 8.0, 4.0, 2.0], dtype=torch.float32, device="cuda")  # alpha, scale_q, scale_k, scale_v
    run_fused_proj_gemm_fp8(plan, a8, w8, q8, k8, v8, gate16, wq, wk, cos, sin, qscal, stream=_stream())
    return (q8, k8, v8, gate16)


def _blocked_sf(e: torch.Tensor, mq) -> torch.Tensor:
    """The padded F8_128x4 blob of a logical ``[rows, K/32]`` uint8 scale matrix (whole 128-row atoms, whole 4-block words)."""
    rows, cols = e.shape
    rows_pad, cols_pad = -(-rows // 128) * 128, -(-cols // 4) * 4
    pad = torch.zeros(rows_pad, cols_pad, dtype=torch.uint8, device=e.device)
    pad[:rows, :cols] = e
    return mq.swizzle_sf_rowwise(pad).flatten()


def fork_mxfp8():
    from cudnn.gated_attention_block.kernels.proj_gemm import build_fused_proj_gemm, run_fused_proj_gemm_mxfp8
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import sf_bytes
    from sdpa import mxfp8_quant as mq

    m, k = 256, 1024
    p = _fork_params(quant_mxfp8=True)
    plan = build_fused_proj_gemm(p)
    g = _gen(5)
    a32 = torch.randn(m, k, generator=g, device="cuda") * 0.5
    w32 = torch.randn(p.n_qkvg, k, generator=g, device="cuda") * 0.02
    a8, e_a = mq.quantize_blocks(a32.reshape(m, k // BLOCK, BLOCK), FP8)
    w8, e_w = mq.quantize_blocks(w32.reshape(p.n_qkvg, k // BLOCK, BLOCK), FP8)
    a8, w8 = a8.reshape(m, k).contiguous(), w8.reshape(p.n_qkvg, k).contiguous()
    wq, wk = (1.0 + 0.1 * torch.randn(p.d_head, generator=g, device="cuda")).to(torch.bfloat16), (
        1.0 + 0.1 * torch.randn(p.d_head, generator=g, device="cuda")
    ).to(torch.bfloat16)
    cos, sin = _tables(g, m, p.rope_dim)
    e4m3 = lambda width: torch.zeros((m, width), dtype=torch.uint8, device="cuda").view(FP8)  # noqa: E731
    q8, k8, v8 = e4m3(p.n_q), e4m3(p.n_kv), e4m3(p.n_kv)
    gate16 = torch.zeros((m, p.n_gate), dtype=torch.bfloat16, device="cuda")
    sf = lambda h: torch.zeros((sf_bytes(1, h, m, p.d_head),), dtype=torch.uint8, device="cuda")  # noqa: E731
    sf_q, sf_k, sf_v = sf(p.h_q), sf(p.h_kv), sf(p.h_kv)
    run_fused_proj_gemm_mxfp8(
        plan, a8, _blocked_sf(e_a, mq), w8, _blocked_sf(e_w, mq), q8, k8, v8, gate16, sf_q, sf_k, sf_v, wq, wk, cos, sin, batch=1, seq_len=m, stream=_stream()
    )
    return (q8, k8, v8, gate16, sf_q, sf_k, sf_v)


def cache_write(arm: str):
    """The serving write-through, shaped like the block's two artifacts: ``kv`` = the K / V launch (two operands, the KV heads,
    int32 slots), ``index`` = the indexer band's raw-key launch (one operand, one 128-wide head, int64 slots).  Padded (-1) tokens
    leave their rows at the NaN sentinel; the pools are the output (the sentinel bit pattern is fixed, so the digest is too)."""
    from cudnn.gated_attention_block.kernels.cache_write import compile_cache_write, run_cache_write

    if arm == "kv":
        h, d, two, slot_dtype = 2, 256, True, torch.int32
    elif arm == "index":
        h, d, two, slot_dtype = 1, 128, False, torch.int64
    else:
        raise ValueError(arm)
    t, page_size, n_pages = 100, 16, 12
    g = _gen(6)
    src_a = _randn(g, t, h, d)
    src_b = _randn(g, t, h, d) if two else None
    # an injective slot mapping drawn on the CPU (deterministic for the seed on any device), every seventh token padded
    slot = torch.randperm(n_pages * page_size, generator=torch.Generator().manual_seed(6))[:t].to(slot_dtype)
    slot[::7] = -1
    slot = slot.cuda()
    pool = lambda: torch.full((n_pages, h, page_size, d), float("nan"), device="cuda", dtype=torch.bfloat16)  # noqa: E731
    pool_a, pool_b = pool(), (pool() if two else None)
    r = compile_cache_write(dtype=torch.bfloat16, h=h, d=d, page_size=page_size, slot_dtype=slot_dtype, two_operands=two)
    run_cache_write(r, src_a, src_b, pool_a, pool_b, slot, stream=_stream())
    return (pool_a, pool_b) if two else (pool_a,)


SITES = {
    "elementwise_gate": lambda: elementwise(True),
    "elementwise_copy": lambda: elementwise(False),
    "qk_tma_norm": lambda: qk_norm_rope_tma("norm"),
    "qk_tma_rope_only": lambda: qk_norm_rope_tma("rope_only"),
    "qk_tma_fp8": lambda: qk_norm_rope_tma("fp8"),
    "qk_tma_mx": lambda: qk_norm_rope_tma("mx"),
    "fork_bf16": fork_bf16,
    "fork_fp8": fork_fp8,
    "fork_mxfp8": fork_mxfp8,
}
# the serving cache-write kernel is a separate change: its two arms exist exactly where the module is in the tree (the parent test
# guards its expectations the same way), so a run with no site names never asks a tree for a kernel it does not carry
if importlib.util.find_spec("cudnn.gated_attention_block.kernels.cache_write") is not None:
    SITES["cache_write_kv"] = lambda: cache_write("kv")
    SITES["cache_write_index"] = lambda: cache_write("index")


def main() -> int:
    only = sys.argv[1:] or list(SITES)
    report = {}
    for name in only:
        cc.reset_stats()
        outs = SITES[name]()
        torch.cuda.synchronize()
        report[name] = {"stats": cc.stats(), "digest": _digest(*outs)}
        print(f"site {name}: {report[name]}", file=sys.stderr, flush=True)
    print(json.dumps({"sites": report, "cache_dir": str(cc.get_cache_dir())}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
