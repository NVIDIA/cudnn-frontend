# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ahead-of-time (AOT) export of the SM120 f16/bf16 prefill templates and the split combine.

The prepared pointer host (``prepared_host.host``) binds batch, sequence lengths and strides at run time but keeps the head counts
as plan constants. The AOT host below also takes ``h_q`` and ``h_kv`` at run time, so one exported object serves every head
count (PackGQA kernels still bake the ratio G = h_q / h_kv). Exports go through ``cute.compile(..., options=...)`` and
``export_to_c``: a plain C ABI with the cubin embedded, no tvm-ffi and no FE runtime at the call site. ``aot/`` at the repository
root builds ``libcudnn_frontend_aot_sm120.so`` from these objects; see ``docs/fe-oss-apis/aot_sm120_attention.md``.

The kernel object is the one the template's own ``compile()`` constructs (captured, not rebuilt), so an exported kernel is the
kernel the Python FROST engine runs for the same TemplateParams and knobs.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_driver

from cudnn.frost.template_loader import load_template
from cudnn.sdpa.fwd.kernels._common_blackwell import _bshd

EXPORT_OPTIONS = "--gpu-arch sm_120a"


@cute.jit
def aot_prefill_host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    sinks_ptr: Optional[cute.Pointer],
    problem_size: Tuple[int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    scale_softmax_log2: cutlass.Float32,
    kernel: cutlass.Constexpr,
    d_qk: cutlass.Constexpr[int],
    d_v: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    """``problem_size`` is (B, H_q, H_kv, S_q, S_kv). Under a split, O / LSE are the split-major compact partials (B * split)."""
    b, qh, kh, sq, skv = problem_size
    q = _bshd(q_ptr, b, sq, qh, d_qk, q_strides, False)
    k = _bshd(k_ptr, b, skv, kh, d_qk, k_strides, False)
    v = _bshd(v_ptr, b, skv, kh, d_v, v_strides, False)
    o = _bshd(o_ptr, b * kernel.split_kv, sq, qh, d_v, o_strides, False)
    lse = None
    if cutlass.const_expr(lse_ptr is not None):
        sq64 = cutlass.Int64(sq)
        lse = cute.make_tensor(lse_ptr, cute.make_layout((b * kernel.split_kv, qh, sq), stride=(cutlass.Int64(qh) * sq64, sq64, 1)))
    sinks = None
    if cutlass.const_expr(kernel.has_sink):
        sinks = cute.make_tensor(sinks_ptr, cute.make_layout((qh,), stride=(1,)))
    # seq_q_lens / seq_kv_lens are compiled out (dense, no padding); the kernel never reads them.
    lens_ptr = cute.make_ptr(cutlass.Int32, cutlass.Int64(0), cute.AddressSpace.gmem, assumed_align=4)
    q_lens = cute.make_tensor(lens_ptr, cute.make_layout((b,), stride=(1,)))
    kv_lens = cute.make_tensor(lens_ptr, cute.make_layout((b,), stride=(1,)))
    kernel(
        q,
        k,
        v,
        o,
        lse,
        sinks,
        q_lens,
        kv_lens,
        scale_softmax_log2,
        cutlass.Int32(0),
        None,
        None,
        None,
        cutlass.Int32(0),
        stream,
        prepared=True,
    )


def _template_path(d: int) -> str:
    name = "prefill_d256_f16.py" if d == 256 else "prefill_f16.py"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def build_prefill_kernel(params, d: int, gqa: int, tag: str):
    """The template's own kernel object for ``params`` (a config_sm120.TemplateParams), head dim ``d`` and PackGQA ratio ``gqa``
    (1 when ``params.pack_gqa`` is off), captured from the template's ``compile()`` instead of being re-constructed here."""
    mod = load_template(_template_path(d), params, tag=tag)
    captured = {}

    def capture(kernel, *args, **kwargs):
        captured["kernel"] = kernel
        return None

    real = mod._compile_prepared_host
    mod._compile_prepared_host = capture
    try:
        mod.compile.__wrapped__((12, 0), b=1, qh=gqa, kh=1, sq=1, skv=1, d_qk=d, d_v=d, has_lse=params.split_kv > 1, prepared=True)
    finally:
        mod._compile_prepared_host = real
    return mod, captured["kernel"]


def compile_prefill(params, d: int, gqa: int, kernel_name: str, options: str = EXPORT_OPTIONS):
    """Compile one prefill variant for AOT export; the device kernel symbol becomes ``kernel_name`` + the DSL's ``_0`` suffix."""
    mod, kernel = build_prefill_kernel(params, d, gqa, tag=kernel_name)
    dtype = mod.STORAGE_DTYPE

    def ptr(t, align=16):
        return cute.runtime.make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=align)

    strides = (cutlass.Int64(0),) * 3
    kernel_fn = type(kernel).kernel
    kernel_fn.set_name_prefix(kernel_name, remove_cutlass_symbol=True, keep_mangled_name=False)
    try:
        return cute.compile(
            aot_prefill_host,
            ptr(dtype),
            ptr(dtype),
            ptr(dtype),
            ptr(dtype),
            ptr(cutlass.Float32, 4) if params.split_kv > 1 else None,
            ptr(cutlass.Float32, 4) if params.has_sink else None,
            # Int32 leaves, not Python ints: a Python int is traced as a compile-time constant.
            (cutlass.Int32(0),) * 5,
            strides,
            strides,
            strides,
            strides,
            cutlass.Float32(0),
            kernel,
            d,
            d,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options=options,
        )
    finally:
        kernel_fn.set_name_prefix("cudnn", remove_cutlass_symbol=True)


def compile_combine(dtype: str, has_sink: bool, kernel_name: str, options: str = EXPORT_OPTIONS):
    """The shape-generic split combine (``split_combine.compile_ptr``'s dense half entry: f16/bf16 partials, no LSE output) for
    AOT export. B, H, S_q, D_v, the split count and the output strides are run-time values."""
    from cudnn.sdpa.fwd.kernels.sm100 import split_combine as sc

    elem = sc._ELEM[dtype]
    gmem = cute.AddressSpace.gmem

    def P(t):
        return cute.runtime.make_ptr(t, 16, gmem, assumed_align=t.width // 8)

    entry = sc._host_ptr_sink if has_sink else sc._host_ptr
    extra = (P(cutlass.Float32),) if has_sink else ()
    # The decode-grid kernel is traced behind a run-time-false branch (DECODE_GRID_OFF) and never launched; it is renamed so
    # that every symbol in the object carries the export's name.
    kernels = {"rows": sc._combine_kernel, "decode": sc._combine_decode_kernel, "unused": sc._combine_decode_grid_kernel}
    for suffix, k in kernels.items():
        k.set_name_prefix(f"{kernel_name}_{suffix}", remove_cutlass_symbol=True, keep_mangled_name=False)
    try:
        return cute.compile(
            entry,
            P(elem),
            P(cutlass.Float32),
            P(elem),
            None,
            (cutlass.Int32(0),) * 4,
            cutlass.Int32(0),
            (cutlass.Int64(0),) * 4,
            (cutlass.Int64(0),) * 3,
            *extra,
            False,
            sc.DECODE_GRID_OFF,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options=options,
        )
    finally:
        for k in kernels.values():
            k.set_name_prefix("cudnn", remove_cutlass_symbol=True)
