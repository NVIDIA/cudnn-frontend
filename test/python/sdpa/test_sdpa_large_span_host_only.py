# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no kernel launch) pins for operands that span more than 2**31 elements.

Rule S7 (``python/cudnn/sdpa/AGENTS.md``): promote the index to Int64 BEFORE the stride multiply. Each test here
lowers the kernel for a declared geometry past the limit and inspects what was generated, so it runs on any lane
(no SM120 / SM100 part needed):

* SM120 backward, compact path: ``dq_accum`` / dQ / dK-dV-workspace addressing. A wrapped Int32 index shows up as a
  ``mul.wide.s32`` on the address path; the widened form is Int64 arithmetic.
* Triton QAT kernels: the batch*head offset must be a 64-bit product.
* FlexAttention varlen / Sage FP8 BSA: the host declines instead of letting an Int32 offset wrap.
"""

import glob
import os
import re
import subprocess
import sys
import tempfile
import textwrap

import pytest

pytestmark = pytest.mark.L0

_SM120_BWD_PROBE = textwrap.dedent("""
    import sys
    from cudnn.sdpa.bwd.kernels.sm120 import bprop_f16 as m
    b, s, h, kvh = (int(x) for x in sys.argv[1:5])
    m.compile((12, 0), b=b, qh=h, sq=s, skv=s, d_qk=128, kvh=kvh)
""")


def _lower_sm120_bwd(b, s, h, kvh):
    """PTX text of every kernel of the SM120 f16 backward chain, lowered for sm_120a with no device."""
    with tempfile.TemporaryDirectory() as dump:
        env = dict(
            os.environ,
            CUTE_DSL_ARCH="sm_120a",
            CUTE_DSL_KEEP_PTX="1",
            CUTE_DSL_DUMP_DIR=dump,
            CUDNN_FRONTEND_DISABLE_COMPILED_CACHE="1",  # an sm_120a artifact must never reach the live cache
        )
        run = subprocess.run([sys.executable, "-c", _SM120_BWD_PROBE, *map(str, (b, s, h, kvh))], env=env, capture_output=True, text=True)
        if run.returncode != 0:
            pytest.skip(f"the installed CuTe DSL cannot lower sm_120a here: {run.stderr[-300:]}")
        texts = []
        for path in glob.glob(os.path.join(dump, "**", "*.ptx"), recursive=True):
            with open(path) as f:
                texts.append(f.read())
        text = "\n".join(texts)
    entries = {}
    for chunk in text.split(".entry ")[1:]:
        entries[re.match(r"(\w+?)(?:_tensorptr|_object)", chunk).group(1)] = chunk
    return entries


@pytest.mark.parametrize("kvh", [0, 16], ids=["mha", "gqa"])
def test_sm120_bwd_compact_dq_and_workspace_addressing_is_64_bit(kvh):
    """B4 S32K H128 D128 is exactly 2**31 elements per BSHD tensor (and per fp32 dq_accum / dK-dV workspace).

    Before the fix the convert kernel and the main kernel formed ``(batch * S + row) * H ...`` in Int32 and widened
    only at the byte multiply (``mul.wide.s32``); only the already-wide sites (and two small fp32 stats/semaphore
    indices in the main kernel) may keep that instruction now.
    """
    entries = _lower_sm120_bwd(4, 32768, 128, kvh)
    convert = next(v for k, v in entries.items() if "convert_dq_kernel" in k)
    main = next(v for k, v in entries.items() if "SM120FusedMultiHeadAttentionFP16Backward" in k)
    assert convert.count("mul.wide.s32") == 0
    assert main.count("mul.wide.s32") <= 2


def test_qat_triton_batch_head_offset_is_64_bit():
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from cudnn.sdpa.bwd.qat import _nvfp4

    signature = {
        "k_ptr": "*bf16",
        "v_ptr": "*bf16",
        "fake_k_ptr": "*bf16",
        "fake_v_ptr": "*bf16",
        **{name: "i32" for name in ("stride_b", "stride_h", "stride_s", "stride_d", "fake_stride_b", "fake_stride_h", "fake_stride_s", "fake_stride_d")},
        "num_heads": "i32",
        "seqlen_kv": "i32",
        "block_n": "constexpr",
        "head_dim": "constexpr",
    }
    constexprs = {"block_n": 128, "head_dim": 128, "stride_d": 1, "fake_stride_d": 1}
    ptx = None
    try:
        ptx = triton.compile(ASTSource(fn=_nvfp4.fake_quantize_kv, signature=signature, constexprs=constexprs), target=GPUTarget("cuda", 100, 32)).asm["ptx"]
    except Exception as exc:  # no sm_100 ptxas in this Triton build
        pytest.skip(f"Triton cannot lower sm_100 here: {exc}")
    assert ptx is not None
    # stride_b * (batch_head // num_heads) and stride_h * (batch_head % num_heads): 64-bit products
    assert len(re.findall(r"mul\.lo\.s64", ptx)) >= 2


@pytest.mark.parametrize("tokens,limit_ok", [(2**14, True), (2**21, False)], ids=["below_limit", "past_int32"])
def test_flex_varlen_declines_int32_span(tokens, limit_ok):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device to build the operand descriptors")
    from cudnn.flex_attention.plan.validation import validate_create_mask_plan_inputs

    heads, dim = 8, 128  # 2**21 * 8 * 128 = 2**31 elements, one past INT32_MAX
    q = None
    try:
        q = torch.empty((tokens, heads, dim), device="cuda", dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        pytest.skip("needs 4 GiB of device memory")
    assert q is not None
    cu = torch.tensor([0, tokens], dtype=torch.int32, device="cuda")
    mask = torch.zeros((1, 1, tokens), dtype=torch.int32, device="cuda")
    call = dict(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=tokens, max_seqlen_k=tokens)
    if limit_ok:
        validate_create_mask_plan_inputs(mask, q, q, q, **call)
    else:
        with pytest.raises(ValueError, match=r"2\*\*31"):
            validate_create_mask_plan_inputs(mask, q, q, q, **call)


def test_bsa_sage_fp8_q_batch_stride_must_fit_int32():
    from types import SimpleNamespace

    from cudnn.block_sparse_attention._interface import _sm100_blk64_require_int32_sage_q_strides

    def q(batch_stride):
        return SimpleNamespace(stride=lambda dim=None: (batch_stride, 2**27, 128, 1) if dim is None else (batch_stride, 2**27, 128, 1)[dim])

    with pytest.raises(ValueError, match="int32"):
        _sm100_blk64_require_int32_sage_q_strides(q(2**31))
    _sm100_blk64_require_int32_sage_q_strides(q(2**30))
