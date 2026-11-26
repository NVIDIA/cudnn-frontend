# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no kernel launch) pins for operands that span more than 2**31 elements.

Rule S7 (``python/cudnn/sdpa/AGENTS.md``): promote the index to Int64 BEFORE the stride multiply. Each test here
lowers the kernel for a declared geometry past the limit and inspects what was generated, so it runs on any lane
(no SM120 / SM100 part needed):

* SM120 backward, compact path: ``dq_accum`` / dQ / dK-dV-workspace addressing. Distinguish widening after a
  potentially wrapped Int32 offset from a safe full 32x32-to-64 product.
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
    from cudnn.sdpa.bwd.kernels.sm120.f16_fused import kernel as m
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
            # A failing project import fails the test; only a missing compiler target is a skip.
            assert "ImportError" not in run.stderr and "ModuleNotFoundError" not in run.stderr, run.stderr[-2000:]
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


def _non_tile_wide_products(ptx, max_tile_id):
    """Exclude widening products of a proven, non-wrapping tile-only index.

    Tile offsets originate at ctaid.x and may be copied or scaled in Int32
    while their full range fits. Adding a batch/head/row offset invalidates
    this proof: a subsequent widening multiply cannot undo an earlier wrap.
    Unknown instructions also invalidate the destination's proof.
    """
    tile_max = {"%ctaid.x": max_tile_id}
    products = []
    for instruction in ptx.split(";"):
        if "mul.wide.s32" in instruction:
            products.append(instruction.strip())
        match = re.search(r"(?:^|[\n{}])\s*(@!?%\w+\s+)?([a-z]\w*(?:\.[\w:]+)+)\s+([^;{}]+)$", instruction)
        if match is None:
            for register in re.findall(r"%r\w+", instruction):
                tile_max.pop(register, None)
            continue
        predicate, op, operands = match.groups()
        dst, *args = [arg.strip() for arg in operands.split(",")]
        bound = None
        if op == "mul.wide.s32":
            for index, arg in enumerate(args):
                other = args[1 - index]
                if arg in tile_max and re.fullmatch(r"-?\d+", other) and -(2**31) <= int(other) < 2**31:
                    products.pop()  # The two Int32 operands produce the full Int64 product.
                    break
        elif op in ("mov.u32", "mov.b32", "mov.s32") and args[0] in tile_max:
            bound = tile_max[args[0]]
        elif op == "shl.b32" and args[0] in tile_max and re.fullmatch(r"\d+", args[1]):
            shift = int(args[1])
            if shift < 32:
                bound = tile_max[args[0]] << shift
        elif op in ("mul.lo.s32", "mul.lo.u32"):
            for index, arg in enumerate(args):
                if arg in tile_max and re.fullmatch(r"\d+", args[1 - index]):
                    bound = tile_max[arg] * int(args[1 - index])
                    break
        tile_max.pop(dst, None)
        if predicate is None and bound is not None and bound <= 2**31 - 1:
            tile_max[dst] = bound
    return products


@pytest.mark.parametrize(
    "offset, count",
    [
        ("shl.b32 %r2, %r1, 6;", 0),
        ("mul.lo.s32 %r2, %r1, 64;", 0),
        ("shl.b32 %r2, %r1, 17;", 1),
        ("add.s32 %r2, %r1, %r_batch;", 1),
        ("shl.b32 %r2, %r1, 6;\nadd.s32 %r2, %r2, %r_batch;", 1),
        ("@%p1 mov.u32 %r2, %r1;", 1),
        ("shl.b32 %r2, %r1, 6;\nmov.b64 {%r2, %r3}, %rd_batch;", 1),
    ],
    ids=["tile_shift", "tile_multiply", "wrapped_tile", "mixed_offset", "redefined_offset", "predicated_definition", "vector_definition"],
)
def test_wide_product_filter_requires_bounded_tile_offset(offset, count):
    ptx = "mov.u32 %r1, %ctaid.x;\n" + offset + "\nmul.wide.s32 %rd1, %r2, 16384;"
    assert len(_non_tile_wide_products(ptx, 32767)) == count


@pytest.mark.parametrize("b, kvh", [(4, 0), (4, 16), (5, 0), (5, 16)], ids=["mha", "gqa", "mha_past_int32", "gqa_past_int32"])
def test_sm120_bwd_compact_dq_and_workspace_addressing_is_64_bit(b, kvh):
    """Compile compact tensors at and beyond 2**31 elements without allocating them.

    Only the two small stats/semaphore indices may retain unproven widening
    products in the main kernel. A tile-only offset that fits Int32 may also
    use mul.wide.s32: the product itself is Int64, even when its operand is not.
    Batch/row/head combinations remain subject to the original regression check.
    """
    entries = _lower_sm120_bwd(b, 32768, 128, kvh)
    convert = next(v for k, v in entries.items() if "convert_dq_kernel" in k)
    main = next(v for k, v in entries.items() if "SM120FusedMultiHeadAttentionFP16Backward" in k)
    assert convert.count("mul.wide.s32") == 0
    # Each KV tile covers at least one row, so gridDim.x <= S_KV.
    products = _non_tile_wide_products(main, 32768 - 1)
    assert len(products) <= 2, "\n".join(products)


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
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("cudnn.flex_attention supports SM90 and newer")
    from cudnn.flex_attention.plan.validation import validate_create_mask_plan_inputs
    from cudnn.flex_attention.runtime.arch import SUPPORTED_ARCHES

    major, minor = torch.cuda.get_device_capability()
    if limit_ok and major * 10 + minor not in SUPPORTED_ARCHES:
        # The span decline comes first and holds on any part; an in-limit span goes on to the arch check (SM12x is not served).
        pytest.skip(f"cudnn.flex_attention serves SM{', SM'.join(map(str, SUPPORTED_ARCHES))}; this device is SM{major}{minor}")

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
