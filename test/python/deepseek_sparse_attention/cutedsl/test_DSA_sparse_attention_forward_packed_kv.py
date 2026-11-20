# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 DSA sparse attention forward on a packed KV cache (``kv_format`` ``mxfp8_528`` / ``nvfp4_288``).

The packed plan widens every gathered row exactly to BF16 and runs the BF16 kernel's MMAs, so it returns, byte for byte,
what the BF16 plan returns on the widened pool (``block_scaled_kv_reference.widen_kv``). Every numerical check here
compares the raw bytes of all four outputs (``out``, ``max_logits``, ``lse``, ``lse_indexer``) with the BF16 plan's on
the same indices; one L1 case also checks the independent torch reference.

- Rows: quantized ``randn`` rows, and random code bytes (every non-NaN encoding) with scales in a moderate range.
- Indices: invalid ids (-1, other negatives, ids >= S_kv), duplicates, slots past ``topk_length``, an empty row.
- Pools: padded rows (a view of a wider buffer) and a row-offset view; a misaligned pool raises and is never copied.
- Rule 11: the staged raw rows are released behind a proxy fence after their last load (PTX pin), and a warm stress.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = [pytest.mark.gpu_exclusive, pytest.mark.xdist_group(name="gpu_exclusive")]

if not all(hasattr(torch, name) for name in ("float8_e8m0fnu", "float4_e2m1fn_x2")):
    pytest.skip("torch build lacks float8_e8m0fnu / float4_e2m1fn_x2", allow_module_level=True)

from deepseek_sparse_attention.cutedsl.block_scaled_kv_reference import ROW_LAYOUTS, pack_rows, quantize_rows, widen_rows  # noqa: E402
from deepseek_sparse_attention.cutedsl.dsa_reference import ref_sparse_attention_forward  # noqa: E402

_FORMATS = tuple(ROW_LAYOUTS)
# Shared-memory loads per lane of a staged tile (four rows): per row, eight code loads and one (MXFP8) or two (NVFP4)
# 16-byte scale loads.
_STAGED_LOADS = {"mxfp8_528": 4 * (8 + 1), "nvfp4_288": 4 * (8 + 2)}
_CODE_CONVERSION = {"mxfp8_528": "cvt.rn.bf16x2.e4m3x2", "nvfp4_288": "cvt.rn.bf16x2.e2m1x2"}
_D = 512


def _require_packed_kv():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")
    try:
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_forward._tma_gather4 import TMA_GATHER4_AVAILABLE
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if not TMA_GATHER4_AVAILABLE:
        pytest.skip("the installed CuTe DSL lacks the gather4 TMA bridge the packed path needs")
    return DSA


def _quantized_rows(fmt: str, s_kv: int, generator: torch.Generator, *, row_pitch=None) -> torch.Tensor:
    return quantize_rows(torch.randn((s_kv, _D), generator=generator, device="cuda"), fmt, row_pitch=row_pitch)


def _random_byte_rows(fmt: str, s_kv: int, generator: torch.Generator) -> torch.Tensor:
    """Rows of random code bytes (every non-NaN encoding) with scales of moderate magnitude, so every output is finite."""
    code_dtype, code_bytes, scale_bytes, scale_dtype = ROW_LAYOUTS[fmt]
    codes = torch.randint(0, 256, (s_kv, code_bytes), generator=generator, device="cuda", dtype=torch.int32)
    if fmt == "mxfp8_528":
        codes = torch.where((codes & 0x7F) == 0x7F, codes ^ 0x01, codes)  # E4M3 0x7F / 0xFF are NaN
        scales = torch.randint(117, 137, (s_kv, scale_bytes), generator=generator, device="cuda", dtype=torch.int32)  # 2**-10 .. 2**9
    else:
        exponent_mantissa = torch.randint(0x20, 0x58, (s_kv, scale_bytes), generator=generator, device="cuda", dtype=torch.int32)
        sign = torch.randint(0, 2, (s_kv, scale_bytes), generator=generator, device="cuda", dtype=torch.int32) << 7
        scales = exponent_mantissa | sign  # E4M3 2**-4 .. 2**4, both signs
    return pack_rows(codes.to(torch.uint8).view(code_dtype), scales.to(torch.uint8).view(scale_dtype))


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


def _assert_same_bytes(got, want, label: str):
    for name in ("out", "max_logits", "lse", "lse_indexer"):
        if want[name] is None:
            assert got[name] is None, f"{label}: {name}"
            continue
        same = _bits(got[name]) == _bits(want[name])
        assert bool(same.all()), f"{label}: {name} differs in {int((~same).sum())} of {same.numel()} values; first at {(~same).nonzero()[:3].tolist()}"


def _run_pair(DSA, q, rows, fmt, topk_idxs, **kwargs):
    """The packed plan on ``rows`` and the BF16 plan on their widened pool, same indices and options."""
    packed = DSA.sparse_attention_forward_wrapper(q, rows, topk_idxs, kv_format=fmt, **kwargs)
    bf16 = DSA.sparse_attention_forward_wrapper(q, widen_rows(rows, fmt), topk_idxs, **kwargs)
    torch.cuda.synchronize()
    return packed, bf16


# (heads, logical K, indexer_topk, topk_length form). H128 at indexer_topk 2048 runs the H64 kernel on two head windows.
_EQUALITY_CASES = {
    "h64-k65": (64, 65, 0, None),
    "h64-k640-ragged": (64, 640, 0, "ragged"),
    "h128-k2048-indexer2048-ragged": (128, 2048, 2048, "ragged"),
}


@pytest.mark.L0
@pytest.mark.parametrize("case", tuple(_EQUALITY_CASES), ids=tuple(_EQUALITY_CASES))
@pytest.mark.parametrize("fmt", _FORMATS)
def test_DSA_packed_kv_forward_equals_bf16_on_the_widened_pool(fmt, case):
    DSA = _require_packed_kv()
    heads, logical_topk, indexer_topk, length_form = _EQUALITY_CASES[case]
    generator = torch.Generator(device="cuda").manual_seed(4100 + logical_topk + heads)
    s_q, s_kv = 5, 1000
    q = (torch.randn((s_q, heads, _D), generator=generator, device="cuda") / 8).to(torch.bfloat16)
    topk_idxs = torch.randint(0, s_kv, (s_q, logical_topk), generator=generator, device="cuda", dtype=torch.int32)
    topk_length = None
    if length_form == "ragged":
        lengths = (logical_topk, logical_topk // 2 + 1, 0, 64, 65)
        topk_length = torch.tensor([max(length, indexer_topk) if i < 2 else length for i, length in enumerate(lengths)], dtype=torch.int32, device="cuda")
    attn_sink = torch.linspace(-0.5, 0.5, heads, device="cuda") if heads == 128 else None
    for kind, rows in (("quantized", _quantized_rows(fmt, s_kv, generator)), ("random-bytes", _random_byte_rows(fmt, s_kv, generator))):
        packed, bf16 = _run_pair(DSA, q, rows, fmt, topk_idxs, topk_length=topk_length, attn_sink=attn_sink, indexer_topk=indexer_topk)
        assert bool(torch.isfinite(packed["out"].float()).all())
        _assert_same_bytes(packed, bf16, f"{fmt} {case} {kind}")


@pytest.mark.L0
@pytest.mark.parametrize("fmt", _FORMATS)
def test_DSA_packed_kv_forward_invalid_slots_read_nothing(fmt):
    """Invalid ids and slots past ``topk_length`` contribute nothing: a NaN row they point at never leaks, and an empty row
    returns the sentinel (O = 0, max_logits = -inf, lse = +inf)."""
    DSA = _require_packed_kv()
    generator = torch.Generator(device="cuda").manual_seed(4211)
    s_kv, logical_topk = 300, 128
    rows = _quantized_rows(fmt, s_kv, generator)
    nan_row = 7
    code_bytes = ROW_LAYOUTS[fmt][1]
    rows[nan_row, code_bytes] = 0xFF if fmt == "mxfp8_528" else 0x7F  # the first block's scale is NaN
    assert bool(widen_rows(rows, fmt)[nan_row].isnan().any())
    valid = torch.randint(0, s_kv, (logical_topk,), generator=generator, device="cuda", dtype=torch.int32)
    valid = torch.where(valid == nan_row, valid + 1, valid)
    invalid = torch.tensor([-1, -7, s_kv, s_kv + 5, 2**31 - 1, -(2**31)], dtype=torch.int32, device="cuda")
    topk_idxs = torch.stack(
        (
            torch.cat((invalid, valid[: logical_topk - len(invalid)])),  # invalid ids mixed with valid ones
            torch.cat(
                (valid[:40], valid[:40], torch.full((logical_topk - 80,), nan_row, dtype=torch.int32, device="cuda"))
            ),  # duplicates; NaN row past the length
            torch.full((logical_topk,), -1, dtype=torch.int32, device="cuda"),  # all invalid
            valid,  # length 0
        )
    )
    topk_length = torch.tensor([logical_topk, 80, logical_topk, 0], dtype=torch.int32, device="cuda")
    q = (torch.randn((4, 64, _D), generator=generator, device="cuda") / 8).to(torch.bfloat16)
    packed, bf16 = _run_pair(DSA, q, rows, fmt, topk_idxs, topk_length=topk_length)
    _assert_same_bytes(packed, bf16, f"{fmt} invalid slots")
    assert bool(torch.isfinite(packed["out"].float()).all())
    for empty in (2, 3):
        assert bool((packed["out"][empty] == 0).all())
        assert bool((packed["max_logits"][empty] == float("-inf")).all())
        assert bool((packed["lse"][empty] == float("inf")).all())


@pytest.mark.L0
@pytest.mark.parametrize("fmt", _FORMATS)
def test_DSA_packed_kv_forward_reads_pool_views_in_place(fmt, monkeypatch):
    """Padded rows and a row-offset view are read in place, never copied; a misaligned pool raises ``ValueError``."""
    DSA = _require_packed_kv()
    from cudnn.deepseek_sparse_attention.sparse_attention_forward import _interface_sm100

    copied = []
    contiguous_aligned = _interface_sm100._contiguous_aligned

    def recording_contiguous_aligned(tensor, alignment):
        copied.append(tensor.dtype)
        return contiguous_aligned(tensor, alignment)

    monkeypatch.setattr(_interface_sm100, "_contiguous_aligned", recording_contiguous_aligned)
    generator = torch.Generator(device="cuda").manual_seed(4307)
    record = ROW_LAYOUTS[fmt][1] + ROW_LAYOUTS[fmt][2]
    s_kv, logical_topk = 257, 192
    padded = _quantized_rows(fmt, s_kv, generator, row_pitch=record + 16)  # padding bytes are 0xFF
    compact = padded[:, :record].contiguous()
    q = (torch.randn((3, 64, _D), generator=generator, device="cuda") / 8).to(torch.bfloat16)
    topk_idxs = torch.randint(0, s_kv - 1, (3, logical_topk), generator=generator, device="cuda", dtype=torch.int32)

    want = DSA.sparse_attention_forward_wrapper(q, compact, topk_idxs, kv_format=fmt)
    got = DSA.sparse_attention_forward_wrapper(q, padded[:, :record], topk_idxs, kv_format=fmt)
    torch.cuda.synchronize()
    _assert_same_bytes(got, want, f"{fmt} padded rows")
    want = DSA.sparse_attention_forward_wrapper(q, compact[1:].contiguous(), topk_idxs, kv_format=fmt)
    got = DSA.sparse_attention_forward_wrapper(q, padded[1:, :record], topk_idxs, kv_format=fmt)
    torch.cuda.synchronize()
    _assert_same_bytes(got, want, f"{fmt} row-offset view")
    assert torch.uint8 not in copied, "a packed pool was handed to the copying normalization"

    misaligned = torch.empty((s_kv * record + 8,), dtype=torch.uint8, device="cuda")[8:].view(s_kv, record)
    with pytest.raises(ValueError, match="16-byte aligned base"):
        DSA.sparse_attention_forward_wrapper(q, misaligned, topk_idxs, kv_format=fmt)


@pytest.mark.L0
def test_DSA_packed_kv_forward_declines_unserved_declarations():
    DSA = _require_packed_kv()
    device = torch.device("cuda")
    topk = torch.zeros((1, 2048), dtype=torch.int32, device=device)

    def rows(record):
        return torch.empty((2, record), dtype=torch.uint8, device=device)

    q_bf16 = torch.empty((1, 64, 512), dtype=torch.bfloat16, device=device)
    for fmt in _FORMATS:
        record = ROW_LAYOUTS[fmt][1] + ROW_LAYOUTS[fmt][2]
        assert DSA.SparseAttentionForward(q_bf16, rows(record), topk, kv_format=fmt).check_support()
        with pytest.raises(NotImplementedError, match=f"does not serve kv_format='{fmt}' on SM10.*served with bfloat16 Q"):
            DSA.SparseAttentionForward(q_bf16.half(), rows(record), topk, kv_format=fmt).check_support()
        # H128 D512 at indexer_topk <= 1024 runs the dedicated H128 kernel, which reads typed KV only.
        q_h128 = torch.empty((1, 128, 512), dtype=torch.bfloat16, device=device)
        for indexer_topk in (0, 512, 1024):
            with pytest.raises(NotImplementedError, match="reads typed KV only"):
                DSA.SparseAttentionForward(q_h128, rows(record), topk, kv_format=fmt, indexer_topk=indexer_topk).check_support()
        assert DSA.SparseAttentionForward(q_h128, rows(record), topk, kv_format=fmt, indexer_topk=2048).check_support()
    q_576 = torch.empty((1, 64, 576), dtype=torch.bfloat16, device=device)
    for fmt, q, record in (("fp8_528", q_bf16, 528), ("fp8_bf16rope_656", q_576, 656), ("nvfp4_fp8rope_352", q_576, 352)):
        with pytest.raises(NotImplementedError, match=f"does not serve kv_format='{fmt}' on SM10.*the served formats are"):
            DSA.SparseAttentionForward(q, rows(record), topk, kv_format=fmt).check_support()


def _staged_row_release(ptx: str, fmt: str):
    """Line numbers, in the widening loop of ``ptx``, of the staged-row loads, the code conversions, the proxy fences
    after the last load and the first ``mbarrier.arrive`` after it (the ``raw_free`` release)."""
    lines = ptx.splitlines()
    conversions = [i for i, line in enumerate(lines) if _CODE_CONVERSION[fmt] in line]
    assert conversions, f"no {_CODE_CONVERSION[fmt]} in the rendering"
    acquire = max(i for i in range(conversions[0]) if "mbarrier.try_wait" in lines[i])
    loads = [i for i in range(acquire, len(lines)) if lines[i].lstrip().startswith("ld.shared")]
    arrive = next(i for i in range(conversions[-1], len(lines)) if "mbarrier.arrive" in lines[i])
    loads = [i for i in loads if i < arrive]
    fences = [i for i in range(loads[-1], arrive) if "fence.proxy.async.shared::cta" in lines[i]]
    return loads, conversions, fences, arrive


@pytest.mark.L0
@pytest.mark.parametrize("fmt", _FORMATS)
def test_DSA_packed_kv_forward_releases_staged_rows_behind_a_proxy_fence(fmt, tmp_path):
    """Rule 11, fence form: a raw stage is refilled by TMA (an async-proxy write), so the widening loop releases it
    (``raw_free``) behind ``fence.proxy.async.shared::cta`` after the tile's last load and its conversions. Compiled in
    a child process that keeps the PTX; the warm stress below is the run-time detector."""
    _require_packed_kv()
    probe = textwrap.dedent(f"""
        import torch
        from cudnn import DSA
        rows = torch.zeros((256, {ROW_LAYOUTS[fmt][1] + ROW_LAYOUTS[fmt][2]}), dtype=torch.uint8, device="cuda")
        q = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="cuda")
        DSA.sparse_attention_forward_wrapper(q, rows, torch.zeros((1, 64), dtype=torch.int32, device="cuda"), kv_format={fmt!r})
        torch.cuda.synchronize()
        """)
    dump, cache = tmp_path / "dump", tmp_path / "cache"
    dump.mkdir()
    cache.mkdir()
    env = dict(os.environ, CUTE_DSL_DUMP_DIR=str(dump), CUTE_DSL_KEEP="ptx", CUTE_DSL_CACHE_DIR=str(cache))
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=1200, env=env)
    assert proc.returncode == 0, proc.stderr[-4000:]
    ptx_files = sorted(dump.rglob("*head64*.ptx"))
    assert len(ptx_files) == 1, f"expected one H64 rendering, found {[p.name for p in ptx_files]}"
    ptx = ptx_files[0].read_text()
    loads, conversions, fences, arrive = _staged_row_release(ptx, fmt)
    assert len(loads) == _STAGED_LOADS[fmt], f"expected {_STAGED_LOADS[fmt]} staged-row loads, found {len(loads)}"
    assert fences, "no fence.proxy.async.shared::cta between the last staged-row load and the raw_free release"
    assert conversions[-1] < fences[-1], "a code conversion follows the release fence"
    lines = ptx.splitlines()
    releases = [lines[i].strip() for i in range(loads[-1] + 1, fences[-1]) if "mbarrier" in lines[i] or "bar.sync" in lines[i] or "barrier." in lines[i]]
    assert not releases, f"a barrier operation precedes the release fence: {releases[:3]}"


_L1_CASES = {
    "h64-k2048-indexer512": (64, 2048, 512),
    "h96-k1100-indexer1024": (96, 1100, 1024),
    "h64-k2048-indexer1024": (64, 2048, 1024),
}


@pytest.mark.L1
@pytest.mark.parametrize("case", tuple(_L1_CASES), ids=tuple(_L1_CASES))
@pytest.mark.parametrize("fmt", _FORMATS)
def test_DSA_packed_kv_forward_matrix(fmt, case):
    DSA = _require_packed_kv()
    heads, logical_topk, indexer_topk = _L1_CASES[case]
    generator = torch.Generator(device="cuda").manual_seed(4409 + heads + logical_topk + indexer_topk)
    s_q, s_kv = 7, 3000
    q = (torch.randn((s_q, heads, _D), generator=generator, device="cuda") / 8).to(torch.bfloat16)
    topk_idxs = torch.randint(-2, s_kv + 2, (s_q, logical_topk), generator=generator, device="cuda", dtype=torch.int32)
    topk_length = torch.randint(indexer_topk, logical_topk + 1, (s_q,), generator=generator, device="cuda", dtype=torch.int32)
    attn_sink = torch.linspace(-1.0, 1.0, heads, device="cuda")
    for kind, rows in (("quantized", _quantized_rows(fmt, s_kv, generator)), ("random-bytes", _random_byte_rows(fmt, s_kv, generator))):
        packed, bf16 = _run_pair(DSA, q, rows, fmt, topk_idxs, topk_length=topk_length, attn_sink=attn_sink, indexer_topk=indexer_topk)
        _assert_same_bytes(packed, bf16, f"{fmt} {case} {kind}")


@pytest.mark.L1
@pytest.mark.parametrize("fmt", _FORMATS)
def test_DSA_packed_kv_forward_matches_the_reference(fmt):
    DSA = _require_packed_kv()
    generator = torch.Generator(device="cuda").manual_seed(4513)
    s_q, s_kv, heads, logical_topk, indexer_topk = 3, 500, 64, 640, 512
    rows = _quantized_rows(fmt, s_kv, generator)
    q = (torch.randn((s_q, heads, _D), generator=generator, device="cuda") / 8).to(torch.bfloat16)
    topk_idxs = torch.randint(0, s_kv, (s_q, logical_topk), generator=generator, device="cuda", dtype=torch.int32)
    attn_sink = torch.linspace(-0.5, 0.5, heads, device="cuda")
    result = DSA.sparse_attention_forward_wrapper(q, rows, topk_idxs, kv_format=fmt, attn_sink=attn_sink, indexer_topk=indexer_topk)
    ref_out, ref_max, ref_lse, ref_lse_indexer = ref_sparse_attention_forward(
        q, widen_rows(rows, fmt), attn_sink, topk_idxs, indexer_topk=indexer_topk, return_full=True
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(result["out"].float(), ref_out.float(), atol=8e-4, rtol=3.01 / 128)
    torch.testing.assert_close(result["max_logits"], ref_max, atol=1e-6, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse"], ref_lse, atol=1e-6, rtol=2.01 / 65536)
    torch.testing.assert_close(result["lse_indexer"], ref_lse_indexer, atol=1e-6, rtol=2.01 / 65536)


@pytest.mark.L1
@pytest.mark.parametrize("fmt", _FORMATS)
def test_DSA_packed_kv_forward_warm_stress(fmt):
    """Rule 11 run-time detector: every staging slot is refilled many times per launch (32 tiles per CTA), warm and behind
    shared-memory poisoning, against the BF16 plan's bytes on the widened pool. The kernel runs one CTA per SM."""
    DSA = _require_packed_kv()
    import warm_stress

    generator = torch.Generator(device="cuda").manual_seed(4621)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    s_q, s_kv, heads, logical_topk = 2 * sms, 4096, 64, 2048
    rows = _random_byte_rows(fmt, s_kv, generator)
    q = (torch.randn((s_q, heads, _D), generator=generator, device="cuda") / 8).to(torch.bfloat16)
    topk_idxs = torch.randint(0, s_kv, (s_q, logical_topk), generator=generator, device="cuda", dtype=torch.int32)
    reference = DSA.sparse_attention_forward_wrapper(q, widen_rows(rows, fmt), topk_idxs)
    plan = DSA.SparseAttentionForward(q, rows, topk_idxs, kv_format=fmt)
    assert plan.check_support()
    plan.compile()
    outputs = {
        "out": torch.empty_like(reference["out"]),
        "max_logits": torch.empty_like(reference["max_logits"]),
        "lse": torch.empty_like(reference["lse"]),
    }
    torch.cuda.synchronize()
    case = warm_stress.StressCase(
        name=f"dsa forward packed kv {fmt}",
        launch=lambda: plan.execute(q, rows, topk_idxs, out=outputs["out"], max_logits=outputs["max_logits"], lse=outputs["lse"]),
        outputs=outputs,
        thread_blocks=s_q,
        refills_per_launch=s_q * (logical_topk // 64),
        reference={name: reference[name] for name in outputs},
    )
    warm_stress.assert_warm_stress(case, validation=warm_stress.stress_validation())
