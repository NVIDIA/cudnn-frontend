# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""sm120 (consumer Blackwell) matmul template gate.

Two layers, mirroring ``test_matmul.py``:

* Wiring tests (no sm_120 GPU needed — they run on the sm100 CI too): the
  registry / tile-config / compiler plumbing for ``sm120_matmul.py`` — catalog
  family, template routing, the v1 scope gates, the ≤8-element epilogue chunk
  clamp with STG-only rendering, and (on any CUDA GPU) a full source render.

* End-to-end correctness on an sm_120 GPU: (config × dtype-pair × shape)
  sweeps asserting bit-tight equality vs torch-fp32 (small-integer inputs keep
  the reduction exact), plus batched / batch-broadcast, epilogue fusion,
  narrow output rows, and the engine's auto-select path.

CUDNN_GEMM_TEST_FULL=1 expands the config axis to every sm120 catalog
geometry. Also runnable as a script (forwards argv to pytest).
"""

from __future__ import annotations

import ast
import os
import sys

import pytest
import torch

from gemm_test_utils import (
    Plan as _plan,
    vp as _vp,
    resolve as _resolve,
)

pytestmark = [pytest.mark.L0]


import cudnn
import cudnn.gemm.frost  # noqa: F401  — installs the cudnn.pygraph recorder hook
from cudnn.gemm.frost.graph_analyzer import analyze
from cudnn.gemm.frost.kernel_registry import PIPELINE_ARCH_RANGES
from cudnn.gemm.frost.tile_config import CATALOG, ConfigSm120, by_name

# --- arch gate ---------------------------------------------------------------


def _active_sm() -> int | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


_SM = _active_sm()
_SM120_RANGES = PIPELINE_ARCH_RANGES["sm120"]

# The e2e tests JIT + LAUNCH the sm120 warp-MMA template; gate on consumer
# Blackwell so wrong-arch machines skip instead of failing in the launch.
requires_sm120 = pytest.mark.skipif(
    _SM is None or not (120 <= _SM < 130),
    reason="needs a consumer-Blackwell GPU (120 <= SM < 130), have " + ("none" if _SM is None else f"sm_{_SM}"),
)

requires_any_gpu = pytest.mark.skipif(
    _SM is None,
    reason="rendering sizes the SMEM/L2 budgets from the active GPU",
)


_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
_CUDNN_DTYPE = {
    "bf16": cudnn.data_type.BFLOAT16,
    "fp16": cudnn.data_type.HALF,
    "fp8_e4m3": cudnn.data_type.FP8_E4M3,
    "fp8_e5m2": cudnn.data_type.FP8_E5M2,
}
_ELEM_BYTES = {"bf16": 2, "fp16": 2, "fp8_e4m3": 1, "fp8_e5m2": 1}


# Shape menu: tile-aligned baseline + M-OOB + N-OOB + K-OOB + combined. Every N
# is a multiple of 8, so 2-byte outputs keep 16-byte rows — one compiled anchor
# per (config, dtype) serves the whole menu (the kernel is shape-agnostic).
_WEIRD_SHAPES: tuple[tuple[int, int, int], ...] = (
    # Tile-aligned baseline.
    (384, 768, 384),
    (640, 384, 512),
    (256, 1280, 256),
    (512, 1024, 640),  # K = 5×128
    # M-OOB (N, K aligned).
    (255, 256, 256),  # one row short of a tile
    (200, 256, 256),  # deep inside a partial tile
    # N-OOB (predicated pair stores / TMA global-extent clip).
    (256, 200, 256),
    # K-OOB (bf16/fp16 only for K=200; FP8 skips via the 16B TMA stride rule).
    (256, 256, 200),
    (256, 256, 96),  # smaller than one K_BYTES=128 BF16 tile
    # M + N + K OOB.
    (255, 200, 240),
)

# (input_dtype, output_dtype) pairs — the sm120 MMA menu (fp32-accumulated).
_CORE_DTYPE_PAIRS: tuple[tuple[str, str], ...] = (
    ("bf16", "bf16"),
    ("fp16", "fp16"),
    ("fp8_e4m3", "fp16"),
    ("fp8_e5m2", "fp16"),
    ("fp8_e4m3", "bf16"),
)

# Curated config subset — each entry covers a distinct template corner. The
# 48/144/16 N-tiles are synthesized by name (the catalog walks N in 32s). The
# full sm120 catalog sweep is opt-in via CUDNN_GEMM_TEST_FULL=1.
_QUICK_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm120_128x128x128_128x128x32_cluster1x1_1ctamma",  # baseline
    "CONFIG_sm120_128x256x128_128x256x32_cluster1x1_1ctamma",  # large N
    "CONFIG_sm120_128x64x128_128x64x32_cluster1x1_1ctamma",  # narrow N
    "CONFIG_sm120_128x128x64_128x128x32_cluster1x1_1ctamma",  # K_BYTES=64 (s64b AB swizzle)
    "CONFIG_sm120_64x128x128_64x128x32_cluster1x1_1ctamma",  # cta_m=64 (16-row warp tile)
    "CONFIG_sm120_128x48x128_128x48x32_cluster1x1_1ctamma",  # N%32 != 0, odd n-frag tail
    "CONFIG_sm120_128x144x128_128x144x32_cluster1x1_1ctamma",  # 9 n-frags/warp (4 pairs + tail)
    "CONFIG_sm120_64x16x128_64x16x32_cluster1x1_1ctamma",  # minimum tile (single-n-frag warp)
    "CONFIG_sm120_64x64x64_64x64x32_cluster1x1_1ctamma",  # cta_m=64 + K_BYTES=64
    "CONFIG_sm120_128x256x64_128x256x32_cluster1x1_1ctamma",  # K_BYTES=64, large N
)

_BATCHED_CONFIGS: tuple[str, ...] = (
    "CONFIG_sm120_128x128x128_128x128x32_cluster1x1_1ctamma",
    "CONFIG_sm120_64x128x128_64x128x32_cluster1x1_1ctamma",
)

_BATCHED_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (1, 384, 768, 384),
    (2, 640, 384, 512),
    (3, 255, 256, 256),  # M-OOB
    (2, 256, 200, 240),  # N + K OOB
)

_BATCH_BROADCAST_CASES = tuple((side, (2, 384, 256, 384)) for side in ("A", "B"))


def _sweep_config_names() -> list[str]:
    """Quick subset by default; the whole sm120 catalog under CUDNN_GEMM_TEST_FULL=1."""
    if os.environ.get("CUDNN_GEMM_TEST_FULL", "0") == "1":
        return [f"{c.name}_1ctamma" for c in CATALOG if c.pipeline == "sm120"]
    return list(_QUICK_CONFIGS)


def _shape_id(s: tuple[int, int, int]) -> str:
    return f"{s[0]}x{s[1]}x{s[2]}"


def _dtype_id(p: tuple[str, str]) -> str:
    return f"{p[0]}->{p[1]}"


def _config_id(name: str) -> str:
    return name.removeprefix("CONFIG_sm120_")


# --- compatibility gate -------------------------------------------------------


def _compatible(cfg, M: int, N: int, K: int, in_dtype: str, out_dtype: str) -> tuple[bool, str]:
    """Reject only shapes the sm120 kernel can't service. Returns (ok, reason)."""
    in_eb = _ELEM_BYTES[in_dtype]
    out_eb = _ELEM_BYTES[out_dtype]
    if cfg.cta_tile_k_bytes % in_eb != 0:
        return False, f"K_BYTES={cfg.cta_tile_k_bytes} not divisible by in_elem_bytes={in_eb}"
    # K-major A and B: the TMA contiguous extent is K on both sides.
    if (K * in_eb) % 16 != 0:
        return False, f"K*in_eb={K * in_eb} not 16B-aligned (TMA contiguous-extent rule); " f"{in_dtype!r} needs K % {16 // in_eb} == 0"
    # The pair epilogue stores 2 output elements per thread.
    if (N * out_eb) % 4 != 0:
        return False, f"N*out_eb={N * out_eb} not 4B-aligned — the (n, n+1) pair store needs it"
    return True, ""


# --- graph + data + reference ------------------------------------------------


def _a_stride_batched(M: int, K: int, a_major: str) -> list[int]:
    return [M * K, K, 1] if a_major == "k" else [M * K, 1, M]


def _b_stride_batched(N: int, K: int, b_major: str) -> list[int]:
    return [N * K, 1, K] if b_major == "k" else [N * K, N, 1]


def _build_graph(
    M: int,
    N: int,
    K: int,
    in_dtype: str = "bf16",
    out_dtype: str = "bf16",
    a_major: str = "k",
    b_major: str = "k",
    out_major: str = "n",
) -> cudnn.pygraph:
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, a_major))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, b_major))
    C = g.matmul(A=A, B=B, name="mm")
    if out_major == "m":
        C.set_stride([M * N, 1, M])
    C.set_output(True)
    if out_dtype != in_dtype:
        C.set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _build_batched_graph(batch: int, M: int, N: int, K: int, in_dtype: str, out_dtype: str, a_batch=None, b_batch=None) -> cudnn.pygraph:
    """Rank-3 batched matmul; pass ``a_batch``/``b_batch``=1 for a broadcast side."""
    g = cudnn.pygraph(
        io_data_type=_CUDNN_DTYPE[in_dtype],
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[a_batch or batch, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[b_batch or batch, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True)
    if out_dtype != in_dtype:
        C.set_data_type(_CUDNN_DTYPE[out_dtype])
    return g


def _build_bias_relu_graph(M: int, N: int, K: int) -> cudnn.pygraph:
    """matmul -> per-col bias -> relu: an aux + pointwise epilogue chain (aux
    chains take the STG path — the TMA-store gate excludes aux)."""
    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    A = g.tensor(name="A", dim=[1, M, K], stride=_a_stride_batched(M, K, "k"))
    B = g.tensor(name="B", dim=[1, K, N], stride=_b_stride_batched(N, K, "k"))
    C = g.matmul(A=A, B=B, name="mm")
    bias = g.tensor(name="bias0", dim=[1, N], stride=[N, 1])
    D = g.bias(input=C, bias=bias, name="bias")
    E = g.relu(input=D, name="relu")
    E.set_output(True)
    return g


def _mkdata(batch: int, M: int, N: int, K: int, in_dtype: str, out_dtype: str, seed: int = 0):
    """Small-integer inputs ⇒ exact FP32 reduction ⇒ kernel and reference differ
    only by the final deterministic downcast. Rank-3, K-major."""
    torch.manual_seed(seed)
    rng = (-3, 3) if in_dtype.startswith("fp8") else (-2, 2)
    a = torch.empty(batch, M, K, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    b = torch.empty(batch, N, K, dtype=torch.int32).random_(*rng).to(dtype=_TORCH_DTYPE[in_dtype], device="cuda")
    c = torch.empty(batch, M, N, dtype=_TORCH_DTYPE[out_dtype], device="cuda")
    return a, b, c


def _reference(a: torch.Tensor, b: torch.Tensor, out_dtype: str) -> torch.Tensor:
    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    return ref.to(_TORCH_DTYPE[out_dtype])


def _assert_bit_tight(c, ref, header: str) -> None:
    """Both sides reduce exactly in FP32 and downcast the same way ⇒ equality."""
    diff = (c.to(torch.float32) - ref.to(torch.float32)).abs()
    bad = int((diff > 0).sum().item())
    assert bad == 0, (
        f"\n  {header}"
        f"\n  bad:       {bad}/{diff.numel()} ({100 * bad / diff.numel():.2f}%)"
        f"\n  max|diff|: {float(diff.max().item()):.4g}"
        f"\n  max|ref|:  {float(ref.abs().max().item()):.4g}"
        f"\n  hint:      sample c[0,0,:8]   = {c[0, 0, :8].to(torch.float32).tolist()}"
        f"\n             sample ref[0,0,:8] = {ref[0, 0, :8].to(torch.float32).tolist()}"
    )


# --- compile cache (session-scoped) --------------------------------------------


@pytest.fixture(scope="session")
def _compile_cache() -> dict:
    """Maps a case key → Plan | ("skip"|"fail", msg). Cases visit in (config,
    dtype, shape) order, so each (config, dtype) block shares one compile."""
    return {}


def _cached_outcome(entry):
    if isinstance(entry, tuple) and entry[0] in ("skip", "fail"):
        kind, msg = entry
        if kind == "skip":
            pytest.skip(msg)
        pytest.fail(msg, pytrace=False)
    return entry


def _plan_or_skip(cache, key, build_graph, cfg, cta_group, **plan_kw):
    """JIT the anchor graph; the engine's clean "unsupported" rejections SKIP,
    any other compile error FAILS."""
    try:
        compiled = _plan(build_graph(), config=cfg, cta_group=cta_group, **plan_kw)
    except Exception as e:
        first = str(e).splitlines()[0] if str(e) else ""
        if isinstance(e, NotImplementedError) or (isinstance(e, ValueError) and "no kernel template" in str(e)):
            msg = first[:300]
            cache[key] = ("skip", msg)
            pytest.skip(msg)
        msg = f"JIT compile failed: {type(e).__name__}: {first[:200]}"
        cache[key] = ("fail", msg)
        pytest.fail(msg, pytrace=False)
    cache[key] = compiled
    return compiled


def _pick_anchor(cfg, in_dt: str, out_dt: str) -> tuple[int, int, int] | None:
    for shape in _WEIRD_SHAPES:
        ok, _ = _compatible(cfg, *shape, in_dt, out_dt)
        if ok:
            return shape
    return None


def _get_compiled(cache: dict, cfg, in_dt: str, out_dt: str, cta_group: int):
    key = (cfg.name, in_dt, out_dt, cta_group)
    if key in cache:
        return _cached_outcome(cache[key])
    anchor = _pick_anchor(cfg, in_dt, out_dt)
    if anchor is None:
        msg = f"no menu shape is compatible with ({cfg.name}, {in_dt}->{out_dt})"
        cache[key] = ("skip", msg)
        pytest.skip(msg)
    return _plan_or_skip(cache, key, lambda: _build_graph(*anchor, in_dt, out_dt), cfg, cta_group)


# =============================================================================
# Wiring tests — no sm_120 GPU required (registry / config / compiler gates).
# =============================================================================


def test_sm120_registry_wiring() -> None:
    """The sm120 template is registered with its own family, class and gates."""
    from cudnn.gemm.frost.kernel_registry import (
        MMA_TYPE_SUPPORT,
        TEMPLATES,
        GraphType,
        Sm120KernelTemplate,
    )

    # SM 12.x is in the family's active range (whatever else the range covers).
    assert any(lo <= 120 < hi for lo, hi in _SM120_RANGES)

    (tmpl,) = [t for t in TEMPLATES if t.pipeline == "sm120"]
    assert tmpl.file == "sm120_matmul.py"
    assert isinstance(tmpl, Sm120KernelTemplate)
    # Warp-scoped MMA: 1-CTA only, no multi-GEMM (no per-GEMM operand indexing).
    assert tmpl.cta_group == 1 and not tmpl.supports_multi_gemm
    assert tmpl.graph_type is GraphType.MATMUL and not tmpl.mainloop

    # dtype support mirrors the sm100 matmul pipeline (incl. int8 and fp8 mixes)
    assert ("bf16", "bf16", "fp32") in MMA_TYPE_SUPPORT["sm120"][GraphType.MATMUL]
    assert ("int8", "int8", "int32") in MMA_TYPE_SUPPORT["sm120"][GraphType.MATMUL]

    # the template file itself ships with the package
    from pathlib import Path

    import cudnn.gemm.frost.compiler as C

    assert (Path(C.__file__).parent / "kernel_templates" / "sm120_matmul.py").is_file()


def test_sm120_tile_config_family() -> None:
    """ConfigSm120: catalog membership, name round-trip, geometry guards, and
    the auto path's as_pipeline conversion (cluster + block size move)."""
    from cudnn.gemm.frost.tile_config import as_pipeline

    sm120 = [c for c in CATALOG if c.pipeline == "sm120"]
    assert len(sm120) == 2 * 8 * 2  # m {128,64} × n {32..256/32} × kb {128,64}
    assert all(isinstance(c, ConfigSm120) for c in sm120)
    assert all(c.cgrp_size_mn == (1, 1) and c.threads_per_cta == 384 for c in sm120)

    cfg = by_name("CONFIG_sm120_128x128x128_128x128x32_cluster1x1")
    assert isinstance(cfg, ConfigSm120)
    assert cfg.name == "CONFIG_sm120_128x128x128_128x128x32_cluster1x1"
    # non-catalog sm120 geometries synthesize with the family's 12-warp block
    c48 = by_name("CONFIG_sm120_128x48x128_128x48x32_cluster1x1")
    assert c48.threads_per_cta == 384 and c48.cgrp_size_mn == (1, 1)

    # a clustered sm120 NAME pins to 1x1 and so fails the canonical round-trip;
    # the free axes (16-col n-frag granularity, no MMA-M split) reject directly
    with pytest.raises(KeyError, match="round-trips"):
        by_name("CONFIG_sm120_128x128x128_128x128x32_cluster2x1")
    with pytest.raises(NotImplementedError, match="cta_tile_n"):
        by_name("CONFIG_sm120_128x24x128_128x24x32_cluster1x1")
    with pytest.raises(NotImplementedError):
        by_name("CONFIG_sm120_256x128x128_128x128x32_cluster1x1")

    # as_pipeline: an sm100 auto pick crosses over, and ConfigSm120 pins its
    # family-fixed axes (cluster -> 1x1, threads -> 384) by itself
    picked = by_name("CONFIG_sm100_128x256x128_128x256x32_cluster2x1")
    conv = as_pipeline(picked, "sm120")
    assert isinstance(conv, ConfigSm120)
    assert conv.cgrp_size_mn == (1, 1) and conv.threads_per_cta == 384
    assert conv.cta_tile_mn == picked.cta_tile_mn

    # resolve() understands the legacy test names
    r_cfg, r_group = _resolve("CONFIG_sm120_128x128x128_128x128x32_cluster1x1_1ctamma")
    assert r_cfg is cfg and r_group == 1


def test_sm120_template_routing() -> None:
    """select_template: (sm120 config, cta_group=1) → the sm120 template; the
    strategies sm120 doesn't have decline with the registry's clean error."""
    from cudnn.gemm.frost.kernel_registry import select_template

    chain = analyze(_build_graph(256, 256, 128))
    cfg = by_name("CONFIG_sm120_128x128x128_128x128x32_cluster1x1")
    assert select_template(chain, cfg, 1).file == "sm120_matmul.py"
    with pytest.raises(ValueError, match="no kernel template"):
        select_template(chain, cfg, 2)
    # sm100 configs keep routing to the sm100 family
    cfg100 = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    assert select_template(chain, cfg100, 1).file == "sm100_matmul_1ctamma.py"


def test_sm120_scope_gates() -> None:
    """The v1 scope contract rejects through the registry (never a template
    AssertionError mid-render): K-major-only inputs, N-major output, and a
    pair-storable epilogue chunk."""
    from cudnn.gemm.frost.kernel_registry import select_template

    cfg = by_name("CONFIG_sm120_128x128x128_128x128x32_cluster1x1")
    tmpl = select_template(analyze(_build_graph(256, 256, 128)), cfg, 1)

    def scope_reject(**graph_kw):
        return tmpl._extra_reject(analyze(_build_graph(256, 256, 128, **graph_kw)), cfg)

    assert scope_reject() is None
    assert "K-major" in scope_reject(a_major="m")
    assert "K-major" in scope_reject(b_major="n")
    assert "N-major" in scope_reject(out_major="m")

    # the funnel never offers sm120 points for an out-of-scope chain
    from cudnn.gemm.frost.kernel_registry import candidates

    nmaj = analyze(_build_graph(256, 256, 128, b_major="n"))
    assert not [t for t, _c in candidates(nmaj) if t.pipeline == "sm120"]


def test_sm120_epi_vec_clamp_and_stg_only() -> None:
    """sm120 always renders the transposed-STG epilogue, with the chunk capped
    at the 8-element fragment row run; the sm100 derivation is untouched."""
    from cudnn.gemm.frost.compiler import _epi_vec_bytes, _use_tma_store_epi

    chain = analyze(_build_graph(384, 768, 384))
    for name in (
        "CONFIG_sm120_128x128x128_128x128x32_cluster1x1",
        "CONFIG_sm120_64x128x128_64x128x32_cluster1x1",
        "CONFIG_sm120_128x48x128_128x48x32_cluster1x1",
    ):
        c = by_name(name)
        v = _epi_vec_bytes(chain, c, 1)
        assert v == 16, name  # 8-element bf16 run (template: 8 % _STG_V == 0)
        assert _use_tma_store_epi(chain, c, 1) is False, name

    # the sm100 chunk derivation and its TMA-store gate are untouched
    cfg100 = by_name("CONFIG_sm100_128x128x128_128x128x32_cluster1x1")
    v100 = _epi_vec_bytes(chain, cfg100, 1)
    assert v100 >= 16 and _use_tma_store_epi(chain, cfg100, 1) is True

    # 8B-aligned output rows (N=100 bf16) narrow the chunk to 4 elements
    narrow = analyze(_build_graph(256, 100, 256))
    assert _epi_vec_bytes(narrow, by_name("CONFIG_sm120_128x128x128_128x128x32_cluster1x1"), 1) == 8


@requires_any_gpu
def test_sm120_render_smoke() -> None:
    """Render the sm120 template end-to-end (real tile constants + epilogue
    snippets) on whatever GPU is active — no cute.compile, so this covers the
    sm100 CI too. The source must be marker-free, parseable, and carry the
    STG-only sm120 contract constants."""
    from cudnn.gemm.frost.compiler import _epi_vec_bytes, _render_template
    from cudnn.gemm.frost.epilogue_codegen import generate

    chain = analyze(_build_graph(384, 768, 384))
    for name in (
        "CONFIG_sm120_128x128x128_128x128x32_cluster1x1",
        "CONFIG_sm120_128x48x128_128x48x32_cluster1x1",  # odd n-frag tail
    ):
        cfg = by_name(name)
        vec = _epi_vec_bytes(chain, cfg, 1)
        snippets = generate(chain, vec_bytes_epi=vec, output_elem_bytes=2)  # empty tma_slots == STG everywhere
        src = _render_template(chain, snippets, cfg, 1)
        assert "@@" not in src, "leftover injection markers"
        ast.parse(src)
        assert "cudnn_frost_sm120_matmul_" in src
        assert "threads_per_cta = 384" in src
        assert f"vec_bytes_epi = {vec}" in src
        assert "_STG_EPI_BYTES" in src, "transposed-STG arm must be rendered"
        assert "tma_c_desc" not in src, "TMA-store arm must be stripped"


# =============================================================================
# End-to-end correctness — needs an sm_120 GPU.
# =============================================================================


@requires_sm120
@pytest.mark.parametrize("shape", _WEIRD_SHAPES, ids=[_shape_id(s) for s in _WEIRD_SHAPES])
@pytest.mark.parametrize("in_dt,out_dt", _CORE_DTYPE_PAIRS, ids=[_dtype_id(p) for p in _CORE_DTYPE_PAIRS])
@pytest.mark.parametrize("config_name", _sweep_config_names(), ids=[_config_id(n) for n in _sweep_config_names()])
def test_sm120_matmul(
    _compile_cache,
    config_name: str,
    in_dt: str,
    out_dt: str,
    shape: tuple[int, int, int],
) -> None:
    """One (config, dtype-pair, shape); incompatible combos SKIP, else bit-tight."""
    cfg, cta_group = _resolve(config_name)
    ok, reason = _compatible(cfg, *shape, in_dt, out_dt)
    if not ok:
        pytest.skip(reason)

    compiled = _get_compiled(_compile_cache, cfg, in_dt, out_dt, cta_group)

    M, N, K = shape
    a, b, c = _mkdata(1, M, N, K, in_dt, out_dt)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()

    _assert_bit_tight(
        c,
        _reference(a, b, out_dt),
        f"config: {config_name}\n  dtype:     {in_dt} -> {out_dt}\n  shape:     {M}x{N}x{K}",
    )


@requires_sm120
@pytest.mark.parametrize("bshape", _BATCHED_SHAPES, ids=[f"B{s[0]}_{s[1]}x{s[2]}x{s[3]}" for s in _BATCHED_SHAPES])
@pytest.mark.parametrize("config_name", _BATCHED_CONFIGS, ids=[_config_id(n) for n in _BATCHED_CONFIGS])
def test_sm120_matmul_batched(_compile_cache, config_name: str, bshape) -> None:
    """Rank-3 batches ride gridDim.z; the CLC scheduler steals across planes."""
    batch, M, N, K = bshape
    in_dt = out_dt = "bf16"
    cfg, cta_group = _resolve(config_name)

    # Keyed by batch too: batch=1 bakes the degenerate-broadcast const branch
    # (matmul_a_batch == 1), so one anchor cannot serve both batch classes.
    key = ("batched", cfg.name, in_dt, out_dt, batch)
    if key in _compile_cache:
        compiled = _cached_outcome(_compile_cache[key])
    else:
        compiled = _plan_or_skip(
            _compile_cache,
            key,
            lambda: _build_batched_graph(batch, M, N, K, in_dt, out_dt),
            cfg,
            cta_group,
        )

    a, b, c = _mkdata(batch, M, N, K, in_dt, out_dt)
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()
    _assert_bit_tight(c, _reference(a, b, out_dt), f"config: {config_name}  batched {batch}x{M}x{N}x{K}")


@requires_sm120
@pytest.mark.parametrize("case", _BATCH_BROADCAST_CASES, ids=[f"broadcast{s}" for s, _ in _BATCH_BROADCAST_CASES])
def test_sm120_batch_broadcast(case) -> None:
    """One operand batch-broadcast (batch=1 input against a batch>1 GEMM)."""
    side, (batch, M, N, K) = case
    cfg, cta_group = _resolve("CONFIG_sm120_128x128x128_128x128x32_cluster1x1_1ctamma")
    a_batch = 1 if side == "A" else batch
    b_batch = 1 if side == "B" else batch
    compiled = _plan(
        _build_batched_graph(batch, M, N, K, "bf16", "bf16", a_batch=a_batch, b_batch=b_batch),
        config=cfg,
        cta_group=cta_group,
    )
    a, _, _ = _mkdata(a_batch, M, N, K, "bf16", "bf16")
    _, b, _ = _mkdata(b_batch, M, N, K, "bf16", "bf16", seed=1)
    c = torch.empty(batch, M, N, dtype=torch.bfloat16, device="cuda")
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()
    ref = torch.einsum(
        "bmk,bnk->bmn",
        a.to(torch.float32).expand(batch, -1, -1),
        b.to(torch.float32).expand(batch, -1, -1),
    ).to(torch.bfloat16)
    _assert_bit_tight(c, ref, f"batch-broadcast {side}, {batch}x{M}x{N}x{K}")


@requires_sm120
@pytest.mark.parametrize(
    "config_name",
    [
        "CONFIG_sm120_128x128x128_128x128x32_cluster1x1_1ctamma",  # aligned warp N-frags
        "CONFIG_sm120_128x48x128_128x48x32_cluster1x1_1ctamma",  # odd n-frag tail
    ],
    ids=["128x128", "128x48_tail"],
)
def test_sm120_bias_relu_epilogue(config_name: str) -> None:
    """Per-col bias + relu through the epilogue: the per-col vector aux load
    plus the predicated pair stores."""
    M, N, K = 256, 240, 128
    cfg, cta_group = _resolve(config_name)
    compiled = _plan(_build_bias_relu_graph(M, N, K), config=cfg, cta_group=cta_group)
    assert compiled.aux_names == ["bias0"]

    a, b, c = _mkdata(1, M, N, K, "bf16", "bf16")
    torch.manual_seed(1)
    bias = torch.empty(1, N, dtype=torch.int32).random_(-2, 2).to(dtype=torch.bfloat16, device="cuda")
    compiled(_vp(compiled, a, b, c, bias))
    torch.cuda.synchronize()

    ref = torch.einsum("bmk,bnk->bmn", a.to(torch.float32), b.to(torch.float32))
    ref = torch.relu(ref + bias.to(torch.float32)).to(torch.bfloat16)
    _assert_bit_tight(c, ref, f"bias+relu, {config_name}, {M}x{N}x{K}")


@requires_sm120
def test_sm120_narrow_output_rows() -> None:
    """8B-aligned output rows (N=100 bf16) narrow the chunk to 4 elements and
    still store correct predicated pairs."""
    M, N, K = 256, 100, 256
    cfg, cta_group = _resolve("CONFIG_sm120_128x128x128_128x128x32_cluster1x1_1ctamma")
    compiled = _plan(_build_graph(M, N, K), config=cfg, cta_group=cta_group)
    a, b, c = _mkdata(1, M, N, K, "bf16", "bf16")
    compiled(_vp(compiled, a, b, c))
    torch.cuda.synchronize()
    _assert_bit_tight(c, _reference(a, b, "bf16"), f"narrow output rows, {M}x{N}x{K}")


@requires_sm120
def test_sm120_out_of_scope_jit_rejects() -> None:
    """An out-of-scope chain raises NotImplementedError from the jit gates —
    never an AssertionError from the template's render asserts."""
    from cudnn.gemm.frost.compiler import jit_from_cudnn_graph

    cfg = by_name("CONFIG_sm120_128x128x128_128x128x32_cluster1x1")
    with pytest.raises(NotImplementedError, match="K-major"):
        jit_from_cudnn_graph(_build_graph(256, 256, 128, b_major="n"), config=cfg, cta_group=1)
    with pytest.raises(NotImplementedError, match="N-major"):
        jit_from_cudnn_graph(_build_graph(256, 256, 128, out_major="m"), config=cfg, cta_group=1)


@requires_sm120
def test_sm120_probe_and_auto_select() -> None:
    """The engine path: probe accepts with its DEFAULT arguments (the sm100
    default geometry is re-targeted to the family the auto path builds), and
    build_gemm_plan compiles + runs an sm120 kernel without a caller config."""
    from cudnn.gemm.frost.compiler import probe_supported
    from cudnn.gemm.frost.graph_analyzer import build_gemm_plan
    from cudnn.gemm.frost.kernel_registry import preferred_strategy
    from cudnn.gemm.frost.tile_config import DEFAULT_CONFIG

    M, N, K = 384, 768, 384
    g = _build_graph(M, N, K)
    cfg, grp = preferred_strategy(analyze(g), DEFAULT_CONFIG, 2)
    assert cfg.pipeline == "sm120" and grp == 1  # warp MMA: cta_group clamps to 1
    probe_supported(g)  # must not raise on an sm_120 GPU

    compiled = build_gemm_plan(g)
    assert compiled.config.pipeline == "sm120"
    a, b, c = _mkdata(1, M, N, K, "bf16", "bf16")
    bd = compiled.binding
    compiled({bd.a_operands[0]: a, bd.b_operands[0]: b, bd.outputs[0]: c})
    torch.cuda.synchronize()
    _assert_bit_tight(c, _reference(a, b, "bf16"), f"auto-select ({compiled.config.name}), {M}x{N}x{K}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
