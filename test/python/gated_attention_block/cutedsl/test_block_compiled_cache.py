# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The block's forward kernels through the persistent compiled-plan cache.

Every compile site of ``kernels/elementwise.py``, ``kernels/qk_norm_rope_tma.py``, ``kernels/cache_write.py`` (the serving
write-through) and the three forks of the rendered GEMM template (``kernels/proj_gemm_norm_rope.py`` + ``_fp8`` / ``_mxfp8``) goes
through ``cudnn.frost.compiled_cache.compile_cached``:
the module dict each site already kept stays the in-process memo, and behind it the on-disk cache hands the next process the
exported object instead of re-tracing and re-running ptxas.  Keys follow ``compiled_cache.template_key``: the module's source
digest (the template loader's file + params digest for the forks, a file digest for the three plainly imported modules) joined with
the function tag and the dict key's plain values (``None`` -- not cacheable -- for a tensor / device / stream, so the keys stay the
Constexpr tuples the dicts use).

The host tests pin the keys, the memo and the two documented switches with ``cute.compile`` stubbed; the Rubin test runs
every site in a second process and asserts it reloads every kernel and writes the same bytes.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

from cudnn.frost import compiled_cache as cc  # noqa: E402
from cudnn.gated_attention_block.kernels import elementwise as kern_ew  # noqa: E402
from cudnn.gated_attention_block.kernels import proj_gemm as kern_pg  # noqa: E402
from cudnn.gated_attention_block.kernels import qk_norm_rope_tma as kern_tma  # noqa: E402

pytestmark = pytest.mark.L0
requires_rubin = pytest.mark.requires_rubin  # the suite conftest's registered marker: an accept test, skipped off SM107

# The serving cache-write kernel (``kernels/cache_write.py``, the paged-KV serving form's write-through) is a separate change: its
# site is pinned here exactly where the module is in the tree and skipped BY NAME where it is not -- the pins turn on the day the
# kernel lands and never leave this suite red on a tree without it.  The probe is the module's PRESENCE (``find_spec``), never a
# swallowed ImportError: a tree that carries the kernel but cannot import it fails here loudly instead of skipping.
_CACHE_WRITE_MODULE = "cudnn.gated_attention_block.kernels.cache_write"
kern_cw = importlib.import_module(_CACHE_WRITE_MODULE) if importlib.util.find_spec(_CACHE_WRITE_MODULE) is not None else None
requires_cache_write_kernel = pytest.mark.skipif(kern_cw is None, reason="kernels/cache_write.py (the serving cache-write kernel) is not in this tree")

_KERNELS = Path(kern_ew.__file__).parent
# file -> (the function tag template_key joins, the exported symbol)
_SITES = {
    "elementwise.py": ("compile_elementwise_gate", "cudnn_block_elementwise_gate"),
    "qk_norm_rope_tma.py": ("compile_qk_norm_rope_tma", "cudnn_block_qk_norm_rope_tma"),
    "proj_gemm_norm_rope.py": ("compile", "frost_gemm_norm_rope"),
    "proj_gemm_norm_rope_fp8.py": ("compile", "frost_gemm_norm_rope_fp8"),
    "proj_gemm_norm_rope_mxfp8.py": ("compile", "frost_gemm_norm_rope_mxfp8"),
}
if kern_cw is not None:
    _SITES["cache_write.py"] = ("compile_cache_write", "cudnn_block_cache_write")
_CHILD = str(Path(__file__).with_name("block_compiled_cache_child.py"))
# every arm the child runs; the elementwise, TMA and cache-write arms share their module's symbol, each is its own compiled object
_CHILD_SITES = (
    "elementwise_gate",
    "elementwise_copy",
    "qk_tma_norm",
    "qk_tma_rope_only",
    "qk_tma_fp8",
    "qk_tma_mx",
    "fork_bf16",
    "fork_fp8",
    "fork_mxfp8",
) + (("cache_write_kv", "cache_write_index") if kern_cw is not None else ())


@pytest.fixture(autouse=True)
def _fresh_memo():
    """Every test starts from an empty in-process memo (and leaves one): an object memoised under a stubbed compile must not
    reach a later test, and an earlier test's real object must not serve a stubbed key here."""
    cc.clear_memo()
    cc.reset_stats()
    yield
    cc.clear_memo()


def _recorder(calls):
    """A stand-in for ``compile_cached`` that records its call and mints a fresh object (no trace, no files)."""

    def fake(fn, *args, cache_key, symbol="kernel", **kwargs):
        calls.append(dict(fn=fn, args=args, cache_key=cache_key, symbol=symbol, options=kwargs.get("options")))
        return object()

    return fake


def _stub_cute_compile(monkeypatch, calls):
    import cutlass.cute as cute

    monkeypatch.setattr(cute, "compile", lambda fn, *a, **k: calls.append((fn, a, k)) or object())


# ---------------------------------------------------------------------------------------------------------------------------------
# Static detector: every site is routed, none is bare
# ---------------------------------------------------------------------------------------------------------------------------------


def test_every_block_compile_site_goes_through_the_compiled_plan_cache():
    """No bare ``cute.compile(`` is left in the kernel files (the five develop sites, plus the cache write where its module is in the
    tree); each passes ``compile_cached`` its key and symbol."""
    for name, (tag, symbol) in _SITES.items():
        src = (_KERNELS / name).read_text()
        assert "cute.compile(" not in src, f"{name}: a bare cute.compile( site bypasses the compiled-plan cache"
        assert "compile_cached(" in src and f'symbol="{symbol}"' in src and f'"{tag}")' in src, f"{name}: the site must pass cache_key= and symbol={symbol!r}"
    for mod in (m for m in (kern_ew, kern_tma, kern_cw) if m is not None):
        # imported normally, so each names its own source identity (the loader stamps the forks')
        assert mod.FROST_SOURCE_DIGEST == hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------------------
# Keys: the module digest + the function tag + the memo dict's key
# ---------------------------------------------------------------------------------------------------------------------------------


def test_the_plain_modules_key_on_their_file_digest_and_the_memo_dict_key(monkeypatch):
    """``elementwise`` and ``qk_norm_rope_tma`` hand ``compile_cached`` ``template_key(globals(), {"key": <the dict key>}, <tag>)``
    -- the file digest, the tag, and exactly the Constexpr tuple the dict memoises under -- with the site's symbol and options."""
    calls = []
    monkeypatch.setattr(kern_ew, "compile_cached", _recorder(calls))
    monkeypatch.setattr(kern_ew, "compiled_cache", {})
    monkeypatch.setattr(kern_ew, "current_device", lambda: 0)
    kern_ew.compile_elementwise_gate(dtype=torch.bfloat16, h=8, d=256, has_gate=True)
    (key,) = kern_ew.compiled_cache.keys()
    assert key == ("torch.bfloat16", 8, 256, True, 128, 2, True, False, 0)
    assert calls[0]["cache_key"] == f"{kern_ew.FROST_SOURCE_DIGEST}|compile_elementwise_gate|{[('key', key)]!r}"
    assert calls[0]["cache_key"] == cc.template_key(vars(kern_ew), {"key": key}, "compile_elementwise_gate")
    assert (calls[0]["symbol"], calls[0]["options"]) == ("cudnn_block_elementwise_gate", "--enable-tvm-ffi")
    assert kern_ew.compile_elementwise_gate(dtype=torch.bfloat16, h=8, d=256, has_gate=True).compiled is kern_ew.compiled_cache[key]
    assert len(calls) == 1, "a repeat request is served by the module dict, not by another compile_cached call"

    calls = []
    monkeypatch.setattr(kern_tma, "compile_cached", _recorder(calls))
    monkeypatch.setattr(kern_tma, "compiled_cache", {})
    monkeypatch.setattr(kern_tma, "current_device", lambda: 0)
    r = kern_tma.compile_qk_norm_rope_tma(dtype=torch.bfloat16, h_q=8, h_kv=2, d=256, rope_dim=64, eps=1e-6, want_rstd=False, tile_rows=8)
    (key,) = kern_tma.compiled_cache.keys()
    assert key[:6] == ("torch.bfloat16", 8, 2, 256, 64, 8) and key[-4:] == (0, True, False, False)  # ..., device, apply_norm, fp8_out, mx_out
    assert len(key) == 16 and all(isinstance(v, (str, int, bool)) for v in key)
    assert calls[0]["cache_key"] == cc.template_key(vars(kern_tma), {"key": key}, "compile_qk_norm_rope_tma")
    assert calls[0]["cache_key"].startswith(f"{kern_tma.FROST_SOURCE_DIGEST}|compile_qk_norm_rope_tma|[('key', ('torch.bfloat16', 8, 2, 256, 64, 8, ")
    assert (calls[0]["symbol"], calls[0]["options"]) == ("cudnn_block_qk_norm_rope_tma", "--enable-tvm-ffi")
    assert r.compiled is kern_tma.compiled_cache[key]


@requires_cache_write_kernel
def test_the_cache_write_site_keys_on_its_file_digest_and_the_memo_dict_key(monkeypatch):
    """``cache_write`` (the serving write-through) hands ``compile_cached`` ``template_key(globals(), {"key": <the dict key>},
    "compile_cache_write")`` -- the file digest, the tag, and exactly the Constexpr tuple the dict memoises under (the slot dtype and
    the operand count included: one artifact per value, never a shared one) -- with the site's symbol and options."""
    calls = []
    monkeypatch.setattr(kern_cw, "compile_cached", _recorder(calls))
    monkeypatch.setattr(kern_cw, "compiled_cache", {})
    monkeypatch.setattr(kern_cw, "current_device", lambda: 0)
    kv = dict(dtype=torch.bfloat16, h=2, d=256, page_size=16, slot_dtype=torch.int64)
    r = kern_cw.compile_cache_write(**kv)
    (key,) = kern_cw.compiled_cache.keys()
    assert key == ("torch.bfloat16", 2, 256, 16, "torch.int64", True, 128, 2, True, False, 0)
    assert all(isinstance(v, (str, int, bool)) for v in key)
    assert calls[0]["cache_key"] == f"{kern_cw.FROST_SOURCE_DIGEST}|compile_cache_write|{[('key', key)]!r}"
    assert calls[0]["cache_key"] == cc.template_key(vars(kern_cw), {"key": key}, "compile_cache_write")
    assert (calls[0]["symbol"], calls[0]["options"]) == ("cudnn_block_cache_write", "--enable-tvm-ffi")
    assert r.compiled is kern_cw.compiled_cache[key]
    assert kern_cw.compile_cache_write(**kv).compiled is r.compiled
    assert len(calls) == 1, "a repeat request is served by the module dict, not by another compile_cached call"
    kern_cw.compile_cache_write(**{**kv, "slot_dtype": torch.int32})
    kern_cw.compile_cache_write(dtype=torch.bfloat16, h=1, d=128, page_size=16, slot_dtype=torch.int64, two_operands=False)
    assert len(calls) == 3 and len({c["cache_key"] for c in calls}) == 3 and len(kern_cw.compiled_cache) == 3


@pytest.mark.parametrize(
    "template, params, symbol",
    [
        (kern_pg._FUSED_TEMPLATE, dict(), "frost_gemm_norm_rope"),
        (kern_pg._FUSED_TEMPLATE_FP8, dict(quant_fp8=True), "frost_gemm_norm_rope_fp8"),
        (kern_pg._FUSED_TEMPLATE_MXFP8, dict(quant_mxfp8=True), "frost_gemm_norm_rope_mxfp8"),
    ],
    ids=["bf16", "fp8", "mxfp8"],
)
def test_the_forks_key_on_the_loader_digest(monkeypatch, template, params, symbol):
    """A fork loaded through the template loader carries ``FROST_SOURCE_DIGEST`` (the file + its params) and ``compile()`` hands
    ``compile_cached`` ``<digest>|compile|[]`` with the fork's symbol; the SAME file imported plainly carries no digest, so its
    key is ``None`` -- not cacheable -- and it compiles as before."""
    from cudnn.frost.template_loader import load_template

    # params no other test loads, so the lru_cache'd compile() stub below can never be served to a real caller; cleared anyway
    p = kern_pg.NormRopeFusionParams(d_head=256, rope_dim=64, h_q=6, h_kv=3, eps=1e-6, **params)
    mod = load_template(template, p, tag="proj_gemm_norm_rope_cachekey")
    assert mod.FROST_SOURCE_DIGEST == hashlib.sha256(Path(template).read_bytes() + repr(p).encode("utf-8")).hexdigest()[:16]
    calls = []
    monkeypatch.setattr(mod, "_compile_cached", _recorder(calls))
    try:
        mod.compile()
        assert calls[0]["cache_key"] == f"{mod.FROST_SOURCE_DIGEST}|compile|[]" == cc.template_key(vars(mod), {}, "compile")
        assert calls[0]["symbol"] == symbol and "--enable-tvm-ffi" in calls[0]["options"]
    finally:
        mod.compile.cache_clear()
    plain = importlib.import_module("cudnn.gated_attention_block.kernels." + Path(template).stem)
    assert "FROST_SOURCE_DIGEST" not in vars(plain)
    calls = []
    monkeypatch.setattr(plain, "_compile_cached", _recorder(calls))
    try:
        plain.compile()
        assert calls[0]["cache_key"] is None and calls[0]["symbol"] == symbol
    finally:
        plain.compile.cache_clear()


# ---------------------------------------------------------------------------------------------------------------------------------
# The two documented switches, and the dict as the first memo
# ---------------------------------------------------------------------------------------------------------------------------------


def test_the_disable_switch_compiles_as_before_and_persists_nothing(monkeypatch, tmp_path):
    """``CUDNN_FRONTEND_DISABLE_COMPILED_CACHE=1``: the site reaches ``cute.compile`` directly (counted ``bypassed``), the cache root
    stays empty, and the object is still memoised in-process (``compile_cached`` hands a key's object out again whatever the files'
    state); with the switch off the same request is a counted miss that still writes nothing for an object that cannot be exported."""
    calls = []
    _stub_cute_compile(monkeypatch, calls)
    monkeypatch.setattr(kern_ew, "compiled_cache", {})
    monkeypatch.setattr(kern_ew, "current_device", lambda: 0)
    monkeypatch.setattr(cc, "_dir_override", None)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    gate = dict(dtype=torch.bfloat16, h=8, d=256, has_gate=True)
    kern_ew.compile_elementwise_gate(**gate)
    assert len(calls) == 1 and cc.stats()["bypassed"] == 1 and cc.stats()["misses"] == 0
    assert not any(tmp_path.iterdir()), "a disabled cache must not touch its root"
    kern_ew.compiled_cache.clear()
    kern_ew.compile_elementwise_gate(**gate)
    assert len(calls) == 1 and cc.stats()["memo_hits"] == 1, "a bypassed object is memoised too: a dict miss of a known key is no trace"

    monkeypatch.delenv(cc._ENV_DISABLE)
    cc.clear_memo()
    if any(v == "unknown" for v in cc.environment_manifest().values()):
        pytest.skip("no CUDA device: the manifest cannot identify this environment, so the cache bypasses by design")
    kern_ew.compiled_cache.clear()
    kern_ew.compile_elementwise_gate(**gate)
    assert len(calls) == 2 and cc.stats()["misses"] == 1, "with the switch off the site goes to the files and misses"
    assert not list(tmp_path.rglob("entry.json")), "a stub object has no export_to_c: nothing is committed"


def test_the_memo_switch_off_lets_a_trace_time_spy_see_every_dict_miss(monkeypatch):
    """The module dict stays the FIRST memo (a repeat key never reaches ``compile_cached``); behind it, with
    ``CUDNN_FRONTEND_COMPILED_CACHE_INPROCESS_MEMO=0`` every dict miss is a real trace (a spy on ``cute.compile`` counts one call per
    miss), and with the memo on a dict miss of a key this process already compiled is served from the memo without a trace."""
    calls = []
    _stub_cute_compile(monkeypatch, calls)
    monkeypatch.setattr(kern_ew, "compiled_cache", {})
    monkeypatch.setattr(kern_ew, "current_device", lambda: 0)
    monkeypatch.setenv(cc._ENV_DISABLE, "1")
    monkeypatch.setenv(cc._ENV_MEMO, "0")
    gate = dict(dtype=torch.bfloat16, h=8, d=256, has_gate=True)
    kern_ew.compile_elementwise_gate(**gate)
    kern_ew.compile_elementwise_gate(**{**gate, "has_gate": False})
    assert len(calls) == 2, "two keys, two traces"
    kern_ew.compile_elementwise_gate(**gate)
    assert len(calls) == 2, "a repeat key is the module dict's hit"
    kern_ew.compiled_cache.clear()
    kern_ew.compile_elementwise_gate(**gate)
    assert len(calls) == 3 and cc.stats()["memo_hits"] == 0, "memo off: a dict miss is a trace the spy sees"

    monkeypatch.delenv(cc._ENV_MEMO)
    cc.clear_memo()
    kern_ew.compiled_cache.clear()
    first = kern_ew.compile_elementwise_gate(**gate).compiled
    kern_ew.compiled_cache.clear()
    again = kern_ew.compile_elementwise_gate(**gate).compiled
    assert len(calls) == 4 and again is first and cc.stats()["memo_hits"] == 1, "memo on: the dict miss is served from the memo, no trace"


# ---------------------------------------------------------------------------------------------------------------------------------
# Rubin: a second process reloads every kernel and writes the same bytes
# ---------------------------------------------------------------------------------------------------------------------------------


def _run_child(cache_dir):
    env = dict(os.environ, CUDNN_FRONTEND_COMPILED_CACHE=str(cache_dir))
    env.pop(cc._ENV_DISABLE, None)  # a functional run's default may have switched the cache off
    env.pop(cc._ENV_MEMO, None)
    out = subprocess.run([sys.executable, _CHILD], env=env, capture_output=True, text=True, timeout=1500)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@requires_rubin
def test_a_second_process_reloads_every_block_kernel_and_writes_the_same_bytes(tmp_path):
    """Process one compiles every site (one miss each, nothing bypassed, every export committed); process two reloads every one of
    them (one hit each, no miss) and every output digest is bitwise the first process's.  The records on disk name the sites."""
    first = _run_child(tmp_path)
    assert set(first["sites"]) == set(_CHILD_SITES)
    for site, rec in first["sites"].items():
        s = rec["stats"]
        assert (s["misses"], s["hits"], s["bypassed"], s["export_failed"]) == (1, 0, 0, 0), (site, s)
    records = [json.loads(p.read_text()) for p in (tmp_path / cc._SCHEMA).glob("*/*/entry.json")]
    assert len(records) == len(_CHILD_SITES), "one committed entry per compiled arm"
    assert {r["symbol"] for r in records} == {symbol for _, symbol in _SITES.values()}
    for r in records:
        tag = {symbol: tag for tag, symbol in _SITES.values()}[r["symbol"]]
        assert f"|{tag}|" in r["key"], r

    second = _run_child(tmp_path)
    for site, rec in second["sites"].items():
        s = rec["stats"]
        assert (s["misses"], s["hits"], s["bypassed"]) == (0, 1, 0), (site, s)
        assert rec["digest"] == first["sites"][site]["digest"], f"{site}: the reloaded kernel must write exactly what the compiled one did"
