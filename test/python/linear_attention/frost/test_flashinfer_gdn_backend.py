# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FlashInfer backend ``"frost_fe"`` of the fused GDN decode and verify steps (``cudnn.linear_attention.frost.flashinfer_gdn_backend``).
Skipped without FlashInfer's experimental Qwen 3.5 GDN block.  Host: the import-time registration (idempotent, ``replace``), the typed
declines through FlashInfer's own probe (form, geometry by field, device by compute capability, the DSL gate, the decode row envelope,
the verify step-shape envelope: tokens per request against the pick's shared-memory budget on a cc 10.0 reading and on a 99 KB one, a
part whose shared memory holds no verify form), the build-time
dtype / device declines, both runners' argument mapping onto their entry's call contract and their refusal of an unprepared binding, and
a subprocess proof that the probe loads no kernel, and the per-row trip of the docs page's served / declined table through both twins
(the probe and the entries' own ``check_support``).  Device (an SM100- or SM120-class part, cc 10.x / 12.x, with a serving DSL): FlashInfer's OWN decode and verify
tests -- the torch reference at decode rows 1 .. 64 and at verify requests 1 / 2 / 4 x tokens 4 / 6, the CUDA-graph replays, the
dim-first conv pool (served by decode, a typed decline for verify) -- run against ``frost_fe`` (FlashInfer's checks, the cuDNN Frontend backend), the verify
reference under the 148-SM launch policy and with the two-CTAs-per-SM form forced at every shape, the fp32-parameter decode decline at
plan time, the capture guards, and bitwise agreement with FlashInfer's built-in ``frost`` backend where that one is served (cc 10.0):
decode on its reference cases, verify on the benchmark's own input sets at requests 1 / 2 / 4 x 6 tokens (all accepted, and random
acceptance with a padding request).  The FlashInfer tests and benchmark are imported from the source tree next to the installed
package; a wheel ships none, and those cells skip."""

import functools
import importlib
import importlib.util
import subprocess
import sys
import textwrap
import types
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")
fi_api = pytest.importorskip("flashinfer.experimental.qwen_3_5_gdn_block.api", reason="FlashInfer with the experimental Qwen 3.5 GDN block is not installed")
fi_block = pytest.importorskip("flashinfer.qwen_3_5_gdn_block", reason="FlashInfer's public qwen_3_5_gdn_block shim is not installed")

from flashinfer.experimental.qwen_3_5_gdn_block import support as fi_support  # noqa: E402

Qwen35GdnGeometry = fi_block.Qwen35GdnGeometry
prepare_qwen_3_5_gdn_block = fi_block.prepare_qwen_3_5_gdn_block
qwen_3_5_gdn_block_decode = fi_block.qwen_3_5_gdn_block_decode
qwen_3_5_gdn_block_support = fi_block.qwen_3_5_gdn_block_support
qwen_3_5_gdn_block_verify = fi_block.qwen_3_5_gdn_block_verify

from cudnn.linear_attention.frost import flashinfer_gdn_backend as backend  # noqa: E402  (the import registers "frost_fe")
from cudnn.linear_attention.frost import gdn_decode_fused as entry  # noqa: E402
from cudnn.linear_attention.frost import gdn_verify_fused as ventry  # noqa: E402
from cudnn.linear_attention.frost.kernel import gdn_verify_fused_config as kvcfg  # noqa: E402

pytestmark = [pytest.mark.L0]

GEOMETRY = Qwen35GdnGeometry(**backend.GEOMETRY)
CPU = torch.device("cpu")
F32, BF16 = torch.float32, torch.bfloat16
ROWS = [1, 2, 3, 4, 8, 16, 64]  # FlashInfer's decode rows (the last row a padding row from two rows on)
VERIFY_SHAPES = [(1, 4), (1, 6), (2, 4), (2, 6), (4, 4), (4, 6)]  # FlashInfer's verify (requests, tokens per request)
SM100_READINGS = dict(SMEM_PER_SM_BYTES=233472, SMEM_OPTIN_BYTES=232448)  # a cc 10.0 part: 228 KB per SM, 227 KB per-CTA opt-in
SM100_SMS = 148
SM120_READINGS = dict(
    SMEM_PER_SM_BYTES=kvcfg.SM120_SMEM_PER_SM_BYTES, SMEM_OPTIN_BYTES=kvcfg.SM120_SMEM_OPTIN_BYTES
)  # a cc 12.0 part: 100 KB per SM, 99 KB opt-in
SM120_SMS = 170
POLICY_SMS = {"148 SMs": SM100_SMS, "two CTAs per SM": 47}  # the SM-count input of the entry's pick per launch policy
_REAL_VERIFY_FORM = backend.verify_form  # the unpatched probe path, for cells that patch it to a host twin and back


def _probe(device, form="decode", **kw):
    return qwen_3_5_gdn_block_support(device, GEOMETRY, form, backend.BACKEND, **kw)


def _spec(device=CPU, rows=2, param_dtypes=(F32, BF16, BF16), form="decode", tokens=1):
    """A plan spec; for ``verify`` ``rows`` is the requests count and ``tokens`` the tokens per request."""
    return fi_api.Qwen35GdnPlanSpec(backend.BACKEND, form, rows, tokens, GEOMETRY, torch.device(device), tuple(param_dtypes))


def _tensors(rows, device, slots=None, seed=0):
    """One decode call's tensors in FlashInfer's serving layout: ``qkv`` / ``z`` views of one QKVZ slab, the conv pool channel-innermost
    (``[slots, 3, C]`` transposed to the ``[slots, C, 3]`` view), the kernel-native parameter dtypes."""
    slots = rows + 1 if slots is None else slots
    g = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g).to(device)

    qkvz = (rn(rows, 16384) * 2).bfloat16()
    return dict(
        qkv=qkvz[:, :10240],
        z=qkvz[:, 10240:].view(rows, 48, 128),
        hidden=rn(rows, 5120).bfloat16(),
        ba_weight=(rn(96, 5120) / 5120**0.5).bfloat16(),
        A_log=rn(48) * 0.5,
        dt_bias=(rn(48) * 0.5).bfloat16(),
        ssm_state=rn(slots, 48, 128, 128) * 0.1,
        state_indices=torch.arange(1, rows + 1, dtype=torch.int32).to(device),
        norm_weight=(1 + 0.2 * rn(128)).bfloat16(),
        out_scale=torch.tensor([0.02], device=device),
        conv_state=rn(slots, 3, 10240).bfloat16().transpose(1, 2),
        conv_weight=(0.3 * rn(10240, 4)).bfloat16(),
        out=torch.empty(rows, 48, 128, dtype=torch.float8_e4m3fn, device=device),
    )


def _verify_tensors(requests, tokens, device, seed=0):
    """One verify call's tensors in FlashInfer's layout: ``state_indices`` ``[requests, T]``, the conv pool channel-innermost (``[slots, T + 2,
    C]`` transposed to the ``[slots, C, T + 2]`` view), ``cu_seqlens`` / ``num_accepted_tokens`` int32, every request fully accepted."""
    rows = requests * tokens
    t = _tensors(rows, device, slots=rows + 1, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    t["conv_state"] = torch.randn(rows + 1, tokens + 2, 10240, generator=g).to(device).bfloat16().transpose(1, 2)
    t["state_indices"] = torch.arange(1, rows + 1, dtype=torch.int32).view(requests, tokens).to(device)
    t["cu_seqlens"] = torch.arange(0, rows + 1, tokens, dtype=torch.int32).to(device)
    t["num_accepted_tokens"] = torch.full((requests,), tokens, dtype=torch.int32).to(device)
    return t


def _part_verify_form(readings, sms, requests, tokens_per_request):
    """The verify form a part (``readings``, ``sms``) picks for a step shape, for a host without one: the module pick with that part's
    readings, resolved (no requests count = the pick at one request, as the backend reads it)."""
    requests = 1 if requests is None else int(requests)
    cfg = kvcfg.default_cfg(requests, sms, readings["SMEM_PER_SM_BYTES"], readings["SMEM_OPTIN_BYTES"], tokens_per_request)
    return kvcfg.resolve_cfg(replace(cfg, **readings), requests)


def _sm100_verify_form(device, requests=None, norm_eps=1e-6, tokens_per_request=None):
    """The verify form a cc 10.0 part with 148 SMs picks (``verify_form``'s twin for a host without one)."""
    return _part_verify_form(SM100_READINGS, SM100_SMS, requests, tokens_per_request)


def _sm120_verify_form(device, requests=None, norm_eps=1e-6, tokens_per_request=None):
    """The verify form a 99 KB-opt-in part with 170 SMs (cc 12.0) picks for a step shape (``verify_form``'s twin for a host without one)."""
    return _part_verify_form(SM120_READINGS, SM120_SMS, requests, tokens_per_request)


# ---------------------------------------------------------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------------------------------------------------------
def test_import_registered_the_backend_once():
    reg = fi_api.get_backend(backend.BACKEND)
    assert reg is not None and reg.support_fn is backend.support and reg.build_fn is backend.build
    assert backend.register_frost_fe() == backend.BACKEND and fi_api.get_backend(backend.BACKEND) is reg, "a second import / call is a no-op"
    assert backend.register_frost_fe(replace=True) == backend.BACKEND
    again = fi_api.get_backend(backend.BACKEND)
    assert again is not reg and again.build_fn is backend.build and again.support_fn is backend.support
    with pytest.raises(ValueError, match="already registered"):
        fi_api.register_backend(backend.BACKEND, backend.support, lambda spec: None)  # a foreign backend does not take the name silently
    assert fi_api.get_backend(backend.BACKEND) is again
    assert backend.geometry_off(GEOMETRY) == {} and GEOMETRY.conv_dim == 10240 and GEOMETRY.conv_state_len(1) == 3 and GEOMETRY.conv_state_len(6) == 8
    assert backend.FORMS == fi_support.FORMS == ("decode", "verify")


def test_register_names_the_flashinfer_floor_without_the_registry(monkeypatch):
    """A FlashInfer before the block's backend registry (PR #8131) is an ImportError naming that floor, not an AttributeError."""
    assert "8131" in backend.FLASHINFER_FLOOR
    monkeypatch.delattr(fi_api, "register_backend")
    with pytest.raises(ImportError, match="8131"):
        backend.register_frost_fe(replace=True)
    assert fi_api.get_backend(backend.BACKEND) is not None, "the registration in place stays"


def test_an_unknown_form_is_declined_by_kind_form():
    """FlashInfer's own gate stops an unknown form before any backend sees it; the backend's own answer has the same kind."""
    sup = _probe(CPU, "prefill")
    assert (sup.served, sup.kind) == (False, "form"), sup
    decline = backend.support(CPU, GEOMETRY, "prefill", None, None)
    assert (decline.kind, "serves the forms" in decline.reason) == ("form", True), decline
    with pytest.raises(NotImplementedError, match="serves the forms"):
        backend.build(_spec(form="prefill"))


@pytest.fixture
def as_sm100(monkeypatch):
    """The support question as it reads on a cc 10.0 part (148 SMs) with a serving DSL: the device fact FlashInfer's helper reads, both
    entries' DSL gates and the verify form's device readings patched, nothing else."""
    monkeypatch.setattr(fi_support, "device_cc", lambda device: (10, 0))
    monkeypatch.setattr(entry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(ventry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(backend, "verify_form", _sm100_verify_form)


@pytest.fixture
def as_sm120(monkeypatch):
    """The support question as it reads on a cc 12.0 part (170 SMs, 100 KB of shared memory per SM, 99 KB per-CTA opt-in) with a serving
    DSL: the device fact FlashInfer's helper reads, the DSL gates, and the verify entry's DEVICE READERS patched to that part -- the real
    ``verify_form`` / ``verify_max_tokens`` run on the host, picking through the real entry."""
    monkeypatch.setattr(fi_support, "device_cc", lambda device: (12, 0))
    monkeypatch.setattr(entry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(ventry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(ventry, "device_shared_memory", lambda device: (SM120_READINGS["SMEM_PER_SM_BYTES"], SM120_READINGS["SMEM_OPTIN_BYTES"]))
    monkeypatch.setattr(ventry, "device_multiprocessor_count", lambda device: SM120_SMS)


@pytest.mark.parametrize("form", backend.FORMS)
def test_other_geometries_are_declined_by_field_name(as_sm100, form):
    kw = dict(tokens_per_request=6) if form == "verify" else {}
    for field, value in (("d_model", 4096), ("num_v_heads", 32), ("num_k_heads", 8), ("head_k_dim", 64), ("head_v_dim", 64), ("conv_kernel", 3)):
        g = Qwen35GdnGeometry(**dict(backend.GEOMETRY, **{field: value}))
        assert backend.geometry_off(g) == {field: value}
        sup = qwen_3_5_gdn_block_support("cuda:0", g, form, backend.BACKEND, **kw)
        assert (sup.served, sup.kind) == (False, "geometry") and field in sup.reason and form in sup.reason, (field, sup)
    assert _probe("cuda:0", form, **kw).served, "the kernels' geometry with another norm_eps is served (the epsilon is compiled into a plan)"
    assert qwen_3_5_gdn_block_support("cuda:0", Qwen35GdnGeometry(**backend.GEOMETRY, norm_eps=1e-5), form, backend.BACKEND, **kw).served


@pytest.mark.parametrize("form", backend.FORMS)
def test_devices_are_declined_by_compute_capability(monkeypatch, form):
    monkeypatch.setattr(entry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(ventry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(backend, "verify_form", _sm100_verify_form)
    kw = dict(tokens_per_request=6) if form == "verify" else {}
    sup = _probe(CPU, form, **kw)  # the real probe on a CPU device
    assert (sup.served, sup.kind) == (False, "device") and "no CUDA device" in sup.reason and form in sup.reason, sup
    for cc, token in (((8, 0), "SM80"), ((9, 0), "SM90"), ((11, 0), "SM110"), ((13, 0), "SM130")):
        monkeypatch.setattr(fi_support, "device_cc", lambda device, cc=cc: cc)
        sup = _probe("cuda:0", form, **kw)
        assert (sup.served, sup.kind) == (False, "device") and token in sup.reason and "10.x" in sup.reason and "12.x" in sup.reason, sup
    assert entry.SUPPORTED_CC_RANGES == ventry.SUPPORTED_CC_RANGES == ((100, 109), (120, 129))
    assert entry.SUPPORTED_CC_TEXT == ventry.SUPPORTED_CC_TEXT and not hasattr(entry, "SUPPORTED_CC_MAJOR") and not hasattr(ventry, "SUPPORTED_CC_MAJOR")
    for cc in ((10, 0), (10, 3), (10, 7), (12, 0), (12, 1)):
        monkeypatch.setattr(fi_support, "device_cc", lambda device, cc=cc: cc)
        assert _probe("cuda:0", form, **kw).served, cc


@pytest.mark.parametrize("form", backend.FORMS)
def test_dsl_gate_is_a_dsl_decline(monkeypatch, form):
    monkeypatch.setattr(fi_support, "device_cc", lambda device: (10, 0))
    monkeypatch.setattr(backend, "verify_form", _sm100_verify_form)
    name = "decode" if form == "decode" else "verify"
    monkeypatch.setattr(
        entry if form == "decode" else ventry, "_dsl_version_error", lambda: f"the fused GDN {name} step requires nvidia-cutlass-dsl >= 4.8.0; found 4.7.1"
    )
    sup = _probe("cuda:0", form, **(dict(tokens_per_request=6) if form == "verify" else {}))
    assert (sup.served, sup.kind) == (False, "dsl") and "4.8.0" in sup.reason and "4.7.1" in sup.reason, sup


def test_row_envelope(as_sm100):
    assert backend.DECODE_MAX_ROWS == entry.MAX_ROWS == 65535, "one source: the entry's own bound (its check_support declines past it)"
    assert _probe("cuda:0").served and _probe("cuda:0", requests=1).served
    assert _probe("cuda:0", requests=backend.DECODE_MAX_ROWS, tokens_per_request=1).served
    sup = _probe("cuda:0", requests=backend.DECODE_MAX_ROWS + 1)
    assert (sup.served, sup.kind) == (False, "shape") and str(backend.DECODE_MAX_ROWS) in sup.reason, sup
    sup = _probe("cuda:0", requests=4, tokens_per_request=2)  # FlashInfer's own gate: decode takes one token per request
    assert (sup.served, sup.kind) == (False, "shape"), sup


def test_verify_step_shape_envelope(as_sm100):
    """The verify form on a 148-SM cc 10.0 reading: tokens per request from 2 up to the picked form's shared-memory budget (the cluster
    pair at one request holds 35, the one-CTA form at requests 2 .. 3 holds 8, the two-CTAs-per-SM form from 4 requests on holds 45), one
    grid row per request up to 65535."""
    pair, one_cta, two_cta = _sm100_verify_form("cuda:0", 1), _sm100_verify_form("cuda:0", 3), _sm100_verify_form("cuda:0", 4)
    assert (pair.TWO_CTAS_PER_HEAD, pair.CTAS_PER_HEAD, pair.STATE_STREAM, pair.max_tokens()) == (True, 2, False, 35)
    assert (one_cta.TWO_CTAS_PER_HEAD, one_cta.CTAS_PER_SM, one_cta.STATE_STREAM, one_cta.max_tokens()) == (False, 1, False, 8)
    assert _sm100_verify_form("cuda:0", 2) == one_cta
    assert (two_cta.MIN_CTAS_PER_SM, two_cta.STATE_STREAM, two_cta.max_tokens()) == (2, True, 45)
    assert _probe("cuda:0", "verify").served and _probe("cuda:0", "verify", requests=3).served
    for requests, tokens in VERIFY_SHAPES + [(1, 2), (1, 35), (2, 8), (3, 8), (4, 45), (16, 45), (backend.VERIFY_MAX_REQUESTS, 2)]:
        assert _probe("cuda:0", "verify", requests=requests, tokens_per_request=tokens).served, (requests, tokens)
    assert _probe("cuda:0", "verify", tokens_per_request=35).served, "without a requests count the one-request pick's budget answers"
    for requests, tokens, token in (
        (1, 36, "two CTAs per head"),
        (2, 9, "one CTA per SM"),
        (3, 9, "one CTA per SM"),
        (4, 46, "two CTAs per SM"),
        (None, 36, "two CTAs per head"),
    ):
        sup = _probe("cuda:0", "verify", requests=requests, tokens_per_request=tokens)
        assert (sup.served, sup.kind) == (False, "shape") and token in sup.reason and f"tokens_per_request={tokens}" in sup.reason, (requests, tokens, sup)
    sup = _probe("cuda:0", "verify", requests=2, tokens_per_request=1)
    assert (sup.served, sup.kind) == (False, "shape") and "decode form" in sup.reason, sup
    assert backend.VERIFY_MAX_REQUESTS == ventry.MAX_REQUESTS == 65535, "one source: the entry's own bound"
    sup = _probe("cuda:0", "verify", requests=backend.VERIFY_MAX_REQUESTS + 1, tokens_per_request=4)
    assert (sup.served, sup.kind) == (False, "shape") and str(backend.VERIFY_MAX_REQUESTS) in sup.reason, sup
    assert (
        fi_support.unsupported_reason("cuda:0", GEOMETRY, "verify", backend.BACKEND, requests=1, tokens_per_request=36)
        == _probe("cuda:0", "verify", requests=1, tokens_per_request=36).reason
    )


def test_verify_step_shape_envelope_on_a_99kb_part(as_sm120):
    """The verify form on a 99 KB-opt-in part with 170 SMs (the cc 12.0 readings) through the REAL probe path (the entry's device readers
    patched): the pick is token-aware -- the 2-deep-ring pair at one request and the one-CTA landing-slab form at two and more hold 8
    tokens, the landing-slab pair holds 17 at every requests count -- so the served envelope is 2 .. 17 tokens per request everywhere,
    ``verify_max_tokens`` answers 17, and T 18 is a ``shape`` decline naming the deepest form (two CTAs per head) and the bound; the
    probe and the entry's own ``max_tokens_for`` agree (no served-then-raise shape exists)."""
    k = ventry.GdnVerifyFused("cuda:0", GEOMETRY.norm_eps)
    assert k.readings == (SM120_READINGS["SMEM_PER_SM_BYTES"], SM120_READINGS["SMEM_OPTIN_BYTES"]) and k.sm_count == SM120_SMS
    assert replace(k.cfg, **SM100_READINGS) == kvcfg.ONE_CTA_NBUF1_CFG, "the start form: the one-CTA form this part holds"
    for requests, tokens, want in (
        (1, None, kvcfg.PAIR_NBUF2_CFG),
        (1, 6, kvcfg.PAIR_NBUF2_CFG),
        (1, 12, kvcfg.PAIR_NBUF1_CFG),
        (2, 6, kvcfg.ONE_CTA_NBUF1_CFG),
        (4, 17, kvcfg.PAIR_NBUF1_CFG),
    ):
        got = backend.verify_form("cuda:0", requests, GEOMETRY.norm_eps, tokens)
        assert replace(got, NORM_FORM=None, **SM100_READINGS) == want and got == _sm120_verify_form("cuda:0", requests, GEOMETRY.norm_eps, tokens), (
            requests,
            tokens,
        )
    assert backend.verify_form("cuda:0") == backend.verify_form("cuda:0", 1, GEOMETRY.norm_eps, 2), "no step shape: the pick at one request, two tokens"
    for requests in (None, 1, 2, 3, 4, 8, 16):
        assert backend.verify_max_tokens("cuda:0", requests) == 17 == k.max_tokens_for(1 if requests is None else requests), requests
    assert _probe("cuda:0", "verify").served and _probe("cuda:0", "verify", requests=3).served
    for requests, tokens in [(1, t) for t in range(2, 18)] + [(2, 8), (2, 9), (3, 8), (4, 17), (16, 17), (backend.VERIFY_MAX_REQUESTS, 17)]:
        assert _probe("cuda:0", "verify", requests=requests, tokens_per_request=tokens).served, (requests, tokens)
    for requests in (1, 2, 4, None):
        sup = _probe("cuda:0", "verify", requests=requests, tokens_per_request=18)
        assert (sup.served, sup.kind) == (False, "shape"), (requests, sup)
        assert "two CTAs per head" in sup.reason and "at most 17" in sup.reason and "tokens_per_request=18" in sup.reason and "101376 B" in sup.reason, sup
    assert _probe("cuda:0", "decode").served and _probe("cuda:0", "decode", requests=64).served


DOCS_PAGE = Path(__file__).resolve().parents[4] / "docs" / "operations" / "linear_attention" / "gdn_decode_flashinfer_backend.md"  # <repo>/docs/...


def _docs_table_rows():
    """The docs page's served / declined table as ``{question: answer}`` (the GDN support matrix, maintained by hand)."""
    rows = {}
    for line in DOCS_PAGE.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("| ") else []
        if len(cells) == 2 and cells[0] not in ("question", "---"):
            rows[cells[0]] = cells[1]
    return rows


def test_docs_served_and_declined_rows_are_reachable_with_the_published_bounds(monkeypatch):
    """The docs page's served / declined table is the GDN support matrix: its ``device`` and ``verify step shape`` rows name exactly the
    compute capabilities and the per-part token bounds the code computes, and every decline those rows publish is reachable through BOTH
    twins -- FlashInfer's probe (``support``) and the entries' own ``check_support`` -- with the published tokens in its text (a row no
    code path reaches, or a bound the page misquotes, fails here)."""
    rows = _docs_table_rows()
    assert {"device", "verify step shape", "decode rows"} <= set(rows), sorted(rows)
    device_row, shape_row = rows["device"], rows["verify step shape"]
    for token in ("10.x", "12.x", "cc 10.0", "cc 12.0", "kind `device`"):
        assert token in device_row, (token, device_row)
    assert entry.SUPPORTED_CC_TEXT == ventry.SUPPORTED_CC_TEXT == "10.x (SM100-class) or 12.x (SM120-class)"
    assert entry.SUPPORTED_CC_RANGES == ventry.SUPPORTED_CC_RANGES == ((100, 109), (120, 129))
    for token in ("35, 8 and 45", "17", "kind `shape`", "`verify_max_tokens(device, requests)`"):
        assert token in shape_row, (token, shape_row)
    assert [kvcfg.max_tokens_served(r, SM100_SMS) for r in (1, 2, 3, 4)] == [35, 8, 8, 45]
    sm120 = (SM120_READINGS["SMEM_PER_SM_BYTES"], SM120_READINGS["SMEM_OPTIN_BYTES"])
    assert all(kvcfg.max_tokens_served(r, SM120_SMS, *sm120) == 17 for r in (1, 2, 3, 4, 8, 16))
    assert all(replace(kvcfg.default_cfg(r, SM120_SMS, *sm120, 8), **SM120_READINGS).max_tokens() == 8 for r in (1, 2, 4))  # "8 with the picked form"
    # the device row through both twins: every declined capability by name, every served one through
    monkeypatch.setattr(entry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(ventry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(backend, "verify_form", _sm100_verify_form)
    dspec, vspec = _spec(rows=2), _spec(form="verify", rows=1, tokens=4)
    dargs = backend.FrostFeDecodeRunner(dspec)._args(fi_api.bind_inputs(dspec, **_tensors(2, CPU)))
    vkw = backend.FrostFeVerifyRunner(vspec)._kwargs(fi_api.bind_inputs(vspec, **_verify_tensors(1, 4, CPU)))
    for cc, name in (((9, 0), "SM90"), ((11, 0), "SM110"), ((13, 0), "SM130")):
        monkeypatch.setattr(fi_support, "device_cc", lambda device, cc=cc: cc)
        for form, kw in (("decode", {}), ("verify", dict(requests=1, tokens_per_request=4))):
            sup = _probe("cuda:0", form, **kw)
            assert (sup.served, sup.kind) == (False, "device") and name in sup.reason and "10.x" in sup.reason and "12.x" in sup.reason, sup
        with pytest.raises(NotImplementedError, match=rf"10\.x .*12\.x.*{cc[0]}\.{cc[1]}"):
            entry.check_support(*dargs, device_cc=cc)
        with pytest.raises(NotImplementedError, match=rf"10\.x .*12\.x.*{cc[0]}\.{cc[1]}"):
            ventry.check_support(**vkw, device_cc=cc)
    for cc in ((10, 0), (10, 3), (10, 7), (12, 0), (12, 1)):
        monkeypatch.setattr(fi_support, "device_cc", lambda device, cc=cc: cc)
        assert _probe("cuda:0", "decode").served and _probe("cuda:0", "verify", requests=1, tokens_per_request=4).served, cc
        entry.check_support(*dargs, device_cc=cc)
        ventry.check_support(**vkw, device_cc=cc)
    # the verify step-shape row through both twins: the cc 10.0 bound at one request (35) and the 99 KB bound (17)
    monkeypatch.setattr(fi_support, "device_cc", lambda device: (10, 0))
    sup = _probe("cuda:0", "verify", requests=1, tokens_per_request=36)
    assert (sup.served, sup.kind) == (False, "shape") and "at most 35" in sup.reason and "two CTAs per head" in sup.reason, sup
    vkw36 = backend.FrostFeVerifyRunner(_spec(form="verify", rows=1, tokens=36))._kwargs(
        fi_api.bind_inputs(_spec(form="verify", rows=1, tokens=36), **_verify_tensors(1, 36, CPU))
    )
    with pytest.raises(NotImplementedError, match="at most 35 on this part"):
        ventry.check_support(**vkw36, device_cc=(10, 0), cfg=_sm100_verify_form("cuda:0", 1))
    monkeypatch.setattr(fi_support, "device_cc", lambda device: (12, 0))
    monkeypatch.setattr(ventry, "device_shared_memory", lambda device: sm120)
    monkeypatch.setattr(ventry, "device_multiprocessor_count", lambda device: SM120_SMS)
    monkeypatch.setattr(backend, "verify_form", _REAL_VERIFY_FORM)  # the real probe path again, now reading the 99 KB part
    sup = _probe("cuda:0", "verify", requests=1, tokens_per_request=18)
    assert (sup.served, sup.kind) == (False, "shape") and "at most 17" in sup.reason and "two CTAs per head" in sup.reason, sup
    assert _probe("cuda:0", "verify", requests=4, tokens_per_request=17).served
    vkw18 = backend.FrostFeVerifyRunner(_spec(form="verify", rows=1, tokens=18))._kwargs(
        fi_api.bind_inputs(_spec(form="verify", rows=1, tokens=18), **_verify_tensors(1, 18, CPU))
    )
    k120 = ventry.GdnVerifyFused("cuda:0", GEOMETRY.norm_eps)
    with pytest.raises(NotImplementedError, match="at most 17 on this part"):
        ventry.check_support(**vkw18, device_cc=(12, 0), cfg=k120.cfg_for(1, 18))
    ventry.check_support(**vkw, device_cc=(12, 0), cfg=k120.cfg_for(1, 4))


def test_verify_probe_declines_a_part_whose_shared_memory_holds_no_form_by_kind_device(monkeypatch):
    """A device whose shared memory holds no verify form (a 40 KB opt-in): the entry's typed NotImplementedError at construction reaches
    the probe as kind ``device`` naming the opt-in -- never a raised error inside ``support`` -- with and without a step shape; the
    decode form, whose one-CTA footprint every served part holds, is unaffected by the verify reading."""
    monkeypatch.setattr(fi_support, "device_cc", lambda device: (10, 0))
    monkeypatch.setattr(entry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(ventry, "_dsl_version_error", lambda: None)
    monkeypatch.setattr(ventry, "device_shared_memory", lambda device: (40960, 40000))
    monkeypatch.setattr(ventry, "device_multiprocessor_count", lambda device: SM120_SMS)
    for kw in (dict(requests=2, tokens_per_request=6), dict(tokens_per_request=2), dict(requests=1, tokens_per_request=18)):
        sup = _probe("cuda:0", "verify", **kw)
        assert (sup.served, sup.kind) == (False, "device") and "40000-byte" in sup.reason and "holds no verify form" in sup.reason, (kw, sup)
    assert _probe("cuda:0", "verify", requests=2).served, "no tokens count: the budget is not asked"
    with pytest.raises(NotImplementedError, match="40000-byte"):
        backend.verify_max_tokens("cuda:0", 2)
    assert _probe("cuda:0", "decode", requests=4).served


def test_verify_token_bound_reads_the_device_and_reports_a_failed_reading_as_a_device_decline(monkeypatch):
    monkeypatch.setattr(fi_support, "device_cc", lambda device: (10, 0))
    monkeypatch.setattr(ventry, "_dsl_version_error", lambda: None)
    seen = []

    def form(device, requests=None, norm_eps=1e-6, tokens_per_request=None):
        seen.append((device, requests, norm_eps, tokens_per_request))
        return _sm100_verify_form(device, requests, norm_eps, tokens_per_request)

    monkeypatch.setattr(backend, "verify_form", form)
    assert _probe("cuda:0", "verify", requests=2, tokens_per_request=6).served and seen == [("cuda:0", 2, GEOMETRY.norm_eps, 6)]
    assert _probe("cuda:0", "verify", requests=2).served and len(seen) == 1, "no tokens count: no device reading needed"

    def failing(device, requests=None, norm_eps=1e-6, tokens_per_request=None):
        raise RuntimeError("cuDeviceGetAttribute failed")

    monkeypatch.setattr(backend, "verify_form", failing)
    sup = _probe("cuda:0", "verify", requests=2, tokens_per_request=6)
    assert (sup.served, sup.kind) == (False, "device") and "cuDeviceGetAttribute failed" in sup.reason, sup

    def no_form(device, requests=None, norm_eps=1e-6, tokens_per_request=None):
        raise NotImplementedError("gdn_verify_fused: this part's 40000-byte shared-memory opt-in per CTA holds no verify form")

    monkeypatch.setattr(backend, "verify_form", no_form)
    sup = _probe("cuda:0", "verify", requests=2, tokens_per_request=6)
    assert (sup.served, sup.kind) == (False, "device") and "40000-byte" in sup.reason, sup  # the entry's typed decline of a part with no form

    def regressed(device, requests=None, norm_eps=1e-6, tokens_per_request=None):
        raise ValueError("a config regression")

    monkeypatch.setattr(backend, "verify_form", regressed)
    sup = _probe("cuda:0", "verify", requests=2, tokens_per_request=6)
    assert (sup.served, sup.kind) == (False, "backend") and "ValueError" in sup.reason, sup  # not a device decline: the raising probe is named by type


def test_build_declines_parameter_dtypes_before_any_device_work():
    for dtypes in ((F32, F32, F32), (BF16, BF16, BF16), (F32, F32, BF16), (F32, BF16, F32), (BF16, BF16, BF16)):
        with pytest.raises(NotImplementedError, match="dt_bias torch.bfloat16 and norm_weight torch.bfloat16"):
            backend.build(_spec(param_dtypes=dtypes))  # a CPU spec: the dtypes are the decline, not the device
        with pytest.raises(NotImplementedError, match="CUDA devices"):
            backend.build(_spec(form="verify", tokens=4, param_dtypes=dtypes))  # the verify kernel reads every dtype choice: the device is the decline
    assert backend._param_dtypes_error((F32, BF16, BF16)) is None
    with pytest.raises(NotImplementedError, match="decode kernel runs on CUDA devices"):
        backend.build(_spec())  # the kernel-native dtypes on a CPU device
    with pytest.raises(NotImplementedError, match="verify kernel runs on CUDA devices"):
        backend.build(_spec(form="verify", tokens=4))


def test_runner_maps_flashinfer_inputs_onto_the_entry_call_contract():
    spec = _spec(rows=2)
    t = _tensors(2, CPU)
    x = fi_api.bind_inputs(spec, **t)  # FlashInfer's own operand checks admit the set
    runner = backend.FrostFeDecodeRunner(spec)  # a host entry (no SM count to pick by); the output slots are allocated on the CPU
    args = runner._args(x)
    assert args[0] == 2
    assert [
        a is b
        for a, b in zip(
            args[1:],
            (x.qkv, x.conv_state, x.conv_weight, x.A_log, x.dt_bias, x.ssm_state, x.state_indices, x.z, x.norm_weight, x.out_scale, x.hidden, x.ba_weight),
        )
    ] == [True] * 12
    entry.check_support(*args, device_cc=(10, 0))  # the entry's host checks accept FlashInfer's serving layout (channel-innermost conv pool)
    dim_first = fi_api.bind_inputs(spec, **dict(t, conv_state=t["conv_state"].contiguous()))  # [slots, C, 3]: tap-contiguous channels
    assert tuple(dim_first.conv_state.stride()) == (30720, 3, 1)
    entry.check_support(*runner._args(dim_first), device_cc=(10, 0))
    with pytest.raises(RuntimeError, match="eagerly"):
        runner.run(x)  # nothing compiled for this binding: refused before any launch (the CUDA-graph guard, visible on the host)
    assert runner.buffers is runner.kernel.outputs(2) and runner.scratch_bf16 is runner.buffers["out"]
    assert runner.kernel.compiled is backend._shared_compiled(_spec(rows=4)), "plans of one form, device and epsilon share the compiled table"
    other = fi_api.Qwen35GdnPlanSpec(backend.BACKEND, "decode", 2, 1, Qwen35GdnGeometry(**backend.GEOMETRY, norm_eps=1e-5), CPU, (F32, BF16, BF16))
    assert backend._shared_compiled(other) is not runner.kernel.compiled
    assert backend._shared_compiled(_spec(form="verify", rows=2, tokens=4)) is not runner.kernel.compiled, "the two forms never share a table"


def test_verify_runner_maps_flashinfer_inputs_onto_the_entry_call_contract():
    spec = _spec(form="verify", rows=2, tokens=4)
    t = _verify_tensors(2, 4, CPU)
    x = fi_api.bind_inputs(spec, **t)  # FlashInfer's own operand checks admit the set
    runner = backend.FrostFeVerifyRunner(spec)  # a host entry (no device readings: the one-CTA form); the buffers are allocated on the CPU
    kw = runner._kwargs(x)
    assert list(kw) == list(ventry.OPERANDS) + ["out", "out_bf16"]
    assert [kw[name] is getattr(x, name) for name in ventry.OPERANDS] == [True] * 14 and kw["out"] is x.out and kw["out_bf16"] is None
    ventry.check_support(**kw, device_cc=(10, 0))  # the entry's host checks accept FlashInfer's verify layout (channel-contiguous conv pool)
    assert ventry.step_shape(kw["state_indices"]) == (spec.requests, spec.tokens_per_request) == (2, 4)
    dim_first = fi_api.bind_inputs(spec, **dict(t, conv_state=t["conv_state"].contiguous()))  # [slots, C, T + 2]: FlashInfer admits it ...
    with pytest.raises(NotImplementedError, match="channel-contiguous"):  # ... the verify kernel does not fuse it: the typed decline prepare raises
        ventry.check_support(**runner._kwargs(dim_first), device_cc=(10, 0))
    with pytest.raises(RuntimeError, match="eagerly"):
        runner.run(x)  # nothing compiled for this binding: refused before any launch (the CUDA-graph guard, visible on the host)
    assert runner.buffers is runner.kernel.outputs(2, 4) and runner.buffers["counts"].shape == (2 * kvcfg.TICKET_STRIDE,)
    assert runner.kernel.compiled is backend._shared_compiled(_spec(form="verify", rows=4, tokens=6)), "verify plans of one device and epsilon share"
    assert runner.kernel.compiled is not backend._shared_compiled(_spec(rows=2))
    bf16 = fi_api.bind_inputs(spec, **dict(t, out_bf16=torch.empty(8, 48, 128, dtype=BF16)))
    assert runner._kwargs(bf16)["out_bf16"] is bf16.out_bf16


def test_runner_launch_re_checks_the_binding():
    """FlashInfer's ``bind_inputs`` checks dtypes and shapes; the entries re-check the whole binding (alignment and layout too) on every
    launch, before the compiled kernel sees the pointers: with a kernel planted under the binding's compile key, a misaligned operand is a
    typed decline at ``run``, never a launch -- on both runners."""

    def launched(*args):
        raise AssertionError("the planted kernel was launched on a mismatched binding")

    spec = _spec(rows=2)
    x = fi_api.bind_inputs(spec, **_tensors(2, CPU))
    runner = backend.FrostFeDecodeRunner(spec)
    args = runner._args(x)
    runner.kernel.compiled[runner.kernel.key(args[0], args[1], args[2], args[6], args[8])] = launched
    with pytest.raises(ValueError, match="norm weight|aligned"):
        runner.run(replace(x, norm_weight=torch.ones(136, dtype=BF16)[1:129]))  # the right dtype and shape at a 2-byte offset
    vspec = _spec(form="verify", rows=2, tokens=4)
    vx = fi_api.bind_inputs(vspec, **_verify_tensors(2, 4, CPU))
    vrunner = backend.FrostFeVerifyRunner(vspec)
    kw = vrunner._kwargs(vx)
    vrunner.kernel.compiled[
        vrunner.kernel.key(kw["qkv"], kw["z"], kw["A_log"], kw["dt_bias"], kw["ssm_state"], kw["state_indices"], kw["norm_weight"], kw["conv_state"])
    ] = launched
    with pytest.raises(ValueError, match="norm_weight"):
        vrunner.run(replace(vx, norm_weight=torch.ones(136, dtype=BF16)[1:129]))


def test_support_probe_loads_no_kernel_module():
    code = textwrap.dedent("""
        import sys
        import cudnn.linear_attention.frost.flashinfer_gdn_backend as b
        from flashinfer.qwen_3_5_gdn_block import Qwen35GdnGeometry, qwen_3_5_gdn_block_support
        for form, kw in (("decode", {}), ("verify", dict(requests=2, tokens_per_request=6))):
            sup = qwen_3_5_gdn_block_support("cpu", Qwen35GdnGeometry(**b.GEOMETRY), form, b.BACKEND, **kw)
            assert not sup.served and sup.kind == "device", sup
        kernels = ("gdn_decode_fused_f16", "gdn_verify_fused_f16", "flashinfer.experimental.qwen_3_5_gdn_block.backends.frost.gdn", "flashinfer.experimental.qwen_3_5_gdn_block.backends.frost.gdn_verify")
        loaded = [m for m in sys.modules if m.endswith(kernels)]
        assert not loaded, loaded
        print("PROBE-OK")
        """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "PROBE-OK" in r.stdout, r.stderr[-3000:]


# ---------------------------------------------------------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------------------------------------------------------
def _device_or_skip(form="decode", **kw):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    dev = torch.device("cuda", torch.cuda.current_device())
    sup = _probe(dev, form, **kw)
    if not sup.served:
        pytest.skip(f"frost_fe is not served here ({form}): {sup.kind}: {sup.reason}")
    return dev


def _flashinfer_root():
    import flashinfer

    root = Path(flashinfer.__file__).resolve().parents[1]
    if not (root / "tests" / "experimental" / "qwen_3_5_gdn_block" / "test_frost_decode.py").is_file():
        pytest.skip("FlashInfer's tests/experimental/qwen_3_5_gdn_block is not next to the installed package (a source tree is needed)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _flashinfer_tests():
    """FlashInfer's own GDN block tests and torch reference (``ref``, ``dec``, ``ver``): importable from a source tree next to the
    installed package only.  The cells below call these modules' internals (their test functions, ``reference``, a plan's ``_runner``)
    as FlashInfer PR #8131 laid them out -- a FlashInfer that reshapes its test package skips here rather than failing, and the cells
    that read ``plan._runner`` are pinned to that layout."""
    _flashinfer_root()
    modules = None  # assigned in the try below; pytest.skip() in the except never returns
    try:
        modules = types.SimpleNamespace(
            ref=importlib.import_module("tests.experimental.qwen_3_5_gdn_block.reference"),
            dec=importlib.import_module("tests.experimental.qwen_3_5_gdn_block.test_frost_decode"),
            ver=importlib.import_module("tests.experimental.qwen_3_5_gdn_block.test_frost_verify"),
        )
    except ImportError as exc:
        pytest.skip(f"FlashInfer's test package did not import: {exc}")
    return modules


def _flashinfer_bench():
    """FlashInfer's ``benchmarks/bench_qwen_3_5_gdn_block.py`` (its input sets in the serving layout), from the source tree."""
    path = _flashinfer_root() / "benchmarks" / "bench_qwen_3_5_gdn_block.py"
    if not path.is_file():
        pytest.skip("FlashInfer's benchmarks/bench_qwen_3_5_gdn_block.py is not next to the installed package")
    spec = importlib.util.spec_from_file_location("flashinfer_bench_qwen_3_5_gdn_block", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pytest.skip(f"FlashInfer's benchmark did not import: {exc}")
    return module


@pytest.mark.parametrize("rows", ROWS)
def test_flashinfer_decode_reference_test_passes_against_frost_fe(monkeypatch, rows):
    """FlashInfer's ``test_decode_matches_reference`` with its entry point bound to ``backend="frost_fe"``: its tolerances, the cuDNN Frontend kernel."""
    dev = _device_or_skip()
    fi = _flashinfer_tests()
    monkeypatch.setattr(fi.dec, "qwen_3_5_gdn_block_decode", functools.partial(qwen_3_5_gdn_block_decode, backend=backend.BACKEND))
    fi.dec.test_decode_matches_reference(dev, rows)


def test_flashinfer_cuda_graph_replay_test_passes_against_frost_fe(monkeypatch):
    dev = _device_or_skip()
    fi = _flashinfer_tests()
    monkeypatch.setattr(fi.dec, "prepare_qwen_3_5_gdn_block", functools.partial(prepare_qwen_3_5_gdn_block, backend=backend.BACKEND))
    fi.dec.test_decode_cuda_graph_replay_matches_eager(dev)


def test_dim_first_conv_pool_is_served_and_fp32_parameters_are_declined():
    dev = _device_or_skip()
    fi = _flashinfer_tests()
    case = fi.ref.make_case(4, 1, seed=7, device=dev, form="decode", conv_dim_first=True)  # [slots, C, 3], the kernel-native dtypes
    ssm, conv = case.ssm_state.clone(), case.conv_state.clone()
    _, out = fi.dec._outputs(4, dev)
    q = qwen_3_5_gdn_block_decode(**fi.dec._tensors(case, ssm, conv), geometry=fi.dec.GEOMETRY, backend=backend.BACKEND, out_bf16=out)
    torch.cuda.synchronize()
    fi.dec._check(case, q, out, ssm, conv)
    with pytest.raises(NotImplementedError, match="norm_weight"):  # FlashInfer's fp32-parameter cell: declined at plan time, by name
        prepare_qwen_3_5_gdn_block("decode", geometry=GEOMETRY, device=dev, requests=4, backend=backend.BACKEND, param_dtypes=(F32, F32, F32))


def test_capture_of_an_unprepared_binding_is_refused_until_an_eager_run_compiled_it():
    """The compiled table is shared by every plan of one device and epsilon, so the cold binding must carry a compile-key fact no earlier
    cell produced: a wider QKVZ slab (another qkv / z row stride).  Under capture that binding is refused; after one eager run the same
    binding captures, and the replay is bitwise the eager run."""
    dev = _device_or_skip()
    fi = _flashinfer_tests()
    rows = 2
    plan = prepare_qwen_3_5_gdn_block("decode", geometry=GEOMETRY, device=dev, requests=rows, backend=backend.BACKEND)
    case = fi.ref.make_case(rows, 1, seed=3, device=dev, form="decode")
    slab = torch.zeros(rows, 16384 + 1024, dtype=torch.bfloat16, device=dev)  # row stride 17408 (a 16-byte multiple): a new compile key
    slab[:, :10240] = case.qkv
    slab[:, 10240:16384] = case.z.reshape(rows, 6144)
    case.qkv, case.z = slab[:, :10240], slab[:, 10240:16384].view(rows, 48, 128)
    q, out = fi.dec._outputs(rows, dev)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="eagerly"):
        with torch.cuda.graph(graph):
            plan.run(**fi.dec._tensors(case, case.ssm_state.clone(), case.conv_state.clone()), out=q, out_bf16=out)
    ssm_a, conv_a = case.ssm_state.clone(), case.conv_state.clone()
    q_a, out_a = fi.dec._outputs(rows, dev)
    plan.run(**fi.dec._tensors(case, ssm_a, conv_a), out=q_a, out_bf16=out_a)  # the eager run compiles the binding
    ssm_b, conv_b = case.ssm_state.clone(), case.conv_state.clone()
    q_b, out_b = fi.dec._outputs(rows, dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(**fi.dec._tensors(case, ssm_b, conv_b), out=q_b, out_bf16=out_b)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q_a.view(torch.uint8), q_b.view(torch.uint8)) and torch.equal(out_a, out_b)
    assert torch.equal(ssm_a, ssm_b) and torch.equal(conv_a, conv_b)
    fi.dec._check(case, q_b, out_b, ssm_b, conv_b)


def test_bitwise_with_the_built_in_frost_backend():
    dev = _device_or_skip()
    fi = _flashinfer_tests()
    sup = qwen_3_5_gdn_block_support(dev, GEOMETRY, "decode", "frost")
    if not sup.served:
        pytest.skip(f"FlashInfer's built-in frost backend is not served here: {sup.reason}")
    for rows in (1, 2, 4, 8):
        case = fi.ref.make_case(rows, 1, seed=100 + rows, device=dev, form="decode")
        got = {}
        for name in ("frost", backend.BACKEND):
            ssm, conv = case.ssm_state.clone(), case.conv_state.clone()
            q, out = fi.dec._outputs(rows, dev)
            qwen_3_5_gdn_block_decode(**fi.dec._tensors(case, ssm, conv), geometry=GEOMETRY, backend=name, out=q, out_bf16=out)
            torch.cuda.synchronize()
            got[name] = (q.view(torch.uint8), out, ssm, conv)
        for a, b, what in zip(got["frost"], got[backend.BACKEND], ("fp8 out", "bf16 out", "ssm state", "conv state")):
            assert torch.equal(a, b), f"rows {rows}: {what} differs between frost and frost_fe"


# --- verify ---------------------------------------------------------------------------------------------------------------
def _verify_device_or_skip(requests, tokens):
    return _device_or_skip("verify", requests=requests, tokens_per_request=tokens)


@pytest.mark.parametrize("requests, tokens", VERIFY_SHAPES, ids=str)
def test_flashinfer_verify_reference_test_passes_against_frost_fe(monkeypatch, requests, tokens):
    """FlashInfer's ``test_verify_matches_reference`` under its device launch policy with its entry point bound to ``backend="frost_fe"``:
    its tolerances (the bf16 output within 2^-6 relative + 1e-3, > 99.9 % of the live fp8 bytes, the conv pool bitwise, written state
    slots within 1e-4, untouched slots untouched, padding outputs zero), the cuDNN Frontend kernel."""
    dev = _verify_device_or_skip(requests, tokens)
    fi = _flashinfer_tests()
    monkeypatch.setattr(fi.ver, "qwen_3_5_gdn_block_verify", functools.partial(qwen_3_5_gdn_block_verify, backend=backend.BACKEND))
    fi.ver.test_verify_matches_reference(dev, requests, tokens, "device")


@pytest.mark.parametrize("policy", sorted(POLICY_SMS))
@pytest.mark.parametrize("requests, tokens", VERIFY_SHAPES, ids=str)
def test_flashinfer_verify_reference_passes_under_the_sm_count_policies(requests, tokens, policy):
    """FlashInfer's verify reference with the entry's SM-count pick steered the way its own test steers the built-in backend (its ``b200``
    policy sets the wrapper's SM count before the first compile): the 148-SM policy (on a cc 10.0 part the cluster pair at one request,
    the one-CTA-per-SM form at 2 .. 3, two CTAs per SM from 4) and, with an SM count below one request's grid, the beyond-the-grid form at
    every shape.  The expected form is the module pick on that SM count and THIS part's readings (the budget decides which form of the
    regime's ladder serves: the streamed two-CTAs-per-SM form on a cc 10.0 part, the one-CTA landing-slab form on a 99 KB opt-in), so a
    part with another SM count or budget pins its own switch points."""
    dev = _verify_device_or_skip(requests, tokens)
    fi = _flashinfer_tests()
    plan = prepare_qwen_3_5_gdn_block("verify", geometry=GEOMETRY, device=dev, requests=requests, tokens_per_request=tokens, backend=backend.BACKEND)
    sms = POLICY_SMS[policy]
    plan._runner.kernel.sm_count = sms  # the pick's SM-count input, before the first compile
    form = plan._runner.kernel.cfg_for(requests, tokens)
    want = kvcfg.default_cfg(requests, sms, *plan._runner.kernel.readings, tokens)
    assert replace(form, NORM_FORM=None, **SM100_READINGS) == want, (policy, requests, form)
    assert form.TWO_CTAS_PER_HEAD == (requests * 96 <= sms) and (form.STATE_STREAM or form.NBUF == 1) == (requests * 48 > sms), (policy, requests, form)
    case = fi.ver.make_case(requests, tokens, seed=100 * requests + tokens, device=dev, form="verify")
    ssm, conv = case.ssm_state.clone(), case.conv_state.clone()
    q, out = fi.ver._outputs(requests * tokens, dev)
    assert plan.run(**fi.ver._tensors(case, ssm, conv), out=q, out_bf16=out) is q
    torch.cuda.synchronize()
    fi.ver._check(case, q, out, ssm, conv)


def test_flashinfer_verify_cuda_graph_replay_test_passes_against_frost_fe(monkeypatch):
    dev = _verify_device_or_skip(2, 4)
    fi = _flashinfer_tests()
    monkeypatch.setattr(fi.ver, "prepare_qwen_3_5_gdn_block", functools.partial(prepare_qwen_3_5_gdn_block, backend=backend.BACKEND))
    fi.ver.test_verify_cuda_graph_replay_matches_eager(dev)


def test_flashinfer_verify_dim_first_conv_pool_decline_passes_against_frost_fe(monkeypatch):
    """FlashInfer's ``test_verify_declines_a_conv_pool_it_cannot_fuse``: the dim-first pool is a typed decline at plan time, by layout."""
    dev = _verify_device_or_skip(1, 4)
    fi = _flashinfer_tests()
    monkeypatch.setattr(fi.ver, "qwen_3_5_gdn_block_verify", functools.partial(qwen_3_5_gdn_block_verify, backend=backend.BACKEND))
    fi.ver.test_verify_declines_a_conv_pool_it_cannot_fuse(dev)


def test_verify_capture_of_an_unprepared_binding_is_refused_until_an_eager_run_compiled_it():
    """The verify twin of the decode guard: a cold binding (a wider QKVZ slab) is refused under capture; after one eager run it captures
    and the replay is bitwise the eager run."""
    requests, tokens = 2, 4
    dev = _verify_device_or_skip(requests, tokens)
    fi = _flashinfer_tests()
    rows = requests * tokens
    plan = prepare_qwen_3_5_gdn_block("verify", geometry=GEOMETRY, device=dev, requests=requests, tokens_per_request=tokens, backend=backend.BACKEND)
    case = fi.ver.make_case(requests, tokens, seed=11, device=dev, form="verify")
    slab = torch.zeros(rows, 16384 + 1024, dtype=torch.bfloat16, device=dev)  # row stride 17408: a compile key no earlier cell produced
    slab[:, :10240] = case.qkv
    slab[:, 10240:16384] = case.z.reshape(rows, 6144)
    case.qkv, case.z = slab[:, :10240], slab[:, 10240:16384].view(rows, 48, 128)
    q, out = fi.ver._outputs(rows, dev)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="eagerly"):
        with torch.cuda.graph(graph):
            plan.run(**fi.ver._tensors(case, case.ssm_state.clone(), case.conv_state.clone()), out=q, out_bf16=out)
    ssm_a, conv_a = case.ssm_state.clone(), case.conv_state.clone()
    q_a, out_a = fi.ver._outputs(rows, dev)
    plan.run(**fi.ver._tensors(case, ssm_a, conv_a), out=q_a, out_bf16=out_a)  # the eager run compiles the binding
    ssm_b, conv_b = case.ssm_state.clone(), case.conv_state.clone()
    q_b, out_b = fi.ver._outputs(rows, dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(**fi.ver._tensors(case, ssm_b, conv_b), out=q_b, out_bf16=out_b)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(q_a.view(torch.uint8), q_b.view(torch.uint8)) and torch.equal(out_a, out_b)
    assert torch.equal(ssm_a, ssm_b) and torch.equal(conv_a, conv_b)
    fi.ver._check(case, q_b, out_b, ssm_b, conv_b)


def _clone_input_set(base):
    """A fresh copy of one benchmark input set with the SAME strides: ``qkv`` / ``z`` views of one QKVZ clone, the conv pool cloned in its
    channel-innermost layout, a bf16 output added."""
    x = {k: v.clone() for k, v in base.items() if k not in ("qkv", "z", "conv_state")}
    qkvz = base["qkv"].as_strided((base["qkv"].shape[0], base["qkv"].stride(0)), (base["qkv"].stride(0), 1)).clone()
    c = base["qkv"].shape[1]
    x["qkv"] = qkvz[:, :c]
    x["z"] = qkvz[:, c:].view_as(base["z"])
    x["conv_state"] = base["conv_state"].clone(memory_format=torch.preserve_format)
    assert x["conv_state"].stride() == base["conv_state"].stride() and x["conv_state"].stride(1) == 1
    x["out_bf16"] = torch.empty_like(base["z"])
    return x


@pytest.mark.parametrize("requests", [1, 2, 4])
def test_verify_bitwise_with_the_built_in_frost_backend_on_the_bench_input_sets(requests):
    """FlashInfer's built-in ``frost`` verify backend and ``frost_fe`` on IDENTICAL inputs -- the benchmark's own input sets (its
    ``make_inputs("verify", requests, 6, device)``: the serving layout, every token accepted) and the same sets with random acceptance
    counts and a padding request -- at requests 1 / 2 / 4 x 6 tokens: the fp8 output, the bf16 normed value, the SSM state pool and the
    conv state pool byte for byte."""
    tokens = 6
    dev = _verify_device_or_skip(requests, tokens)
    bench = _flashinfer_bench()
    sup = qwen_3_5_gdn_block_support(dev, GEOMETRY, "verify", "frost", requests=requests, tokens_per_request=tokens)
    if not sup.served:
        pytest.skip(f"FlashInfer's built-in frost verify backend is not served here: {sup.reason}")
    for arm in ("all accepted", "random acceptance + padding"):
        torch.manual_seed(2000 + requests)
        base = bench.make_inputs("verify", requests, tokens, dev)
        assert tuple(base["state_indices"].shape) == (requests, tokens) and base["conv_state"].stride(1) == 1
        if arm != "all accepted":
            base["num_accepted_tokens"] = torch.randint(1, tokens + 1, (requests,), device=dev, dtype=torch.int32)
            if requests > 1:
                base["state_indices"][-1] = 0  # a padding request (every slot 0), as FlashInfer's tests lay it out
        got = {}
        for name in ("frost", backend.BACKEND):
            x = _clone_input_set(base)
            plan = prepare_qwen_3_5_gdn_block("verify", geometry=GEOMETRY, device=dev, requests=requests, tokens_per_request=tokens, backend=name)
            assert plan.run(**x) is x["out"]
            torch.cuda.synchronize()
            got[name] = x
        f, g = got["frost"], got[backend.BACKEND]
        for key, what in (("out", "fp8 out"), ("out_bf16", "bf16 out"), ("ssm_state", "ssm state"), ("conv_state", "conv state")):
            a, b = f[key], g[key]
            if key == "out":
                a, b = a.view(torch.uint8), b.view(torch.uint8)
            assert torch.equal(a, b), f"requests {requests}, {arm}: {what} differs between frost and frost_fe"
        assert not torch.equal(g["ssm_state"], base["ssm_state"]) and not torch.equal(g["conv_state"], base["conv_state"]), "the pools advanced"
        if arm != "all accepted" and requests > 1:
            assert not g["out"][-tokens:].view(torch.uint8).any() and not g["out_bf16"][-tokens:].any(), "the padding request's rows are zero"
