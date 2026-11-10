# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The served / declined table of the fused GDN decode / verify steps' docs page (``docs/operations/linear_attention/
gdn_decode_flashinfer_backend.md``) is the de-facto support matrix of the two entries.  This host-only module -- no FlashInfer, no CUDA
device -- pins its ``device`` and ``verify step shape`` rows to the code that produces them: the admission ranges and text of both
entries, the budget-derived token bounds of both validated parts, and the typed declines by name through the entries' own
``check_support``, so a lane without FlashInfer still checks the published table.  The FlashInfer-side twin of every row (the probe) is
``test_flashinfer_gdn_backend.py::test_docs_served_and_declined_rows_are_reachable_with_the_published_bounds``."""

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.linear_attention.frost import gdn_decode_fused as dentry  # noqa: E402
from cudnn.linear_attention.frost import gdn_verify_fused as ventry  # noqa: E402
from cudnn.linear_attention.frost.kernel import gdn_verify_fused_config as kvcfg  # noqa: E402
from test_gdn_decode_fused import _binding_args, make_operands as decode_operands  # noqa: E402  (the suite's operand builders)
from test_gdn_verify_fused import make_operands as verify_operands  # noqa: E402

pytestmark = [pytest.mark.L0]

DOCS_PAGE = Path(__file__).resolve().parents[4] / "docs" / "operations" / "linear_attention" / "gdn_decode_flashinfer_backend.md"  # <repo>/docs/...
SM100_READING = (kvcfg.SM100_SMEM_PER_SM_BYTES, kvcfg.SM100_SMEM_OPTIN_BYTES)  # 233,472 / 232,448: the cc 10.0 part validated
SM120_READING = (kvcfg.SM120_SMEM_PER_SM_BYTES, kvcfg.SM120_SMEM_OPTIN_BYTES)  # 102,400 / 101,376: the cc 12.0 part validated
SM100_SMS, SM120_SMS = 148, 170
DECLINED_CCS = ((9, 0), (11, 0), (13, 0))
SERVED_CCS = ((10, 0), (10, 3), (12, 0), (12, 1))


def docs_table_rows():
    """The docs page's served / declined table as ``{question: answer}`` (the GDN support matrix, maintained by hand)."""
    rows = {}
    for line in DOCS_PAGE.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("| ") else []
        if len(cells) == 2 and cells[0] not in ("question", "---"):
            rows[cells[0]] = cells[1]
    return rows


def test_the_docs_table_names_the_capabilities_and_bounds_the_code_computes():
    """The ``device`` row names both served families and both validated parts, the ``verify step shape`` row names the per-part token
    bounds -- and the code agrees: one admission text and one range tuple on both entries, 35 / 8 / 45 by regime on the cc 10.0 readings,
    8 with the picked form and 17 through the landing-slab pair on the cc 12.0 readings."""
    rows = docs_table_rows()
    assert {"device", "verify step shape", "decode rows"} <= set(rows), sorted(rows)
    device_row, shape_row = rows["device"], rows["verify step shape"]
    for token in ("10.x", "12.x", "cc 10.0", "cc 12.0", "kind `device`"):
        assert token in device_row, (token, device_row)
    assert dentry.SUPPORTED_CC_TEXT == ventry.SUPPORTED_CC_TEXT == "10.x (SM100-class) or 12.x (SM120-class)"
    assert dentry.SUPPORTED_CC_RANGES == ventry.SUPPORTED_CC_RANGES == ((100, 109), (120, 129))
    for cc in SERVED_CCS:
        assert dentry.cc_supported(cc) and ventry.cc_supported(cc), cc
    for cc in DECLINED_CCS + (None,):
        assert not dentry.cc_supported(cc) and not ventry.cc_supported(cc), cc
    for token in ("35, 8 and 45", "17", "kind `shape`", "`verify_max_tokens(device, requests)`"):
        assert token in shape_row, (token, shape_row)
    assert [kvcfg.max_tokens_served(r, SM100_SMS, *SM100_READING) for r in (1, 2, 3, 4, 8)] == [35, 8, 8, 45, 45]
    assert all(kvcfg.max_tokens_served(r, SM120_SMS, *SM120_READING) == 17 for r in (1, 2, 3, 4, 8, 16))
    for r in (1, 2, 4):
        on = dict(SMEM_PER_SM_BYTES=SM120_READING[0], SMEM_OPTIN_BYTES=SM120_READING[1])
        assert replace(kvcfg.default_cfg(r, SM120_SMS, *SM120_READING, 8), **on).max_tokens() == 8, r  # "8 with the picked form"
        assert kvcfg.default_cfg(r, SM120_SMS, *SM120_READING, 17) == kvcfg.PAIR_NBUF1_CFG, r  # "17 through the landing-slab pair"


def test_the_device_row_is_reachable_through_both_entries_check_support():
    """Every capability the row declines is declined by name through both entries' ``check_support`` (the text names both served
    families and the device), every capability it serves passes the architecture gate -- CPU operands, no device."""
    dargs = _binding_args(decode_operands())
    vops = verify_operands(requests=1, tokens=4)
    for cc in DECLINED_CCS:
        with pytest.raises(NotImplementedError, match=rf"10\.x .*12\.x.*{cc[0]}\.{cc[1]}"):
            dentry.check_support(*dargs, device_cc=cc)
        with pytest.raises(NotImplementedError, match=rf"10\.x .*12\.x.*{cc[0]}\.{cc[1]}"):
            ventry.check_support(**vops, device_cc=cc)
    for cc in SERVED_CCS:
        dentry.check_support(*dargs, device_cc=cc)
        ventry.check_support(**vops, device_cc=cc)


def test_the_verify_step_shape_row_is_reachable_through_the_entry(monkeypatch):
    """The row's bounds through the entry: on the cc 10.0 readings the one-request pick holds 35 tokens (36 is the typed decline naming
    35); on the cc 12.0 readings (device readers stubbed) the pick holds 17 at every requests count (18 is the typed decline naming 17,
    17 passes on the landing-slab pair); and a check without a form is bounded by the module default form, said as such."""
    on100 = replace(kvcfg.default_cfg(1, SM100_SMS, *SM100_READING), SMEM_PER_SM_BYTES=SM100_READING[0], SMEM_OPTIN_BYTES=SM100_READING[1])
    assert on100 == kvcfg.PAIR_CFG and on100.max_tokens() == 35
    ventry.check_support(**verify_operands(requests=1, tokens=35), device_cc=(10, 0), cfg=kvcfg.resolve_cfg(on100, 1))
    with pytest.raises(NotImplementedError, match="at most 35 on this part"):
        ventry.check_support(**verify_operands(requests=1, tokens=36), device_cc=(10, 0), cfg=kvcfg.resolve_cfg(on100, 1))
    monkeypatch.setattr(ventry, "device_shared_memory", lambda device: SM120_READING)
    monkeypatch.setattr(ventry, "device_multiprocessor_count", lambda device: SM120_SMS)
    k120 = ventry.GdnVerifyFused("cuda:0", 1e-6)
    assert all(k120.max_tokens_for(r) == 17 for r in (1, 2, 4, 16))
    for requests in (1, 4):
        ventry.check_support(**verify_operands(requests=requests, tokens=17), device_cc=(12, 0), cfg=k120.cfg_for(requests, 17))
        with pytest.raises(NotImplementedError, match="at most 17 on this part"):
            ventry.check_support(**verify_operands(requests=requests, tokens=18), device_cc=(12, 0), cfg=k120.cfg_for(requests, 18))
        assert (
            replace(k120.cfg_for(requests, 17), NORM_FORM=None, SMEM_PER_SM_BYTES=SM100_READING[0], SMEM_OPTIN_BYTES=SM100_READING[1]) == kvcfg.PAIR_NBUF1_CFG
        )
    with pytest.raises(
        NotImplementedError, match=r"module default form \(TPR 4, NBUF 3\) on the cc 10\.0 readings, which holds at most 8 tokens per request; got 12"
    ):
        ventry.check_support(**verify_operands(requests=1, tokens=12), device_cc=(12, 0))  # no form: the module default form's bound, never "on this part"
