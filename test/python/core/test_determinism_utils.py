# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

import determinism_utils as checks

pytestmark = pytest.mark.L0


def test_bit_comparison_rejects_signed_zero_and_nan_payload_changes():
    for left, right in ((0, -2147483648), (0x7FC00001, 0x7FC00002)):
        a = torch.tensor([left], dtype=torch.int32).view(torch.float32)
        b = torch.tensor([right], dtype=torch.int32).view(torch.float32)
        checks.assert_bitwise({"out": a}, {"out": a.clone()})
        with pytest.raises(AssertionError, match="bytes differ"):
            checks.assert_bitwise({"out": a}, {"out": b})


def test_comparison_requires_all_outputs_and_metadata():
    expected = {"out": torch.ones(2), "dq": torch.ones(2)}
    with pytest.raises(AssertionError, match="names changed"):
        checks.assert_bitwise({"out": expected["out"]}, expected)
    for changed in (torch.ones(1, 2), torch.ones(2, dtype=torch.float64)):
        with pytest.raises(AssertionError, match="shape or dtype"):
            checks.assert_bitwise({"out": changed, "dq": expected["dq"]}, expected)


def test_snapshot_detaches_reused_storage_and_rejects_nonfinite():
    value = torch.ones(3)
    saved = checks.snapshot({"out": value})
    value.zero_()
    assert torch.equal(saved["out"], torch.ones(3))
    for invalid in (torch.nan, torch.inf, -torch.inf):
        with pytest.raises(AssertionError, match="non-finite"):
            checks.snapshot({"out": torch.tensor([invalid])})
    checks.snapshot({"lse": torch.tensor([-torch.inf])})
    with pytest.raises(AssertionError, match="no outputs"):
        checks.snapshot({})


def test_replay_detects_changed_gradient_with_reused_storage():
    gradient = torch.zeros(2, device="cuda")

    def launch():
        gradient.add_(1)
        return {"out": torch.ones_like(gradient), "dq": gradient}

    with pytest.raises(AssertionError, match="replay 1 dq:.*bytes differ"):
        checks.assert_replays(launch, repeats=2)
    with pytest.raises(AssertionError, match="at least two"):
        checks.assert_replays(launch, repeats=1)


def test_infinity_sentinels_are_explicit_and_keep_sign_and_nan_checks(tmp_path, monkeypatch):
    outputs = {"lse": torch.tensor([torch.inf]), "max_logits": torch.tensor([-torch.inf])}
    allowed = {"lse": torch.inf, "max_logits": -torch.inf}
    checks.assert_replays(lambda: outputs, repeats=2, allowed_inf=allowed)
    for invalid in (-torch.inf, torch.nan):
        with pytest.raises(AssertionError, match="non-finite"):
            checks.snapshot({"lse": torch.tensor([invalid])}, allowed_inf=allowed)
    with pytest.raises(AssertionError, match="non-finite"):
        checks.snapshot(outputs)
    with pytest.raises(AssertionError, match="non-finite"):
        checks.snapshot({"out": torch.tensor([torch.inf])}, allowed_inf=allowed)
    with pytest.raises(AssertionError, match="only infinity"):
        checks.snapshot(outputs, allowed_inf={"lse": torch.nan})
    monkeypatch.setattr(checks, "runtime", lambda: {"sm": [10, 0]})
    monkeypatch.delenv("CUDNN_DETERMINISM_GOLDENS", raising=False)
    monkeypatch.setenv("CUDNN_DETERMINISM_RECORD", str(tmp_path))
    checks.assert_golden("sentinels", {}, outputs, allowed_inf=allowed)
    monkeypatch.delenv("CUDNN_DETERMINISM_RECORD")
    monkeypatch.setenv("CUDNN_DETERMINISM_GOLDENS", str(tmp_path))
    checks.assert_golden("sentinels", {}, outputs, allowed_inf=allowed)


def test_golden_requires_exact_outputs_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "runtime", lambda: {"sm": [10, 0]})
    monkeypatch.delenv("CUDNN_DETERMINISM_GOLDENS", raising=False)
    monkeypatch.setenv("CUDNN_DETERMINISM_RECORD", str(tmp_path))
    inputs, outputs = {"q": torch.ones(2)}, {"out": torch.ones(2), "dq": torch.zeros(2)}
    checks.assert_golden("control", inputs, outputs)
    with pytest.raises(FileExistsError):
        checks.assert_golden("control", inputs, outputs)
    record = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert set(record["outputs"]) == {"out", "dq"}
    monkeypatch.delenv("CUDNN_DETERMINISM_RECORD")
    monkeypatch.setenv("CUDNN_DETERMINISM_GOLDENS", str(tmp_path))
    checks.assert_golden("control", inputs, outputs)
    for changed in ({"out": torch.zeros(2), "dq": outputs["dq"]}, {"out": outputs["out"]}):
        with pytest.raises(AssertionError, match="golden mismatch"):
            checks.assert_golden("control", inputs, changed)
    with pytest.raises(pytest.fail.Exception, match="no golden"):
        checks.assert_golden("missing", inputs, outputs)
