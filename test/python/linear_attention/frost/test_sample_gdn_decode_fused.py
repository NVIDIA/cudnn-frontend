# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The numbered sample of the fused GDN decode step (``samples/frost/linear_attention/26_gdn_decode_fused.py``), loaded by path and run
through its ``main()``: on a device the entry serves (compute capability 10.x or 12.x with the CuTe DSL at its floor) the sample must
print ``[26] PASS`` -- its own checks against the fp64-accumulated PyTorch step, the replay and the batch-invariance comparison -- and the
line is recorded for the job log; where the entry does not serve the device, ``NOT RUN`` with the reason is recorded in its place, so a
job log never reads the same with and without the self-check.  On any other CUDA device the sample must print the entry's typed decline
after ``[26] SKIP``; without a CUDA device, and with a cuDNN Frontend that does not carry the entry, it must print its own ``SKIP``
sentence (both patched in: no kernel is involved, so these two cases run on every GPU class)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

import frost_routing  # noqa: E402

from cudnn.linear_attention.frost import gdn_decode_fused as entry  # noqa: E402

pytestmark = [pytest.mark.L0]

SAMPLE = Path(__file__).resolve().parents[4] / "samples/frost/linear_attention/26_gdn_decode_fused.py"
RECORD = "sample 26 gdn decode fused"  # the key of the job-log record: the sample's PASS line, or NOT RUN with the reason


def _load_sample():
    spec = importlib.util.spec_from_file_location("gdn_decode_fused_sample", SAMPLE)
    sample = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sample)
    return sample


def _run_sample(capsys) -> str:
    """Load the script by path, call its ``main()`` with the defaults and return what it printed."""
    _load_sample().main()
    return capsys.readouterr().out


def _served_or_reason():
    """None when this device is one the entry serves, else the reason it is not (the entry's own gates)."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    cc = tuple(torch.cuda.get_device_capability())
    if not entry.cc_supported(cc):
        return f"the fused GDN decode step serves compute capability {entry.SUPPORTED_CC_TEXT}, this device is {cc}"
    return entry._dsl_version_error()


def _served_or_skip():
    """Skip unless the entry serves this device.  The reason is recorded first: the CI jobs run without ``-r``, so a skip reason never
    reaches the job log by itself."""
    reason = _served_or_reason()
    if reason is not None:
        frost_routing.measured(RECORD, f"NOT RUN: {reason}")
        pytest.skip(reason)


def _verdict(out: str) -> str:
    """The one ``[26] PASS`` line of a run that did not skip."""
    verdict = [line for line in out.splitlines() if line.startswith("[26] PASS")]
    assert len(verdict) == 1 and "[26] SKIP" not in out, out
    return verdict[0]


def test_sample_skips_by_name_without_a_cuda_device(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _run_sample(capsys).splitlines() == ["[26] SKIP  needs a CUDA GPU"]


def test_sample_skips_by_version_without_the_entry(monkeypatch, capsys):
    """A released package next to a newer checkout of the samples: the sample names the version it found, before any device work."""
    sample = _load_sample()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sample, "cudnn", SimpleNamespace(__version__="1.32.3"))
    sample.main()
    assert capsys.readouterr().out.splitlines() == ["[26] SKIP  needs a cuDNN Frontend that carries cudnn.GdnDecodeFused (1.33.0 or newer); this is 1.32.3"]


def test_sample_prints_the_typed_decline_on_an_unserved_device(capsys):
    reason = _served_or_reason()
    if reason is None or not torch.cuda.is_available():
        pytest.skip("needs a CUDA device the entry declines" if reason is None else reason)
    lines = _run_sample(capsys).splitlines()
    assert len(lines) == 1 and lines[0].startswith("[26] SKIP  ") and len(lines[0]) > len("[26] SKIP  "), lines


def test_a_skipped_self_check_is_recorded_for_the_job_log(monkeypatch):
    """A job whose device the entry does not serve must not read like one that ran the sample (the same counts, no line): the skip
    leaves ``NOT RUN`` with the reason under the key the PASS line is recorded under."""
    recorded = {}
    monkeypatch.setattr(frost_routing, "measured", recorded.__setitem__)
    monkeypatch.setattr(entry, "cc_supported", lambda cc: False)
    with pytest.raises(pytest.skip.Exception):
        _served_or_skip()
    assert list(recorded) == [RECORD] and recorded[RECORD].startswith("NOT RUN: the fused GDN decode step serves compute capability "), recorded


def test_sample_checks_itself_on_a_served_device(capsys):
    _served_or_skip()
    verdict = _verdict(_run_sample(capsys))
    assert verdict.endswith("replay bitwise, batch invariance bitwise)"), verdict  # the default run compares 1 row with 4
    frost_routing.measured(RECORD, f"cc {torch.cuda.get_device_capability()} {verdict}")


def test_sample_claims_batch_invariance_only_after_comparing_row_counts(capsys):
    """With a single row count no run is compared with another, and the PASS line must say so instead of claiming the comparison."""
    _served_or_skip()
    _load_sample().main(row_counts=(2,))
    verdict = _verdict(capsys.readouterr().out)
    assert verdict.endswith("replay bitwise, batch invariance not checked (one row count))"), verdict
