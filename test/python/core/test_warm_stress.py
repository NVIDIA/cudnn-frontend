# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The warm stress driver (``warm_stress.py``) against fake launches: what it reports, what it refuses, when it passes.

No kernel of the package is launched here; a "launch" is a torch write into the output, so a defect can be planted at a
chosen launch and byte.  The ring that validates the stress on a real GPU is ``sdpa/frost/test_tile_dsl_consume_then_arrive.py``.
"""

import pytest
import torch

import warm_stress as ws

pytestmark = pytest.mark.L0

DEVICE = torch.device("cuda")
VALIDATED = ws.Validation(True, "warm stress validated on the test part: the unsafe twin failed 200 of 200 launches", 200, 200)
NOT_VALIDATED = ws.Validation(False, "warm stress not validated on the test part: the unsafe twin read 0 of 200", 200, 0)


@pytest.fixture(autouse=True)
def _no_job_log_records(monkeypatch):
    """The fake kernels of this module are not measurements: keep them out of the run's ``measured`` summary."""
    import frost_routing

    monkeypatch.setattr(frost_routing, "measured", lambda key, text: None)


class FakeKernel:
    """Writes ``good`` into ``out`` on every launch; a planted launch number writes something else (or nothing)."""

    def __init__(self, records=64, dtype=torch.int32):
        self.good = (torch.arange(records, device=DEVICE) * 3 + 1).to(dtype)
        self.out = torch.zeros(records, dtype=dtype, device=DEVICE)
        self.launches = 0
        self.poisons = 0
        self.plant = {}  # launch number -> callable(out)

    def launch(self):
        number = self.launches
        self.launches += 1
        if number in self.plant:
            self.plant[number](self.out)
        else:
            self.out.copy_(self.good)

    def poison(self):
        self.poisons += 1

    def case(self, **overrides):
        fields = dict(
            name="fake",
            launch=self.launch,
            outputs={"out": self.out},
            thread_blocks=4 * torch.cuda.get_device_properties(DEVICE).multi_processor_count,
            refills_per_launch=3,
            reference={"out": self.good},
            locate=lambda name, offset: f"record {offset // 4}",
        )
        fields.update(overrides)
        return ws.StressCase(**fields)


def _run(kernel, **kwargs):
    kwargs.setdefault("flush_mib", 0)
    kwargs.setdefault("poison", kernel.poison)
    return ws.warm_stress(kernel.case(), **kwargs)


def test_a_clean_kernel_reports_zero_and_runs_every_block():
    kernel = FakeKernel()
    report = _run(kernel, warm=40, poisoned=30)
    assert report.failures == 0 and report.launches == 70 and report.complete and not report.wrong_cold
    assert [(b.block, b.launches) for b in report.blocks] == [("warm", 40), ("poisoned", 30)]
    assert kernel.poisons == 30, "one poison launch in front of every poisoned launch"
    assert kernel.launches == 2 + 70, "two cold launches, then the blocks"
    text = report.text()
    assert "0 wrong of 40 launches" in text and "equal the independent reference" in text


def test_a_planted_byte_is_reported_with_launch_offset_bytes_and_location():
    kernel = FakeKernel()

    def corrupt(out):
        out.copy_(kernel.good)
        out.view(torch.uint8)[4 * 9 + 2] ^= 0x40  # record 9, byte 2

    kernel.plant[2 + 17] = corrupt  # warm launch 17 (two cold launches come first)
    report = _run(kernel, warm=40, poisoned=10)
    first = report.first
    assert report.failures == 1 and first is not None
    assert (first.block, first.launch, first.output, first.byte_offset, first.differing_bytes) == ("warm", 17, "out", 38, 1)
    assert first.where == "record 9"
    got, expected = bytes.fromhex(first.got), bytes.fromhex(first.expected)
    assert len(got) == 16 and got[38 - 32] == expected[38 - 32] ^ 0x40 and got[:6] == expected[:6]
    assert report.blocks[0].stopped == "first failure" and report.blocks[0].launches == 18
    assert report.blocks[1].stopped.startswith("not run"), "the poisoned block does not run behind a failure"
    with pytest.raises(AssertionError, match="warm launch 17"):
        kernel2 = FakeKernel()
        kernel2.plant[2 + 17] = corrupt
        ws.assert_warm_stress(kernel2.case(), validation=VALIDATED, warm=40, poisoned=10, flush_mib=0, poison=kernel2.poison)


def test_counting_on_reports_every_failing_launch_per_quarter():
    kernel = FakeKernel()
    for number in (3, 4, 30, 39):
        kernel.plant[2 + number] = lambda out: out.zero_()
    report = _run(kernel, warm=40, poisoned=0, stop_at_first=False)
    block = report.blocks[0]
    assert (block.failures, block.launches, block.quarters) == (4, 40, [2, 0, 0, 2])
    assert block.first.launch == 3 and "failing launches per quarter [2, 0, 0, 2]" in report.text()


def test_a_launch_that_writes_nothing_is_caught_by_the_sentinel():
    kernel = FakeKernel()
    kernel.plant[2 + 5] = lambda out: None  # the output keeps the sentinel fill
    report = _run(kernel, warm=10, poisoned=0)
    assert report.failures == 1 and report.first.launch == 5
    assert report.first.got == f"{ws.SENTINEL_BYTE:02x}" * 16


@pytest.mark.parametrize("payload", ["nan_payload", "signed_zero"])
def test_the_compare_is_on_bytes_not_on_float_values(payload):
    """A NaN with another payload and a negative zero are different bytes; a float compare would call the first different from
    itself in every launch and the second equal."""
    kernel = FakeKernel(dtype=torch.float32)
    if payload == "nan_payload":
        kernel.good.view(torch.int32)[5] = 0x7FC00001
        other = 0x7FC00002
    else:
        kernel.good[5] = 0.0
        other = -0x80000000

    def corrupt(out):
        out.copy_(kernel.good)
        out.view(torch.int32)[5] = other

    clean = _run(kernel, warm=8, poisoned=0)
    assert clean.failures == 0, "the same NaN bytes in every launch are not a failure"
    kernel.plant[kernel.launches + 2 + 3] = corrupt
    report = _run(kernel, warm=8, poisoned=0)
    assert report.failures == 1 and report.first.launch == 3 and report.first.byte_offset // 4 == 5


def test_wrong_cold_launches_stop_the_stress_and_fail_the_assert():
    kernel = FakeKernel()
    kernel.plant[1] = lambda out: out.zero_()  # the second cold launch differs
    report = _run(kernel, warm=10, poisoned=10)
    assert report.wrong_cold and not report.cold_stable and report.launches == 0
    assert all(b.stopped.startswith("not run") for b in report.blocks)
    kernel = FakeKernel()
    kernel.good = kernel.good + 1  # the kernel is stable and disagrees with the independent reference
    case = kernel.case(reference={"out": kernel.good - 1})
    with pytest.raises(AssertionError, match="DIFFER FROM the independent reference"):
        ws.assert_warm_stress(case, validation=VALIDATED, warm=5, poisoned=0, flush_mib=0)


def test_wrong_cold_launches_are_counted_on_against_an_independent_reference():
    """The control's mode: an unsafe kernel whose cold launches are wrong too is still counted, against the independent reference."""
    kernel = FakeKernel()
    for number in range(0, 12, 2):
        kernel.plant[number] = lambda out: out.zero_()
    report = _run(kernel, warm=10, poisoned=0, stop_at_first=False)
    assert report.wrong_cold and report.blocks[0].launches == 10 and report.blocks[0].failures == 5


def test_without_an_independent_reference_only_stability_is_tested():
    kernel = FakeKernel()
    report = ws.warm_stress(kernel.case(reference=None), warm=6, poisoned=0, flush_mib=0)
    assert report.failures == 0 and report.cold_equals_reference is None and report.reference_kind == "cold launches only"
    assert "an error every launch repeats passes" in report.text()


def test_a_listed_output_that_no_launch_writes_is_refused_without_a_reference():
    """A copy of the kernel's output (what ``.contiguous()`` or a dtype cast returns) is never written.  Without an independent
    reference both cold launches leave the sentinel fill, the cold reference becomes the sentinel and every launch equals it."""
    kernel = FakeKernel()
    copy = kernel.out.clone()
    with pytest.raises(ValueError, match="still hold the sentinel fill after both cold launches"):
        ws.warm_stress(kernel.case(outputs={"out": copy}, reference=None), warm=100, poisoned=0, flush_mib=0)
    assert kernel.launches == 2, "refused behind the two cold launches, before any block"
    # one written output beside one that is not: still refused, and the message names the unwritten one
    kernel = FakeKernel()
    with pytest.raises(ValueError, match=r"\['stale'\] still hold the sentinel fill"):
        ws.warm_stress(kernel.case(outputs={"out": kernel.out, "stale": kernel.out.clone()}, reference=None), warm=4, poisoned=0, flush_mib=0)


def test_an_unwritten_output_fails_against_a_reference_and_the_report_names_it():
    kernel = FakeKernel()
    case = kernel.case(outputs={"out": kernel.out.clone()})
    report = ws.warm_stress(case, warm=10, poisoned=0, flush_mib=0)
    assert report.wrong_cold and report.cold_equals_reference is False and report.cold_unwritten == ["out"] and report.launches == 0
    assert "not written by either cold launch (still the sentinel fill): 'out'" in report.text()
    with pytest.raises(AssertionError, match="not written by either cold launch"):
        ws.assert_warm_stress(case, validation=VALIDATED, warm=10, poisoned=0, flush_mib=0)


def test_without_a_reference_the_assert_needs_cold_reference_launches():
    """No independent reference and no cache flush in front of the reference launches: the reference is two back-to-back
    launches of the kernel under test.  The report says so; the assert skips it as incomplete instead of passing."""
    report = ws.warm_stress(FakeKernel().case(reference=None), warm=4, poisoned=0, flush_mib=0)
    assert report.failures == 0 and not report.cold_flushed and "reference not cold" in report.text()
    with pytest.raises(pytest.skip.Exception, match="warm stress incomplete.*the reference launches were not cold"):
        ws.assert_warm_stress(FakeKernel().case(reference=None), validation=VALIDATED, warm=4, poisoned=0, flush_mib=0)
    # with a cache flush the same case is a pass
    passed = ws.assert_warm_stress(FakeKernel().case(reference=None), validation=VALIDATED, warm=4, poisoned=0, flush_mib=1)
    assert passed.cold_flushed and passed.failures == 0


def test_build_launchers_never_raises_and_a_missing_control_is_reported(monkeypatch):
    """``build_launchers`` is the setup step a test runs in front of its kernel watchdog; what it could not build is reported by
    the validation (not validated, not available), never raised."""
    import cudnn.frost.buffers as buffers

    monkeypatch.setattr(ws, "_CONTROL", {})
    monkeypatch.setattr(ws, "_VALIDATION", {})
    monkeypatch.setattr(ws, "_poison_launcher", lambda device: (None, "no DSL here"))
    monkeypatch.setattr(buffers, "cutedsl_state", lambda: (False, None))
    assert ws.build_launchers(DEVICE, cluster=2) is None
    assert list(ws._CONTROL.values()) == [(None, "ImportError: the control ring needs the CuTe DSL at the FROST floor")]
    validation = ws.stress_validation(DEVICE, cluster=2)
    assert not validation.reproduced and not validation.available and "the control ring could not be built (ImportError" in validation.text
    assert ws._CONTROL == {}, "the control ring is dropped once its validation is cached"
    ws.build_launchers(DEVICE, cluster=2)
    assert ws._CONTROL == {}, "and it is not built again behind a cached validation"


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(refills_per_launch=0), "never refills the stage"),
        (dict(thread_blocks=0), "thread_blocks"),
        (dict(outputs={}), "every tensor the kernel writes"),
        (dict(reference={"other": torch.zeros(64, dtype=torch.int32, device=DEVICE)}), "the reference names"),
        (dict(reference={"out": torch.zeros(8, dtype=torch.int32, device=DEVICE)}), "exactly the bytes"),
        (dict(reference={"out": torch.full((64,), -0x3C3C3C3D, dtype=torch.int32, device=DEVICE)}), "equals the sentinel fill"),
    ],
)
def test_a_case_that_cannot_fail_is_refused(overrides, message):
    kernel = FakeKernel()
    with pytest.raises(ValueError, match=message):
        ws.warm_stress(kernel.case(**overrides), warm=4, poisoned=0, flush_mib=0)


def test_a_non_contiguous_output_is_refused():
    kernel = FakeKernel()
    wide = torch.zeros(64, 2, dtype=torch.int32, device=DEVICE)
    with pytest.raises(ValueError, match="not contiguous"):
        ws.warm_stress(kernel.case(outputs={"out": wide[:, 0]}), warm=4, poisoned=0, flush_mib=0)


def test_a_zero_is_a_pass_only_behind_a_reproduced_control_and_a_complete_run():
    kernel = FakeKernel()
    report = ws.assert_warm_stress(kernel.case(), validation=VALIDATED, warm=20, poisoned=20, flush_mib=0, poison=kernel.poison)
    assert report.failures == 0 and report.complete and "control: warm stress validated" in report.text()
    with pytest.raises(pytest.skip.Exception, match="not validated on the test part"):
        ws.assert_warm_stress(FakeKernel().case(), validation=NOT_VALIDATED, warm=20, poisoned=0, flush_mib=0)
    with pytest.raises(pytest.skip.Exception, match="warm stress incomplete"):
        ws.assert_warm_stress(FakeKernel().case(), validation=VALIDATED, warm=10**9, poisoned=0, flush_mib=0, budget_s=0.2)


def test_a_mismatch_fails_even_where_the_control_did_not_reproduce():
    kernel = FakeKernel()
    kernel.plant[2 + 1] = lambda out: out.zero_()
    with pytest.raises(AssertionError, match="1 wrong of 2 launches"):
        ws.assert_warm_stress(kernel.case(), validation=NOT_VALIDATED, warm=20, poisoned=0, flush_mib=0)


def test_the_report_states_thread_blocks_per_sm_and_flags_a_thin_grid():
    sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    thin = ws.warm_stress(FakeKernel().case(thread_blocks=sm_count), warm=2, poisoned=0, flush_mib=0)
    assert thin.blocks_per_sm == 1.0 and "below the control's geometry" in thin.text()
    full = ws.warm_stress(FakeKernel().case(thread_blocks=3 * sm_count), warm=2, poisoned=0, flush_mib=0)
    assert "3.00 per SM" in full.text() and "below the control's geometry" not in full.text()


def test_a_missing_poison_launch_is_reported_never_dropped(monkeypatch):
    monkeypatch.setattr(ws, "_poison_launcher", lambda device: (None, "no DSL here"))
    report = ws.warm_stress(FakeKernel().case(), warm=4, poisoned=4, flush_mib=0)
    assert report.blocks[1].stopped == "not run: no poison launch (no DSL here)" and not report.complete
    with pytest.raises(pytest.skip.Exception, match="incomplete"):
        ws.assert_warm_stress(FakeKernel().case(), validation=VALIDATED, warm=4, poisoned=4, flush_mib=0)


def test_launch_counts_and_rate_bounds():
    assert ws.launches_for(1e-3) == 2995 and ws.launches_for(0.05) == 59 and ws.launches_for(0.01) == 299
    assert ws.launches_for(1e-3, 0.99) == 4603
    assert abs(ws.upper_bound(0, 3000) - 9.98e-4) < 2e-6
    assert abs(ws.upper_bound(0, 200000) - 1.498e-5) < 1e-7
    assert abs(ws.upper_bound(1, 100) - 0.0466) < 5e-4, "exact one-sided bound for one failure of a hundred"
    assert ws.upper_bound(100, 100) == 1.0
    with pytest.raises(ValueError):
        ws.launches_for(0.0)


# ---------------------------------------------------------------------------
# The rendering pin on hand-written PTX
# ---------------------------------------------------------------------------

_PTX_HEAD = """
.visible .entry kernel(
    .param .u64 kernel_param_0,
    .param .u32 kernel_param_1
)
{
    ld.param.u64 %rd1, [kernel_param_0];
    ld.param.u32 %r9, [kernel_param_1];
    ld.shared.v4.b32 {%r10, %r11, %r12, %r13}, [%r1];
    ld.shared.v2.b32 {%r14, %r15}, [%r1+16];
"""


def _ptx(body):
    return _PTX_HEAD + body + "\n    ret;\n}\n"


def test_release_dependence_follows_the_address_to_the_loads():
    dependent = _ptx("""
    add.s32 %r20, %r10, %r11;
    add.s32 %r21, %r20, %r12;
    add.s32 %r22, %r21, %r13;
    add.s32 %r23, %r22, %r14;
    add.s32 %r24, %r23, %r15;
    and.b32 %r25, %r24, %r9;
    shl.b32 %r26, %r25, 3;
    add.s32 %r27, %r2, %r26;
    mapa.shared::cluster.u32 %r28, %r27, %r3;
    mbarrier.arrive.expect_tx.shared.b64 %rd9, [%r2+64], %r4;
    @%p1 mbarrier.arrive.release.cta.shared::cluster.b64 _, [%r28];""")
    sites = ws.assert_release_dependence(dependent, sites=1, words=6)
    assert len(sites) == 1, "an expect_tx arrive is a producer's arm, not a release site"
    assert sites[0].parameter == "kernel_param_1" and sites[0].loads == 2 and sites[0].loaded_registers == 6


def test_release_dependence_sees_a_dropped_value_a_visible_zero_and_a_guard():
    # one vector left out of the fold (or cancelled by the compiler): fewer loaded registers reach the arrive
    partial = _ptx("""
    add.s32 %r20, %r10, %r11;
    add.s32 %r21, %r20, %r12;
    add.s32 %r22, %r21, %r13;
    and.b32 %r25, %r22, %r9;
    shl.b32 %r26, %r25, 3;
    add.s32 %r27, %r2, %r26;
    mbarrier.arrive.shared::cta.b64 _, [%r27];""")
    with pytest.raises(AssertionError, match="computed from 4 loaded registers, expected at least 6"):
        ws.assert_release_dependence(partial, sites=1, words=6)
    # the mask is not a kernel parameter (an inline-asm zero): no dependent site
    visible_zero = _ptx("""
    add.s32 %r20, %r10, %r11;
    mov.b32 %r30, 0;
    and.b32 %r25, %r20, %r30;
    add.s32 %r27, %r2, %r25;
    mbarrier.arrive.shared::cta.b64 _, [%r27];""")
    with pytest.raises(AssertionError, match="expected 1 arrive site"):
        ws.assert_release_dependence(visible_zero, sites=1, words=1)
    # the parameter is only compared (an entry guard): the arrive takes the plain address
    guarded = _ptx("""
    setp.ne.b32 %p3, %r9, 0;
    @%p3 bra $L__BB0_9;
    add.s32 %r27, %r2, 8208;
    mbarrier.arrive.shared::cta.b64 _, [%r27];
    add.s32 %r20, %r10, %r11;""")
    found = ws.release_dependence(guarded)
    assert len(found) == 1 and not found[0].dependent and found[0].loaded_registers == 0


@pytest.mark.parametrize("words", [7, 8])
def test_release_dependence_counts_tmem_words_before_reuse(words):
    registers = ", ".join(f"%r{i}" for i in range(10, 18))
    body = "\n".join(f"    add.s32 %r{20+i}, %r{19+i}, %r{10+i};" for i in range(1, words))
    ptx = f"""
    ld.param.u32 %r9, [kernel_param_1];
    tcgen05.ld.sync.aligned.32x32b.x8.b32 {{{registers}}}, [%r1];
    tcgen05.wait::ld.sync.aligned;
    mov.b32 %r20, %r10;
{body}
    and.b32 %r30, %r{19+words}, %r9;
    shl.b32 %r31, %r30, 3;
    add.s32 %r32, %r2, %r31;
    mbarrier.arrive.shared::cta.b64 _, [%r32];
    """
    if words == 7:
        with pytest.raises(AssertionError, match="computed from 7 loaded registers, expected at least 8"):
            ws.assert_release_dependence(ptx, sites=1, words=8)
    else:
        (site,) = ws.assert_release_dependence(ptx, sites=1, words=8)
        assert (site.parameter, site.loads, site.loaded_registers) == ("kernel_param_1", 1, 8)
