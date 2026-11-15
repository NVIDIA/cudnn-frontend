# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A warm stress with byte references, for kernels that hand a shared-memory stage back to its producer.

A kernel that releases a stage while one of its shared-memory loads is still unread can receive the next fill of that stage
(``python/cudnn/AGENTS.md`` Rule 11).  The failure is timing-dependent: a few launches in a hundred thousand, every launch of
one shape and none of another, no cold launch at all.  A functional test, a "launch 1 equals launch 2" check and a replay
loop of a handful of launches pass such a kernel.  This helper is the run-time detector:

* a cold reference -- two launches behind a cache flush must agree byte for byte, and equal the case's independent
  reference when it has one;
* ``warm`` launches back to back and ``poisoned`` launches each behind a launch that overwrites every SM's shared memory,
  every output byte compared after every launch; outputs are refilled with a sentinel pattern before each launch, so a
  launch that wrote nothing differs from the reference, and a listed output that neither cold launch wrote (a copy of the
  kernel's tensor) is refused where there is no independent reference to differ from;
* a positive control (:func:`stress_validation`): the unsafe twin of the self-test ring (``warm_stress_ring.py``) must be
  seen to fail in this process on this GPU.  A zero count is reported as a pass only then; elsewhere the test is skipped
  with ``warm stress not validated on <part>``.

What a pass means: a RATE BOUND for the shape that was launched (0 of N bounds the per-launch rate at about 3 / N), on this
GPU, behind this control.  The control shows that the GPU and driver expose the class in the ring's geometry -- three thread
blocks per SM, every lane of a load in the same shared-memory banks.  The same unsafe ring reads zero with one thread block
per SM, so the report states the thread blocks per SM of the case and flags fewer than two.  The proof of the order is the
disassembly of the rendering (Rule 11); :func:`assert_release_dependence` pins it in the generated PTX.

A test that guards its launches with a process watchdog calls :func:`build_launchers` and builds its own kernel in front of
the watchdog: the first CuTe DSL import of a process can take minutes on a cold file system, and only launches may be timed
as launches.

``sdpa/frost/test_tile_dsl_consume_then_arrive.py`` is the usage example; ``core/test_warm_stress.py`` pins the report.
"""

import math
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

DEFAULT_WARM = 3000
DEFAULT_POISONED = 3000
DEFAULT_BUDGET_S = 60.0
CONTROL_LAUNCHES = 200
SENTINEL_BYTE = 0xC3
_CHECK_CLOCK_EVERY = 64


@dataclass
class StressCase:
    """One kernel shape under stress.

    ``launch``: ONE launch on the launch path that ships; it may return before the kernel finishes.
    ``outputs``: EVERY tensor the kernel writes, by name (contiguous).  A written tensor that is not listed is not compared.
    List the storage the kernel writes, never a copy of it (``.contiguous()`` of a strided view and a dtype cast return
    copies): an output that both cold launches leave at the sentinel fill is refused when there is no ``reference``.
    ``thread_blocks``: thread blocks of one launch (the grid's product).
    ``refills_per_launch``: how often the stage under test is refilled inside one launch; zero is refused, because a shape
    that never refills the stage cannot fail.
    ``reference``: independent expected bytes per output (a host or torch model of the same bytes).  Without it the helper
    tests stability across launches only: an error that every launch repeats passes.
    ``reset``: restores state the kernel mutates; the inputs must stay untouched between launches.
    ``locate``: ``(output name, byte offset) -> text`` naming what that byte belongs to.
    """

    name: str
    launch: Callable[[], None]
    outputs: Dict[str, torch.Tensor]
    thread_blocks: int
    refills_per_launch: int
    reference: Optional[Dict[str, torch.Tensor]] = None
    reset: Optional[Callable[[], None]] = None
    locate: Optional[Callable[[str, int], str]] = None


@dataclass
class Mismatch:
    block: str
    launch: int
    output: str
    byte_offset: int
    differing_bytes: int
    got: str
    expected: str
    where: str = ""

    def text(self) -> str:
        where = f" ({self.where})" if self.where else ""
        return (
            f"{self.block} launch {self.launch}: output '{self.output}' differs in {self.differing_bytes} byte(s), first at byte {self.byte_offset}{where}; "
            f"16 bytes from {self.byte_offset - self.byte_offset % 16}: got {self.got} expected {self.expected}"
        )


@dataclass
class BlockResult:
    block: str
    requested: int
    launches: int = 0
    failures: int = 0
    seconds: float = 0.0
    first: Optional[Mismatch] = None
    stopped: str = ""  # "", "first failure", "budget", "not run: <why>"
    quarters: List[int] = field(default_factory=lambda: [0, 0, 0, 0])

    @property
    def complete(self) -> bool:
        return self.launches == self.requested


@dataclass
class Validation:
    reproduced: bool
    text: str
    launches: int = 0
    failures: int = 0
    available: bool = True


@dataclass
class StressReport:
    case: str
    part: str
    sm_count: int
    thread_blocks: int
    refills_per_launch: int
    reference_kind: str  # "independent" or "cold launches only"
    cold_flushed: bool = True
    cold_stable: bool = True
    cold_equals_reference: Optional[bool] = None
    cold_mismatch: Optional[Mismatch] = None
    cold_unwritten: List[str] = field(default_factory=list)  # outputs both cold launches left at the sentinel fill
    blocks: List[BlockResult] = field(default_factory=list)
    validation: Optional[Validation] = None

    @property
    def blocks_per_sm(self) -> float:
        return self.thread_blocks / max(self.sm_count, 1)

    @property
    def wrong_cold(self) -> bool:
        return (not self.cold_stable) or self.cold_equals_reference is False

    @property
    def failures(self) -> int:
        return sum(b.failures for b in self.blocks)

    @property
    def launches(self) -> int:
        return sum(b.launches for b in self.blocks)

    @property
    def complete(self) -> bool:
        return all(b.complete for b in self.blocks)

    @property
    def first(self) -> Optional[Mismatch]:
        for b in self.blocks:
            if b.first is not None:
                return b.first
        return self.cold_mismatch

    def text(self) -> str:
        lines = [f"warm stress '{self.case}' on {self.part}"]
        geometry = f"  shape: {self.thread_blocks} thread blocks = {self.blocks_per_sm:.2f} per SM, {self.refills_per_launch} refill(s) of the stage per launch"
        if self.blocks_per_sm < 2.0:
            geometry += "  [below the control's geometry (3 thread blocks per SM): a zero for this shape bounds little]"
        lines.append(geometry)
        behind = "behind a cache flush" if self.cold_flushed else "(reference not cold: no cache flush in front of them)"
        cold = f"  cold reference: two launches {behind} {'agree' if self.cold_stable else 'DIFFER'}"
        if self.cold_equals_reference is not None:
            cold += f"; {'equal' if self.cold_equals_reference else 'DIFFER FROM'} the independent reference"
        else:
            cold += "; no independent reference (stability across launches only: an error every launch repeats passes)"
        lines.append(cold)
        if self.cold_mismatch is not None:
            lines.append("    " + self.cold_mismatch.text())
        if self.cold_unwritten:
            names = ", ".join(repr(name) for name in self.cold_unwritten)
            lines.append(f"    not written by either cold launch (still the sentinel fill): {names} -- list the tensor the kernel writes, not a copy")
        for b in self.blocks:
            if b.stopped.startswith("not run"):
                lines.append(f"  {b.block}: {b.stopped}")
                continue
            bound = upper_bound(b.failures, b.launches) if b.launches else 1.0
            line = f"  {b.block}: {b.failures} wrong of {b.launches} launches (requested {b.requested}) in {b.seconds:.1f} s"
            if b.failures == 0 and b.launches:
                line += f"; per-launch rate below {100.0 * bound:.3g} % (95 %)"
            if b.stopped:
                line += f"; stopped: {b.stopped}"
            if b.failures and not b.stopped:
                line += f"; failing launches per quarter {b.quarters}"
            lines.append(line)
            if b.first is not None:
                lines.append("    first: " + b.first.text())
        if self.validation is not None:
            lines.append("  control: " + self.validation.text)
        return "\n".join(lines)


def launches_for(rate: float, confidence: float = 0.95) -> int:
    """Launches needed to see at least one failure with ``confidence`` at a per-launch ``rate``."""
    if not 0.0 < rate < 1.0 or not 0.0 < confidence < 1.0:
        raise ValueError("rate and confidence must lie strictly between 0 and 1")
    return int(math.ceil(math.log(1.0 - confidence) / math.log(1.0 - rate)))


def upper_bound(failures: int, launches: int, confidence: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) upper bound of the per-launch rate after ``failures`` of ``launches``."""
    if launches <= 0 or failures < 0 or failures > launches:
        raise ValueError("need 0 <= failures <= launches and launches > 0")
    if failures == launches:
        return 1.0
    if failures == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / launches)

    def cdf(p: float) -> float:  # P(X <= failures) for X ~ Binomial(launches, p)
        total = 0.0
        for k in range(failures + 1):
            log_term = math.lgamma(launches + 1) - math.lgamma(k + 1) - math.lgamma(launches - k + 1) + k * math.log(p) + (launches - k) * math.log1p(-p)
            total += math.exp(log_term)
        return total

    lo, hi = failures / launches, 1.0 - 1e-15
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) > 1.0 - confidence:
            lo = mid
        else:
            hi = mid
    return hi


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).view(torch.uint8)


def _part(device) -> str:
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    return f"{props.name}, cc {major}.{minor}, {props.multi_processor_count} SMs"


def _refuse(case: StressCase) -> None:
    if case.refills_per_launch < 1:
        raise ValueError(f"warm stress '{case.name}': refills_per_launch must be at least 1 -- a shape that never refills the stage under test cannot fail")
    if case.thread_blocks < 1:
        raise ValueError(f"warm stress '{case.name}': thread_blocks must be the number of thread blocks of one launch")
    if not case.outputs:
        raise ValueError(f"warm stress '{case.name}': list every tensor the kernel writes in `outputs`")
    for name, out in case.outputs.items():
        if not out.is_contiguous():
            raise ValueError(f"warm stress '{case.name}': output '{name}' is not contiguous; pass the kernel's own contiguous storage")
        if out.numel() == 0:
            raise ValueError(f"warm stress '{case.name}': output '{name}' is empty")
    if case.reference is not None:
        if set(case.reference) != set(case.outputs):
            raise ValueError(f"warm stress '{case.name}': the reference names {sorted(case.reference)} but the outputs are {sorted(case.outputs)}")
        for name, ref in case.reference.items():
            out = case.outputs[name]
            if not ref.is_contiguous() or ref.numel() * ref.element_size() != out.numel() * out.element_size():
                raise ValueError(f"warm stress '{case.name}': reference '{name}' must be contiguous and hold exactly the bytes of the output")


def _mismatch(case: StressCase, block: str, launch: int, name: str, got: torch.Tensor, expected: torch.Tensor) -> Mismatch:
    differ = got != expected
    count = int(differ.sum().item())
    offset = int(differ.nonzero()[0].item())
    start = offset - offset % 16
    got_hex = bytes(got[start : start + 16].tolist()).hex()
    expected_hex = bytes(expected[start : start + 16].tolist()).hex()
    where = case.locate(name, offset) if case.locate is not None else ""
    return Mismatch(block, launch, name, offset, count, got_hex, expected_hex, where)


def _prepare(case: StressCase, views: Dict[str, torch.Tensor], device) -> None:
    if case.reset is not None:
        case.reset()
    for view in views.values():
        view.fill_(SENTINEL_BYTE)
    torch.cuda.synchronize(device)


def _first_difference(case: StressCase, block: str, launch: int, views: Dict[str, torch.Tensor], expected: Dict[str, torch.Tensor]) -> Optional[Mismatch]:
    for name, view in views.items():
        if not torch.equal(view, expected[name]):
            return _mismatch(case, block, launch, name, view, expected[name])
    return None


def _poison_launcher(device):
    """``(callable, why not)``: the launch that overwrites every SM's shared memory, built once per process and device."""
    key = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    if key not in _POISON:
        try:
            import warm_stress_ring

            _POISON[key] = (warm_stress_ring.Poison(device=device).launch, "")
        except Exception as exc:  # no DSL, or a part the poison kernel does not build for: the block is reported as not run
            _POISON[key] = (None, f"{type(exc).__name__}: {str(exc)[:200]}")
    return _POISON[key]


def _control_ring(device, cluster: int):
    """``(ring, why not)``: the unsafe twin of the test ring for this device and cluster size, built once per process."""
    device = torch.device(device)
    key = (device.index if device.index is not None else torch.cuda.current_device(), int(cluster))
    if key not in _CONTROL:
        try:
            from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old

            installed, version = cutedsl_state()
            if not installed or cutedsl_too_old(version):
                raise ImportError("the control ring needs the CuTe DSL at the FROST floor")
            import warm_stress_ring

            _CONTROL[key] = (warm_stress_ring.Ring("plain", device=device, cluster=cluster), "")
        except Exception as exc:  # reported by stress_validation as "not validated ... could not be built"
            _CONTROL[key] = (None, f"{type(exc).__name__}: {str(exc)[:160]}")
    return _CONTROL[key]


_POISON: Dict[int, tuple] = {}
_CONTROL: Dict[tuple, tuple] = {}
_VALIDATION: Dict[tuple, Validation] = {}


def build_launchers(device=None, *, cluster: int = 1) -> None:
    """Import the CuTe DSL and compile what the stress launches besides the case: the control ring and the poison kernel.

    Both are cached per process and device.  Call this, and build the kernel under test, BEFORE entering a kernel watchdog:
    the first DSL import of a process can take minutes on a cold file system, and a watchdog that also covers it reports a
    slow import as a hung kernel.  It never raises; what could not be built is reported by :func:`stress_validation` and by
    the poisoned block.
    """
    device = torch.device("cuda") if device is None else torch.device(device)
    key = (device.index if device.index is not None else torch.cuda.current_device(), int(cluster))
    if key not in _VALIDATION:  # a cached validation needs no ring
        _control_ring(device, cluster)
    _poison_launcher(device)


def warm_stress(
    case: StressCase,
    *,
    warm: int = DEFAULT_WARM,
    poisoned: int = DEFAULT_POISONED,
    flush_mib: int = 512,
    stop_at_first: bool = True,
    device=None,
    budget_s: Optional[float] = None,
    poison: Optional[Callable[[], None]] = None,
) -> StressReport:
    """Run the stress and return the report; raises only for a case that cannot fail (``ValueError``): one refused up front, or
    one without a reference whose listed output no cold launch wrote.

    ``budget_s``: wall-clock budget of the launch blocks.  A block that uses up its share ends at a check point and reports
    the launches it reached -- an incomplete block, never a longer run and never a kernel timeout.  ``stop_at_first=False``
    counts on after a failure (the control uses it) and, given an independent reference, also when the cold launches are wrong.
    """
    _refuse(case)
    first_out = next(iter(case.outputs.values()))
    device = first_out.device if device is None else torch.device(device)
    views = {name: _byte_view(out) for name, out in case.outputs.items()}
    reference = None if case.reference is None else {name: _byte_view(ref) for name, ref in case.reference.items()}
    if reference is not None:
        for name, ref in reference.items():
            if bool((ref == SENTINEL_BYTE).all().item()):
                raise ValueError(f"warm stress '{case.name}': reference '{name}' equals the sentinel fill; a launch that writes nothing would pass")
    report = StressReport(
        case=case.name,
        part=_part(device),
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
        thread_blocks=int(case.thread_blocks),
        refills_per_launch=int(case.refills_per_launch),
        reference_kind="independent" if reference is not None else "cold launches only",
    )

    # ---- the cold reference: two launches, each behind a rewritten flush buffer ----
    flush = None
    if flush_mib > 0:
        try:
            flush = torch.zeros((flush_mib << 20) // 4, dtype=torch.float32, device=device)
        except torch.OutOfMemoryError:
            report.cold_flushed = False
    else:
        report.cold_flushed = False
    cold = []
    for _ in range(2):
        _prepare(case, views, device)
        if flush is not None:
            flush.add_(1.0)
            torch.cuda.synchronize(device)
        case.launch()
        torch.cuda.synchronize(device)
        cold.append({name: view.clone() for name, view in views.items()})
    del flush
    report.cold_unwritten = [name for name in views if all(bool((launch[name] == SENTINEL_BYTE).all().item()) for launch in cold)]
    if report.cold_unwritten and reference is None:
        raise ValueError(
            f"warm stress '{case.name}': output(s) {report.cold_unwritten} still hold the sentinel fill after both cold launches -- the launch does not "
            "write the listed tensor (list the kernel's own storage, never a copy made by `.contiguous()` or a cast); "
            "without an independent reference every launch would equal that fill and pass"
        )
    for name in views:
        if not torch.equal(cold[0][name], cold[1][name]):
            report.cold_stable = False
            report.cold_mismatch = _mismatch(case, "cold", 1, name, cold[1][name], cold[0][name])
            break
    if reference is not None:
        report.cold_equals_reference = True
        for name in views:
            if not torch.equal(cold[0][name], reference[name]):
                report.cold_equals_reference = False
                report.cold_mismatch = _mismatch(case, "cold", 0, name, cold[0][name], reference[name])
                break
    expected = reference if reference is not None else cold[0]
    plan = [("warm", int(warm)), ("poisoned", int(poisoned))]
    if report.wrong_cold and (stop_at_first or reference is None):
        report.blocks = [BlockResult(block, requested, stopped="not run: the cold reference is wrong") for block, requested in plan if requested > 0]
        return report

    # ---- the launch blocks ----
    started = time.monotonic()
    remaining_blocks = sum(1 for _, requested in plan if requested > 0)
    for block, requested in plan:
        if requested <= 0:
            continue
        result = BlockResult(block, requested)
        report.blocks.append(result)
        before = None
        if block == "poisoned":
            before = poison
            if before is None:
                before, why = _poison_launcher(device)
                if before is None:
                    result.stopped = f"not run: no poison launch ({why})"
                    remaining_blocks -= 1
                    continue
        deadline = None
        if budget_s is not None:
            deadline = time.monotonic() + max(budget_s - (time.monotonic() - started), 0.0) / remaining_blocks
        remaining_blocks -= 1
        t0 = time.monotonic()
        for index in range(requested):
            if deadline is not None and index % _CHECK_CLOCK_EVERY == 0 and time.monotonic() > deadline:
                result.stopped = "budget"
                break
            _prepare(case, views, device)
            if before is not None:
                before()
            case.launch()
            torch.cuda.synchronize(device)
            result.launches += 1
            found = _first_difference(case, block, index, views, expected)
            if found is not None:
                result.failures += 1
                result.quarters[min(4 * index // requested, 3)] += 1
                if result.first is None:
                    result.first = found
                if stop_at_first:
                    result.stopped = "first failure"
                    break
        result.seconds = time.monotonic() - t0
        if result.stopped == "first failure":
            for later_block, later_requested in plan[plan.index((block, requested)) + 1 :]:
                if later_requested > 0:
                    report.blocks.append(BlockResult(later_block, later_requested, stopped="not run: stopped at the first failure"))
            break
    return report


def stress_validation(device=None, *, cluster: int = 1, launches: int = CONTROL_LAUNCHES) -> Validation:
    """The positive control, once per process, device and launch shape: does the unsafe twin of the test ring fail HERE?

    ``cluster``: the cluster size of the control launch (1, or 2 for the control of a kernel launched in clusters of two).
    The result is cached and recorded through ``frost_routing.measured`` so that it reaches the job log on a pass too.
    The control ring is built here unless :func:`build_launchers` built it already (in front of the caller's watchdog).
    """
    device = torch.device("cuda") if device is None else torch.device(device)
    key = (device.index if device.index is not None else torch.cuda.current_device(), int(cluster))
    if key in _VALIDATION:
        return _VALIDATION[key]
    part = _part(device)
    ring, why = _control_ring(device, cluster)
    if ring is None:
        validation = Validation(False, f"warm stress not validated on {part}: the control ring could not be built ({why})", available=False)
    else:
        import warm_stress_ring

        case = StressCase(
            name=f"control: unsafe twin of the test ring (cluster {cluster})",
            launch=ring.launch,
            outputs={"sums": ring.out},
            thread_blocks=ring.blocks,
            refills_per_launch=ring.refills_per_launch,
            reference={"sums": ring.reference},
            locate=warm_stress_ring.locate,
        )
        report = warm_stress(case, warm=launches, poisoned=0, flush_mib=0, stop_at_first=False, device=device)
        failures, done = report.failures, report.launches
        if failures > 0:
            text = f"warm stress validated on {part}: the unsafe twin of the test ring failed {failures} of {done} launches ({ring.blocks} thread blocks, cluster {cluster})"
        else:
            text = f"warm stress not validated on {part}: the unsafe twin read 0 of {done} ({ring.blocks} thread blocks, cluster {cluster})"
        validation = Validation(failures > 0, text, done, failures)
    _VALIDATION[key] = validation
    _CONTROL.pop(key, None)  # the result is cached; the ring and its input are not kept
    try:
        import frost_routing

        frost_routing.measured(f"warm_stress control cluster {cluster}", validation.text)
    except Exception:  # outside the test tree's pytest session: the text is in the report
        pass
    return validation


def assert_warm_stress(case: StressCase, *, validation: Optional[Validation] = None, budget_s: Optional[float] = DEFAULT_BUDGET_S, **kwargs) -> StressReport:
    """The test entry: FAIL on any differing byte, SKIP where the stress is not validated or did not complete, else pass.

    A mismatch fails whether or not the control reproduced.  A zero count is a pass only when the control saw the unsafe
    twin fail on this GPU in this process, every block ran its full count, and the reference is independent or cold (a case
    without ``reference`` whose reference launches ran without the cache flush has neither); otherwise the test is skipped
    -- a coverage gap, neither a pass nor a failure.  The record goes through ``frost_routing.measured`` in every outcome.
    """
    import pytest

    first_out = next(iter(case.outputs.values()))
    device = kwargs.get("device") or first_out.device
    if validation is None:
        validation = stress_validation(device)
    report = warm_stress(case, budget_s=budget_s, **kwargs)
    report.validation = validation
    text = report.text()
    try:
        import frost_routing

        frost_routing.measured(f"warm_stress {case.name}", text)
    except Exception:
        pass
    if report.failures or report.wrong_cold:
        raise AssertionError(text)
    if not validation.reproduced:
        pytest.skip(text)
    if not report.complete:
        pytest.skip("warm stress incomplete (a coverage gap, not a pass)\n" + text)
    if case.reference is None and not report.cold_flushed:
        pytest.skip("warm stress incomplete (a coverage gap, not a pass): no independent reference and the reference launches were not cold\n" + text)
    return report


# ---------------------------------------------------------------------------
# The rendering pin: is the address of an arrive computed from the loaded registers through the kernel-argument mask?
# ---------------------------------------------------------------------------

_STAGE_LOADS = ("ld.shared", "tcgen05.ld")
_REGISTER = re.compile(r"%[a-z]+\d+")
_NO_RESULT = ("st.", "bra", "barrier", "bar.", "cp.async", "fence", "mbarrier.init", "ret", "prefetch", "griddepcontrol", "trap", "exit")


@dataclass
class ArriveSite:
    line: int
    text: str
    dependent: bool  # the address reaches an `and` whose mask comes from a kernel parameter
    parameter: str = ""  # the parameter symbol of the mask
    loaded_registers: int = 0  # distinct shared/TMEM destination registers the masked value is computed from
    loads: int = 0  # shared/TMEM load instructions among them


def release_dependence(ptx_text: str) -> List[ArriveSite]:
    """Every ``mbarrier.arrive`` (not ``expect_tx``) of a kernel's PTX with what its address is computed from.

    A definition-use walk, not an instruction pattern: from the address operand of the arrive backwards through every
    instruction that defines a register on the way (flow-insensitive: every definition of a register counts).  The site is
    ``dependent`` when the walk meets an ``and`` one of whose operands is computed from a kernel parameter alone
    (``ld.param``) -- the release-zero -- and it reports how many shared-memory or TMEM destination registers the other operand is
    computed from.  The walk does not look through a shared-memory load's own address.
    """
    instructions = []  # (line number, predicate register, opcode, results, sources, parameter symbol, text)
    for number, raw in enumerate(ptx_text.splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith(("//", ".", "$", "{", "}")) or text.endswith(":"):
            continue
        text = text.rstrip(";").strip()
        predicate = ""
        if text.startswith("@"):
            head, _, text = text.partition(" ")
            predicate = head.lstrip("@!")
            text = text.strip()
        opcode, _, operands = text.partition(" ")
        operands = operands.strip()
        opcode = opcode.strip()
        results, rest = [], operands
        if not opcode.startswith(_NO_RESULT):
            if operands.startswith("{"):
                group, _, rest = operands.partition("}")
                results = _REGISTER.findall(group)
            else:
                first, _, rest = operands.partition(",")
                results = _REGISTER.findall(first) if first.strip().startswith("%") else []
                if not results:
                    rest = operands
        sources = _REGISTER.findall(rest) + ([predicate] if predicate.startswith("%") else [])
        symbol = ""
        if opcode.startswith("ld.param"):
            match = re.search(r"\[([^\]]+)\]", operands)
            symbol = match.group(1) if match else "?"
        instructions.append((number, predicate, opcode, results, sources, symbol, raw.strip()))

    definitions: Dict[str, List[int]] = {}
    for index, (_, _, _, results, _, _, _) in enumerate(instructions):
        for register in results:
            definitions.setdefault(register, []).append(index)

    def closure(registers):
        """Indices of every instruction the registers are computed from; stage loads and parameters end the walk."""
        seen, todo = set(), list(registers)
        visited = set()
        while todo:
            register = todo.pop()
            if register in visited:
                continue
            visited.add(register)
            for index in definitions.get(register, ()):
                if index in seen:
                    continue
                seen.add(index)
                opcode = instructions[index][2]
                if opcode.startswith((*_STAGE_LOADS, "ld.param")):
                    continue
                todo.extend(instructions[index][4])
        return seen

    sites = []
    for number, _, opcode, _, _, _, text in instructions:
        if not opcode.startswith("mbarrier.arrive") or "expect_tx" in opcode:
            continue
        match = re.search(r"\[([^\]]+)\]", text)
        address = _REGISTER.findall(match.group(1)) if match else []
        site = ArriveSite(number, text, False)
        for index in sorted(closure(address)):
            _, _, and_opcode, _, and_sources, _, _ = instructions[index]
            if not and_opcode.startswith("and.") or len(and_sources) != 2:
                continue
            for mask, value in ((and_sources[0], and_sources[1]), (and_sources[1], and_sources[0])):
                mask_from = closure([mask])
                symbols = {instructions[i][5] for i in mask_from if instructions[i][2].startswith("ld.param")}
                if not symbols or any(instructions[i][2].startswith(_STAGE_LOADS) for i in mask_from):
                    continue
                loads = [i for i in closure([value]) if instructions[i][2].startswith(_STAGE_LOADS)]
                wanted = closure([value])
                registers = set()
                for i in loads:
                    for register in instructions[i][3]:
                        # a destination register counts when some instruction of the walk reads it
                        if any(register in instructions[j][4] for j in wanted if j != i) or register == value:
                            registers.add(register)
                if len(registers) > site.loaded_registers:
                    site.dependent = True
                    site.parameter = sorted(symbols)[0]
                    site.loaded_registers = len(registers)
                    site.loads = len(loads)
        sites.append(site)
    return sites


def assert_release_dependence(ptx_text: str, *, sites: int, words: int) -> List[ArriveSite]:
    """Pin a rendering: exactly ``sites`` arrives whose address is computed, through an ``and`` with a kernel parameter, from at
    least ``words`` loaded shared-memory or TMEM registers each.  Returns every arrive site of the PTX for further checks."""
    found = release_dependence(ptx_text)
    dependent = [site for site in found if site.dependent]
    listing = "\n".join(f"  line {s.line}: dependent={s.dependent} loaded registers={s.loaded_registers} ({s.loads} loads) :: {s.text}" for s in found)
    assert (
        len(dependent) == sites
    ), f"expected {sites} arrive site(s) that depend on loaded values through the kernel-argument mask, found {len(dependent)}:\n{listing}"
    for site in dependent:
        assert (
            site.loaded_registers >= words
        ), f"the arrive at PTX line {site.line} is computed from {site.loaded_registers} loaded registers, expected at least {words}:\n{listing}"
    return found
