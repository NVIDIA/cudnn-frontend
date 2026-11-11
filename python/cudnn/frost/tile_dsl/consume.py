# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Consume, then arrive: an mbarrier release whose address depends on the values loaded from the stage it frees.

THE HAZARD.  A consumer that reads a pipeline stage with ordinary shared-memory loads (``ld.shared``, ``ldmatrix``) and then
releases the stage -- any arrive after which the producer may refill it -- can receive the bytes of the NEXT fill in every
load whose destination registers have not been read when the release executes.  Nothing faults and nothing hangs.  Writing
the use in front of the arrive does not help: the compiler moves pure arithmetic, compares, warp votes and shuffles behind
a release that does not depend on them.  ``python/cudnn/AGENTS.md`` Rule 11 has the rule and what was measured.

WHAT THIS MODULE EMITS.  Every form computes

    off  = (wrapping 32-bit sum of every loaded word) & release_zero
    addr = mb + off elements                      (8 * off bytes for an mbarrier array)

and arrives on ``addr``.  ``release_zero`` is a kernel argument that the kernel's own host passes as the literal 0, so
``off`` is 0 for every data value and the arrive lands exactly where the plain arrive would.  Neither the optimizer nor the
assembler knows the argument's value, so the address register of the arrive is computed from every loaded register: each of
them is read in front of the release in the generated code.  For W loaded 32-bit words the cost is ceil(W / 2) + 2 integer
instructions per release (three-input adds, one AND, one multiply-add); no load, no store, no shared-memory word.

THE CONTRACT.

* ``consumed`` is REQUIRED: every value the calling lanes loaded from the stage since they acquired it -- scalars, vectors,
  register tiles, nested tuples / lists.  64-bit values contribute both halves, 8- and 16-bit values their zero-extended bits,
  floats their bit patterns.  Pass each loaded value ONCE and pass the loaded value itself, never a value derived from it
  (a difference, a negation, a checksum of other passed values): a derived term can cancel the term it was computed from and
  leave that load unread, with no error anywhere.  ``None``, an empty sequence, a Python number, a compile-time constant, a
  Boolean and the same value twice are refused at trace time.
* ``zero`` is REQUIRED: the :class:`ReleaseZero` made by :func:`release_zero` from the kernel's release-zero argument.
* THE RELEASE-ZERO ARGUMENT HAS EXACTLY ONE USE IN THE KERNEL: ``release_zero(arg)``.  Never compare it, branch on it, assert
  or assume its value, or use it in any other expression.  A kernel that knows the value has no ordering left: an entry guard
  ``if arg == 0:`` round the consumer lets the optimizer replace the mask by a constant, the fold disappears and the loads
  are unread at the release again -- without any error.  The forms refuse to trace when the argument already has another
  use.  A non-zero argument is undefined behaviour (the arrive lands on another word of shared memory); nothing in the kernel
  may test for it.  A zero the compiler or the assembler can see (a literal, an inline-asm ``mov``) removes the ordering the
  same way, which is why the mask has to come from outside the kernel.
* LANE LEDGER UNCHANGED.  No form elects for the caller.  Each arrives where and as often as its plain twin
  (``barrier.arrive``, ``mbar_arrive_on_peer``, ``scheduler.read_tile_id_arrive``): one arrive (times ``count``) per calling
  lane whose ``pred`` is true.  Init counts do not move when a site changes over.
* ELECTED ARRIVES TAKE ``pred=``.  Every calling lane computes the address, one predicated arrive fires.  Keep the call out
  of a lane branch: the unconditional fold in front of a predicated arrive is the form that was measured.
* SCOPE.  The forms order the calling lanes' own loads; a warp that loads releases through them itself.  A release without
  an address operand (a CTA barrier in front of a refill, a peer send) is outside this module: Rule 11 names the alternatives.

    @cute.kernel
    def kernel(..., rz_arg: cutlass.Int32):            # the release-zero: one more kernel argument, last
        rz = release_zero(rz_arg)                      # once, at the top of the kernel
        ...
        w = stage.load(base, vector_size=4, alignment=16)
        consume_then_arrive(mb_empty[s], w, rz, pred=nvvm.elect_sync())
        ... w is used here, or later ...

    @cute.jit
    def host(...):
        kernel(..., cutlass.Int32(0)).launch(...)       # the only value ever passed

``test/python/sdpa/frost/test_tile_dsl_consume_then_arrive.py`` pins the generated chain of every form and runs it on a ring
whose unsafe twin fails; ``test/python/warm_stress.py`` is the run-time detector for a kernel that adopts it.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import vector as _vector
from cutlass.experimental import primitives as nvvm

from .barrier import MBarrier
from .regtile import RegTile
from .scheduler import _read_tile_id_lane_geometry

__all__ = [
    "ReleaseZero",
    "consume_then_arrive",
    "consume_then_arrive_on_peer",
    "consume_then_read_tile_id_arrive",
    "release_zero",
]

_MASK_OP = "arith.andi"


class ReleaseZero:
    """The kernel's release-zero argument, checked once by :func:`release_zero`.  A trace-time object: it emits no code."""

    __slots__ = ("_word",)

    def __init__(self, word):
        self._word = word

    @property
    def word(self):
        return self._word


def _same_value(a, b) -> bool:
    # The DSL's value wrappers overload ``==`` to EMIT a compare; identity of two IR values is the base-class comparison.
    return bool(_ir.Value.__eq__(a, b))


def _is_ir_constant(value) -> bool:
    owner = value.owner
    return not isinstance(owner, _ir.Block) and getattr(owner, "name", "") == "arith.constant"


def release_zero(kernel_arg) -> ReleaseZero:
    """Wrap the kernel's release-zero ARGUMENT for the ``consume_then_*`` forms.

    ``kernel_arg`` must be the 32-bit integer parameter of the enclosing ``@cute.kernel`` itself -- the one its host passes as
    the literal 0 -- and this call must be its only use (module docstring).  A Python number, a compile-time constant, a
    computed value or a value of another type raises ``TypeError``: a mask the compiler can see through removes the ordering
    without any other error.
    """
    if not isinstance(kernel_arg, (cutlass.Int32, cutlass.Uint32)):
        raise TypeError(
            f"release_zero: expected the kernel's Int32 release-zero argument, got {type(kernel_arg).__name__}; a Python number "
            "or another type cannot carry the ordering (the mask must be a kernel argument the host passes as 0)"
        )
    value = kernel_arg.value
    if not isinstance(value, _ir.Value):
        raise TypeError(
            f"release_zero: got the compile-time constant {value!r}; the release-zero must be a KERNEL ARGUMENT -- a zero the "
            "compiler can see is folded away and the loads are unread at the release again"
        )
    owner = value.owner
    if not isinstance(owner, _ir.Block):
        what = "a constant" if _is_ir_constant(value) else f"the result of {getattr(owner, 'name', type(owner).__name__)}"
        raise TypeError(f"release_zero: got {what}; pass the kernel's own release-zero parameter, unmodified")
    parent = owner.owner
    regions = getattr(parent, "regions", None)
    if regions is None or len(regions) == 0 or len(regions[0].blocks) == 0 or regions[0].blocks[0] != owner:
        raise TypeError("release_zero: the value is a region argument (a loop- or branch-carried value), not a parameter of the kernel")
    zero = ReleaseZero(kernel_arg)
    _check_zero(zero, "release_zero")
    return zero


def _i32(value):
    return cutlass.Int32(value)


def _scalar_words(value, who: str) -> list:
    """The 32-bit words of ONE scalar IR value (a read of every bit of it), as ``Int32``."""
    ty = value.type
    i32 = _ir.IntegerType.get_signless(32)
    if _ir.IntegerType.isinstance(ty):
        width = _ir.IntegerType(ty).width
        if width == 1:
            raise TypeError(f"{who}: a Boolean is a compare result, not a loaded value -- pass the loaded word itself")
        if width == 32:
            return [_i32(value)]
        if width < 32:
            return [_i32(_arith.extui(i32, value))]
        if width == 64:
            hi = _arith.shrui(value, _arith.constant(_ir.IntegerType.get_signless(64), 32))
            return [_i32(_arith.trunci(i32, value)), _i32(_arith.trunci(i32, hi))]
        raise TypeError(f"{who}: unsupported integer width {width} in the consumed values")
    if _ir.F32Type.isinstance(ty):
        return [_i32(_arith.bitcast(i32, value))]
    if _ir.F16Type.isinstance(ty) or _ir.BF16Type.isinstance(ty):
        return [_i32(_arith.extui(i32, _arith.bitcast(_ir.IntegerType.get_signless(16), value)))]
    if _ir.F64Type.isinstance(ty):
        return _scalar_words(_arith.bitcast(_ir.IntegerType.get_signless(64), value), who)
    raise TypeError(f"{who}: unsupported type {ty} in the consumed values")


def _value_words(value, who: str) -> list:
    if _ir.VectorType.isinstance(value.type):
        count = 1
        for extent in _ir.VectorType(value.type).shape:
            count *= int(extent)
        if count < 1 or len(_ir.VectorType(value.type).shape) != 1:
            raise TypeError(f"{who}: unsupported vector type {value.type} in the consumed values")
        words = []
        for index in range(count):
            words.extend(_scalar_words(_vector.extract(value, [], [index]), who))
        return words
    return _scalar_words(value, who)


def _extract_source(value):
    """``(source vector, static position)`` when ``value`` is one element extracted from a vector, else ``None``."""
    owner = value.owner
    if isinstance(owner, _ir.Block) or getattr(owner, "name", "") != "vector.extract":
        return None
    try:
        position = tuple(int(p) for p in owner.opview.static_position)
        return owner.operands[0], position
    except Exception:  # an extract with a dynamic position: nothing to compare
        return None


def _consumed_values(consumed, who: str) -> list:
    """Flatten ``consumed`` into its IR values, refusing what cannot carry the ordering (module docstring)."""
    values = []

    def visit(item):
        if item is None:
            raise TypeError(f"{who}: `consumed` holds None; pass every value the calling lanes loaded from the stage")
        if isinstance(item, (tuple, list)):
            if len(item) == 0:
                raise TypeError(f"{who}: `consumed` holds an empty sequence; there is no spelling that passes nothing")
            for element in item:
                visit(element)
            return
        if isinstance(item, RegTile):
            item = item.vec
        if isinstance(item, (bool, int, float)):
            raise TypeError(f"{who}: `consumed` holds the Python number {item!r}; pass the loaded values")
        if isinstance(item, ReleaseZero):
            raise TypeError(f"{who}: the release-zero is not a consumed value (argument order: mb, consumed, zero)")
        if isinstance(item, cutlass.Boolean):
            raise TypeError(f"{who}: `consumed` holds a Boolean; a compare result is not a loaded value -- pass the loaded word itself")
        if isinstance(item, cutlass.Numeric):
            if not isinstance(item.value, _ir.Value):
                raise TypeError(f"{who}: `consumed` holds the compile-time constant {item.value!r}; pass the loaded values")
            item = item.value
        if not isinstance(item, _ir.Value):
            raise TypeError(f"{who}: unsupported object of type {type(item).__name__} in `consumed`")
        if _is_ir_constant(item):
            raise TypeError(f"{who}: `consumed` holds a constant; pass the loaded values")
        values.append(item)

    visit(consumed)
    for index, value in enumerate(values):
        for earlier in values[:index]:
            if _same_value(value, earlier):
                raise TypeError(f"{who}: the same value is passed twice in `consumed`; pass each loaded value once")
        source = _extract_source(value)
        if source is None:
            continue
        for other_index, other in enumerate(values):
            if other_index == index:
                continue
            if _same_value(other, source[0]):
                raise TypeError(f"{who}: `consumed` holds a vector and one of its elements; pass each loaded value once")
            other_source = _extract_source(other)
            if other_index < index and other_source is not None and other_source[1] == source[1] and _same_value(other_source[0], source[0]):
                raise TypeError(f"{who}: the same vector element is passed twice in `consumed`; pass each loaded value once")
    return values


def _check_zero(zero, who: str):
    if not isinstance(zero, ReleaseZero):
        raise TypeError(f"{who}: `zero` must be the ReleaseZero returned by release_zero(<the kernel's release-zero argument>), got {type(zero).__name__}")
    for use in zero.word.value.uses:
        name = use.owner.name
        if name != _MASK_OP:
            raise TypeError(
                f"{who}: the kernel's release-zero argument has a use that is not this module's mask ({name}).  Its only use may "
                "be release_zero(arg): a compare, branch, assert or any other expression on it lets the compiler replace the mask "
                "by a constant, and the loads are unread at the release again without any error"
            )


def _check_pred(pred, who: str):
    if pred is not None and not isinstance(pred, (cutlass.Boolean, _ir.Value)):
        raise TypeError(f"{who}: `pred` must be a lane predicate computed in the kernel (e.g. nvvm.elect_sync()) or None, got {pred!r}")
    if isinstance(pred, cutlass.Boolean) and not isinstance(pred.value, _ir.Value):
        raise TypeError(f"{who}: `pred` is the compile-time constant {pred.value!r}; omit it for an arrive on every calling lane")


def _release_address(mb, consumed, zero, who: str):
    """The barrier address plus ``(sum of every consumed word) & release_zero`` elements: the plain address at run time."""
    _check_zero(zero, who)
    fold = None
    for value in _consumed_values(consumed, who):
        for word in _value_words(value, who):
            fold = word if fold is None else fold + word
    offset = fold & zero.word
    base = mb.smem_ptr if isinstance(mb, MBarrier) else mb
    if hasattr(base, "subview"):
        return base.subview(offset)
    return base + offset


def _check_count(count, who: str):
    if count is None:
        return
    if isinstance(count, bool) or (isinstance(count, int) and count < 1) or not isinstance(count, (int, cutlass.Int32, cutlass.Uint32)):
        raise TypeError(f"{who}: `count` must be a positive arrival count (an int or an Int32 value), got {count!r}")


def _count_register(count):
    """The arrive count as a REGISTER operand of a predicated inline-PTX arrive.

    ``inline_ptx(predicate=)`` can be lowered onto a trailing immediate operand's value (``barrier.arrive_expect_tx`` has the
    record), so the count never reaches the template as an immediate: a Python int is materialised by a ``mov``, a traced
    value passes through an identity ``mov``.
    """
    if isinstance(count, int):
        return nvvm.inline_ptx(f"mov.b32 $0, {count};", write_only_types=[cutlass.Int32])
    return nvvm.inline_ptx("mov.b32 $0, $1;", write_only_types=[cutlass.Int32], read_only_args=[cutlass.Int32(count)])


@cute.jit
def consume_then_arrive(mb, consumed, zero, *, pred=None, count=None):
    """Local arrive on ``mb`` (a barrier pointer or an ``MBarrier``) that depends on every value in ``consumed``.

    ``pred=None``: one arrive per calling lane, as ``barrier.arrive``.  ``pred=<lane predicate>``: every calling lane computes
    the address, ONE predicated ``mbarrier.arrive.shared::cta.b64`` fires on the lanes whose predicate is true (the elected
    form: ``pred=nvvm.elect_sync()``).  ``count=n``: each arriving lane arrives ``n`` times in one instruction; the barrier's
    init count is ``n`` x the arriving lanes.
    """
    _check_pred(pred, "consume_then_arrive")
    _check_count(count, "consume_then_arrive")
    addr = _release_address(mb, consumed, zero, "consume_then_arrive")
    if cutlass.const_expr(pred is None):
        if cutlass.const_expr(count is None):
            nvvm.mbarrier_arrive(addr)
        else:
            nvvm.mbarrier_arrive(addr, count=count)
    elif cutlass.const_expr(count is None):
        nvvm.inline_ptx("mbarrier.arrive.shared::cta.b64 _, [{$r0}];", read_only_args=[addr], predicate=pred)
    else:
        nvvm.inline_ptx(
            "mbarrier.arrive.shared::cta.b64 _, [{$r0}], {$r1};",
            read_only_args=[addr, _count_register(count)],
            predicate=pred,
        )


@cute.jit
def consume_then_arrive_on_peer(mb, peer_cta_id, consumed, zero, *, pred=None, release_cta: cutlass.Constexpr[bool] = True):
    """One arrive per calling lane (whose ``pred`` is true) on the PEER CTA's copy of ``mb``; the address depends on ``consumed``.

    The offset is added in the local window, then ``mapa`` translates the address, then the remote arrive fires: the arms of
    ``barrier.mbar_arrive_on_peer``.  ``release_cta=True`` (the default here) is ``mbarrier.arrive.release.cta.shared::cluster``,
    the form that helper documents for a read-then-credit across CTAs; ``release_cta=False`` is the relaxed cluster form, for a
    site whose measurement asks for it.  The loaded registers are the calling CTA's: a release issued by ANOTHER CTA than the
    one that loaded cannot be ordered by this form.  A credit to one CTA only (a leader) is this form with the caller's predicate.
    """
    _check_pred(pred, "consume_then_arrive_on_peer")
    addr = _release_address(mb, consumed, zero, "consume_then_arrive_on_peer")
    peer_mb = nvvm.mapa(addr, peer_cta_id)
    if cutlass.const_expr(pred is None):
        if cutlass.const_expr(release_cta):
            nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CTA, relaxed=False)
        else:
            nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CLUSTER, relaxed=True)
    elif cutlass.const_expr(release_cta):
        nvvm.inline_ptx("mbarrier.arrive.release.cta.shared::cluster.b64 _, [{$r0}];", read_only_args=[peer_mb], predicate=pred)
    else:
        nvvm.inline_ptx("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [{$r0}];", read_only_args=[peer_mb], predicate=pred)


@cute.jit
def consume_then_read_tile_id_arrive(mb, cga_size: int, consumed, zero):
    """The scheduler slot credit of ``scheduler.read_tile_id_arrive`` with every decoded slot word in ``consumed``.

    The calling warp lands ONE arrive on EVERY CTA of the cluster, from the lanes and at the peers of the plain helper's
    predicated arm (``lane_stride = 32 // cga_size``; lane ``c * lane_stride`` credits CTA ``c``), so
    ``SUM(issuing lanes with peer == c) == 1`` per calling warp and CTA and no ``READ_TILE_ARRIVERS`` init count changes.
    ``cga_size == 1`` is the elected local arrive.  Only the predicated lowering exists here; a kernel whose measurement
    prefers the plain helper's branch arm adds that arm together with its own measurement.
    """
    if cutlass.const_expr(cga_size == 1):
        consume_then_arrive(mb, consumed, zero, pred=nvvm.elect_sync())
    else:
        lane_stride, shift = _read_tile_id_lane_geometry(cga_size)  # trace-time; raises on a non-power-of-two cga_size
        lane = cute.arch.thread_idx()[0] & cutlass.Int32(31)
        peer = lane >> cutlass.Int32(shift)
        pred = ((lane & cutlass.Int32(lane_stride - 1)) == cutlass.Int32(0)) & (peer < cutlass.Int32(cga_size))
        consume_then_arrive_on_peer(mb, peer, consumed, zero, pred=pred, release_cta=True)
