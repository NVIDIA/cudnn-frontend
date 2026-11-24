# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the row argmax decode step's test modules: the reference, the operand builders (built on the HOST), the id
tables, the rows that state the rule one case at a time, and the device gate.  Not a test module (no ``test_`` prefix); imported
by name from the sibling modules (pytest puts their directory on ``sys.path``).

THE REFERENCE (:func:`reference_columns`) is written here from the rule's words -- torch on the host; nothing of the package's
kernel or tile library is imported -- and it is a statement about every byte of the result:

    the column of a row is the LOWEST column whose value is greater than -inf and not below any other value of the row;
    0 when the row holds no such value.  A NaN is greater than nothing and below nothing.  +0 and -0 are equal.

With a table the result is the table's element at that column, bit for bit; without one it is the column as int32.  The rule is
exact, so the reference is the kernel's result on every part: there is no bracket and no tolerance anywhere in these suites.

THE RULE ROWS (:func:`rule_rows`) carry their column BY HAND, next to the sentence of the rule each one stands for; a host cell
holds the reference to those hand-written columns, and the device cells hold the kernel to both.
"""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.row_argmax.frost import row_argmax_decode as entry  # noqa: E402

WIDTH = 65536
BF16, I32, I64 = torch.bfloat16, torch.int32, torch.int64
TABLES = ("none", "int32", "int64")
RESULT = {"none": I32, "int32": I32, "int64": I64}
INF = float("inf")
NAN = float("nan")
#: the thread shape of the served form, as the rule rows place their ties: 1024 threads, eight elements per thread per pass
THREADS, VEC = 1024, 8


def make_entry(device, width=WIDTH):
    return entry.RowArgmaxDecode(device, width)


def column_of(thread, scan=0, element=0):
    """The column the thread ``thread`` reads as element ``element`` of its vector of pass ``scan``."""
    return VEC * (thread + THREADS * scan) + element


# ---------------------------------------------------------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------------------------------------------------------
def reference_columns(x):
    """The rule on a host copy of ``x`` (bf16 ``[rows, width]``): int64 ``[rows]``.  A value is ELIGIBLE when it is greater than
    -inf (a NaN and -inf are not); the row's column is the lowest eligible column whose value equals the largest eligible value,
    and 0 when no column is eligible."""
    x = x.detach().cpu()
    rows, width = x.shape
    position = torch.arange(width, dtype=I64)
    columns = torch.empty(rows, dtype=I64)
    for start in range(0, rows, 64):  # bounded host memory at the row ceiling
        values = x[start : start + 64].float()  # bf16 -> fp32 is exact
        eligible = values > -INF
        largest = torch.where(eligible, values, torch.full_like(values, -INF)).amax(dim=1, keepdim=True)
        holds = eligible & (values == largest)
        lowest = torch.where(holds, position, torch.full_like(position, width)).amin(dim=1)
        columns[start : start + 64] = torch.where(lowest == width, torch.zeros_like(lowest), lowest)
    return columns


def reference(x, ids=None):
    """The result of a call on the host: the column as int32 without a table, else the table's element at the column."""
    columns = reference_columns(x)
    return columns.to(I32) if ids is None else ids.detach().cpu()[columns]


def result_bytes(t):
    """A result tensor's bytes on the host."""
    return t.detach().cpu().contiguous().view(torch.uint8)


def assert_rule(got, x, ids=None, label=""):
    """Every byte of a launch's result is the reference's: the dtype, the shape and the bytes."""
    want = reference(x, ids)
    assert got.dtype == want.dtype and tuple(got.shape) == tuple(
        want.shape
    ), f"{label}: result {got.dtype} {tuple(got.shape)}, want {want.dtype} {tuple(want.shape)}"
    wrong = (got.detach().cpu() != want).nonzero().reshape(-1)
    assert (
        wrong.numel() == 0
    ), f"{label}: {wrong.numel()} of {want.numel()} rows differ from the rule (first: row {int(wrong[0])}, got {int(got[wrong[0]])}, want {int(want[wrong[0]])})"
    assert torch.equal(result_bytes(got), result_bytes(want)), f"{label}: the result's bytes"


# ---------------------------------------------------------------------------------------------------------------------------
# operands and id tables (built on the host: one seed gives one set of tensors on every GPU)
# ---------------------------------------------------------------------------------------------------------------------------
def make_rows(rows, device="cpu", seed=0, kind="random", width=WIDTH):
    """``x`` bf16 ``[rows, width]`` from a CPU generator, moved to ``device``.  ``random``: normal values under a per-row
    magnitude spread over twelve octaves (bf16 holds eight significant bits, so the largest value of a row is often held by
    several columns).  ``ties``: small integers, so every row holds its maximum thousands of times.  ``special``: the random
    draw with, by row index modulo six, NaN scattered through the row; NaN and -inf; +inf (several columns hold it); a row of
    negative values with zeros of both signs (the maximum is a zero); all of them; nothing -- and whole rows of -inf, of NaN, of
    zeros and of negative zeros at fixed row indices."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "ties":
        x = torch.randint(-3, 4, (rows, width), generator=gen).to(BF16)
    elif kind in ("random", "special"):
        magnitude = torch.exp2(torch.empty(rows, 1).uniform_(-6.0, 6.0, generator=gen))
        x = (torch.randn(rows, width, generator=gen) * magnitude).to(BF16)
        if kind == "special":
            draw = torch.rand(rows, width, generator=gen)
            flavour = torch.arange(rows).reshape(rows, 1) % 6
            x = torch.where(flavour == 3, -x.abs(), x)
            for flavours, low, high, value in (
                ((0, 1, 4), 0.00, 0.02, NAN),
                ((1, 4), 0.02, 0.04, -INF),
                ((2, 4), 0.04, 0.0402, INF),
                ((3, 4), 0.05, 0.0502, 0.0),
                ((3, 4), 0.06, 0.0602, -0.0),
            ):
                member = torch.zeros(rows, 1, dtype=torch.bool)
                for one in flavours:
                    member |= flavour == one
                x[member & (draw >= low) & (draw < high)] = value
            for row, value in ((7, -INF), (9, NAN), (10, 0.0), (11, -0.0)):
                if row < rows:
                    x[row] = value
    else:
        raise ValueError(kind)
    return x.to(device)


def make_table(kind, device="cpu", seed=0, width=WIDTH):
    """The id table of a kind: ``None`` for ``"none"``; else a permutation of the columns pushed through a map that no column
    order survives -- int32 ids of both signs, int64 ids whose upper 32 bits differ from id to id (a 32-bit copy of an id, or the
    id of a neighbouring column, is another value)."""
    if kind == "none":
        return None
    gen = torch.Generator(device="cpu").manual_seed(1000 + seed)
    permutation = torch.randperm(width, generator=gen)
    if kind == "int32":
        return (permutation * 32749 - 1_000_000_007).to(I32).to(device)
    assert kind == "int64", kind
    return (permutation.to(I64) * 0x1_0003_0007 - (1 << 45)).to(device)


def rule_rows(width=WIDTH):
    """``(names, x, columns)``: one row per sentence of the rule, with the column it must give written BY HAND.  ``x`` is bf16
    ``[len(names), width]`` on the host.  Unless a case says otherwise the row is filled with -7 and the named columns are set."""
    last = width - 1
    lowest_finite = float(torch.tensor(-0x0081, dtype=torch.int16).view(BF16))  # the pattern 0xFF7F: the most negative finite bf16
    largest_finite = float(torch.tensor(0x7F7F, dtype=torch.int16).view(BF16))
    least_positive = float(torch.tensor(0x0001, dtype=torch.int16).view(BF16))  # the smallest bf16 subnormal
    cases = []

    def case(name, column, fill=-7.0, **at):
        row = torch.full((width,), fill, dtype=BF16)
        for where, value in at.items():
            row[int(where[1:])] = value
        cases.append((name, row, column))

    def at(**columns):
        return {f"c{column}": value for column, value in columns.items()}

    case("the maximum in column 0", 0, c0=3.0)
    case("the maximum in the last column", last, **{f"c{last}": 3.0})
    case("every element equal: the lowest column", 0, fill=2.5)
    case("a row of zeros: column 0", 0, fill=0.0)
    case("a row of negative zeros: column 0", 0, fill=-0.0)
    case("-0 below +0 in column order: equal, the lower column", 4, fill=-1.0, c4=-0.0, c9=0.0)
    case("+0 below -0 in column order: equal, the lower column", 4, fill=-1.0, c4=0.0, c9=-0.0)
    case("a row of -inf: no value above -inf, column 0", 0, fill=-INF)
    case("a row of NaN: no value above -inf, column 0", 0, fill=NAN)
    case("NaN then -inf: no value above -inf, column 0", 0, fill=-INF, c0=NAN)
    case("NaN in column 0 beside values: never chosen", 1, fill=-1.0, c0=NAN, c1=5.0, c2=4.0)
    case("NaN everywhere but one column", 40000, fill=NAN, c40000=-3.0)
    case("NaN in the columns before and behind the maximum", 30001, c30000=NAN, c30001=1.0, c30002=NAN)
    case("one value above -inf, the most negative finite one", 777, fill=-INF, c777=lowest_finite)
    case("+inf is a value: its lowest column", 20, fill=largest_finite, c500=INF, c20=INF)
    case("the largest finite value below nothing but +inf", 12345, c12345=largest_finite)
    case("-inf beside finite values is never the maximum", 9, fill=-INF, c9=lowest_finite, c8=NAN)
    case("the smallest positive value above a row of zeros", 9, fill=0.0, c9=least_positive)
    case("zero above a row of negative values", 60000, fill=-least_positive, c60000=-0.0)
    case("a tie inside one vector of one thread", column_of(2, 0, 0), **at(**{str(column_of(2, 0, 0)): 1.0, str(column_of(2, 0, 3)): 1.0}))
    case("a tie in one thread across two passes", column_of(5, 0, 1), **at(**{str(column_of(5, 0, 1)): 1.0, str(column_of(5, 1, 0)): 1.0}))
    case("a tie across two threads of one warp", column_of(3), **at(**{str(column_of(3)): 1.0, str(column_of(7)): 1.0}))
    case("a tie across threads: the lower column in the HIGHER thread", column_of(7, 2), **at(**{str(column_of(7, 2)): 1.0, str(column_of(3, 5)): 1.0}))
    case("a tie across two warps", column_of(40), **at(**{str(column_of(40)): 1.0, str(column_of(1000)): 1.0}))
    case("a tie across warps: the lower column in the HIGHER warp", column_of(1000), **at(**{str(column_of(1000)): 1.0, str(column_of(40, 3)): 1.0}))
    case("a tie between the first and the last column", 0, **at(**{"0": 1.0, str(last): 1.0}))
    case(
        "the maximum in the last element of the last thread's first vector",
        column_of(THREADS - 1, 0, VEC - 1),
        **at(**{str(column_of(THREADS - 1, 0, VEC - 1)): 1.0}),
    )
    names = [name for name, _, _ in cases]
    return names, torch.stack([row for _, row, _ in cases]), torch.tensor([column for _, _, column in cases], dtype=I64)


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the row argmax decode step serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry._dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    return dev
