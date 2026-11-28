# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise determinism and batch invariance of the RMSNorm forward under the BATCH_INVARIANT note.

Forward only: the note selects a forward engine, and the backward graph does not take it.
"""

from pathlib import Path

import pytest
import torch

from bi_rmsnorm_utils import EPS, adversarial, build, exact, weight
from determinism_utils import assert_bitwise, assert_golden, assert_replays

pytestmark = pytest.mark.L0

CASES = [
    pytest.param(torch.bfloat16, torch.bfloat16, 4096, id="bf16-h4096"),
    pytest.param(torch.bfloat16, torch.bfloat16, 4800, id="bf16-h4800-uneven-lanes"),
    pytest.param(torch.bfloat16, torch.float32, 8192, id="bf16-fp32w-h8192"),
    pytest.param(torch.float16, torch.float16, 5120, id="fp16-h5120"),
    pytest.param(torch.float32, torch.float32, 2048, id="fp32-h2048"),
    pytest.param(torch.bfloat16, torch.bfloat16, 128, id="bf16-h128-subwarp"),
    pytest.param(torch.bfloat16, torch.bfloat16, 100, id="bf16-h100-8byte-granules"),
    pytest.param(torch.bfloat16, torch.bfloat16, 7, id="bf16-h7-2byte-granules"),
    pytest.param(torch.bfloat16, torch.float32, 1002, id="bf16-fp32w-h1002-4byte-granules"),
    pytest.param(torch.float16, torch.float16, 4097, id="fp16-h4097-2byte-granules"),
    pytest.param(torch.float32, torch.float32, 2049, id="fp32-h2049-4byte-granules"),
]
ROWS = 135
GOLDENS = Path(__file__).with_name("goldens")


def _reference_close(x, w, out):
    """An independent FP64 reference, separate from the byte comparisons."""
    xd = x.double()
    ref = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + EPS) * w.double()
    tol = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[out.dtype]
    torch.testing.assert_close(out.double(), ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
@pytest.mark.parametrize("in_dtype,w_dtype,hidden", CASES)
def test_replays_and_exact_model(bi_device, in_dtype, w_dtype, hidden, training):
    x = adversarial(ROWS, hidden, in_dtype, 11, bi_device)
    w = weight(hidden, w_dtype, 12, bi_device)
    graph, run = build(x, w, training=training)
    assert graph.selected_engine.name == "batch_invariant_rmsnorm"
    got = assert_replays(run)
    y, rstd = exact(x, w)
    expected = {"y": y, "rstd": rstd} if training else {"y": y}
    assert_bitwise(got, expected, "exact model")
    _reference_close(x, w, got["y"])


@pytest.mark.parametrize("in_dtype,w_dtype,hidden", CASES)
def test_rows_are_batch_invariant(bi_device, in_dtype, w_dtype, hidden):
    x = adversarial(ROWS, hidden, in_dtype, 21, bi_device)
    w = weight(hidden, w_dtype, 22, bi_device)
    full = build(x, w, training=True)[1]()

    def rows_of(index):
        return {name: value[index] for name, value in full.items()}

    rows = [129, 0, 129, 134, 17]
    assert_bitwise(build(x[rows], w, training=True)[1](), rows_of(rows), "reordered and duplicated rows")
    assert_bitwise(build(x[129:130], w, training=True)[1](), rows_of(slice(129, 130)), "one row")
    assert_bitwise(build(x[:64], w, training=True)[1](), rows_of(slice(0, 64)), "leading rows")
    neighbours = adversarial(ROWS, hidden, in_dtype, 23, bi_device)
    neighbours[129] = x[129]
    changed = build(neighbours, w, training=True)[1]()
    assert_bitwise({name: value[129] for name, value in changed.items()}, rows_of(129), "changed neighbours")


@pytest.mark.parametrize("in_dtype,w_dtype,hidden", CASES)
def test_training_matches_inference(bi_device, in_dtype, w_dtype, hidden):
    x = adversarial(ROWS, hidden, in_dtype, 31, bi_device)
    w = weight(hidden, w_dtype, 32, bi_device)
    assert_bitwise({"y": build(x, w, training=True)[1]()["y"]}, build(x, w)[1](), "training y vs inference y")


@pytest.mark.parametrize("hidden", [128, 4096, 4800, 7168, 100, 1002, 4100, 4097])
def test_every_placement_computes_the_class(bi_device, hidden):
    """Every placement, at granules of 16, 8, 4 and 2 bytes (H = 4096, 100, 1002 and 4097)."""
    from cudnn.norm.batch_invariant._knobs import knob_space
    from cudnn.norm.batch_invariant.kernel import build as build_kernel

    x = adversarial(ROWS, hidden, torch.bfloat16, 41, bi_device)
    w = weight(hidden, torch.bfloat16, 42, bi_device)
    y_ref, rstd_ref = exact(x, w)
    schedule, _ = build_kernel(ROWS, hidden, "bf16", "bf16", "bf16", True, bi_device.index)
    placements = knob_space(schedule.cls)
    assert len(placements) > 1
    stream = torch.cuda.current_stream()
    for knobs in placements:
        sched, compiled = build_kernel(ROWS, hidden, "bf16", "bf16", "bf16", True, bi_device.index, knobs)
        # The scheduled grid, and small grids that make each CTA walk many row groups (the staged prefetch path).
        for grid in (sched.grid, 1, 7):
            y = torch.full_like(x, float("nan"))
            rstd = torch.full((ROWS, 1), float("nan"), dtype=torch.float32, device=bi_device)
            compiled.execute(x.data_ptr(), w.data_ptr(), y.data_ptr(), rstd.data_ptr(), ROWS, hidden, hidden, EPS, grid, stream.cuda_stream)
            assert_bitwise({"y": y, "rstd": rstd}, {"y": y_ref, "rstd": rstd_ref}, f"placement {knobs}, grid {grid}")


def test_rows_across_waves(bi_device):
    """Enough rows for several waves of resident CTAs, compared with small launches of the same rows."""
    from cudnn.norm.batch_invariant._heuristic import resident_ctas
    from cudnn.norm.batch_invariant._schedule import part_for, select_schedule

    hidden, rows = 4096, 24576
    schedule = select_schedule(rows, hidden, "bf16", bi_device.index)
    part, sm_count = part_for(bi_device.index)
    wave = resident_ctas(schedule.knobs, schedule.cls, part) * sm_count * schedule.knobs.rows_per_cta
    assert wave < rows // 2, "the launch must span several waves"
    x = adversarial(rows, hidden, torch.bfloat16, 51, bi_device)
    w = weight(hidden, torch.bfloat16, 52, bi_device)
    full = assert_replays(build(x, w)[1], repeats=3)
    picks = [0, 1, wave, rows // 2, rows - 1]
    assert_bitwise(build(x[picks], w)[1](), {"y": full["y"][picks]}, "rows of later waves")


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_row_count_overrides(bi_device, training):
    """One graph runs every row count through shape overrides, with the bytes of a graph built for that count."""
    from cudnn.norm.batch_invariant._heuristic import reachable
    from cudnn.norm.batch_invariant._schedule import part_for, select_schedule

    hidden, most = 4096, 24576
    x = adversarial(most, hidden, torch.bfloat16, 71, bi_device)
    w = weight(hidden, torch.bfloat16, 72, bi_device)
    graph, run = build(x, w, training=training, override=True)
    # One row count per placement the plan may pick, the smallest that picks it, and the declared count.
    first = {}
    for rows in range(1, most + 1):
        first.setdefault(select_schedule(rows, hidden, "bf16", bi_device.index).knobs, rows)
    assert set(first) == reachable(select_schedule(most, hidden, "bf16", bi_device.index).cls, *part_for(bi_device.index))
    for rows in sorted({*first.values(), most}):
        y_ref, rstd_ref = exact(x[:rows], w)
        expected = {"y": y_ref, "rstd": rstd_ref} if training else {"y": y_ref}
        assert_bitwise(run(rows), expected, f"{rows} rows through overrides")
        assert_bitwise(run(rows), build(x[:rows], w, training=training)[1](), f"{rows} rows against a graph built for them")


@pytest.mark.parametrize(
    "case, declared, width, shape, x_strides, y_strides, match",
    [
        ("undeclared overrides", False, 4096, [32, 4096], [4096, 1], [4096, 1], "is_override_shape_enabled"),
        ("normalized axes", True, 2048, [64, 2048], [2048, 1], [2048, 1], "row axes"),
        ("unaligned row stride", True, 4100, [64, 4096], [4100, 1], [4096, 1], "row strides"),
        ("overlapping Y rows", True, 4096, [64, 4096], [4096, 1], [2048, 1], "Y row stride"),
        ("rows beyond 2^28", True, 4096, [2**28, 4096], [4096, 1], [4096, 1], "index range"),
    ],
)
def test_rejected_overrides_raise(bi_device, case, declared, width, shape, x_strides, y_strides, match):
    """Overrides need a graph that declares them, and may change the row count and an aligned row stride only."""
    w = weight(4096, torch.bfloat16, 74, bi_device)
    graph, _ = build(adversarial(64, 4096, torch.bfloat16, 73, bi_device), w, override=declared)
    xt, wt, et, yt = (graph.find_tensor(name) for name in ("X", "scale", "epsilon", "rmsnorm::Y"))
    x = adversarial(64, width, torch.bfloat16, 75, bi_device)
    y = torch.empty(64, 4096, dtype=torch.bfloat16, device=bi_device)
    with pytest.raises(ValueError, match=match):
        graph.execute(
            {xt: x, wt: w.view(1, -1), et: torch.full((1, 1), EPS), yt: y},
            override_uids=[xt.get_uid(), yt.get_uid()],
            override_shapes=[shape, shape],
            override_strides=[x_strides, y_strides],
        )


@pytest.mark.parametrize(
    "case, match",
    [("short inverse RMS", "bytes"), ("X dtype", "declared dtype"), ("FP64 epsilon", "host FP32 scalar")],
)
def test_mismatched_bindings_raise(bi_device, case, match):
    """A bound buffer must hold the bytes this execute addresses, in the declared dtype."""
    hidden = 4096
    x = adversarial(64, hidden, torch.bfloat16, 78, bi_device)
    w = weight(hidden, torch.bfloat16, 79, bi_device)
    graph, _ = build(x, w, training=True)
    xt, wt, et, yt, it = (graph.find_tensor(name) for name in ("X", "scale", "epsilon", "rmsnorm::Y", "rmsnorm::INV_VARIANCE"))
    pack = {xt: x, wt: w.view(1, -1), et: torch.full((1, 1), EPS), yt: torch.empty_like(x), it: torch.empty(64, 1, device=bi_device)}
    if case == "short inverse RMS":
        pack[it] = torch.empty(32, 1, device=bi_device)
    elif case == "X dtype":
        pack[xt] = x.half()
    else:
        pack[et] = torch.full((1, 1), EPS, dtype=torch.float64)
    with pytest.raises(ValueError, match=match):
        graph.execute(pack)


@pytest.mark.parametrize("operand", ["X", "scale", "rmsnorm::Y"])
def test_misaligned_pointers_raise(bi_device, operand):
    """Each operand is read or written one granule at a time, so its pointer must be aligned to that access."""
    hidden = 4096
    x = adversarial(64, hidden, torch.bfloat16, 76, bi_device)
    w = weight(hidden, torch.bfloat16, 77, bi_device)
    graph, _ = build(x, w)
    xt, wt, et, yt = (graph.find_tensor(name) for name in ("X", "scale", "epsilon", "rmsnorm::Y"))
    pack = {xt: x, wt: w.view(1, -1), et: torch.full((1, 1), EPS), yt: torch.empty_like(x)}
    tensor = graph.find_tensor(operand)
    shifted = torch.empty(pack[tensor].numel() + 1, dtype=torch.bfloat16, device=bi_device)[1:].view(pack[tensor].shape)
    shifted.copy_(pack[tensor])
    pack[tensor] = shifted
    with pytest.raises(ValueError, match="aligned"):
        graph.execute(pack)


def test_rank3_input(bi_device):
    hidden = 4096
    x = adversarial(48, hidden, torch.bfloat16, 61, bi_device)
    w = weight(hidden, torch.bfloat16, 62, bi_device)
    expected = build(x, w, training=True)[1]()
    got = build(x.view(4, 12, hidden), w, training=True)[1]()
    assert_bitwise({name: value.reshape(48, -1) for name, value in got.items()}, expected, "rank-3 input")


def test_kernel_row_strides(bi_device):
    """The kernel addresses rows through independent X and Y row strides."""
    from cudnn.norm.batch_invariant.kernel import build as build_kernel

    hidden, rows = 4096, 48
    base = adversarial(2 * rows, 2 * hidden, torch.bfloat16, 63, bi_device)
    x = base[::2, :hidden]  # row stride 4 * hidden elements
    w = weight(hidden, torch.bfloat16, 64, bi_device)
    y_ref, _ = exact(x.contiguous(), w)
    y_store = torch.full((rows, 3 * hidden), float("nan"), dtype=torch.bfloat16, device=bi_device)
    schedule, compiled = build_kernel(rows, hidden, "bf16", "bf16", "bf16", False, bi_device.index)
    compiled.execute(
        x.data_ptr(), w.data_ptr(), y_store.data_ptr(), 0, rows, x.stride(0), y_store.stride(0), EPS, schedule.grid, torch.cuda.current_stream().cuda_stream
    )
    assert_bitwise({"y": y_store[:, :hidden].contiguous()}, {"y": y_ref}, "strided rows")
    assert torch.isnan(y_store[:, hidden:].float()).all(), "the kernel wrote outside its rows"


@pytest.mark.parametrize("training", [False, True], ids=["inference", "training"])
def test_known_bytes(bi_device, training):
    hidden = 4096
    x = adversarial(37, hidden, torch.bfloat16, 71, bi_device)
    w = weight(hidden, torch.bfloat16, 72, bi_device)
    out = assert_replays(build(x, w, training=training)[1])
    y, rstd = exact(x, w)
    assert_bitwise(out, {"y": y, "rstd": rstd} if training else {"y": y}, "exact model")
    name = f"batch_invariant_rmsnorm_bf16_h4096_{'training' if training else 'inference'}"
    assert_golden(name, {"x": x, "w": w}, out, baseline_dir=GOLDENS)
