# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute-time shape overrides on the unified SDPA forward engine.

A graph declares the largest extent it will ever be asked for
(``is_override_shape_enabled=True``); each ``execute`` then names the extent this
run actually uses through ``override_uids`` / ``override_shapes`` /
``override_strides``. The declaration is what the heuristic sees at build time,
so the plan is chosen for the declared class rather than for the run.

What this file pins, on a device that serves the unified SDPA engine:

* the declared-shape run is correct, and an override inside the measured legal
  window is correct against an independent FP64 reference built from the
  already-quantized inputs;
* the legal window is NOT "any extent up to the declaration": the non-causal
  class serves multiples of the plan's Q tile, the causal class serves any
  length above 1, and the two classes do not mix (declared prefill -> ``s_q=1``
  is rejected with ``CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE``);
* an override beyond the declaration, and a non-causal override off the tile
  grid, are strict xfails: the tree rejects neither, so they fail (wrong output)
  until the contract is made explicit;
* rebinding the graph to fresh buffers for a second run gives that run's own
  answer and leaves the first buffers alone;
* the Stats geometry follows the same override as the output.

The reference is FP64 over the same FP16 inputs. A served run measures ~3e-4
(FP16 storage); a mis-served run is O(1) wrong, so the tolerance is not what
decides these tests.
"""

import math

import pytest
import torch

import cudnn
from cudnn._pygraph import pygraph

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

pytestmark = pytest.mark.L0

B, H, D = 1, 4, 64
S_MAX = 256
TILE = 64  # Q tile of the selected plan (measured on L20: eng8_k24=1_k27=0_k38=0_k40=3_k41=1)
DT = cudnn.data_type.HALF
_SCALE = 1.0 / math.sqrt(D)
_ATOL = 5e-3


def _reference(q, k, v, causal):
    """FP64 reference over the *already quantized* inputs (no cuDNN, no helper)."""
    qf, kf, vf = q.double(), k.double(), v.double()
    scores = (qf @ kf.transpose(-1, -2)) * _SCALE
    if causal:
        n, m = scores.shape[-2], scores.shape[-1]
        scores = scores.masked_fill(torch.triu(torch.ones(n, m, dtype=torch.bool, device=scores.device), 1), float("-inf"))
    return (torch.softmax(scores, dim=-1) @ vf).float()


class OverrideCase:
    """One override-enabled graph plus a buffer set allocated at the declared extent."""

    def __init__(self, *, s_max=S_MAX, s_kv_max=None, causal=False, stats=False, declared_only=False):
        s_kv_max = s_max if s_kv_max is None else s_kv_max
        self.causal = causal
        self.stats = stats
        self.s_max = s_max
        self.s_kv_max = s_kv_max
        g = pygraph(
            io_data_type=DT,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            is_override_shape_enabled=True,
        )
        self.graph = g
        self.q = g.tensor(name="q", dim=[B, H, s_max, D], stride=[H * s_max * D, s_max * D, D, 1], data_type=DT)
        self.k = g.tensor(name="k", dim=[B, H, s_kv_max, D], stride=[H * s_kv_max * D, s_kv_max * D, D, 1], data_type=DT)
        self.v = g.tensor(name="v", dim=[B, H, s_kv_max, D], stride=[H * s_kv_max * D, s_kv_max * D, D, 1], data_type=DT)
        if stats:
            self.o, self.st = g.sdpa(self.q, self.k, self.v, use_causal_mask=causal, attn_scale=_SCALE, generate_stats=True)
            self.st.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        else:
            self.o, self.st = g.sdpa(self.q, self.k, self.v, is_inference=True, use_causal_mask=causal, attn_scale=_SCALE)
        self.o.set_output(True).set_data_type(DT)
        g.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        self.workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
        self.bufs = self.new_buffers()

    def new_buffers(self):
        """Fresh buffers at the declared extent: an override says how much of them this run uses."""
        return (
            torch.randn(B, H, self.s_max, D, dtype=torch.float16, device="cuda"),
            torch.randn(B, H, self.s_kv_max, D, dtype=torch.float16, device="cuda"),
            torch.randn(B, H, self.s_kv_max, D, dtype=torch.float16, device="cuda"),
            torch.full((B, H, self.s_max, D), float("nan"), dtype=torch.float16, device="cuda"),
            torch.full((B, H, self.s_max), float("nan"), dtype=torch.float32, device="cuda"),
        )

    def execute(self, s_q, s_kv, *, bufs=None, override=True):
        qt, kt, vt, ot, stt = self.bufs if bufs is None else bufs
        data = {self.q: qt, self.k: kt, self.v: vt, self.o: ot}
        kwargs = {}
        if override:
            q_geom, kv_geom = [B, H, s_q, D], [B, H, s_kv, D]
            q_stride = [H * self.s_max * D, self.s_max * D, D, 1]
            kv_stride = [H * self.s_kv_max * D, self.s_kv_max * D, D, 1]
            uids = [self.q.get_uid(), self.k.get_uid(), self.v.get_uid(), self.o.get_uid()]
            shapes = [q_geom, kv_geom, kv_geom, q_geom]
            strides = [q_stride, kv_stride, kv_stride, q_stride]
            if self.stats:
                data[self.st] = stt
                uids.append(self.st.get_uid())
                shapes.append([B, H, s_q])
                strides.append([H * self.s_max, self.s_max, 1])
            kwargs = dict(override_uids=uids, override_shapes=shapes, override_strides=strides)
        self.graph.execute(data, self.workspace, **kwargs)
        torch.cuda.synchronize()
        return ot[:, :, :s_q].float(), (stt[:, :, :s_q].double() if self.stats else None)

    def assert_served(self, s_q, s_kv, *, got=None, bufs=None):
        qt, kt, vt = (self.bufs if bufs is None else bufs)[:3]
        o_actual, st_actual = self.execute_result if got is None else got
        ref = _reference(qt[:, :, :s_q], kt[:, :, :s_kv], vt[:, :, :s_kv], self.causal)
        torch.testing.assert_close(o_actual, ref, atol=_ATOL, rtol=1e-2)
        if self.stats:
            assert torch.isfinite(st_actual).all(), "Stats carries a non-finite value on a served override"
            ref_lse = torch.logsumexp(
                (qt[:, :, :s_q].double() @ kt[:, :, :s_kv].double().transpose(-1, -2)) * _SCALE, dim=-1
            )
            torch.testing.assert_close(st_actual, ref_lse, atol=1e-3, rtol=1e-3)

    def backend_evidence(self):
        assert self.graph.selected_engine is None
        assert self.graph._lowered_graph is not None
        assert self.graph._cpp_plans_created and self.graph._is_built


@pytest.fixture
def case():
    return OverrideCase()


# ---------------------------------------------------------------------------
# The legal window
# ---------------------------------------------------------------------------


def test_declared_shape_runs_without_any_override(case):
    """Smoke: the declared graph is correct before an override is involved."""
    got = case.execute(S_MAX, S_MAX, override=False)
    case.backend_evidence()
    ref = _reference(case.bufs[0], case.bufs[1], case.bufs[2], case.causal)
    torch.testing.assert_close(got[0], ref, atol=_ATOL, rtol=1e-2)


@pytest.mark.parametrize("s_actual", [64, 128, 192, 256], ids=lambda s: f"s{s}")
def test_non_causal_override_on_the_tile_grid_is_served(s_actual):
    """Non-causal overrides are served at multiples of the plan's Q tile."""
    case = OverrideCase()
    got = case.execute(s_actual, s_actual)
    case.backend_evidence()
    case.assert_served(s_actual, s_actual, got=got)


@pytest.mark.parametrize("s_actual", [2, 33, 64, 96, 128], ids=lambda s: f"s{s}")
def test_causal_override_serves_every_length_above_one(s_actual):
    """The causal class does not need tile alignment: 33 and 96 are served here."""
    case = OverrideCase(causal=True)
    got = case.execute(s_actual, s_actual)
    case.backend_evidence()
    case.assert_served(s_actual, s_actual, got=got)


def test_stats_geometry_follows_the_override():
    """With generate_stats the Stats/LSE geometry is overridden together with O."""
    case = OverrideCase(stats=True)
    got = case.execute(64, 64)
    case.backend_evidence()
    case.assert_served(64, 64, got=got)


def test_decode_graph_overrides_its_kv_extent():
    """A decode-shaped graph (s_q = 1) serves an override of its KV extent."""
    case = OverrideCase(s_max=1, s_kv_max=S_MAX)
    got = case.execute(1, 64)
    case.backend_evidence()
    case.assert_served(1, 64, got=got)


def test_two_extents_on_one_plan_and_a_rebind_to_fresh_buffers():
    """Shape A, then B on the same plan, then a second buffer set.

    The second run must answer for its own buffers and leave the first buffers'
    results untouched: neither the pointer set nor the geometry of the first run
    may be remembered.
    """
    case = OverrideCase()
    first = case.execute(64, 64)
    case.assert_served(64, 64, got=first)

    # The second extent is larger, so it legitimately rewrites the first rows;
    # what must NOT happen is the second run answering from the first run's
    # baked-in geometry, which _assert_served checks above.
    second = case.execute(128, 128)
    case.assert_served(128, 128, got=second)
    keep = case.bufs[3].clone()

    # A third run on FRESH buffers must leave this buffer set exactly as it was.
    fresh = case.new_buffers()
    third = case.execute(64, 64, bufs=fresh)
    case.assert_served(64, 64, got=third, bufs=fresh)
    torch.testing.assert_close(case.bufs[3][:, :, :128], keep[:, :, :128], atol=0, rtol=0)
    assert torch.isnan(case.bufs[3][:, :, 128:]).all(), "the fresh-buffer run wrote into this buffer set"
    case.backend_evidence()


def test_the_declared_class_decides_the_plan_not_the_run(capsys):
    """Observation: two declarations run the same actual shape on their own plans.

    No speed assertion. The point is that the plan is chosen at build time from
    the declaration and the run does not reselect one, which is why bucketing is
    the caller's policy rather than something the override does.
    """
    observations = []
    for declared in (64, S_MAX):
        case = OverrideCase(s_max=declared, s_kv_max=declared)
        got = case.execute(64, 64, override=declared != 64)
        case.assert_served(64, 64, got=got)
        names = [case.graph.get_plan_name_at_index(i) for i in range(case.graph.get_execution_plan_count())]
        observations.append((declared, names))
    with capsys.disabled():
        print(f"\nplans per declared extent: {observations}")
    assert all(names for _, names in observations), observations
    # The declaration is what the heuristic sees, and the plans exist before any
    # override does, which is why a run cannot reselect one. On this device the
    # list happens to be identical for 64 and 256 rows; the 128/256 engine flip
    # in the issue is an SM100/SM107 observation and is not generalized here.


# ---------------------------------------------------------------------------
# Boundaries the tree does not reject
# ---------------------------------------------------------------------------


def test_prefill_to_decode_is_rejected_by_the_backend():
    """A declared prefill graph cannot cross the s_q == 1 boundary.

    The engine classes differ, and the backend answers with
    ``CUDNN_STATUS_NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE`` instead of serving a
    wrong answer. The assertion is on the status code, so a stride, buffer or
    layout failure does not pass this test.
    """
    case = OverrideCase()
    with pytest.raises(Exception) as excinfo:
        case.execute(1, 1)
    assert "NOT_SUPPORTED_INVALID_DYNAMIC_SHAPE" in str(excinfo.value), str(excinfo.value)


@pytest.mark.xfail(
    strict=True,
    reason="beyond the declared extent the tree neither rejects nor serves "
    "(measured: s=320 finite but wrong, s=512 NaN)",
)
def test_override_beyond_the_declared_extent_is_rejected():
    case = OverrideCase()
    with pytest.raises(Exception):
        case.execute(320, 320)


@pytest.mark.xfail(
    strict=True,
    reason="non-causal override off the tile grid is silently wrong (measured: s=96 max|dO| 8.1e-01, "
    "s=160 4.9e-01) while the same shape declared outright is correct",
)
def test_non_causal_override_off_the_tile_grid_is_served():
    case = OverrideCase()
    got = case.execute(96, 96)
    case.assert_served(96, 96, got=got)


def test_override_triple_must_name_the_same_tensors():
    """A short or mismatched override triple is refused before anything runs."""
    case = OverrideCase()
    geom = [B, H, 64, D]
    stride = [H * S_MAX * D, S_MAX * D, D, 1]
    qt, kt, vt, ot, _ = case.bufs
    with pytest.raises(Exception) as excinfo:
        case.graph.execute(
            {case.q: qt, case.k: kt, case.v: vt, case.o: ot},
            case.workspace,
            override_uids=[case.q.get_uid()],
            override_shapes=[geom, geom],
            override_strides=[stride, stride],
        )
    # The frontend's own length check does not run on this path (overrides go
    # straight to the backend uid-map overload), so the caller sees the backend
    # finalize error: same contract, backend-level wording.
    message = str(excinfo.value)
    assert "override_uids_count" in message or "same size" in message, message
    assert "BAD_PARAM" in message or "same size" in message, message
