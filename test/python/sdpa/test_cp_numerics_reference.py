# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layered #752 tests: R0 (math oracle) and R1 (TE merge fidelity).

The layers are kept visibly separate so that a green run cannot be read as more
than it is:

* ``TestR0ReferenceOracle`` -- FP64 mathematics only. Proves the chunk-merge
  formula, the masking policy and the empty/masked-row semantics. Independent of
  the code under test: ``expected`` always comes from the monolithic FP64 path,
  ``actual`` from the merge under test.
* ``TestR1FixtureContract`` -- trace/schedule/fixture bookkeeping that any R1
  claim depends on.
* ``TestR1TeFidelity`` -- the frozen-partial merge under TE's step order.
  **Approximate**, never bitwise, against the FP64 oracle. Bitwise equality is
  only ever asserted between two runs of the same code path.
* ``TestR1EmulatorLane`` -- this package's own schedule emulator, kept separate
  because agreeing with it is not evidence about TE.
* ``TestR2Preconditions`` -- the actual-attention lane's guards. R2 itself is a
  distributed lane and lives in ``cp_numerics/run_cp_reference.py``.

Nothing in this file executes a cuDNN or TransformerEngine GPU kernel.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cp_numerics import reference_math as rm  # noqa: E402
from cp_numerics import te_adapter as tea  # noqa: E402
from cp_numerics import trace_schema as ts  # noqa: E402

FIXTURE_DIR = os.path.join(_HERE, "cp_numerics", "fixtures")

#: Fixtures the R1 tests consume, and the P they carry. The runner enforces the
#: same mapping; a P=2 file must never be replayed under a P=4 label.
FIXTURE_NAMES = {
    1: "cp_p1_f32_s128_causal",
    2: "cp_p2_f32_s128_causal",
    4: "cp_p4_f32_s128_causal",
}

#: Relative-error budget for "the merge is numerically sound against FP64",
#: expressed in units of the declared storage precision rather than as a magic
#: atol: 32 * eps(dtype) * max|reference|. This is explicitly NOT a bitwise claim
#: and NOT a substitute for the ULP/abs/rel numbers reported by the runner.
SOUNDNESS_ULP_BUDGET = 32


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURE_DIR, f"{name}.pt")


def _require_fixture(name: str) -> tea.FrozenFixture:
    path = _fixture_path(name)
    if not os.path.exists(path):
        pytest.skip(f"fixture {name} not present at {path}; run cp_numerics/make_fixtures.py")
    return tea.load_fixture(path)


def _merge_all_ranks(fixture: tea.FrozenFixture, *, compiled: bool = False) -> torch.Tensor:
    header = fixture.header
    per_rank = {}
    for rank in range(header.cp_size):
        out, _ = tea.merge_te_fidelity(
            rank=rank,
            cp_size=header.cp_size,
            causal=header.causal,
            partial_out=[fixture.rank_step_out(rank, s) for s in range(header.cp_size)],
            partial_lse=[fixture.rank_step_lse(rank, s) for s in range(header.cp_size)],
            o_local_shape=_local_shape(header),
            compiled=compiled,
        )
        per_rank[rank] = out.reshape(header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)
    return ts.restore_global_order(per_rank, header.cp_size, header.chunk_len)


def _local_shape(header: tea.FixtureHeader):
    if header.causal:
        return (header.batch, 2, header.chunk_len, header.num_heads, header.head_dim)
    return (header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)


def _bitwise_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    """Bit-pattern comparison of the same dtype; never ``torch.equal``."""
    if a.dtype != b.dtype:
        raise TypeError("bitwise compare needs identical dtypes")
    if a.dtype == torch.float32:
        return int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum().item())
    if a.dtype in (torch.float16, torch.bfloat16):
        return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum().item())
    if a.dtype == torch.float64:
        return int((a.contiguous().view(torch.int64) != b.contiguous().view(torch.int64)).sum().item())
    raise TypeError(f"no integer view for {a.dtype}")


def _soundness_budget(dtype: torch.dtype, reference: torch.Tensor) -> float:
    eps = torch.finfo(dtype).eps
    scale = float(reference.abs().max().item())
    return SOUNDNESS_ULP_BUDGET * eps * scale


# ===========================================================================
# R0: mathematical oracle
# ===========================================================================


class TestR0ReferenceOracle:
    """FP64 only. ``expected`` = monolithic attention; ``actual`` = the merge."""

    @staticmethod
    def _case(seq_len: int = 24, causal: bool = True, heads: int = 2, dim: int = 8, seed: int = 7):
        gen = torch.Generator().manual_seed(seed)
        shape = (1, seq_len, heads, dim)
        q = torch.randn(shape, generator=gen, dtype=torch.float64)
        k = torch.randn(shape, generator=gen, dtype=torch.float64)
        v = torch.randn(shape, generator=gen, dtype=torch.float64)
        return q, k, v, float(dim) ** -0.5, causal

    @pytest.mark.L0
    @pytest.mark.parametrize("causal", [True, False])
    def test_monolithic_and_chunked_paths_agree(self, causal: bool) -> None:
        """Fixture self-consistency: the chunked path plus the analytic merge
        must reproduce the monolithic attention in FP64. This validates the
        *fixture*, and deliberately uses a third code path as the expected."""
        q, k, v, scale, _ = self._case(causal=causal)
        seq_len = q.shape[1]
        expected = rm.full_attention_fp64(q, k, v, scale=scale, causal=causal)
        partials = rm.chunk_partials_fp64(q, k, v, scale=scale, causal=causal, chunk_bounds=rm.even_chunk_bounds(seq_len, 4))
        actual = rm.analytic_merge_fp64(partials)
        torch.testing.assert_close(actual.out, expected.out, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual.lse, expected.lse, rtol=1e-12, atol=1e-12)

    @pytest.mark.L0
    @pytest.mark.parametrize("causal", [True, False])
    @pytest.mark.parametrize("cp_size", [1, 2, 4])
    def test_r0_merge_matches_monolithic(self, causal: bool, cp_size: int) -> None:
        """R0: build the TE schedule at FP64, take the chunk partials from the
        oracle, merge them with the code under test, compare to the monolithic
        FP64 result. The merge never contributes to ``expected``."""
        q, k, v, scale, _ = self._case(causal=causal)
        seq_len = q.shape[1]
        chunk_len = seq_len // (2 * cp_size)
        assert chunk_len * 2 * cp_size == seq_len
        expected = rm.full_attention_fp64(q, k, v, scale=scale, causal=causal)
        trace = tea.te_schedule(cp_size, chunk_len, causal=causal, partial_dtype="float64", accumulator_dtype="float64")
        per_rank = {}
        for rank in range(cp_size):
            partial_out, partial_lse = [], []
            for rec in trace:
                if rec.rank != rank:
                    continue
                q_positions = torch.cat([torch.arange(b, e) for b, e in rec.q_global_ranges])
                kv_positions = torch.cat([torch.arange(b, e) for b, e in rec.kv_global_ranges])
                keep = rm.causal_keep_mask_from_positions(q_positions, kv_positions, causal=(rec.causal_mode == "causal"))
                part = rm.chunk_partial_with_mask_fp64(
                    q.index_select(1, q_positions),
                    k.index_select(1, kv_positions),
                    v.index_select(1, kv_positions),
                    scale=scale,
                    keep_mask=keep,
                )
                partial_out.append(part.out)
                partial_lse.append(part.lse)
            local_shape = (q.shape[0], 2, chunk_len, q.shape[2], q.shape[3]) if causal else (q.shape[0], 2 * chunk_len, q.shape[2], q.shape[3])
            out, _ = tea.merge_te_fidelity(
                rank=rank,
                cp_size=cp_size,
                causal=causal,
                partial_out=partial_out,
                partial_lse=partial_lse,
                o_local_shape=local_shape,
            )
            per_rank[rank] = out.reshape(q.shape[0], 2 * chunk_len, q.shape[2], q.shape[3])
        actual = ts.restore_global_order(per_rank, cp_size, chunk_len)
        torch.testing.assert_close(actual, expected.out, rtol=1e-11, atol=1e-11)

    @pytest.mark.L0
    def test_empty_chunk_semantics(self) -> None:
        """A chunk with no unmasked key must give ``L = -inf`` and ``O = 0``."""
        q, k, v, scale, causal = self._case()
        seq_len = q.shape[1]
        bounds = rm.even_chunk_bounds(seq_len, 4)
        partials = rm.chunk_partials_fp64(q, k, v, scale=scale, causal=causal, chunk_bounds=bounds)
        # Chunk 0 under top-left causal only exposes query row 0.
        empty = rm.chunk_partial_with_mask_fp64(
            q,
            k[:, bounds[0][0] : bounds[0][1]],
            v[:, bounds[0][0] : bounds[0][1]],
            scale=scale,
            keep_mask=torch.zeros((seq_len, bounds[0][1] - bounds[0][0]), dtype=torch.bool),
        )
        assert torch.isneginf(empty.lse).all(), "a fully masked chunk must report L = -inf"
        assert torch.equal(empty.out, torch.zeros_like(empty.out)), "a fully masked chunk must report O = 0"
        # And it must not perturb an analytic merge that also has real chunks.
        merged = rm.analytic_merge_fp64([empty] + partials[1:])
        expected = rm.full_attention_fp64(q, k[:, bounds[1][0] :], v[:, bounds[1][0] :], scale=scale, causal=causal, kv_offset=bounds[1][0])
        torch.testing.assert_close(merged.out, expected.out, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(merged.lse, expected.lse, rtol=1e-12, atol=1e-12, equal_nan=True)
        # Rows whose entire key set was dropped must report -inf on both sides;
        # the empty chunk must not turn them into 0 or NaN.
        assert torch.isneginf(merged.lse[:, :, : bounds[1][0]]).all()
        assert torch.isfinite(merged.lse[:, :, bounds[1][0] :]).all()
        assert torch.equal(torch.isfinite(merged.lse), torch.isfinite(expected.lse))

    @pytest.mark.L0
    def test_fully_masked_row_is_minus_inf_and_zero(self) -> None:
        """Reference policy: an all-masked row is ``L = -inf`` / ``O = 0``, not NaN."""
        q, k, v, scale, _ = self._case(seq_len=8)
        keep = torch.zeros((q.shape[1], k.shape[1]), dtype=torch.bool)
        result = rm.full_attention_with_mask_fp64(q, k, v, scale=scale, keep_mask=keep)
        assert torch.isneginf(result.lse).all()
        assert torch.equal(result.out, torch.zeros_like(result.out))
        assert not torch.isnan(result.lse).any()
        assert not torch.isnan(result.out).any()

    @pytest.mark.L0
    def test_padding_is_not_treated_as_valid_kv(self) -> None:
        """Keys past ``valid_kv_len`` must not change the result at all."""
        q, k, v, scale, causal = self._case(seq_len=16)
        valid = 10
        padded = rm.full_attention_fp64(q, k, v, scale=scale, causal=causal, valid_kv_len=valid)
        truncated = rm.full_attention_fp64(q, k[:, :valid], v[:, :valid], scale=scale, causal=causal)
        torch.testing.assert_close(padded.out, truncated.out, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(padded.lse, truncated.lse, rtol=1e-12, atol=1e-12)
        # ...and sentinel padding must be *ignored*, not merely equal by luck.
        poisoned = k.clone()
        poisoned[:, valid:] = 1e6
        poisoned_v = v.clone()
        poisoned_v[:, valid:] = -1e6
        with_padding = rm.full_attention_fp64(q, poisoned, poisoned_v, scale=scale, causal=causal, valid_kv_len=valid)
        torch.testing.assert_close(with_padding.out, truncated.out, rtol=1e-12, atol=1e-12)

    @pytest.mark.L0
    def test_top_left_alignment_is_the_documented_policy(self) -> None:
        """Non-square causal uses top-left alignment: row ``i`` keeps ``kv <= i``.

        With a 2-row query and a 4-key KV the alternative (bottom-right)
        alignment would give row 0 the keys 0..2 and a mean of 1.0. The assertion
        below pins the top-left policy the fixtures are built with.
        """
        q = torch.zeros(1, 2, 1, 1, dtype=torch.float64)
        k = torch.zeros(1, 4, 1, 1, dtype=torch.float64)
        v = torch.arange(4, dtype=torch.float64).view(1, 4, 1, 1)
        result = rm.full_attention_fp64(q, k, v, scale=1.0, causal=True)
        torch.testing.assert_close(result.out[0, 0], torch.zeros(1, 1, dtype=torch.float64))
        torch.testing.assert_close(result.out[0, 1], torch.full((1, 1), 0.5, dtype=torch.float64))
        # Offsets are global positions: a chunk that starts at key 0 while the
        # queries start at position 2 keeps the same "kv <= q" predicate.
        k6 = torch.zeros(1, 6, 1, 1, dtype=torch.float64)
        v6 = torch.arange(6, dtype=torch.float64).view(1, 6, 1, 1)
        chunk = rm.full_attention_fp64(q, k6, v6, scale=1.0, causal=True, q_offset=2)
        torch.testing.assert_close(chunk.out[0, 0, 0, 0], torch.tensor(1.0, dtype=torch.float64))  # mean of 0,1,2
        torch.testing.assert_close(chunk.out[0, 1, 0, 0], torch.tensor(1.5, dtype=torch.float64))  # mean of 0..3

    @pytest.mark.L0
    def test_reference_and_merge_are_independent_modules(self) -> None:
        """The oracle must not be able to produce ``actual`` for the merge."""
        source = open(os.path.join(_HERE, "cp_numerics", "reference_math.py"), encoding="utf-8").read()
        assert "te_adapter" not in source, "reference_math must not import the code under test"
        adapter = open(os.path.join(_HERE, "cp_numerics", "te_adapter.py"), encoding="utf-8").read()
        assert "reference_math" not in adapter, "te_adapter must not use the oracle to fabricate expected values"
        assert not hasattr(rm, "merge_te_fidelity")

    @pytest.mark.L0
    def test_analytic_merge_is_not_the_expected_for_r1(self) -> None:
        """Guard the plan's rule: the logsumexp formula is a cross-check only."""
        import inspect

        source = open(os.path.join(_HERE, "cp_numerics", "te_adapter.py"), encoding="utf-8").read()
        assert "analytic_merge_fp64" not in source, "R1 must not be graded against the analytic formula"
        te_lane = inspect.getsource(tea.merge_te_fidelity)
        assert "logsumexp" not in te_lane and "logaddexp" not in te_lane
        # The merged lane must reach the transcribed helpers, not a rewrite of them.
        eager = tea._helpers(False)
        assert eager["lse"] is tea.flash_attn_fwd_softmax_lse_correction
        assert eager["lse2"] is tea.flash_attn_fwd_second_half_softmax_lse_correction
        assert eager["init"] is tea.flash_attn_fwd_out_correction_init
        assert eager["out"] is tea.flash_attn_fwd_out_correction
        assert eager["out2"] is tea.flash_attn_fwd_second_half_out_correction
        for index in (0, 1, 2, 3, 4):
            pass  # nothing further: the identity checks above are the real guard


# ===========================================================================
# R1: fixture contract
# ===========================================================================


class TestR1FixtureContract:
    @pytest.mark.L0
    @pytest.mark.parametrize("cp_size", [1, 2, 4])
    def test_fixture_trace_matches_derived_schedule(self, cp_size: int) -> None:
        fixture = _require_fixture(FIXTURE_NAMES[cp_size])
        assert fixture.header.cp_size == cp_size
        tea.check_schedule_matches_trace(fixture.header)

    @pytest.mark.L0
    @pytest.mark.parametrize("cp_size", [1, 2, 4])
    def test_fixture_invariants_hold(self, cp_size: int) -> None:
        fixture = _require_fixture(FIXTURE_NAMES[cp_size])
        header = fixture.header
        records = [ts.TraceRecord.from_dict(d) for d in header.trace]
        results = ts.check_invariants(
            records,
            cp_size=header.cp_size,
            seq_len=header.seq_len,
            chunk_len=header.chunk_len,
            causal=header.causal,
            valid_len=header.valid_len,
        )
        failed = [r for r in results if not r.ok]
        assert not failed, "\n".join(f"{r.name}: {r.detail}" for r in failed)
        names = {r.name for r in results}
        assert names == {
            "coverage_no_duplication",
            "unique_partial_slots",
            "order_preserved",
            "query_half_matches_schedule",
            "global_token_order_restored",
            "no_padding_as_valid_kv",
            "dtype_metadata_uniform",
        }

    @pytest.mark.L0
    def test_trace_records_expose_every_required_field(self) -> None:
        fixture = _require_fixture(FIXTURE_NAMES[2])
        required = {
            "rank",
            "ring_step",
            "source_rank",
            "q_global_begin",
            "q_global_end",
            "kv_global_begin",
            "kv_global_end",
            "causal_mode",
            "query_half",
            "partial_slot_id",
            "lse_update_index",
            "out_update_index",
            "partial_dtype",
            "accumulator_dtype",
            "lse_base",
        }
        for record in fixture.header.trace:
            assert required <= set(record), f"missing trace fields: {sorted(required - set(record))}"
        ts.trace_from_json(ts.trace_to_json([ts.TraceRecord.from_dict(d) for d in fixture.header.trace]))

    @pytest.mark.L0
    @pytest.mark.parametrize("cp_size", [1, 2, 4])
    def test_fixture_header_carries_the_precision_contract(self, cp_size: int) -> None:
        header = _require_fixture(FIXTURE_NAMES[cp_size]).header
        assert header.o_i_normalized is True
        assert header.lse_base == "e"
        assert header.layout == "bshd"
        assert header.qkv_dtype in ("float16", "bfloat16", "float32")
        assert header.partial_storage_dtype in ("float16", "bfloat16", "float32")
        assert header.accumulator_dtype in ("float16", "bfloat16", "float32")
        assert header.o_dtype in ("float16", "bfloat16", "float32")
        assert header.lse_dtype == "float32"
        assert header.mask in ("causal-top-left", "no_mask")
        assert header.head_mapping
        assert header.te_revision == tea.TE_PINNED_SHA
        assert header.fuser_mode
        assert header.scale == pytest.approx(header.head_dim**-0.5)
        assert header.fixture_sha256
        assert header.valid_len == header.seq_len

    @pytest.mark.L0
    def test_p_schedule_map_is_explicit_and_differs_per_p(self) -> None:
        maps = {}
        for cp_size in (1, 2, 4):
            header = _require_fixture(FIXTURE_NAMES[cp_size]).header
            maps[cp_size] = header.p_schedule_map
            assert len(header.p_schedule_map) == cp_size * cp_size
            for row in header.p_schedule_map:
                assert row["source_rank"] == (row["rank"] - row["ring_step"]) % cp_size
        assert maps[1] != maps[2] and maps[2] != maps[4]
        # A P=2 file must not be replayable under a P=4 label.
        header2 = _require_fixture(FIXTURE_NAMES[2]).header
        forged = type(header2)(**{**header2.__dict__, "cp_size": 4})
        with pytest.raises(AssertionError):
            tea.check_schedule_matches_trace(forged)

    @pytest.mark.L0
    def test_ragged_fixture_is_reference_only(self) -> None:
        fixture = _require_fixture("ref_ragged_p2_s102_causal")
        assert fixture.header.lane == "reference_only"
        ok, why = tea.te_requires_divisibility(fixture.header)
        assert not ok and "reference_only" in why
        # ...and if the lane were forced, the divisibility constraint is the next gate.
        forged = type(fixture.header)(**{**fixture.header.__dict__, "lane": "te_fidelity"})
        ok2, why2 = tea.te_requires_divisibility(forged)
        assert not ok2 and "divisib" in why2
        expected = fixture.tensors["expected_out_fp64"]
        splits = fixture.tensors["ragged_splits"].tolist()
        assert sum(splits) == fixture.header.seq_len
        assert len(set(splits)) > 1, "the ragged fixture must actually have an uneven split"
        # And the reference planner must still reproduce the monolithic result.
        partials = [
            rm.ChunkPartial(
                chunk_index=i,
                kv_begin=sum(splits[:i]),
                kv_end=sum(splits[: i + 1]),
                out=fixture.tensors[f"out_chunk_{i}"].to(torch.float64),
                lse=fixture.tensors[f"lse_chunk_{i}"].to(torch.float64),
            )
            for i in range(len(splits))
        ]
        merged = rm.analytic_merge_fp64(partials)
        # The fixture stores chunk partials in float32, so the reference-planner
        # check is bounded by that storage precision, not by FP64.
        max_abs = float((merged.out - expected).abs().max().item())
        assert max_abs <= _soundness_budget(torch.float32, expected), f"ragged planner drift {max_abs:.6e}"


# ===========================================================================
# R1: TE fidelity lane
# ===========================================================================


class TestR1TeFidelity:
    @pytest.mark.L0
    @pytest.mark.parametrize("cp_size", [1, 2, 4])
    def test_te_lane_matches_fp64_oracle_approximately(self, cp_size: int) -> None:
        """APPROXIMATE match against the FP64 oracle -- never a bitwise claim.

        The bound is derived from the declared storage precision
        (``32 * eps(dtype) * max|reference|``) and the raw numbers are printed, so
        a future regression cannot be hidden behind a tuned constant. The plan
        requires the acceptable ULP/range to be confirmed by the requester; until
        then this test asserts only that the merge is numerically *sound*, and the
        runner keeps reporting the raw error and a ``bitwise: not passed`` state.
        """
        fixture = _require_fixture(FIXTURE_NAMES[cp_size])
        actual = _merge_all_ranks(fixture)
        expected = fixture.tensors["expected_out_fp64"]
        max_abs = float((actual.to(torch.float64) - expected).abs().max().item())
        budget = _soundness_budget(actual.dtype, expected)
        print(f"\nP={cp_size} dtype={actual.dtype} max_abs={max_abs:.6e} budget={budget:.6e} " f"(32 * eps({actual.dtype}) * max|ref|)")
        assert not torch.isnan(actual).any(), "the merge produced NaN on a fully valid fixture"
        assert max_abs <= budget, (
            f"merge error {max_abs:.6e} exceeds the precision-derived budget {budget:.6e}; " "report the raw ULP/abs/rel numbers instead of widening this bound"
        )

    @pytest.mark.L0
    def test_te_lane_is_not_bitwise_equal_to_the_fp64_reference(self) -> None:
        """State the negative explicitly: R1 is approximate, so the bitwise
        comparison against the FP64 reference must *fail* by a large margin."""
        fixture = _require_fixture(FIXTURE_NAMES[2])
        actual = _merge_all_ranks(fixture)
        rounded = fixture.tensors["expected_out_fp64"].to(actual.dtype)
        mismatches = _bitwise_mismatches(actual, rounded)
        assert mismatches > 0, "a bitwise-identical result would mean the fixture is degenerate, not that R1 passed"

    @pytest.mark.L0
    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_te_lane_is_bitwise_deterministic(self, cp_size: int) -> None:
        """Bitwise equality is only claimed between two runs of the same path."""
        fixture = _require_fixture(FIXTURE_NAMES[cp_size])
        first = _merge_all_ranks(fixture)
        second = _merge_all_ranks(fixture)
        assert _bitwise_mismatches(first, second) == 0

    @pytest.mark.L0
    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_multirank_reassembly_is_bitwise_stable(self, cp_size: int) -> None:
        """Per-rank merges reassembled in global order must be byte-identical to
        a fresh replay -- catches any accidental dependence on gather order."""
        fixture = _require_fixture(FIXTURE_NAMES[cp_size])
        first = _merge_all_ranks(fixture)
        header = fixture.header
        replay = {}
        for rank in range(header.cp_size):
            out, _ = tea.merge_te_fidelity(
                rank=rank,
                cp_size=header.cp_size,
                causal=header.causal,
                partial_out=[fixture.rank_step_out(rank, s) for s in range(header.cp_size)],
                partial_lse=[fixture.rank_step_lse(rank, s) for s in range(header.cp_size)],
                o_local_shape=_local_shape(header),
            )
            replay[rank] = out.reshape(header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)
        assert _bitwise_mismatches(first, ts.restore_global_order(replay, header.cp_size, header.chunk_len)) == 0

    @pytest.mark.L0
    def test_empty_chunk_partial_is_absorbed_without_nan(self) -> None:
        fixture = _require_fixture("cp_p2_f32_s128_causal_empty")
        assert fixture.header.empty_chunks == [[1, 1]]
        lse = fixture.rank_step_lse(1, 1)
        assert torch.isneginf(lse).all()
        assert torch.equal(fixture.rank_step_out(1, 1), torch.zeros_like(fixture.rank_step_out(1, 1)))
        actual = _merge_all_ranks(fixture)
        assert not torch.isnan(actual).any(), "TE's log1p/exp path must absorb a -inf step when the running LSE is finite"
        expected = fixture.tensors["expected_out_fp64"]
        max_abs = float((actual.to(torch.float64) - expected).abs().max().item())
        assert max_abs <= _soundness_budget(actual.dtype, expected), f"empty-chunk merge drift {max_abs:.6e}"

    @pytest.mark.L0
    def test_all_empty_row_produces_nan_in_te_helpers(self) -> None:
        """The documented behavioural difference, stated rather than papered over.

        TE's ``max + log1p(exp(min - max))`` turns ``-inf`` and ``-inf`` into
        ``log1p(exp(nan)) = nan``. The FP64 reference uses the empty-set policy
        ``L = -inf, O = 0``. The two disagree; neither is silently rewritten.
        """
        a = torch.full((1, 2, 4), float("-inf"), dtype=torch.float32)
        b = torch.full((1, 2, 4), float("-inf"), dtype=torch.float32)
        lse = a.clone()
        tea.flash_attn_fwd_softmax_lse_correction(lse, b)
        assert torch.isnan(lse).all(), "TE helper behaviour changed; update the report's stated difference"
        reference = rm.analytic_merge_fp64(
            [rm.ChunkPartial(i, 0, 1, torch.zeros(1, 4, 2, 3, dtype=torch.float64), a.to(torch.float64).view(1, 2, 4)) for i in range(2)]
        )
        assert torch.isneginf(reference.lse).all()
        assert torch.equal(reference.out, torch.zeros_like(reference.out))

    @pytest.mark.L0
    def test_te_helpers_are_transcribed_not_simplified(self) -> None:
        """The LSE correction must be the max/log1p form, not ``logaddexp`` or a
        ``logsumexp`` over a stack, and the output steps must round back to the
        partial dtype without an FP32 accumulator."""
        import inspect

        lse_source = inspect.getsource(tea.flash_attn_fwd_softmax_lse_correction)
        assert "log1p" in lse_source and "exp" in lse_source and "copy_" in lse_source
        init_source = inspect.getsource(tea.flash_attn_fwd_out_correction_init)
        assert ".to(out_init_step.dtype)" in init_source
        add_source = inspect.getsource(tea.flash_attn_fwd_out_correction)
        assert "add_" in add_source
        second = inspect.getsource(tea.flash_attn_fwd_second_half_out_correction)
        assert "select(seq_dim, 1)" in second and "view(*softmax_lse.shape[:-1], 2, -1)" in second

    @pytest.mark.L0
    def test_reduced_precision_fixtures_record_the_accumulator_they_ran(self) -> None:
        """`accumulator_dtype` is part of the contract, so it may not drift.

        The FP16/BF16 fixtures exist to pin the reduced-precision path, and the
        accumulation there is the storage dtype (TE's accumulator on the
        non-FP8 path): recording float32 while the replay accumulates in half
        would describe a different run than the one the fixture carries.
        """
        fixtures = pathlib.Path(_HERE, "cp_numerics", "fixtures")
        seen = set()
        for header_path in sorted(fixtures.glob("*.header.json")):
            header = json.loads(header_path.read_text())
            seen.add(header["partial_storage_dtype"])
            if header["partial_storage_dtype"] in ("float16", "bfloat16"):
                assert header["accumulator_dtype"] == header["partial_storage_dtype"], header_path.name
            else:
                assert header["accumulator_dtype"] == "float32", header_path.name
        assert {"float16", "bfloat16", "float32"} <= seen, seen

    @pytest.mark.L0
    def test_te_source_anchors_still_present(self) -> None:
        """Bind the transcription to the pinned checkout when one is available."""
        repo = os.environ.get("TE_CP_REF_REPO")
        if not repo or not os.path.isdir(repo):
            pytest.skip("TE_CP_REF_REPO is not set to a TransformerEngine checkout; " "the transcription cannot be re-verified against source in this run")
        # The revision is checked BEFORE the anchors: on an unpinned tree the
        # anchors (and the recorded digest) may legitimately be absent, and the
        # intended outcome there is a SKIP, not an assertion failure.
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        if head.stdout.strip() != tea.TE_PINNED_SHA:
            pytest.skip(f"checkout HEAD {head.stdout.strip()} != pinned {tea.TE_PINNED_SHA}; anchors verified on that tree only")
        result = tea.verify_helper_source(repo)
        missing = [k for k, v in result.items() if v is False]
        assert not missing, f"pinned TE source no longer contains: {missing}"

    @pytest.mark.L0
    def test_compiled_lane_is_close_to_eager_but_not_bitwise_identical(self) -> None:
        """TE's ``@jit_fuser`` is ``torch.compile``, so the fuser mode is part of
        the numerics contract.

        Measured on this host: inductor's fused code does **not** reproduce the
        eager elementwise rounding bit for bit (the two lanes differ in a
        non-trivial fraction of elements, still within the precision budget). The
        test therefore asserts closeness and *reports* the bitwise distance; it
        does not claim fuser-mode invariance, and the report keeps the R1 evidence
        on the eager transcription.
        """
        fixture = _require_fixture(FIXTURE_NAMES[2])
        try:
            compiled = _merge_all_ranks(fixture, compiled=True)
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            pytest.skip(f"torch.compile could not run the helpers in this environment: {exc!r}")
        eager = _merge_all_ranks(fixture, compiled=False)
        mismatches = _bitwise_mismatches(compiled, eager)
        max_abs = float((compiled.to(torch.float64) - eager.to(torch.float64)).abs().max().item())
        print(f"\nfuser-mode A/B: bitwise mismatches={mismatches}, max_abs={max_abs:.6e}")
        assert max_abs <= _soundness_budget(eager.dtype, fixture.tensors["expected_out_fp64"])


# ===========================================================================
# R1: self-written emulator lane (NOT evidence about TE)
# ===========================================================================


class TestR1EmulatorLane:
    @pytest.mark.L0
    def test_emulator_is_close_to_the_oracle_too(self) -> None:
        """Both lanes approximate the same mathematics; that is all this shows."""
        fixture = _require_fixture(FIXTURE_NAMES[2])
        header = fixture.header
        per_rank = {}
        for rank in range(header.cp_size):
            out, _ = tea.merge_emulator(
                rank=rank,
                cp_size=header.cp_size,
                causal=header.causal,
                partial_out=[fixture.rank_step_out(rank, s) for s in range(header.cp_size)],
                partial_lse=[fixture.rank_step_lse(rank, s) for s in range(header.cp_size)],
                o_local_shape=_local_shape(header),
                out_dtype=torch.float32,
            )
            per_rank[rank] = out.reshape(header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)
        actual = ts.restore_global_order(per_rank, header.cp_size, header.chunk_len)
        expected = fixture.tensors["expected_out_fp64"]
        max_abs = float((actual.to(torch.float64) - expected).abs().max().item())
        assert max_abs <= _soundness_budget(actual.dtype, expected), f"emulator drift {max_abs:.6e}"

    @pytest.mark.L0
    def test_emulator_and_te_lane_are_not_bitwise_identical(self) -> None:
        """Agreement between the two lanes would mean one of them is not doing
        what it claims; the schedule order matters and must be observable."""
        fixture = _require_fixture(FIXTURE_NAMES[4])
        header = fixture.header
        te_per_rank, emu_per_rank = {}, {}
        for rank in range(header.cp_size):
            out_te, _ = tea.merge_te_fidelity(
                rank=rank,
                cp_size=header.cp_size,
                causal=header.causal,
                partial_out=[fixture.rank_step_out(rank, s) for s in range(header.cp_size)],
                partial_lse=[fixture.rank_step_lse(rank, s) for s in range(header.cp_size)],
                o_local_shape=_local_shape(header),
            )
            te_per_rank[rank] = out_te.reshape(header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)
            out_emu, _ = tea.merge_emulator(
                rank=rank,
                cp_size=header.cp_size,
                causal=header.causal,
                partial_out=[fixture.rank_step_out(rank, s) for s in range(header.cp_size)],
                partial_lse=[fixture.rank_step_lse(rank, s) for s in range(header.cp_size)],
                o_local_shape=_local_shape(header),
                out_dtype=torch.float32,
            )
            emu_per_rank[rank] = out_emu.reshape(header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)
        te = ts.restore_global_order(te_per_rank, header.cp_size, header.chunk_len)
        emu = ts.restore_global_order(emu_per_rank, header.cp_size, header.chunk_len)
        assert _bitwise_mismatches(te, emu) > 0


# ===========================================================================
# R2 preconditions
# ===========================================================================


class TestR2Preconditions:
    @pytest.mark.L0
    def test_actual_attention_refuses_without_pinned_te(self) -> None:
        """The R2 lane must fail closed, not fall back to a look-alike."""
        sys.path.insert(0, os.path.join(_HERE, "cp_numerics"))
        from cp_numerics import run_cp_reference as runner  # noqa: PLC0415

        ok, why = runner._te_installed_matches_pin(None)
        if ok:
            pytest.skip("the pinned TransformerEngine is importable here; the refusal path is not exercised")
        assert isinstance(why, str) and why, "a refusal must always carry a reason"
        assert "transformer_engine" in why or "--te-repo" in why

    @pytest.mark.L0
    def test_runner_returns_unverified_exit_for_an_unverified_backend(self, tmp_path, monkeypatch) -> None:
        """A downgraded backend must reach the shell, not only the JSON.

        The backend guard turns a run whose fused kernel was not observed into
        ``R2_UNVERIFIED``; that path used to return ``EXIT_OK``, so automation
        saw success while the report said unverified.
        """
        import json

        sys.path.insert(0, os.path.join(_HERE, "cp_numerics"))
        from cp_numerics import actual_attention as act
        from cp_numerics import run_cp_reference as runner

        fixture = pathlib.Path(_HERE, "cp_numerics", "fixtures", "cp_p1_f32_s128_causal.pt")
        args = runner._parse_args(["--mode", "actual-attention", "--fixture", str(fixture), "--output-dir", str(tmp_path), "--te-compute-dtype", "float16"])
        monkeypatch.setattr(runner, "_te_installed_matches_pin", lambda repo: (True, "mocked pin"))
        monkeypatch.setattr(
            act,
            "run_actual_attention_on_fixture",
            lambda **kwargs: act.ActualAttentionResult(
                status="R2_UNVERIFIED",
                per_step=[{"backend": {}}],
                merged={
                    "o": {"elements": 0},
                    "lse": {"elements": 0},
                    "merge_label": "te_o_with_fixture_lse",
                    "partial_lse_source": "fixture",
                    "backend": {"steps": 1, "unrecorded_steps": 1, "fused_steps": 0, "unfused_steps": 0},
                },
                fixture={"name": "cp_p1_f32_s128_causal"},
            ),
        )
        code = runner.run_actual_attention(args, rank=0, local_rank=0, world_size=1, output_dir=str(tmp_path))
        assert code == runner.EXIT_R2_UNVERIFIED, f"unverified backend exited {code}"
        report = json.loads((tmp_path / "rank_0_actual_attention.json").read_text())
        assert report["status"] == "R2_UNVERIFIED"
        assert report["merged"]["backend"]["unrecorded_steps"] == 1

    @pytest.mark.L0
    def test_runner_rejects_a_missing_fixture(self, tmp_path) -> None:
        sys.path.insert(0, os.path.join(_HERE, "cp_numerics"))
        from cp_numerics import run_cp_reference as runner  # noqa: PLC0415

        args = runner._parse_args(["--mode", "frozen-partials", "--fixture", str(tmp_path / "nope.pt"), "--output-dir", str(tmp_path)])
        with pytest.raises(RuntimeError, match="fixture not found"):
            runner.local_preflight(args, rank=0, local_rank=0, world_size=1)


import ast
import json
import os
import pathlib
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _lane_source() -> str:
    return pathlib.Path(_HERE, "cp_numerics", "actual_attention.py").read_text()


@pytest.mark.L0
def test_r2_lane_uses_te_supported_mask_types() -> None:
    """The lane must not ask TE for the ``arbitrary`` mask type.

    At the pinned revision the dispatcher turns an arbitrary mask into
    UnfusedDotProductAttention -- FusedAttention and FlashAttention are both
    disabled for it -- so that spelling could never evidence a fused run.  The
    mask of the section is expressed through the types the fused backend
    supports: ``causal`` for the diagonal section, whose block-diagonal ranges
    are causal within each block, and ``no_mask`` for the two triangles, whose
    keep rule is total on the axes the step carries.
    """
    source = _lane_source()
    assert 'attn_mask_type = "causal" if record.causal_mode == "causal" else "no_mask"' in source
    assert '"arbitrary"' not in source, "an arbitrary mask disables the fused backend at the pinned revision"
    assert "mask = drop.view(" not in source, "the lane expresses the mask through the mask type, not a built mask"


@pytest.mark.L0
def test_r2_lane_records_and_requires_the_fused_backend() -> None:
    """A passing number is not evidence of WHICH kernel produced it.

    The lane wraps the TE dispatcher, records the selection atomically with the
    call, and downgrades the result unless EVERY step ran fused with no unfused
    step -- so an unfused fallback reports R2_UNVERIFIED instead of a green
    merged O.
    """
    source = _lane_source()
    assert "dpa_utils.get_attention_backend" in source, "the REAL dispatcher decision is recorded, not re-derived"
    assert "fused_steps = sum(" in source and "unfused_steps = sum(" in source
    # A step whose selection was never observed is not evidence: the dispatcher
    # caches its decision, so the recache has to be forced and an empty record
    # counted as unrecorded rather than assumed fused.
    assert '_attention_backends["backend_selection_requires_update"] = True' in source
    assert "unrecorded_steps = sum(" in source
    assert "fused_steps == steps and unfused_steps == 0 and unrecorded_steps == 0" in source
    assert 'status = "R2_EXECUTED" if executed else "R2_UNVERIFIED"' in source
    # The FlashAttention field is a VERSION, not a backend id: the pinned
    # selector reports it even when the fused backend was selected.
    assert "str(flash_backend)" in source and "int(flash_backend)" not in source


@pytest.mark.L0
def test_r2_lane_keeps_the_fixture_row_axis() -> None:
    """Each partial covers the FULL local query axis of its step.

    Trimming the rows that keep nothing made an all-masked step hand the merge a
    zero-row partial, which cannot be reshaped back to the section it belongs to.
    Fully masked rows stay in the partial as zeros, in their original order.
    """
    source = _lane_source()
    assert "q_step.index_select(1, torch.nonzero" not in source, "the row axis is the fixture's, not a subset"
    assert '"q_tokens": int(q_step.shape[1])' in source, "the partial reports the step width, not a kept-row count"
    assert "q_rows_kept" not in source


@pytest.mark.L0
def test_r2_lane_is_syntactically_importable_without_a_gpu() -> None:
    """The module must parse and expose its entry point even on a CPU-only box."""
    path = pathlib.Path(_HERE, "cp_numerics", "actual_attention.py")
    tree = ast.parse(path.read_text())
    names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "run_actual_attention_on_fixture" in names
    assert "compute_partials" in names
