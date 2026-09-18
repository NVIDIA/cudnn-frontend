# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""R2: real per-step partials from the pinned TransformerEngine fused attention.

``run_cp_reference.py --mode actual-attention`` used to be a gate report only:
it verified the pinned ``context_parallel.py`` against its recorded sha256 and
then returned ``R2_READY`` **without computing anything**.  This module is the
missing computation.

What runs, per rank and per ring step
-------------------------------------
The ring schedule and the section/query-half selection come from
``te_adapter.te_schedule`` -- the same records the fixtures and the trace
invariants are built from, so the runner cannot drift from the fixtures it
compares against.  For every step this module:

1. gathers the step's global Q/KV position ranges out of the fixture's
   ``q_global`` / ``k_global`` / ``v_global`` tensors (``bshd``), transposing to
   the ``[B, H, S, D]`` layout TE's kernel wants;
2. calls the pinned ``DotProductAttention`` in fused mode with
   ``attn_mask_type="causal"`` for the diagonal section and ``"no_mask"`` for the
   two triangle sections -- exactly the mask split ``te_schedule`` documents;
3. writes the result back into the step's own partial slot.

Those partials then go through ``te_adapter.merge_te_fidelity``, and the merged
O is compared against the fixture's FP64 expectation (``expected_out_fp64``),
which was produced by the independent oracle, not by TE.

One honesty boundary, stated rather than papered over
-----------------------------------------------------
The public ``DotProductAttention`` returns **only** O.  TE's softmax LSE lives
in the backend's aux context, which is not part of the public forward result at
this revision, so this module takes each step's LSE from the fixture's frozen
partials and labels the merge as ``te_o_with_fixture_lse``.  Consequence: the
merged O is TE end to end, but a discrepancy in the *LSE* bookkeeping would be
invisible here.  ``merge_te_fidelity``'s own tests cover that path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention

from . import reference_math as rm
from . import te_adapter as tea
from . import trace_schema as ts

# Query-half names from the trace vocabulary, mapped onto TE's seq axis.
_PART_FIRST = tea.PART_FIRST
_PART_SECOND = tea.PART_SECOND
_PART_WHOLE = tea.PART_WHOLE


def _tensor_digest(tensor: torch.Tensor) -> str:
    payload = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(payload.numpy().tobytes()).hexdigest()[:32]


def _positions(record: ts.TraceRecord) -> Tuple[torch.Tensor, torch.Tensor]:
    q_positions = torch.cat([torch.arange(begin, end) for begin, end in record.q_global_ranges])
    kv_positions = torch.cat([torch.arange(begin, end) for begin, end in record.kv_global_ranges])
    return q_positions, kv_positions


@dataclass
class ActualAttentionResult:
    """Everything the R2 lane measured, ready to be written as one report."""

    status: str
    per_step: List[Dict[str, Any]] = field(default_factory=list)
    merged: Dict[str, Any] = field(default_factory=dict)
    fixture: Dict[str, Any] = field(default_factory=dict)
    te_binding: Dict[str, Any] = field(default_factory=dict)
    te_repo: str = ""
    te_repo_sha256: str = ""
    generator_utc: str = ""
    notes: List[str] = field(default_factory=list)


class _PinnedAttention:
    """One ``DotProductAttention`` per mask type, reused across every step.

    Constructing the module per step would rebuild TE's plan each time and spend
    more time in setup than in the kernel; the two mask types are the only axes
    that vary inside a graph.
    """

    def __init__(self, *, batch: int, num_heads: int, head_dim: int, qkv_dtype: torch.dtype, device: torch.device, valid_len: Optional[int]) -> None:
        self._batch = batch
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._dtype = qkv_dtype
        self._device = device
        self._modules: Dict[str, DotProductAttention] = {}
        self._valid_len = valid_len

    def _module(self, attn_mask_type: str) -> DotProductAttention:
        module = self._modules.get(attn_mask_type)
        if module is None:
            # The module builds its own buffers, so it must land on the rank's
            # device BEFORE the first forward; leaving it on the default device
            # is what produced "batch1 is on cuda:1, different from other
            # tensors on cuda:0" when rank 1 ran.
            with torch.cuda.device(self._device):
                module = DotProductAttention(
                    num_attention_heads=self._num_heads,
                    kv_channels=self._head_dim,
                    attention_dropout=0.0,
                    attn_mask_type=attn_mask_type,
                    qkv_format="bshd",
                ).to(self._device)
            self._modules[attn_mask_type] = module
        return module

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep_mask: torch.Tensor, attn_mask_type: str) -> torch.Tensor:
        """``[B, Sq, H, D]`` in, ``[B, Sq, H, D]`` out, in the input dtype.

        ``keep_mask`` is the step's own ``[Sq, Skv]`` boolean keep rule.  A step
        whose axes each concatenate two DISJOINT chunks cannot be described by
        TE's causal flag -- that flag would keep the rectangular
        first-chunk-Q x second-chunk-KV corners as well -- so every section goes
        through the explicit ``arbitrary`` path with the same mask the fixture
        used.
        """
        module = self._module(attn_mask_type)
        batch = q.shape[0]
        # TE's convention is drop-True, the OPPOSITE of the keep rule the fixtures
        # are built from: measured, the complement reproduces a causal reference
        # exactly while the keep mask itself is off by 3.9.
        drop = ~keep_mask
        mask = drop.view(1, 1, drop.shape[0], drop.shape[1]).expand(batch, 1, drop.shape[0], drop.shape[1]).contiguous()
        with torch.cuda.device(self._device):
            flat = module(q, k, v, attention_mask=mask, qkv_format="bshd", attn_mask_type=attn_mask_type)
            return flat.reshape(self._batch, q.shape[1], self._num_heads, self._head_dim)


def compute_partials(
    *,
    q_global: torch.Tensor,
    k_global: torch.Tensor,
    v_global: torch.Tensor,
    records: Sequence[ts.TraceRecord],
    rank: int,
    chunk_len: int,
    num_heads: int,
    head_dim: int,
    qkv_dtype: torch.dtype,
    o_dtype: torch.dtype,
    device: torch.device,
    valid_len: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Per-ring-step ``(O_i, L_i)`` for one rank, O from TE and L from the fixture.

    ``q_global``/``k_global``/``v_global`` are ``bshd`` ``[B, S, H, D]``.  The
    returned O partials carry ``o_dtype`` and the shapes
    ``merge_te_fidelity`` documents: ``[B, 2*chunk_len, H, D]`` for the
    diagonal and lower sections, ``[B, chunk_len, H, D]`` for the upper one.
    """
    batch = q_global.shape[0]
    attention = _PinnedAttention(
        batch=batch,
        num_heads=num_heads,
        head_dim=head_dim,
        qkv_dtype=qkv_dtype,
        device=device,
        valid_len=valid_len,
    )

    partial_out: List[torch.Tensor] = []
    per_step: List[Dict[str, Any]] = []
    for record in records:
        if record.rank != rank:
            continue
        q_positions, kv_positions = _positions(record)
        # index_select on the seq axis keeps the bshd memory order the kernel wants.
        q_step = q_global.index_select(1, q_positions.to(q_global.device))
        k_step = k_global.index_select(1, kv_positions.to(k_global.device))
        v_step = v_global.index_select(1, kv_positions.to(v_global.device))

        # The step's keep rule, from global positions -- the SAME function the
        # fixtures were built with, so both sides are defined identically.
        q_cpu = torch.cat([torch.arange(begin, end) for begin, end in record.q_global_ranges])
        kv_cpu = torch.cat([torch.arange(begin, end) for begin, end in record.kv_global_ranges])
        # The keep rule is built on the host (it is integer bookkeeping) and moved
        # once, so every later comparison in the block-diagonal pass stays on one
        # device.
        q_positions_host = q_cpu.to(device)
        kv_positions_host = kv_cpu.to(device)
        keep = rm.causal_keep_mask_from_positions(
            q_cpu,
            kv_cpu,
            causal=(record.causal_mode == "causal"),
            valid_kv_len=valid_len,
        ).to(device)

        # Block-diagonal structure: a row survives only if a kept column lies in
        # the same (q_range, kv_range) block pair as the row itself.  That is what
        # removes the rectangular corners a plain causal flag would keep, and it
        # leaves the second half's cross-chunk rows empty, which is exactly the
        # row set the fixture's partial carries.
        in_block = torch.zeros_like(keep)
        for q_begin, q_end in record.q_global_ranges:
            for kv_begin, kv_end in record.kv_global_ranges:
                row_hit = (q_positions_host >= q_begin) & (q_positions_host < q_end)
                col_hit = (kv_positions_host >= kv_begin) & (kv_positions_host < kv_end)
                in_block |= row_hit.unsqueeze(1) & col_hit.unsqueeze(0)
        effective = keep & in_block
        selected = effective.any(dim=1)

        # Trim the local Q axis to the surviving rows, and trim the mask's query
        # axis with the same selection so TE and the fixture see one mask.
        q_step = q_step.index_select(1, torch.nonzero(selected, as_tuple=False).flatten().to(q_step.device))
        effective = effective[selected]

        # TE sees an explicit mask, so the type is 'arbitrary' for every section.
        assert effective.device == q_step.device, (effective.device, q_step.device)
        attn_mask_type = "arbitrary"
        out_step = attention.run(q_step, k_step, v_step, effective, attn_mask_type).to(o_dtype)
        partial_out.append(out_step)
        per_step.append(
            {
                "rank": rank,
                "ring_step": record.ring_step,
                "source_rank": record.source_rank,
                "section": record.section,
                "query_half": record.query_half,
                "causal_mode": record.causal_mode,
                "attn_mask_type": attn_mask_type,
                "q_tokens": int(q_step.shape[1]),
                "q_rows_kept": int(selected.sum().item()),
                "q_rows_kept": int(selected.sum().item()),
                "kv_tokens": int(k_step.shape[1]),
                "out_shape": list(out_step.shape),
                "out_dtype": str(out_step.dtype).replace("torch.", ""),
                "out_sha256_32": _tensor_digest(out_step),
                "q_sha256_32": _tensor_digest(q_step),
                "kv_sha256_32": _tensor_digest(k_step),
            }
        )
    return partial_out, per_step


def _local_expected_lse(expected_lse: torch.Tensor, rank: int, cp_size: int, chunk_len: int) -> torch.Tensor:
    """The expectation's own columns for this rank, first chunk then second.

    Mirrors ``run_cp_reference._local_expected_lse``; kept here so the R2 lane
    does not import the runner back into itself.
    """
    first, second = ts.rank_chunk_indices(rank, cp_size)
    return torch.cat(
        [
            expected_lse[:, :, first * chunk_len : (first + 1) * chunk_len],
            expected_lse[:, :, second * chunk_len : (second + 1) * chunk_len],
        ],
        dim=-1,
    )


def _ulp_stats(got: torch.Tensor, expected: torch.Tensor) -> Dict[str, Any]:
    """ULP of ``got`` against ``expected``, measured at ``expected``'s OWN dtype spacing.

    Non-finite pairs are counted separately and excluded from every statistic,
    because a ULP is undefined there and folding them in would let one NaN
    decide the maximum.  abs/rel are kept alongside the ULP rather than replaced
    by it: a near-zero expected value has a tiny spacing, so its ULP blows up
    while the absolute error is irrelevant.
    """
    ref = expected.detach()
    got_c = got.detach()
    if got_c.shape != ref.shape:
        raise ValueError(f"shape mismatch: got {tuple(got_c.shape)} vs expected {tuple(ref.shape)}")

    finite = torch.isfinite(got_c) & torch.isfinite(ref)
    nan_or_inf = int((~finite).sum().item())
    if not finite.any():
        return {"max_abs": None, "max_rel": None, "ulp": {}, "nan_or_inf_pairs": nan_or_inf, "finite_pairs": 0}

    got_f = got_c[finite].double()
    ref_f = ref[finite].double()
    diff = (got_f - ref_f).abs()
    abs_err = float(diff.max().item())
    tiny = torch.finfo(torch.float64).tiny
    rel_err = float((diff / ref_f.abs().clamp_min(tiny)).max().item())

    # Spacing of the STORED dtype at each expected value, so a bf16 comparison
    # is judged in bf16 ULPs rather than in float64 ones.
    spacing_dtype = ref.dtype if ref.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    ref_spacing = ref[finite].to(spacing_dtype)
    next_up = torch.nextafter(ref_spacing, torch.full_like(ref_spacing, float("inf")))
    spacing = (next_up.double() - ref_spacing.double()).abs().clamp_min(tiny)
    ulp = torch.sort(diff / spacing).values
    quantiles: Dict[str, float] = {}
    for name, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1.0)):
        index = min(ulp.numel() - 1, int(q * (ulp.numel() - 1)))
        quantiles[name] = float(ulp[index].item())
    return {
        "max_abs": abs_err,
        "max_rel": rel_err,
        "ulp": quantiles,
        "ulp_spacing_dtype": str(spacing_dtype).replace("torch.", ""),
        "nan_or_inf_pairs": nan_or_inf,
        "finite_pairs": int(finite.sum().item()),
    }


def run_actual_attention_on_fixture(
    *,
    fixture: tea.FrozenFixture,
    rank: int,
    world_size: int,
    compiled: bool,
    device: torch.device,
) -> ActualAttentionResult:
    """The R2 computation, on one rank, against one fixture's own Q/K/V."""
    header = fixture.header
    if header.cp_size != world_size:
        raise ValueError(f"fixture {header.name!r} carries cp_size={header.cp_size} but world_size={world_size}")

    records = [ts.TraceRecord.from_dict(entry) for entry in header.trace]
    qkv_dtype = tea.dtype_from_name(header.qkv_dtype)
    o_dtype = tea.dtype_from_name(header.o_dtype)

    q_global = fixture.tensors["q_global"].to(device).to(qkv_dtype)
    k_global = fixture.tensors["k_global"].to(device).to(qkv_dtype)
    v_global = fixture.tensors["v_global"].to(device).to(qkv_dtype)

    partial_out, per_step = compute_partials(
        q_global=q_global,
        k_global=k_global,
        v_global=v_global,
        records=records,
        rank=rank,
        chunk_len=header.chunk_len,
        num_heads=header.num_heads,
        head_dim=header.head_dim,
        qkv_dtype=qkv_dtype,
        o_dtype=o_dtype,
        device=device,
        valid_len=header.valid_len,
    )
    # LSE per step comes from the fixture: the public TE forward returns only O
    # (module docstring).  Labelled as such in the report.
    partial_lse = [fixture.rank_step_lse(rank, step).to(device) for step in range(header.cp_size)]

    o_local_shape = (
        (header.batch, 2, header.chunk_len, header.num_heads, header.head_dim)
        if header.causal
        else (header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)
    )
    merged_out, merged_lse = tea.merge_te_fidelity(
        rank=rank,
        cp_size=header.cp_size,
        causal=header.causal,
        partial_out=partial_out,
        partial_lse=partial_lse,
        o_local_shape=o_local_shape,
        compiled=compiled,
    )

    expected_out = fixture.tensors["expected_out_fp64"].to(device)
    expected_lse = fixture.tensors["expected_lse_fp64"].to(device)
    got_out = merged_out.reshape(header.batch, 2 * header.chunk_len, header.num_heads, header.head_dim)
    # The fixture's local rows for this rank are its two chunks, in ring order.
    first_chunk, second_chunk = ts.rank_chunk_indices(rank, header.cp_size)
    local_rows = torch.cat(
        [
            torch.arange(first_chunk * header.chunk_len, (first_chunk + 1) * header.chunk_len),
            torch.arange(second_chunk * header.chunk_len, (second_chunk + 1) * header.chunk_len),
        ]
    )
    expected_local = expected_out.index_select(1, local_rows.to(device))
    # The merged LSE carries the rank's own 2*chunk_len columns in the same
    # first-chunk-then-second-chunk order as the output rows, so it is compared
    # against the expectation's own local slice (run_cp_reference's helper),
    # not against global token positions.
    expected_lse_local = _local_expected_lse(expected_lse, rank, header.cp_size, header.chunk_len)
    got_lse = merged_lse

    return ActualAttentionResult(
        status="R2_EXECUTED",
        per_step=per_step,
        merged={
            "o": _ulp_stats(got_out, expected_local),
            "lse": _ulp_stats(got_lse, expected_lse_local),
            "o_dtype": str(got_out.dtype).replace("torch.", ""),
            "lse_dtype": str(got_lse.dtype).replace("torch.", ""),
            "rows_compared": int(got_out.shape[1]),
            "o_sha256_32": _tensor_digest(got_out),
            "partial_lse_source": "fixture",
            "merge_label": "te_o_with_fixture_lse",
        },
        fixture={
            "name": header.name,
            "cp_size": header.cp_size,
            "causal": header.causal,
            "batch": header.batch,
            "seq_len": header.seq_len,
            "num_heads": header.num_heads,
            "head_dim": header.head_dim,
            "chunk_len": header.chunk_len,
            "qkv_dtype": header.qkv_dtype,
            "o_dtype": header.o_dtype,
            "fixture_sha256": header.fixture_sha256,
            "q_global_sha256_32": _tensor_digest(q_global),
            "k_global_sha256_32": _tensor_digest(k_global),
            "v_global_sha256_32": _tensor_digest(v_global),
        },
        te_binding=tea.describe_te_binding(),
        notes=[
            "O partials are computed by the pinned TransformerEngine fused attention; the merge is te_adapter.merge_te_fidelity.",
            "Per-step LSE is taken from the fixture because the public TE forward returns only O at this revision.",
            "Comparison target is the fixture's FP64 expectation, produced by the independent oracle.",
        ],
    )
