# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer's paged cuDNN DECODE graph, re-declared byte for byte, against FE's DEFAULT placement on cc 10.7.

FlashInfer's ``cudnn_batch_decode_with_kv_cache`` builds one FE graph per (shape, flags): a dense 4-D Q ``(b, h_qo, q_len, d)`` --
``(b, h, d)`` for a single token, the ``(b, q_len, h, d)`` view transposed for a multi-token (speculative / MTP) step -- over the
page pools ``[pages, H_kv, page, D]`` bound by ``tensor_like`` (any in-page layout), the block tables as ``(b, 1, max_pages, 1)``
``tensor_like`` tensors, the per-request KV lengths as a ``(b, 1, 1, 1)`` int32 ``tensor_like`` with ``set_is_pass_by_value(False)``
(the padding mask), the Q lengths likewise, ``paged_attention_max_seq_len_kv``, no Stats, the bottom-right causal diagonal when
``q_len > 1``, and the CUDA-graph replay hint when its caller captures.  The decode twin of ``test_flashinfer_shaped_sdpa.py``.

What it pins (the decode-tile placement bands of ``placement._place_sm107_f16``, the MEASURED frontier of the tile against the backend's
default plan on this very graph): on every LEAD cell FE's DEFAULT plan is the FROST row ``sdpa_fwd_prefill_sm107`` served by the Rubin
d256 decode tile (the route readback: the kernel template is ``decode_d256_f16``); on every TRAIL cell the default is the backend's
plan, by name.  Both regimes (the replay hint set and unset) agree on the verdict.  The default plan is also EXECUTED under the D2H
detector and compared with the backend's plan pinned on the SAME graph (the numbers, not only the name); a padded entry (``kv_len = 0``
over the NULL block ``0``) inside a LEAD cell keeps the verdict and writes exactly ``0``; one MTP cell per routed ``q_len`` (2: one
token unit, 3 / 4: two) rides the 32-column tile.  Skips off cc 10.7 and below the DSL floor (the arch-aware probes).
"""

from __future__ import annotations

import math
import zlib

import pytest
import torch

import cudnn
from frost_test_utils import _SM, requires_dsl
from test_sdpa_fwd_decode_d256_sm100 import _pools, _served_by
from test_sdpa_fwd_decode_d256_sm107 import requires_sm107_dsl

requires_cc107 = pytest.mark.skipif(
    _SM != 107, reason="the Rubin paged d256 decode-tile default placement exists on cc 10.7 only; have " + ("none" if _SM is None else f"sm_{_SM}")
)
pytestmark = [pytest.mark.L0, requires_dsl, requires_cc107, requires_sm107_dsl]

# flashinfer/cudnn/decode.py UIDs, kept identical.
Q_UID, K_UID, V_UID = 1, 2, 3
ACTUAL_SEQ_LENS_Q_UID, ACTUAL_SEQ_LENS_KV_UID = 100, 101
BLOCK_TABLES_K_UID, BLOCK_TABLES_V_UID = 201, 202
O_UID = 1000
FROST_PREFIX = "sdpa_fwd_prefill_sm107["
FROST_ENGINE = "frost:sdpa_fwd_prefill_sm107"
DECODE_TILE = "decode_d256_f16"
D = 256


class _DecodeCase:
    """One FlashInfer decode call: the buffers, the graph it builds and the two plans of interest (FE's default, the backend's)."""

    def __init__(self, *, heads, b, kv, q_len=1, page=16, hnd=False, dtype=torch.bfloat16, dead_entry=None, seed=0):
        torch.manual_seed(seed)
        dev = torch.device("cuda")
        self.h_q, self.h_kv = (int(x) for x in heads.split("/"))
        self.b, self.kv, self.q_len, self.page, self.dtype = b, kv, q_len, page, dtype
        self.scale = 1.0 / math.sqrt(D)
        # the per-request KV lengths: random inside the declared KV, at least q_len (the diagonal needs them); the dead entry 0 over the NULL block
        lens = torch.randint(max(q_len, 1), kv + 1, (b,), generator=torch.Generator().manual_seed(seed)).tolist()
        self.dead = dead_entry
        if dead_entry is not None:
            lens[dead_entry] = 0
        self.lens = lens
        k_dense = torch.randn(b, kv, self.h_kv, D, device=dev, dtype=dtype)
        v_dense = torch.randn(b, kv, self.h_kv, D, device=dev, dtype=dtype)
        self.k_cache, self.v_cache, bt = _pools(k_dense, v_dense, page, hnd, seed)  # [pages, H_kv, page, D] containers (HND compact or NHD by strides)
        self.max_pages = int(bt.shape[2])
        self.block_tables = bt.view(b, self.max_pages).contiguous()
        if dead_entry is not None:
            self.block_tables[dead_entry] = 0  # the serving stacks' NULL block for a padded entry
        self.seq_lens_kv = torch.tensor(lens, dtype=torch.int32, device=dev).view(b, 1, 1, 1)
        self.seq_lens_q = torch.full((b, 1, 1, 1), q_len, dtype=torch.int32, device=dev)
        # FlashInfer's q: (b * q_len, h, d) rows; the graph sees (b, h, q_len, d) at the view's strides ((h*d, d, h*d, 1) for q_len 1)
        self.q_rows = torch.randn(b * q_len, self.h_q, D, device=dev, dtype=dtype)
        self.q_graph = self.q_rows.view(b, q_len, self.h_q, D).transpose(1, 2)
        self.io = cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16

    def build(self, handle, *, replay):
        b, h, s, d = self.q_graph.shape
        kw = dict(is_cuda_graph_replay_expected=True) if replay else {}
        g = cudnn.pygraph(handle=handle, io_data_type=self.io, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT, **kw)
        q = g.tensor(name="q", dim=(b, h, s, d), stride=tuple(self.q_graph.stride()), data_type=self.io)
        k = g.tensor_like(self.k_cache)
        v = g.tensor_like(self.v_cache)
        q.set_uid(Q_UID)
        k.set_uid(K_UID)
        v.set_uid(V_UID)
        nd_bt = self.block_tables.reshape(b, 1, self.max_pages, 1)
        tk = g.tensor_like(nd_bt)
        tk.set_uid(BLOCK_TABLES_K_UID)
        tv = g.tensor_like(nd_bt)
        tv.set_uid(BLOCK_TABLES_V_UID)
        slq = g.tensor_like(self.seq_lens_q)
        slq.set_uid(ACTUAL_SEQ_LENS_Q_UID)
        slk = g.tensor_like(self.seq_lens_kv)
        slk.set_uid(ACTUAL_SEQ_LENS_KV_UID)
        slk.set_is_pass_by_value(False)
        mask = dict(use_causal_mask_bottom_right=True) if s > 1 else {}
        o, _ = g.sdpa(
            name="sdpa",
            q=q,
            k=k,
            v=v,
            seq_len_q=slq,
            seq_len_kv=slk,
            use_padding_mask=True,
            generate_stats=False,
            attn_scale=self.scale,
            paged_attention_k_table=tk,
            paged_attention_v_table=tv,
            paged_attention_max_seq_len_kv=self.kv,
            compute_data_type=cudnn.data_type.FLOAT,
            **mask,
        )
        o.set_uid(O_UID).set_output(True).set_dim([b, h, s, d]).set_stride([s * h * d, d, h * d, 1]).set_data_type(self.io)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        return g

    def pack(self, out):
        return {
            Q_UID: self.q_rows,
            K_UID: self.k_cache,
            V_UID: self.v_cache,
            O_UID: out,
            ACTUAL_SEQ_LENS_Q_UID: self.seq_lens_q,
            ACTUAL_SEQ_LENS_KV_UID: self.seq_lens_kv,
            BLOCK_TABLES_K_UID: self.block_tables.reshape(self.b, 1, self.max_pages, 1),
            BLOCK_TABLES_V_UID: self.block_tables.reshape(self.b, 1, self.max_pages, 1),
        }

    def run_default(self, *, replay):
        """FE's DEFAULT plan: name, the engine that serves it, the kernel template (the route readback) and its output."""
        handle = cudnn.create_handle()
        g = self.build(handle, replay=replay)
        g.check_support()
        g.build_plans()
        idx = int(g._plan_index)
        eng = g.selected_engine
        facts = dict(
            plan_name=g.get_plan_name_at_index(idx),
            served_by=("frost:" + eng.name) if eng is not None else "cudnn-backend",
            template=_served_by(g, idx) if eng is not None else None,
            replay_hint=bool(getattr(g, "is_cuda_graph_replay_expected", False)),
        )
        out = torch.full((self.b * self.q_len, self.h_q, D), float("nan"), device="cuda", dtype=self.dtype)
        ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
        torch.cuda.set_sync_debug_mode("error")  # Rule 3: the lengths and page tables are read on device
        try:
            g.execute(self.pack(out), ws, handle=handle)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        return facts, out

    def run_backend(self):
        """The backend's plan pinned by name on the SAME graph (the first non-FROST entry of the ranked list)."""
        from cudnn.engines import is_python_engine

        handle = cudnn.create_handle()
        g = self.build(handle, replay=False)
        idx = next(
            i
            for i in range(g.get_execution_plan_count())
            if not is_python_engine(g.get_engine_and_knobs_at_index(i)[0]) and g.get_plan_name_at_index(i) != "backend_heuristics"
        )
        g.select_plan(idx)
        g.check_support()
        g.build_plans()
        out = torch.full((self.b * self.q_len, self.h_q, D), float("nan"), device="cuda", dtype=self.dtype)
        ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
        g.execute(self.pack(out), ws, handle=handle)
        torch.cuda.synchronize()
        return g.get_plan_name_at_index(idx), out


def _assert_default(case: _DecodeCase, verdict: str, *, replay: bool):
    facts, out = case.run_default(replay=replay)
    assert facts["replay_hint"] is replay
    if verdict == "LEAD":
        assert facts["served_by"] == FROST_ENGINE and facts["plan_name"].startswith(FROST_PREFIX), facts
        assert facts["template"] == DECODE_TILE, facts  # the route readback: the Rubin d256 decode tile (its split combine rides it)
    else:
        assert facts["served_by"] == "cudnn-backend" and not facts["plan_name"].startswith(FROST_PREFIX), facts
    assert not torch.isnan(out.float()).any(), "the default plan left NaN in O"
    return facts, out


def _assert_numbers(case: _DecodeCase, out_default: torch.Tensor):
    """The default plan's O against the backend's plan on the same graph, every live row; a dead entry's rows exactly 0 (the tile's
    empty-range select; the backend's convention for a keyless row is its own, so it is not compared there)."""
    backend_name, out_ref = case.run_backend()
    assert not backend_name.startswith(FROST_PREFIX)
    live = torch.ones(case.b, dtype=torch.bool)
    if case.dead is not None:
        live[case.dead] = False
        dead_rows = out_default.view(case.b, case.q_len, case.h_q, D)[case.dead]
        assert dead_rows.abs().max().item() == 0.0, "a padded entry (kv_len 0 over the NULL block) must write O := 0"
    a = out_default.view(case.b, case.q_len, case.h_q, D)[live].float()
    r = out_ref.view(case.b, case.q_len, case.h_q, D)[live].float()
    torch.testing.assert_close(a, r, atol=2e-2 if case.dtype == torch.float16 else 5e-2, rtol=0)


# The cells: the measured frontier of the three bands (placement.py's module docstring) -- LEAD inside, TRAIL one step outside.
_CELLS = [
    # id, kwargs, verdict
    ("24-2_b32_kv8k_q1_tokenmajor", dict(heads="24/2", b=32, kv=8192), "LEAD"),  # the first band, FlashInfer's benchmark pool layout
    ("24-2_b8_kv8k_q1_hnd", dict(heads="24/2", b=8, kv=8192, hnd=True), "LEAD"),  # the 2**16 frontier on vLLM's head-major pool
    ("32-2_b32_kv8k_q1_p32", dict(heads="32/2", b=32, kv=8192, page=32), "LEAD"),
    ("24-2_b32_kv8k_q1_f16", dict(heads="24/2", b=32, kv=8192, dtype=torch.float16), "LEAD"),
    ("64-2_b32_kv8k_q1", dict(heads="64/2", b=32, kv=8192), "LEAD"),  # group 32 inside its band (64 units x 8k >= 2**17)
    ("24-2_b4_kv8k_q2", dict(heads="24/2", b=4, kv=8192, q_len=2), "LEAD"),  # the MTP rows: one token unit of the 32-column tile
    ("32-2_b16_kv8k_q3", dict(heads="32/2", b=16, kv=8192, q_len=3), "LEAD"),  # two token units
    ("24-2_b32_kv4k_q4", dict(heads="24/2", b=32, kv=4096, q_len=4), "LEAD"),  # two token units
    ("24-2_b32_kv8k_q1_dead_entry", dict(heads="24/2", b=32, kv=8192, dead_entry=5), "LEAD"),  # a padded entry keeps the verdict
    ("24-2_b4_kv4k_q1", dict(heads="24/2", b=4, kv=4096), "TRAIL"),  # 2**15 tokens in flight: mixed, the backend stays first
    ("64-2_b4_kv8k_q1", dict(heads="64/2", b=4, kv=8192), "TRAIL"),  # group 32 one step below its band (the parity cell)
    ("24-2_b4_kv8k_q2_p32", dict(heads="24/2", b=4, kv=8192, q_len=2, page=32), "TRAIL"),  # the MTP rows at page 32: unmeasured
]


@pytest.mark.parametrize("replay", [False, True], ids=["eager", "replay"])
@pytest.mark.parametrize("cid, kwargs, verdict", _CELLS, ids=[c[0] for c in _CELLS])
def test_flashinfer_decode_graph_default_plan_is_the_rubin_decode_tile_inside_the_measured_bands(cid, kwargs, verdict, replay, monkeypatch):
    """FE's DEFAULT plan on FlashInfer's decode graph: the Rubin d256 decode tile on every LEAD cell (engine, plan name and the served
    template), the backend by name on every TRAIL cell, the same verdict with the replay hint set and unset; the numbers of the default
    plan against the backend's plan on the same graph (eager cells; the replay-hinted default is the same tile with the split first).
    The opt-in flag is UNSET for the default run, as in a FlashInfer process: with it set every FROST row is placed first."""
    assert tuple(int(x) for x in cudnn.__version__.split(".")[:3]) >= (1, 32, 0)
    # FlashInfer's environment: the opt-in flag UNSET -- the suite's autouse fixture sets it, and with it set every FROST row is placed
    # first whatever the verdict, which would hide the TRAIL half of this pin (the frost suite's own pattern for a default-placement cell).
    monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
    case = _DecodeCase(**kwargs, seed=zlib.crc32(cid.encode()) % 1000)  # a per-cell seed that is stable across processes
    _, out = _assert_default(case, verdict, replay=replay)
    if not replay:
        _assert_numbers(case, out)
