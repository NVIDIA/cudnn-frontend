# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer-shaped SDPA prefill graphs under the frost opt-in.

FlashInfer (flashinfer/cudnn/prefill.py) declares its ragged prefill as a dense
``(b, h, s_max, d)`` graph whose Q/O batch stride is ONE TOKEN (``h * d``,
equal to the sequence stride) with per-tensor ragged offsets, packed
``(T, h, d)`` buffers, padded ``(b, s_max, h)`` LSE, per-batch lengths as
``(b, 1, 1, 1)`` int32 device tensors and a bottom-right causal mask. Two
forms exist: the legacy element-unit offsets, and the token-unit direct form
(``cu_seq_len_q/kv`` + ``set_ragged_offset_multiplier``, cuDNN >= 9.24).

Contract under test for the frost plan: accepted at check_support => builds,
runs, and O / valid LSE rows match the cuDNN backend plan. A decline is xfail
with the reason (today: the THD stride-order check at b > 1). Known contract
gaps carry ``xfail(strict=True)`` so a fix flips them loud.
"""

from __future__ import annotations

import math

import pytest
import torch

import cudnn
from cudnn.engines import is_python_engine
from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell

pytestmark = [pytest.mark.L0, requires_pre_rubin_blackwell, requires_dsl]

# flashinfer/cudnn/prefill.py UIDs, kept identical.
Q_UID, K_UID, V_UID, O_UID, STATS_UID = 0, 1, 2, 3, 4
RAGGED_Q_UID, RAGGED_K_UID, RAGGED_V_UID, RAGGED_O_UID, RAGGED_STATS_UID = 10, 11, 12, 13, 14
SEQ_Q_UID, SEQ_KV_UID = 20, 21
FROST_PREFIX = "sdpa_fwd_prefill_"


def _plan_indices(g):
    frost = backend = None
    for i in range(g.get_execution_plan_count()):
        name = g.get_plan_name_at_index(i)
        engine_id, _ = g.get_engine_and_knobs_at_index(i)
        if name.startswith(FROST_PREFIX) and frost is None:
            frost = i
        elif backend is None and not is_python_engine(engine_id) and name != "backend_heuristics":
            backend = i
    return frost, backend


def _frost_decline_reasons(g) -> str:
    """Why every offered frost SDPA-forward row declined ``g`` (engine name: reason)."""
    from cudnn.engines import manifest

    fam = next(f for f in manifest.MANIFEST if f.name == "frost_sdpa_fwd")
    reasons = []
    for name, engine_id in fam.offered_ids().items():
        engine = manifest.engine_for_id(engine_id)
        if engine is None:
            continue
        reason = engine._decline_reason(g, None)
        if reason is not None and "requires SM" not in reason:  # rows for other archs are not news
            reasons.append(f"{name}: {reason}")
    return "; ".join(reasons) or "no frost row offered on this device"


class _Case:
    """One FlashInfer prefill call: lengths, packed buffers, and the graph it builds."""

    def __init__(self, lens_q, lens_kv, *, h_q=8, h_kv=4, d=128, causal=True, tokens_form=False, dtype=torch.bfloat16):
        torch.manual_seed(0)
        dev = torch.device("cuda")
        self.b, self.h_q, self.h_kv, self.d, self.causal, self.tokens_form = len(lens_q), h_q, h_kv, d, causal, tokens_form
        self.lens_q = torch.tensor(lens_q, dtype=torch.int32, device=dev)
        self.lens_kv = torch.tensor(lens_kv, dtype=torch.int32, device=dev)
        self.s_q, self.s_kv = max(lens_q), max(lens_kv)
        zero = torch.zeros(1, dtype=torch.int32, device=dev)
        self.cu_q = torch.cat([zero, torch.cumsum(self.lens_q, 0).int()])
        self.cu_kv = torch.cat([zero, torch.cumsum(self.lens_kv, 0).int()])
        t_q, t_kv = int(self.cu_q[-1]), int(self.cu_kv[-1])
        self.q = torch.randn(t_q, h_q, d, device=dev, dtype=dtype)
        self.k = torch.randn(t_kv, h_kv, d, device=dev, dtype=dtype)
        self.v = torch.randn(t_kv, h_kv, d, device=dev, dtype=dtype)
        self.scale = 1.0 / math.sqrt(d)
        self.dtype = dtype
        self.t_q = t_q

    def build(self, handle):
        b, h_q, h_kv, d, s_q, s_kv = self.b, self.h_q, self.h_kv, self.d, self.s_q, self.s_kv
        g = cudnn.pygraph(
            handle=handle, io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
        )
        dt = cudnn.datatypes._torch_to_cudnn_data_type(self.dtype)
        s_stride, h_stride, d_stride = self.q.stride()
        # Q: batch stride == one token (h*d) == the s stride; ragged offsets say where each batch starts.
        q = g.tensor(name="q", dim=(b, h_q, s_q, d), stride=(h_q * d, h_stride, s_stride, d_stride), data_type=dt)
        k = g.tensor(name="k", dim=(b, h_kv, s_kv, d), stride=(h_kv * d * s_kv, d, h_kv * d, 1), data_type=dt)
        v = g.tensor(name="v", dim=(b, h_kv, s_kv, d), stride=(h_kv * d * s_kv, d, h_kv * d, 1), data_type=dt)
        rq = g.tensor_like(self.cu_q, name="ragged_q")
        rk = g.tensor_like(self.cu_kv, name="ragged_k")
        rv = g.tensor_like(self.cu_kv, name="ragged_v")
        for t, uid in ((rq, RAGGED_Q_UID), (rk, RAGGED_K_UID), (rv, RAGGED_V_UID)):
            t.set_uid(uid)
        q.set_ragged_offset(rq)
        k.set_ragged_offset(rk)
        v.set_ragged_offset(rv)
        if self.tokens_form:
            q.set_ragged_offset_multiplier(h_q * d)
            k.set_ragged_offset_multiplier(h_kv * d)
            v.set_ragged_offset_multiplier(h_kv * d)
            cu_q = g.tensor_like(self.cu_q, name="cu_seq_lens_q")
            cu_kv = g.tensor_like(self.cu_kv, name="cu_seq_lens_kv")
            cu_q.set_uid(SEQ_Q_UID)
            cu_kv.set_uid(SEQ_KV_UID)
            seq_kwargs = dict(cu_seq_len_q=cu_q, cu_seq_len_kv=cu_kv, implementation=cudnn.attention_implementation.UNIFIED)
        else:
            lq = g.tensor_like(self.lens_q.view(b, 1, 1, 1), name="actual_seq_lens_q")
            lkv = g.tensor_like(self.lens_kv.view(b, 1, 1, 1), name="actual_seq_lens_kv")
            lq.set_uid(SEQ_Q_UID)
            lkv.set_uid(SEQ_KV_UID)
            seq_kwargs = dict(seq_len_q=lq, seq_len_kv=lkv)
        out_t, stats_t = g.sdpa(
            name="sdpa",
            q=q,
            k=k,
            v=v,
            **seq_kwargs,
            use_padding_mask=True,
            attn_scale=self.scale,
            generate_stats=True,
            use_causal_mask_bottom_right=self.causal,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        ro = g.tensor_like(self.cu_q, name="ragged_o")
        ro.set_uid(RAGGED_O_UID)
        out_t.set_ragged_offset(ro)
        rs = g.tensor_like(self.cu_q, name="ragged_stats")
        rs.set_uid(RAGGED_STATS_UID)
        stats_t.set_ragged_offset(rs)
        if self.tokens_form:
            out_t.set_ragged_offset_multiplier(h_q * d)
            stats_t.set_ragged_offset_multiplier(h_q)
        out_t.set_uid(O_UID).set_output(True).set_dim([b, h_q, s_q, d]).set_stride([s_q * d * h_q, d, d * h_q, 1]).set_data_type(dt)
        stats_t.set_uid(STATS_UID).set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim([b, h_q, s_q, 1]).set_stride([s_q * h_q, 1, h_q, 1])
        for t, uid in ((q, Q_UID), (k, K_UID), (v, V_UID)):
            t.set_uid(uid)
        g.validate()
        g.build_operation_graph()
        return g

    def pack(self, out, lse):
        b, h_q, h_kv, d = self.b, self.h_q, self.h_kv, self.d
        if self.tokens_form:
            offs_q, offs_kv, offs_o, offs_s = self.cu_q, self.cu_kv, self.cu_q, self.cu_q
        else:  # legacy element-unit offsets
            offs_q, offs_kv, offs_o, offs_s = self.cu_q * (h_q * d), self.cu_kv * (h_kv * d), self.cu_q * (h_q * d), self.cu_q * h_q
        pack = {
            Q_UID: self.q,
            K_UID: self.k,
            V_UID: self.v,
            O_UID: out,
            STATS_UID: lse,
            RAGGED_Q_UID: offs_q,
            RAGGED_K_UID: offs_kv,
            RAGGED_V_UID: offs_kv,
            RAGGED_O_UID: offs_o,
            RAGGED_STATS_UID: offs_s,
        }
        if self.tokens_form:
            pack[SEQ_Q_UID], pack[SEQ_KV_UID] = self.cu_q, self.cu_kv
        else:
            pack[SEQ_Q_UID], pack[SEQ_KV_UID] = self.lens_q.view(b, 1, 1, 1), self.lens_kv.view(b, 1, 1, 1)
        return pack

    def run(self, *, use_frost: bool):
        handle = cudnn.create_handle()
        g = self.build(handle)
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        frost, backend = _plan_indices(g)
        if use_frost and frost is None:
            # The family heuristics list only rows whose capability check admits
            # the graph, so "not listed" IS the decline; ask the rows why.
            return ("declined", "recommend", _frost_decline_reasons(g))
        idx = frost if use_frost else backend
        assert idx is not None, f"no backend plan; plans={[g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]}"
        g.select_plan(idx)
        try:
            g.check_support()
        except (NotImplementedError, cudnn.cudnnGraphNotSupportedError) as exc:
            return ("declined", "check_support", str(exc))
        g.build_plans()  # accepted above: a failure from here on is the finding, not a decline
        out = torch.empty_like(self.q)  # packed (T, h, d)
        lse = torch.empty(self.t_q, self.h_q, device="cuda", dtype=torch.float32)  # packed (T, h) stats
        ws = torch.empty(max(g.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
        g.execute(self.pack(out, lse), ws, handle=handle)
        torch.cuda.synchronize()
        return ("ok", out, lse)


def _accept_means_run(case: _Case):
    ref = case.run(use_frost=False)
    assert ref[0] == "ok", f"the cuDNN backend itself declined this FlashInfer graph: {ref}"
    got = case.run(use_frost=True)
    if got[0] == "declined":
        pytest.xfail(f"frost declined at {got[1]}: {got[2][:220]}")
    torch.testing.assert_close(got[1].float(), ref[1].float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(got[2], ref[2], atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("form", ["legacy_offsets", "tokens"])
@pytest.mark.parametrize("d", [128, 192])
def test_ragged_prefill_batch_of_two(form, d):
    """FlashInfer's main prefill path: b > 1 ragged Q/K/V/O with packed stats."""
    _accept_means_run(_Case([68, 87], [400, 512], d=d, tokens_form=form == "tokens"))


@pytest.mark.parametrize("form", ["legacy_offsets", "tokens"])
def test_ragged_prefill_single_sequence(form):
    """b == 1 (the form frost's THD path serves today): O and packed LSE must match the backend."""
    _accept_means_run(_Case([68], [400], tokens_form=form == "tokens"))
