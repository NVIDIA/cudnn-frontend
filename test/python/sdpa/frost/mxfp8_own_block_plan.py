# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared GPU plan for the ``sdpa_mxfp8(causal_exclude_own_block=32)`` suites (no ``test_`` prefix, so a
directory sweep collects each suite once)."""

import math

import torch

import cudnn
from cudnn.sdpa.fwd import engines

from frost_test_utils import _is_plan_for, make_dense_stats, select_engine
from mxfp8_test_reference import _quantize

OWN = 32
_E8M0 = cudnn.data_type.FP8_E8M0
_F8_128X4 = cudnn.tensor_reordering.F8_128x4
_ROLES = ("q", "k", "v", "dq", "dk", "dv")
_PAYLOADS = ("q", "k", "v")


_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_ITYPE = {"e4m3": cudnn.data_type.FP8_E4M3, "e5m2": cudnn.data_type.FP8_E5M2}
_PIN = engines.engine_name(arch="sm100", mxfp8=True)


class _Plan:
    """One built, pinned plan for a dense BSHD (B, H_q / H_kv, S, 128) sdpa_mxfp8 graph with BF16 O and Stats.

    ``pack_gqa`` pins the packed (True) or unpacked (False) leg; ``split`` pins the smallest KV split above 1
    the heuristics offer for the shape.  Both default to the heuristics' best guess; ``self.knobs`` records
    the pinned plan's knobs."""

    def __init__(self, b, h_q, h_kv, s, in_key, *, own_block=OWN, bottom_right=False, pack_gqa=None, split=False):
        self.shape = (b, h_q, h_kv, s)
        self.in_key = in_key
        self.knobs = None
        d = 128
        g = cudnn.pygraph(io_data_type=_ITYPE[in_key], intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

        def qkv(h, name):
            return g.tensor(dim=[b, h, s, d], stride=[s * h * d, d, h * d, 1], data_type=_ITYPE[in_key], name=name)

        def sf(dims):
            return g.tensor(dim=list(dims), stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1], data_type=_E8M0, reordering_type=_F8_128X4)

        self.sf_dims = {}
        rows = lambda n: math.ceil(n / 128) * 128  # noqa: E731 -- the F8_128x4 atom pads S to whole 128-row tiles
        self.q, self.k, self.v = qkv(h_q, "q"), qkv(h_kv, "k"), qkv(h_kv, "v")
        self.dq = sf((b, h_q, rows(s), 4))
        self.dk = sf((b, h_kv, rows(s), 4))
        self.dv = sf((b, h_kv, math.ceil(s / 32 / 4) * 4, rows(d)))
        mask = dict(use_causal_mask_bottom_right=True) if bottom_right else dict(use_causal_mask=True)
        kw = dict(causal_exclude_own_block=own_block) if own_block else {}
        self.o, self.stats, _ = g.sdpa_mxfp8(
            q=self.q,
            k=self.k,
            v=self.v,
            descale_q=self.dq,
            descale_k=self.dk,
            descale_v=self.dv,
            attn_scale=1.0 / math.sqrt(d),
            generate_stats=True,
            **mask,
            **kw,
        )
        self.o.set_output(True).set_dim([b, h_q, s, d]).set_stride([s * h_q * d, d, h_q * d, 1]).set_data_type(cudnn.data_type.BFLOAT16)
        self.stats.set_output(True).set_dim([b, h_q, s, 1]).set_stride([h_q * s, s, 1, 1]).set_data_type(cudnn.data_type.FLOAT)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        split_kv = None
        if split:
            offered = {
                g.plans[i].knobs.split_kv
                for i in range(len(g.plans))
                if _is_plan_for(g.get_plan_name_at_index(i), _PIN) and (pack_gqa is None or g.plans[i].knobs.pack_gqa == pack_gqa)
            }
            splits = sorted(n for n in offered if (n or 1) > 1)
            assert splits, f"the heuristics offer no split-KV plan for {self.shape}"
            split_kv = splits[0]
        self.knobs = select_engine(g, _PIN, pack_gqa=pack_gqa, split_kv=split_kv).knobs
        g.check_support()
        g.build_plans()
        self.graph = g

    def quantize(self, qf, kf, vf):
        """MXFP8-quantize fp32 BHSD inputs once: (payloads in BSHD storage, SF, fp32 dequantized operands)."""
        b, h_q, h_kv, s = self.shape
        fp8 = _FP8[self.in_key]
        q8, sfq, dqq, (sqp, dsc) = _quantize(qf, b, h_q, s, 128, fp8, columnwise=False)
        k8, sfk, dqk, (skp, _) = _quantize(kf, b, h_kv, s, 128, fp8, columnwise=False)
        v8, sfv, dqv, (ssc, dvp) = _quantize(vf, b, h_kv, s, 128, fp8, columnwise=True)
        bshd = lambda x8: x8.permute(0, 2, 1, 3).contiguous().transpose(1, 2)  # noqa: E731
        inputs = {
            self.q: bshd(q8),
            self.k: bshd(k8),
            self.v: bshd(v8),
            self.dq: sfq.view(torch.uint8).reshape(b, h_q, sqp, dsc),
            self.dk: sfk.view(torch.uint8).reshape(b, h_kv, skp, dsc),
            self.dv: sfv.view(torch.uint8).reshape(b, h_kv, ssc, dvp),
        }
        return inputs, (q8.float() * dqq, k8.float() * dqk, v8.float() * dqv)

    def execute(self, inputs):
        """Run on FRESH output buffers (NaN-poisoned, so an unwritten element cannot pass a comparison)."""
        b, h_q, _, s = self.shape
        o = torch.full((b, s, h_q, 128), float("nan"), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        lse = make_dense_stats(b, h_q, s, "contiguous").fill_(float("nan"))
        workspace = torch.empty(max(self.graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
        self.graph.execute({**inputs, self.o: o, self.stats: lse}, workspace)
        torch.cuda.synchronize()
        return o, lse.squeeze(-1)

    def roles(self, inputs):
        """This plan's input dict, keyed by role name instead of graph tensor."""
        return {r: inputs[getattr(self, r)] for r in _ROLES}

    def pack(self, sequences):
        """Assemble this plan's inputs from per-sequence role dicts (each from a B = 1 plan), reusing the
        quantized payload and scale bytes as they are -- no requantization.  Payloads keep BSHD storage;
        the F8_128x4 scale tensors are batch-major, so the batch concatenates along dim 0."""
        assert len(sequences) == self.shape[0], (len(sequences), self.shape)
        out = {}
        for r in _ROLES:
            parts = [seq[r] for seq in sequences]
            if r in _PAYLOADS:
                out[getattr(self, r)] = torch.cat([x.transpose(1, 2) for x in parts], 0).transpose(1, 2)
            else:
                out[getattr(self, r)] = torch.cat(parts, 0)
        return out


def _inputs(b, h_q, h_kv, s, device="cuda"):
    return (
        torch.randn(b, h_q, s, 128, device=device) * 0.5,
        torch.randn(b, h_kv, s, 128, device=device) * 0.5,
        torch.randn(b, h_kv, s, 128, device=device) * 0.5,
    )
