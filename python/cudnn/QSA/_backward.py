# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The launch sequence of the Qwen Sparse Attention (QSA) backward: its plan, compiled kernels and launch.

The kernels come from ``kernels/bprop_sm100.py``. On SM107 the query-major, key-major and dense pair kernels of
``kernels/bprop_sm107.py`` take their place when the driver offers the oversized shared memory carveout.
"""

import hashlib
import math
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

import cutlass.cute as cute
from cuda.bindings import driver as cuda_driver
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from cudnn.api_base import ws_align
from cudnn.frost.compiled_cache import compile_cached, template_key

from .kernels import bprop_sm100
from .kernels.bprop_sm100 import (
    BF16,
    D,
    DENSE_QUERIES_MAX,
    DP_G,
    DP_TT,
    DP_U_MAX,
    DQ_PARTS_DEFAULT,
    F32,
    GQ,
    I32,
    PART_W,
    POS,
    PREP_ROWS,
    QR,
    RANGE_PAIRS,
    SPARSE_FOLD,
    SPARSE_WINDOW_QUERIES,
    _bitmap_host,
    _dpair_host,
    _fill_host,
    _fin_host,
    _pad,
    _part_host,
    _prep_host,
    _rec_w,
    _scan_host,
    _seq_fill_host,
    _seq_scan_host,
    _sparse_key_host,
    _sparse_query_host,
    _window_host,
)

FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
SOURCE_DIGESTS = (bprop_sm100.FROST_SOURCE_DIGEST,)
_SM107_SOURCE_DIGEST = hashlib.sha256(Path(__file__).with_name("kernels").joinpath("bprop_sm107.py").read_bytes()).hexdigest()[:16]

KV_PASS = 32768  # KV pass of the query-major kernel (tokens), so one pass's K/V blocks stay L2 resident
# From this length on a key holds few enough queries that single-query stages are common, and skipping their empty
# K-step pays for the per-stage branch.
KEY_SKIP_MIN_T = 196608
DP_U = 320  # tuned query groups per dense work unit, small enough to backfill the key-major grid's tail evenly

# where the sequence bounds come from
SEQ_SINGLE = 0  # the T rows are one sequence, or documents the caller did not describe
SEQ_CU = 1  # the caller's cu_seqlens
SEQ_ORIGIN = 2  # derived from the per-token sequence origins (tok_base of the bitmap API)


def max_packed_len(hkv: int, gq: int = GQ) -> int:
    # The kernels address the [T, gq * hkv, 256] tensors with 32-bit element offsets.
    return (2**31 - 1) // (gq * hkv * D)


def _dense_cap(max_len: int, single: bool) -> int:
    # the most dense queries a sequence can have under the device-side _dense_queries rule: the one sequence of
    # max_len tokens, or any packed sequence up to max_len tokens (dense throughout just below the crossover)
    if single:
        return max_len if max_len - DENSE_QUERIES_MAX < SPARSE_FOLD else DENSE_QUERIES_MAX
    return min(max_len, DENSE_QUERIES_MAX + SPARSE_FOLD - 1)


class LaunchPlan:
    """Launch geometry and scratch layout for one packed length, KV head count and sequence source.

    ``seq_mode`` SEQ_SINGLE treats the T rows as one sequence, SEQ_CU takes the caller's ``cu_seqlens`` of ``num_seqs``
    sequences, SEQ_ORIGIN derives the sequences on the device from the per-token origins (at most ``num_seqs`` of
    them). The per-sequence dense / sparse split is decided on the device. ``rubin`` runs the query-major, key-major and
    dense pair kernels of ``bprop_sm107`` (SM107 only).
    """

    def __init__(
        self,
        seq_len: int,
        num_sms: int,
        hkv: int,
        seq_mode: int,
        num_seqs: int = 1,
        max_seq_len: Optional[int] = None,
        arch: int = 100,
        gq: int = GQ,
        dq_parts: int = DQ_PARTS_DEFAULT,
        rubin: bool = False,
    ):
        T = int(seq_len)
        assert hkv in (1, 2), f"{hkv} KV heads"
        assert dq_parts in (1, 3), f"{dq_parts} dS parts"
        assert gq == GQ or (gq == 6 and hkv == 1), f"{gq} query heads per KV head with {hkv} KV heads"
        assert 1 <= T <= max_packed_len(hkv, gq), f"sequence length {T} outside [1, {max_packed_len(hkv, gq)}]"
        assert not rubin or arch == 107, f"the SM107 kernels on SM{arch}"
        HKV = hkv
        HQ = gq * hkv
        REC_W = _rec_w(hkv)
        PAD = _pad(hkv)
        self.rubin = bool(rubin)
        self.hkv = hkv
        self.gq = gq
        self.dq_parts = dq_parts
        self.arch = int(arch)
        self.seq_len = T
        self.seq_mode = seq_mode
        self.seq_count = max(int(num_seqs), 1)
        max_len = T if max_seq_len is None else min(int(max_seq_len), T)
        dense_cap = _dense_cap(max_len, seq_mode == SEQ_SINGLE)
        self.has_sparse = dense_cap < max_len
        # one selection bit per key token of the dense prefix, a row per dense query: the first dense_cap rows of one
        # sequence, any row of packed ones
        self.nw = (dense_cap + 31) // 32
        bitmap_rows = min(T, dense_cap) if seq_mode == SEQ_SINGLE else T
        self.nsel = (bitmap_rows * self.nw + 64 + 3) // 4 * 4
        # Key tiles of the dense pair kernel cover the buffer, or only its dense prefix when the rows are one sequence.
        dense_tokens = T if seq_mode != SEQ_SINGLE else min(T, dense_cap)
        self.dense_tiles = (dense_tokens + DP_TT - 1) // DP_TT
        gd = (dense_cap + DP_G - 1) // DP_G
        # A (key tile, unit) pair of CTAs walks its query groups serially behind one key tile load and one dK / dV flush.
        # Without a sparse grid to backfill, the unit hands every pair slot one wave of a sequence's causal work (key
        # tile t sees the groups from its first token up), so the fixed cost is paid once per slot and the waves stay
        # balanced.
        causal_work = HKV * sum(gd - t * (DP_TT // DP_G) for t in range((dense_cap + DP_TT - 1) // DP_TT))
        self.dense_unit = DP_U if self.has_sparse else max(1, min(DP_U_MAX, -(-causal_work // (num_sms // 2))))
        self.num_sms = num_sms
        # the groups a key tile can reach: the dense queries of the sequences holding its tokens
        reach = (DP_TT + dense_cap + DP_G - 1) // DP_G
        self.dense_parts = (reach + self.dense_unit - 1) // self.dense_unit
        self.pass_tokens = KV_PASS
        self.num_passes = (T + self.pass_tokens - 1) // self.pass_tokens
        assert self.num_passes < POS, f"sequence length {T} needs {self.num_passes} KV passes, the offset rows hold {POS - 1}"
        # Sparse queries: exactly T - dense_cap of a single sequence, up to T of packed ones (counted on the device).
        nsq = 0 if not self.has_sparse else (T - dense_cap if seq_mode == SEQ_SINGLE else T)
        self.window_capacity = min(nsq, SPARSE_WINDOW_QUERIES)
        self.query_windows = tuple((lo, min(lo + SPARSE_WINDOW_QUERIES, nsq)) for lo in range(0, nsq, SPARSE_WINDOW_QUERIES))
        # the query-major kernel's item is a query, or a pair of queries with one KV head
        self.sparse_ctas = min(num_sms, (self.window_capacity + 1) // 2 if hkv == 1 else self.window_capacity)
        # inverted index of the sparse queries: ranges of QR queries, one key per (range, block), one slot per pair
        self.num_ranges = (self.window_capacity + QR - 1) // QR
        assert self.num_ranges <= 2, "the key-major kernel walks the active keys of at most two ranges"
        # a range's keys are the first tokens of the buffer, padded to the index scan's 8 keys per thread
        self.keys_per_range = (T + 7) // 8 * 8
        self.num_keys = self.num_ranges * self.keys_per_range
        self.num_pairs = self.num_ranges * RANGE_PAIRS
        self.key_name = f"key_skip{1 if T >= KEY_SKIP_MIN_T else 0}"
        self.key_ctas = num_sms
        nz = T * HKV * D if self.has_sparse else 0
        dense_dkv_size = T * HKV * D + 64
        keys = (self.num_keys + 31) // 16 * 16
        prep_vecs = max(T * HQ // PREP_ROWS, nz // 8, (self.nsel + 4) // 4, keys // 4, T * HQ * D // 8, dense_dkv_size // 8)
        self.prep_blocks = min(num_sms * 8, (prep_vecs + 255) // 256)
        self.fin_blocks = min(num_sms * 8, (T * HKV * D // 8 + 255) // 256)
        sp = self.has_sparse
        cu_len = {SEQ_SINGLE: 4, SEQ_CU: 0, SEQ_ORIGIN: (T + 1 + 3) // 4 * 4}[seq_mode]
        T4 = (T + 3) // 4 * 4
        # Without the sparse path its buffers shrink to nothing, except the dK/dV accumulators, which keep one
        # never-read PAD chunk that both kernels alias.
        self.chunks = [
            ("delta", (T, HQ), torch.float32),
            ("sparse_dk_acc", (nz + PAD,), torch.float32),
            ("sparse_dv_acc", (nz + PAD if sp else 0,), torch.float32),
            ("dense_block_bitmap", (self.nsel + 4,), torch.int32),
            ("dq_acc", (T * HQ * D,), torch.float32),
            ("dense_dk_acc", (dense_dkv_size,), torch.float32),
            ("dense_dv_acc", (dense_dkv_size,), torch.float32),
            ("seq_lens", (cu_len,), torch.int32),
            ("seq_sparse_base", ((self.seq_count + 3) // 4 * 4,), torch.int32),
            ("sparse_count", (4,), torch.int32),
            ("window_count", (4 if sp else 0,), torch.int32),
            ("seq_start", (T4,), torch.int32),
            ("dense_hi", (T4,), torch.int32),
            ("nseq", (4,), torch.int32),
            ("dense_unit", (4,), torch.int32),
            ("first_group", (((T + DP_TT - 1) // DP_TT + 3) // 4 * 4,), torch.int32),  # seq_fill resets every tile of the buffer
            ("active_count", (4 if sp else 0,), torch.int32),
            ("sparse_queries", ((nsq + 3) // 4 * 4 if sp else 0,), torch.int32),
            ("sparse_blocks", (self.window_capacity * PART_W + 16 if sp else 0,), torch.int32),
            ("sparse_pass_offsets", (self.window_capacity * POS + 16 if sp else 0,), torch.int32),
            ("sparse_block_slots", (self.window_capacity * PART_W + 16 if sp else 0,), torch.int32),
            ("key_pair_counts", (keys if sp else 0,), torch.int32),
            ("key_slot_offsets", (keys if sp else 0,), torch.int32),
            ("key_slot_cursors", (keys if sp else 0,), torch.int32),
            ("active_keys", (keys if sp else 0,), torch.int32),
            ("slot_queries", ((self.num_pairs + 31) // 16 * 16 if sp else 0,), torch.int32),
            ("pair_records", (self.num_pairs * REC_W + 16 if sp else 0,), torch.int32),
        ]

    def workspace_bytes(self) -> int:
        return sum(ws_align(math.prod(shape) * dtype.itemsize) for _, shape, dtype in self.chunks)

    def carve(self, carver) -> dict:
        """Scratch views from a ``cudnn.api_base.WorkspaceCarver`` over the caller's workspace."""
        scratch = {name: carver.take(math.prod(shape), dtype).view(shape) for name, shape, dtype in self.chunks if math.prod(shape)}
        if not self.has_sparse:
            scratch["sparse_dv_acc"] = scratch["sparse_dk_acc"]
            scratch["key_pair_counts"] = scratch["dense_block_bitmap"][self.nsel :]
            scratch["sparse_queries"] = scratch["seq_sparse_base"]
        return scratch


_PLANS: dict = {}
_PLANS_LOCK = threading.Lock()
_PLANS_MAX = 64  # packed training changes the token and block counts every step, so the least recent plans go


def plan_for(
    seq_len: int,
    device: torch.device,
    hkv: int,
    seq_mode: int,
    num_seqs: int = 1,
    gq: int = GQ,
    dq_parts: int = DQ_PARTS_DEFAULT,
    max_seq_len: Optional[int] = None,
) -> LaunchPlan:
    """The plan of a geometry, kept for the most recent geometries of the process."""
    key = (int(seq_len), hkv, seq_mode, int(num_seqs), device.index, int(gq), None if max_seq_len is None else int(max_seq_len), int(dq_parts))
    with _PLANS_LOCK:
        plan = _PLANS.pop(key, None)
        if plan is None:
            cc = torch.cuda.get_device_capability(device)
            arch = cc[0] * 10 + cc[1]
            rubin = False
            if arch == 107:
                from .kernels import bprop_sm107

                # without the oversized shared memory carveout the SM100 kernels serve SM107
                rubin = bprop_sm107.supported(device.index if device.index is not None else torch.cuda.current_device())
            plan = LaunchPlan(
                seq_len,
                torch.cuda.get_device_properties(device).multi_processor_count,
                hkv,
                seq_mode,
                num_seqs,
                max_seq_len,
                arch=arch,
                gq=gq,
                dq_parts=dq_parts,
                rubin=rubin,
            )
            if len(_PLANS) >= _PLANS_MAX:
                _PLANS.pop(next(iter(_PLANS)))
        _PLANS[key] = plan
        return plan


def _fake(dtype, shape, sym, align=16):
    full = tuple(sym if s is None else s for s in shape)
    order = tuple(range(len(shape) - 1, -1, -1))
    return make_fake_compact_tensor(dtype, full, stride_order=order, assumed_align=align)


@lru_cache(maxsize=None)
def _compile_args(hkv: int, gq: int, parts: int, rubin: bool) -> dict:
    """Storage-free kernel signatures, reused across plans. The shared cache owns the CUDA objects."""
    HQ = gq * hkv
    query_host, key_host, dense_host = _sparse_query_host, _sparse_key_host, _dpair_host
    if rubin:
        from .kernels import bprop_sm107

        query_host, key_host, dense_host = bprop_sm107._sparse_query_host, bprop_sm107._sparse_key_host, bprop_sm107._dpair_host
    sym_t = cute.sym_int64()
    f_q = _fake(BF16, (None, HQ, D), sym_t)
    f_kv = _fake(BF16, (None, hkv, D), sym_t)
    f_out = _fake(F32, (None, HQ, D), sym_t)
    f_lse = _fake(F32, (HQ, None), sym_t, align=4)
    f_delta = _fake(F32, (None, HQ), sym_t)
    f_h = _fake(F32, (None,), cute.sym_int64(divisibility=8))
    f_dq_acc = _fake(F32, (None,), cute.sym_int64(divisibility=8))
    f_dense_dkv = _fake(F32, (None,), cute.sym_int64(divisibility=8))

    def f_ids():
        # the compacted block keys [T, capacity]
        return make_fake_compact_tensor(I32, (sym_t, cute.sym_int64()), stride_order=(1, 0), assumed_align=4)

    def f_i():
        # scratch int32 vectors, padded to 4 entries
        return _fake(I32, (None,), cute.sym_int64(divisibility=4))

    def f_c():
        # caller int32 vectors of any length
        return _fake(I32, (None,), cute.sym_int64(), align=4)

    stream = make_fake_stream(use_tvm_ffi_env_stream=False)
    dense_fakes = (
        f_kv,
        f_kv,
        f_q,
        f_q,
        f_lse,
        f_delta,
        f_i(),
        f_dq_acc,
        f_dense_dkv,
        f_dense_dkv,
        f_i(),
        f_i(),
        f_i(),
        f_i(),
        f_i(),
        I32(1),
        I32(1),
        I32(1),
        I32(1),
        I32(0),
        F32(1),
    )
    fin_fakes = (f_h, f_h, f_dense_dkv, f_dense_dkv, f_dq_acc, f_kv, f_kv, f_q, I32(1))
    builders = {
        "seq_scan": lambda: (_seq_scan_host, f_c(), f_c(), f_i(), f_i(), f_i(), f_i(), I32(1), I32(1), I32(0), I32(0), I32(1), I32(1), stream),
        "seq_fill": lambda: (_seq_fill_host, f_c(), f_i(), f_i(), f_i(), f_i(), f_i(), f_i(), I32(1), stream),
        "prep": lambda: (
            _prep_host,
            f_out,
            f_q,
            f_delta,
            f_h,
            f_h,
            f_i(),
            f_i(),
            f_dq_acc,
            f_dense_dkv,
            f_dense_dkv,
            f_i(),
            I32(1),
            hkv,
            gq,
            stream,
        ),
        "bitmap": lambda: (_bitmap_host, f_ids(), f_c(), f_c(), f_c(), f_c(), f_i(), f_i(), f_i(), f_i(), f_dq_acc, I32(1), I32(8), stream),
        "window": lambda: (_window_host, f_i(), f_i(), f_i(), I32(0), I32(1), stream),
        "part": lambda: (_part_host, f_ids(), f_c(), f_c(), f_dq_acc, f_i(), f_i(), f_i(), f_i(), f_i(), I32(1), I32(1), I32(1), I32(1), hkv, gq, stream),
        "scan": lambda: (_scan_host, f_i(), f_i(), f_i(), f_i(), f_i(), f_i(), I32(1), I32(1), stream),
        "fill": lambda: (_fill_host, f_i(), f_i(), f_i(), f_i(), f_i(), f_i(), f_i(), I32(1), I32(1), I32(1), stream),
        "query": lambda: (
            query_host,
            f_q,
            f_kv,
            f_kv,
            f_q,
            f_lse,
            f_delta,
            f_dq_acc,
            f_i(),
            f_i(),
            f_i(),
            f_i(),
            f_i(),
            f_c(),
            f_c(),
            f_i(),
            I32(1),
            I32(1),
            F32(1),
            hkv,
            gq,
            parts,
            stream,
        ),
        "dense": lambda: (dense_host, *dense_fakes, 0, hkv, gq, parts, stream),
        "dense_pdl": lambda: (dense_host, *dense_fakes, 1, hkv, gq, parts, stream),
        "fin": lambda: (_fin_host, *fin_fakes, 0, hkv, gq, stream),
        "fin_sparse": lambda: (_fin_host, *fin_fakes, 1, hkv, gq, stream),
    }
    for skip in (0, 1):
        builders[f"key_skip{skip}"] = lambda skip=skip: (
            key_host,
            f_q,
            f_q,
            f_i(),
            f_i(),
            f_i(),
            f_i(),
            f_h,
            f_h,
            f_i(),
            f_i(),
            I32(1),
            I32(1),
            skip,
            hkv,
            gq,
            stream,
        )
    return {name: build() for name, build in builders.items()}


def compile_kernels(plan: LaunchPlan) -> dict:
    """Compile the launch sequence of the plan's path through the device-aware compiled cache.

    Every extent is symbolic, so one set of artifacts serves all sequence lengths of a path on one architecture.
    """
    names = ["seq_scan", "seq_fill", "prep", "bitmap"]
    if plan.has_sparse:
        names += ["window", "part", "scan", "fill", "query", plan.key_name, "dense_pdl", "fin_sparse"]
    else:
        names += ["dense", "fin"]
    key = template_key(
        globals(),
        {
            "hkv": plan.hkv,
            "gq": plan.gq,
            "parts": plan.dq_parts,
            "arch": plan.arch,
            "sources": SOURCE_DIGESTS,
            "sm107": _SM107_SOURCE_DIGEST if plan.rubin else None,
        },
        "compile_kernels",
    )
    args = _compile_args(plan.hkv, plan.gq, plan.dq_parts, plan.rubin)
    options = f"--enable-tvm-ffi --gpu-arch=sm_{plan.arch}a"
    return {name: compile_cached(*args[name], cache_key=key, symbol=f"qsa_backward_{name}", options=options) for name in names}


def launch(plan, fns, q, k, v, out, dout, lse, ids, counts, first, lo, hi, dq, dk, dv, scratch, scale, stream, cu_seqlens=None, tok_base=None):
    """Enqueue the launch sequence on ``stream``. Inputs, outputs and scratch are caller-owned.

    ``ids`` [T, capacity] holds every query's block keys, ``counts`` [T] their number, ``first`` [NB] the first token
    of a block key, ``lo`` / ``hi`` [T] the inclusive key bounds. ``cu_seqlens`` (SEQ_CU) or ``tok_base`` (SEQ_ORIGIN)
    describe the packed sequences.
    """
    s = scratch
    T = plan.seq_len
    HKV = plan.hkv
    scale = F32(scale)
    dense_done_count = s["dense_block_bitmap"][plan.nsel :]
    cu = cu_seqlens if plan.seq_mode == SEQ_CU else s["seq_lens"]
    origins = tok_base if plan.seq_mode == SEQ_ORIGIN else s["nseq"]
    num_seqs = plan.seq_count if plan.seq_mode == SEQ_CU else 1
    fns["seq_scan"](
        cu,
        origins,
        s["nseq"],
        s["dense_unit"],
        s["seq_sparse_base"],
        s["sparse_count"],
        T,
        num_seqs,
        plan.seq_mode,
        plan.dense_unit if plan.has_sparse else 0,
        HKV,
        plan.num_sms,
        stream,
    )
    fns["seq_fill"](cu, s["nseq"], s["seq_sparse_base"], s["seq_start"], s["dense_hi"], s["sparse_queries"], s["first_group"], T, stream)
    fns["prep"](
        out,
        dout,
        s["delta"],
        s["sparse_dk_acc"],
        s["sparse_dv_acc"],
        s["dense_block_bitmap"],
        s["key_pair_counts"],
        s["dq_acc"],
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        s["dense_hi"],
        plan.prep_blocks,
        stream,
    )
    fns["bitmap"](
        ids, counts, first, lo, hi, s["seq_start"], s["dense_hi"], s["dense_block_bitmap"], s["first_group"], s["dq_acc"], plan.nw, plan.gq * HKV * D, stream
    )
    dense_args = (
        k,
        v,
        q,
        dout,
        lse,
        s["delta"],
        s["dense_block_bitmap"],
        s["dq_acc"],
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        dense_done_count,
        s["seq_start"],
        s["dense_hi"],
        s["dense_unit"],
        s["first_group"],
        plan.nw,
        plan.dense_tiles,
        plan.dense_unit,
        plan.dense_parts,
        0 if plan.has_sparse else 1,
        scale,
    )
    fin_args = (
        s["sparse_dk_acc"],
        s["sparse_dv_acc"],
        s["dense_dk_acc"],
        s["dense_dv_acc"],
        s["dq_acc"],
        dk,
        dv,
        dq,
        plan.fin_blocks,
        stream,
    )
    if plan.has_sparse:
        window_count = s["window_count"]
        KR = plan.keys_per_range
        for qlo, qhi in plan.query_windows:
            queries = s["sparse_queries"][qlo : (qhi + 3) // 4 * 4]
            fns["window"](s["sparse_count"], window_count, s["key_pair_counts"], qlo, qhi - qlo, stream)
            fns["part"](
                ids,
                counts,
                first,
                s["dq_acc"],
                queries,
                window_count,
                s["sparse_blocks"],
                s["sparse_pass_offsets"],
                s["key_pair_counts"],
                qhi - qlo,
                plan.pass_tokens,
                plan.num_passes,
                KR,
                stream,
            )
            fns["scan"](
                window_count,
                s["key_pair_counts"],
                s["key_slot_offsets"],
                s["key_slot_cursors"],
                s["active_keys"],
                s["active_count"],
                KR,
                plan.num_ranges,
                stream,
            )
            fns["fill"](
                queries,
                window_count,
                s["sparse_blocks"],
                s["sparse_pass_offsets"],
                s["key_slot_cursors"],
                s["sparse_block_slots"],
                s["slot_queries"],
                qhi - qlo,
                plan.num_passes,
                KR,
                stream,
            )
            fns["query"](
                q,
                k,
                v,
                dout,
                lse,
                s["delta"],
                s["dq_acc"],
                s["sparse_blocks"],
                s["sparse_pass_offsets"],
                s["sparse_block_slots"],
                s["pair_records"],
                queries,
                lo,
                hi,
                window_count,
                plan.sparse_ctas,
                plan.num_passes,
                scale,
                stream,
            )
            # Consume the window before its records and inverted index are reused.
            fns[plan.key_name](
                q,
                dout,
                s["pair_records"],
                s["key_slot_offsets"],
                s["key_pair_counts"],
                s["slot_queries"],
                s["sparse_dk_acc"],
                s["sparse_dv_acc"],
                s["active_keys"],
                s["active_count"],
                plan.key_ctas,
                KR,
                stream,
            )
        # The dense grid backfills only the final key-major grid, so it launches once, after the windows.
        fns["dense_pdl"](*dense_args, stream)
        fns["fin_sparse"](*fin_args)
    else:
        fns["dense"](*dense_args, stream)
        fns["fin"](*fin_args)


def dsl_cuda_major():
    # the toolkit flavor of the DSL compiler, which torch.version.cuda and CUDA_HOME do not identify
    from cutlass.base_dsl.version_info import CUDA_VERSION

    return CUDA_VERSION.major


def backward(grad_out, q, k, v, out_fp32, lse, sel_blocks, lo, hi, tok_base, block_first, block_last, scale, arch, dq_parts=DQ_PARTS_DEFAULT):
    """``cudnn.ops.qsa_backward`` on SM100 / SM103 / SM107: the compacted selection through the launch sequence above."""
    from cudnn.api_base import TupleDict, WorkspaceCarver
    from . import _metadata

    total, hkv = q.shape[0], k.shape[1]
    if hkv == 1 and dsl_cuda_major() < 13:
        # the one-KV-head key-major kernel crashes the CUDA 12 DSL compiler
        raise NotImplementedError(
            "QSA backward with one KV head requires the CUDA 13 build of CuTe DSL: install 'nvidia-cutlass-dsl[cu13]>=4.7.0' with a CUDA 13 compatible driver"
        )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    ids, counts = _metadata.compact(sel_blocks, lo, hi, block_first, block_last, arch, stream)
    # the sequences come from tok_base on the device. There are at most as many as blocks.
    plan = plan_for(total, q.device, hkv, SEQ_ORIGIN, num_seqs=min(total, block_first.shape[0]), gq=q.shape[1] // hkv, dq_parts=dq_parts)
    fns = compile_kernels(plan)
    workspace = torch.empty(plan.workspace_bytes(), dtype=torch.uint8, device=q.device)
    scratch = plan.carve(WorkspaceCarver(workspace, plan.workspace_bytes(), "qsa_backward"))
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    launch(plan, fns, q, k, v, out_fp32, grad_out, lse, ids, counts, block_first, lo, hi, dq, dk, dv, scratch, float(scale), stream, tok_base=tok_base)
    return TupleDict(dq=dq, dk=dk, dv=dv)
