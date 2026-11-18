# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-plan public-S forward bytes; backward and autotuned plan changes are outside this contract."""

import pytest
import torch

import cudnn
from cudnn.engines.manifest import MANIFEST
from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
from determinism_utils import assert_bitwise, assert_golden, assert_replays, snapshot
from test_sdpa_prepared_fp8 import _case, _check, _cuda_graph


def cases(arch):
    if arch == "sm100":
        bodies = [
            ("d128_fp8_multi_q", 128, 128, 1, 128),
            ("d128_fp8_multi_q", 128, 128, 2, 128),
            ("d128_fp8_dual_kv", 128, 128, 1, 128),
            ("d128_fp8_swap_ab", 128, 128, 1, 16),
            ("d192_fp8_multi_q", 192, 128, 1, 128),
            ("d192_fp8_multi_q", 192, 128, 2, 128),
            ("d256_fp8_single_q", 256, 256, 1, 128),
            ("d512_fp8_role_split", 512, 512, 4, 128),
        ]
    elif arch == "sm107":
        bodies = [
            ("d128_fp8_multi_q", 128, 128, 2, 128),
            ("d192_fp8_multi_q", 192, 128, 2, 128),
            ("d256_fp8_single_q", 256, 256, 1, 128),
            ("d512_fp8_role_split", 512, 512, 4, 128),
            ("d128_fp8_swap_ab", 128, 128, 1, 16),
        ]
    else:
        assert arch == "sm120"
        bodies = [("fp8_general", d, d, 1, 128) for d in (64, 128, 256)] + [("d512_fp8_paired", 512, 512, 1, 64)]
    result = []
    for owner, d, dv, cga, tile in bodies:
        for half in ((False, True) if arch == "sm107" and "swap_ab" not in owner else (False,)):
            # Unsplit nonpaged D128 FP8 THD is CGA2-only on SM100/SM103.
            dense_only = "swap_ab" in owner or "dual_kv" in owner or (arch == "sm100" and d == 128 and cga == 1)
            for thd in ((False,) if dense_only else (False, True)):
                label = f"{arch}-{owner}-d{d}-cga{cga}-{'f16' if half else 'f32'}-{'thd' if thd else 'dense'}"
                result.append(pytest.param((arch, owner, d, dv, cga, tile, half, thd), id=label))
    return result


def check_public_s_batch_and_golden(case, source):
    arch, owner, d, dv, cga, tile, half, thd = case
    sq, skv = (3 if "swap_ab" in owner else 129), 257
    name = f"sdpa_fwd_{arch}_{owner}"
    family = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    knobs = SdpaFwdKnobs(tile_m=tile, tile_n=64 if arch == "sm120" and d == 512 else 128, cga=cga, pack_gqa=False, split_kv=1, sched_policy=0)
    record = (family.offered_ids()[name], knobs.to_public())
    cache = {}

    def build(b):
        if b not in cache:
            kwargs = {"softmax_precision": cudnn.data_type.HALF if half else cudnn.data_type.FLOAT} if arch == "sm107" else {}
            cache[b] = _case(
                arch=arch, b=b, sq=sq, skv=skv, d=d, dv=dv, thd=thd, record=record, sdpa_kwargs=kwargs, s_scale_source=source, s_scale_pair=(256.0, 1 / 256)
            )
            assert cache[b][0].selected_engine.name == name
            assert cache[b][0].plans[cache[b][0]._plan_index].knobs.split_kv == 1
        return cache[b]

    def batch_view(tensor, b, role):
        if not thd:
            return tensor
        seq = skv if role in ("k", "v") else sq
        return tensor.reshape(b, seq, *tensor.shape[1:]).transpose(1, 2)

    pool = build(3)[3]
    # Quantize once. Preserve those exact payloads across every batch composition.
    samples = [{role: batch_view(pool[role], 3, role)[i].clone() for role in ("q", "k", "v")} for i in range(3)]
    samples_before = [snapshot(sample, allowed_inf={}) for sample in samples]
    output_names = ("o", "lse", "amax_o")

    def run(order, *, capture=False, fresh=False):
        b = len(order)
        graph, vp, ws, bufs, tensors = build(b)
        if fresh:
            for role in tuple(bufs):
                if role in tensors:
                    bufs[role] = bufs[role].clone()
                    vp[tensors[role]] = bufs[role]
        for slot, request in enumerate(order):
            for role in ("q", "k", "v"):
                batch_view(bufs[role], b, role)[slot].copy_(samples[request][role])
        inputs = {role: value for role, value in bufs.items() if role not in output_names}
        before = snapshot(inputs, allowed_inf={})
        for role in output_names:
            bufs[role].fill_(float("nan"))
        graph.execute(vp, ws)
        if capture:
            with _cuda_graph() as replay:
                with torch.cuda.graph(replay):
                    graph.execute(vp, ws)
                for role in output_names:
                    bufs[role].fill_(float("nan"))
                replay.replay()
        _check(bufs, thd=thd, b=b, sq=sq, skv=skv)
        assert_bitwise(inputs, before, "input mutation")
        return snapshot({role: bufs[role] for role in output_names}, allowed_inf={})

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        singles = [run((i,), capture=True, fresh=True) for i in range(3)]
        for order in ((0, 1, 2), (2, 0, 1), (0, 0, 2)):
            result = run(order, capture=True, fresh=True)
            for slot, request in enumerate(order):
                actual = {role: batch_view(result[role], len(order), role)[slot] for role in ("o", "lse")}
                expected = {role: batch_view(singles[request][role], 1, role)[0] for role in ("o", "lse")}
                assert_bitwise(actual, expected, f"batch {order}, slot {slot}")
            expected_amax = torch.stack([singles[i]["amax_o"] for i in order]).amax(0)
            assert_bitwise({"amax_o": result["amax_o"]}, {"amax_o": expected_amax}, f"batch Amax {order}")
        assert_replays(lambda: run((2, 0, 1), fresh=True), allowed_inf={})
        for actual, expected in zip(samples, samples_before):
            assert_bitwise(actual, expected, "sample payload mutation")
        # Changing only a neighbor must not change request zero.
        samples[1] = {role: (-value.float()).to(value.dtype) for role, value in samples[1].items()}
        changed = run((0, 1, 2), capture=True)
        assert_bitwise(
            {role: batch_view(changed[role], 3, role)[0] for role in ("o", "lse")},
            {role: batch_view(singles[0][role], 1, role)[0] for role in ("o", "lse")},
            "changed neighbor",
        )
        golden_inputs = samples_before[0] | {role: pool[role] for role in ("descale_q", "descale_k", "descale_v", "scale_o")}
        golden_inputs["public_s_pair"] = torch.tensor([256.0, 1 / 256], dtype=torch.float32)
        assert_golden(
            f"public_s_{arch}_{owner}_d{d}_cga{cga}_{'f16' if half else 'f32'}_{'thd' if thd else 'dense'}_{source}", golden_inputs, singles[0], allowed_inf={}
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
