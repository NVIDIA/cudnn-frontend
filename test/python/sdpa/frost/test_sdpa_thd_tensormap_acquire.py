# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Immutable K/V maps must be acquired again after every THD setup launch."""

import pytest
import torch

from frost_test_utils import _SM, requires_dsl
from test_utils import torch_fork_set_rng

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(not 100 <= _SM <= 107, reason="SM100/SM107 templates")]


@pytest.mark.parametrize(
    "d,dv,cga",
    [(128, 128, 2), (256, 256, 2), (192, 128, 1), (192, 128, 2), (512, 512, 2)],
    ids=["d128_cga2", "d256_cga2", "d192_d128_cga1", "d192_d128_cga2", "d512_cga2"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@torch_fork_set_rng(seed=1300)
def test_thd_tensormaps_rebind_and_replay(d, dv, cga, dtype):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    if _SM == 107 and cga == 1:
        pytest.skip("D192 half single-CTA exceeds the current SM107 shared-memory carveout")

    q_lengths = [193, 97, 33]
    b, h, hk, sq, sk = 3, 8, 2, 193, 257
    tq = sum(q_lengths)

    def tensor_view(raw, heads, length, width):
        return raw.as_strided((b, heads, length, width), (length * heads * width, width, heads * width, 1))

    def inputs(k_lengths):
        q = torch.full((b * sq, h, d), float("nan"), device="cuda", dtype=dtype)
        k = torch.full((b * sk, hk, d), float("nan"), device="cuda", dtype=dtype)
        v = torch.full((b * sk, hk, dv), float("nan"), device="cuda", dtype=dtype)
        q[:tq].normal_()
        k[: sum(k_lengths)].normal_()
        v[: sum(k_lengths)].normal_()
        out = torch.empty((b * sq, h, dv), device="cuda", dtype=dtype)
        stats = torch.empty((b * sq, h), device="cuda", dtype=torch.float32)
        return dict(
            q=q,
            k=k,
            v=v,
            out=out,
            stats=stats,
            q_lens=torch.tensor(q_lengths, dtype=torch.int32, device="cuda"),
            k_lens=torch.tensor(k_lengths, dtype=torch.int32, device="cuda"),
        )

    def views(x):
        return (
            tensor_view(x["q"], h, sq, d),
            tensor_view(x["k"], hk, sk, d),
            tensor_view(x["v"], hk, sk, dv),
            tensor_view(x["out"], h, sq, dv),
            x["stats"].as_strided((b, h, sq), (sq * h, 1, h)),
        )

    first_lengths = [257, 161, 65]
    first = inputs(first_lengths)
    q, k, v, out, stats = views(first)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_lse=stats,
        thd=True,
        is_causal=True,
        causal_bottom_right=True,
        seq_kv_lens_present=True,
        cga=cga,
        sched_policy=1,
        split_kv=1,
        pack_gqa=False,
    )
    assert api.check_support()
    api.compile()
    assert api._k_mod.CFG.THD_VARLEN and not getattr(api._k_mod, "PAGED_KV", False)
    assert api._k_mod.CFG.CTA_MMA == cga
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")

    def run(x):
        api.execute(*views(x)[:4], x["stats"], seq_q_lens=x["q_lens"], seq_kv_lens=x["k_lens"], workspace=workspace)

    def poison(x):
        x["out"].fill_(float("nan"))
        x["stats"].fill_(float("nan"))

    def check(x, k_lengths):
        qb = kb = 0
        for nq, nk in zip(q_lengths, k_lengths):
            q = x["q"][qb : qb + nq].double().transpose(0, 1)
            k = x["k"][kb : kb + nk].double().transpose(0, 1).repeat_interleave(h // hk, dim=0)
            v = x["v"][kb : kb + nk].double().transpose(0, 1).repeat_interleave(h // hk, dim=0)
            scores = q @ k.transpose(-1, -2) * d**-0.5
            mask = torch.arange(nk, device="cuda")[None, :] > torch.arange(nq, device="cuda")[:, None] + nk - nq
            scores.masked_fill_(mask, -float("inf"))
            prob = scores.softmax(-1)
            expected = prob @ v
            bound = torch.finfo(dtype).eps / 2 * (prob @ v.abs() + expected.abs()) + 2e-5
            error = (x["out"][qb : qb + nq].transpose(0, 1).double() - expected).abs()
            assert torch.all(error <= bound), float((error / bound).max())
            torch.testing.assert_close(x["stats"][qb : qb + nq].T.double(), scores.logsumexp(-1), atol=2e-4, rtol=0)
            qb += nq
            kb += nk

    poison(first)
    run(first)
    check(first, first_lengths)

    # Reuse the compiled plan with new pointers and a different packed total.
    second_lengths = [241, 145, 47]
    second = inputs(second_lengths)
    poison(second)
    run(second)
    check(second, second_lengths)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(second)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run(second)
        poison(second)
        graph.replay()
        check(second, second_lengths)

        # Setup must republish the shorter extent on replay. The retired K/V
        # rows become NaN, so a stale descriptor corrupts the final partial tile.
        third_lengths = [225, 129, 35]
        second["k_lens"].copy_(torch.tensor(third_lengths, dtype=torch.int32))
        second["q"][:tq].normal_()
        for name in ("k", "v"):
            second[name][: sum(third_lengths)].normal_()
            second[name][sum(third_lengths) :].fill_(float("nan"))
        poison(second)
        graph.replay()
        check(second, third_lengths)
    finally:
        graph.reset()


@pytest.mark.parametrize(
    "d,dv,block_scaled",
    [(128, 128, False), (192, 128, False), (256, 256, False), (512, 512, False), (128, 128, True), (192, 128, True)],
    ids=["d128_fp8", "d192_fp8", "d256_fp8", "d512_fp8", "d128_mxfp8", "d192_mxfp8"],
)
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"])
@torch_fork_set_rng(seed=1313)
def test_quantized_thd_tensormaps_rebind_and_replay(d, dv, block_scaled, input_dtype):
    """Rebuilt FP8 maps must clip poisoned tails after rebinding and replay."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    if _SM == 107 and block_scaled:
        pytest.skip("SM107 MXFP8 does not support THD")
    if _SM != 107 and d == 256:
        pytest.skip("The changed D256 FP8 template is SM107 only")

    q_lengths = [193, 97, 33]
    b, h, hk, sq, sk = 3, 8, 2, 193, 257
    tq = sum(q_lengths)
    ones = torch.ones(1, dtype=torch.float32, device="cuda")

    def tensor_view(raw, heads, length):
        width = raw.shape[-1]
        return raw.as_strided((b, heads, length, width), (length * heads * width, width, heads * width, 1))

    def update(x, k_lengths):
        x["k_lens"].copy_(torch.tensor(k_lengths, dtype=torch.int32))
        for name, heads, length, lens in (("q", h, sq, q_lengths), ("k", hk, sk, k_lengths), ("v", hk, sk, k_lengths)):
            width = dv if name == "v" else d
            values, dequantized, factors = [], [], []
            for n in lens:
                source = torch.randn(1, heads, n, width, dtype=torch.float32, device="cuda") * 0.25
                if block_scaled:
                    dd, ds, sf_d, sd, ss, sf_s = quantize_to_mxfp8(source, 1, heads, n, width, 32, input_dtype, with_ref=True)
                    value, scale, sf = (sd, ss, sf_s) if name == "v" else (dd, ds, sf_d)
                    tiles = (n + 127) // 128
                    if name == "v":
                        packed_sf = sf.view(torch.uint8).reshape(width // 128, heads, tiles, 512).permute(1, 2, 0, 3).contiguous().reshape(heads, tiles, -1)
                    else:
                        packed_sf = sf.view(torch.uint8).reshape(heads, tiles, -1)
                    factors.append(packed_sf)
                    reference = value.double() * scale.reshape(1, heads, n, width).double()
                else:
                    value = source.to(input_dtype)
                    reference = value.double()
                values.append(value.squeeze(0).transpose(0, 1))
                dequantized.append(reference.squeeze(0).transpose(0, 1))
            packed = torch.cat(values)
            x[name].copy_(torch.full_like(x[name], float("nan"), dtype=torch.float32).to(input_dtype))
            x[name][: packed.shape[0]].copy_(packed)
            x["reference"][name] = torch.cat(dequantized)
            if block_scaled:
                sf = torch.cat(factors, dim=1)
                x["sf_" + name].fill_(255)
                x["sf_" + name][:, : sf.shape[1]].copy_(sf)

    def inputs(k_lengths):
        x = dict(reference={}, q_lens=torch.tensor(q_lengths, dtype=torch.int32, device="cuda"), k_lens=torch.empty(b, dtype=torch.int32, device="cuda"))
        for name, heads, length in (("q", h, sq), ("k", hk, sk), ("v", hk, sk)):
            width = dv if name == "v" else d
            x[name] = torch.empty(b * length, heads, width, dtype=input_dtype, device="cuda")
            if block_scaled:
                x["sf_" + name] = torch.empty(heads, b * ((length + 127) // 128), ((width + 127) // 128) * 512, dtype=torch.uint8, device="cuda")
        x["out"] = torch.empty(b * sq, h, dv, dtype=torch.bfloat16, device="cuda")
        x["stats"] = torch.empty(b * sq, h, dtype=torch.float32, device="cuda")
        update(x, k_lengths)
        return x

    def views(x):
        return (
            tensor_view(x["q"], h, sq),
            tensor_view(x["k"], hk, sk),
            tensor_view(x["v"], hk, sk),
            tensor_view(x["out"], h, sq),
            x["stats"].as_strided((b, h, sq), (sq * h, 1, h)),
        )

    first_lengths = [257, 161, 65]
    first = inputs(first_lengths)
    q, k, v, out, stats = views(first)
    cga = 1 if d == 256 or (d == 512 and block_scaled and _SM != 107) else 2
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=out,
        sample_lse=stats,
        thd=True,
        is_causal=True,
        causal_bottom_right=True,
        seq_kv_lens_present=True,
        cga=cga,
        sched_policy=1,
        split_kv=1,
        pack_gqa=False,
        pertensor_fp8=not block_scaled,
        has_amax_o=False,
    )
    assert api.check_support()
    api.compile()
    assert api._k_mod.CFG.THD_VARLEN and api._k_mod.CFG.CTA_MMA == cga
    workspace = torch.empty(api.scratch_workspace_bytes(), dtype=torch.uint8, device="cuda")

    def run(x):
        kwargs = (
            {"sf_" + name: x["sf_" + name] for name in ("q", "k", "v")} if block_scaled else dict(descale_q=ones, descale_k=ones, descale_v=ones, scale_o=ones)
        )
        api.execute(*views(x)[:4], x["stats"], seq_q_lens=x["q_lens"], seq_kv_lens=x["k_lens"], workspace=workspace, **kwargs)

    def poison(x):
        x["out"].fill_(float("nan"))
        x["stats"].fill_(float("nan"))

    def check(x, k_lengths):
        qb = kb = 0
        for nq, nk in zip(q_lengths, k_lengths):
            q = x["reference"]["q"][qb : qb + nq].transpose(0, 1)
            k = x["reference"]["k"][kb : kb + nk].transpose(0, 1).repeat_interleave(h // hk, dim=0)
            v = x["reference"]["v"][kb : kb + nk].transpose(0, 1).repeat_interleave(h // hk, dim=0)
            scores = q @ k.transpose(-1, -2) * d**-0.5
            mask = torch.arange(nk, device="cuda")[None, :] > torch.arange(nq, device="cuda")[:, None] + nk - nq
            scores.masked_fill_(mask, -float("inf"))
            expected = scores.softmax(-1) @ v
            # Existing quantized suites' half-output bounds include P rounding.
            torch.testing.assert_close(x["out"][qb : qb + nq].transpose(0, 1).double(), expected, atol=0.05 if block_scaled else 0.04, rtol=0)
            # D192 MXFP8 mixes quadratic exp2 approximations into the row sum.
            # The unchanged kernel has the same LSE (bitwise); its maximum
            # error over this binding/replay probe is 2.40e-4 on B200.
            lse_atol = 4e-4 if block_scaled and d == 192 else 2e-4
            torch.testing.assert_close(x["stats"][qb : qb + nq].T.double(), scores.logsumexp(-1), atol=lse_atol, rtol=0)
            qb += nq
            kb += nk

    poison(first)
    run(first)
    check(first, first_lengths)
    second_lengths = [241, 145, 47]
    second = inputs(second_lengths)
    poison(second)
    run(second)
    check(second, second_lengths)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(second)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            run(second)
        poison(second)
        graph.replay()
        check(second, second_lengths)
        third_lengths = [225, 129, 35]
        update(second, third_lengths)
        poison(second)
        graph.replay()
        check(second, third_lengths)
    finally:
        graph.reset()
