# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent FP8 reference data for the SM100/SM103 D256 backward owner."""

from types import SimpleNamespace

import cudnn
import torch

from cudnn.api_base import TensorDesc
from sdpa.fp8_ref import compute_ref, compute_ref_backward
from sdpa.helpers import get_fp8_descale_factor, get_fp8_scale_factor

NAME = "sdpa_bwd_sm100_d256_fp8"
GRADS = ("dQ", "dK", "dV")
AMAX = tuple("amax_" + n for n in (*GRADS, "dP"))
DTYPES = {
    torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
    torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
    torch.float16: cudnn.data_type.HALF,
    torch.bfloat16: cudnn.data_type.BFLOAT16,
}


def make_case(itype, otype, *, b=1, h=2, hk=1, sq=17, sk=33, causal=False, left=None, seed=1737):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def draw(s, heads):
        x = torch.randn(b, s, heads, 256, generator=gen).cuda()
        scale = get_fp8_scale_factor(x.abs().max().item(), itype)
        return (x * scale).to(itype), 1 / scale

    q, qs = draw(sq, h)
    k, ks = draw(sk, hk)
    v, vs = draw(sk, hk)
    do, dos = draw(sq, h)
    scale = 256**-0.5
    ps = get_fp8_scale_factor(1.0, itype)
    mask = dict(left_bound=left, right_bound=0 if causal else None, diag_align=cudnn.diagonal_alignment.TOP_LEFT if causal else None)
    o, stats, om = compute_ref(q, k, v, scale, qs, ks, vs, ps, 1 / ps, itype, itype, **mask)
    o = o.contiguous()
    ods = get_fp8_descale_factor(om, itype)

    def reference(return_intermediates=False):
        return compute_ref_backward(
            q,
            k,
            v,
            o,
            do,
            scale,
            qs,
            ks,
            vs,
            ps,
            1 / ps,
            itype,
            ods,
            dos,
            otype,
            stats=stats,
            return_intermediates=return_intermediates,
            quantize_grads=False,
            **mask,
        )

    dq, dk, dv, _, amdp, amdq, amdk, amdv, inter = reference("amax")
    dps = get_fp8_scale_factor(amdp, itype)
    grad_scales = dict(zip(GRADS, (get_fp8_scale_factor(x, otype) for x in (amdq, amdk, amdv))))
    values = dict(
        descale_q=qs,
        descale_k=ks,
        descale_v=vs,
        descale_s=1 / ps,
        scale_s=ps,
        descale_o=ods,
        descale_dO=dos,
        descale_dP=1 / dps,
        scale_dQ=grad_scales["dQ"],
        scale_dK=grad_scales["dK"],
        scale_dV=grad_scales["dV"],
        scale_dP=dps,
    )
    return SimpleNamespace(
        itype=itype,
        otype=otype,
        b=b,
        h=h,
        hk=hk,
        sq=sq,
        sk=sk,
        causal=causal,
        left=left,
        scale=scale,
        inputs=dict(q=q, k=k, v=v, o=o, dO=do, stats=stats.contiguous()),
        scalars={n: torch.tensor([x], device="cuda", dtype=torch.float32) for n, x in values.items()},
        refs={n: (x * grad_scales[n]).to(otype) for n, x in zip(GRADS, (dq, dk, dv))},
        ref_amax=dict(zip(AMAX, (amdq, amdk, amdv, inter["ds_scaled"].item() / dps))),
        grad_scales=grad_scales,
        reference=reference,
        operands=dict(dQ=k.float() * ks, dK=q.float() * qs, dV=do.float() * dos),
        flip_units=dict(dQ=1 / dps, dK=1 / dps, dV=1 / ps),
        keys=dict(dQ=sk, dK=sq, dV=sq),
    )


def view(x):
    return x.transpose(1, 2)


def desc(x):
    return TensorDesc(x.dtype, tuple(x.shape), tuple(x.stride()), TensorDesc._compute_stride_order(x.shape, x.stride()), x.device)


def standalone(case, request_amax=AMAX):
    from cudnn.sdpa.bwd.kernels.sm100.d256_fp8.host import SdpaBwdDslSm100D256Fp8

    outputs = {n: torch.empty_like(case.inputs[src], dtype=case.otype) for n, src in zip(GRADS, ("q", "k", "v"))}
    amax = {n: torch.empty((1,), device="cuda") for n in request_amax}
    tensors = [view(case.inputs[n]) for n in ("q", "k", "v", "o", "dO")]
    tensors += [case.inputs["stats"]] + [view(outputs[n]) for n in GRADS]
    api = SdpaBwdDslSm100D256Fp8(
        *map(desc, tensors),
        amax_requested=tuple(amax),
        scale_softmax=case.scale,
        is_causal=case.causal,
        window_size_left=None if case.left is None else case.left - 1,
    )
    api.check_support()
    api.compile()
    ws = torch.empty(api.scratch_workspace_bytes(), device="cuda", dtype=torch.uint8)

    def launch():
        api.execute(*tensors, workspace=ws, **case.scalars, **amax)

    return SimpleNamespace(api=api, outputs={**outputs, **amax}, workspace=ws, launch=launch)


def execute(run):
    for x in run.outputs.values():
        x.fill_(float("nan"))
    run.workspace.fill_(255)
    run.launch()
    return {n: x.clone() for n, x in run.outputs.items()}


def check(case, outputs):
    from sdpa.fp8 import assert_close_fp8_grad

    for n in GRADS:
        got = outputs[n].float() / case.grad_scales[n]
        want = case.refs[n].float() / case.grad_scales[n]
        assert torch.isfinite(got).all(), n
        assert_close_fp8_grad(
            got,
            want,
            0.08,
            0.2,
            tag=n,
            keys=case.keys[n],
            operand=case.operands[n],
            flip_unit=case.flip_units[n],
            intermediates=lambda selection: case.reference(selection)[8],
            fp8_dtype=case.itype,
            out_dtype=case.otype,
        )
    for n in AMAX:
        if n in outputs:
            # dP is max|dS| before quantization, not max|dO V^T|.
            tol = dict(atol=1e-4, rtol=1e-2) if n == "amax_dP" else dict(atol=0.08, rtol=0.2)
            torch.testing.assert_close(outputs[n], torch.full_like(outputs[n], case.ref_amax[n]), **tol)


def snapshot_inputs(case):
    from determinism_utils import snapshot

    empty = (
        torch.arange(case.sq, device="cuda") >= (case.sk + case.left - 1) if case.left is not None else torch.zeros(case.sq, device="cuda", dtype=torch.bool)
    )
    expected = empty[None, None, :, None].expand_as(case.inputs["stats"])
    assert torch.equal(torch.isneginf(case.inputs["stats"]), expected), "Stats infinity must occur only on fully masked rows"
    return snapshot({**case.inputs, **case.scalars}, allowed_inf={"stats": -torch.inf})


def graph_plan(case, request_amax=AMAX, record=None):
    from test_sdpa_bwd_fp8_sm107 import _graph_and_ports

    graph, ports = _graph_and_ports(
        case.b,
        case.h,
        case.sq,
        256,
        hkv=case.hk,
        skv=case.sk,
        causal=case.causal,
        left_bound=case.left,
        fp8=DTYPES[case.itype],
        grad_dtypes=(DTYPES[case.otype],) * 3,
        request_amax=request_amax,
        deterministic=True,
        scale=case.scale,
    )
    graph.validate()
    graph.build_operation_graph()
    if record is None:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    else:
        graph.create_execution_plan(*record)
        graph.select_plan(0)
    graph.check_support()
    graph.build_plans()
    record = graph.get_engine_and_knobs_at_index(graph._plan_index)
    assert record == (20637, {}), record
    outputs = {n: torch.empty_like(case.inputs[src], dtype=case.otype) for n, src in zip(GRADS, ("q", "k", "v"))}
    amax = {n: torch.empty((1,), device="cuda") for n in request_amax}
    pack = {ports[n]: x if n == "stats" else view(x) for n, x in case.inputs.items()}
    pack.update({ports[n]: x for n, x in case.scalars.items()})
    pack.update({ports[n]: view(x) for n, x in outputs.items()})
    pack.update({ports[n]: x for n, x in amax.items()})
    ws = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def launch():
        graph.execute(pack, ws)

    return SimpleNamespace(graph=graph, record=record, ports=ports, pack=pack, outputs={**outputs, **amax}, workspace=ws, launch=launch)
