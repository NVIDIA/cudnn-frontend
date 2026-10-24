# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A/B the atomic and deterministic CSA compressor backward class APIs.

The two variants share all input/output addresses. Each round reverses measurement
order (AB, then BA) to suppress boost-clock/order bias; ``--construct-order`` also
reverses which independently compiled plan is built first. ``kernel_us`` measures
only ``execute`` (the atomic dAPE buffer is cleared before its event); ``operation_us``
includes the atomic path's required dAPE clear and every deterministic reduction
launch. Allocations and JIT are outside both measurements.

Run this script against an unchanged-source checkout with ``--atomic-only`` to record
the independent upstream control. The atomic constructor deliberately omits the new
keyword, so the same script works before and after this change.
"""

import argparse
import json
import statistics

import torch
from cudnn.csa import CSACompressorBackward
from cudnn.csa.compressor.compressor_sm100 import _BWD_ROWS

# On the seeded 1M case, both deterministic trees stay within 2.1e-4 of the
# FP64 sum while the atomic result varies by more than 1.9e-3.
_DAPE_FP64_ATOL = 2.5e-4

SHAPES = (
    ("r4-c2-d128-1x8192", [8192], 128, 4, 2),
    ("r4-c2-d128-3x8192", [8192] * 3, 128, 4, 2),
    ("r4-c2-d512-1x8192", [8192], 512, 4, 2),
    ("r4-c2-d512-3x8192", [8192] * 3, 512, 4, 2),
    ("r128-c1-d128-1x8192", [8192], 128, 128, 1),
    ("r128-c2-d128-3x8192", [8192] * 3, 128, 128, 2),
    ("r128-c1-d512-1x8192", [8192], 512, 128, 1),
    ("r128-c2-d512-3x8192", [8192] * 3, 512, 128, 2),
)

LONG_CONTEXT_SHAPES = (
    (
        "r4-c1-d128-packed256k-cp32-or64",
        [1024, 3072, 8192, 16384, 32768, 65536, 135168],
        128,
        4,
        1,
    ),
    (
        "r4-c1-d128-packed1m-cp64",
        [4096, 12288, 32768, 65536, 131072, 262144, 540672],
        128,
        4,
        1,
    ),
)


def _make_inputs(lens, d, ratio, coff):
    """Seeded, contiguous class-API inputs."""
    total = sum(lens)
    width = coff * d
    generator = torch.Generator(device="cpu").manual_seed(1234)
    kv = torch.randn(total, width, generator=generator).to(torch.bfloat16).cuda()
    score = torch.randn(total, width, generator=generator).mul_(1.5).to(torch.bfloat16).cuda()
    ape = torch.randn(ratio, width, generator=generator).mul_(0.25).cuda()
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0)], dtype=torch.int32, device="cuda")
    blocks = torch.tensor([length // ratio for length in lens]).cumsum(0)
    cuc = torch.tensor([0, *blocks], dtype=torch.int32, device="cuda")
    total_comp = int(cuc[-1].item())
    grad_out = torch.randn(total_comp, d, generator=generator).to(torch.bfloat16).cuda()
    return kv, score, ape, cu, cuc, grad_out


def _build_plan(inputs, ratio, coff, deterministic):
    """Compile one plan; only the deterministic plan receives the new keyword."""
    kv, score, ape, cu, cuc, grad_out = inputs
    kwargs = dict(
        sample_kv=kv,
        sample_score=score,
        sample_ape=ape,
        sample_cu_seqlens=cu,
        sample_cu_seqlens_comp=cuc,
        sample_out=torch.empty_like(grad_out, device="meta"),
        ratio=ratio,
        coff=coff,
    )
    if deterministic:
        kwargs["deterministic"] = True
    plan = CSACompressorBackward(**kwargs)
    plan.check_support()
    plan.compile()
    return plan


def _event_us(fn):
    """One synchronized CUDA-event timing sample."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def _dape_fp64_reference(workspace, total_comp, ratio, coff, d):
    """Sum the reducer's FP32 producer partials independently in FP64."""
    nelems = ratio * coff * d
    if ratio == 4:
        npartials = (total_comp + _BWD_ROWS - 1) // _BWD_ROWS
    else:
        # Ratio 128 chooses rows_per_cta from the runtime device shape; its
        # workspace (like generic ratio 2) contains only the producer planes.
        npartials = workspace.numel() // (nelems * 4)
    partials = workspace[: npartials * nelems * 4].view(torch.float32).view(npartials, ratio, coff * d)
    return partials.sum(dim=0, dtype=torch.float64)


def _bench_shape(name, lens, d, ratio, coff, args):
    inputs = _make_inputs(lens, d, ratio, coff)
    kv, score, ape, cu, cuc, grad_out = inputs
    variants = [False] if args.atomic_only else [False, True]
    if args.construct_order == "deterministic-first":
        variants.reverse()
    plans = {det: _build_plan(inputs, ratio, coff, det) for det in variants}

    grad_kv = torch.empty_like(kv)
    grad_score = torch.empty_like(score)
    grad_ape = torch.empty_like(ape)
    workspace_bytes = plans[True].scratch_workspace_bytes() if True in plans else 0
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda") if workspace_bytes else None

    def execute(det):
        operands = (kv, score, ape, cu, cuc, grad_out, grad_kv, grad_score, grad_ape)
        if det:
            plans[det].execute(*operands, workspace=workspace)
        else:
            plans[det].execute(*operands)

    for _ in range(args.warmup):
        grad_ape.zero_()
        execute(False)
        if True in plans:
            execute(True)
    torch.cuda.synchronize()

    accuracy = {}
    if True in plans:
        grad_ape.zero_()
        execute(False)
        torch.cuda.synchronize()
        atomic_kv, atomic_score, atomic_ape = grad_kv.clone(), grad_score.clone(), grad_ape.clone()
        grad_ape.fill_(float("nan"))
        execute(True)
        torch.cuda.synchronize()
        assert torch.equal(grad_kv, atomic_kv)
        assert torch.equal(grad_score, atomic_score)
        dape_reference = _dape_fp64_reference(workspace, grad_out.shape[0], ratio, coff, d)
        accuracy["atomic_dape_fp64_max_abs"] = (atomic_ape.double() - dape_reference).abs().max().item()
        accuracy["deterministic_dape_fp64_max_abs"] = (grad_ape.double() - dape_reference).abs().max().item()
        torch.testing.assert_close(grad_ape.double(), dape_reference, rtol=0, atol=_DAPE_FP64_ATOL)
        deterministic_ape = grad_ape.clone()
        grad_ape.fill_(float("nan"))
        execute(True)
        torch.cuda.synchronize()
        assert torch.equal(grad_ape, deterministic_ape)

    kernel = {det: [] for det in plans}
    operation = {det: [] for det in plans}
    for round_idx in range(args.iters):
        order = list(plans)
        if round_idx % 2:
            order.reverse()
        for det in order:
            if not det:
                grad_ape.zero_()
            kernel[det].append(_event_us(lambda det=det: execute(det)))
            if det:
                operation[det].append(_event_us(lambda: execute(True)))
            else:
                operation[det].append(_event_us(lambda: (grad_ape.zero_(), execute(False))))

    result = {
        "shape": name,
        "workspace_mib": workspace_bytes / (1024 * 1024),
        "atomic_kernel_us": statistics.median(kernel[False]),
        "atomic_operation_us": statistics.median(operation[False]),
        **accuracy,
    }
    if True in plans:
        result.update(
            deterministic_kernel_us=statistics.median(kernel[True]),
            deterministic_operation_us=statistics.median(operation[True]),
        )
        result["kernel_slowdown"] = result["deterministic_kernel_us"] / result["atomic_kernel_us"]
        result["operation_slowdown"] = result["deterministic_operation_us"] / result["atomic_operation_us"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--construct-order", choices=("atomic-first", "deterministic-first"), default="atomic-first")
    parser.add_argument("--atomic-only", action="store_true")
    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument("--quick", action="store_true", help="Run one generic and one ratio=128 shape")
    shape_group.add_argument(
        "--long-context",
        action="store_true",
        help="Run the unequal packed 256K/CP32-or-64 and 1M/CP64 production shapes",
    )
    args = parser.parse_args()
    shapes = LONG_CONTEXT_SHAPES if args.long_context else (SHAPES[::4] if args.quick else SHAPES)
    results = [_bench_shape(*shape, args) for shape in shapes]
    print(json.dumps({"construct_order": args.construct_order, "results": results}, indent=2))


if __name__ == "__main__":
    main()
