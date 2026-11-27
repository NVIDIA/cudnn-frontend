# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two same-signature BF16 grouped GEMM wrapper calls overlapping on two streams match their serial results bitwise.

The dynamic scheduler's tile counter and the discrete-mode TMA descriptors live in the kernel workspace, so overlapping
calls that shared a cached plan's workspace stole each other's tiles or descriptors (#2881). 32 experts x 512 rows at
N=4096, K=1024 overlapped on GB200 every run. SM90 always runs its dynamic (atomic-counter) scheduler, whose counter lives
in the same workspace.
"""

import pytest
import torch

import cudnn
from cudnn.tensor_adapter import has_wgmma
from determinism_utils import assert_bitwise, snapshot
from gemm.cutedsl.test_grouped_gemm_bf16_utils import make_grouped_gemm_bf16_problem
from gemm.cutedsl.test_grouped_gemm_dglu_bf16_utils import make_grouped_gemm_dglu_bf16_problem
from gemm.cutedsl.test_grouped_gemm_glu_bf16_utils import make_grouped_gemm_glu_bf16_problem
from gemm.cutedsl.test_grouped_gemm_sm90 import override_schedule
from gemm.cutedsl.test_grouped_gemm_wgrad_bf16_utils import make_grouped_gemm_wgrad_bf16_problem

pytestmark = pytest.mark.L0

EXPERTS, ROWS, N, K = 32, 512, 4096, 1024
ITERATIONS = 40
ALPHA = tuple(0.5 + index / EXPERTS for index in range(EXPERTS))


def _make(op, discrete, seed):
    """A canonical-layout problem; the per-seed scale keeps the two overlapping problems' A and B bytes apart."""
    if op == "wgrad":
        problem = make_grouped_gemm_wgrad_bf16_problem(group_k_list=(ROWS,) * EXPERTS, m=N, n=K, a_major="m", b_major="n", discrete=discrete)
        problem["a"].mul_(0.5 * seed)
        problem["b"].mul_(0.5 * seed)
        return problem
    if op == "dglu":
        problem = make_grouped_gemm_dglu_bf16_problem(m=EXPERTS * ROWS, n=N, k=K, experts=EXPERTS, discrete=discrete, alpha_values=ALPHA, beta_values=ALPHA)
        problem["c"] = problem["c"].squeeze(-1)
    else:
        make = make_grouped_gemm_glu_bf16_problem if op == "glu" else make_grouped_gemm_bf16_problem
        problem = make(m=EXPERTS * ROWS, n=N, k=K, experts=EXPERTS, discrete=discrete, alpha_values=ALPHA)
    problem["a"] = problem["a"].squeeze(-1).mul_(0.5 * seed)
    # In place, so the discrete b_ptrs see it too.
    problem["b_storage"].mul_(0.5 * seed)
    problem["prob"] = problem["prob"].view(-1)
    return problem


def _launch(op, problem, sched, arch="sm100", generate_c=True):
    """One ``*_{arch}`` wrapper call on the current stream; returns its outputs (fresh buffers)."""
    if op == "wgrad":
        output = torch.empty_like(problem["output"])
        ptrs = torch.tensor([expert.data_ptr() for expert in output], dtype=torch.int64, device="cuda")
        cudnn.grouped_gemm_wgrad_wrapper_sm100(
            a_tensor=problem["a"],
            b_tensor=problem["b"],
            sfa_tensor=None,
            sfb_tensor=None,
            offsets_tensor=problem["offsets"],
            output_mode="discrete",
            wgrad_ptrs=ptrs,
            acc_dtype=torch.float32,
            wgrad_dtype=torch.bfloat16,
        )
        return {"wgrad": output}
    if op == "dglu":
        result = getattr(cudnn, f"grouped_gemm_dglu_wrapper_{arch}")(
            a_tensor=problem["a"],
            c_tensor=problem["c"],
            sfa_tensor=None,
            padded_offsets=problem["offsets"],
            alpha_tensor=problem["alpha"],
            beta_tensor=problem["beta"],
            prob_tensor=problem["prob"],
            dprob_tensor=torch.zeros_like(problem["prob"]),
            b_tensor=problem["b_storage"],
            d_dtype=torch.bfloat16,
            use_dynamic_sched=sched == "dynamic",
        )
        # dprob is an atomic sum across N tiles, so only D is compared bitwise.
        return {"d": result["d_row_tensor"]}
    kwargs = dict(
        a_tensor=problem["a"],
        padded_offsets=problem["offsets"],
        alpha_tensor=problem["alpha"],
        prob_tensor=problem["prob"],
        c_dtype=torch.bfloat16,
        d_dtype=torch.bfloat16,
        generate_c=generate_c,
        use_dynamic_sched=sched == "dynamic",
    )
    if problem["b_ptrs"] is not None:
        kwargs.update(b_ptrs=problem["b_ptrs"], n=N, b_dtype=torch.bfloat16)
    elif op == "dgrad":
        # The forward weights read N-major, as the dgrad GEMM does.
        kwargs.update(b_tensor=problem["b_storage"].transpose(1, 2).contiguous(), b_major="n")
    else:
        kwargs.update(b_tensor=problem["b_storage"], b_major="k")
    if op == "glu":
        result = getattr(cudnn, f"grouped_gemm_glu_wrapper_{arch}")(sfa_tensor=None, **kwargs)
    else:
        result = getattr(cudnn, f"grouped_gemm_wrapper_{arch}")(**kwargs)
    return {"c": result["c_tensor"], "d": result["d_tensor"]} if generate_c else {"d": result["d_tensor"]}


CASES = [
    ("fwd", False, "static"),
    ("fwd", False, "dynamic"),
    ("dgrad", False, "dynamic"),
    ("glu", False, "dynamic"),
    ("fwd", True, "static"),
    ("fwd", True, "dynamic"),
    ("glu", True, "static"),
    ("glu", True, "dynamic"),
    ("wgrad", True, None),
]


@pytest.mark.parametrize("op,discrete,sched", CASES, ids=[f"{op}-{'discrete' if discrete else 'dense'}-{sched or 'default'}" for op, discrete, sched in CASES])
def test_sm100_bf16_overlapping_streams(op, discrete, sched):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100/103 is required")
    _assert_overlap_matches_serial(op, discrete, sched)


def _assert_overlap_matches_serial(op, discrete, sched, arch="sm100", generate_c=True):
    pair = [_make(op, discrete, seed) for seed in (1, 2)]
    expected = [snapshot(_launch(op, problem, sched, arch, generate_c)) for problem in pair]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for iteration in range(ITERATIONS):
        torch.cuda.synchronize()
        results = []
        for stream, problem in zip(streams, pair):
            with torch.cuda.stream(stream):
                results.append(_launch(op, problem, sched, arch, generate_c))
        torch.cuda.synchronize()
        for index, (result, reference) in enumerate(zip(results, expected)):
            assert_bitwise(snapshot(result), reference, f"iteration {iteration} call {index}")


# N=4096, K=1024: the shape rule picks pingpong for fwd / dgrad / GLU with C; the forced cases cover each schedule. dGLU
# has one schedule.
SM90_CASES = [(op, schedule) for op in ("fwd", "dgrad", "glu") for schedule in (None, "cooperative", "pingpong")] + [("dglu", None)]


@pytest.mark.parametrize("op,schedule", SM90_CASES, ids=[f"{op}-{schedule or 'heuristic'}" for op, schedule in SM90_CASES])
def test_sm90_bf16_overlapping_streams(op, schedule):
    if not torch.cuda.is_available() or not has_wgmma(torch.cuda.get_device_capability()[0]):
        pytest.skip("SM90 is required")
    with override_schedule(schedule):
        # The forced pingpong GLU runs without the saved C; the heuristic case covers pingpong with C.
        _assert_overlap_matches_serial(op, False, "dynamic", "sm90", generate_c=not (op == "glu" and schedule == "pingpong"))
