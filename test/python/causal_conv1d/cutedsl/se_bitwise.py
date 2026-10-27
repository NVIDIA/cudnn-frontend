# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw-bit comparison against the SE entry points exported by cuDNN Frontend."""

import hashlib
import json
import math
from pathlib import Path

import torch

MANIFEST = json.loads(Path(__file__).with_name("se_levels_cases.json").read_text())
CASES = MANIFEST["cases"]


def bitwise_difference(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    integer = {2: torch.int16, 4: torch.int32, 8: torch.int64}[actual.element_size()]
    a, b = actual.detach().contiguous().flatten(), expected.detach().contiguous().flatten()
    mismatches, nonfinite, max_abs, first = 0, 0, 0.0, []
    for offset in range(0, a.numel(), 1 << 20):
        aa, bb = a[offset : offset + (1 << 20)], b[offset : offset + (1 << 20)]
        different = aa.view(integer) != bb.view(integer)
        count = int(different.sum())
        mismatches += count
        nonfinite += int((~torch.isfinite(aa) | ~torch.isfinite(bb)).sum())
        if count:
            max_abs = max(max_abs, float((aa.double() - bb.double()).abs().max()))
            indices = different.nonzero().flatten()[: max(0, 4 - len(first))].cpu().tolist()
            mask = (1 << (8 * a.element_size())) - 1
            for index in indices:
                first.append(
                    {
                        "index": offset + index,
                        "actual": float(aa[index]),
                        "expected": float(bb[index]),
                        "actual_bits": hex(int(aa.view(integer)[index]) & mask),
                        "expected_bits": hex(int(bb.view(integer)[index]) & mask),
                    }
                )
    return {
        "equal": mismatches == 0 and nonfinite == 0,
        "mismatched_elements": mismatches,
        "elements": a.numel(),
        "nonfinite_elements": nonfinite,
        "max_abs": max_abs,
        "first": first,
    }


ATOMIC_OUTPUTS = frozenset({"dw", "db", "dwp", "dwm", "dskip"})


def parameter_difference(actual, expected, reduction_size):
    """Use compute-dtype tolerances; never inherit the input's half precision."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    atol = 1e-10 if actual.dtype == torch.float64 else max(1e-4, 2e-5 * math.sqrt(reduction_size))
    rtol = 1e-10 if actual.dtype == torch.float64 else 2e-4
    a, b = actual.double(), expected.double()
    error = (a - b).abs()
    limit = atol + rtol * b.abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    failures = int((~finite | (error > limit)).sum())
    return {
        "accepted": failures == 0,
        "atol": atol,
        "rtol": rtol,
        "failed_elements": failures,
        "max_abs": float(error.max()),
        "max_tolerance_fraction": float((error / limit).max()),
    }


def run_case(case):
    import cudnn
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda
    from cudnn.causal_conv1d_se.kernel import launch_fwd
    from cudnn.causal_conv1d_se.backward_kernel import launch_bwd
    from cudnn.causal_conv1d_se.b2b_kernel import launch_b2b_fwd, launch_b2b_bwd

    kernel_dir = Path(__file__).resolve().parents[4] / "python/cudnn/causal_conv1d_se"

    def kernel_hashes():
        return {name: hashlib.sha256((kernel_dir / name).read_bytes()).hexdigest() for name in ("kernel.py", "backward_kernel.py", "b2b_kernel.py")}

    source_hashes = kernel_hashes()
    B, D, L = case["shape"]
    K, Km = case["width"], case["mixer_width"]
    b2b, nwh = case["family"] == "b2b", case["family"] == "nwh"
    backward = case["direction"] == "backward"
    dtype = getattr(torch, case["dtype"])
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    typecode = {torch.float32: 0, torch.float64: 1, torch.float16: 2, torch.bfloat16: 9}[dtype]
    gradcode = 1 if compute == torch.float64 else 0
    seed = 20261023 + int(hashlib.sha256(case["name"].encode()).hexdigest()[:8], 16)
    gen = torch.Generator().manual_seed(seed)

    def rand(shape):
        return (torch.randn(shape, generator=gen) * (0.25 if b2b else 1.0)).to(device="cuda", dtype=dtype)

    x = rand((B, 3 * D, L) if b2b else ((B, L, D) if nwh else (B, D, L)))
    w = rand((3 * D, K) if b2b else ((K, D) if nwh else (D, K)))
    bias = rand((D,))
    yshape = (B, D, L) if b2b else x.shape
    dy = rand(yshape)
    raw_stream = torch.cuda.current_stream().cuda_stream
    stream = cuda.CUstream(raw_stream)
    wm, saved = (rand((D, Km)), torch.empty(yshape, device="cuda", dtype=dtype)) if b2b else (None, None)
    if b2b and backward:
        auxiliary = torch.empty_like(saved)
        cudnn.b2b_causal_conv1d_forward(raw_stream, *(t.data_ptr() for t in (x, w, wm, bias, saved, auxiliary)), B, D, L, K, Km, typecode)
        del auxiliary
    if backward:
        reference = {"dx": torch.empty_like(x)}
        if b2b:
            reference.update(dwp=torch.empty_like(w, dtype=compute), dwm=torch.empty_like(wm, dtype=compute), dskip=torch.empty_like(bias, dtype=compute))
        else:
            reference.update(dw=torch.empty_like(w, dtype=compute), db=torch.empty_like(bias, dtype=compute))
    else:
        reference = {"y": torch.empty(yshape, device="cuda", dtype=dtype)}
        if b2b:
            reference["yg"] = torch.empty_like(reference["y"])
    candidate = {name: torch.empty_like(value) for name, value in reference.items()}
    dz = torch.empty_like(x, dtype=compute) if backward and not b2b else None
    if b2b:
        function = launch_b2b_bwd if backward else launch_b2b_fwd
        tensors = [x, w, wm, bias] + ([saved, dy] if backward else []) + list(candidate.values())
        constants = [K, Km]
        binding = "b2b_causal_conv1d_" + case["direction"]
    else:
        function = launch_bwd if backward else launch_fwd
        tensors = [x, w, bias] + ([dy, dz] if backward else []) + list(candidate.values())
        binding = ("causal_conv1d_nwh_" if nwh else "causal_conv1d_") + case["direction"]
        if backward:
            constants = [K, case["silu"], nwh, 32 if nwh else 4]
        else:
            if nwh:
                vector = 1 if dtype == torch.float64 or K >= 64 else (4 if D % 4 == 0 and (D > 768 or B * D * L > 10_000_000) else (2 if D % 2 == 0 else 1))
                tile = 4 if K >= 32 else (16 if dtype == torch.float32 and B * D * L >= 8_000_000 else (32 if B * D * L >= 32_000_000 else 8))
            else:
                tile, vector = 8, 8 if L % 8 == 0 and dtype != torch.float64 else 1
            constants = [K, case["silu"], nwh, tile, vector, nwh and vector > 1 and torch.cuda.get_device_capability()[0] >= 10]
    compiled = cute.compile(function, *(from_dlpack(t, assumed_align=16) for t in tensors), stream, *constants, options="--enable-tvm-ffi")

    def backend_run():
        if backward:
            for name, value in reference.items():
                if name != "dx":
                    value.zero_()
        inputs = [x, w, wm, bias] + ([saved, dy] if backward else []) if b2b else [x, w, bias] + ([dy] if backward else [])
        dimensions = [B, D, L, K, Km, typecode] if b2b else [B, D, L, K, typecode]
        if backward:
            dimensions.append(gradcode)
        if not b2b:
            dimensions.append(int(case["silu"]))
        getattr(cudnn, binding)(raw_stream, *(t.data_ptr() for t in inputs + list(reference.values())), *dimensions)

    result = {
        "case": case,
        "seed": seed,
        "gpu": torch.cuda.get_device_name(),
        "cc": list(torch.cuda.get_device_capability()),
        "frontend": cudnn.__file__,
        "backend_version": cudnn.backend_version(),
        "backend_binding": binding,
        "gradient_dtype": str(compute) if backward else None,
        "saved_y_source": "common backend forward" if b2b and backward else None,
        "acceptance_contract": "exact_data_atomic_parameter_tolerance_v1",
        "kernel_sha256": source_hashes,
        "trials": [],
    }
    for trial in range(max(3, case["iterations"])):
        for value in list(reference.values()) + list(candidate.values()):
            value.fill_(float("nan"))
        backend_run()
        compiled(*tensors, stream)
        row = {"cross": {name: bitwise_difference(candidate[name], reference[name]) for name in reference}}
        row["parameters"] = {name: parameter_difference(candidate[name], reference[name], B * L) for name in reference if name in ATOMIC_OUTPUTS}
        if trial == 0:
            first_backend = {name: value.clone() for name, value in reference.items()}
            first_candidate = {name: value.clone() for name, value in candidate.items()}
        else:
            row["backend_repeat"] = {name: bitwise_difference(reference[name], first_backend[name]) for name in reference}
            row["candidate_repeat"] = {name: bitwise_difference(candidate[name], first_candidate[name]) for name in reference}
        result["trials"].append(row)
    result["bitwise_match"] = all(v["equal"] for row in result["trials"] for v in row["cross"].values())
    result["backend_repeatable"] = all(v["equal"] for row in result["trials"][1:] for v in row["backend_repeat"].values())
    result["candidate_repeatable"] = all(v["equal"] for row in result["trials"][1:] for v in row["candidate_repeat"].values())
    result["data_bitwise_match"] = all(v["equal"] for row in result["trials"] for name, v in row["cross"].items() if name not in ATOMIC_OUTPUTS)
    result["parameters_accepted"] = all(v["accepted"] for row in result["trials"] for v in row["parameters"].values())
    result["accepted"] = result["data_bitwise_match"] and result["parameters_accepted"]
    if kernel_hashes() != source_hashes:
        raise RuntimeError("Kernel sources changed during the test")
    return result
