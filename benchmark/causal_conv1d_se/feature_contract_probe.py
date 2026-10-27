# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe backend contracts beyond the manifest's zeroed-gradient, small-batch cases."""

import argparse
import json
from pathlib import Path
import sys

import torch
import cudnn
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from cudnn.causal_conv1d_se import kernel, backward_kernel, b2b_kernel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test/python/causal_conv1d/cutedsl"))
    from se_bitwise import parameter_difference, bitwise_difference

    records = []
    generator = torch.Generator().manual_seed(20261026)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def rand(shape):
        return (torch.randn(shape, generator=generator) * 0.25).cuda().bfloat16()

    for family in ("ncw", "nwh", "b2b"):
        B, D, L, K, Km = 2, 16, 64, 3, 7
        b2b, nwh = family == "b2b", family == "nwh"
        x = rand((B, 3 * D, L) if b2b else ((B, L, D) if nwh else (B, D, L)))
        w = rand((3 * D, K) if b2b else ((K, D) if nwh else (D, K)))
        b = rand((D,))
        dy = rand((B, D, L) if b2b else x.shape)
        if b2b:
            wm = rand((D, Km))
            y, yg = torch.empty_like(dy), torch.empty_like(dy)
            cudnn.b2b_causal_conv1d_forward(int(stream), *(t.data_ptr() for t in (x, w, wm, b, y, yg)), B, D, L, K, Km, 9)
            inputs = [x, w, wm, b, y, dy]
            names = ["dx", "dwp", "dwm", "dskip"]
            outputs = [
                torch.empty_like(x),
                torch.zeros_like(w, dtype=torch.float32),
                torch.zeros_like(wm, dtype=torch.float32),
                torch.zeros_like(b, dtype=torch.float32),
            ]
            dimensions = [B, D, L, K, Km, 9, 0]
            constants = [K, Km]
            function = b2b_kernel.launch_b2b_bwd
            backend = cudnn.b2b_causal_conv1d_backward
        else:
            inputs = [x, w, b, dy]
            names = ["dx", "dw", "db"]
            outputs = [torch.empty_like(x), torch.zeros_like(w, dtype=torch.float32), torch.zeros_like(b, dtype=torch.float32)]
            dimensions = [B, D, L, K, 9, 0, 0]
            constants = [K, False, nwh, 32 if nwh else 4]
            function = backward_kernel.launch_bwd
            backend = cudnn.causal_conv1d_nwh_backward if nwh else cudnn.causal_conv1d_backward
        for _ in range(2):
            backend(int(stream), *(t.data_ptr() for t in inputs + outputs), *dimensions)
        actual = [torch.empty_like(outputs[0])] + [torch.zeros_like(t) for t in outputs[1:]]
        tensors = inputs + ([] if b2b else [torch.empty_like(x, dtype=torch.float32)]) + actual
        for t in outputs[1:]:
            t.fill_(0.25)
        for _ in range(2):
            backend(int(stream), *(t.data_ptr() for t in inputs + outputs), *dimensions)
        for t in actual[1:]:
            t.fill_(0.25)
        compiled = cute.compile(function, *(from_dlpack(t, assumed_align=16) for t in tensors), stream, *constants, True, options="--enable-tvm-ffi")
        for _ in range(2):
            compiled(*tensors, stream)
        checks = {
            name: (bitwise_difference(out, ref) if name == "dx" else parameter_difference(out, ref, 2 * B * L))
            for name, out, ref in zip(names, actual, outputs)
        }
        assert all(v.get("equal", v.get("accepted", False)) for v in checks.values()), (family, checks)
        records.append({"feature": "parameter_gradient_accumulation", "family": family, "prefill": 0.25, "calls": 2, "accepted": True, "checks": checks})
    B, D, L, K = 65536, 4, 8, 3
    x, w, b = rand((B, D, L)), rand((D, K)), rand((D,))
    reference, actual = torch.empty_like(x), torch.empty_like(x)
    cudnn.causal_conv1d_forward(int(stream), *(t.data_ptr() for t in (x, w, b, reference)), B, D, L, K, 9, 0)
    torch.cuda.synchronize()
    assert torch.isfinite(reference).all()
    compiled = cute.compile(
        kernel.launch_fwd, *(from_dlpack(t, assumed_align=16) for t in (x, w, b, actual)), stream, K, False, False, 8, 8, False, options="--enable-tvm-ffi"
    )
    compiled(x, w, b, actual, stream)
    check = bitwise_difference(actual, reference)
    assert check["equal"], check
    records.append({"feature": "batch_grid_limit", "shape": [B, D, L], "backend": "pass", "accepted": True, "checks": check})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "records": records}, indent=2) + "\n")
    print(json.dumps({"accumulation_contracts_pass": 3, "large_batch_pass": True}, indent=2))


if __name__ == "__main__":
    main()
