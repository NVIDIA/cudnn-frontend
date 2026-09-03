# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qlen=1 M64 forward comparison for cluster development."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch

repo = Path(__file__).resolve().parents[2]
cudnn = types.ModuleType("cudnn")
cudnn.__path__ = [str(repo / "python" / "cudnn")]
cudnn.__file__ = str(repo / "python" / "cudnn" / "__init__.py")
cudnn.HSTUBwdSm100 = object
cudnn.HSTUFwdSm100 = object
sys.modules["cudnn"] = cudnn

hstu = types.ModuleType("cudnn.hstu_attention")
hstu.__path__ = [str(repo / "python" / "cudnn" / "hstu_attention")]
sys.modules["cudnn.hstu_attention"] = hstu

benchmark_path = repo / "benchmark" / "hstu_attention" / "benchmark_hstu_qlen1.py"
spec = importlib.util.spec_from_file_location("hstu_q1_benchmark", benchmark_path)
assert spec is not None and spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

device = torch.device("cuda")
dtype = torch.bfloat16
heads = 4
head_dim = 128
k_lengths = [1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 2048, 2049, 3072]
batch = len(k_lengths)
generator = torch.Generator(device=device)
generator.manual_seed(20260903)
q = torch.randn((batch, heads, head_dim), dtype=dtype, device=device, generator=generator) * 0.2
k = torch.randn((sum(k_lengths), heads, head_dim), dtype=dtype, device=device, generator=generator) * 0.2
v = torch.randn_like(k)
cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
cu_k = torch.zeros(batch + 1, dtype=torch.int32, device=device)
cu_k[1:] = torch.tensor(k_lengths, dtype=torch.int32, device=device).cumsum(0)
tensors = {
    "q": q,
    "k": k,
    "v": v,
    "do": torch.zeros_like(q),
    "cu_q": cu_q,
    "cu_k": cu_k,
    "k_lengths": k_lengths,
}

outputs = {}
implementations = (
    "tc",
    "tc-m64-16dp-tail-kv5",
    "tc-m64-16dp-tail-kv5-split2",
    "tc-m64-16dp-tail-kv5-split4",
)
for implementation in implementations:
    out, run, _ = benchmark._compile_forward(tensors, max(k_lengths), (-1, 0), 0.7, 2048.0, implementation)
    run()
    torch.cuda.synchronize()
    outputs[implementation] = out.cpu().float()

reference = benchmark._reference_forward(q.cpu().float(), k.cpu().float(), v.cpu().float(), k_lengths, 0.7, 2048.0)
for row, length in enumerate(k_lengths):
    tc_error = (outputs["tc"][row] - reference[row]).abs().max().item()
    errors = {implementation: (outputs[implementation][row] - reference[row]).abs().max().item() for implementation in implementations}
    cross_errors = {implementation: (outputs[implementation][row] - outputs["tc"][row]).abs().max().item() for implementation in implementations[1:]}
    print(f"LEN {length:4d} ERRORS {errors} CROSS {cross_errors}", flush=True)
    if length in (128, 2048):
        print("  REF", reference[row, 0, :8].tolist(), flush=True)
        print("  TC ", outputs["tc"][row, 0, :8].tolist(), flush=True)
        print("  M64", outputs["tc-m64-16dp-tail-kv5"][row, 0, :8].tolist(), flush=True)
