# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Does the TMA alignment gate check the extent where TMA cares about the stride?

_tma_alignment_reject checks ``extent * bits % 128 == 0`` -- K for a k-major
operand. Its docstring says TMA encodes "the contiguous input dimension's
STRIDE in 16-byte units". Those are the same number only when rows are dense.

Outer strides are free at runtime: _contiguous_dim only asks which dim has
stride 1, and the max-allocation override case depends on that freedom. So a
caller can name a [m, k] corner of an [mb, kb] allocation whose ROW STRIDE is
not 16-byte aligned while k is.

    k  = 64 at bf16 -> 64 * 2  = 128 bytes, aligned
    kb = 72 at bf16 -> 72 * 2  = 144 bytes, NOT a multiple of 16

If the gate accepts that and the result disagrees with the backend, the gate is
checking the wrong quantity. Run twice, with CUDNN_FRONTEND_ENABLE_FROST_ENGINES
unset and set to 1.
"""

import os

import torch

import cudnn

BF16, F32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT
MB, NB, KB = 256, 256, 72  # KB * 2 bytes = 144, not 16-byte aligned
M, N, K = 128, 192, 64  # K * 2 bytes = 128, aligned


def small(*shape):
    return torch.empty(*shape, dtype=torch.int32).random_(-2, 2).to(torch.bfloat16).cuda()


def run(a, b, c):
    g = cudnn.pygraph(
        io_data_type=BF16,
        intermediate_data_type=F32,
        compute_data_type=F32,
        is_dynamic_shape_enabled=True,
        is_override_shape_enabled=True,
    )
    A = g.tensor(name="A", uid=1, dim=[1, MB, KB], stride=[MB * KB, KB, 1], data_type=BF16)
    B = g.tensor(name="B", uid=2, dim=[1, KB, NB], stride=[KB * NB, 1, KB], data_type=BF16)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16).set_uid(3)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    g.check_support()
    g.build_plans()
    name = g.get_plan_name_at_index(g._plan_index)

    handle = cudnn.create_handle()
    uids = [1, 2, 3]
    shapes = [[1, M, K], [1, K, N], [1, M, N]]
    strides = [[MB * KB, KB, 1], [NB * KB, 1, KB], [MB * NB, NB, 1]]
    wsz = g.get_workspace_size_plan_at_index(g._plan_index, handle, uids, shapes, strides)
    ws = torch.empty(max(wsz, 1), device="cuda", dtype=torch.uint8)
    c.zero_()
    g.execute({1: a, 2: b, 3: c}, ws, handle=handle, override_uids=uids, override_shapes=shapes, override_strides=strides)
    torch.cuda.synchronize()
    return name


def main():
    tag = "FROST on " if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES") == "1" else "FROST off"
    a, b = small(1, MB, KB), small(1, NB, KB)
    c = torch.zeros(1, MB, NB, dtype=torch.bfloat16, device="cuda")
    ref = torch.einsum("bmk,bnk->bmn", a[:, :M, :K].float(), b[:, :N, :K].float()).to(torch.bfloat16)
    try:
        name = run(a, b, c)
    except Exception as exc:
        print(f"{tag} | refused: {type(exc).__name__}: {str(exc)[:150]}")
        return
    got = c[:, :M, :N]
    err = (got.float() - ref.float()).abs().max().item()
    print(f"{tag} | {name:14s} accepted the launch; correct={torch.equal(got, ref)} max|d|={err:.1f}")


if __name__ == "__main__":
    main()
