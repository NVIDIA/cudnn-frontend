# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 dSwiGLU recomputed activation with row x N past 2^31 elements (about 24 GB of device memory).

The activation is written with per-thread stores from a memory tensor, so its row offsets must be 64-bit: expert 0
covers rows 0 .. M - 2048 (its tiles past 2^31 / N rows need a 64-bit tile offset) and expert 1 starts past 2^31 / N
rows (a 64-bit expert offset).  A and C repeat one block of rows, so any stray store into them is detected.
"""

import pytest
import torch

pytestmark = pytest.mark.L1

ROWS, K, N = 532480, 128, 4096  # ROWS x N = 2,181,038,080 > 2^31
BLOCK = 16384
NEEDED_BYTES = 30 * 2**30


@pytest.fixture(autouse=True)
def require_sm100_and_memory():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")
    if torch.cuda.mem_get_info()[0] < NEEDED_BYTES:
        pytest.skip("needs about 30 GB of free device memory")


def repeated_rows(columns, scale, generator):
    block = (torch.randn((BLOCK, columns), generator=generator, device="cuda") * scale).to(torch.bfloat16)
    out = torch.empty((ROWS, columns), device="cuda", dtype=torch.bfloat16)
    for row in range(0, ROWS, BLOCK):
        out[row : row + BLOCK] = block[: min(BLOCK, ROWS - row)]
    return out


def expected_activation(c, beta, prob, rows):
    pair = (c[rows].float() * beta[:, None]).reshape(len(rows), N // 32, 2, 32)
    gate, up = pair[:, :, 0].reshape(len(rows), N), pair[:, :, 1].reshape(len(rows), N)
    return (torch.nn.functional.silu(gate) * up * prob[rows, None]).bfloat16()


def test_dglu_activation_rows_past_2g_elements():
    import cudnn

    generator = torch.Generator(device="cuda").manual_seed(2031)
    a = repeated_rows(K, 0.125, generator)
    c = repeated_rows(2 * N, 2.0, generator)
    weights = (torch.randn((2, N, K), generator=generator, device="cuda") * 0.125).to(torch.bfloat16)
    prob = torch.rand(ROWS, generator=generator, device="cuda") * 0.9 + 0.1
    split = ROWS - 2048
    offsets = torch.tensor([split, ROWS], dtype=torch.int32, device="cuda")
    beta = torch.tensor([0.5, 1.5], device="cuda")
    activation = torch.full((ROWS, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    cudnn.grouped_gemm_dglu_wrapper_sm100(
        a_tensor=a,
        c_tensor=c,
        sfa_tensor=None,
        padded_offsets=offsets,
        alpha_tensor=torch.tensor([1.0, 0.5], device="cuda"),
        beta_tensor=beta,
        prob_tensor=prob,
        dprob_tensor=torch.zeros(ROWS, device="cuda"),
        b_tensor=weights,
        d_dtype=torch.bfloat16,
        act_func="dswiglu",
        activation_tensor=activation,
    )
    torch.cuda.synchronize()

    boundary = 2**31 // N
    for first, last in ((0, 512), (boundary - 512, boundary + 512), (split - 256, split), (split, ROWS)):
        rows = torch.arange(first, last, device="cuda")
        expert_beta = beta[(rows >= split).long()]
        expected = expected_activation(c, expert_beta, prob, rows)
        torch.testing.assert_close(activation[first:last], expected, rtol=0.008, atol=2e-5, msg=f"rows {first}..{last}")
    assert not torch.isnan(activation).any(), "activation rows left unwritten"
    for name, tensor in (("a", a), ("c", c)):
        for row in range(BLOCK, ROWS, BLOCK):
            count = min(BLOCK, ROWS - row)
            assert torch.equal(tensor[row : row + count], tensor[:count]), f"{name} rows {row}.. changed by a stray store"
