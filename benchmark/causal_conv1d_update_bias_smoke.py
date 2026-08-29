# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ComputeLab correctness smoke for optional decode causal-conv bias."""

from pathlib import Path

import cudnn
import torch
import torch.nn.functional as F

SOURCE_CUDNN = Path(__file__).resolve().parents[1] / "python" / "cudnn"
if str(SOURCE_CUDNN) not in cudnn.__path__:
    cudnn.__path__.insert(0, str(SOURCE_CUDNN))

from cudnn.ops import causal_conv1d_update  # noqa: E402


@torch.no_grad()
def main():
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    x = torch.randn((128, 4096), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    weight = torch.randn((4096, 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    bias = torch.randn((4096,), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    initial_state = torch.randn((128, 4096, 4), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.25
    rows = {}
    for name, value in (("none", None), ("bias", bias)):
        state = initial_state.clone()
        expected_state = torch.cat((initial_state[..., 1:], x.unsqueeze(-1)), dim=-1)
        accumulator = (expected_state.float() * weight.float().unsqueeze(0)).sum(-1)
        if value is not None:
            accumulator = accumulator + value.float()
        expected = F.silu(accumulator).to(torch.bfloat16)
        actual = causal_conv1d_update(x, state, weight, bias=value, activation="silu")
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(state.view(torch.int16), expected_state.view(torch.int16), atol=0, rtol=0)
        rows[name] = float((actual.float() - expected.float()).abs().max())
    print(
        {
            "device": torch.cuda.get_device_name(),
            "shape": list(x.shape),
            "max_abs": rows,
        }
    )


if __name__ == "__main__":
    main()
