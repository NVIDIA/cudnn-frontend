# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM90 bias forward O/Stats repeatability and batch isolation on the D512 body (engine 20540) and the dedicated D128
template (engine 20541); backward is unsupported.

Fixed plans are compared within one GPU/compiler/runtime. No cross-runtime byte identity
is claimed. The query and key tails are partial tiles of either body, and row 0 is empty.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, assert_golden, snapshot
from frost_test_utils import _SM, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, pytest.mark.skipif(_SM != 90, reason="requires native Hopper")]


def _check_determinism(dtype, bias_fp32, d, causal, *, tile, kernel, golden):
    """Replays, batch reordering and insertion with changed neighbours, captured replay and the golden, for the plan the
    standalone adapter builds at ``tile`` (64 pins the D512 body; 128 the D128 template), whose module must be
    ``kernel``."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm90

    # CPU seeding preserves inputs independently of the GPU's SM count.
    gen = torch.Generator().manual_seed(2193)
    base = {
        "q": (torch.randn(3, 64, 129, d, generator=gen) * 0.2).to(dtype).cuda(),
        "k": (torch.randn(3, 16, 257, d, generator=gen) * 0.2).to(dtype).cuda(),
        "v": (torch.randn(3, 16, 257, d, generator=gen) * 0.2).to(dtype).cuda(),
        "bias": (torch.randn(3, 64, 129, 272, generator=gen) * 0.7).to(torch.float32 if bias_fp32 else dtype).cuda(),
    }
    # Include an empty logical row and a non-tile-aligned key tail.
    base["bias"][:, :, 0, :257] = -torch.inf

    def build(order, *, changed_neighbors=False):
        index = torch.tensor(order, dtype=torch.int64, device="cuda")
        tensors = {name: value.index_select(0, index) for name, value in base.items()}
        if changed_neighbors:
            for i, source in enumerate(order):
                if source != 0:
                    for name in ("q", "k", "v"):
                        tensors[name][i].mul_(-0.5)
                    tensors["bias"][i].mul_(0.5)
        tensors["bias"] = tensors["bias"][..., :257]
        tensors["o"] = torch.empty_like(tensors["q"])
        tensors["lse"] = torch.empty(len(order), 64, 129, dtype=torch.float32, device="cuda")
        api = SdpaFwdDslSm90(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            tensors["o"],
            sample_lse=tensors["lse"],
            sample_bias=tensors["bias"],
            scale_softmax=d**-0.5,
            pack_gqa=False,
            is_causal=causal,
            tile_m=tile,
            tile_n=tile,
        )
        api.check_support()
        api.compile()
        assert api._dense_spec.has_bias
        assert api._k_mod.__file__.replace("\\", "/").endswith(kernel)

        def run():
            tensors["o"].fill_(torch.nan)
            tensors["lse"].fill_(torch.nan)
            api.execute(tensors["q"], tensors["k"], tensors["v"], tensors["o"], tensors["lse"], bias_tensor=tensors["bias"])
            return {name: tensors[name] for name in ("o", "lse")}

        return run, tensors

    run, tensors = build([0, 1, 2])
    original = assert_replays(run, repeats=4)
    # Independent FP64 reference, distinct from the bytewise comparisons.
    q, k, v = (tensors[name].double() for name in ("q", "k", "v"))
    scores = (q @ k.repeat_interleave(4, 1).transpose(-1, -2)) * d**-0.5 + tensors["bias"].double()
    if causal:
        # Square-free causal: top-left alignment on S_q = 129, S_kv = 257.
        scores.masked_fill_(torch.arange(257, device="cuda")[None, :] > torch.arange(129, device="cuda")[:, None], -torch.inf)
    reference = scores.softmax(-1).nan_to_num() @ v.repeat_interleave(4, 1)
    torch.testing.assert_close(original["o"].double(), reference, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(original["lse"].double(), scores.logsumexp(-1), atol=3e-4, rtol=3e-4)
    want = {name: value[:1] for name, value in original.items()}
    for order, changed in [([0], False), ([2, 0, 1], False), ([0, 0, 2, 1], True)]:
        fresh, fresh_tensors = build(order, changed_neighbors=changed)
        actual = assert_replays(fresh, repeats=3)
        for i, source in enumerate(order):
            if source == 0:
                assert_bitwise({name: value[i : i + 1] for name, value in actual.items()}, want, f"batch {order}, changed={changed}, index={i}")
        captured = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(captured, stream=stream):
            fresh()
        torch.cuda.current_stream().wait_stream(stream)
        captured.replay()
        assert_bitwise(snapshot({name: fresh_tensors[name] for name in ("o", "lse")}), actual, "captured replay")
        captured.reset()

    assert_golden(golden, {name: tensors[name] for name in ("q", "k", "v", "bias")}, original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("bias_fp32", [False, True], ids=["io-bias", "fp32-bias"])
@pytest.mark.parametrize("d,causal", [(128, False), (128, True), (256, True)], ids=["d128-dense", "d128-causal", "d256-causal"])
def test_sm90_bias_forward_determinism(dtype, bias_fp32, d, causal):
    """Engine 20540 (the 64 x 64 tile is pinned: the adapter takes the D128 template for an exact (128, 128) bias in the
    Q/K/V dtype otherwise)."""
    from cutlass.base_dsl.version_info import CUDA_VERSION

    _check_determinism(
        dtype,
        bias_fp32,
        d,
        causal,
        tile=64,
        kernel="/sm90/f16_three_wg/kernel.py",
        golden=f"sdpa_sm90_bias_fwd_{str(dtype).split('.')[-1]}_{'fp32' if bias_fp32 else 'io'}_d{d}_causal{int(causal)}_cuda{CUDA_VERSION.major}{CUDA_VERSION.minor}",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_sm90_d128_bias_forward_determinism(dtype, causal):
    """Engine 20541, the adapter's record (persistent for both: 3 x 64 heads x 2 Q tiles = 384 items, so every CTA
    walks at least two), bias in the Q/K/V dtype. Its 128-row Q tile holds 128 of the 129 rows, so the second tile has
    one live row; the 257 keys leave a one-key last KV tile."""
    from cutlass.base_dsl.version_info import CUDA_VERSION

    _check_determinism(
        dtype,
        False,
        128,
        causal,
        tile=128,
        kernel="/sm90/d128_f16_multi_q/kernel.py",
        golden=f"sdpa_sm90_d128_bias_fwd_{str(dtype).split('.')[-1]}_io_causal{int(causal)}_cuda{CUDA_VERSION.major}{CUDA_VERSION.minor}",
    )
