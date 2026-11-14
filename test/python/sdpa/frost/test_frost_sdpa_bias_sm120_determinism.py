# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D128 bias forward O/Stats repeatability and batch isolation; backward is unsupported.

Fixed plans are compared within one GPU/compiler/runtime. No cross-tile or
cross-runtime byte identity is claimed. The key tail refills the K/V pipeline.
"""

import pytest
import torch

from determinism_utils import assert_bitwise, assert_replays, assert_golden, snapshot
from frost_test_utils import requires_blackwell_geforce, requires_dsl

pytestmark = [pytest.mark.L0, requires_dsl, requires_blackwell_geforce]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("bias_fp32", [False, True], ids=["io-bias", "fp32-bias"])
@pytest.mark.parametrize("tile_m,causal", [(64, False), (128, True)], ids=["m64-dense", "m128-causal"])
def test_sm120_bias_forward_determinism(dtype, bias_fp32, tile_m, causal):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm120
    from cutlass.base_dsl.version_info import CUDA_VERSION

    # CPU seeding preserves inputs independently of the GPU's SM count.
    gen = torch.Generator().manual_seed(2193)
    base = {
        "q": (torch.randn(3, 64, 129, 128, generator=gen) * 0.2).to(dtype).cuda(),
        "k": (torch.randn(3, 16, 257, 128, generator=gen) * 0.2).to(dtype).cuda(),
        "v": (torch.randn(3, 16, 257, 128, generator=gen) * 0.2).to(dtype).cuda(),
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
        api = SdpaFwdDslSm120(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            tensors["o"],
            sample_lse=tensors["lse"],
            sample_bias=tensors["bias"],
            scale_softmax=128**-0.5,
            tile_m=tile_m,
            pack_gqa=False,
            is_causal=causal,
        )
        api.check_support()
        api.compile()
        assert api._dense_spec.has_bias
        assert api._k_mod.__file__.replace("\\", "/").endswith("/sm120/prefill_f16.py")

        def run():
            tensors["o"].fill_(torch.nan)
            tensors["lse"].fill_(torch.nan)
            api.execute(tensors["q"], tensors["k"], tensors["v"], tensors["o"], tensors["lse"], bias_tensor=tensors["bias"])
            return {name: tensors[name] for name in ("o", "lse")}

        return run, tensors

    run, tensors = build([0, 1, 2])
    original = assert_replays(run, repeats=4)
    # Independent FP64 reference is distinct from the bytewise comparisons.
    q, k, v = (tensors[name].double() for name in ("q", "k", "v"))
    scores = (q @ k.repeat_interleave(4, 1).transpose(-1, -2)) * 128**-0.5 + tensors["bias"].double()
    if causal:
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

    assert_golden(
        f"sdpa_sm120_bias_fwd_{str(dtype).split('.')[-1]}_{'fp32' if bias_fp32 else 'io'}_m{tile_m}_causal{int(causal)}_cuda{CUDA_VERSION.major}{CUDA_VERSION.minor}",
        {name: tensors[name] for name in ("q", "k", "v", "bias")},
        original,
    )
