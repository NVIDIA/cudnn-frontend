# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A cached wgrad plan must acquire descriptors rewritten between graph nodes."""

import pytest
import torch
import cudnn


@pytest.mark.L0
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("accumulate", [False, True])
def test_wgrad_tma_descriptor_reuse(discrete, ragged, accumulate):
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Exercises the SM100 blockscaled wgrad kernel.")

    stream = torch.cuda.Stream()
    calls, outputs = [], []
    # Identical signatures reuse one plan/workspace, but the device helper must
    # rebind its descriptors to distinct input, scale and output addresses.
    for value, scale_byte, tokens in ((1, 127, (128, 128, 0, 256)), (2, 128, (256, 0, 128, 128))):
        a = torch.full((512, 256), value, device="cuda").to(torch.float8_e4m3fn).T
        b = torch.ones((512, 256), device="cuda").to(torch.float8_e4m3fn)
        # Inputs are constant, so their physical layout is also valid in ragged mode.
        sa = torch.full((256, 16), scale_byte, device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
        sb = sa.clone()
        offsets = torch.tensor(tokens, device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
        if discrete:
            dest = [torch.empty((256, 256), device="cuda", dtype=torch.bfloat16) for _ in tokens]
            pointers = torch.tensor([out.data_ptr() for out in dest], device="cuda", dtype=torch.int64)
            output_args = dict(output_mode="discrete", wgrad_ptrs=pointers)
        else:
            dense = torch.empty((4, 256, 256), device="cuda", dtype=torch.bfloat16)
            dest = list(dense.unbind())
            output_args = dict(output_mode="dense", wgrad_tensor=dense)
        calls.append(
            dict(
                a_tensor=a,
                b_tensor=b,
                sfa_tensor=sa,
                sfb_tensor=sb,
                offsets_tensor=offsets,
                **output_args,
                wgrad_dtype=torch.bfloat16,
                acc_dtype=torch.float32,
                sf_vec_size=32,
                input_order="tensor_ragged" if ragged else "tensor2d",
                accumulate_on_output=accumulate,
                current_stream=stream.cuda_stream,
            )
        )
        scale = 2 ** (scale_byte - 127)
        outputs.extend((out, n * value * scale * scale + int(accumulate)) for out, n in zip(dest, tokens))

    def run():
        for kwargs in calls:
            cudnn.grouped_gemm_wgrad_wrapper_sm100(**kwargs)

    def check(fn):
        # Reset outside capture on every replay: stale warmup outputs must not
        # hide a missing store, and accumulation always starts from a known value.
        for out, _ in outputs:
            out.fill_(1 if accumulate else -1)
        fn()
        for out, expected in outputs:
            torch.testing.assert_close(out, torch.full_like(out, expected), rtol=0, atol=0)

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            check(run)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for _ in range(8):
            check(graph.replay)
    torch.cuda.current_stream().wait_stream(stream)
