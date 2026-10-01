# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent consumers must reacquire per-sequence maps on every launch."""

import pytest

torch = pytest.importorskip("torch")

from linear_attention.test_la import (
    FWD_TOL,
    STATE_TOL,
    assert_rms_close,
    backend,
    make_case,
    reference,
    run_fwd,
    window,
)

pytestmark = [pytest.mark.L0, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", ["gdn", "gdp"])
@pytest.mark.parametrize(
    "dtype,checkpoints",
    [(torch.bfloat16, False), (torch.float16, False), (torch.bfloat16, True)],
    ids=["bf16", "fp16", "bf16_checkpoints"],
)
def test_persistent_batch_maps_rebind_and_replay(backend, variant, dtype, checkpoints):
    # More heads than resident CTAs exercises repeated maps and sequence changes
    # in the same consumer. The zero-length sequence moves after capture.
    heads = max(256, 2 * torch.cuda.get_device_properties(0).multi_processor_count)
    # Bound the recurrent FP64 oracle under sanitizer while retaining a live
    # checkpoint boundary and more work per sequence than resident CTAs.
    lengths = [65, 0, 1, 1] if checkpoints else [129, 0, 65, 31]
    held_bindings = []
    for seed in (1313, 1314):
        case = make_case(variant, dtype, seq_lens=lengths, H=heads, K=64, V=64, seed=seed)
        held_bindings.append(case)  # Keep the previous allocation alive across rebind.

        def launch():
            kwargs = {"checkpoint_every_n_tokens": 64 * case.n} if checkpoints else {}
            return run_fwd(backend, case, output_final_state=True, **kwargs)

        @torch.no_grad()
        def check(outputs):
            expected = reference(case)
            for name, actual, want, tolerance in zip(("O", "final_state"), outputs, expected, (FWD_TOL[dtype], STATE_TOL[dtype])):
                assert torch.isfinite(actual).all(), name
                assert_rms_close(name, actual, want.reshape(actual.shape), tolerance)
            if checkpoints:
                row = 0
                bounds = case.cu.tolist()
                for start, end in zip(bounds, bounds[1:]):
                    for prefix in range(0, end - start, 64):
                        actual = outputs[2][row]
                        assert torch.isfinite(actual).all()
                        if prefix == 0:
                            assert not actual.any()
                        else:
                            _, state = reference(window(case, start, start + prefix))
                            assert_rms_close("checkpoint", actual, state[0], STATE_TOL[dtype])
                        row += 1

        check(launch())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = launch()
        changed_bounds = [0, 0, lengths[0], lengths[0] + lengths[3], sum(lengths)]
        for boundaries in (case.cu.tolist(), changed_bounds):
            case.cu.copy_(torch.tensor(boundaries, device="cuda", dtype=case.cu.dtype))
            case.q.mul_(0.75)
            case.v.neg_()
            for output in captured:
                output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check(captured)
        graph.reset()
