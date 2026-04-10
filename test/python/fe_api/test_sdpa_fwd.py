"""
Tests for SDPA forward SM100 API and wrapper.
"""

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.test_sdpa_fwd_utils import (
    allocate_sdpa_fwd_input_tensors,
    allocate_sdpa_fwd_output_tensors,
    check_ref_sdpa_fwd,
    sdpa_fwd_init,
    with_sdpa_fwd_params,
)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_sdpa_fwd_params
def test_sdpa_fwd_compile_execute(
    layout,
    dtype,
    is_causal,
    window_size,
    request,
):
    try:
        from cudnn.sdpa import SdpafwdSm100D256
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = sdpa_fwd_init(
        request=request,
        layout=layout,
        dtype=dtype,
        is_causal=is_causal,
        mma_tiler_mn=(128, 128),
        window_size=window_size,
    )
    inputs = allocate_sdpa_fwd_input_tensors(cfg)
    outputs = allocate_sdpa_fwd_output_tensors(cfg, inputs)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    sdpa_fwd = SdpafwdSm100D256(
        sample_q=inputs["q"],
        sample_k=inputs["k"],
        sample_v=inputs["v"],
        sample_o=outputs["o"],
        sample_lse=outputs["lse"],
        sample_cum_seqlen_q=inputs["cum_seqlen_q"],
        sample_cum_seqlen_k=inputs["cum_seqlen_kv"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        is_causal=cfg["is_causal"],
        window_size=cfg["window_size"],
    )

    try:
        assert sdpa_fwd.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    sdpa_fwd.compile()
    sdpa_fwd.execute(
        q_tensor=inputs["q"],
        k_tensor=inputs["k"],
        v_tensor=inputs["v"],
        o_tensor=outputs["o"],
        lse_tensor=outputs["lse"],
        cum_seqlen_q_tensor=inputs["cum_seqlen_q"],
        cum_seqlen_k_tensor=inputs["cum_seqlen_kv"],
        current_stream=stream,
    )
    torch.cuda.synchronize()

    check_ref_sdpa_fwd(
        cfg=cfg,
        inputs=inputs,
        outputs=outputs,
        skip_ref=cfg["skip_ref"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_sdpa_fwd_params
def test_sdpa_fwd_wrapper(
    layout,
    dtype,
    is_causal,
    window_size,
    request,
):
    try:
        from cudnn.sdpa import sdpa_fwd_wrapper_sm100_d256
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = sdpa_fwd_init(
        request=request,
        layout=layout,
        dtype=dtype,
        is_causal=is_causal,
        mma_tiler_mn=(128, 128),
        window_size=window_size,
    )
    inputs = allocate_sdpa_fwd_input_tensors(cfg)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    wrapper_outputs = None
    try:
        for _ in range(2):
            wrapper_outputs = sdpa_fwd_wrapper_sm100_d256(
                q_tensor=inputs["q"],
                k_tensor=inputs["k"],
                v_tensor=inputs["v"],
                cum_seqlen_q_tensor=inputs["cum_seqlen_q"],
                cum_seqlen_k_tensor=inputs["cum_seqlen_kv"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                is_causal=cfg["is_causal"],
                window_size=cfg["window_size"],
                current_stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    assert wrapper_outputs is not None
    torch.cuda.synchronize()

    check_ref_sdpa_fwd(
        cfg=cfg,
        inputs=inputs,
        outputs={
            "o": wrapper_outputs["o_tensor"],
            "lse": wrapper_outputs["lse_tensor"],
        },
        skip_ref=cfg["skip_ref"],
    )
