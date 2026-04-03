"""
Tests for SDPA backward SM100 API and wrapper.
"""

import pytest
import torch

from test_utils import torch_fork_set_rng
from fe_api.test_sdpa_bwd_utils import (
    allocate_sdpa_bwd_input_tensors,
    allocate_sdpa_bwd_output_tensors,
    check_ref_sdpa_bwd,
    sdpa_bwd_init,
    with_sdpa_bwd_params,
)


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_sdpa_bwd_params
def test_sdpa_bwd_compile_execute(
    layout,
    dtype,
    is_causal,
    window_size,
    request,
):
    try:
        from cudnn.sdpa import SdpabwdSm100D256
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = sdpa_bwd_init(
        request=request,
        layout=layout,
        dtype=dtype,
        is_causal=is_causal,
        mma_tiler_mn=(128, 128),
        dkdv_mma_tiler_mn=(128, 64),
        window_size=window_size,
    )
    inputs = allocate_sdpa_bwd_input_tensors(cfg)
    outputs = allocate_sdpa_bwd_output_tensors(cfg)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    sdpa_bwd = SdpabwdSm100D256(
        sample_q=inputs["q"],
        sample_k=inputs["k"],
        sample_v=inputs["v"],
        sample_o=inputs["o"],
        sample_do=inputs["do"],
        sample_lse=inputs["lse"],
        sample_dq=outputs["dq"],
        sample_dk=outputs["dk"],
        sample_dv=outputs["dv"],
        sample_cum_seqlen_q=inputs["cum_seqlen_q"],
        sample_cum_seqlen_k=inputs["cum_seqlen_kv"],
        mma_tiler_mn=cfg["mma_tiler_mn"],
        dkdv_mma_tiler_mn=cfg["dkdv_mma_tiler_mn"],
        is_causal=cfg["is_causal"],
        window_size=cfg["window_size"],
    )

    try:
        assert sdpa_bwd.check_support(), "Unsupported testcase"
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    sdpa_bwd.compile()
    sdpa_bwd.execute(
        q_tensor=inputs["q"],
        k_tensor=inputs["k"],
        v_tensor=inputs["v"],
        o_tensor=inputs["o"],
        do_tensor=inputs["do"],
        lse_tensor=inputs["lse"],
        dq_tensor=outputs["dq"],
        dk_tensor=outputs["dk"],
        dv_tensor=outputs["dv"],
        cum_seqlen_q_tensor=inputs["cum_seqlen_q"],
        cum_seqlen_k_tensor=inputs["cum_seqlen_kv"],
        current_stream=stream,
    )
    torch.cuda.synchronize()

    check_ref_sdpa_bwd(
        cfg=cfg,
        inputs=inputs,
        outputs=outputs,
        skip_ref=cfg["skip_ref"],
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_sdpa_bwd_params
def test_sdpa_bwd_wrapper(
    layout,
    dtype,
    is_causal,
    window_size,
    request,
):
    try:
        from cudnn.sdpa import sdpa_bwd_wrapper_sm100_d256
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn optional dependencies not installed")

    cfg = sdpa_bwd_init(
        request=request,
        layout=layout,
        dtype=dtype,
        is_causal=is_causal,
        mma_tiler_mn=(128, 128),
        dkdv_mma_tiler_mn=(128, 64),
        window_size=window_size,
    )
    inputs = allocate_sdpa_bwd_input_tensors(cfg)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Run twice to exercise wrapper cache path.
    wrapper_outputs = None
    try:
        for _ in range(2):
            wrapper_outputs = sdpa_bwd_wrapper_sm100_d256(
                q_tensor=inputs["q"],
                k_tensor=inputs["k"],
                v_tensor=inputs["v"],
                o_tensor=inputs["o"],
                do_tensor=inputs["do"],
                lse_tensor=inputs["lse"],
                cum_seqlen_q_tensor=inputs["cum_seqlen_q"],
                cum_seqlen_k_tensor=inputs["cum_seqlen_kv"],
                mma_tiler_mn=cfg["mma_tiler_mn"],
                dkdv_mma_tiler_mn=cfg["dkdv_mma_tiler_mn"],
                is_causal=cfg["is_causal"],
                window_size=cfg["window_size"],
                current_stream=stream,
            )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    assert wrapper_outputs is not None
    torch.cuda.synchronize()

    check_ref_sdpa_bwd(
        cfg=cfg,
        inputs=inputs,
        outputs={
            "dq": wrapper_outputs["dq_tensor"],
            "dk": wrapper_outputs["dk_tensor"],
            "dv": wrapper_outputs["dv_tensor"],
        },
        skip_ref=cfg["skip_ref"],
    )
