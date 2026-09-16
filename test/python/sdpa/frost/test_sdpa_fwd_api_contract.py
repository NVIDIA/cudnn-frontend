# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Architecture-neutral public API contract tests for FROST SDPA forward."""

import inspect
from pathlib import Path

import pytest
import torch

from cudnn.sdpa.fwd.api_dsl import SdpaFwdDsl, SdpaFwdDslSm80, SdpaFwdDslSm100, SdpaFwdDslSm120


@pytest.mark.L0
def test_new_constructor_parameters_follow_the_legacy_positional_prefix():
    """New optional operands must not change legacy positional bindings."""
    params = list(inspect.signature(SdpaFwdDsl.__init__).parameters)
    legacy_tail = [
        "paged_table_stride",
        "paged_table_v_stride",
        "thd_stats_padded",
    ]
    assert params[params.index("paged_table_stride") : params.index("thd_stats_padded") + 1] == legacy_tail
    extension_start = params.index("sample_amax_o")
    assert params[extension_start : extension_start + 3] == ["sample_amax_o", "pv_bf16", "stats_log2"]
    assert inspect.signature(SdpaFwdDsl.__init__).parameters["stats_log2"].default is False


@pytest.mark.L0
@pytest.mark.parametrize(
    ("api_cls", "device_cc"),
    [
        (SdpaFwdDslSm80, (8, 0)),
        (SdpaFwdDslSm100, (10, 7)),
        (SdpaFwdDslSm120, (12, 0)),
    ],
    ids=["sm80", "sm107", "sm120"],
)
def test_pv_bf16_rejects_unsupported_architectures(monkeypatch, api_cls, device_cc):
    """Only the pre-Rubin SM100 implementation may accept hybrid PV BF16."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: device_cc)
    q = torch.empty((1, 4, 128, 128), dtype=torch.float8_e4m3fn, device="cuda")
    k = torch.empty((1, 2, 128, 128), dtype=torch.float8_e4m3fn, device="cuda")
    v = torch.empty((1, 2, 128, 128), dtype=torch.bfloat16, device="cuda")
    o = torch.empty((1, 4, 128, 128), dtype=torch.bfloat16, device="cuda")
    api = api_cls(q, k, v, o, dtype_o=torch.bfloat16, pv_bf16=True)

    with pytest.raises(NotImplementedError, match="pre-Rubin SM100"):
        api.check_support()


@pytest.mark.L0
def test_d192_hybrid_elides_pv_scale_factor_storage_and_transactions():
    """The hybrid D192 specialization must compile out unused P/V SF traffic."""
    kernel = Path(__file__).parents[4] / "python/cudnn/sdpa/fwd/kernels/sm100/prefill_d192_d128_mxfp8.py"
    source = kernel.read_text()

    assert "sP_SF = sQ_SF" in source
    assert "sV_SF = sK_SF" in source
    assert "V_SF_EXPECT_BYTES = 0 if CFG.PV_BF16 else" in source


@pytest.mark.L0
def test_d256_hybrid_uses_bf16_pv_and_elides_scale_factor_traffic():
    """D256 must keep the non-causal block-scaled operator out of BMM2."""
    kernel = Path(__file__).parents[4] / "python/cudnn/sdpa/fwd/kernels/sm100/prefill_d256_mxfp8.py"
    source = kernel.read_text()

    assert "V_STORAGE_DTYPE = cutlass.BFloat16 if CFG.PV_BF16 else STORAGE_DTYPE" in source
    assert "P_STORAGE_DTYPE = cutlass.BFloat16 if CFG.PV_BF16 else STORAGE_DTYPE" in source
    assert "if cutlass.const_expr(CFG.PV_BF16):" in source
    assert "kind=nvvm.Tcgen05MMAKind.F16" in source
    assert "sP_SF = sQ_SF" in source
    assert "sV_SF = sK_SF" in source
    assert "V_SF_EXPECT_BYTES = 0 if CFG.PV_BF16 else" in source


@pytest.mark.L0
def test_hybrid_execute_uses_cached_v_scale_factor_dummy():
    """Hybrid execution must not materialize a sliced SF tensor per launch."""
    source = inspect.getsource(SdpaFwdDslSm100._execute_mxfp8)

    assert "sf_k_v[..., : km.SF_SMEM_SIZE_V].contiguous()" not in source
    assert 'f"pv_bf16_sf_v_{b}_{h_kv}_{n_kv_tiles}_{km.SF_SMEM_SIZE_V}"' in source
