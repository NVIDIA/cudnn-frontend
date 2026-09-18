# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level contracts for the leakage-safe SM100 MXFP8 backward path."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.L0


_ROOT = Path(__file__).parents[4]
_DQ = _ROOT / "python/cudnn/sdpa/bwd/kernels/bprop_dq_d256_mxfp8_sm100.py"
_DKDV = _ROOT / "python/cudnn/sdpa/bwd/kernels/bprop_dkdv_d256_mxfp8_sm100.py"
_API = _ROOT / "python/cudnn/sdpa/bwd/api_dsl_mxfp8_sm100.py"
_BWD_TEST = _ROOT / "test/python/sdpa/frost/test_sdpa_bwd_mxfp8_sm100.py"


def test_d256_dq_sequence_reduction_is_bf16():
    source = _DQ.read_text()
    mma = source[source.index("# compute dQ - using self.cta_group") : source.index("cluster_layout_vmnk =")]

    assert "dSK_tiled_mma = sm100_utils.make_trivial_tiled_mma(" in mma
    assert "cutlass.BFloat16" in mma
    assert "make_blockscaled_trivial_tiled_mma" not in mma

    assert "K_f16 = make_kv_head_batch_tensor(K_f16" in source
    assert "KT = make_transposed_tensor(K_f16" in source
    assert "MemRange[cutlass.BFloat16, cute.cosize(KT_smem_layout_staged)]" in source
    assert "MemRange[cutlass.BFloat16, cute.cosize(dS_smem_layout_staged)]" in source
    assert "dSK_tiled_mma.set(tcgen05.Field.SFA" not in source
    assert "dSK_tiled_mma.set(tcgen05.Field.SFB" not in source


def test_d256_dq_does_not_repack_or_load_columnwise_k_scale():
    source = _DQ.read_text()
    api = _API.read_text()

    for obsolete in (
        "SF_KT",
        "SFK_mn",
        "sfK_mn",
        "sfkmn",
        "dSK_tiled_mma_sfb",
        "dSK_mma_tiler_sfb",
        "tma_copy_sfK_mn",
        '"dq_sf_kt"',
    ):
        assert obsolete not in source

    assert '("dq_sf_kt", "sf_k_T"' not in api
    assert 'sf_bufs["dq_sf_kt"]' not in api

    # The row-wise K scales still feed the Q @ K block-scaled MMA.
    for required in ("sSFK_smem_layout_staged", "tma_atom_sfK", "tma_copy_sfK_bytes", "tSTtSFK"):
        assert required in source


def test_d256_dq_uses_independent_two_stage_k_and_one_stage_aux_pipelines():
    source = _DQ.read_text()
    setup_begin = source.index("def _setup_pipeline_stages_and_sf_tilers")
    setup = source[setup_begin : source.index("@cute.jit", setup_begin)]
    load = source[source.index("    def load(") : source.index("    def sfv_s2t_helper(")]
    mma_begin = source.index("    def mma_interleaved(")
    mma = source[mma_begin : source.index("    def mma(", mma_begin)]
    helper = source[source.index("    def sfv_s2t_helper(") : source.index("    def mma_interleaved(")]
    pipeline_helpers = source[source.index("    def make_and_init_load_mma_K_pipeline") :]

    assert "self.load_mma_K_stage = 2" in setup
    assert "self.load_mma_aux_stage = 1" in setup
    assert "self.KT_load_mma_K_stage = self.load_mma_aux_stage" in setup
    assert "self.SFV_load_mma_K_stage = self.load_mma_aux_stage * self.k_halves" in setup
    assert "load_mma_aux_mbar_ptr" in source
    assert "make_and_init_load_mma_aux_pipeline" in source
    assert "2 * self.tma_copy_K_bytes + sfk_tx_multiplier * self.tma_copy_sfK_bytes" in pipeline_helpers
    assert "kt_tx_multiplier * self.tma_copy_KT_bytes" in pipeline_helpers
    assert "+ 2 * self.tma_copy_V_bytes" in pipeline_helpers
    assert "+ sfv_tx_multiplier * self.tma_copy_sfV_bytes" in pipeline_helpers

    for ring in ("K", "aux"):
        assert f"load_mma_{ring}_producer_state" in load
        assert f"cumulative_trip_count % Int32(2 * self.load_mma_{ring}_stage)" in load
        assert f"load_mma_{ring}_pipeline.sync_object_full.get_barrier" in load
        assert f"load_mma_{ring}_consumer_state" in mma
        assert f"load_mma_{ring}_pipeline.consumer_wait" in mma
        assert f"load_mma_{ring}_pipeline.consumer_release" in mma
        assert f"load_mma_{ring}_consumer_state.advance()" in mma

    assert "tma_barrier_K_inner" in load
    assert "tma_barrier_aux_inner" in load
    assert "tKsK[None, k_stage]" in load
    assert "tKTsKT[None, aux_stage]" in load
    assert "tVsV[None, aux_stage]" in load
    refill = load[load.index("        while iter_count > 0:") :]
    assert refill.index("tKsK[None, k_stage]") < refill.index("for sfk_k_half")
    assert refill.index("for sfk_k_half") < refill.index("load_mma_K_producer_state.advance()")
    assert refill.index("load_mma_K_producer_state.advance()") < refill.index("load_mma_aux_pipeline.producer_acquire")
    assert refill.index("load_mma_aux_pipeline.producer_acquire") < refill.index("tKTsKT[None, aux_stage]")
    assert "load_mma_aux_pipeline.consumer_wait" in helper


def test_d256_dkdv_sequence_reductions_are_bf16():
    source = _DKDV.read_text()
    mma_begin = source.index("# dK = dS @ Q")
    mma = source[mma_begin : source.index("cluster_layout_vmnk =", mma_begin)]

    for name in ("dSQ_tiled_mma", "PdO_tiled_mma"):
        assert f"{name} = sm100_utils.make_trivial_tiled_mma(" in mma
    assert mma.count("cutlass.BFloat16") >= 2
    assert "make_blockscaled_trivial_tiled_mma" not in mma

    assert "QT = make_transposed_tensor(Q_16bits" in source
    assert "dOT = make_transposed_tensor(dO_16bits" in source
    assert "MemRange[cutlass.BFloat16, cute.cosize(dS_smem_layout_staged)]" in source
    assert "MemRange[cutlass.BFloat16, cute.cosize(P_smem_layout_staged)]" in source
    assert "dSQ_tiled_mma.set(tcgen05.Field.SFA" not in source
    assert "dSQ_tiled_mma.set(tcgen05.Field.SFB" not in source
    assert "PdO_tiled_mma.set(tcgen05.Field.SFA" not in source
    assert "PdO_tiled_mma.set(tcgen05.Field.SFB" not in source

    for obsolete in (
        "Q_MN",
        "dO_MN",
        "SF_QT",
        "SF_DOT",
        "dSQ_tiled_mma_sfb",
        "PdO_tiled_mma_sfb",
        "SFQ_mn",
        "SFDO_mn",
        "sDS_scale_exchange",
    ):
        assert obsolete not in source


def test_qwen_leakage_probe_does_not_compile_the_test_quantizer():
    source = _BWD_TEST.read_text()
    helper = source[
        source.index("def _run_leakage_only(") : source.index(
            "# --------------------------------------------------------------------------- #", source.index("def _run_leakage_only(")
        )
    ]
    test = source[source.index("def test_causal_sequence_reductions_do_not_leak_masked_rows") : source.index("def test_gqa")]

    assert "_quantize(" not in helper
    assert "quantize_to_mxfp8" not in helper
    assert "_run_leakage_only(" in test
    assert "hq=8" in test and "hkv=1" in test
