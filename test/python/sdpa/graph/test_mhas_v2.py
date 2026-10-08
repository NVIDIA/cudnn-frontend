# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This script tests cuDNN front-end attention.
The recommended way to run tests:
> pytest -vv -s -rA test_mhas_v2.py
"""

import cudnn
import pytest
import contextlib
import random
import torch
import sys
import os
from datetime import datetime

from sdpa.random_config import (
    ExecConfig,
    generate_test_seeds,
    RandomizationContext,
    RandomBatchSize,
    RandomBlockSize,
    RandomSequenceLength,
    RandomHiddenDimSize,
    RandomHeadGenerator,
    RandomChoice,
    SlidingWindowMaskGenerator,
    get_strides_from_layout,
)
from sdpa.fp16 import exec_sdpa
from sdpa.fp8 import exec_sdpa_fp8
from sdpa.mxfp8 import exec_sdpa_mxfp8
from sdpa.blocked import fetch_blocked_tests
from sdpa.helpers import print_section_begin, print_section_end
from sdpa.softmax_knobs import served_softmax_knob_sets, knob_set_for_case, knob_set_label, is_default_knob_set
import frost_routing

# fmt: off

if __name__ == "__main__":
    print("This is pytest script.")
    sys.exit(0)

class SDPATestConfig:
    __slots__ = ['gpu_arch', 'gpu_info', 'cudnn_ver', 'blocked_tests', 'implementation', 'cfg']

    def __init__(self, *, gpu_arch, gpu_info, cudnn_ver, blocked_tests, implementation):
        assert type(gpu_arch) == type(gpu_info) == type(cudnn_ver) == str, "expecting strings as arguments"
        assert isinstance(blocked_tests, list), "argument 'blocked_tests' must be list"

        # Initialize all attributes to None.
        for k in self.__slots__:
            setattr(self, k, None)

        self.gpu_arch      = gpu_arch
        self.gpu_info      = gpu_info
        self.cudnn_ver     = cudnn_ver
        self.blocked_tests = blocked_tests

        self.implementation = implementation

        self.cfg = ExecConfig()


    def showConfig(self, test_no, request):
        is_dryrun = request.config.option.dryrun
        print()
        print_section_begin("DRY-RUN" if is_dryrun else "")
        print(f"#### Test #{test_no[0]} of {test_no[1]} at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")
        print(f"test_name        = {request.node.name}")
        print(f"platform_info    = {self.gpu_arch} ({self.gpu_info}), cudnn_ver={self.cudnn_ver}")
        print()
        print(self.cfg.to_repro_cmd(request.module.__file__))
        print(flush=True)


@pytest.fixture(scope="package")
def env_info(request):
    assert torch.cuda.is_available(), "no CUDA device"

    gpu_type = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    device   = torch.device('cuda:0')
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    gpu_arch     = f"SM_{gpu_type[0]}{gpu_type[1]}"
    gpu_info     = f"{sm_count} SM-s, {gpu_name}"
    cudnn_ver    = str(torch.backends.cudnn.version())

    blocked_tests = fetch_blocked_tests(gpu_arch, cudnn_ver)

    return {"gpu_arch": gpu_arch, "gpu_info": gpu_info, "cudnn_ver": cudnn_ver, "blocked_tests": blocked_tests}

# These options are common to all test lists
data_type_options      = {torch.float16 : 1, torch.bfloat16 : 2}
diag_alignment_options = [cudnn.diagonal_alignment.TOP_LEFT, cudnn.diagonal_alignment.BOTTOM_RIGHT]
implementation_options = [cudnn.attention_implementation.AUTO, cudnn.attention_implementation.COMPOSITE, cudnn.attention_implementation.UNIFIED]
implementation_names   = ['cudnn.attention_implementation.AUTO', 'cudnn.attention_implementation.COMPOSITE', 'cudnn.attention_implementation.UNIFIED']

# sink_token in the s_q == 1 sweeps: the FROST SM100 f16/bf16 engine serves it (dense and
# paged; FROST engines are opt-in via CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1), the cuDNN
# backend engines decline it (C++ support surface), and exec_sdpa can only report that
# decline as a WAIVED skip -- so the sweeps draw the sink only when FROST is on. Both arms
# carry the same total weight (4), i.e. the same randint(0, 3) call: the two modes consume
# the rng identically and a config differs between them in the sink flag alone (CPython's
# randint rejection-samples 32-bit words, so the total weight, not the number of options,
# fixes the consumption; checked identical over 20000 seeds).
def _frost_engines_enabled(engine="sdpa_fwd_prefill_sm100"):
    # Whether the manifest OFFERS the row: the SM100/SM120 f16 rows are default
    # candidates (placed per measured shard), the others still answer to
    # CUDNN_FRONTEND_ENABLE_FROST_ENGINES.
    from cudnn.engines.manifest import MANIFEST
    fam = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd")
    return engine in fam.offered_ids()

def _frost_sm100_unavailable_reason(engine="sdpa_fwd_prefill_sm100"):
    """Why the FROST SM100 f16/bf16 row would NOT serve a graph here, or None when it
    would: the engine must be offered by the manifest, the device a pre-Rubin Blackwell (cc 10.0-10.6,
    the row's arch domain) and a CuTe DSL at the FROST floor importable (the row declines
    without one and the native backend then serves the graph)."""
    if not _frost_engines_enabled(engine):
        return f"{engine} is not offered by the manifest (opt-in row without CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1)"
    major, minor = torch.cuda.get_device_capability()
    if not (100 <= major * 10 + minor <= 106):
        return f"{engine} serves cc 10.0-10.6 only; device is cc {major}.{minor}"
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        return "needs the cutedsl extra (nvidia-cutlass-dsl) at the FROST floor"
    return None

def _sq1_sink_token():
    # Draw the sink only where the FROST SM100 row can serve it (opt-in AND arch AND DSL,
    # not the opt-in alone): elsewhere a sink at s_q == 1 is declined by every engine and
    # exec_sdpa could only report it as a WAIVED skip, silently losing the draw.
    if _frost_sm100_unavailable_reason() is None:
        return RandomChoice({True : 1, False : 3})
    return RandomChoice({False : 4})

# # ==================================
# # L0 fprop tests
# # ==================================
@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=512, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=256, d_v_min=1, d_v_max=256, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128), (256, 256)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        with_sink_token=RandomChoice({True : 1, False : 3}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L1
def test_sdpa_random_fwd_unified_L1(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=256, d_v_min=1, d_v_max=256, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128), (256, 256)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_bias=RandomChoice({True : 1, False : 3}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "cu_padded" : 1, "full" : 1}),
        with_unfuse_fma=RandomChoice({True : 1, False : 1}),  # Randomly enable unfuse_fma for SM100
        with_sink_token=RandomChoice({True : 1, False : 3}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 bprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=512, rng_seed=844), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_bwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=8, max=16),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=512, d_v_min=1, d_v_max=512, head_dim_distribution={"d_qk=d_v":5, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128), (256,256), (512,512)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 4, "full" : 1}),
        is_deterministic=RandomChoice({True : 3, False : 1}),
        with_sink_token=RandomChoice({True : 1, False : 3}),
        with_stats_log2=RandomChoice({True : 1, False : 3}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 fprop tests with s_q=1
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=256, rng_seed=111), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_sq1_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":100, "s_q=s_kv":1, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        # ragged = packed-THD decode (the serving shape); was never drawn here,
        # which is how the d=192 THD-decode view overflow (GitHub #980) hid.
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 1, "full" : 1}),
        with_sink_token=_sq1_sink_token(),
        # dropout not supported with s_q==1
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=111), ids=lambda p: f"test{p[0]}")
@pytest.mark.L1
def test_sdpa_random_sq1_unified_L1(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":100, "s_q=s_kv":1, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        with_sink_token=_sq1_sink_token(),
        # dropout not supported with s_q==1
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# fmt: on


@pytest.mark.L0
@pytest.mark.skipif(cudnn.backend_version() < 92400, reason="ragged offset multiplier requires cuDNN >= 9.24")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("offset_dtype,use_multiplier", [(torch.int32, True), (torch.int64, False)], ids=["int32_tokens", "int64_elements"])
@pytest.mark.parametrize(
    "s_q,h_q,ragged_stats",
    [(1, 8, True), (1, 8, False), (1, 2, True), (2, 8, True)],
    ids=["decode_gqa_ragged", "decode_gqa_padded_stats", "decode_mha_ragged", "prefill_gqa_ragged"],
)
def test_sdpa_ragged_decode_stats(cudnn_handle, request, dtype, offset_dtype, use_multiplier, s_q, h_q, ragged_stats):
    """Single-token GQA must write every head's LSE, even when O is correct."""
    from cudnn.engines.engine_ids import is_backend_engine

    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("unified SDPA requires SM80 or newer")

    b, h_kv, d = 3, 2, 128
    kv_lengths = [128, 256, 128]
    rng = torch.Generator(device="cuda").manual_seed(6783545)
    q_gpu = torch.randn(b * s_q, h_q, d, device="cuda", dtype=dtype, generator=rng)
    k_gpu = torch.randn(sum(kv_lengths), h_kv, d, device="cuda", dtype=dtype, generator=rng)
    v_gpu = torch.randn(k_gpu.shape, device="cuda", dtype=dtype, generator=rng)
    o_gpu = torch.full_like(q_gpu, float("nan"))
    stats_gpu = torch.full((b, s_q, h_q, 1), float("nan"), device="cuda").transpose(1, 2)
    cu_q_gpu = torch.arange(b + 1, dtype=torch.int32, device="cuda") * s_q
    cu_kv_gpu = torch.tensor([0, 128, 384, 512], dtype=torch.int32, device="cuda")

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )
    pack = {}

    def offset(cu, token_stride):
        data = cu.to(offset_dtype) * (1 if use_multiplier else token_stride)
        desc = graph.tensor_like(data)
        pack[desc] = data
        return desc

    def packed_tensor(data, heads, max_seq, cu):
        desc = graph.tensor(dim=[b, heads, max_seq, d], stride=[max_seq * heads * d, d, heads * d, 1])
        desc.set_ragged_offset(offset(cu, heads * d))
        if use_multiplier:
            desc.set_ragged_offset_multiplier(heads * d)
        pack[desc] = data
        return desc

    q = packed_tensor(q_gpu, h_q, s_q, cu_q_gpu)
    k = packed_tensor(k_gpu, h_kv, max(kv_lengths), cu_kv_gpu)
    v = packed_tensor(v_gpu, h_kv, max(kv_lengths), cu_kv_gpu)
    cu_q, cu_kv = graph.tensor_like(cu_q_gpu), graph.tensor_like(cu_kv_gpu)
    pack.update({cu_q: cu_q_gpu, cu_kv: cu_kv_gpu})
    o, stats = graph.sdpa(
        q=q,
        k=k,
        v=v,
        generate_stats=True,
        attn_scale=d**-0.5,
        use_padding_mask=True,
        cu_seq_len_q=cu_q,
        cu_seq_len_kv=cu_kv,
        implementation=cudnn.attention_implementation.UNIFIED,
    )
    o.set_output(True).set_dim([b, h_q, s_q, d]).set_stride([s_q * h_q * d, d, h_q * d, 1])
    o.set_ragged_offset(offset(cu_q_gpu, h_q * d))
    if use_multiplier:
        o.set_ragged_offset_multiplier(h_q * d)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT).set_dim(stats_gpu.shape).set_stride(stats_gpu.stride())
    if ragged_stats:
        stats.set_ragged_offset(offset(cu_q_gpu, h_q))
        if use_multiplier:
            stats.set_ragged_offset_multiplier(h_q)
    pack.update({o: o_gpu, stats: stats_gpu})

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    # Pin a backend plan: a FROST pass cannot prove this native codegen regression.
    backend_plans = [i for i in range(graph.get_execution_plan_count()) if is_backend_engine(graph.get_engine_and_knobs_at_index(i)[0])]
    if not backend_plans:
        pytest.skip("no unified backend plan on this device")
    graph.select_plan(backend_plans[0])
    graph.check_support()
    print("Ragged Stats backend plan:", graph.get_plan_name_at_index(backend_plans[0]))
    selected_engine, _ = graph.get_engine_and_knobs_at_index(backend_plans[0])
    if torch.cuda.get_device_capability() == (10, 7) and s_q == 1 and selected_engine in (10, 18):
        # NVBug 6813175 affects the native 10X/107 engines (global indices 10/18): their ragged
        # decode codegen emits a TMEM Stats round-trip wider than the ISA allows. The heuristic
        # may instead select the working eng8 plan; that must run without this marker. A native
        # 10/18 plan that builds successfully remains a strict XPASS so its fix is noticed.
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=cudnn.cudnnGraphNotSupportedError,
                reason="Selected native 10X/107 ragged-decode plan fails NVRTC (NVBug 6813175)",
            )
        )
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()  # Inputs were created on the torch stream; the fixture handle owns another stream.
    graph.execute(pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # Independent fp64 reference; check O as well as every q head's Stats.
    kv_start = 0
    stats_refs = []
    for batch, kv_len in enumerate(kv_lengths):
        q_ref = q_gpu[batch * s_q : (batch + 1) * s_q].double().transpose(0, 1)
        k_ref = k_gpu[kv_start : kv_start + kv_len].double().transpose(0, 1).repeat_interleave(h_q // h_kv, dim=0)
        v_ref = v_gpu[kv_start : kv_start + kv_len].double().transpose(0, 1).repeat_interleave(h_q // h_kv, dim=0)
        scores = (q_ref @ k_ref.transpose(-1, -2)) * d**-0.5
        o_ref = (scores.softmax(-1) @ v_ref).transpose(0, 1).to(dtype)
        torch.testing.assert_close(o_gpu[batch * s_q : (batch + 1) * s_q], o_ref, atol=5e-3, rtol=1e-2)
        stats_refs.append(scores.logsumexp(-1).float().unsqueeze(-1))
        kv_start += kv_len

    # Known issue on older backends (NVBug 6783545). Register only AFTER all O
    # checks, so an unrelated plan/build/output failure is never an expected failure.
    # A backport to an older version is an XPASS that asks us to retire this marker.
    if cudnn.backend_version() < 92700 and s_q == 1 and h_q > h_kv and ragged_stats:
        request.node.add_marker(pytest.mark.xfail(strict=True, raises=AssertionError, reason="cuDNN < 9.27: ragged decode GQA Stats (NVBug 6783545)"))
    torch.testing.assert_close(stats_gpu, torch.stack(stats_refs), atol=1e-4, rtol=1e-4)


# fmt: off

# # =====================================================
# # L0 lean attention, s_kv=513..4096
# # =====================================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=256, rng_seed=222), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_lean_attn_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=513, s_kv_max=8192, s_q_distribution={"s_q=1":100, "s_q=s_kv":0, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        # ragged = packed-THD decode against a long KV (GitHub #980 coverage).
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 1, "full" : 1}),
        with_sink_token=_sq1_sink_token(),
        # dropout not supported with s_q==1
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=222), ids=lambda p: f"test{p[0]}")
@pytest.mark.L1
def test_sdpa_random_lean_attn_unified_L1(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=513, s_kv_max=8192, s_q_distribution={"s_q=1":100, "s_q=s_kv":0, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        with_sink_token=_sq1_sink_token(),
        # dropout not supported with s_q==1
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)

# # =====================================================================
# # L0 paged decode / MTP, GQA groups that do not divide the 128-row tile
# # (FROST partial PackGQA: G=12 packs 4, G=6 packs 2, G=3 / G=5 unpacked)
# # =====================================================================

def _require_frost_sm100(engine="sdpa_fwd_prefill_sm100"):
    """The functions below ASSERT that a FROST engine served the graph, so they
    run only where that engine is offered: a pre-Rubin Blackwell (cc 10.0-10.6)
    with the FROST engines opted in (read the frontend's way, as the s_q == 1
    sweeps' sink draw does -- _frost_engines_enabled) and a usable CuTe DSL (the
    row declines without one and the native backend could then serve the graph,
    which the routing assertion must not count as a failure of FROST).
    Elsewhere they skip instead of failing. Same prerequisites as the s_q == 1
    sweeps' sink draw (_frost_sm100_unavailable_reason)."""
    reason = _frost_sm100_unavailable_reason(engine)
    if reason is not None:
        pytest.skip(f"{reason}: this test asserts FROST routing")


def _require_frost_fp8_paged_leads():
    """Paged FP8 is backend-first by placement (placement._place_sm100_fp8), so the
    default walk reaches the FROST FP8 row only with the opt-in flag, which ranks it first."""
    _require_frost_sm100(FROST_FP8_ENGINE)
    from cudnn.engines.manifest import opt_in_engines_enabled

    if not opt_in_engines_enabled():
        pytest.skip("paged FP8 is backend-first without CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1: this test asserts FROST routing")


def _exec_sdpa_on_frost(cfg, request, cudnn_handle, engine="sdpa_fwd_prefill_sm100", cga=None, template=None, tensor_checker=None):
    """exec_sdpa, then assert the FROST engine served the graph: the harness
    tallies the serving engine in frost_routing after build_plans, and a FROST
    decline silently falls through to the native backend, so a green run alone
    proves nothing about routing.  (A WAIVED skip inside exec_sdpa skips before
    the assertion.)  Two optional pins name WHICH plan of that engine served it:
    ``cga`` pins the selected plan's TILE_CGA_M knob (frost_routing.LAST_PLAN):
    on sdpa_fwd_prefill_sm100's d128 flavor 1 IS the decode tile and 2 the
    prefill pipeline, so a test that means the d128 decode tile asserts the
    tile, not just the engine; ``template`` pins the kernel template the plan
    lowered onto -- the SM100 d256 flavor lowers onto decode_d256_f16 or
    prefill_d256_f16, tallied as "frost:<engine>:<template>".  ``tensor_checker`` is forwarded to exec_sdpa (an
    exact check on the outputs before the tolerance compare)."""
    keys   = [f"frost:{engine}"] + ([f"frost:{engine}:{template}"] if template else [])
    before = [frost_routing.snapshot().get(k, 0) for k in keys]
    # The assertion is "FROST served it": opt FROST in for the call so the placement
    # tree (sdpa/fwd/placement.py) ranks ours first even on a shard measured behind
    # the backend -- the routing, not the default winner, is under test here.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        exec_sdpa(cfg, request, cudnn_handle, tensor_checker=tensor_checker)
    after  = [frost_routing.snapshot().get(k, 0) for k in keys]
    for key, b, a in zip(keys, before, after):
        assert a == b + 1, f"expected {key!r} to serve this graph; routing tally: {frost_routing.snapshot()}"
    if cga is not None:
        served, knobs = frost_routing.LAST_PLAN
        assert served == engine and knobs is not None and knobs.cga == cga, f"expected TILE_CGA_M={cga} on {engine!r}, got {served} {knobs}"


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=64, rng_seed=2004), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_gqa_partial_pack_frost_L0(env_info, test_no, request, cudnn_handle):

    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32, with_high_probability=[4, 32]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8, s_kv_min=8, s_kv_max=4096, s_q_distribution={"s_q=1":2, "s_q=s_kv":0, "s_q=random":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=128, d_qk_max=128, d_v_min=128, d_v_max=128, head_dim_distribution={"d_qk=d_v":1}),
        # GQA groups over 8 KV heads that do not divide the 128-row Q tile: 96/8 (G=12, packs 4),
        # 48/8 (G=6, packs 2), 24/8 (G=3, unpacked), 40/8 (G=5, unpacked).
        head_count=RandomChoice({(96, 8, 8) : 3, (48, 8, 8) : 2, (24, 8, 8) : 1, (40, 8, 8) : 1}),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=3, no_mask=1),
        diag_align=RandomChoice({cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"padded" : 1}),
        # page sizes the FROST paged contract serves (a multiple of 8 dividing the 128-row KV tile)
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16, 32, 128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.showConfig(test_no, request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=48, rng_seed=6607857), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_thd_decode_frost_L0(env_info, test_no, request, cudnn_handle):
    """FlashInfer's prefill-style paged graph at one token per sequence (nvbug
    6607857): ragged Q/O/Stats (ragged offsets + per-batch lengths) over page
    pools + block tables at S_q == 1, d128, GQA.  Every other ragged graph keeps
    the cga2 prefill THD leg; this one must ride the d128 decode tile's ragged-Q
    leg (TILE_CGA_M=1, PackGQA, KV split + combine placing the ragged rows) --
    asserted through the routing tally and the serving template."""

    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32, with_high_probability=[4, 8]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=16, s_kv_max=8192, s_q_distribution={"s_q=1":100, "s_q=s_kv":0, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=128, d_qk_max=128, d_v_min=128, d_v_max=128, head_dim_distribution={"d_qk=d_v":1}),
        head_count=RandomChoice({(64, 8, 8) : 2, (32, 8, 8) : 2, (16, 2, 2) : 1, (8, 8, 8) : 1}),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=1),
        diag_align=RandomChoice({cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 1}),
        # page sizes the FROST paged contract serves (a multiple of 8 dividing the 128-row KV tile)
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16, 32, 128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.showConfig(test_no, request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, cga=1, template="decode_d128_f16")


PARTIAL_PACK_PINNED_S_Q = [1, 2]


@pytest.mark.parametrize("s_q", PARTIAL_PACK_PINNED_S_Q, ids=["decode", "mtp2_bottom_right"])
@pytest.mark.L0
def test_sdpa_fwd_paged_gqa_partial_pack_pinned_frost_L0(env_info, s_q, request, cudnn_handle):
    """96 query heads over 8 KV heads (GQA ratio 12) at d128 over a 16-token page
    pool, b=32, mixed KV lengths <= 4096 -- the paged decode shape FlashInfer's
    cuDNN backend sends; bottom-right causal at s_q=2 (MTP).  Pinned so a
    bisect lands on one config.  FROST must serve it: the d128 f16 kernel packs
    4 of the 12 heads per token row-group (partial PackGQA) instead of running
    one live row per 512-row cluster."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2004,
        rng_geom_seed=2004,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=False,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=32,
        d_qk=128,
        d_v=128,
        s_q=s_q,
        s_kv=4096,
        h_q=96,
        h_k=8,
        h_v=8,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT,
        left_bound=None,
        right_bound=0 if s_q > 1 else None,
        seq_len_q=[s_q] * 32,
        # tile / page boundaries, a partial last page, 1-token and 0-token sequences
        seq_len_kv=[4096, 4095, 4000, 3073, 3072, 2048, 2047, 1536, 1025, 1024, 1000, 777, 513, 512, 511, 300,
                    257, 256, 255, 200, 129, 128, 127, 100, 65, 64, 33, 17, 16, 15, 1, 0],
        implementation=cudnn.attention_implementation.AUTO,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, len(PARTIAL_PACK_PINNED_S_Q)), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)

# # ==================================================================
# # L0 pinned decode graphs with an attention sink (FROST-served)
# # ==================================================================
#
# sink_token at s_q==1 is served by the FROST SM100 f16/bf16 engine
# (sdpa_fwd_prefill_sm100), dense and paged; the cuDNN backend engines still
# decline it (C++ support surface), so exec_sdpa's WAIVED skip cannot tell a
# still-rejecting validator from a working feature. The s_q == 1 sweeps draw the
# sink natively when FROST is on (_sq1_sink_token); the pinned graphs here
# ASSERT that FROST served them (_require_frost_sm100 / _exec_sdpa_on_frost above).
#
# Two tiles can serve them since #1094: on the d128 flavor a decode / MTP shape
# (s_q * PACK_G <= 128) rides the d128 decode tile (sm100/decode_d128_f16.py,
# TILE_CGA_M=1), whose sink fold carries the same keyless-row select as the
# prefill kernels'; every other flavor's decode graph runs its prefill kernel.
# The routing tally names the engine only, so each pinned graph asserts its
# tile (cga=): the two d128-envelope graphs the decode tile, the d256 pair the
# prefill kernel (prefill_d256_f16.py -- (256, 256) has no decode tile,
# engines.py cgas_by_d_shape lists (128, 128) and (192, 128) only).

@pytest.mark.L0
def test_sdpa_paged_decode_sink_sliding_window_frost_L0(env_info, request, cudnn_handle):
    """Deterministic decode graph as FlashInfer hands it to cuDNN for a d=64,
    64/8-head model with an attention sink and a 128-token left window (the
    GPT-OSS decode shape): bf16, paged KV (page 16), s_q=1, s_kv=2048 with mixed
    per-batch lengths (a full cache, a page-unaligned one, 129 = one KV tile plus
    a token, 16 = one page), sink_token, diagonal_band_left_bound=128 under
    BOTTOM_RIGHT alignment with right_bound=0. Served by FROST on the d128 decode
    tile (d128 envelope, PackGQA group 8: 1 * 8 <= 128 rows; TILE_CGA_M=1
    asserted); the backend engines decline sink at s_q==1."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2002,
        rng_geom_seed=2002,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=4,
        d_qk=64,
        d_v=64,
        s_q=1,
        s_kv=2048,
        h_q=64,
        h_k=8,
        h_v=8,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT,
        left_bound=128,
        right_bound=0,
        seq_len_q=[1, 1, 1, 1],
        seq_len_kv=[2048, 1337, 129, 16],
        with_sink_token=True,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, 1), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, cga=1)


@pytest.mark.L0
def test_sdpa_paged_decode_sink_keyless_rows_frost_L0(env_info, request, cudnn_handle):
    """Multi-token decode geometry from the review of PR #1095: bf16, paged KV
    (page 16), d=128, 4/1 heads, s_q=4 under BOTTOM_RIGHT alignment with
    right_bound=0, one batch holding a single live key -- three of its four rows
    have no key at all, so the sink is their whole mass and O must be 0 -- and one
    with a full 128-key cache, sink_token. The harness draws the sink from
    N(0, 0.5); the -120 sink that underflowed the FROST fold on exactly this
    geometry is pinned in test/python/sdpa/frost/test_sdpa_fwd_paged_sm100.py
    (test_paged_graph_keyless_rows_sink_magnitude). The backend can serve an
    s_q=4 sink graph too; the routing assertion keeps FROST the engine under test,
    on the d128 decode tile (4 * 4 <= 128 rows; TILE_CGA_M=1 asserted). The d256
    twin below runs the same geometry on the prefill kernel."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=1095,
        rng_geom_seed=1095,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=2,
        d_qk=128,
        d_v=128,
        s_q=4,
        s_kv=128,
        h_q=4,
        h_k=1,
        h_v=1,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT,
        right_bound=0,
        seq_len_q=[4, 4],
        seq_len_kv=[1, 128],
        with_sink_token=True,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, 1), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, cga=1)


PAGED_DECODE_SINK_D256_CASES = [
    # case_id, s_q, batches, h_q, h_kv, s_kv, left_bound, seq_len_kv, template
    # S_q * G <= 16 packed Q rows lower onto the d256 decode tile (decode_d256_f16);
    # more rows stay on the prefill kernel (prefill_d256_f16) -- config_sm100.decode_d256_q_tile.
    ("gpt_oss_shaped_sq1", 1, 4, 64, 8, 2048, 128, [2048, 1337, 129, 16], "decode_d256_f16"),
    ("keyless_rows_sq4", 4, 2, 4, 1, 128, None, [1, 128], "decode_d256_f16"),
    ("keyless_rows_sq32_prefill_tile", 32, 2, 4, 1, 128, None, [1, 128], "prefill_d256_f16"),
]

@pytest.mark.parametrize("case_id,s_q,batches,h_q,h_kv,s_kv,left_bound,seq_len_kv,template", PAGED_DECODE_SINK_D256_CASES, ids=[c[0] for c in PAGED_DECODE_SINK_D256_CASES])
@pytest.mark.L0
def test_sdpa_paged_decode_sink_d256_frost_L0(env_info, case_id, s_q, batches, h_q, h_kv, s_kv, left_bound, seq_len_kv, template, request, cudnn_handle):
    """The two pinned sink graphs above at d=256, plus a prefill-tile case. The
    (256, 256) flavor lowers a decode-shaped graph (S_q * G <= 16 packed Q rows)
    onto the d256 decode tile, decode_d256_f16.py, whose sink fold keeps the sink
    logit on keyless rows (O := 0, LSE := sink) like the prefill kernels' select
    from #1095; more rows run prefill_d256_f16.py's PAGED_KV specialization with
    the prefill fold. ``template`` pins which one served (frost_routing tally).
    gpt_oss_shaped_sq1: bf16, paged (page 16), s_q=1, 64/8 heads (8 rows), sink +
    left window 128 under BOTTOM_RIGHT with right_bound=0, the d64 graph's mixed
    lengths. keyless_rows_sq4: bf16, paged, 4/1 heads, s_q=4 (16 rows), one batch
    with a single live key (three keyless rows) and one with a full 128-key cache
    -- test_paged_graph_keyless_rows_sink_magnitude[d256]'s geometry with the
    harness's N(0, 0.5) sink. keyless_rows_sq32_prefill_tile: the same at s_q=32
    (128 rows: above the routed maximum, so the prefill kernel serves it; 31
    keyless rows in the one-key batch)."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=1256,
        rng_geom_seed=1256,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=batches,
        d_qk=256,
        d_v=256,
        s_q=s_q,
        s_kv=s_kv,
        h_q=h_q,
        h_k=h_kv,
        h_v=h_kv,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT,
        left_bound=left_bound,
        right_bound=0,
        seq_len_q=[s_q] * batches,
        seq_len_kv=seq_len_kv,
        with_sink_token=True,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, len(PAGED_DECODE_SINK_D256_CASES)), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, template=template)

# # ==================================
# # L0 ragged tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=256, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=256, d_v_min=1, d_v_max=256, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128), (256, 256)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        with_sink_token=RandomChoice({True : 1, False : 3}),
        ragged_stats_layout=RandomChoice({"token_major" : 1, "head_major" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.L0
@pytest.mark.parametrize("side,seq_lens,minimum,default_capacity", [
    ("q", [], 256, 320),
    ("kv", [], 128, 192),
    ("q", [0, 7], 7, 64),
    ("kv", [0, 7], 7, 64),
    ("q", [0, 0], 0, 64),
    ("kv", [0, 0], 0, 64),
])
def test_ragged_capacity_uses_effective_seq_lens(side, seq_lens, minimum, default_capacity):
    cfg = ExecConfig(
        batches=2, h_q=8, h_k=8, h_v=8, s_q=128, s_kv=64, d_qk=128, d_v=128,
        data_type=torch.float16, is_ragged=True, **{f"seq_len_{side}": seq_lens},
    )
    total = f"total_{side}"
    cfg.fill_derived_fields()
    assert getattr(cfg, total) == default_capacity

    setattr(cfg, total, minimum)
    cfg.fill_derived_fields()
    assert getattr(cfg, total) == minimum

    setattr(cfg, total, minimum - 1)
    with pytest.raises(AssertionError, match=total):
        cfg.fill_derived_fields()


@pytest.mark.L0
def test_ragged_stride_gaps_stable_under_stride_overrides():
    """The seeded per-tensor token and head gaps must not depend on which strides
    were explicitly provided, head gaps must respect the 16-byte rule, and turning
    head gaps off must leave the token gaps of the same seed unchanged."""
    from sdpa.random_config import ExecConfig, compute_packed_strides

    names = ("q", "k", "v", "o")
    base = dict(
        batches=2, h_q=8, h_k=8, h_v=8, s_q=64, s_kv=64, d_qk=128, d_v=128,
        data_type=torch.float16, is_ragged=True, rng_geom_seed=7,
    )
    plain = ExecConfig(**base)
    plain.fill_derived_fields()
    token_only = ExecConfig(**base, with_ragged_head_gap=False)
    token_only.fill_derived_fields()

    pinned_q = (64 * 8 * 128, 128, 8 * 128, 1)  # explicit packed Q, no gap
    pinned = ExecConfig(**base, stride_q=pinned_q)
    pinned.fill_derived_fields()
    assert pinned.stride_q == pinned_q
    assert (pinned.stride_k, pinned.stride_v, pinned.stride_o) == (plain.stride_k, plain.stride_v, plain.stride_o)

    quantum = 16 // plain.data_type.itemsize
    head_gaps, token_gaps = [], []
    for name in names:
        _, h, _, d = getattr(plain, f"shape_{name}")
        _, head_stride, token_stride, _ = getattr(plain, f"stride_{name}")
        head_gap, token_gap = head_stride - d, token_stride - h * head_stride
        assert head_gap >= 0 and head_gap % quantum == 0
        assert token_gap >= 0 and token_gap % (h * d) == 0
        # the head-gap knob must not disturb the token gap drawn for the same seed
        _, off_head_stride, off_token_stride, _ = getattr(token_only, f"stride_{name}")
        assert off_head_stride == d and off_token_stride - h * d == token_gap
        head_gaps.append(head_gap)
        token_gaps.append(token_gap)
    assert any(head_gaps) and any(token_gaps)

    # Auto-packed fallbacks: cu / multiplier offset forms (#538) and 1-byte
    # (fp8) data types (#537) derive packed strides regardless of the default.
    packed = {n: compute_packed_strides(getattr(plain, f"shape_{n}")) for n in names}
    for override in (dict(is_cu_seq_len=True), dict(with_ragged_offset_multiplier=True), dict(data_type=torch.float8_e4m3fn)):
        cfg = ExecConfig(**{**base, **override})
        cfg.fill_derived_fields()
        assert all(getattr(cfg, f"stride_{n}") == packed[n] for n in names), override


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L1
def test_sdpa_random_fwd_ragged_unified_L1(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=256, d_v_min=1, d_v_max=256, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128), (256, 256)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "cu_ragged" : 1, "padded" : 0, "full" : 0}),
        with_sink_token=RandomChoice({True : 1, False : 3}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# The ragged offset multiplier (CUDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER) is only
# supported on the unified SDPA forward engine; backward/composite engines reject a
# non-default multiplier. The attribute itself requires cuDNN 9.24.0.
@pytest.mark.skipif(
    cudnn.backend_version() < 92400,
    reason="ragged offset multiplier requires cuDNN >= 9.24.0",
)
@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L1
def test_sdpa_random_fwd_ragged_offset_multiplier_unified_L1(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.UNIFIED)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=256, d_v_min=1, d_v_max=256, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128), (256, 256)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged_mult" : 1, "cu_ragged_mult" : 1}),
        with_sink_token=RandomChoice({True : 1, False : 3}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    # Multiplier is only supported on the unified forward engine.
    test.cfg.implementation = cudnn.attention_implementation.UNIFIED
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=384, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_bwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=8, max=16),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=256, d_v_min=1, d_v_max=256, head_dim_distribution={"d_qk=d_v":5, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128), (256,256)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        is_deterministic=RandomChoice({True : 3, False : 1}),
        ragged_stats_layout=RandomChoice({"token_major" : 1, "head_major" : 1}),
        with_sink_token=RandomChoice({True : 1, False : 3}),
        with_stats_log2=RandomChoice({True : 1, False : 3}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 paged tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=384, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=64, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 0}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1,32,128]),
        with_sink_token=RandomChoice({True : 1, False : 3}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=64, s_kv_min=1, s_kv_max=512, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "cu_padded" : 1, "full" : 0}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1,32,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)

# # ==========================================================
# # L0 paged decode on the FROST SM100 row (split-KV, ragged max)
# # ==========================================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=2001), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_decode_split_frost_L0(env_info, test_no, request, cudnn_handle):
    """Paged decode (s_q=1) with the declared KV maximum drawn freely -- almost
    never a multiple of the 128-row KV tile, the FlashInfer spelling -- at a
    small batch, where the heuristic proposes a KV split. Every draw stays
    inside the SM100 row's paged contract (d_qk == d_v <= 256, page size a
    multiple of 8 dividing 128 or a multiple of it, padded, sink-free so the
    KV split is proposed) and must be served by FROST; the reference check
    covers the split + combine path.
    Own seed and function: widening test_sdpa_fwd_paged_L0 would reshuffle
    every downstream draw of that sweep."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1,2]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=129, s_kv_max=16384, s_q_distribution={"s_q=1":1}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=32, d_qk_max=256, d_v_min=32, d_v_max=256, head_dim_distribution={"d_qk=d_v":1}, with_high_probability=[(64,64), (128,128), (256,256)]),
        head_count=RandomHeadGenerator(min=1, max=16, head_group_options=(1, 4, 2)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=2, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"padded" : 1}),
        block_size=RandomBlockSize(min=8, max=1024, with_high_probability=[16,32,64,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.showConfig(test_no, request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)


PAGED_DECODE_SPLIT_FROST_CASES = [
    # (id, diag_align, right_bound): bottom-right causal is FlashInfer's MTP
    # spelling; no mask is its plain decode spelling (the one the split gate
    # over-declined: with no band covering the KV tail, a declared max that is
    # not a 128-multiple was mistaken for synthesized KV-tail padding).
    ("brcm",    cudnn.diagonal_alignment.BOTTOM_RIGHT, 0),
    ("no_mask", cudnn.diagonal_alignment.TOP_LEFT,     None),
]

@pytest.mark.parametrize("case_id,diag_align,right_bound", PAGED_DECODE_SPLIT_FROST_CASES, ids=[c[0] for c in PAGED_DECODE_SPLIT_FROST_CASES])
@pytest.mark.L0
def test_sdpa_fwd_paged_decode_split_frost_pinned_L0(env_info, case_id, diag_align, right_bound, request, cudnn_handle):
    """The FlashInfer paged-decode shape the split over-decline hit, pinned so a
    bisect lands on it: b=2, GQA 8:1, d=128, page 16, s_q=1, declared KV max
    4000 (not a multiple of the 128-row KV tile), per-batch lengths [4000, 3000].
    FROST must serve it; its leading plan is the KV split."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2001,
        rng_geom_seed=2001,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=False,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=2,
        d_qk=128,
        d_v=128,
        s_q=1,
        s_kv=4000,
        h_q=8,
        h_k=1,
        h_v=1,
        block_size=16,
        diag_align=diag_align,
        left_bound=None,
        right_bound=right_bound,
        seq_len_q=[1, 1],
        seq_len_kv=[4000, 3000],
        implementation=cudnn.attention_implementation.AUTO,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, len(PAGED_DECODE_SPLIT_FROST_CASES)), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)

# # ==================================
# # L0 fprop block mask tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_unified_block_mask_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_block_mask = True
    test.cfg.implementation = cudnn.attention_implementation.UNIFIED
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)

# # ==================================
# # L0 paged KV with MLA-shaped head dims (d_qk != d_v) on the FROST SM100 row
# # ==================================

# The routing gate and the served-engine assertion are the shared
# _require_frost_sm100() / _exec_sdpa_on_frost() above.


def _paged_mla_randomization(s_q_s_kv):
    """The paged MLA sweep space shared by the decode and prefill-shaped fuzzes:
    the native d192x128 flavor and mixed dims such as (256, 128) / (64, 192) that
    ride the d256 envelope, page sizes {16, 32, 64, 128} (the FlashInfer / vLLM
    range), MHA and GQA head groups, every mask family, both f16 dtypes."""
    return RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,8]),
        s_q_s_kv=s_q_s_kv,
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=8, d_qk_max=256, d_v_min=8, d_v_max=256, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":2}, with_high_probability=[(192,128), (256,128), (64,192)]),
        head_count=RandomHeadGenerator(min=4, max=32, head_group_options=(1, 1, 0)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=4, left_window_only=2, right_window_only=1, band_around_diag=1, no_mask=6),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 2}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 0}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16,32,64,128]),
    )


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=64, rng_seed=2006), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_mla_decode_L0(env_info, test_no, request, cudnn_handle):
    """Paged decode / MTP (s_q in [1, 8]) over MLA-shaped head dims: the native
    d192x128 flavor ((192, 128); envelope for e.g. (136, 72)) and mixed dims such as
    (256, 128) / (64, 192) on the d256 envelope, next to the square draws the d128 /
    d256 flavors already served. Every graph must run and must be served by FROST
    (asserted through the routing tally): under the opt-in the row leads wherever it
    is eligible. At these shapes the d192x128 and d256 flavors run their prefill tile
    (only d128 has a decode tile; the tracker's gaps table records the measured
    d192x128 decode gap and its follow-up); the reference check covers the paged
    loader either way."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with _paged_mla_randomization(
        RandomSequenceLength(s_q_min=1, s_q_max=8, s_kv_min=2, s_kv_max=8192, s_q_distribution={"s_q=1":6, "s_q=s_kv":0, "s_q=random":4, "s_q>s_kv":0}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.showConfig(test_no, request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=64, rng_seed=2007), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_mla_frost_L0(env_info, test_no, request, cudnn_handle):
    """Paged PREFILL-shaped queries (s_q in [64, 256]: chunked prefill over a paged
    cache) with MLA-shaped head dims on the FROST SM100 engine -- the d192x128 paged
    loader under every mask, page size and head group the decode sweep draws, at a Q
    extent where its prefill geometry is the right tile (B200 32/8, s_q=64,
    bottom-right causal: 244 us on FROST vs 917 us on the backend's prefill engine).
    Every graph that runs must be served by FROST (asserted through the routing
    tally). The s_q = s_kv branch of the sequence generator is off: it does not clamp
    to s_q_max and would draw full-square prefills up to 8192 rows."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with _paged_mla_randomization(
        RandomSequenceLength(s_q_min=64, s_q_max=256, s_kv_min=64, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":0, "s_q=random":10, "s_q>s_kv":0}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.showConfig(test_no, request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)


@pytest.mark.L0
def test_sdpa_fwd_paged_d192x128_decode_frost_L0(env_info, request, cudnn_handle):
    """Deterministic decode over a paged (192, 128) cache: b=8, 32/32 heads (a
    Kimi-Linear-class MLA layer), s_q=1, page 16, bf16, mixed per-batch KV lengths
    up to 4096 incl. a page/tile-boundary length, a one-token and a single-page
    sequence -- FlashInfer's spelling (no mask, per-batch seq_len_q of 1). Must run
    and must be served by sdpa_fwd_prefill_sm100 (a native decline fails here
    instead of passing green). This is the shape the d192x128 paged loader was
    measured on (the tracker's gaps table) and where its decode tile will land."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2006,
        rng_geom_seed=2006,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=8,
        d_qk=192,
        d_v=128,
        s_q=1,
        s_kv=4096,
        h_q=32,
        h_k=32,
        h_v=32,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=None,
        right_bound=None,
        seq_len_q=[1, 1, 1, 1, 1, 1, 1, 1],
        seq_len_kv=[4096, 3000, 17, 1, 2048, 129, 4095, 512],
        implementation=cudnn.attention_implementation.AUTO,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, 1), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("d_qk, d_v, h_q, h_kv", [(256, 128, 32, 32), (64, 192, 32, 8)], ids=["d256x128_mha", "d64x192_gqa"])
@pytest.mark.L0
def test_sdpa_fwd_paged_envelope_decode_frost_L0(env_info, request, cudnn_handle, d_qk, d_v, h_q, h_kv):
    """Deterministic decode over a paged cache on the mixed-dims ENVELOPE shapes this
    row newly admits onto the d256 flavor -- (256, 128) 32/32 MHA and (64, 192) 32/8
    GQA, head dims FROST serves zero-padded to the flavor's width. b=8, s_q=1, page
    16, bf16, mixed per-batch KV lengths up to 4096. Must run and must be served by
    FROST: the head-dim gate is the flavor the lowering SELECTS, and the previous
    "exactly one dim > 128" approximation declined both (INVERTED from a decline)."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2008,
        rng_geom_seed=2008,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=8,
        d_qk=d_qk,
        d_v=d_v,
        s_q=1,
        s_kv=4096,
        h_q=h_q,
        h_k=h_kv,
        h_v=h_kv,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=None,
        right_bound=None,
        seq_len_q=[1, 1, 1, 1, 1, 1, 1, 1],
        seq_len_kv=[4096, 3000, 17, 1, 2048, 129, 4095, 512],
        implementation=cudnn.attention_implementation.AUTO,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, 1), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)


@pytest.mark.L0
def test_sdpa_fwd_paged_d192x128_prefill_frost_L0(env_info, request, cudnn_handle):
    """Deterministic prefill-shaped queries over a paged (192, 128) cache: b=4, 32/8
    GQA, s_q=128 with per-batch query lengths (a full chunk, a partial chunk, one
    token, a chunk as long as its KV), page 16, bf16, bottom-right causal, mixed KV
    lengths up to 4096 -- chunked prefill on a Kimi-Linear-class layer. Must run and
    must be served by FROST (a native decline fails here instead of passing green)."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2007,
        rng_geom_seed=2007,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=4,
        d_qk=192,
        d_v=128,
        s_q=128,
        s_kv=4096,
        h_q=32,
        h_k=8,
        h_v=8,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT,
        left_bound=None,
        right_bound=0,
        seq_len_q=[128, 77, 1, 128],
        seq_len_kv=[4096, 3000, 129, 128],
        implementation=cudnn.attention_implementation.AUTO,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, 1), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)

# # ==================================
# # L0 paged d512 (DSv4-class) tests on the FROST SM100 engine
# # ==================================
#
# The d512 f16/bf16 flavor of sdpa_fwd_prefill_sm100 serves paged KV: native
# 512/512 and the (256, 512] envelope zero-padded. Every config below runs
# unpinned and must land on FROST -- the routing gate and the served-engine
# assertion are the shared _require_frost_sm100() / _exec_sdpa_on_frost() above
# (which opts FROST in for the call, so the default placement -- backend-first
# for paged d512 at s_q == 1 until the d512 decode tile lands -- is not what is
# under test here; the kernel is). The measured gap on the FlashInfer shape is
# a tracker row (SUPPORT_MATRIX_TRACKER.md, "Gaps at a glance").

def _require_paged_d512_env():
    """Environment gates for the d512 paged configs below, checked BEFORE the
    strict wrapper so a missing prerequisite stays a skip: cuDNN >= 9.25.0 (the
    configs carry zero-length KV entries, which validate_config waives below
    it), then _require_frost_sm100() -- the SM100 class (cc 10.0-10.6) and a
    usable CuTe DSL (a missing or too-old DSL is a skip, not a failure: the
    native backend serving the graph then is a legitimate fallback, not a
    FROST regression)."""
    if cudnn.backend_version() < 92500:
        pytest.skip("zero sequence length SDPA requires cuDNN 9.25.0 or higher")
    _require_frost_sm100()


@contextlib.contextmanager
def _must_run(request):
    """A function that claims every one of its draws is covered must not lose
    draws to the harness's WAIVED skips: inside this block a skip FAILS (dry runs
    excepted). Environment gates run before it (_require_paged_d512_env)."""
    try:
        yield
    except pytest.skip.Exception as e:
        if request.config.option.dryrun:
            raise
        pytest.fail(f"this config must run, not skip: {e}", pytrace=False)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=64, rng_seed=2010), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_d512_frost_L0(env_info, test_no, request, cudnn_handle):
    """Paged KV decode / speculative-decode (s_q in [1, 8]) on the d512 (DSv4-class)
    f16/bf16 flavor of the FROST SM100 engine: head dims in (256, 512] select the
    d512 kernel (d=384 rides its envelope zero-padded), GQA / MQA, page sizes
    16 .. 128, per-batch KV lengths incl. 0 and 1, no mask or (bottom-right)
    causal. Every draw must run on FROST unpinned -- strict (_must_run), so a
    WAIVED skip fails instead of silently thinning the claimed coverage. The KV
    pool extent draws from 2: a 1-token pool under s_q=1 is the s_q == s_kv == 1
    geometry the harness waives as a known issue (sdpa/fp16.py), which strict
    would fail (1 draw in ~1800 at s_kv_min=1); per-batch KV lengths still draw
    0 and 1."""
    _require_paged_d512_env()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,8]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8, s_kv_min=2, s_kv_max=4096, s_q_distribution={"s_q=1":4, "s_q=s_kv":0, "s_q=random":4, "s_q>s_kv":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=264, d_qk_max=512, d_v_min=264, d_v_max=512, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(512,512), (384,384)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(0, 2, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=1, no_mask=2),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 2}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 0}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16,32,64,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.showConfig(test_no, request)

    with _must_run(request):
        _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)


def _paged_d512_pinned_cfg(*, batches, s_q, h_q, h_kv, seq_len_q, seq_len_kv):
    """b x (h_q / h_kv) x s_q, d_qk = d_v = 512, padded paged KV (page 16) up to
    4096, bf16, inference, no mask -- the FlashInfer paged-wrapper contract."""
    cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2010,
        rng_geom_seed=2010,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=False,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=batches,
        d_qk=512,
        d_v=512,
        s_q=s_q,
        s_kv=4096,
        h_q=h_q,
        h_k=h_kv,
        h_v=h_kv,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=None,
        right_bound=None,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
    )
    cfg.fill_derived_fields()
    return cfg


PAGED_D512_PINNED_KV_HEADS = [1, 8]


@pytest.mark.parametrize("h_kv", PAGED_D512_PINNED_KV_HEADS, ids=["mqa_64_1", "gqa_64_8"])
@pytest.mark.L0
def test_sdpa_fwd_paged_d512_decode_frost_pinned_L0(env_info, h_kv, request, cudnn_handle):
    """DSv4-class d512 paged decode, pinned: b=8, 64 q heads over 1 (MQA) or 8
    (GQA) KV heads, d_qk = d_v = 512, s_q=1, mixed KV lengths up to 4096 incl. 0
    and 1, page 16, bf16 -- the FlashInfer decode-wrapper shape. Must run (a
    WAIVED skip fails) and be served by FROST under the opt-in. This is the shape
    the d512 prefill tile was measured on against the backend's paged decode
    engine (tracker gap row; placement keeps the backend first there by default
    until the d512 decode tile lands); a bisect of that number lands here."""
    _require_paged_d512_env()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = _paged_d512_pinned_cfg(batches=8, s_q=1, h_q=64, h_kv=h_kv, seq_len_q=[1] * 8, seq_len_kv=[4096, 1, 0, 300, 1024, 2048, 77, 129])
    test.showConfig((request.node.name, len(PAGED_D512_PINNED_KV_HEADS)), request)

    with _must_run(request):
        _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)


@pytest.mark.L0
def test_sdpa_fwd_paged_d512_prefill_frost_pinned_L0(env_info, request, cudnn_handle):
    """Paged PREFILL-shaped d512 (chunked prefill over a page pool), pinned: b=4,
    GQA 16:2, d_qk = d_v = 512, s_q=128 with per-batch q lengths incl. partial
    chunks, KV lengths up to 4096, page 16, bf16, no mask. Must run (strict) and
    be served by FROST under the opt-in -- the routing tally must show FROST."""
    _require_paged_d512_env()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = _paged_d512_pinned_cfg(batches=4, s_q=128, h_q=16, h_kv=2, seq_len_q=[128, 100, 128, 64], seq_len_kv=[4096, 300, 1024, 2048])
    test.showConfig((request.node.name, 1), request)

    with _must_run(request):
        _exec_sdpa_on_frost(test.cfg, request, cudnn_handle)

# # ==================================
# # L0 fprop bias tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_bias_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=4096, s_kv_min=1, s_kv_max=4096, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=1),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        is_bias=RandomChoice({True : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)

# # ==================================
# # L0 bprop bias tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_bwd_bias_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=8, max=16),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=4096, s_kv_min=1, s_kv_max=4096, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=256, d_v_min=1, d_v_max=256, head_dim_distribution={"d_qk=d_v":5, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128), (256,256)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 4, "full" : 1}),
        is_bias=RandomChoice({True : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 FP8 fprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=384, rng_seed=999), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_fwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[4]),
        s_q_s_kv=RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1": 2, "s_q=s_kv": 5, "s_q=random": 2}),
        # d up to 256: the fp8 forward sweep stopped at 192 while the backward
        # sweep drew 256, which is how the d=256 fp8 forward defect (GitHub
        # #981, FROST d256 fp8 TMEM race under masking) went unexercised here.
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=256, d_v_min=64, d_v_max=256, head_dim_distribution={"d_qk=d_v": 2, "d_qk=random": 1}, with_high_probability=[(64, 64), (128, 128), (192, 128), (256, 256)]),
        head_count=RandomHeadGenerator(min=1, max=16, head_group_options=(1, 5, 2)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        # Block-scaled O epilogue (FROST d128 per-tensor FP8 only): FP4 O + E4M3
        # scales per 16 d (16) or E4M3 O + UE8M0 scales per 32 d (32), with the
        # sf_o output. exec_sdpa_fp8 folds it to 0 on configs the epilogue does
        # not serve (paged / ragged / d != 128), so the draw stays a plain fp8 run there.
        o_block_scale=RandomChoice({0: 6, 16: 1, 32: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        # KNOWN GAP: a dense "padded" draw currently runs as full — exec_sdpa_fp8
        # binds seq_len tensors only for the paged and ragged paths, so the
        # padding mask is never applied here (only the ragged fp8 suites below
        # exercise real padding).
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 1, "full": 1}),
        with_sink_token=RandomChoice({True : 1, False : 2}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.AUTO)
    test.showConfig(test_no, request)

    # Randomly enable unfuse_fma via environment variable for SM100
    unfuse_fma = rng.choice([True, False])
    test.cfg.with_unfuse_fma = unfuse_fma
    if unfuse_fma:
        os.environ["CUDNN_UNFUSE_FMA"] = "1"
    elif "CUDNN_UNFUSE_FMA" in os.environ:
        del os.environ["CUDNN_UNFUSE_FMA"]

    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] == 10:
        rescale_threshold = rng.choice([0.0, 2.0, 4.0])
    else:
        rescale_threshold = 0.0
    test.cfg.rescale_threshold = rescale_threshold
    os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    try:
        exec_sdpa_fp8(test.cfg, request, cudnn_handle)
    finally:
        if "CUDNN_UNFUSE_FMA" in os.environ:
            del os.environ["CUDNN_UNFUSE_FMA"]
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


# # ==================================
# # L0 FP8 bprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=384, rng_seed=998), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_bwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(s_q_min=64, s_q_max=8192, s_kv_min=64, s_kv_max=8192, s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=192, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128), (192, 128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 0, "full": 1}),
        is_deterministic=RandomChoice({True: 1, False: 1}),
        with_sink_token=RandomChoice({True : 1, False : 2}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request)

    test.cfg.rescale_threshold = 0.0
    os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    try:
        exec_sdpa_fp8(test.cfg, request, cudnn_handle)
    finally:
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


# # ==================================
# # L0 FP8 paged attention tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=997), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_fwd_paged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(s_q_min=64, s_q_max=256, s_kv_min=64, s_kv_max=512, s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=128, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128)]),
        head_count=RandomHeadGenerator(min=1, max=4, head_group_options=(1, 2, 0)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 1, "full": 0}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16, 32, 64]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.showConfig(test_no, request)

    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] == 10:
        rescale_threshold = rng.choice([0.0, 2.0, 4.0])
    else:
        rescale_threshold = 0.0
    test.cfg.rescale_threshold = rescale_threshold
    os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    try:
        exec_sdpa_fp8(test.cfg, request, cudnn_handle)
    finally:
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


# # ==================================
# # L0 FP8 paged decode / paged prefill on the FROST SM100 per-tensor FP8 engine
# # ==================================
#
# FlashInfer-shaped fp8 KV-cache decode / MTP: sdpa_fp8 over E4M3/E5M2 page pools,
# s_q in [1, 8] ("s_q=1" weighted), GQA, page sizes {16, 32, 64, 128}, per-batch KV
# lengths (partial last pages, zero-length sequences), and the masks a decode step
# spells: none (plain decode), a causal upper bound top-left or bottom-right (MTP,
# each batch's diagonal anchored at its own KV length) and a left sliding window.
# Every request has >= 1 query token (the harness zeroes a dead Q row on BOTH sides
# of the compare, so a draw of all-zero Q lengths would pass vacuously).
#
# Routing. FROST engines are opt-in; under the opt-in the fp8 row's proposal leads the
# plan list for every paged fp8 graph it accepts, and every draw of the decode fuzz is
# inside its paged envelope (d <= 128 at the fp8 graphs' 16-granularity, dense Q,
# contract page sizes, bottom-right only under a causal bound), so each config ASSERTS
# that sdpa_fwd_prefill_sm100_fp8 served it over the DEFAULT walk: the harness only
# tallies which engine ran, and a silent fall-through to the backend would otherwise
# hide a decline. The pinned cases are the FlashInfer-shaped 64/4 decode graph (the
# capability win: without a Stats output cuDNN 9.26's backend engine fails to build it)
# and a prefill-shaped chunked-prefill graph. All three opt in to the harness's
# dead-page NaN poison (cfg.paged_nan_dead_pages): the FROST kernel promises a TMA-OOB
# page -1 for every table slot past a sequence's live pages, so a dereferenced dead slot
# fails the compare instead of passing silently. The harness pages K/V as HND pools
# ([pages, H_kv, page, D]) only; NHD pools are covered by the strict twins' hnd
# parametrization in test/python/sdpa/frost/test_sdpa_fwd_paged_sm100.py. The measured
# gap to the backend's decode engine at decode shapes is a kernel follow-up
# (SUPPORT_MATRIX_TRACKER.md gaps table: fp8 d128 decode tile), not a routing rule.

FROST_FP8_ENGINE = "sdpa_fwd_prefill_sm100_fp8"
FROST_FP8_ENGINE_KEY = f"frost:{FROST_FP8_ENGINE}"

def _exec_sdpa_fp8_expect_frost(cfg, request, cudnn_handle, strict=True, engine=FROST_FP8_ENGINE):
    """The fp8 twin of _exec_sdpa_on_frost: exec_sdpa_fp8, then assert the FROST FP8
    engine ``engine`` (the SM100 row by default; the cc 10.7 sweeps pass their own row's
    name) served the graph (the harness only tallies which engine ran; a decline falls
    through to the backend silently). ``strict`` turns the harness's own skips (a validator
    rejection, "unsupported forward graph") into failures: a config that claims FROST
    coverage must run."""
    import frost_routing
    key = f"frost:{engine}"
    before = frost_routing.snapshot().get(key, 0)
    try:
        exec_sdpa_fp8(cfg, request, cudnn_handle)
    except pytest.skip.Exception as e:
        if strict and not request.config.option.dryrun:
            pytest.fail(f"FROST-asserting fp8 config must run, not skip: {e}", pytrace=False)
        raise
    after = frost_routing.snapshot().get(key, 0)
    assert after == before + 1, f"expected {key} to serve this graph; routing tally: {frost_routing.snapshot()}"


@pytest.mark.L0
@pytest.mark.parametrize("b,h,hk,d,skv,page,dtype,qlens,kvlens", [
    (4, 7, 7, 64, 175, 128, torch.float8_e5m2, [2, 2, 7, 1], [34, 133, 121, 32]),
    (5, 5, 1, 96, 3439, 32, torch.float8_e4m3fn, [1, 1, 5, 3, 1], [3050, 128, 2572, 2010, 3009]),
])
def test_sdpa_fp8_paged_equal_head_and_query_axes(request, cudnn_handle, monkeypatch, b, h, hk, d, skv, page, dtype, qlens, kvlens):
    """BSHD allocation axes must bind as BHSD even when H == S hides the swap."""
    _require_frost_fp8_paged_leads()
    cfg = ExecConfig(
        data_type=dtype, output_type=dtype, rng_geom_seed=827, rng_data_seed=1045226910,
        is_infer=True, is_paged=True, paged_nan_dead_pages=True, is_padding=True,
        is_ragged=False, is_cu_seq_len=False, batches=b, h_q=h, h_k=hk, h_v=hk,
        s_q=h, s_kv=skv, d_qk=d, d_v=d, block_size=page, seq_len_q=qlens, seq_len_kv=kvlens,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT, rescale_threshold=4.0,
    )
    cfg.fill_derived_fields()
    monkeypatch.setenv("CUDNN_RESCALE_THRESHOLD", "4.0")
    _exec_sdpa_fp8_expect_frost(cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=64, rng_seed=2005), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_fwd_paged_decode_frost_L0(env_info, test_no, request, cudnn_handle):
    """Decode / MTP-shaped (s_q <= 8) fp8 paged graphs over the DEFAULT plan walk, each
    asserting the FROST fp8 row served it (block note above); a harness skip stays a skip."""
    _require_frost_fp8_paged_leads()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=16, with_high_probability=[8, 16]),
        s_q_s_kv=RandomSequenceLength(s_q_min=1, s_q_max=8, s_kv_min=1, s_kv_max=4096, s_q_distribution={"s_q=1": 6, "s_q=s_kv": 0, "s_q=random": 4}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=128, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128)], multiple_of=16),
        head_count=RandomHeadGenerator(min=4, max=32, head_group_options=(1, 6, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=5, causal=3, left_window_only=2),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1, cudnn.diagonal_alignment.BOTTOM_RIGHT: 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 1, "full": 0}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16, 32, 64, 128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.cfg.paged_nan_dead_pages = True
    # A decode / MTP step has at least one query token per request; the KV lengths stay
    # free (0 and partial-page sequences are the point of the paged draw).
    test.cfg.seq_len_q = [max(1, n) for n in test.cfg.seq_len_q]
    # Bottom-right alignment needs a causal upper bound (right_bound == 0 on the causal and
    # sliding-window draws); a no-mask draw is plain decode and stays top-left.
    if test.cfg.right_bound is None:
        test.cfg.diag_align = cudnn.diagonal_alignment.TOP_LEFT
    # The FROST FP8 kernels bake a 4-binade lazy-rescale threshold (config_sm100.rescale_threshold)
    # and the backend honors the same value; the reference mirrors what it is handed, so pin it.
    test.cfg.rescale_threshold = 4.0
    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    try:
        # Set inside the try, after the blocked-test skip: a config that skips must not
        # leak the value into the worker (the serializers read it into later graphs' JSON).
        os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)
        _exec_sdpa_fp8_expect_frost(test.cfg, request, cudnn_handle, strict=False)
    finally:
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


@pytest.mark.L0
def test_sdpa_fp8_fwd_paged_decode_pinned_frost_L0(env_info, request, cudnn_handle):
    """FlashInfer-shaped fp8 KV-cache decode: B=32, 64/4 heads (PackGQA), d128, S_q=1,
    page 16, mixed per-batch KV lengths up to 4096 (partial last pages, a zero-length
    and a one-token sequence, page and tile boundaries); e4m3 pools, f16 O. Over the
    default walk, asserting the FROST fp8 row served it (what a bisect of this path
    needs; a harness skip fails). Without Stats (the FlashInfer spelling) cuDNN 9.26's
    backend engine fails to build this graph -- the real-data twin over that spelling
    is test_paged_graph_fp8_flashinfer_shaped_decode_default_walk in
    test/python/sdpa/frost/test_sdpa_fwd_paged_sm100.py."""
    _require_frost_fp8_paged_leads()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.float8_e4m3fn,
        output_type=torch.float16,
        rng_data_seed=2005,
        rng_geom_seed=2005,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        paged_nan_dead_pages=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=32,
        d_qk=128,
        d_v=128,
        s_q=1,
        s_kv=4096,
        h_q=64,
        h_k=4,
        h_v=4,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=None,
        right_bound=None,
        seq_len_q=[1] * 32,
        seq_len_kv=[4096, 1, 0, 17, 16, 15, 128, 129, 127, 2048, 3000, 4095, 33, 1000, 1279, 512,
                    4096, 7, 8, 9, 640, 1023, 1024, 1025, 2047, 300, 77, 2500, 3999, 100, 200, 4000],
        rescale_threshold=4.0,
        implementation=cudnn.attention_implementation.AUTO,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, 1), request)

    try:
        os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)
        _exec_sdpa_fp8_expect_frost(test.cfg, request, cudnn_handle, strict=True)
    finally:
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


@pytest.mark.L0
def test_sdpa_fp8_fwd_paged_prefill_pinned_frost_L0(env_info, request, cudnn_handle):
    """Paged (chunked) prefill over fp8 pools: B=4, 16/2 heads (PackGQA), d128, s_q=128,
    page 16, e4m3 pools, f16 O, per-batch Q lengths (full, partial, one token) and KV
    lengths (full, partial last page, one token, zero). Over the default walk, asserting
    the FROST fp8 row served it (a fall-through to the backend or a harness skip fails)."""
    _require_frost_fp8_paged_leads()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.float8_e4m3fn,
        output_type=torch.float16,
        rng_data_seed=2006,
        rng_geom_seed=2006,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        paged_nan_dead_pages=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=4,
        d_qk=128,
        d_v=128,
        s_q=128,
        s_kv=1024,
        h_q=16,
        h_k=2,
        h_v=2,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=None,
        right_bound=None,
        seq_len_q=[128, 100, 1, 64],
        seq_len_kv=[1024, 300, 17, 0],
        rescale_threshold=4.0,
        implementation=cudnn.attention_implementation.AUTO,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, 1), request)

    try:
        os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)
        _exec_sdpa_fp8_expect_frost(test.cfg, request, cudnn_handle, strict=True)
    finally:
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


# # ==================================
# # L0 FP8 THD (ragged) fprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=996), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_fwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(s_q_min=64, s_q_max=256, s_kv_min=64, s_kv_max=512, s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=128, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 1, "cu_ragged": 1, "cu_ragged_mult": 1, "padded": 0, "full": 0}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
    test.showConfig(test_no, request)

    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] == 10:
        rescale_threshold = rng.choice([0.0, 2.0, 4.0])
    else:
        rescale_threshold = 0.0
    test.cfg.rescale_threshold = rescale_threshold
    os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    try:
        exec_sdpa_fp8(test.cfg, request, cudnn_handle)
    finally:
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


# # ==================================
# # L0 FP8 THD (ragged) bprop tests
# # ==================================

@pytest.mark.skipif(
    cudnn.backend_version() <= 92100,
    reason="ragged FP8 backward requires cuDNN > 9.21.0",
)
@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=995), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_bwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(s_q_min=64, s_q_max=8192, s_kv_min=64, s_kv_max=8192, s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=128, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 1, "padded": 0, "full": 0}),
        is_deterministic=RandomChoice({True: 1, False: 1}),
        with_sink_token=RandomChoice({True: 1, False: 2}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request)

    test.cfg.rescale_threshold = 0.0
    os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    try:
        exec_sdpa_fp8(test.cfg, request, cudnn_handle)
    finally:
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


# # ==================================
# # L0 MXFP8 fprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=384, rng_seed=1001), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_mxfp8_fwd_L0(env_info, test_no, request, cudnn_handle):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 SDPA requires Blackwell (SM100+)")

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4),
        s_q_s_kv=RandomSequenceLength(s_q_min=128, s_q_max=8192, s_kv_min=128, s_kv_max=8192, s_q_distribution={"s_q=1": 0, "s_q=s_kv": 1, "s_q=random": 1}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=256, d_v_min=64, d_v_max=256, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128), (192, 128), (256, 256)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 3, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float16: 2, torch.bfloat16: 1}),  # FP16 more often for tighter tolerance testing
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        # full-only: the sdpa_mxfp8 API has no seq_len/padding arguments, so a
        # "padded" draw would silently run dense-full (exec_sdpa_mxfp8 never
        # reads seq_len_q/kv) and inflate padded coverage. When the API grows
        # seq-len support, re-add padded/ragged draws — the shared
        # packed_token_capacity / convert_uniform_to_packed helpers then give
        # the NaN-poisoned capacity tails that catch the GitHub #624 class.
        is_ragged_or_padded_or_full=RandomChoice({"full": 1}),
        with_sink_token=RandomChoice({True : 1, False : 2}),
        # Block-scaled O on the FROST d128 MXFP8 kernel: FP4 O + E4M3/16 scales (16)
        # or E4M3 O + UE8M0/32 scales (32) with the sf_o output. exec_sdpa_mxfp8
        # folds it to 0 on configs the epilogue does not serve (d != 128, unfuse_fma,
        # a ragged KV tail without a covering causal band, FROST engines off, an arch
        # without an MXFP8 engine row such as SM120), so the draw stays a plain mxfp8 run there.
        o_block_scale=RandomChoice({0: 6, 16: 1, 32: 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_mxfp8 = True
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.AUTO)

    # Randomly enable unfuse_fma via environment variable for SM100
    unfuse_fma = rng.choice([True, False])
    test.cfg.with_unfuse_fma = unfuse_fma
    if unfuse_fma:
        os.environ["CUDNN_UNFUSE_FMA"] = "1"
    elif "CUDNN_UNFUSE_FMA" in os.environ:
        del os.environ["CUDNN_UNFUSE_FMA"]

    compute_capability = torch.cuda.get_device_capability()
    if compute_capability[0] == 10:
        rescale_threshold = rng.choice([0.0, 2.0, 4.0])
    else:
        rescale_threshold = 0.0
    test.cfg.rescale_threshold = rescale_threshold
    os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)

    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    try:
        exec_sdpa_mxfp8(test.cfg, request, cudnn_handle)
    finally:
        if "CUDNN_UNFUSE_FMA" in os.environ:
            del os.environ["CUDNN_UNFUSE_FMA"]
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]

# # ==================================
# # L0 MXFP8 paged KV cache on the FROST SM100 MXFP8 row
# # ==================================
#
# sdpa_mxfp8 over page pools whose F8_128x4 descales page with K/V (page_size % 128). The
# cuDNN backend declines every MXFP8 paged graph, so each config asserts FROST routing.

FROST_MXFP8_ENGINE = "sdpa_fwd_prefill_sm100_mxfp8"
FROST_MXFP8_ENGINE_KEY = f"frost:{FROST_MXFP8_ENGINE}"


def _exec_sdpa_mxfp8_expect_frost(cfg, request, cudnn_handle, strict=True, engine=FROST_MXFP8_ENGINE):
    """exec_sdpa_mxfp8, then assert the FROST MXFP8 row ``engine`` (the SM100 row by default;
    the cc 10.7 sweeps pass their own row's name) served the graph; with ``strict`` a harness
    skip (every engine declined) is a failure, so an unserved config cannot pass silently."""
    key = f"frost:{engine}"
    before = frost_routing.snapshot().get(key, 0)
    try:
        exec_sdpa_mxfp8(cfg, request, cudnn_handle)
    except pytest.skip.Exception as e:
        if strict and not request.config.option.dryrun:
            pytest.fail(f"FROST-asserting mxfp8 config must run, not skip: {e}", pytrace=False)
        raise
    after = frost_routing.snapshot().get(key, 0)
    assert after == before + 1, f"expected {key} to serve this graph; routing tally: {frost_routing.snapshot()}"


def _run_mxfp8_paged(test, test_no, request, cudnn_handle):
    test.showConfig(test_no, request)
    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    try:
        os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)
        _exec_sdpa_mxfp8_expect_frost(test.cfg, request, cudnn_handle)
    finally:
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=64, rng_seed=2006), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_mxfp8_fwd_paged_decode_frost_L0(env_info, test_no, request, cudnn_handle):
    """Decode / MTP-shaped (s_q <= 8) MXFP8 paged graphs at page sizes 128 / 256, d256, over the
    default plan walk, each asserting the FROST MXFP8 row served it."""
    _require_frost_sm100(FROST_MXFP8_ENGINE)

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=16, with_high_probability=[4, 16]),
        s_q_s_kv=RandomSequenceLength(s_q_min=1, s_q_max=8, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1": 6, "s_q=s_kv": 0, "s_q=random": 4}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=256, d_qk_max=256, d_v_min=256, d_v_max=256, head_dim_distribution={"d_qk=d_v": 1}),
        head_count=RandomHeadGenerator(min=2, max=32, head_group_options=(1, 8, 2)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float16: 1, torch.bfloat16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=5, causal=3, left_window_only=2),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1, cudnn.diagonal_alignment.BOTTOM_RIGHT: 1}),
        is_ragged_or_padded_or_full=RandomChoice({"padded": 1}),
        with_sink_token=RandomChoice({True: 1, False: 3}),
        block_size=RandomBlockSize(min=128, max=256, with_high_probability=[128, 256]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_mxfp8 = True
    test.cfg.is_paged = True
    # A decode / MTP step has at least one query token per request. KV lengths stay free.
    test.cfg.seq_len_q = [max(1, n) for n in test.cfg.seq_len_q]
    # Bottom-right alignment needs a causal upper bound.
    if test.cfg.right_bound is None:
        test.cfg.diag_align = cudnn.diagonal_alignment.TOP_LEFT
    # The FROST MXFP8 kernels bake the 4-binade lazy-rescale threshold.
    test.cfg.rescale_threshold = 4.0
    _run_mxfp8_paged(test, test_no, request, cudnn_handle)


MXFP8_PAGED_PINNED_CASES = [
    # (id, batches, h_q, h_kv, s_q, s_kv, block_size, diag_align, right_bound, seq_len_kv)
    ("qwen35_decode", 32, 32, 2, 1, 4096, 128, cudnn.diagonal_alignment.TOP_LEFT, None, [4096, 1, 0, 129, 2049, 4095, 130, 3000] * 4),
    ("qwen35_mtp4_br", 8, 32, 2, 4, 4096, 128, cudnn.diagonal_alignment.BOTTOM_RIGHT, 0, [4096, 4, 700, 129, 2049, 4095, 130, 3000]),
    ("chunked_prefill_512", 2, 8, 2, 512, 8192, 256, cudnn.diagonal_alignment.BOTTOM_RIGHT, 0, [8192, 700]),
]


@pytest.mark.parametrize("case_id,batches,h_q,h_kv,s_q,s_kv,block_size,diag_align,right_bound,seq_len_kv", MXFP8_PAGED_PINNED_CASES, ids=[c[0] for c in MXFP8_PAGED_PINNED_CASES])
@pytest.mark.L0
def test_sdpa_mxfp8_fwd_paged_pinned_frost_L0(env_info, case_id, batches, h_q, h_kv, s_q, s_kv, block_size, diag_align, right_bound, seq_len_kv, request, cudnn_handle):
    """Qwen3.5-shaped MXFP8 paged decode / MTP and a chunked-prefill step, pinned."""
    _require_frost_sm100(FROST_MXFP8_ENGINE)
    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.float8_e4m3fn,
        output_type=torch.bfloat16,
        rng_data_seed=2006,
        rng_geom_seed=2006,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_mxfp8=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=batches,
        d_qk=256,
        d_v=256,
        s_q=s_q,
        s_kv=s_kv,
        h_q=h_q,
        h_k=h_kv,
        h_v=h_kv,
        block_size=block_size,
        diag_align=diag_align,
        right_bound=right_bound,
        rescale_threshold=4.0,
        seq_len_q=[s_q] * batches,
        seq_len_kv=list(seq_len_kv),
    )
    test.cfg.fill_derived_fields()
    _run_mxfp8_paged(test, (request.node.name, len(MXFP8_PAGED_PINNED_CASES)), request, cudnn_handle)


# # ==================================
# # L0 MXFP8 bprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=512, rng_seed=1002), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_mxfp8_bwd_L0(env_info, test_no, request, cudnn_handle):
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 SDPA requires Blackwell (SM100+)")

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # d_qk = d_v = 256 is served by the frontend-only FROST engine
    # (sdpa_bwd_sm100_mxfp8, opt-in, BSHD-physical only); the cuDNN backend
    # serves up to d_qk=192/d_v=128. Draws no engine serves skip.
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4),
        s_q_s_kv=RandomSequenceLength(s_q_min=256, s_q_max=8192, s_kv_min=256, s_kv_max=8192, s_q_distribution={"s_q=1": 0, "s_q=s_kv": 1, "s_q=random": 1}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=256, d_v_min=64, d_v_max=256, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128), (192, 128), (256, 256)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 0}),
        output_type=RandomChoice({torch.float16: 2, torch.bfloat16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 0, "full": 1}),
        is_deterministic=RandomChoice({True: 1, False: 0}),
        with_sink_token=RandomChoice({True : 1, False : 2}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
        test.cfg.use_causal_mask = test.cfg.left_bound is None and test.cfg.right_bound == 0

    test.cfg.is_mxfp8 = True
    test.cfg.is_infer = False

    test.cfg.rescale_threshold = 0.0
    os.environ["CUDNN_RESCALE_THRESHOLD"] = str(test.cfg.rescale_threshold)

    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    try:
        exec_sdpa_mxfp8(test.cfg, request, cudnn_handle)
    finally:
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]

# # ===================
# # Single repro test
# # ===================

MIXED_SEQ_LEN_FORM_CASES = [
    ("q", cudnn.diagonal_alignment.TOP_LEFT, None),
    ("kv", cudnn.diagonal_alignment.TOP_LEFT, None),
    ("q", cudnn.diagonal_alignment.BOTTOM_RIGHT, 0),
    ("kv", cudnn.diagonal_alignment.BOTTOM_RIGHT, 0),
]


@pytest.mark.parametrize(
    "cu_sides,diag_align,right_bound",
    MIXED_SEQ_LEN_FORM_CASES,
    ids=["cu_q", "cu_kv", "cu_q_brcm", "cu_kv_brcm"],
)
@pytest.mark.L0
def test_sdpa_mixed_seq_len_forms_L0(env_info, cu_sides, diag_align, right_bound, request, cudnn_handle):
    """Mixed-form sequence lengths: cumulative on one side, per-batch on the other.

    Deterministic configs with non-uniform per-batch lengths, so misreading one
    side's form cannot produce a passing result. The bottom-right causal cases
    guard the DiagonalBandMask alignment derivation, which must treat the two
    sides' forms independently. Requires cuDNN 9.25+ (skips below via exec_sdpa).
    """
    # Mixed forms are gated in the UNIFIED surface; request it explicitly rather
    # than relying on AUTO resolving to it.
    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.UNIFIED)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=1234,
        rng_geom_seed=5678,
        is_alibi=False,
        is_infer=True,
        is_paged=False,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=True,
        cu_seq_len_sides=cu_sides,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=4,
        d_qk=64,
        d_v=64,
        s_q=256,
        s_kv=512,
        h_q=3,
        h_k=3,
        h_v=3,
        diag_align=diag_align,
        left_bound=None,
        right_bound=right_bound,
        seq_len_q=[128, 100, 256, 37],
        seq_len_kv=[96, 64, 512, 200],
        implementation=cudnn.attention_implementation.UNIFIED,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, len(MIXED_SEQ_LEN_FORM_CASES)), request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 paged decode / MTP tests on the FROST d128 decode tile (SM100, opt-in)
# # ==================================

# The routing gate and the served-engine assertion are the shared
# _require_frost_sm100() / _exec_sdpa_on_frost(..., cga=1) above: cga=1 pins the
# d128 DECODE tile behind sdpa_fwd_prefill_sm100's TILE_CGA_M knob.


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=2008), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_decode_tile_frost_L0(env_info, test_no, request, cudnn_handle):
    """Paged decode / MTP shapes the FROST d128 decode tile serves: s_q in [1, 8]
    (mostly 1), GQA groups that divide the 128-row tile (4/8/16), one that does
    not (96/8: G=12 packs 4 -- partial PackGQA -- with three packed heads per KV
    head) and MQA, d in
    {64, 128} (d64 rides the d128 envelope), page sizes 16..128, no mask /
    bottom-right causal / sliding window. No sink draw here: the decode tile's sink
    arm is pinned by test_sdpa_paged_decode_sink_* above. Asserts the decode tile served the graph.
    """
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32, with_high_probability=[1,8,32]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8, s_kv_min=16, s_kv_max=4096, s_q_distribution={"s_q=1":6, "s_q=random":4}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=128, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v":1}, with_high_probability=[(64,64), (128,128)]),
        head_count=RandomChoice({(64, 4, 4) : 3, (64, 8, 8) : 3, (32, 2, 2) : 2, (16, 1, 1) : 1, (8, 8, 8) : 1, (96, 8, 8) : 2}),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=4, left_window_only=2, no_mask=6),
        diag_align=RandomChoice({cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"padded" : 1}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16,32,64,128]),
        with_sink_token=RandomChoice({False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    # (head_count is a RandomChoice of (h_q, h_k, h_v) triples, not a
    # RandomHeadGenerator: every group here rides the decode tile -- 4 / 8 / 16
    # / MQA / MHA pack whole, 96/8 packs 4 of its 12 heads (s_q * 4 <= 32 rows).)
    test.cfg.is_paged = True
    # Decode / MTP: every batch carries all its s_q query tokens (FlashInfer's contract).
    test.cfg.seq_len_q = [test.cfg.s_q] * test.cfg.batches
    test.cfg.fill_derived_fields()
    test.showConfig(test_no, request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, cga=1)


@pytest.mark.L0
def test_sdpa_fwd_paged_decode_tile_flashinfer_pinned_L0(env_info, request, cudnn_handle):
    """The FlashInfer paged-decode shape (Qwen3-235B: b=32, H=64/4, d=128, s_q=1,
    s_kv=4096, page 16, bf16, padding mask, no causal mask) pinned deterministically,
    asserting FROST's d128 decode tile (TILE_CGA_M=1) serves it. This is the shape
    the decode tile was measured on (B200: 117.9 us prefill tile -> decode tile, see
    sm100/decode_d128_f16.py); a git bisect of that number lands here.
    """
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2008,
        rng_geom_seed=2008,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=False,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=32,
        d_qk=128,
        d_v=128,
        s_q=1,
        s_kv=4096,
        h_q=64,
        h_k=4,
        h_v=4,
        block_size=16,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=None,
        right_bound=None,
        seq_len_q=[1] * 32,
        seq_len_kv=[4096, 1, 129, 4000, 2048, 17, 4096, 3333] * 4,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, 1), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, cga=1)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=2003), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_decode_mtp_frost_L0(env_info, test_no, request, cudnn_handle):
    """Paged GQA decode (S_q=1) and MTP (S_q in [2, 8]) on the FROST SM100 d128
    flavor: bottom-right causal weighted, head groups up to 16:1 (incl. a group
    that does not divide the tile), page sizes 16..128, padded per-batch lengths.
    Every draw is inside the FROST paged contract and every unit fits one decode
    tile (S_q * PACK_G <= 128), so the d128 decode tile (TILE_CGA_M=1) must serve
    it -- a wider fuzz of that tile than test_sdpa_fwd_paged_decode_tile_frost_L0:
    top-left alignment, right-window and band masks, any d_qk / d_v <= 128,
    batches to 64, KV to 8192."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=64, with_high_probability=[8,32]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8, s_kv_min=8, s_kv_max=8192, s_q_distribution={"s_q=1":6, "s_q=random":4}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=8, d_qk_max=128, d_v_min=8, d_v_max=128, head_dim_distribution={"d_qk=d_v":3, "d_qk=random":1}, with_high_probability=[(128,128), (64,64)]),
        head_count=RandomChoice({(64, 4, 4) : 3, (64, 8, 8) : 2, (16, 4, 4) : 1, (24, 2, 2) : 1, (8, 1, 1) : 1, (32, 32, 32) : 1}),  # (h_q, h_k, h_v): 16:1, 8:1, 4:1, 12:1 (partial PackGQA: packs 4 of 12), MQA, MHA
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=4, right_window_only=1, band_around_diag=1, no_mask=6),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 3}),
        is_ragged_or_padded_or_full=RandomChoice({"padded" : 1}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16,32,64,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    # Decode / MTP: every batch carries all its s_q query tokens (FlashInfer's contract).
    test.cfg.seq_len_q = [test.cfg.s_q] * test.cfg.batches
    test.cfg.fill_derived_fields()
    test.showConfig(test_no, request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, cga=1)


# FlashInfer's cudnn paged decode graph (Qwen3-235B-style 64/4 heads, d128, page 16) and its MTP sibling.
FI_PAGED_DECODE_CASES = [
    # (s_q, diag_align, right_bound, cga): cga pins the tile behind sdpa_fwd_prefill_sm100's
    # TILE_CGA_M knob -- 1 the d128 decode tile (S_q * PACK_G <= 128), 2 the prefill pipeline.
    (1,  cudnn.diagonal_alignment.TOP_LEFT,     None, 1),  # S_q=1, no mask: the wrapper's spelling today
    (4,  cudnn.diagonal_alignment.BOTTOM_RIGHT, 0,    1),  # MTP S_q=4, bottom-right causal: 64 packed rows
    (16, cudnn.diagonal_alignment.BOTTOM_RIGHT, 0,    2),  # 16 speculative tokens: 256 packed rows, one cga2 cluster
]


@pytest.mark.parametrize("s_q,diag_align,right_bound,cga", FI_PAGED_DECODE_CASES, ids=["decode_sq1", "mtp_sq4_brcm", "chunk_sq16_brcm_prefill_tile"])
@pytest.mark.L0
def test_sdpa_fwd_paged_decode_fi_shapes_frost_L0(env_info, s_q, diag_align, right_bound, cga, request, cudnn_handle):
    """FlashInfer's paged decode graph pinned: b=32, 64/4 heads, d128 bf16, page 16,
    per-batch KV lengths mixed up to 4096 (page and tile boundaries, 1, and for the
    causal cases a length below S_q), served by the FROST SM100 engine on the tile
    the heuristics select for the unit's rows: the decode tile for decode and MTP,
    the cga2 prefill tile (on the plain scheduler -- the one-cluster rule) for a
    16-token speculative chunk whose 256 packed rows overflow one decode tile."""
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    seq_len_kv = [4096, 4095, 4080, 3000, 2048, 2047, 1024, 129, 128, 127, 100, 17, 16, 15, 8, 1] * 2
    seq_len_kv[-1] = 3  # below S_q for MTP: keyless bottom-right rows
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2003,
        rng_geom_seed=2003,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=False,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=32,
        d_qk=128,
        d_v=128,
        s_q=s_q,
        s_kv=4096,
        h_q=64,
        h_k=4,
        h_v=4,
        block_size=16,
        diag_align=diag_align,
        left_bound=None,
        right_bound=right_bound,
        seq_len_q=[s_q] * 32,
        seq_len_kv=seq_len_kv,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, len(FI_PAGED_DECODE_CASES)), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, cga=cga)


@pytest.mark.L0
def test_sdpa_fwd_dense_mtp_decode_tile_sink_keyless_rows_frost_L0(env_info, request, cudnn_handle):
    """Multi-token decode geometry from the review of PR #1094: bf16, dense padded
    KV (declared s_kv=128), d=128, 4/1 heads, s_q=4 under BOTTOM_RIGHT alignment
    with right_bound=0, one batch holding a single live key -- three of its four
    rows have no key at all, so the sink is their whole mass and O must be 0 --
    and one with all 128 keys, sink_token, on the d128 decode tile. Dense because
    the FROST row declines paged + sink today. The harness draws the sink from
    N(0, 0.5); the -120 sink that underflowed the decode tile's fold on exactly
    this geometry (O = NaN, LSE = -inf) is pinned in
    test/python/sdpa/frost/test_sdpa_fwd_decode_d128_sm100.py
    (test_decode_kernel_keyless_rows_sink_magnitude and its graph-path twin).
    The backend can serve an s_q=4 sink graph too; the routing assertion keeps
    the decode tile the kernel under test.
    """
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=1094,
        rng_geom_seed=1094,
        is_alibi=False,
        is_infer=True,
        is_paged=False,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=False,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=2,
        d_qk=128,
        d_v=128,
        s_q=4,
        s_kv=128,
        h_q=4,
        h_k=1,
        h_v=1,
        diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT,
        left_bound=None,
        right_bound=0,
        seq_len_q=[4, 4],
        seq_len_kv=[1, 128],
        with_sink_token=True,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, 1), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, cga=1)


@pytest.mark.L0
def test_sdpa_thd_batch_stride_int32_overflow_L0(env_info, request, cudnn_handle):
    """Packed-THD decode whose whole-buffer size exceeds INT32_MAX elements.

    The FROST THD views bound the extent-1 batch dim with ``T * token_stride``;
    the kernel ABI checks every stride against the int32 range, so a packed KV
    buffer of more than 2^31 elements failed at execute with "Out of bound
    k_tensor.strides[0]" (GitHub #980, found by the dsv3/kimi_k3 model suites:
    h=128, d_qk=192, ~57k packed KV tokens). The random sweeps in this file
    cannot reach that boundary within their memory budget (their largest packed
    stride is 32 * 8192 * 32 * 192 = 1.6e9), so this pins it deterministically:
    128 heads x d_qk=192 x 90,800 packed KV tokens = 2.23e9 > 2^31. K alone is
    4.4 GiB -- that size is the bug's precondition, not a knob.
    """
    if torch.cuda.get_device_properties(0).total_memory < 16 * 2**30:
        pytest.skip("needs a >4 GiB packed K buffer (plus V and reference)")
    seq_len_kv = [11400] * 4 + [11300] * 4  # 90,800 tokens; each K token is 128*192 elements
    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=980,
        rng_geom_seed=980,
        is_alibi=False,
        is_infer=True,
        is_paged=False,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=False,
        is_ragged=True,
        with_ragged_token_gap=False,  # packed contract: token stride = h*d exactly
        with_ragged_head_gap=False,
        is_dropout=False,
        is_determin=False,
        batches=len(seq_len_kv),
        d_qk=192,
        d_v=64,  # keeps V at ~1.5 GiB; only K needs to cross the boundary
        s_q=1,
        s_kv=max(seq_len_kv),
        h_q=128,
        h_k=128,
        h_v=128,
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        left_bound=None,
        right_bound=None,
        seq_len_q=[1] * len(seq_len_kv),
        seq_len_kv=seq_len_kv,
    )
    test.cfg.fill_derived_fields()
    assert sum(seq_len_kv) * test.cfg.h_k * test.cfg.d_qk > 2**31, "config must cross the int32 element boundary"
    test.showConfig((request.node.name, 1), request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 paged decode d256 (FROST decode tile) tests
# # ==================================

# The two templates the SM100 d256 row lowers onto (frost_routing tallies
# "frost:<engine>:<template>", sdpa/helpers.note_frost_routing); the gate and the
# routing assertion are the shared _require_frost_sm100 / _exec_sdpa_on_frost.
FROST_D256_DECODE_TEMPLATE = "decode_d256_f16"
FROST_D256_PREFILL_TEMPLATE = "prefill_d256_f16"
# Packed Q rows (S_q x GQA group) the adapter routes onto the decode tile
# (config_sm100.D256_DECODE_ROUTED_MAX_Q_ROWS); the prefill d256 tile serves the rest.
FROST_D256_DECODE_MAX_ROWS = 16


def _decode_d256_heads(rng):
    """Head groups the decode tile packs whole: 8:1 (two tokens per 16-row tile at
    S_q = 1) and 16:1 (one), over 1, 2 or 4 KV heads (Qwen3-Next 16/2, Qwen3.5 32/2)."""
    group = rng.choice([8, 16])
    h_k = rng.choice([1, 2, 4])
    return group * h_k, h_k, h_k


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=2009), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_decode_d256_frost_L0(env_info, test_no, request, cudnn_handle):
    """Decode-shaped paged d256 (Qwen3-Next / Qwen3.5 class): S_q in [1, 8] over 8:1
    and 16:1 groups, page sizes 16..128, both diagonal alignments with causal /
    sliding-window / band masks, mixed per-batch lengths (zeros included); inference
    graphs (the harness runs a backward otherwise, and paged KV is forward-only).
    The FROST engine row must serve every config -- the decode tile
    (sm100/decode_d256_f16.py) while S_q x group <= 16 rows, the prefill d256 tile
    above that -- so the routing tally is asserted: a decline would fall back to
    the backend silently and pass on the wrong kernel, and a decode-shaped draw
    quietly served by the prefill tile would pass on the slower one.
    """
    _require_frost_sm100()

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[4, 8]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":6, "s_q=random":4}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=256, d_qk_max=256, d_v_min=256, d_v_max=256, head_dim_distribution={"d_qk=d_v":1}),
        head_count=_decode_d256_heads,
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, band_around_diag=5, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 2}),
        is_ragged_or_padded_or_full=RandomChoice({"padded" : 1}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16,32,64,128]),
        with_sink_token=RandomChoice({False : 1}),  # the paged d256 sink arm is pinned above (test_sdpa_paged_decode_sink_d256_frost_L0)
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    test.showConfig(test_no, request)

    # A packed group (8 or 16 divides the tile) rides the decode tile while its
    # S_q x group rows fit; past that the prefill d256 tile serves the graph.
    decode_shaped = test.cfg.s_q * (test.cfg.h_q // test.cfg.h_k) <= FROST_D256_DECODE_MAX_ROWS
    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, template=FROST_D256_DECODE_TEMPLATE if decode_shaped else None)


# (model heads, s_q, diagonal alignment, right bound, the template that must serve it)
D256_DECODE_PINNED_CASES = [
    (32, 1, cudnn.diagonal_alignment.TOP_LEFT,     None, FROST_D256_DECODE_TEMPLATE),   # Qwen3.5 32/2: 16 packed rows
    (32, 2, cudnn.diagonal_alignment.BOTTOM_RIGHT, 0,    FROST_D256_PREFILL_TEMPLATE),  # Qwen3.5 MTP: 32 rows, the unrouted wide tile
    (16, 2, cudnn.diagonal_alignment.BOTTOM_RIGHT, 0,    FROST_D256_DECODE_TEMPLATE),   # Qwen3-Next 16/2 MTP: 16 packed rows
]


@pytest.mark.parametrize("h_q,s_q,diag_align,right_bound,expect_template", D256_DECODE_PINNED_CASES, ids=["qwen35_sq1", "qwen35_sq2_brcm", "qwen3next_sq2_brcm"])
@pytest.mark.L0
def test_sdpa_fwd_paged_decode_d256_qwen35_frost_L0(env_info, h_q, s_q, diag_align, right_bound, expect_template, request, cudnn_handle):
    """Qwen3.5 / Qwen3-Next decode as served: b=32, 32/2 or 16/2 heads, d=256, page
    16, mixed KV lengths up to 4096 -- the S_q = 1 step and the S_q = 2 MTP step
    (bottom-right causal).  Pins WHICH template serves each: the decode tile
    (`decode_d256_f16`) for the 16-row shapes it was built for, and the prefill
    d256 tile for the 32-row Qwen3.5 MTP step -- the 32-column decode tile is
    compiled but not routed (an eager regression: 90 us unsplit / 96-103 us split
    against the prefill tile's 66 us on B200), so a change that routes it must
    come with the numbers and flip this pin.
    """
    _require_frost_sm100()

    rng = random.Random(2009)
    seq_len_kv = [rng.randint(s_q, 4096) for _ in range(32)]
    seq_len_kv[0] = 4096
    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = ExecConfig(
        data_type=torch.bfloat16,
        rng_data_seed=2009,
        rng_geom_seed=2009,
        is_alibi=False,
        is_infer=True,
        is_paged=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=True,
        is_cu_seq_len=False,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=32,
        d_qk=256,
        d_v=256,
        s_q=s_q,
        s_kv=4096,
        h_q=h_q,
        h_k=2,
        h_v=2,
        diag_align=diag_align,
        left_bound=None,
        right_bound=right_bound,
        seq_len_q=[s_q] * 32,
        seq_len_kv=seq_len_kv,
        block_size=16,
    )
    test.cfg.fill_derived_fields()
    test.showConfig((request.node.name, len(D256_DECODE_PINNED_CASES)), request)

    _exec_sdpa_on_frost(test.cfg, request, cudnn_handle, template=expect_template)


@pytest.mark.L0
@pytest.mark.gpu_exclusive
@pytest.mark.parametrize("d_qk,d_v", [(128, 128), (192, 128), (256, 256), (512, 512)])
@pytest.mark.parametrize("binder", ["native", "python"])
def test_sdpa_thd_output_stride_int64(d_qk, d_v, binder, monkeypatch):
    """Every half THD arch/flavor preserves wide strides through device descriptor setup."""
    import math
    from sdpa.frost.frost_test_utils import _dsl_usable

    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if not 100 <= sm < 120:
        pytest.skip("requires an SM100 or SM107 attention engine")
    usable, reason = _dsl_usable()
    if not usable:
        pytest.skip(reason)
    from cudnn.sdpa.fwd.engines import engine_name
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    b, hq, hk, ql, kl = 2, 4, 2, 1, 64
    row_stride = 2**32 + hq * d_v
    if torch.cuda.mem_get_info()[0] < 2 * row_stride + 2**30:
        pytest.skip("wide physical row-stride regression needs 9 GiB free")
    bf16, i32 = cudnn.data_type.BFLOAT16, cudnn.data_type.INT32
    graph = cudnn.pygraph(io_data_type=bf16, intermediate_data_type=cudnn.data_type.FLOAT,
                          compute_data_type=cudnn.data_type.FLOAT, is_override_shape_enabled=True)
    def lengths(name):
        return graph.tensor(dim=[b + 1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=i32, name=name)
    cu_q, cu_kv = lengths("cu_q"), lengths("cu_kv")
    off_q, off_k, off_v, off_o, off_lse = [lengths(n) for n in ("off_q", "off_k", "off_v", "off_o", "off_lse")]
    def operand(name, h, s, d, offset):
        return graph.tensor(dim=[b, h, s, d], stride=[s * h * d, d, h * d, 1], data_type=bf16,
                            name=name).set_ragged_offset(offset)
    q, k, v = operand("q", hq, ql, d_qk, off_q), operand("k", hk, kl, d_qk, off_k), operand("v", hk, kl, d_v, off_v)
    o, stats = graph.sdpa(q=q, k=k, v=v, generate_stats=True, attn_scale=1 / math.sqrt(d_qk),
                         use_padding_mask=True, cu_seq_len_q=cu_q, cu_seq_len_kv=cu_kv,
                         max_total_seq_len_q=b * ql, max_total_seq_len_kv=b * kl)
    o.set_output(True).set_dim([b, hq, ql, d_v]).set_stride([ql * hq * d_v, d_v, hq * d_v, 1]).set_ragged_offset(off_o)
    stats.set_output(True).set_dim([b, hq, ql, 1]).set_stride([ql * hq, 1, hq, 1]).set_data_type(cudnn.data_type.FLOAT).set_ragged_offset(off_lse)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    want = engine_name(arch="sm107" if sm >= 107 else "sm100")
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    graph.select_plan(next(i for i, name in enumerate(names) if name == want or name.startswith(want + "[")))
    graph.check_support()
    graph.build_plans()
    spec = graph._compiled_plans[graph._plan_index]._prepared.spec
    if binder == "native":
        assert spec.native is not None
    elif hasattr(spec, "native"):
        from sdpa.frost import sdpa_binding_reference as binding_reference

        binding_reference.use_reference(spec)  # Compare the independent Python binder with the same compiled host.
    q_buf = torch.zeros((b * ql, hq, d_qk), device="cuda", dtype=torch.bfloat16)
    k_buf = torch.zeros((b * kl, hk, d_qk), device="cuda", dtype=torch.bfloat16)
    v_buf = torch.ones((b * kl, hk, d_v), device="cuda", dtype=torch.bfloat16)
    v_buf[kl:] *= 2
    o_buf = torch.empty_strided((b * ql, hq, d_v), (row_stride, d_v, 1), device="cuda", dtype=torch.bfloat16)
    lse_buf = torch.empty((b * ql, hq), device="cuda", dtype=torch.float32)
    cq = torch.arange(b + 1, device="cuda", dtype=torch.int32) * ql
    ck = torch.arange(b + 1, device="cuda", dtype=torch.int32) * kl
    pack = {q: q_buf, k: k_buf, v: v_buf, o: o_buf, stats: lse_buf, cu_q: cq, cu_kv: ck,
            off_q: cq * hq * d_qk, off_k: ck * hk * d_qk, off_v: ck * hk * d_v,
            off_o: cq * hq * d_v, off_lse: cq * hq}
    workspace = torch.empty(max(graph.get_workspace_size(), 1), device="cuda", dtype=torch.uint8)
    overrides = dict(override_uids=[o.get_uid()], override_shapes=[[b, hq, ql, d_v]],
                     override_strides=[[hq * d_v, d_v, row_stride, 1]])
    expected = torch.ones((b * ql, hq, d_v), device="cuda", dtype=torch.bfloat16)
    expected[ql:] *= 2
    captured = torch.cuda.CUDAGraph()
    try:
        for replay in (False, True):
            if replay:
                with torch.cuda.graph(captured):
                    graph.execute(pack, workspace, **overrides)
                v_buf.mul_(0.5)
                expected.mul_(0.5)
            o_buf.fill_(float("nan"))
            lse_buf.fill_(float("nan"))
            if replay:
                captured.replay()
            else:
                graph.execute(pack, workspace, **overrides)
            torch.testing.assert_close(o_buf, expected, atol=0, rtol=0)
            torch.testing.assert_close(lse_buf, torch.full_like(lse_buf, math.log(kl)), atol=2e-6, rtol=0)
    finally:
        captured.reset()

# # =====================================================================
# # L0 cc 10.7 sweeps with uniform softmax-lever knob sets (the FROST cc 10.7 rows)
# # =====================================================================
#
# graph.sdpa / sdpa_fp8 / sdpa_mxfp8 carry two python-only FORWARD attributes that only the cc 10.7
# FROST rows serve: softmax_precision=HALF (the f16x2 exponent arm of the quantized kernels) and
# attn_scale_prefolded=True (the caller pre-multiplied Q by attn_scale * log2 e; the kernel applies no
# scale).  Each sweep below draws its geometry like the sweep it twins -- own seed and own draw
# order, so no existing case moves -- over the four EXACT kernel flavors, and assigns the case's knob
# set from its CASE INDEX over the set the row serves for that family / flavor / path
# (sdpa/softmax_knobs.py, the served-domain mirror): exactly uniform, and no rng consumed.  A SET
# attribute makes the graph backend-unlowerable, so a set the rows decline would surface as a
# WAIVED skip -- these sweeps turn that into a failure for every non-default set, and every case
# asserts that the row served it (FROST opted in per call, as _exec_sdpa_on_frost does).  The
# quantized sweeps pin unfuse_fma off and the lazy-rescale threshold at the kernels' baked 4.0.
# Volume: the base count of the twinned sweep times _CC107_SWEEP_MULT (env MHAS_CC107_MULT); off a
# cc 10.7 device the sweeps are not collected at all, so every other lane's count is unchanged.

_CC107_SWEEP_MULT = int(os.environ.get("MHAS_CC107_MULT", "4"))
_CC107_FLAVORS = RandomChoice({(128, 128): 1, (192, 128): 1, (256, 256): 1, (512, 512): 1})


def _device_is_cc107():
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return False
    return 107 <= major * 10 + minor <= 119


# Collection gate of the cc 10.7-only functions: the device, or CUDNN_TEST_TIER_ARCH=cc107 (the tiers' override,
# tiers/README.md), so test_tiers.py and `--collect-only` can list the cc 10.7 SMOKE cells on any host; the in-test
# _require_frost_sm107 gate still skips them on a run device that is not cc 10.7.
_CC107_DEVICE = _device_is_cc107() or os.environ.get("CUDNN_TEST_TIER_ARCH", "").strip().lower() == "cc107"


def _cc107_engine(family):
    """The cc 10.7 FROST forward row of a dtype family ("half", "fp8", "mxfp8")."""
    from cudnn.sdpa.fwd.engines import engine_name
    return engine_name(arch="sm107", fp8=family == "fp8", mxfp8=family == "mxfp8")


def _frost_sm107_unavailable_reason(engine):
    """Why the cc 10.7 FROST row ``engine`` would NOT serve a graph here, or None when it would: a
    cc 10.7-11.9 device, the row offered by the manifest -- asked flag-less first (the half and MXFP8 rows are
    default candidates), then with CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 for the rows that answer to the flag
    (the per-tensor FP8 row; the lever sweeps opt in per call) -- and a CuTe DSL at the FROST floor.  This asks
    whether the row EXISTS here; whether it is offered BY DEFAULT is the separate contract of
    _require_frost_sm107_default (the default-walk / pin / decline tests)."""
    major, minor = torch.cuda.get_device_capability()
    if not (107 <= major * 10 + minor <= 119):
        return f"{engine} serves cc 10.7-11.9 only; device is cc {major}.{minor}"
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
        offered = _frost_engines_enabled(engine)
        if not offered:
            mp.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
            offered = _frost_engines_enabled(engine)
    if not offered:
        return f"{engine} is not offered by the manifest even with CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1"
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        return "needs the cutedsl extra (nvidia-cutlass-dsl) at the FROST floor"
    return None


def _require_frost_sm107(engine):
    """Twin of _require_frost_sm100 for the cc 10.7 rows (the SM100 gate keeps its cc 10.0-10.6
    domain): the sweeps below ASSERT routing onto ``engine``, so elsewhere they skip."""
    reason = _frost_sm107_unavailable_reason(engine)
    if reason is not None:
        pytest.skip(f"{reason}: this test asserts FROST routing")


def _cc107_sweep(base_count, rng_seed):
    """Parametrize a cc 10.7-only sweep over ``base_count * _CC107_SWEEP_MULT`` seeded cases (a fresh
    seed: the geometry hash depends on the count, which is why these are new functions rather than a
    multiplier on the existing sweeps).  Off a cc 10.7 device the function is not collected at all
    (pytest honours ``__test__ = False``), so every other lane's collection count is unchanged; the
    in-test _require_frost_sm107 gate still covers a collection device that differs from the run device."""
    def decorate(fn):
        if not _CC107_DEVICE:
            fn.__test__ = False
            return fn
        return pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=base_count * _CC107_SWEEP_MULT, rng_seed=rng_seed), ids=lambda p: f"test{p[0]}")(fn)
    return decorate


def _cc107_only(fn):
    """A cc 10.7-only (non-sweep) function: not collected off a cc 10.7 device, as _cc107_sweep."""
    if not _CC107_DEVICE:
        fn.__test__ = False
    return fn


def _cc107_bshd_dense_strides(cfg, rng):
    """The cc 10.7 half row serves BSHD-physical Q/K/V/O only (its capability row claims no dense-layout
    normalization, unlike the SM100 half row), while the shared draw randomizes the B/H/S stride order
    of dense graphs -- which would route such cases to the backend.  Re-derive the dense strides in
    BSHD order, keeping a padded-stride fuzz (gaps drawn from the geometry rng AFTER the context's own
    draws, so the geometry itself is unchanged; Stats compact token-major).  Ragged configs keep their
    packed layouts, which are BSHD over (H, S, D) already."""
    if cfg.is_ragged:
        return
    elem_align = 16 // cfg.data_type.itemsize
    for name in ("q", "k", "v", "o"):
        gaps = [0, 0, 0, 0]
        if rng.randint(0, 1) == 0:
            gaps = [rng.randint(0, 8) for _ in range(3)] + [elem_align * rng.randint(0, 2)]
        setattr(cfg, f"stride_{name}", get_strides_from_layout(getattr(cfg, f"shape_{name}"), "bshd", gaps))
    cfg.stride_stats = get_strides_from_layout(cfg.shape_stats, "bshd")


def _assign_cc107_knob_set(cfg, test_no, family):
    """Store the case's knob set on the cfg -- so the repro string carries it -- from the set the row
    serves for its family / flavor / path: served[(case - 1) % len(served)].  The default set is stored as
    an EXPLICIT softmax_precision=FLOAT: the same f32 pipeline, but a SET attribute keeps the graph on the
    python engines, so these FROST-asserting sweeps never consult the cuDNN backend (a backend plan build
    is wasted work here, and on cc 10.7 the 9.26 backend crashes planning MXFP8 single-query graphs;
    ``sdpa/fwd/backend_guard.py`` keeps that query out of planning on every known cuDNN build, and the
    default-walk sweep exercises it)."""
    served = served_softmax_knob_sets(family, cfg.d_qk, cfg.d_v, paged=bool(cfg.is_paged), thd=bool(cfg.is_ragged))
    precision, cfg.attn_scale_prefolded = knob_set_for_case(served, test_no[0])
    cfg.softmax_precision = cudnn.data_type.FLOAT if precision is None else precision
    return served


def _expected_softmax_arms(cfg, knob, family):
    """The arm tag (api_dsl.softmax_arms_of) the knob set must have compiled: the exponent ("f16" under
    HALF, else "f32"), "+fold" under the pre-folded scale, and "+fused" when both levers are set on a
    stats-less build and the DSL carries the fused op.  The paged bodies and the single-CTA half legs
    carry no arm constants and read "f32" -- the mirror never draws a lever there."""
    from cudnn.frost.tile_dsl.softmax_f16 import FUSED_SHIFT_CVT_AVAILABLE

    precision, prefolded = knob
    half_exp = precision == cudnn.data_type.HALF
    tag = "f16" if half_exp else "f32"
    if prefolded:
        tag += "+fold"
    if family == "half":
        stats = bool(cfg.is_train or getattr(cfg, "fwd_stats", None) is True)
    else:
        stats = bool(cfg.is_train or getattr(cfg, "fwd_stats", None) is not False)
    if half_exp and prefolded and not stats and FUSED_SHIFT_CVT_AVAILABLE:
        tag += "+fused"
    return tag


def _exec_cc107(test, request, cudnn_handle, family):
    """Run one cc 10.7 case on its FROST row: opt FROST in for the call (and hand the kernels' baked
    lazy-rescale threshold to the engine the way the quantized sweeps do), dispatch to the family's
    expect-frost helper -- which asserts the row's routing tally advanced by one -- and FAIL, not
    skip, when a NON-default knob set waives: the served-domain mirror or the row's claim is wrong.
    A default knob set may still waive for an unrelated reason (a skip, as today)."""
    cfg = test.cfg
    engine = _cc107_engine(family)
    knob = (cfg.softmax_precision, bool(cfg.attn_scale_prefolded))
    non_default = not is_default_knob_set(knob)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        mp.delenv("CUDNN_UNFUSE_FMA", raising=False)
        if cfg.rescale_threshold is not None:
            mp.setenv("CUDNN_RESCALE_THRESHOLD", str(cfg.rescale_threshold))
        try:
            if family == "half":
                _exec_sdpa_on_frost(cfg, request, cudnn_handle, engine=engine)
            elif family == "fp8":
                _exec_sdpa_fp8_expect_frost(cfg, request, cudnn_handle, strict=False, engine=engine)
            else:
                _exec_sdpa_mxfp8_expect_frost(cfg, request, cudnn_handle, strict=False, engine=engine)
        except pytest.skip.Exception as e:
            if non_default and not request.config.option.dryrun:
                layout = "paged THD" if cfg.is_paged else "ragged" if cfg.is_ragged else "padded" if cfg.is_padding else "dense"
                pytest.fail(
                    f"knob set {knob_set_label(knob)} on {request.node.name} ({family}, d={cfg.d_qk}x{cfg.d_v}, {layout}) must run on "
                    f"{engine}, not skip -- the served-domain mirror (sdpa/softmax_knobs.py) or the row's claim is wrong: {e}",
                    pytrace=False,
                )
            raise
    if request.config.option.dryrun:
        return
    # The row served it (asserted above); now the ARM: a lever request that silently traced the default
    # chain would pass every numerical check (the fold is numerically neutral, HALF is within tolerance),
    # so the compiled executor's arm tag is the detector.
    expected = _expected_softmax_arms(cfg, knob, family)
    assert frost_routing.LAST_ARMS == expected, (
        f"knob set {knob_set_label(knob)} on {request.node.name} ({family}, d={cfg.d_qk}x{cfg.d_v}) compiled softmax arms "
        f"{frost_routing.LAST_ARMS!r}, expected {expected!r}"
    )


@_cc107_sweep(512, 10701)
@pytest.mark.L0
def test_sdpa_fwd_cc107_half_L0(env_info, test_no, request, cudnn_handle):
    """Half (f16/bf16) prefill on the cc 10.7 row: test_sdpa_random_fwd_L0's draw over the four exact
    flavors with ragged (THD) added and the forward Stats drawn on / off; knob sets {FLOAT, FLOAT+fold}
    (THD (192, 128): FLOAT only, see sdpa/softmax_knobs.py)."""
    _require_frost_sm107(_cc107_engine("half"))

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=_CC107_FLAVORS,
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 1, "full" : 1}),
        with_sink_token=RandomChoice({True : 1, False : 3}),
        fwd_stats=RandomChoice({True : 1, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    _cc107_bshd_dense_strides(test.cfg, rng)
    _assign_cc107_knob_set(test.cfg, test_no, "half")
    test.showConfig(test_no, request)

    _exec_cc107(test, request, cudnn_handle, "half")


@_cc107_sweep(256, 10702)
@pytest.mark.L0
def test_sdpa_sq1_cc107_half_L0(env_info, test_no, request, cudnn_handle):
    """Half decode on the cc 10.7 row (no decode tile there: s_q == 1 and the MTP s_q <= 4 run the
    prefill bodies, split-KV on d128 / d192x128): test_sdpa_random_sq1_L0's draw over the four exact
    flavors, ragged (packed THD) / padded / full, a causal draw for the MTP rows, Stats on / off;
    knob sets {FLOAT, FLOAT+fold} (THD (192, 128): FLOAT only)."""
    _require_frost_sm107(_cc107_engine("half"))

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        # "s_q=random" draws the MTP step length 1..4 (a quarter of the cases).
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=4, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":3, "s_q=random":1}),
        d_qk_d_v=_CC107_FLAVORS,
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=3, causal=1),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 1, "full" : 1}),
        with_sink_token=RandomChoice({True : 1, False : 3}),
        fwd_stats=RandomChoice({True : 1, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    # Bottom-right alignment needs a causal upper bound; a no-mask draw is plain decode and stays top-left.
    if test.cfg.right_bound is None:
        test.cfg.diag_align = cudnn.diagonal_alignment.TOP_LEFT
    _cc107_bshd_dense_strides(test.cfg, rng)
    _assign_cc107_knob_set(test.cfg, test_no, "half")
    test.showConfig(test_no, request)

    _exec_cc107(test, request, cudnn_handle, "half")


@_cc107_sweep(384, 10703)
@pytest.mark.L0
def test_sdpa_fwd_paged_cc107_half_L0(env_info, test_no, request, cudnn_handle):
    """Paged KV on the cc 10.7 row -- half d128 / d256 with ragged (THD) Q, the paged form that row serves
    (through the shared Blackwell paged bodies; sink-free here -- the sink draw lives in
    test_sdpa_fwd_paged_thd_sink_cc107_half_L0): test_sdpa_fwd_paged_L0's
    prefill-shaped draw (s_q <= 64) over power-of-two page sizes 8..1024 (the paged contract: a
    multiple of 8 dividing the 128-row KV tile, or a multiple of the tile), Stats on / off.  The
    paged bodies apply the scale in-kernel, so the knob set is FLOAT throughout; routing is still
    asserted."""
    _require_frost_sm107(_cc107_engine("half"))

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=64, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10, "s_q>s_kv":3}),
        d_qk_d_v=RandomChoice({(128, 128): 1, (256, 256): 1}),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged" : 1}),
        block_size=RandomBlockSize(min=8, max=1024, with_high_probability=[16, 32, 64, 128]),
        fwd_stats=RandomChoice({True : 1, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_paged = True
    _assign_cc107_knob_set(test.cfg, test_no, "half")
    test.showConfig(test_no, request)

    _exec_cc107(test, request, cudnn_handle, "half")


@_cc107_sweep(384, 10704)
@pytest.mark.L0
def test_sdpa_fp8_fwd_cc107_L0(env_info, test_no, request, cudnn_handle):
    """Per-tensor FP8 prefill + decode (s_q == 1 at weight 2/9, as test_sdpa_fp8_fwd_L0) on the cc 10.7
    FP8 row over the four exact flavors, dense / padded / ragged (THD), e4m3:e5m2 2:1, every output
    dtype, block-scaled O on d128 (the harness folds it off elsewhere), Stats on / off; knob sets
    {FLOAT, HALF} (the fold is declined on per-tensor FP8 by contract)."""
    _require_frost_sm107(_cc107_engine("fp8"))

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[4]),
        s_q_s_kv=RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1": 2, "s_q=s_kv": 5, "s_q=random": 2}),
        d_qk_d_v=_CC107_FLAVORS,
        head_count=RandomHeadGenerator(min=1, max=16, head_group_options=(1, 5, 2)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        o_block_scale=RandomChoice({0: 6, 16: 1, 32: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 1, "padded": 1, "full": 1}),
        with_sink_token=RandomChoice({True : 1, False : 2}),
        fwd_stats=RandomChoice({True : 1, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    # The FROST rows decline unfuse_fma and bake the 4-binade lazy-rescale threshold.
    test.cfg.with_unfuse_fma = False
    test.cfg.rescale_threshold = 4.0
    _assign_cc107_knob_set(test.cfg, test_no, "fp8")
    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    _exec_cc107(test, request, cudnn_handle, "fp8")


@_cc107_sweep(384, 10705)
@pytest.mark.L0
def test_sdpa_mxfp8_fwd_cc107_L0(env_info, test_no, request, cudnn_handle):
    """MXFP8 prefill + decode on the cc 10.7 MXFP8 row (s_q_min = 1, so s_q == 1 is drawn; that row
    runs decode on its prefill bodies): test_sdpa_mxfp8_fwd_L0's draw over the four exact flavors
    (d192x128 is cga2-only, d256 cga1-only -- the heuristics pick), FULL layout, both output dtypes,
    block-scaled O on d128, Stats on / off (the fused HALF+fold arm is stats-less only, so both must be
    drawn); all four knob sets {FLOAT, HALF} x {in-kernel scale, fold}."""
    _require_frost_sm107(_cc107_engine("mxfp8"))

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4),
        s_q_s_kv=RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1": 2, "s_q=s_kv": 5, "s_q=random": 2}),
        d_qk_d_v=_CC107_FLAVORS,
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 3, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float16: 2, torch.bfloat16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        # full-only: sdpa_mxfp8 has no dense seq-len arguments (see test_sdpa_mxfp8_fwd_L0).
        is_ragged_or_padded_or_full=RandomChoice({"full": 1}),
        with_sink_token=RandomChoice({True : 1, False : 2}),
        o_block_scale=RandomChoice({0: 6, 16: 1, 32: 1}),
        fwd_stats=RandomChoice({True : 1, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_mxfp8 = True
    # e5m2 P casts with an attention sink hit a known one-code edge in near-fully-masked causal rows (the
    # kernel's and the reference's fp8 P quantization disagree by one code; the same class as the per-tensor
    # ``known_e5m2_edge_on_rubin`` marker, and independent of the levers) -- keep the sink on e4m3 draws.
    if test.cfg.data_type == torch.float8_e5m2:
        test.cfg.with_sink_token = False
    # The FROST MXFP8 rows serve BSHD-physical Q/K/V/O only (the harness's plain dense draws are
    # BHSD, which routes them to the backend); every case here asserts the row served it.
    test.cfg.bshd_layout = True
    # The FROST rows decline unfuse_fma and bake the 4-binade lazy-rescale threshold.
    test.cfg.with_unfuse_fma = False
    test.cfg.rescale_threshold = 4.0
    _assign_cc107_knob_set(test.cfg, test_no, "mxfp8")
    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    _exec_cc107(test, request, cudnn_handle, "mxfp8")


# ---- P2: SM107 paged THD + sink (paged half attention with sinks for packed serving queries) ----
#
# The cc 10.7 half row serves paged KV through the shared Blackwell d128 / d256 bodies compiled for sm_107a
# (api_dsl._load_sm100_kernel_module's Rubin paged arm, module tag sdpa_fwd_sm107_<d>_paged; the executor's
# kernel_template is the file stem "prefill_d128_f16" / "prefill_d256_f16", which a PAGED graph on cc 10.7 reaches
# through that arm only -- the cc 10.7 siblings raise on paged_kv at module scope -- so the stem is the pinned stand-in
# for the loader's module tag until a cc 10.7-native paged body exists, when the pin must move to the tag).  Those
# bodies carry the
# paged-validated sink fold: a keyless row stores O := 0 / LSE := sink, sinks are indexed per packed query head, so
# PackGQA composes.  Before this block the row declined every paged graph with a sink on cc 10.7; the cuDNN 9.26
# backend serves paged THD + sink at s_q >= 2 and has no engine for a sink at s_q == 1, so paged decode with sinks
# had no plan at all through the common API.  Every must-serve cell is strict (_must_run); sink x split-KV stays
# declined on every row; no cell pins a plan-list order or a heuristic winner (placement and tuning: issue #1472).

_P2_KV24 = [4096, 4095, 4000, 3073, 2048, 1025, 1000, 513, 512, 300, 257, 129, 128, 65, 17, 16, 15, 1, 0, 2047, 1536, 777, 255, 33]
_P2_Q148 = [1] * 8 + [4] * 8 + [8] * 8
_P2_TEMPLATE = {128: "prefill_d128_f16", 256: "prefill_d256_f16"}


def _require_p2_env():
    """Environment gates BEFORE the strict block: zero-length entries need cuDNN >= 9.25; the row must be offered
    on a cc 10.7 device with a DSL that targets sm_107a (_require_frost_sm107)."""
    if cudnn.backend_version() < 92500:
        pytest.skip("zero sequence length SDPA requires cuDNN 9.25.0 or higher")
    _require_frost_sm107(_cc107_engine("half"))


def _p2_cfg(*, dtype, d, h_q, h_kv, b, s_q, seq_len_q, s_kv, seq_len_kv, page, left_bound=None, right_bound=0,
            diag_align=cudnn.diagonal_alignment.BOTTOM_RIGHT, stats=False, stats_layout="token_major", pool="hnd",
            sink=None, poison=True, ragged=True, seed=10720, with_sink=True, cu_seq_len=False, declare_total=False, stats_log2=False):
    cfg = ExecConfig(
        data_type=dtype, rng_data_seed=seed, rng_geom_seed=seed, is_alibi=False, is_infer=True, is_paged=True,
        paged_nan_dead_pages=poison, is_bias=False, is_block_mask=False, is_padding=True, is_cu_seq_len=cu_seq_len,
        is_ragged=ragged, is_dropout=False, is_determin=False, batches=b, d_qk=d, d_v=d, s_q=s_q, s_kv=s_kv,
        h_q=h_q, h_k=h_kv, h_v=h_kv, block_size=page, diag_align=diag_align, left_bound=left_bound,
        right_bound=right_bound, seq_len_q=list(seq_len_q), seq_len_kv=list(seq_len_kv), with_sink_token=with_sink,
        fwd_stats=stats, ragged_stats_layout=stats_layout if ragged else None, paged_pool_layout=pool,
        sink_token_value=sink, with_stats_log2=stats_log2, declare_total_seq_len=declare_total,
        implementation=cudnn.attention_implementation.AUTO,
    )
    if ragged and stats_layout == "head_major":
        from sdpa.random_config import packed_token_capacity
        t_q = packed_token_capacity(cfg.seq_len_q)
        cfg.stride_stats = (h_q * t_q, t_q, 1, 1)   # [h, t] packed Stats, as RandomizationContext derives it
    cfg.fill_derived_fields()
    if not ragged:
        _cc107_bshd_dense_strides(cfg, random.Random(seed))   # the row serves BSHD-physical dense Q / O only
    return cfg


def _p2_keyless_rows(cfg):
    """Packed token indices of the rows that see NO key under the cell's mask -- a request with an empty cache; under
    bottom-right causal (right_bound 0) the first len_q - len_kv rows of a request whose cache is shorter than its query
    count; under top-left causal with a left window the rows whose window ends past the cache.  Such a row holds the
    sink's mass alone and its contract is exact: O := 0, LSE := sink (the kernels SELECT both; see the sink fold in
    sm100/prefill_d128_f16.py)."""
    rows, base = [], 0
    bottom_right = cfg.diag_align == cudnn.diagonal_alignment.BOTTOM_RIGHT
    for len_q, len_kv in zip(cfg.seq_len_q, cfg.seq_len_kv):
        if len_kv == 0:
            keyless = range(len_q)
        elif cfg.right_bound == 0 and bottom_right:
            keyless = range(max(0, len_q - len_kv))
        elif cfg.right_bound == 0 and cfg.left_bound is not None:
            keyless = [i for i in range(len_q) if i - cfg.left_bound > len_kv - 1]
        else:
            keyless = ()
        rows.extend(base + i for i in keyless)
        base += len_q
    return rows


def _p2_exact_keyless_checker(cfg):
    """exec_sdpa tensor_checker asserting the keyless rows BIT-EXACT (torch.equal) before the tolerance compare: O == 0
    on every head and, with natural-log Stats, LSE == the head's sink logit (base-2 Stats scale the sink and are left
    to the tolerance compare).  None when the cell has no keyless row."""
    rows = _p2_keyless_rows(cfg)
    if not rows:
        return None
    from sdpa.fp16 import TensorUid
    shown = f"{rows[:8]}{'...' if len(rows) > 8 else ''}"

    def check(tensors):
        idx = torch.tensor(rows, device="cuda", dtype=torch.int64)
        o = tensors[TensorUid.o][idx]
        assert torch.equal(o, torch.zeros_like(o)), f"keyless rows {shown}: O must be exactly 0 (the sink holds the row's whole mass)"
        with_lse = tensors.get(TensorUid.stats) is not None and not cfg.with_stats_log2
        if with_lse:
            lse = tensors[TensorUid.stats][idx]
            sink = tensors[TensorUid.sink_token].reshape(1, -1, 1).expand_as(lse)
            assert torch.equal(lse, sink), f"keyless rows {shown}: LSE must be exactly the head's sink logit"
        print(f"@@@@ P2 keyless rows exact: {len(rows)} row(s) O == 0" + (" and LSE == sink" if with_lse else ""))

    return check


def _p2_frost(cfg, request, cudnn_handle, *, cga=None, pin=None):
    """Run strictly on the cc 10.7 half row: an explicit softmax_precision=FLOAT (the f32 pipeline the row runs
    anyway) keeps the graph on the python engines as the sibling cc 10.7 sweeps do, so a FROST decline is the
    failure text instead of a backend fallback; _exec_sdpa_on_frost pins the row and the shared paged template;
    _must_run turns a WAIVED skip into a failure.  ``pin`` appends + selects one explicit knob set
    (ExecConfig.plan_pin); the cell's keyless rows are asserted bit-exact (_p2_exact_keyless_checker).  Returns the
    served plan's knobs (asserted unsplit, never a winner)."""
    engine = _cc107_engine("half")
    cfg.softmax_precision = cudnn.data_type.FLOAT
    if pin is not None:
        cfg.plan_pin = {"engine": engine, "knobs": pin}
    with _must_run(request):
        _exec_sdpa_on_frost(cfg, request, cudnn_handle, engine=engine, cga=cga, template=_P2_TEMPLATE[cfg.d_qk], tensor_checker=_p2_exact_keyless_checker(cfg))
    if request.config.option.dryrun:
        return None
    served, knobs = frost_routing.LAST_PLAN
    assert served == engine and knobs is not None, frost_routing.LAST_PLAN
    assert knobs.split_kv in (None, 1), f"sink x split-KV is declined on every row, yet the served plan was {knobs}"
    print(f"@@@@ P2 plan on {engine}: TILE_CGA_M={knobs.cga} PACK_GQA={knobs.pack_gqa} SCHED_POLICY={knobs.sched_policy}")
    return knobs


def _p2_default(cfg, request, cudnn_handle):
    """The user's route: graph.sdpa through the common plan walk with NO opt-in flag and NO python-only attribute,
    so the backend is planned too and placement decides (Rule 9: both routes through the same caller code;
    outputs asserted, the serving engine only printed)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
        with _must_run(request):
            exec_sdpa(cfg, request, cudnn_handle)
    if not request.config.option.dryrun:
        print(f"@@@@ P2 default walk served by {frost_routing.LAST_PLAN[0] or 'the cuDNN backend'}")


@_cc107_sweep(192, 10720)
@pytest.mark.L0
def test_sdpa_fwd_paged_thd_sink_cc107_half_L0(env_info, test_no, request, cudnn_handle):
    """Paged KV on the cc 10.7 half row with the attention sink DRAWN 1:1: ragged (THD) Q over d128 / d256 pools; a
    decode / verification arm (even cases: s_q <= 8, per-request query counts U(0, s_q) so 0 / 1 / 4 / 8-token
    requests mix in one packed batch, b <= 16) and a chunked-prefill arm (odd cases: s_q 64..256, b <= 8); GQA groups
    8 / 16 / MQA / MHA / 12 (12 cannot pack: unpacked); page sizes 8..1024; bottom-right (or top-left) causal / left
    window / right band / band around the diagonal / no mask; the per-batch lengths as seq_len_q/kv or as the cu_seq_len
    (B+1 prefix-sum) form 3:1, the packed totals declared (max_total_seq_len_q/kv) 1:4; Stats off / token-major /
    head-major (head-major on single-request batches only where the logical rows reach the packed capacity -- the b == 1
    guard below and test_sdpa_paged_thd_sink_head_major_batch_one_cc107_L0); HND / NHD pools; dead pool pages
    NaN-poisoned; f16:bf16 1:2.  The sink-free half is the control (served before this change); every case is strict
    and asserts routing onto the row and the f32 arm (_exec_cc107); a served sink graph is unsplit."""
    _require_p2_env()
    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]
    rng = random.Random(geom_seed)
    decode_arm = test_no[0] % 2 == 0
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=16, with_high_probability=[4, 8, 16]) if decode_arm else RandomBatchSize(min=1, max=8, with_high_probability=[1, 4]),
        s_q_s_kv=(RandomSequenceLength(s_q_min=1, s_q_max=8, s_kv_min=16, s_kv_max=8192, s_q_distribution={"s_q=1": 1, "s_q=random": 3})
                  if decode_arm else
                  RandomSequenceLength(s_q_min=64, s_q_max=256, s_kv_min=64, s_kv_max=8192, s_q_distribution={"s_q=random": 3, "s_q=s_kv": 1, "s_q>s_kv": 1})),
        d_qk_d_v=RandomChoice({(128, 128): 1, (256, 256): 1}),
        head_count=RandomChoice({(64, 8, 8): 3, (64, 4, 4): 2, (32, 2, 2): 2, (16, 1, 1): 1, (8, 8, 8): 1, (48, 4, 4): 1}),
        data_type=RandomChoice({torch.float16: 1, torch.bfloat16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=3, left_window_only=1, right_window_only=1, band_around_diag=1, no_mask=1),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1, cudnn.diagonal_alignment.BOTTOM_RIGHT: 2}),
        is_ragged_or_padded_or_full=RandomChoice({"ragged": 3, "cu_ragged": 1}),
        block_size=RandomBlockSize(min=8, max=1024, with_high_probability=[16, 64, 128]),
        with_sink_token=RandomChoice({True: 1, False: 1}),
        fwd_stats=RandomChoice({True: 1, False: 1}),
        ragged_stats_layout=RandomChoice({"token_major": 1, "head_major": 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
    cfg = test.cfg
    if cfg.right_bound is None:   # bottom-right alignment needs a causal upper bound (as test_sdpa_sq1_cc107_half_L0)
        cfg.diag_align = cudnn.diagonal_alignment.TOP_LEFT
    cfg.is_paged = True
    cfg.paged_nan_dead_pages = True
    cfg.paged_pool_layout = "nhd" if rng.random() < 0.5 else "hnd"   # drawn AFTER the context: geometry unchanged
    cfg.declare_total_seq_len = rng.random() < 0.25   # also after the context: the packed totals declared (max_total_seq_len_q/kv)
    if cfg.ragged_stats_layout == "head_major" and cfg.batches == 1 and cfg.s_q < cfg.total_q:
        # A single-request batch declaring head-major Stats (1, h, s_q, 1) over the capacity-strided [h, t_q] buffer
        # with s_q below the 64-rounded packed capacity t_q: check_support accepts it (it checks the packing only) and
        # the native THD binder rejects it at execute ("head-major lse_tensor logical shape must cover bounded packed
        # Q": the declaration's logical rows are bounded against the packed-Q capacity) -- a pre-existing, sink-
        # independent gap on every THD leg, pinned by test_sdpa_paged_thd_sink_head_major_batch_one_cc107_L0 (strict
        # xfail) and named in the tracker's Gaps table.  Only b == 1 trips it: the binder re-describes the harness's
        # [h, t_q] buffer as the declared dims only when that declaration fits the buffer, which is the single-request
        # case; multi-request head-major draws stay head-major and serve.  Keep the b == 1 draw token-major
        # (deterministic, no rng consumed; the geometry is unchanged).
        cfg.ragged_stats_layout = "token_major"
        cfg.stride_stats = get_strides_from_layout(cfg.shape_stats, "bshd")
    _assign_cc107_knob_set(cfg, test_no, "half")
    test.showConfig(test_no, request)
    with _must_run(request):
        _exec_cc107(test, request, cudnn_handle, "half")
    if cfg.with_sink_token and not request.config.option.dryrun:
        assert frost_routing.LAST_PLAN[1].split_kv in (None, 1), frost_routing.LAST_PLAN


_P2_PINNED = [
    # kwargs of _p2_cfg plus "id"; every case runs as a "frost" (row pinned) and a "default" (flag-free common-API) twin.
    dict(id="vllm_verify_d128_64x8_p16", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16),
    dict(id="vllm_verify_d128_64x8_p128_stats", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=128, stats=True),
    dict(id="vllm_verify_d128_64x4_p16_nhd", dtype=torch.bfloat16, d=128, h_q=64, h_kv=4, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16, pool="nhd"),
    dict(id="vllm_verify_d128_64x8_p16_fp16_stats_hm", dtype=torch.float16, d=128, h_q=64, h_kv=8, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16, stats=True, stats_layout="head_major"),
    dict(id="vllm_verify_d256_32x2_p16", dtype=torch.bfloat16, d=256, h_q=32, h_kv=2, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16),
    dict(id="vllm_verify_d256_32x2_p128_stats_nhd", dtype=torch.bfloat16, d=256, h_q=32, h_kv=2, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=128, stats=True, pool="nhd"),
    dict(id="decode_q1_d128_64x8_p16", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=1, seq_len_q=[1] * 24, s_kv=4096, seq_len_kv=_P2_KV24, page=16),
    dict(id="decode_q1_d128_64x8_p128_stats_nhd", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=1, seq_len_q=[1] * 24, s_kv=4096, seq_len_kv=_P2_KV24, page=128, stats=True, pool="nhd"),
    dict(id="decode_q1_d256_32x2_p16_stats", dtype=torch.bfloat16, d=256, h_q=32, h_kv=2, b=24, s_q=1, seq_len_q=[1] * 24, s_kv=4096, seq_len_kv=_P2_KV24, page=16, stats=True),
    dict(id="no_mask_q1_d128_64x8_p16_tl", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=1, seq_len_q=[1] * 24, s_kv=4096, seq_len_kv=_P2_KV24, page=16, right_bound=None, diag_align=cudnn.diagonal_alignment.TOP_LEFT),
    dict(id="left_window128_d128_64x8_p16", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16, left_bound=128),
    dict(id="top_left_window128_d128_64x8_p64_fp16", dtype=torch.float16, d=128, h_q=64, h_kv=8, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=64, left_bound=128, diag_align=cudnn.diagonal_alignment.TOP_LEFT),
    dict(id="gpt_oss_shaped_q1_window128_d128_64x8_sink10", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=4, s_q=1, seq_len_q=[1] * 4, s_kv=2048, seq_len_kv=[2048, 1337, 129, 16], page=16, left_bound=128, stats=True, sink=10.0),
    dict(id="chunked_prefill_d128_32x8_p16_stats", dtype=torch.bfloat16, d=128, h_q=32, h_kv=8, b=4, s_q=128, seq_len_q=[128, 77, 1, 128], s_kv=4096, seq_len_kv=[4096, 3000, 129, 128], page=16, stats=True),
    dict(id="chunked_prefill_d256_16x2_p64_fp16_nhd", dtype=torch.float16, d=256, h_q=16, h_kv=2, b=4, s_q=128, seq_len_q=[128, 77, 1, 128], s_kv=4096, seq_len_kv=[4096, 3000, 129, 128], page=64, pool="nhd"),
    dict(id="mha_zero_q_d128_16x16_p16", dtype=torch.bfloat16, d=128, h_q=16, h_kv=16, b=8, s_q=8, seq_len_q=[1, 0, 4, 8, 0, 1, 4, 8], s_kv=2048, seq_len_kv=[2048, 2048, 1, 0, 16, 17, 128, 129], page=16),
    dict(id="keyless_rows_d128_4x1_sink_m120_stats", dtype=torch.bfloat16, d=128, h_q=4, h_kv=1, b=2, s_q=4, seq_len_q=[4, 4], s_kv=128, seq_len_kv=[1, 128], page=16, stats=True, sink=-120.0),
    dict(id="keyless_rows_d128_4x1_sink_m5_fp16_stats_nhd", dtype=torch.float16, d=128, h_q=4, h_kv=1, b=2, s_q=4, seq_len_q=[4, 4], s_kv=128, seq_len_kv=[1, 128], page=16, stats=True, pool="nhd", sink=-5.0),
    dict(id="keyless_rows_d128_4x1_sink_p3", dtype=torch.bfloat16, d=128, h_q=4, h_kv=1, b=2, s_q=4, seq_len_q=[4, 4], s_kv=128, seq_len_kv=[1, 128], page=16, sink=3.0),
    dict(id="keyless_rows_d128_4x1_sink_p10_stats", dtype=torch.bfloat16, d=128, h_q=4, h_kv=1, b=2, s_q=4, seq_len_q=[4, 4], s_kv=128, seq_len_kv=[1, 128], page=16, stats=True, sink=10.0),
    dict(id="keyless_rows_d256_4x1_sink_m120_stats", dtype=torch.bfloat16, d=256, h_q=4, h_kv=1, b=2, s_q=4, seq_len_q=[4, 4], s_kv=128, seq_len_kv=[1, 128], page=16, stats=True, sink=-120.0),
    dict(id="keyless_rows_d256_4x1_sink_p3_p128_nhd", dtype=torch.bfloat16, d=256, h_q=4, h_kv=1, b=2, s_q=4, seq_len_q=[4, 4], s_kv=128, seq_len_kv=[1, 128], page=128, pool="nhd", sink=3.0),
    dict(id="decode_q1_d128_64x8_p16_stats_hm", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=1, seq_len_q=[1] * 24, s_kv=4096, seq_len_kv=_P2_KV24, page=16, stats=True, stats_layout="head_major"),
    dict(id="vllm_verify_d128_64x8_p16_cu_seq_len", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16, cu_seq_len=True),
    dict(id="vllm_verify_d128_64x8_p16_bounded_stats_hm", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16, stats=True, stats_layout="head_major", declare_total=True),
    dict(id="vllm_verify_d256_32x2_p128_cu_seq_len_bounded_stats", dtype=torch.bfloat16, d=256, h_q=32, h_kv=2, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=128, stats=True, cu_seq_len=True, declare_total=True),
    dict(id="vllm_verify_d128_64x8_p16_stats_log2", dtype=torch.bfloat16, d=128, h_q=64, h_kv=8, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16, stats=True, stats_log2=True),
    dict(id="vllm_verify_d128_16x4_p16_stats_nhd", dtype=torch.bfloat16, d=128, h_q=16, h_kv=4, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16, stats=True, pool="nhd"),
]
_P2_PINNED_CELLS = [pytest.param(case, route, id=f"{case['id']}-{route}") for case in _P2_PINNED for route in ("frost", "default")]


@_cc107_only
@pytest.mark.L0
@pytest.mark.parametrize("case,route", _P2_PINNED_CELLS)
def test_sdpa_paged_thd_sink_verify_cc107_pinned_L0(env_info, case, route, request, cudnn_handle):
    """Serving-shaped paged verify / decode / chunked-prefill graphs with a sink on cc 10.7, pinned so a bisect lands
    on one config: ragged Q over page pools (per-batch lengths, or the cu_seq_len prefix-sum form, with or without the
    packed totals declared), bottom-right (or top-left) causal, mixed per-batch KV lengths (tile and page boundaries, a
    partial last page, one-key and empty caches -> keyless rows: O := 0, LSE := sink, asserted bit-exact on the frost
    route), per-request query counts 1 / 4 / 8 (and 0) in one packed batch, Stats off / token-major / head-major /
    base-2 (head-major on these multi-request batches; the single-request declaration below the packed capacity is
    the detector cell below), HND and NHD pools, f16 / bf16, a left window, GQA 8 / 16 / 4, sink logits at the fold's
    far ends.  Dead pool pages are NaN-poisoned on BOTH routes (neither provider reads a dead table slot).
    route="frost": the row must serve it on the shared paged body (template pinned); route="default": the flag-free
    common-API walk must run and be right (backend or row, whichever placement picks -- printed, not asserted)."""
    _require_p2_env()
    kwargs = dict(case)
    kwargs.pop("id")
    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = _p2_cfg(**kwargs)
    if route == "frost":
        test.cfg.softmax_precision = cudnn.data_type.FLOAT   # the route's attribute (as _p2_frost sets it), printed into the repro
    test.showConfig((request.node.name, len(_P2_PINNED_CELLS)), request)
    if route == "frost":
        _p2_frost(test.cfg, request, cudnn_handle)
    else:
        _p2_default(test.cfg, request, cudnn_handle)


_P2_PINS = [  # d, TILE_CGA_M, PACK_GQA, h_q, h_kv -- every cluster width x packing the sink contract admits
    (128, 2, 0, 64, 8), (128, 2, 1, 64, 8), (128, 1, 0, 64, 8), (128, 1, 1, 64, 8), (128, 2, 1, 32, 2), (128, 2, 1, 16, 4), (128, 1, 1, 16, 4),
    (256, 2, 0, 32, 2), (256, 2, 1, 32, 2), (256, 2, 1, 16, 4),
]


@_cc107_only
@pytest.mark.L0
@pytest.mark.parametrize("d,cga,pack,h_q,h_kv", _P2_PINS, ids=[f"d{d}-cga{c}-pack{p}-{hq}x{hk}" for d, c, p, hq, hk in _P2_PINS])
def test_sdpa_paged_thd_sink_plan_pins_cc107_L0(env_info, d, cga, pack, h_q, h_kv, request, cudnn_handle):
    """Explicit knob admission with a sink (not a winner pin): d128 cga2 and the cga1 two-slab body x PackGQA
    (groups 4, 8 and 16), d256 cga2 x PackGQA (groups 16 and 4), all unsplit, on the verify geometry with Stats.  Each
    set is appended through graph.create_execution_plan and selected strictly, so a decline FAILS and a degraded plan
    cannot pass; TILE_CGA_M / PACK_GQA are read back from the served plan.  The default heuristics never propose
    cga1 with a sink -- a user can still pin it, so it is qualified here (whether to prefer it: issue #1472)."""
    _require_p2_env()
    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = _p2_cfg(dtype=torch.bfloat16, d=d, h_q=h_q, h_kv=h_kv, b=24, s_q=8, seq_len_q=_P2_Q148, s_kv=4096, seq_len_kv=_P2_KV24, page=16, stats=True, seed=10722 + 10 * cga + pack)
    pin = {"TILE_CGA_M": cga, "PACK_GQA": pack, "SPLIT_KV": 1}
    # The route and the pin are part of the case: set before showConfig so the repro string replays them.
    test.cfg.softmax_precision = cudnn.data_type.FLOAT
    test.cfg.plan_pin = {"engine": _cc107_engine("half"), "knobs": pin}
    test.showConfig((request.node.name, len(_P2_PINS)), request)
    knobs = _p2_frost(test.cfg, request, cudnn_handle, cga=cga, pin=pin)
    if knobs is not None:
        assert bool(knobs.pack_gqa) == bool(pack) and knobs.cga == cga, knobs


def _p2_planned_graph(cfg, cudnn_handle):
    """The harness's forward graph, declared by create_forward_graph(plan=False) and planned HERE, so a cell can pin
    a plan and assert the typed decline the harness would otherwise report as a FAIL.  Returns the tensors too
    (keep them alive while the graph is used)."""
    from sdpa import fp16 as fp16_harness
    fp16_harness.validate_config(cfg)
    rng = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)
    _, tensors, _, _ = fp16_harness.allocate_tensors(cfg, rng)
    graph, _ = fp16_harness.create_forward_graph(cfg, tensors, cudnn_handle, plan=False)
    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except cudnn.cudnnGraphNotSupportedError as e:
        # An EMPTY list raises here (the python-only attribute keeps the backend out and the row proposes nothing);
        # the graph is frozen with its facts attached, so the cell can still append and pin its own plan.
        print(f"@@@@ no engine proposed a plan ({e}); the cell pins its own")
    return graph, tensors


@_cc107_only
@pytest.mark.L0
@pytest.mark.parametrize("d", [128, 256])
def test_sdpa_paged_thd_sink_split_declines_cc107_L0(env_info, d, request, cudnn_handle):
    """Sink x split-KV stays declined on the paged THD leg (sm100/split_combine has no sink fold): every plan the row
    proposes for a paged THD + sink graph is unsplit, and pinning SPLIT_KV=2 on its own proposal through
    create_execution_plan is a typed decline at build, never a degraded plan.  (TILE_CGA_M=2 is the row's cluster
    domain under a split; a cga1 pin would decline on the cga domain first and mask the reason under test.)"""
    _require_p2_env()
    cfg = _p2_cfg(dtype=torch.bfloat16, d=d, h_q=32, h_kv=8, b=4, s_q=8, seq_len_q=[1, 4, 8, 8], s_kv=4096, seq_len_kv=[4096, 2048, 129, 16], page=16)
    cfg.softmax_precision = cudnn.data_type.FLOAT   # python engines only: the plan list is the row's proposals
    graph, _tensors = _p2_planned_graph(cfg, cudnn_handle)
    from cudnn.sdpa.fwd.engines import SdpaFwdKnobs
    want = _cc107_engine("half")
    names = [graph.get_plan_name_at_index(i) for i in range(graph.get_execution_plan_count())]
    ours = [i for i, name in enumerate(names) if name == want or name.startswith(want + "[")]
    assert ours, f"the row must propose a plan for paged THD + sink on cc 10.7; plans: {names}"
    for i in ours:
        assert SdpaFwdKnobs.from_public(graph.get_engine_and_knobs_at_index(i)[1]).split_kv in (None, 1), names[i]
    engine_id, knobs = graph.get_engine_and_knobs_at_index(ours[0])
    graph.create_execution_plan(engine_id, {**knobs, cudnn.knob_type.SPLIT_KV: 2, cudnn.knob_type.TILE_CGA_M: 2})
    graph.select_plan(graph.get_execution_plan_count() - 1)
    graph.check_support()   # facts-level: paged THD + sink IS served
    with pytest.raises((NotImplementedError, cudnn.cudnnGraphNotSupportedError), match="sink-free") as decline:
        graph.build_plans()   # knob-level: the pinned split is the typed decline
    print(f"@@@@ P2 split decline: {decline.value}")


@_cc107_only
@pytest.mark.L0
def test_sdpa_paged_sink_dense_queries_decline_cc107_L0(env_info, request, cudnn_handle):
    """The lift is THD-only: dense (BSHD, padded) paged half queries with a sink on cc 10.7 keep a typed decline naming
    the THD requirement (pinned by engine id through create_execution_plan, so the row -- not the walk -- answers)."""
    _require_p2_env()
    cfg = _p2_cfg(dtype=torch.bfloat16, d=128, h_q=32, h_kv=8, b=4, s_q=8, seq_len_q=[1, 4, 8, 8], s_kv=4096, seq_len_kv=[4096, 2048, 129, 16], page=16, ragged=False, poison=False)
    cfg.softmax_precision = cudnn.data_type.FLOAT
    graph, _tensors = _p2_planned_graph(cfg, cudnn_handle)
    from cudnn.engines.manifest import MANIFEST
    engine_id = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()[_cc107_engine("half")]
    graph.create_execution_plan(engine_id, {})
    graph.select_plan(graph.get_execution_plan_count() - 1)
    with pytest.raises((NotImplementedError, cudnn.cudnnGraphNotSupportedError), match="Rubin paged KV requires THD queries") as decline:
        graph.check_support()
    print(f"@@@@ P2 dense decline: {decline.value}")


# The first MULT=4 sweep failure (test340): d256 64/8, one 4-token request in a packed batch of ONE, s_q 5, page 128,
# head-major Stats over the 64-token capacity -- the single-request head-major declaration the native THD binder rejects.
_P2_HEAD_MAJOR_B1 = dict(dtype=torch.bfloat16, d=256, h_q=64, h_kv=8, b=1, s_q=5, seq_len_q=[4], s_kv=1731, seq_len_kv=[889], page=128,
                         right_bound=None, diag_align=cudnn.diagonal_alignment.TOP_LEFT, stats=True, stats_layout="head_major")


@_cc107_only
@pytest.mark.L0
@pytest.mark.parametrize("with_sink", [True, False], ids=["sink", "no_sink"])
@pytest.mark.xfail(strict=True, raises=ValueError, reason="single-request head-major packed Stats below the packed-Q capacity: accepted at check_support, rejected by the native THD binder at execute (SUPPORT_MATRIX_TRACKER.md, Gaps)")
def test_sdpa_paged_thd_sink_head_major_batch_one_cc107_L0(env_info, with_sink, request, cudnn_handle):
    """Detector (strict xfail) for a pre-existing, sink-independent Rule 2 gap on every THD leg: a single-request (b == 1)
    packed batch declaring head-major Stats (1, h, s_q, 1) over a capacity-strided [h, t_q] buffer with s_q < t_q passes
    check_support and build_plans (the row serves it, unsplit) and the first execute raises ``ValueError: cudnn.sdpa:
    head-major lse_tensor logical shape must cover bounded packed Q`` -- the native THD binder bounds the declaration's
    logical rows against the packed-Q capacity, and at execute no plan-walk fallback is possible.  Same failure on
    develop with the sink removed and on nonpaged THD; the token-major twin and every multi-request head-major
    declaration serve.  RED -> fixed when check_support pre-declines the declaration (then make this a typed-decline
    cell) or the binder bounds against the logical rows (then make it a must-serve cell)."""
    _require_p2_env()
    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    test.cfg = _p2_cfg(**_P2_HEAD_MAJOR_B1, with_sink=with_sink)
    test.cfg.softmax_precision = cudnn.data_type.FLOAT
    test.showConfig((request.node.name, 2), request)
    _p2_frost(test.cfg, request, cudnn_handle)


def _p2_fp8_pools_graph(form):
    """graph.sdpa_fp8 over paged K/V pools with a sink (per-tensor FP8 E4M3 in, bf16 O; d128 32/8, b 4, s_q 8, KV 4096, page
    16, bottom-right causal), queries packed ("thd": Q / O carry ragged offsets, the pools stay dense -- the paged THD form
    sdpa.fp16 declares) or dense.  Declared here because the fp8 harness's builder cannot spell ragged Q over paged pools
    (its paged and ragged branches both declare the per-batch length tensors); the operands follow it otherwise: BSHD
    queries, [pages, H, page, D] pools, row-major (b, 1, pages, 1) block tables, scalar (de)scales, the FLOAT softmax."""
    b, h_q, h_kv, s_q, s_kv, d, page = 4, 32, 8, 8, 4096, 128, 16
    e4m3, f32, i32, i64 = cudnn.data_type.FP8_E4M3, cudnn.data_type.FLOAT, cudnn.data_type.INT32, cudnn.data_type.INT64
    table = (s_kv + page - 1) // page
    g = cudnn.pygraph(io_data_type=e4m3, intermediate_data_type=f32, compute_data_type=f32)
    q = g.tensor(dim=(b, h_q, s_q, d), stride=(s_q * h_q * d, d, h_q * d, 1), data_type=e4m3)
    k = g.tensor(dim=(table * b, h_kv, page, d), stride=(page * h_kv * d, page * d, d, 1), data_type=e4m3)
    v = g.tensor(dim=(table * b, h_kv, page, d), stride=(page * h_kv * d, page * d, d, 1), data_type=e4m3)
    k_table = g.tensor(dim=(b, 1, table, 1), stride=(table, table, 1, 1), data_type=i32)
    v_table = g.tensor(dim=(b, 1, table, 1), stride=(table, table, 1, 1), data_type=i32)
    seq_q = g.tensor(dim=(b,), stride=(1,), data_type=i32)
    seq_kv = g.tensor(dim=(b,), stride=(1,), data_type=i32)
    if form == "thd":
        q.set_ragged_offset(g.tensor(dim=(b + 1,), stride=(1,), data_type=i64))
    scalars = [g.tensor(dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=f32) for _ in range(6)]
    sink = g.tensor(dim=(1, h_q, 1, 1), stride=(h_q, 1, 1, 1), data_type=f32)
    o, _stats, _amax_s, amax_o = g.sdpa_fp8(
        q=q, k=k, v=v, descale_q=scalars[0], descale_k=scalars[1], descale_v=scalars[2], scale_s=scalars[3], descale_s=scalars[4], scale_o=scalars[5],
        generate_stats=False, attn_scale=0.125, use_causal_mask=False, use_padding_mask=True, seq_len_q=seq_q, seq_len_kv=seq_kv,
        paged_attention_k_table=k_table, paged_attention_v_table=v_table, paged_attention_max_seq_len_kv=s_kv,
        right_bound=0, diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT, sink_token=sink, softmax_precision=cudnn.data_type.FLOAT,
    )
    o.set_output(True).set_dim((b, h_q, s_q, d)).set_stride((s_q * h_q * d, d, h_q * d, 1)).set_data_type(cudnn.data_type.BFLOAT16)
    if form == "thd":
        o.set_ragged_offset(g.tensor(dim=(b + 1,), stride=(1,), data_type=i64))
    amax_o.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(f32)
    return g


@_cc107_only
@pytest.mark.L0
@pytest.mark.parametrize("form", ["thd", "dense"])
def test_sdpa_paged_sink_fp8_pools_decline_cc107_L0(env_info, form, request, cudnn_handle):
    """The lift is the HALF row's: per-tensor FP8 pools with a sink (graph.sdpa_fp8 over paged K/V, packed THD or dense
    queries, _p2_fp8_pools_graph) keep a typed decline on cc 10.7 at the GRAPH level -- the cc 10.7 FP8 row has no paged
    capability (its `paged_kv = not rubin_row`) -- pinned by engine id through create_execution_plan so the row, not the
    walk, answers (test_sdpa_fwd_dsl_sm107.py::test_sm107_fp8_paged_sink_declines is the host-only facts twin).  The
    explicit FLOAT softmax keeps the backend out of planning (as the quantized cc 10.7 sweeps do), so the walk proposes
    nothing and the pinned plan's check_support is the only answer."""
    _require_p2_env()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")   # the cc 10.7 FP8 row is opt-in: offered so the pin can name it
        graph = _p2_fp8_pools_graph(form)
        graph.validate()
        graph.build_operation_graph()
        try:
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except cudnn.cudnnGraphNotSupportedError as e:
            print(f"@@@@ no engine proposed a plan ({e}); the cell pins its own")
        from cudnn.engines.manifest import MANIFEST
        engine_id = next(f for f in MANIFEST if f.name == "frost_sdpa_fwd").offered_ids()[_cc107_engine("fp8")]
        graph.create_execution_plan(engine_id, {})
        graph.select_plan(graph.get_execution_plan_count() - 1)
        with pytest.raises((NotImplementedError, cudnn.cudnnGraphNotSupportedError), match="paged") as decline:
            graph.check_support()
    print(f"@@@@ P2 fp8 pools decline ({form}): {decline.value}")


# ---- P1: SM107 MXFP8 default provider (graph.sdpa_mxfp8 through the common Graph API, no opt-in flag) ----
# The cc 10.7 MXFP8 row (sdpa_fwd_prefill_sm107_mxfp8) is a DEFAULT manifest candidate that leads the backend on exact
# cc 10.7 (sdpa/fwd/placement.py): a supported dense BSHD MXFP8 graph selects it through the ordinary [A, FALLBACK] walk
# with CUDNN_FRONTEND_ENABLE_FROST_ENGINES deleted and no softmax lever (so the cuDNN backend is planned and ranked behind
# it); explicit diagnostic selection keeps working both ways on that same plan list; the contracts the row does not serve
# decline with a typed error naming the row's reason and the backend's, and nothing is coerced on the way.


def _require_frost_sm107_default(engine):
    """The default-walk contract gate of the pin / decline tests: ``engine`` must be offered WITHOUT
    CUDNN_FRONTEND_ENABLE_FROST_ENGINES (a manifest default slot).  Device and DSL gates skip as usual
    (_require_frost_sm107); a missing default offer on a cc 10.7 device with the DSL is the regression these tests
    exist for, so it FAILS."""
    _require_frost_sm107(engine)
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
        offered = _frost_engines_enabled(engine)
    assert offered, (
        f"{engine} must be a default manifest candidate (engines/manifest.py: EngineSlot without opt_in); "
        f"it is offered only with CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1"
    )


def _exec_mxfp8_default_walk(cfg, request, cudnn_handle):
    """The DEFAULT plan walk of one cc 10.7 MXFP8 case: CUDNN_FRONTEND_ENABLE_FROST_ENGINES DELETED (placement.py ranks,
    the backend is planned like any caller's graph) and no softmax lever, so the row must win the walk, not just serve
    under the flag.  Strict: every draw is inside the row's declared domain, so a harness waive fails."""
    assert cfg.softmax_precision is None and not cfg.attn_scale_prefolded, "the default walk draws no lever"
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
        mp.delenv("CUDNN_UNFUSE_FMA", raising=False)
        mp.setenv("CUDNN_RESCALE_THRESHOLD", str(cfg.rescale_threshold))
        _exec_sdpa_mxfp8_expect_frost(cfg, request, cudnn_handle, strict=True, engine=_cc107_engine("mxfp8"))


@_cc107_sweep(192, 10711)
@pytest.mark.L0
def test_sdpa_mxfp8_fwd_default_walk_cc107_L0(env_info, test_no, request, cudnn_handle):
    """P1: a supported cc 10.7 MXFP8 graph selects the FROST row through the COMMON Graph API -- the opt-in flag
    deleted, no softmax lever (the cuDNN backend IS consulted and ranked), the default [A, FALLBACK] walk.
    test_sdpa_mxfp8_fwd_cc107_L0's draw with its own seed: four exact flavors, FULL, BSHD, e4m3 / e5m2 in, f16 / bf16
    out, every mask, sink on e4m3, Stats on / off, block-scaled O on d128 (admitted flag-less because the harness fold
    follows the manifest), s_q == 1 drawn (the backend planning-crash domain: below the fixed backend the frontend
    records a decline instead of asking, sdpa/fwd/backend_guard.py).  Every case must be served by the row on the f32
    softmax arm.  RED before the flip: the backend serves every BSHD draw and the s_q == 1 cells without a sink take the
    xdist worker down."""
    engine = _cc107_engine("mxfp8")
    _require_frost_sm107_default(engine)  # a manifest regression back to opt-in reads as ONE named failure, not a routing tally

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4),
        s_q_s_kv=RandomSequenceLength(s_q_min=1, s_q_max=8192, s_kv_min=1, s_kv_max=8192, s_q_distribution={"s_q=1": 2, "s_q=s_kv": 5, "s_q=random": 2}),
        d_qk_d_v=_CC107_FLAVORS,
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 3, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float16: 2, torch.bfloat16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        # full-only: sdpa_mxfp8 has no dense seq-len arguments (see test_sdpa_mxfp8_fwd_L0).
        is_ragged_or_padded_or_full=RandomChoice({"full": 1}),
        with_sink_token=RandomChoice({True : 1, False : 2}),
        o_block_scale=RandomChoice({0: 6, 16: 1, 32: 1}),
        fwd_stats=RandomChoice({True : 1, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_mxfp8 = True
    # The e5m2 + sink one-code edge, as the lever sweep: keep the sink on e4m3 draws.
    if test.cfg.data_type == torch.float8_e5m2:
        test.cfg.with_sink_token = False
    # The row's layout; a BHSD declaration is backend territory (Rule 2: no hidden layout copy).
    test.cfg.bshd_layout = True
    # The FROST rows decline unfuse_fma and bake the 4-binade lazy-rescale threshold.
    test.cfg.with_unfuse_fma = False
    test.cfg.rescale_threshold = 4.0
    # No lever on purpose (softmax_precision unset, attn_scale_prefolded False): the backend is planned and ranked.
    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    _exec_mxfp8_default_walk(test.cfg, request, cudnn_handle)
    if not request.config.option.dryrun:
        expected = _expected_softmax_arms(test.cfg, (None, False), "mxfp8")
        assert frost_routing.LAST_ARMS == expected, f"default walk compiled {frost_routing.LAST_ARMS!r}, expected {expected!r}"


# Explicit diagnostic selection on the default (flag-less) plan list: two dense BSHD graphs the before-probe measured with
# backend plans present (d128 causal + sink, d256 dense), fwd_stats=False so the graphs equal the measured contracts; four
# pins each.  ``backend_builds``: whether the backend can BUILD the plan it offers for the graph on the measured backends
# (cuDNN 9.26.0.51 and 9.27.0.28 on cc 10.7): d128 causal + sink runs on eng16; the d256 MXFP8 plans (eng16 / eng3) are
# offered but fail to build (NVRTC: CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED), so a backend pin there must surface
# the backend's own typed decline through the strict pin -- and never run the row.  A backend that builds them is accepted
# too (the pin then runs against the reference like the d128 graph's).
_P1_PIN_GRAPHS = {
    "d128_causal_sink": dict(d=128, s=1024, b=2, h_q=8, h_kv=2, causal=True, sink=True, backend_builds=True),
    "d256_dense": dict(d=256, s=1024, b=2, h_q=8, h_kv=2, causal=False, sink=False, backend_builds=False),
}
# The measured per-plan signature of the backend's d256 / d512 build failure (the strict pin re-raises the plan's own
# text) and the walk-exhaustion header a barred walk raises when every backend entry failed to build.
_P1_BACKEND_BUILD_FAILURE = ("COMPILATION_FAILED", "no plan in the list could be built")
_P1_PINS = ("backend_first", "frost_by_name", "frost_replay", "deselect_frost")
_P1_PIN_CELLS = [(g, p) for g in _P1_PIN_GRAPHS for p in _P1_PINS]


def _p1_pin_cfg(graph_case):
    c = _P1_PIN_GRAPHS[graph_case]
    cfg = ExecConfig(
        data_type=torch.float8_e4m3fn,
        output_type=torch.bfloat16,
        rng_data_seed=10712,
        rng_geom_seed=10712,
        is_alibi=False,
        is_infer=True,
        is_paged=False,
        is_mxfp8=True,
        is_bias=False,
        is_block_mask=False,
        is_padding=False,
        is_ragged=False,
        is_dropout=False,
        is_determin=False,
        batches=c["b"],
        d_qk=c["d"],
        d_v=c["d"],
        s_q=c["s"],
        s_kv=c["s"],
        h_q=c["h_q"],
        h_k=c["h_kv"],
        h_v=c["h_kv"],
        diag_align=cudnn.diagonal_alignment.TOP_LEFT,
        right_bound=0 if c["causal"] else None,
        with_sink_token=c["sink"],
        rescale_threshold=4.0,
        fwd_stats=False,
        bshd_layout=True,
    )
    cfg.fill_derived_fields()
    return cfg


def _p1_is_backend_plan(g, i):
    """Whether plan ``i`` of the ranked list is the backend's (a native engine id, or the delegating backend_heuristics
    entry, which has no (engine_id, knobs) record)."""
    from cudnn.engines.engine_ids import is_backend_engine
    try:
        return is_backend_engine(g.get_engine_and_knobs_at_index(i)[0])
    except NotImplementedError:
        return True


def _p1_plan_pin(pin, frost):
    """A ``plan_pin`` for exec_sdpa_mxfp8: applied to the dense forward graph between create_execution_plans and
    check_support (every failure through pytest.fail -- see exec_sdpa_mxfp8)."""
    from cudnn.engines.engine_ids import FROST_SDPA_FWD_ID_BASE

    def apply(g):
        names = [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]
        fi = [i for i, n in enumerate(names) if n == frost or n.startswith(frost + "[")]
        bi = [i for i in range(len(names)) if _p1_is_backend_plan(g, i)]
        if not fi:
            pytest.fail(f"the default (flag-less) plan list has no {frost} plan: {names}", pytrace=False)
        if not bi:
            pytest.fail(f"the default plan list has no backend plan to pin (cuDNN {cudnn.backend_version_string()}): {names}", pytrace=False)
        if pin == "backend_first":
            g.select_plan(bi[0])
        elif pin == "frost_by_name":
            g.select_plan(fi[0])
        elif pin == "frost_replay":
            engine_id, knobs = g.get_engine_and_knobs_at_index(fi[0])
            if engine_id != FROST_SDPA_FWD_ID_BASE + 16 or not knobs:
                pytest.fail(f"unexpected replay record for the row: {(engine_id, knobs)}", pytrace=False)
            g.create_execution_plan(engine_id, knobs)  # appends; resolves WITHOUT the flag once the slot is a default one
            g.select_plan(g.get_execution_plan_count() - 1)
        else:
            g.deselect_engines([frost])
    return apply


@_cc107_only
@pytest.mark.L0
@pytest.mark.parametrize("graph_case,pin", _P1_PIN_CELLS, ids=[f"{g}-{p}" for g, p in _P1_PIN_CELLS])
def test_sdpa_mxfp8_cc107_explicit_plan_pins_L0(env_info, graph_case, pin, request, cudnn_handle):
    """Explicit diagnostic selection on the default (flag-less) plan list of a dense BSHD cc 10.7 MXFP8 graph: the
    first backend plan pinned by engine id (select_plan, strict), the FROST row pinned by plan name, the row's
    (engine_id, knobs) record replayed through create_execution_plan, and the row barred with deselect_engines so the
    walk lands on the backend.  The harness executes every pin against the MXFP8 reference; Amax_O on a backend pin
    xfails with BACKEND_AMAX_O_ISSUE only (Rule 9: outputs and declines, never plan order).  A backend pin on a graph
    whose backend plans do not build on the measured backends (``backend_builds=False``) must surface the backend's own
    typed build failure through the strict pin / the barred walk -- a decline, never a silent fall-through to the row."""
    frost = _cc107_engine("mxfp8")
    _require_frost_sm107_default(frost)
    cfg = _p1_pin_cfg(graph_case)
    backend_pin = pin in ("backend_first", "deselect_frost")
    backend_builds = _P1_PIN_GRAPHS[graph_case]["backend_builds"]
    key, native = f"frost:{frost}", "native:mxfp8-fwd"
    served = True
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
        mp.delenv("CUDNN_UNFUSE_FMA", raising=False)
        mp.setenv("CUDNN_RESCALE_THRESHOLD", "4.0")
        before = frost_routing.snapshot()
        try:
            with _must_run(request):
                exec_sdpa_mxfp8(cfg, request, cudnn_handle, plan_pin=_p1_plan_pin(pin, frost))
        except pytest.fail.Exception as e:
            # _must_run turned a harness WAIVED skip into a failure.  Legitimate for exactly one case: a backend pin on a
            # graph whose backend plans do not build here -- the strict pin (or the walk with the row barred) surfaces the
            # backend's own typed build failure.  Everything else is the failure it is.
            if not (backend_pin and not backend_builds and any(t in str(e) for t in _P1_BACKEND_BUILD_FAILURE)):
                raise
            served = False
    after = frost_routing.snapshot()
    if pin in ("frost_by_name", "frost_replay"):
        assert after.get(key, 0) == before.get(key, 0) + 1 and frost_routing.LAST_PLAN[0] == frost, frost_routing.LAST_PLAN
    else:
        assert after.get(key, 0) == before.get(key, 0), "a backend pin must not run the row"
        if served:
            assert after.get(native, 0) == before.get(native, 0) + 1 and frost_routing.LAST_PLAN == (None, None), frost_routing.LAST_PLAN
        else:
            assert after.get(native, 0) == before.get(native, 0), "a backend plan that did not build cannot have served"


_P1_FP8_DTYPES = {"e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}
_P1_O_DTYPES = {"bf16": "BFLOAT16", "f16": "HALF", "e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}


def _p1_mxfp8_graph(b=2, hq=8, hk=2, sq=128, skv=2048, d=128, dv=128, page=0, bshd=True, thd=False, stats=True, sink=False, dtype_in="e4m3", o_dtype="bf16", causal=None):
    """One graph.sdpa_mxfp8 graph declared the way test/python/sdpa/mxfp8.py declares it -- dense BSHD / BHSD, paged
    pools behind (b, 1, table, 1) block tables under a padding mask, or the THD twin (ragged offsets on Q/K/V/O/Stats,
    (b, 1, 1, 1) INT32 lengths, dense-capacity SF tensors, token-major packed Stats) -- and the dict of every tensor it
    declared.  ``dtype_in`` e4m3 / e5m2 inputs, ``o_dtype`` bf16 / f16 / e4m3 / e5m2 output, ``causal`` None / "tl" /
    "br" (a zero right band, top-left or bottom-right aligned).  For the decline cells: nothing is ever executed."""
    import math
    ceil_div = lambda a, m: -(-a // m)
    itype = getattr(cudnn.data_type, _P1_FP8_DTYPES[dtype_in])
    g = cudnn.pygraph(io_data_type=itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    T = {}
    d_scale_pad = ceil_div(ceil_div(d, 32), 4) * 4
    dv_pad = ceil_div(dv, 128) * 128

    def sf(dims, name):
        T[name] = g.tensor(dim=dims, stride=(dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1),
                           data_type=cudnn.data_type.FP8_E8M0, reordering_type=cudnn.tensor_reordering.F8_128x4, name=name)

    def ro(name):
        T[name] = g.tensor(dim=(b + 1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT64, name=name)
        return T[name]

    kw = dict(attn_scale=1.0 / math.sqrt(d), generate_stats=stats)
    if thd:
        stride_o = (sq * hq * dv, dv, hq * dv, 1)
        T["q"] = g.tensor(dim=(b, hq, sq, d), stride=(sq * hq * d, d, hq * d, 1), data_type=itype, name="q")
        T["k"] = g.tensor(dim=(b, hk, skv, d), stride=(skv * hk * d, d, hk * d, 1), data_type=itype, name="k")
        T["v"] = g.tensor(dim=(b, hk, skv, dv), stride=(skv * hk * dv, dv, hk * dv, 1), data_type=itype, name="v")
        T["q"].set_ragged_offset(ro("ro_q"))
        T["k"].set_ragged_offset(ro("ro_k"))
        T["v"].set_ragged_offset(ro("ro_v"))
        T["seq_len_q"] = g.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_len_q")
        T["seq_len_kv"] = g.tensor(dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32, name="seq_len_kv")
        sf((b, hq, ceil_div(sq, 128) * 128, d_scale_pad), "sf_q")
        sf((b, hk, ceil_div(skv, 128) * 128, d_scale_pad), "sf_k")
        sf((b, hk, ceil_div(skv, 128) * 4, dv_pad), "sf_v")
        kw.update(use_padding_mask=True, seq_len_q=T["seq_len_q"], seq_len_kv=T["seq_len_kv"])
    else:
        s_q_pad = ceil_div(sq, 128) * 128
        s_kv_pad = ceil_div(skv, 128) * 128
        s_kv_scale_pad = ceil_div(ceil_div(skv, 32), 4) * 4
        b_kv, s_rows = b, skv
        if page:
            table = ceil_div(skv, page)
            b_kv, s_rows = table * b, page
            s_kv_pad = page
            s_kv_scale_pad = ceil_div(page, 32)
        q_stride = (sq * hq * d, d, hq * d, 1) if bshd else (hq * sq * d, sq * d, d, 1)
        stride_o = (sq * hq * dv, dv, hq * dv, 1) if bshd else (hq * sq * dv, sq * dv, dv, 1)
        k_stride = (skv * hk * d, d, hk * d, 1) if (bshd and not page) else (hk * s_rows * d, s_rows * d, d, 1)
        v_stride = (skv * hk * dv, dv, hk * dv, 1) if (bshd and not page) else (hk * s_rows * dv, s_rows * dv, dv, 1)
        T["q"] = g.tensor(dim=(b, hq, sq, d), stride=q_stride, data_type=itype, name="q")
        T["k"] = g.tensor(dim=(b_kv, hk, s_rows, d), stride=k_stride, data_type=itype, name="k")
        T["v"] = g.tensor(dim=(b_kv, hk, s_rows, dv), stride=v_stride, data_type=itype, name="v")
        sf((b, hq, s_q_pad, d_scale_pad), "sf_q")
        sf((b_kv, hk, s_kv_pad, d_scale_pad), "sf_k")
        sf((b_kv, hk, s_kv_scale_pad, dv_pad), "sf_v")
        if page:
            T["seq_len_kv"] = g.tensor(dim=(b,), stride=(1,), data_type=cudnn.data_type.INT32, name="seq_len_kv")
            T["seq_len_q"] = g.tensor(dim=(b,), stride=(1,), data_type=cudnn.data_type.INT32, name="seq_len_q")
            T["k_table"] = g.tensor(dim=(b, 1, table, 1), stride=(table, table, 1, 1), data_type=cudnn.data_type.INT32, name="k_table")
            T["v_table"] = g.tensor(dim=(b, 1, table, 1), stride=(table, table, 1, 1), data_type=cudnn.data_type.INT32, name="v_table")
            kw.update(use_padding_mask=True, seq_len_kv=T["seq_len_kv"], seq_len_q=T["seq_len_q"],
                      paged_attention_k_table=T["k_table"], paged_attention_v_table=T["v_table"], paged_attention_max_seq_len_kv=skv)
    kw.update(q=T["q"], k=T["k"], v=T["v"], descale_q=T["sf_q"], descale_k=T["sf_k"], descale_v=T["sf_v"])
    if sink:
        T["sink"] = g.tensor(dim=(1, hq, 1, 1), stride=(hq, 1, 1, 1), data_type=cudnn.data_type.FLOAT, name="sink")
        kw["sink_token"] = T["sink"]
    if causal:
        kw.update(diagonal_alignment=cudnn.diagonal_alignment.BOTTOM_RIGHT if causal == "br" else cudnn.diagonal_alignment.TOP_LEFT, diagonal_band_right_bound=0)
    o, st, amax = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim((b, hq, sq, dv)).set_stride(stride_o).set_data_type(getattr(cudnn.data_type, _P1_O_DTYPES[o_dtype]))
    T["o"] = o
    if thd:
        o.set_ragged_offset(ro("ro_o"))
    if stats:
        st.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        if thd:
            st.set_dim((b, hq, sq, 1)).set_stride((sq * hq, 1, hq, 1))
            st.set_ragged_offset(ro("ro_stats"))
        else:
            st.set_dim((b, hq, sq, 1)).set_stride((hq * sq, sq, 1, 1))
        T["stats"] = st
    amax.set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    T["amax"] = amax
    return g, T


_P1_ROW_GAP = "the cc 10.7 MXFP8 row serves dense BSHD graphs only; THD and paged MXFP8 are not wired on cc 10.7"
_P1_ROW_EXACT = "serves exact native shapes"
_P1_ROW_BSHD = "Q/K/V/O must be BSHD-physical"
_P1_BACKEND_PAGED = "MXFP8 SDPA over paged K/V caches is not supported by the cuDNN backend."
_P1_GUARD = "backend crashes the process while planning or building single-query MXFP8 SDPA graphs on cc 10.7"
_P1_GUARD_TAIL = "the backend is not consulted for this graph"
_P1_DECLINE_CASES = {  # id: (graph kwargs, expectation, the row's reason -- unread once the expectation is "frost")
    "paged_page64":  (dict(page=64, sq=8),                 "decline",      _P1_ROW_GAP),
    "paged_page128": (dict(page=128, sq=8),                "decline",      _P1_ROW_GAP),   # P6a flips this one value to "frost"
    "thd":           (dict(thd=True),                      "frost_absent", _P1_ROW_GAP),   # the backend may plan THD MXFP8; it NaNs / hangs at execute, so never executed
    "d64":           (dict(d=64, dv=64),                   "frost_absent", _P1_ROW_EXACT),
    "d200":          (dict(d=200, dv=200),                 "frost_absent", _P1_ROW_EXACT),
    "sq1_bhsd":      (dict(sq=1, bshd=False, stats=False), "guard",        _P1_ROW_BSHD),  # the crash domain; LAST so the other cells report first in one run
}


@_cc107_only
@pytest.mark.L0
@pytest.mark.parametrize("case", list(_P1_DECLINE_CASES))
def test_sdpa_mxfp8_cc107_unsupported_requests_decline_L0(case, request):
    """Unsupported cc 10.7 MXFP8 requests decline with a typed error naming the row's reason (and the backend's when it
    declines too); no tensor's dtype / dims / strides and no paging is re-declared on the way; nothing is executed.
    ``decline``: planning raises with both reasons.  ``frost_absent``: the backend may legitimately claim the graph, so
    the row's absence and its reason (graph-level check_support) are asserted and nothing runs; when planning declines
    the error names the row's reason too.  ``guard``: below the fixed backend the frontend answers without consulting
    (or lowering for) the backend; at or above it the backend may serve the BHSD graph.  ``frost``: the row serves the
    request (the P6a hand-off for paged 128 pools, one value in the table): its check_support passes, the flag-less plan
    list carries the row, nothing runs."""
    import re
    from cudnn.engines import manifest
    try:
        from cudnn.sdpa.fwd.backend_guard import SQ1_MXFP8_PLANNING_CRASH_FIXED_IN as fixed_in
    except ImportError:  # no guard module in this tree: every known backend is treated as crashing
        fixed_in = None

    frost = _cc107_engine("mxfp8")
    _require_frost_sm107_default(frost)
    kwargs, expectation, row_reason = _P1_DECLINE_CASES[case]
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", raising=False)
        g, tensors = _p1_mxfp8_graph(**kwargs)
        snap = lambda: {k: (tuple(t.get_dim()), tuple(t.get_stride()), t.get_data_type()) for k, t in tensors.items()}
        before = snap()
        row = next((e for e in manifest.engines_for(g) if e.name == frost), None)
        assert row is not None, f"{frost} is not offered flag-less"
        if expectation == "frost":
            row.check_support(g)  # the row admits the graph: no NotImplementedError
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            names = [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]
            assert any(n == frost or n.startswith(frost + "[") for n in names), names
            assert snap() == before, "planning must not re-declare any tensor"
            return
        error = None
        try:
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        except (cudnn.cudnnGraphNotSupportedError, ValueError) as e:  # ValueError: the python validator's own rejections
            error = e
        with pytest.raises(NotImplementedError, match=re.escape(row_reason)):  # the row's typed reason, always
            row.check_support(g)
        names = [] if error is not None else [g.get_plan_name_at_index(i) for i in range(g.get_execution_plan_count())]
        assert not any(n == frost or n.startswith(frost + "[") for n in names), names
        if expectation == "decline":
            assert isinstance(error, cudnn.cudnnGraphNotSupportedError), error
            msg = str(error)
            assert row_reason in msg and _P1_BACKEND_PAGED in msg and "python engines declined:" in msg, msg
        elif expectation == "frost_absent":
            if error is not None:
                # Both sides named: the backend's own text and, through decline_reasons, the row's reason (d200 / d224:
                # the backend's head-dim validator and the row's exact-shape clause).
                assert isinstance(error, cudnn.cudnnGraphNotSupportedError) and row_reason in str(error), error
            else:
                assert names and all(_p1_is_backend_plan(g, i) for i in range(len(names))), names
        else:  # guard
            if fixed_in is None or cudnn.backend_version() < fixed_in:
                assert isinstance(error, cudnn.cudnnGraphNotSupportedError), error
                assert _P1_GUARD in str(error) and _P1_GUARD_TAIL in str(error) and row_reason in str(error), str(error)
                assert g._lowered_graph is None, "the guard must answer before any C++ lowering"
            else:
                assert error is None and names and g.selected_engine is None
                g.check_support()
                g.build_plans()  # a backend plan builds without buffers; nothing is executed
        assert snap() == before, "a decline must not re-declare any tensor"


# The guard's version bound (sdpa/fwd/backend_guard.py: SQ1_MXFP8_PLANNING_CRASH_FIXED_IN) is a measurement; this detector
# keeps it current over the WHOLE trigger matrix the constant documents: dense / THD, Stats on / off, BSHD / BHSD, batch
# 1 / 2 / 4, KV 128 / 2048 / 4096, no mask / causal top-left / bottom-right, bf16 / f16 / e4m3 / e5m2 O, E4M3 / E5M2 inputs,
# and the d192x128 / d256 / d512 flavors the backend declines -- and over the whole backend LIFECYCLE a serving graph goes
# through: the heuristics query (create_execution_plans), then, with the FROST row barred so the backend's own plan is the
# candidate, check_support and build_plans (nothing executes).  cuDNN 9.28.0 moved the crash from the first stage to the
# last: its heuristics plan every contract, and building the backend's BHSD s_q == 1 plan killed an xdist worker on the cc
# 10.7 CI lane -- a detector that stopped at planning called it clean.  One child interpreter walks the contracts in order
# with the guard DISABLED inside it (the constant set to 0, so backend_guard() answers None), printing BEGIN / STAGE / RESULT
# lines; a crash is rc 139 / -11 of that child, never of the test runner, its stage is the last STAGE line the contract
# printed, and the walk resumes in a fresh child right after the contract that took the previous one down -- one process
# measures a clean backend, one process per crash a crashing one.  The sink contract is the control: every known backend
# plans AND builds it, which proves the probe reaches the backend's heuristics and its plan build at all.
_P1_CRASH_CONTROL = dict(sq=1, stats=False, sink=True)
_P1_CRASH_MATRIX = {
    "dense_sq1": dict(sq=1, stats=False),
    "dense_sq1_stats": dict(sq=1, stats=True),
    "thd_sq1": dict(sq=1, thd=True, stats=False),
    "thd_sq1_stats": dict(sq=1, thd=True, stats=True),
    "bhsd_sq1": dict(sq=1, bshd=False, stats=False),
    "bhsd_sq1_stats": dict(sq=1, bshd=False, stats=True),
    "b1_h1_sq1": dict(b=1, hq=1, hk=1, sq=1, stats=False),
    "b4_sq1": dict(b=4, sq=1, stats=False),
    "kv128_sq1": dict(sq=1, skv=128, stats=False),
    "kv4096_sq1": dict(sq=1, skv=4096, stats=False),
    "causal_tl_sq1": dict(sq=1, stats=False, causal="tl"),
    "causal_br_sq1": dict(sq=1, stats=False, causal="br"),
    "o_f16_sq1": dict(sq=1, stats=False, o_dtype="f16"),
    "o_e4m3_sq1": dict(sq=1, stats=False, o_dtype="e4m3"),
    "o_e5m2_sq1": dict(sq=1, stats=False, o_dtype="e5m2"),
    "e5m2_in_sq1": dict(sq=1, stats=False, dtype_in="e5m2"),
    "e5m2_in_thd_sq1": dict(sq=1, thd=True, stats=False, dtype_in="e5m2"),
    "e5m2_in_bhsd_sq1_stats": dict(sq=1, bshd=False, stats=True, dtype_in="e5m2"),
    "d192x128_sq1": dict(sq=1, d=192, dv=128, stats=False),
    "d256_sq1": dict(sq=1, d=256, dv=256, stats=False),
    "d512_sq1": dict(sq=1, d=512, dv=512, stats=False),
}
_P1_CRASH_PROBE = """
import os, sys
os.environ.pop("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", None)
sys.path[:0] = {paths!r}
import cudnn
import cudnn.sdpa.fwd.backend_guard as guard_module
guard_module.SQ1_MXFP8_PLANNING_CRASH_FIXED_IN = 0  # the guard disabled: the backend is consulted, its plan is built
import test_mhas_v2 as harness
def short(e):
    return (type(e).__name__ + ": " + str(e)[:160]).replace(chr(10), " ")
for name, kwargs in {items!r}:
    print("BEGIN", name, flush=True)
    g, _ = harness._p1_mxfp8_graph(**kwargs)
    try:
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except Exception as e:
        print("RESULT", name, "DECLINED", short(e), flush=True)
        continue
    n = g.get_execution_plan_count()
    print("STAGE", name, "planned", n, flush=True)
    g.deselect_engines([{row!r}])  # the backend's plans alone are the candidates from here on
    try:
        g.check_support()
    except Exception as e:
        print("RESULT", name, "PLANNED", n, "plans; backend plans: none supported --", short(e), flush=True)
        continue
    print("STAGE", name, "supported", flush=True)
    try:
        g.build_plans()
    except Exception as e:
        print("RESULT", name, "PLANNED", n, "plans; backend build failed --", short(e), flush=True)
        continue
    print("RESULT", name, "BUILT", n, "plans; the backend plan built", flush=True)
"""


def _p1_crash_probe(items, timeout=2400):
    """Walk the (name, kwargs) contracts in child interpreters, resuming after every crash: {name: (rc, text, stage)} where rc
    is 0 (text = the RESULT line), 139 / -11 (the child died on this contract; text = the child's last lines, stage = the
    last STAGE it printed: "plan" when none, else "support" after "planned" or "build" after "supported") or the child's
    other exit code (a probe error, neither a plan nor a crash)."""
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    paths = [here, os.path.dirname(os.path.dirname(here))]  # this module; test/python (the sdpa harness, frost_routing)
    results = {}
    start = 0
    while start < len(items):
        batch = items[start:]
        script = _P1_CRASH_PROBE.format(paths=paths, items=batch, row=_cc107_engine("mxfp8"))
        p = subprocess.run([sys.executable, "-c", script], cwd=paths[1], capture_output=True, text=True, timeout=timeout)
        lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip() and "RuntimeWarning" not in l and "AttrBuilder" not in l]
        seen, stage = 0, "plan"
        for l in lines:
            if l.startswith("RESULT "):
                _, name, text = l.split(" ", 2)
                results[name] = (0, text, None)
                seen, stage = seen + 1, "plan"
            elif l.startswith("STAGE "):
                stage = {"planned": "support", "supported": "build"}.get(l.split()[2], stage)
        if seen == len(batch):
            break
        results[batch[seen][0]] = (p.returncode if p.returncode != 0 else 1, "\n".join(lines[-6:]), stage)
        start += seen + 1
    return results


def _p1_crash_verdict(rc, text, stage):
    """One word per contract for the measured record: crash@plan / crash@support / crash@build, built(n), planned(n; ...),
    declined(...), error(rc)."""
    if rc in (139, -11):
        return f"crash@{stage}"
    if rc == 0 and text.startswith("BUILT"):
        return f"built({text.split()[1]})"
    if rc == 0 and text.startswith("PLANNED"):
        n, _, rest = text[len("PLANNED "):].partition("; ")
        reason = rest.split(" -- ", 1)[-1].split(" for ")[0][:60]  # the backend's first clause, not its engine-config dump
        return f"planned({n.split()[0]}; {rest.split(' -- ')[0]}: {reason!r})"
    if rc == 0:
        return f"declined({text[len('DECLINED '):][:60]!r})"
    return f"error(rc {rc})"


@_cc107_only
@pytest.mark.L0
def test_sdpa_mxfp8_cc107_backend_planning_crash_guard_is_current_L0(request):
    """The backend guard's version bound is a measurement, kept current here: with the guard disabled in a child
    interpreter, the installed cuDNN backend plans, supports and builds its own plan for the whole single-query MXFP8
    trigger matrix (one child while it is clean, one more per crash).  Below the recorded fix version (or while none is
    known) at least one contract must take its child down at some stage -- a backend that gets every contract through
    planning AND building cleanly means SQ1_MXFP8_PLANNING_CRASH_FIXED_IN is stale and this test FAILS with the version to
    record (the re-measure procedure sits next to the constant).  At or above the recorded version no contract may crash,
    and the per-contract, per-stage record under "measured" in the run summary is the qualification of that record.
    cc 10.7 only; kept in the FULL tier because it is the tripwire that retires the guard."""
    from cudnn.sdpa.fwd.backend_guard import SQ1_MXFP8_PLANNING_CRASH_FIXED_IN as fixed_in

    if not _device_is_cc107():
        pytest.skip("the backend planning crash is a cc 10.7 measurement")
    if request.config.option.dryrun:
        pytest.skip("dry run mode")
    control = _p1_crash_probe([("sink_control", _P1_CRASH_CONTROL)])["sink_control"]
    assert control[0] == 0 and control[1].startswith("BUILT"), f"the sink control must plan and build on the backend (rc {control[0]}):\n{control[1]}"
    results = _p1_crash_probe(list(_P1_CRASH_MATRIX.items()))
    summary = {name: _p1_crash_verdict(rc, text.splitlines()[-1] if rc == 0 else text, stage) for name, (rc, text, stage) in results.items()}
    crashed = sorted(name for name, (rc, _, _) in results.items() if rc in (139, -11))
    errored = {name: text for name, (rc, text, _) in results.items() if rc not in (0, 139, -11)}
    missing = [name for name in _P1_CRASH_MATRIX if name not in results]
    assert not errored and not missing, f"probe error(s), neither a plan nor a crash: {errored}; unmeasured: {missing}; per contract: {summary}"
    version = cudnn.backend_version_string()
    # The per-contract record is the qualification of the constant: into the run's terminal summary (CI log) through
    # frost_routing.measured -- a passed test's captured stdout never reaches that log.
    frost_routing.measured(
        f"cc 10.7 single-query MXFP8 backend plan / support / build with the guard disabled, cuDNN {version}",
        f"sink_control={_p1_crash_verdict(*control)} " + " ".join(f"{name}={verdict}" for name, verdict in summary.items()),
    )
    if fixed_in is None or cudnn.backend_version() < fixed_in:
        assert crashed, (
            f"cuDNN {version} planned, supported and built every contract of the single-query MXFP8 trigger matrix cleanly: "
            f"SQ1_MXFP8_PLANNING_CRASH_FIXED_IN={fixed_in} is stale -- re-measure the whole matrix per sdpa/fwd/backend_guard.py "
            f"and record {cudnn.backend_version()}; per contract: {summary}"
        )
    else:
        assert not crashed, f"cuDNN {version} still crashes {crashed}: SQ1_MXFP8_PLANNING_CRASH_FIXED_IN={fixed_in} names a build that is not clean; per contract: {summary}"


@pytest.mark.skipif("not config.getoption('--repro')", reason="used with '--repro' only")
@pytest.mark.L0
@pytest.mark.L1
@pytest.mark.L2
@pytest.mark.L3
@pytest.mark.L4
def test_repro(env_info, request, cudnn_handle):
    import ast
    repro_str = request.config.getoption("--repro")
    cfg = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    cfg.cfg = ExecConfig.deserialize(ast.literal_eval(repro_str))
    cfg.showConfig((1,1), request)

    # Set environment variables from config
    if hasattr(cfg.cfg, 'with_unfuse_fma') and cfg.cfg.with_unfuse_fma:
        os.environ["CUDNN_UNFUSE_FMA"] = "1"
    elif "CUDNN_UNFUSE_FMA" in os.environ:
        del os.environ["CUDNN_UNFUSE_FMA"]

    if hasattr(cfg.cfg, 'rescale_threshold') and cfg.cfg.rescale_threshold is not None:
        os.environ["CUDNN_RESCALE_THRESHOLD"] = str(cfg.cfg.rescale_threshold)
    elif "CUDNN_RESCALE_THRESHOLD" in os.environ:
        del os.environ["CUDNN_RESCALE_THRESHOLD"]

    # The softmax levers (softmax_precision / attn_scale_prefolded) are served by the cc 10.7 FROST rows
    # only (the fp8 row opt-in), so a repro of a lever case opts FROST in for the call (as the sweeps that
    # produced it did); a lever-free repro keeps the ambient engine selection.
    lever_case = cfg.cfg.softmax_precision is not None or bool(cfg.cfg.attn_scale_prefolded)
    try:
        with pytest.MonkeyPatch.context() as mp:
            if lever_case:
                mp.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
            if cfg.cfg.is_mxfp8:
                exec_sdpa_mxfp8(cfg.cfg, request, cudnn_handle)
            elif cfg.cfg.data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
                exec_sdpa_fp8(cfg.cfg, request, cudnn_handle)
            else:
                exec_sdpa(cfg.cfg, request, cudnn_handle)
    finally:
        # Clean up environment variables
        if "CUDNN_UNFUSE_FMA" in os.environ:
            del os.environ["CUDNN_UNFUSE_FMA"]
        if "CUDNN_RESCALE_THRESHOLD" in os.environ:
            del os.environ["CUDNN_RESCALE_THRESHOLD"]


@pytest.mark.L0
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("contraction", ["qk", "pv", "dkv", "outer", "empty_batch", "empty_rows", "empty_cols", "empty_reduction"])
def test_sdpa_reference_matmul(contraction, dtype):
    """Unbatched reference GEMMs preserve layouts, GQA broadcasting and autograd."""
    import math
    from sdpa.fp16_ref import _mm, _mm_unbatched

    def values(shape):
        # Small dyadic data makes CPU FP64 an exact oracle, including under TF32.
        return ((torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape) % 17 - 8) / 8).to(device="cuda", dtype=dtype)

    def bshd(s, h, d):
        return values((2, s, h, d)).permute(0, 2, 1, 3).unflatten(1, (2, h // 2))

    if contraction == "qk":
        a, b = bshd(5, 6, 4), bshd(7, 2, 4).transpose(-1, -2)
    elif contraction == "pv":
        a, b = bshd(5, 6, 7), bshd(7, 2, 4)
    elif contraction in ("dkv", "outer"):
        s = 259 if contraction == "dkv" else 1
        a, b = bshd(s, 6, 7).transpose(-1, -2), bshd(s, 6, 4)
    else:
        batch = 0 if contraction == "empty_batch" else 2
        m = 0 if contraction == "empty_rows" else 5
        n = 0 if contraction == "empty_cols" else 7
        k = 0 if contraction == "empty_reduction" else 4
        a, b = values((batch, 2, 3, m, k)), values((batch, 2, 1, k, n))

    a.requires_grad_()
    b.requires_grad_()
    a_cpu = a.detach().cpu().double().requires_grad_()
    b_cpu = b.detach().cpu().double().requires_grad_()
    expected = torch.matmul(a_cpu, b_cpu)
    weights = values(expected.shape)
    expected_grads = torch.autograd.grad(expected, (a_cpu, b_cpu), weights.cpu().double())
    for matmul in (_mm_unbatched, _mm):
        for gradients in (False, True):
            with torch.set_grad_enabled(gradients):
                actual = matmul(a, b)
            assert actual.dtype == dtype and actual.device == a.device
            torch.testing.assert_close(actual.cpu().double(), expected, atol=0, rtol=0)
            if gradients:
                grads = torch.autograd.grad(actual, (a, b), weights)
                for grad, expected_grad in zip(grads, expected_grads):
                    torch.testing.assert_close(grad.cpu().double(), expected_grad, atol=0, rtol=0)
