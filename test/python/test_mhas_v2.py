# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This script tests cuDNN front-end attention.
The recommended way to run tests:
> pytest -vv -s -rA test_mhas_v2.py
"""

import cudnn
import pytest
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
)
from sdpa.fp16 import exec_sdpa
from sdpa.fp8 import exec_sdpa_fp8
from sdpa.mxfp8 import exec_sdpa_mxfp8
from sdpa.blocked import fetch_blocked_tests
from sdpa.helpers import print_section_begin, print_section_end
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
    if torch.cuda.get_device_capability() == (10, 7) and s_q == 1 and cudnn.backend_version() < 92800:
        # NVBug 6813175: the native ragged s_q == 1 decode codegen fails NVRTC on the Rubin-ranked plan on
        # cuDNN 9.26 / 9.27. Fixed in the 9.28.0.13 nightly (the strict marker XPASSed on every SM107 id,
        # pipeline 69411352): 9.28+ must build it -- backend_version() cannot tell .12 from .13, so the
        # floor is the release line, not the build.
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=cudnn.cudnnGraphNotSupportedError,
                reason="Rubin ranks the native ragged-decode plan first and it fails NVRTC on cuDNN 9.26/9.27 (NVBug 6813175)",
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


def _exec_sdpa_on_frost(cfg, request, cudnn_handle, engine="sdpa_fwd_prefill_sm100", cga=None, template=None):
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
    prefill_d256_f16, tallied as "frost:<engine>:<template>"."""
    keys   = [f"frost:{engine}"] + ([f"frost:{engine}:{template}"] if template else [])
    before = [frost_routing.snapshot().get(k, 0) for k in keys]
    # The assertion is "FROST served it": opt FROST in for the call so the placement
    # tree (sdpa/fwd/placement.py) ranks ours first even on a shard measured behind
    # the backend -- the routing, not the default winner, is under test here.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        exec_sdpa(cfg, request, cudnn_handle)
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
    bottom-right causal / sliding window. No sink: paged + sink is declined by the
    FROST row today (a separate change). Asserts the decode tile served the graph.
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
        with_sink_token=RandomChoice({False : 1}),  # paged + sink stays declined by the engine row
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

    try:
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
