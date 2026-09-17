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
        # sink_token not supported with s_q==1
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
        # sink_token not supported with s_q==1
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
    graph.build_plans()
    print("Ragged Stats backend plan:", graph.get_plan_name_at_index(backend_plans[0]))
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
    # Do not waive 9.28+ development builds: they must carry the backend fix.
    # A backport to an older version is an XPASS that asks us to retire this marker.
    if cudnn.backend_version() < 92800 and s_q == 1 and h_q > h_kv and ragged_stats:
        request.node.add_marker(pytest.mark.xfail(strict=True, raises=AssertionError, reason="cuDNN < 9.28: ragged decode GQA Stats (NVBug 6783545)"))
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
        # sink_token not supported with s_q==1
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
        # sink_token not supported with s_q==1
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
    with the FROST engines opted in.  Elsewhere they skip instead of failing."""
    major, minor = torch.cuda.get_device_capability()
    if not (100 <= major * 10 + minor <= 106):
        pytest.skip(f"{engine} serves cc 10.0-10.6 only; device is cc {major}.{minor}")
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES") != "1":
        pytest.skip("CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 required: this test asserts FROST routing")


def _exec_sdpa_on_frost(cfg, request, cudnn_handle, engine="sdpa_fwd_prefill_sm100"):
    """exec_sdpa, then assert the FROST engine served the graph: the harness
    tallies the serving engine in frost_routing after build_plans, and a FROST
    decline silently falls through to the native backend, so a green run alone
    proves nothing about routing.  (A WAIVED skip inside exec_sdpa skips before
    the assertion.)"""
    import frost_routing

    key    = f"frost:{engine}"
    before = frost_routing.snapshot().get(key, 0)
    exec_sdpa(cfg, request, cudnn_handle)
    after  = frost_routing.snapshot().get(key, 0)
    assert after == before + 1, f"expected {engine!r} to serve this graph; routing tally: {frost_routing.snapshot()}"


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

FROST_SM100_ROUTING_KEY = "frost:sdpa_fwd_prefill_sm100"

def _skip_unless_frost_sm100_serves():
    """The FROST SM100 f16/bf16 row serves paged decode on pre-Rubin Blackwell
    (cc 10.0-10.6) when the engines are opted in and the CuTe DSL is usable;
    anywhere else the graph would land on the native backend and the routing
    assertion below would be testing the wrong engine."""
    major, minor = torch.cuda.get_device_capability()
    if not (100 <= major * 10 + minor <= 106):
        pytest.skip("FROST paged decode is served by the SM100 row (cc 10.0-10.6) only")
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES") != "1":
        pytest.skip("requires CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 before import cudnn")
    from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
    installed, version = cutedsl_state()
    if not installed or cutedsl_too_old(version):
        pytest.skip("requires the cutedsl extra (nvidia-cutlass-dsl) at the supported version")

def _exec_sdpa_served_by_frost_sm100(cfg, request, cudnn_handle):
    """exec_sdpa, then assert the FROST SM100 row served the graph. A graph the
    validator waives skips inside exec_sdpa before the tally moves; a graph
    FROST declines is served by the native backend and FAILS here instead of
    passing silently (the tally alone asserts nothing)."""
    before = frost_routing.snapshot().get(FROST_SM100_ROUTING_KEY, 0)
    exec_sdpa(cfg, request, cudnn_handle)
    after = frost_routing.snapshot().get(FROST_SM100_ROUTING_KEY, 0)
    assert after == before + 1, f"expected {FROST_SM100_ROUTING_KEY} to serve this graph; routing tally = {frost_routing.snapshot()}"

def _exec_sdpa_paged_decode_on_frost_sm100(cfg, request, cudnn_handle):
    """exec_sdpa on a paged, decode-shaped graph and assert the FROST SM100 row
    served it -- through the placement the heuristics rank by (#1107): a paged
    decode proposal (s_q <= DECODE_MAX_S_Q) leads the backend's plan only on an
    exact (d_qk, d_v) pair in the row's paged_decode_lead_d_shapes; any other
    pair ranks it behind the backend's entries (fwd/heuristics._yields_to_backend).
    So on a claimed pair the FROST plan must rank first and the unpinned walk must
    land on it; on an unclaimed pair the backend must rank first and the FROST
    plan, still offered, is pinned (select_plan through the select_engine helper)
    so the split + combine path runs on every draw. Either way a FROST decline
    cannot pass on the native backend: offers_engine fails it before the walk,
    and a pin is strict (a pinned plan that fails to build raises inside
    exec_sdpa instead of falling through)."""
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS, engine_name
    from cudnn.sdpa.fwd.heuristics import decode_shaped
    from sdpa.frost.frost_test_utils import _is_plan_for, offers_engine, select_engine

    engine = engine_name()
    assert f"frost:{engine}" == FROST_SM100_ROUTING_KEY
    spec = next(s for s in ENGINE_SPECS if s.name == engine)
    assert cfg.is_paged and decode_shaped(cfg), "placement rule under test is the paged decode one"
    claimed = (cfg.d_qk, cfg.d_v) in spec.paged_decode_lead_d_shapes

    def place(graph):
        names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
        assert offers_engine(graph, engine), f"{engine} declined this paged decode graph; plans = {names}"
        if claimed:
            assert _is_plan_for(names[0], engine), f"({cfg.d_qk}, {cfg.d_v}) claims the paged-decode lead: the FROST plan ranks first; plans = {names}"
        else:
            assert not _is_plan_for(names[0], engine), f"({cfg.d_qk}, {cfg.d_v}) carries no lead claim: the backend ranks first at decode; plans = {names}"
            select_engine(graph, engine)

    before = frost_routing.snapshot().get(FROST_SM100_ROUTING_KEY, 0)
    exec_sdpa(cfg, request, cudnn_handle, plan_select=place)
    after = frost_routing.snapshot().get(FROST_SM100_ROUTING_KEY, 0)
    assert after == before + 1, f"expected {FROST_SM100_ROUTING_KEY} to serve this graph ({'claimed pair, unpinned' if claimed else 'pinned'}); routing tally = {frost_routing.snapshot()}"


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=2001), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_decode_split_frost_L0(env_info, test_no, request, cudnn_handle):
    """Paged decode (s_q=1) with the declared KV maximum drawn freely -- almost
    never a multiple of the 128-row KV tile, the FlashInfer spelling -- at a
    small batch, where the heuristic proposes a KV split. Every draw stays
    inside the SM100 row's paged contract (d_qk == d_v <= 256, page size a
    multiple of 8 dividing 128 or a multiple of it, padded, no sink) and is
    served by FROST -- by the unpinned walk on a pair that claims the
    paged-decode lead, pinned behind the backend-first default on any other
    (_exec_sdpa_paged_decode_on_frost_sm100, which asserts the placement
    either way); the reference check covers the split + combine path.
    Own seed and function: widening test_sdpa_fwd_paged_L0 would reshuffle
    every downstream draw of that sweep."""
    _skip_unless_frost_sm100_serves()

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

    _exec_sdpa_paged_decode_on_frost_sm100(test.cfg, request, cudnn_handle)


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
    _skip_unless_frost_sm100_serves()

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

    _exec_sdpa_served_by_frost_sm100(test.cfg, request, cudnn_handle)

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
