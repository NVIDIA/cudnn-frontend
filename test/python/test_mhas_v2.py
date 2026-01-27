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
from sdpa.blocked import fetch_blocked_tests
from sdpa.helpers import print_section_begin, print_section_end

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
@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 bprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=256, rng_seed=844), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_bwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=8, max=16),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=192, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":5, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 4, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        is_deterministic=RandomChoice({True : 3, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 fprop tests with s_q=1
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=111), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_sq1_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":100, "s_q=s_kv":1, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=111), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_sq1_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":100, "s_q=s_kv":1, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # =====================================================
# # L0 lean attention, s_kv=513..2048
# # =====================================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=222), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_lean_attn_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=513, s_kv_max=2048, s_q_distribution={"s_q=1":100, "s_q=s_kv":0, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=222), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_lean_attn_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=513, s_kv_max=2048, s_q_distribution={"s_q=1":100, "s_q=s_kv":0, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)

# # ==================================
# # L0 ragged tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_ragged_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=256, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_bwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=8, max=16),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=192, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":5, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        is_deterministic=RandomChoice({True : 3, False : 1}),
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

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=64, s_kv_min=1, s_kv_max=512, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1,32,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
        test.cfg.is_paged = True
        test.cfg.implementation=cudnn.attention_implementation.COMPOSITE  # FIXNOW

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
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=64, s_kv_min=1, s_kv_max=512, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1,32,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
        test.cfg.is_paged = True
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)

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
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
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
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=1),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
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
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=192, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":5, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 4, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        is_bias=RandomChoice({True : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 FP8 fprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=999), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_fwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(s_q_min=1, s_q_max=256, s_kv_min=64, s_kv_max=1024, s_q_distribution={"s_q=1": 3, "s_q=s_kv": 5, "s_q=random": 2}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=192, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 2, "d_qk=random": 1}, with_high_probability=[(64, 64), (128, 128), (192, 128)]),
        head_count=RandomHeadGenerator(min=1, max=16, head_group_options=(1, 5, 2)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 0, "full": 1}),
        stats_layout=RandomChoice({"disabled": 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    exec_sdpa_fp8(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 FP8 bprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=64, rng_seed=998), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_bwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(s_q_min=64, s_q_max=256, s_kv_min=64, s_kv_max=256, s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=128, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 0, "full": 1}),
        stats_layout=RandomChoice({"disabled": 1}),
        is_deterministic=RandomChoice({True: 1, False: 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    exec_sdpa_fp8(test.cfg, request, cudnn_handle)


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
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 1, "full": 0}),
        stats_layout=RandomChoice({"disabled": 1}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16, 32, 64]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed, geom_seed)
        test.cfg.is_paged = True
    test.showConfig(test_no, request)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    exec_sdpa_fp8(test.cfg, request, cudnn_handle)


# # ===================
# # Single repro test
# # ===================

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
    exec_sdpa(cfg.cfg, request, cudnn_handle)
