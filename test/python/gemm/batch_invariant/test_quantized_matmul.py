# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from determinism_utils import assert_bitwise, assert_golden, assert_replays
from .matmul import native_matmul
from .quantized_matmul import FORMATS, TENSOR_SCALE, build_quantized, make_case

pytestmark = pytest.mark.L0


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_quantized_golden(gemm_device, fmt, dtype):
    case = make_case(fmt, gemm_device, exact=True)
    out = assert_replays(build_quantized(case, dtype), repeats=2)
    x, w = case.decoded()
    ref = x @ w.transpose(-2, -1)
    if case.alpha is not None:
        ref *= case.alpha
    assert_bitwise(out, {"out": ref.to(dtype)}, "independent FP64 reference")
    a, b, sa, sb = case.inputs()
    inputs = {"a": a, "b": b, "sa": sa, "sb": sb}
    if case.alpha is not None:
        inputs["alpha"] = case.alpha
    name = f"batch_invariant_{fmt}_{'bf16' if dtype == torch.bfloat16 else 'fp16'}"
    assert_golden(name, inputs, out, baseline_dir=Path(__file__).with_name("goldens"))


@pytest.mark.parametrize("fmt", FORMATS)
def test_quantized_direct_training_preserves_rows(gemm_device, fmt):
    case = make_case(fmt, gemm_device, m=257, n=264, k=512)
    training = build_quantized(case, torch.float16)()["out"]
    indices = [256, 0, 256, 17]
    rows = case.select_rows(indices)
    actual = build_quantized(rows, torch.float16)()
    assert_bitwise(actual, {"out": training[..., indices, :]}, "repacked and repeated training rows")


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_quantized_forward_with_native_linear_backward(gemm_device, fmt, dtype):
    case = make_case(fmt, gemm_device, n=128, m=8, exact=True)
    y = build_quantized(case, dtype)()["out"]
    x, w = case.decoded()
    # Differentiate the linear projection in its effective (dequantized) operands.
    # The FP32 alpha multiplies the upstream gradient as well.
    dy = y * (1 / 128)
    if case.alpha is not None:
        dy *= case.alpha
    if dy.ndim == 2:
        dy = dy.unsqueeze(0)
    x = x.reshape(-1, x.size(-2), x.size(-1)).to(dtype)
    w = w.reshape(-1, w.size(-2), w.size(-1)).to(dtype)
    dx = native_matmul(dy, w)
    dw = native_matmul(dy.transpose(-1, -2), x)
    for got, expected in [(dx, dy.double() @ w.double()), (dw, dy.double().transpose(-1, -2) @ x.double())]:
        torch.testing.assert_close(got, expected.to(dtype), rtol=0.008 if dtype == torch.bfloat16 else 0.001, atol=0.002)


def _flashinfer_call(case, dtype, original):
    import flashinfer.gemm
    from flashinfer.gemm.batch_invariant import api

    a, b, sa, sb = case.inputs()
    kwargs = {}
    if case.fmt in TENSOR_SCALE:
        name = "bmm_fp8"
        kwargs["dtype"] = dtype
    else:
        kwargs["out_dtype"] = dtype
        if case.fmt == "groupwise":
            name, b = "gemm_fp8_nt_groupwise", b.T
            kwargs["scale_major_mode"] = "K"
        elif case.fmt == "mxfp8":
            name = "mm_mxfp8"
        elif case.fmt == "mxfp8_mxfp4":
            name = "mm_mxfp8_mxfp4"
            kwargs["alpha"] = case.alpha
        else:
            name = "mm_fp4"
            kwargs.update(block_size=case.block, use_nvfp4=case.fmt == "nvfp4", alpha=case.alpha)
    if original:
        entry = getattr(api, name + "_batch_invariant")
    else:
        entry = getattr(flashinfer.gemm, name)
        kwargs["batch_invariant"] = True
        if name != "mm_mxfp8_mxfp4":
            kwargs["backend"] = "auto"
    return entry(a, b, sa, sb, **kwargs)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype,k", [(torch.bfloat16, 512), (torch.float16, 8192)], ids=["bf16-k512", "fp16-k8192"])
def test_quantized_flashinfer_rollout_matches_training(gemm_device, fmt, dtype, k):
    try:
        import flashinfer
    except ModuleNotFoundError as exc:
        if exc.name != "flashinfer":
            raise
        pytest.skip("FlashInfer is required for the rollout integration tests")
    case = make_case(fmt, gemm_device, m=135, n=128, k=k)
    training = build_quantized(case, dtype)()["out"]
    rows = [130, 0, 130, 7]
    for indices in ([130], rows):
        rollout_case = case.select_rows(indices)
        rollout = {"out": _flashinfer_call(rollout_case, dtype, original=False)}
        assert_bitwise(rollout, {"out": training[..., indices, :]}, "rollout vs direct cuDNN training rows")
        assert_bitwise(rollout, {"out": _flashinfer_call(rollout_case, dtype, original=True)}, "rollout vs original FlashInfer kernel")


@pytest.mark.parametrize("fmt", ["e4m3", "groupwise", "mxfp8", "nvfp4", "mxfp4", "mxfp8_mxfp4"])
@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
def test_quantized_pipeline_refill(gemm_device, fmt):
    from cudnn.gemm.batch_invariant._quantized.config import schedule_for
    from warm_stress import StressCase, assert_warm_stress, stress_validation

    m, n, k = 1024, 4096, 8192
    case = make_case(fmt, gemm_device, m=m, n=n, k=k, batch=1, exact=True)
    x, w = case.decoded()
    expected = x @ w.transpose(-1, -2)
    if case.alpha is not None:
        expected *= case.alpha
    run = build_quantized(case, torch.bfloat16)
    outputs = run()
    props = torch.cuda.get_device_properties(gemm_device)
    in_fmt = {"groupwise": "e4m3_block128", "mxfp8_mxfp4": "mxfp4w8a"}.get(fmt, fmt)
    op = SimpleNamespace(m=m, n=n, k=k, batch=1, in_fmt=in_fmt, b_major="k")
    schedule = schedule_for(op, gemm_device.index)
    cluster_size = schedule.knobs.cluster_m * schedule.knobs.cluster_n
    assert schedule.chunks == 1
    # Even if A alone occupied all opt-in shared memory, its stages would wrap.
    rows_per_block = schedule.knobs.inst_m // (2 if schedule.knobs.use_2cta_instrs else 1)
    bytes_per_k = rows_per_block * case.x.element_size() * case.x.size(-1) // k
    resident_k = props.shared_memory_per_block_optin // bytes_per_k
    assert_warm_stress(
        StressCase(
            f"batch_invariant_{fmt}_pipeline",
            run,
            outputs,
            thread_blocks=schedule.resident_clusters * cluster_size,
            refills_per_launch=(k - 1) // resident_k,
            reference={"out": expected.to(torch.bfloat16)},
        ),
        validation=stress_validation(gemm_device, cluster=cluster_size),
    )
