# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from cudnn.ops import dense_indexer_kl
from phase1_reference import check_reference, reference, run, sample
from test_simplified_sparse_attention import supported_gpu

pytestmark = pytest.mark.L0


@pytest.mark.usefixtures("supported_gpu")
@pytest.mark.parametrize("use_kf", [False, True], ids=["torch", "cutedsl"])
@pytest.mark.parametrize("heads", [16, 32, 96], ids=lambda n: f"bf16-h{n}")
@pytest.mark.parametrize("rows,keys", [(3, 17), (65, 257), (33, 4097)], ids=["tiny", "tile-tail", "split-key"])
def test_forward_backward_reference(use_kf, heads, rows, keys):
    inputs = sample(rows, keys, heads)
    expected, lse = reference(inputs)
    check_reference(run(inputs, use_kf, lse), expected)


def arguments():
    tensors, (starts, ends, valid, denominator) = sample(3, 17, device="cpu", dtype=torch.float64)
    return tensors, dict(
        key_starts=starts,
        key_ends=ends,
        query_valid=valid,
        loss_denominator=denominator,
        teacher_softmax_scale=1 / 16,
        student_softmax_scale=128**-0.5,
        loss_coeff=0.17,
    )


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("query_chunk_size", 0, "chunk sizes"),
        ("key_chunk_size", -1, "chunk sizes"),
        ("key_ends", torch.tensor([18, 17, 0]), "out-of-range"),
        ("loss_denominator", torch.tensor([0.0]), "denominator"),
        ("query_valid", torch.ones(2, dtype=torch.bool), "query_valid"),
    ],
)
def test_invalid_arguments(name, value, match):
    tensors, kwargs = arguments()
    kwargs[name] = value
    with pytest.raises(ValueError, match=match):
        dense_indexer_kl(*tensors, **kwargs)


def test_cpu_reference_and_sbhd_layout():
    tensors, kwargs = arguments()
    tensors = [x.requires_grad_() for x in tensors]
    loss = dense_indexer_kl(*tensors, **kwargs, query_chunk_size=2, key_chunk_size=7)
    expected, _ = reference((tensors, tuple(kwargs[n] for n in ("key_starts", "key_ends", "query_valid", "loss_denominator"))))
    torch.testing.assert_close(loss.float(), expected["loss"], rtol=1e-6, atol=1e-8)
    sbhd = [tensors[0][:, None], tensors[1][:, None], tensors[2][:, None, None], tensors[3][:, None, None]]
    torch.testing.assert_close(dense_indexer_kl(*sbhd, **kwargs), loss)


def test_old_dsl_declines_before_kernel_import(monkeypatch):
    from cudnn.simplified_sparse_attention.cutedsl import phase1

    floors = []

    def too_old(name, *, floor):
        floors.append(floor)
        return "dense_indexer_kl requires CuTe DSL >=4.8.0; installed 4.7.1"

    monkeypatch.setattr(phase1, "cutedsl_requirement_error", too_old)
    monkeypatch.setattr(phase1, "_KF_KERNEL", None)

    def unexpected_import(*args, **kwargs):
        pytest.fail("Imported a kernel before the version gate")

    monkeypatch.setattr(phase1.importlib, "import_module", unexpected_import)
    with pytest.raises(ImportError, match="installed 4.7.1"):
        phase1._phase1_kf_kernel_module()
    assert floors == [(4, 8, 0)]


@pytest.mark.usefixtures("supported_gpu")
def test_capture_declines_before_launch(monkeypatch):
    from cudnn.simplified_sparse_attention.cutedsl import phase1

    tensors, kwargs = arguments()
    tensors = [x.cuda() for x in tensors]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(phase1, "_canonicalize_inputs", lambda *args: pytest.fail("Processed inputs during capture"))
    with pytest.raises(NotImplementedError, match="external CUDA graph capture"):
        dense_indexer_kl(*tensors, **kwargs)


@pytest.mark.usefixtures("supported_gpu")
@pytest.mark.parametrize(
    "case,match",
    [
        ("missing_lse", "requires teacher_lse"),
        ("lse_shape", "teacher_lse must be"),
        ("dtype", "requires BF16"),
        ("scale", "teacher_softmax_scale"),
        ("heads", "Hq in"),
    ],
)
def test_fused_contract_before_import(monkeypatch, case, match):
    from cudnn.simplified_sparse_attention.cutedsl import phase1

    tensors, kwargs = arguments()
    tensors = [x.to(device="cuda", dtype=torch.bfloat16) for x in tensors]
    kwargs.update(use_kf=True, teacher_lse=torch.zeros((3, 16), device="cuda"))
    if case == "missing_lse":
        kwargs["teacher_lse"] = None
    elif case == "lse_shape":
        kwargs["teacher_lse"] = kwargs["teacher_lse"].T
    elif case == "dtype":
        tensors[0] = tensors[0].float()
    elif case == "scale":
        kwargs["teacher_softmax_scale"] = 0.5
    else:
        tensors[0] = tensors[0][:, :8]
    monkeypatch.setattr(phase1, "_phase1_kf_kernel_module", lambda: pytest.fail("Imported a kernel for invalid inputs"))
    with pytest.raises((ValueError, NotImplementedError), match=match):
        dense_indexer_kl(*tensors, **kwargs)


@pytest.mark.usefixtures("supported_gpu")
def test_side_stream_without_host_sync():
    inputs = sample(65, 257)
    expected, lse = reference(inputs)
    run(inputs, True, lse)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    _, kwargs = arguments()
    tensors = inputs[0]
    kwargs.update(zip(("key_starts", "key_ends", "query_valid", "loss_denominator"), inputs[1]))
    with torch.cuda.stream(stream):
        iq, ik = (x.clone().requires_grad_() for x in tensors[2:])
        torch.cuda.set_sync_debug_mode("error")
        try:
            loss = dense_indexer_kl(tensors[0], tensors[1], iq, ik, **kwargs, use_kf=True, teacher_lse=lse)
            dq, dk = torch.autograd.grad(loss * 0.7, (iq, ik))
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    check_reference(dict(loss=loss, dq_idx=dq, dk_idx=dk), expected)
