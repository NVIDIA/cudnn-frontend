# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

from gemm.cutedsl.test_grouped_gemm_wrapper_memo import glu_block_scaled_call, mxfp8_inputs
from gemm.cutedsl.test_grouped_gemm_glu_canonical import natural_inputs
from gemm.cutedsl.test_grouped_gemm_quant_canonical import assert_quant_equal, quant_call

pytestmark = pytest.mark.L0


@pytest.fixture(autouse=True)
def require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 is required")


def operation_call(operation, inputs, **kwargs):
    call = glu_block_scaled_call if operation == "glu" else quant_call
    return call(inputs, use_dynamic_sched=True, **kwargs)


@pytest.mark.parametrize("operation", ["glu", "quant"])
@pytest.mark.parametrize("canonical", [False, True])
def test_external_counter_matches_internal_and_rebinds(operation, canonical):
    inputs = mxfp8_inputs([512] * 4)
    if canonical:
        inputs = natural_inputs(inputs)
    reference = operation_call(operation, inputs)
    module = importlib.import_module(f"cudnn.gemm.cutedsl.grouped.{operation}.api")
    memo = getattr(module, f"_{operation}_wrapper_memo")
    results = []
    counters = []
    for _ in range(3):
        counter = torch.zeros(2, dtype=torch.int32, device="cuda")[1:]
        counters.append(counter)
        results.append(operation_call(operation, inputs, scheduler_counter_tensor=counter))
        if len(results) == 1:
            memo_size = len(memo)
        assert len(memo) == memo_size
    torch.cuda.synchronize()
    for counter, result in zip(counters, results):
        assert counter.item() > 0
        assert_quant_equal(result, reference, inputs["valid_m"])


@pytest.mark.parametrize("operation", ["glu", "quant"])
def test_external_counter_changed_routing_and_streams(operation):
    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    operation_call(operation, inputs, scheduler_counter_tensor=counter)
    inputs["padded_offsets_tensor"].copy_(torch.tensor([0, 256, 2048, 2048], dtype=torch.int32, device="cuda"))
    reference = operation_call(operation, inputs)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    counters = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            counter = torch.zeros(1, dtype=torch.int32, device="cuda")
            counters.append(counter)
            results.append(operation_call(operation, inputs, scheduler_counter_tensor=counter))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for result in results:
        assert_quant_equal(result, reference, inputs["valid_m"])


@pytest.mark.parametrize("operation", ["glu", "quant"])
@pytest.mark.parametrize("invalid", ["dtype", "cpu", "empty", "stride", "rank", "static", "discrete"])
def test_external_counter_rejects_unsupported(operation, invalid):
    inputs = natural_inputs(mxfp8_inputs([512] * 4))
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    options = dict(scheduler_counter_tensor=counter, use_dynamic_sched=True)
    if invalid == "dtype":
        options["scheduler_counter_tensor"] = counter.float()
    elif invalid == "cpu":
        options["scheduler_counter_tensor"] = counter.cpu()
    elif invalid == "empty":
        options["scheduler_counter_tensor"] = counter[:0]
    elif invalid == "stride":
        options["scheduler_counter_tensor"] = torch.zeros(4, dtype=torch.int32, device="cuda")[::2]
    elif invalid == "rank":
        options["scheduler_counter_tensor"] = counter.reshape(1, 1)
    elif invalid == "static":
        options["use_dynamic_sched"] = False
    else:
        b, sfb = inputs["b_tensor"], inputs["sfb_tensor"].view(4, -1)
        options.update(
            b_tensor=None,
            sfb_tensor=None,
            b_ptrs=torch.tensor([b[i].data_ptr() for i in range(4)], dtype=torch.int64, device="cuda"),
            sfb_ptrs=torch.tensor([sfb[i].data_ptr() for i in range(4)], dtype=torch.int64, device="cuda"),
            n=512,
            b_dtype=b.dtype,
        )
    call = glu_block_scaled_call if operation == "glu" else quant_call
    with pytest.raises(ValueError, match="scheduler_counter_tensor"):
        call(inputs, **options)
