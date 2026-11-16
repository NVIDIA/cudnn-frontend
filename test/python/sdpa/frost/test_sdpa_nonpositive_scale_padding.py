# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit padding is a legal control for the unmasked scale-sign defect."""

import cudnn
import pytest
import torch

from cudnn.engines.engine_ids import is_python_engine
from test_sdpa_nonpositive_scale_backend import _check_outputs, _execute, _inputs, native_only  # noqa: F401

pytestmark = pytest.mark.L0


def _reference(data, scale, causal, lengths):
    q, k, v, do = [x.double() for x in data]
    logits = (q @ k.transpose(-1, -2)) * scale
    logits = logits.masked_fill(torch.arange(k.shape[2], device=k.device).reshape(1, 1, 1, -1) >= lengths, -float("inf"))
    if causal:
        logits = logits.masked_fill(torch.ones(q.shape[2], k.shape[2], device=q.device, dtype=torch.bool).triu(1), -float("inf"))
    stats = torch.logsumexp(logits, -1, keepdim=True).float().transpose(1, 2).contiguous().transpose(1, 2)
    o = (torch.softmax(logits, -1) @ v).float().to(data[0].dtype).transpose(1, 2).contiguous().transpose(1, 2)
    p = torch.exp(logits - stats.double())
    ds = p * (do @ v.transpose(-1, -2) - (o.double() * do).sum(-1, keepdim=True)) * scale
    return o, stats, tuple(x.float() for x in (ds @ k, ds.transpose(-1, -2) @ q, p.transpose(-1, -2) @ do))


def _graph(data, o, stats, scale, causal, deterministic, lengths, ragged):
    dtype = cudnn.data_type.HALF if data[0].dtype == torch.float16 else cudnn.data_type.BFLOAT16
    g = cudnn.pygraph(io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    pack = {}

    def tensor(t):
        x = g.tensor_like(t)
        pack[x] = t
        if ragged:
            offsets = torch.tensor([0, t.stride(0), 2 * t.stride(0)], device="cuda", dtype=torch.int64)
            offset = g.tensor_like(offsets)
            pack[offset] = offsets
            x.set_ragged_offset(offset)
        return x

    q, k, v, do = data
    Q, K, V, O, DO, ST = [tensor(t) for t in (q, k, v, o, do, stats)]
    lq = torch.full((2, 1, 1, 1), q.shape[2], device="cuda", dtype=torch.int32)
    LQ, LK = g.tensor_like(lq), g.tensor_like(lengths)
    pack.update({LQ: lq, LK: lengths})
    if isinstance(scale, torch.Tensor):
        descriptor = g.tensor_like(scale)
        pack[descriptor] = scale
        scale = descriptor
    gradients = g.sdpa_backward(
        q=Q,
        k=K,
        v=V,
        o=O,
        dO=DO,
        stats=ST,
        attn_scale=scale,
        use_causal_mask=causal,
        use_padding_mask=True,
        seq_len_q=LQ,
        seq_len_kv=LK,
        use_deterministic_algorithm=deterministic,
    )
    outputs = [torch.full_like(t, float("nan")) for t in (q, k, v)]
    for out, carrier, source in zip(gradients, outputs, (Q, K, V)):
        out.set_output(True).set_dim(carrier.shape).set_stride(carrier.stride())
        if ragged:
            out.set_ragged_offset(source.ragged_offset)
        pack[out] = carrier
    g.validate()
    g.build_operation_graph()
    return g, pack, outputs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.0, -0.125, 0.125, "tensor"], ids=["zero", "negative", "positive", "runtime"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("ragged", [False, True], ids=["padded", "thd"])
def test_explicit_padding_keeps_correct_native_paths(dtype, scale, causal, deterministic, ragged):
    cc = torch.cuda.get_device_capability()
    if ragged and cc[0] in (8, 12):
        pytest.skip("native SM8X/SM12X THD backward does not support packed LSE")
    if deterministic and cc[0] == 10 and cudnn.backend_version() < 91800:
        pytest.skip("Blackwell deterministic backward requires cuDNN 9.18 or newer")
    value = 0.125 if scale == "tensor" else scale
    data, _, _, _ = _inputs(dtype, value, causal, geometry=(2, 129, 192, 128, 2, 2), oracle_dtype=torch.float64)
    lengths = torch.tensor([129, 192], device="cuda", dtype=torch.int32).reshape(2, 1, 1, 1)
    o, stats, grads = _reference(data, value, causal, lengths)
    if not ragged:
        # Older backends require dense BHSD Stats; packed THD keeps token-major
        # LSE instead. The values and the backward oracle stay identical.
        stats = stats.contiguous()
    tensor_scale = torch.tensor([value], device="cuda", dtype=torch.float32).reshape(1, 1, 1, 1) if scale == "tensor" else None
    argument = tensor_scale if tensor_scale is not None else value
    make = lambda: _graph(data, o, stats, argument, causal, deterministic, lengths, ragged)
    g, _, _ = make()
    try:
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except cudnn.cudnnGraphNotSupportedError:
        records = []
    else:
        records = [g.get_engine_and_knobs_at_index(i) for i, p in enumerate(g.plans) if not is_python_engine(p.engine_id)]
    if not records:
        if ragged and deterministic and cc == (10, 7):
            control, _, _ = _graph(data, o, stats, 0.125, causal, deterministic, lengths, ragged)
            try:
                control.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            except cudnn.cudnnGraphNotSupportedError:
                control_records = []
            else:
                control_records = [p for p in control.plans if not is_python_engine(p.engine_id)]
            if not control_records:
                pytest.skip("installed native SM107 deterministic THD backward offers no positive-scale control plan")
        # Blackwell's packed path uses engine 5, independently unsafe even with
        # causal/padding masks. Do not count a native support gap as a skip.
        assert ragged and cc[0] == 10 and (tensor_scale is not None or value <= 0)
        assert 5 in g._backend_engine_exclusions()
        return
    for engine, knobs in records:
        selected, pack, outputs = make()
        selected.create_execution_plan(engine, knobs)
        for current in (0.125, 0.0, -0.125) if tensor_scale is not None else (value,):
            expected_o, expected_stats, grads = _reference(data, current, causal, lengths)
            if tensor_scale is not None:
                tensor_scale.fill_(current)
                o.copy_(expected_o)
                stats.copy_(expected_stats)
            for output in outputs:
                output.fill_(float("nan"))
            _execute(selected, pack)
            for index, (result, reference) in enumerate(zip(outputs, grads)):
                for batch, length in enumerate((129, 129) if index == 0 else (129, 192)):
                    _check_outputs([result[batch, :, :length]], [reference[batch, :, :length]], dtype)
