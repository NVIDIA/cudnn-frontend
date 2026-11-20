# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D512 graph/standalone admission parity without allocating the large tensors."""

import pytest
import torch

import cudnn
from cudnn.api_base import TensorDesc
from cudnn.sdpa.bwd import engines
from cudnn.sdpa.bwd.heuristics import recommend
from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

pytestmark = pytest.mark.L0

_DTYPES = {torch.float16: cudnn.data_type.HALF, torch.bfloat16: cudnn.data_type.BFLOAT16, torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3}
_ROUTES = [
    pytest.param(mx, grad, id=f"{'mxfp8' if mx else 'fp8'}-{str(grad).split('.')[-1]}")
    for mx, grads in ((False, (torch.float16, torch.bfloat16, torch.float8_e4m3fn)), (True, (torch.float16, torch.bfloat16)))
    for grad in grads
]
_CASES = [
    pytest.param({"b": 65535}, None, id="batch-limit"),
    pytest.param({"b": 65536}, "grid", id="batch-over"),
    pytest.param({"h": 65535}, None, id="heads-limit"),
    pytest.param({"h": 65536}, "grid", id="heads-over"),
    pytest.param({"sq": 2**31 - 127}, "tile rounding", id="query-rounding-over"),
    pytest.param({"sk": 2**31 - 127}, "tile rounding", id="kv-rounding-over"),
    pytest.param({"b": 65535, "h": 65535}, "work-id range", id="work-product-over"),
    pytest.param({"b": 4096, "h": 4096, "sq": 128}, "Stats index", id="stats-over"),
    pytest.param({"b": 128, "h": 128, "sq": 130944}, None, id="rounded-delta-limit"),
    pytest.param({"b": 128, "h": 128, "sq": 130945}, "rounded delta index", id="rounded-delta-over"),
    pytest.param({"h": 32768, "sk": 32767}, None, id="tma-stride-below"),
    pytest.param({"h": 32768, "sk": 32768}, "TMA", id="tma-stride-at-limit"),
    pytest.param({"sq": 129, "sk": 257, "h": 4, "hk": 2, "causal": True, "left": 9}, None, id="gqa-window-tails"),
    pytest.param({"scale": None}, None, id="default-scale"),
    pytest.param({"scale": 1e-50}, "finite and positive in FP32", id="scale-underflow"),
    pytest.param({"scale": 3e38}, "finite and positive in FP32", id="scale-log2-overflow"),
]


def _declarations(monkeypatch, cc, mx, grad, *, b=1, h=1, hk=1, sq=2, sk=3, causal=False, left=None, scale=512**-0.5):
    from cudnn.frost import buffers, device
    from cudnn.sdpa.bwd.api_dsl_d512_fp8 import SdpaBwdDslD512Fp8
    from cudnn.sdpa.bwd.api_dsl_d512_mxfp8 import SdpaBwdDslD512Mxfp8

    # Host admission is modeled on every supported part. No compilation or
    # execution is performed, and these cases are not GPU qualification.
    state = lambda: (True, ("nvidia-cutlass-dsl", "4.8.0"))
    for module in (engines, buffers):
        monkeypatch.setattr(module, "cutedsl_state", state)
        monkeypatch.setattr(module, "cutedsl_arch_requirement_error", lambda _: None)
    monkeypatch.setattr(device, "compute_capability", lambda _: cc)

    def desc(dtype, shape, stride, role):
        return TensorDesc(dtype, shape, stride, TensorDesc._compute_stride_order(shape, stride), torch.device("cuda", 0), name=role)

    samples, ports = {}, []
    for role in ("q", "k", "v", "o", "do", "stats", "dq", "dk", "dv"):
        kv = role in ("k", "v", "dk", "dv")
        length, heads = (sk, hk) if kv else (sq, h)
        shape = (b, heads, length, 1 if role == "stats" else 512)
        stride = (h * sq, sq, 1, 1) if role == "stats" else (length * heads * 512, 512, heads * 512, 1)
        dtype = torch.float32 if role == "stats" else grad if role in ("dq", "dk", "dv") or (mx and role == "o") else torch.float8_e4m3fn
        samples["sample_" + role] = desc(dtype, shape, stride, role)
        if role != "stats":
            ports.append((role, shape, stride))
    if mx:
        for role, source in (("q_T", "q"), ("k_T", "k"), ("do_T", "do")):
            samples["sample_" + role] = samples["sample_" + source]
        samples["sample_do_f16"] = desc(grad, samples["sample_o"].shape, samples["sample_o"].stride, "do_f16")
        for role in ("q", "q_T", "k", "k_T", "v", "do", "do_T"):
            length, heads = (sk, hk) if role in ("k", "k_T", "v") else (sq, h)
            # Canonical F8_128x4 bytes; D512 has sixteen blocks per row.
            size = b * heads * ((length + 127) // 128 * 128) * 16
            samples["sample_sf_" + role] = desc(torch.uint8, (size,), (1,), "sf_" + role)

    facts = SdpaGraphFacts(
        is_backward=True,
        b=b,
        h_q=h,
        h_kv=hk,
        s_q=sq,
        s_kv=sk,
        d_qk=512,
        d_v=512,
        dtype=cudnn.data_type.FP8_E4M3,
        dtype_o=_DTYPES[grad],
        is_fp8=not mx,
        is_mxfp8=mx,
        device_cc=cc,
        scale=scale,
        causal=causal,
        window_left=left,
        deterministic=True,
        port_layouts=tuple(ports),
    )
    api_type = SdpaBwdDslD512Mxfp8 if mx else SdpaBwdDslD512Fp8
    api = api_type(**samples, is_causal=causal, window_size_left=left, deterministic=True, scale_softmax=scale)
    name = "sdpa_bwd_d512_mxfp8" if mx else "sdpa_bwd_d512_fp8"
    return facts, api, next(s for s in engines.ENGINE_SPECS if s.name == name), {name: 20615 if mx else 20614}


@pytest.mark.parametrize("cc", [(10, 0), (10, 3), (10, 7)], ids=["sm100", "sm103", "sm107"])
@pytest.mark.parametrize("mx,grad", _ROUTES)
@pytest.mark.parametrize("entry", ["graph", "standalone"])
@pytest.mark.parametrize("geometry,expected", _CASES)
def test_d512_quantized_bounds_match_standalone(monkeypatch, cc, mx, grad, entry, geometry, expected):
    facts, api, spec, offered = _declarations(monkeypatch, cc, mx, grad, **geometry)
    if entry == "standalone":
        if expected is None:
            assert api.check_support()
        else:
            with pytest.raises((NotImplementedError, ValueError), match=expected):
                api.check_support()
        return
    reason = engines.mismatch(spec.capabilities, facts)
    if expected is None:
        assert reason is None, reason
    else:
        assert reason is not None and expected in reason, reason
    for mode in ("A", "FALLBACK"):
        records = recommend(mode, facts, offered)
        assert [p.engine_id for p in records] == ([] if expected else list(offered.values()))
