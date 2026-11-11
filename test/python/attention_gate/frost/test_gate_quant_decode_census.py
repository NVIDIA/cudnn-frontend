# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The exhaustive census of the attention gate + quantize decode step: EVERY (attn, gate) bf16 pair -- 2^32 pairs per scale, 1,024 calls of
``[128, 32768]`` -- against the fp64 reference of ``gate_quant_utils``.  This module is the MEASUREMENT behind the reference's margin
(``gate_quant_utils.MARGIN_ULP32``) and behind its recorded flag (``DEVICE_MODEL_BITWISE``), and their command::

    pytest -m L2 attention_gate/frost/test_gate_quant_decode_census.py -s

It runs at L2 (about 4 s per scale on an SM100-class part plus the fp64 work, about 1.5 GB of device memory): the L0 suite asserts the same
three statements on drawn and adversarial operands and on every finite gate value; this one closes the whole input domain and prints the
numbers a new runtime (another compiler flavour, another torch, another part) is judged by.  NaN / infinite operand elements are outside
the contract and are excluded from the three statements (they are still launched)."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from gate_quant_utils import (  # noqa: E402
    DEVICE_MODEL_BITWISE,
    MARGIN_ULP32,
    SCALES,
    _bits,
    _device_or_skip,
    device_model,
    fp64_bracket,
    product_form_model,
    quantize,
    scale_tensor,
)

from cudnn.attention_gate.frost import gate_quant_decode as entry  # noqa: E402

pytestmark = [pytest.mark.L2]

ROWS, WIDTH = 128, 32768
CHUNK = ROWS * WIDTH  # 2^22 pairs per call
CHUNKS = (1 << 32) // CHUNK


def test_every_bf16_pair_meets_the_reference_and_the_margin_holds_with_room():
    dev = _device_or_skip()
    k = entry.AttentionGateQuantDecode(dev, WIDTH)
    out = torch.empty(ROWS, WIDTH, dtype=torch.float8_e4m3fn, device=dev)
    base = torch.arange(CHUNK, device=dev, dtype=torch.int64)
    scales = {value: scale_tensor(value, dev) for value in SCALES}
    tot = {value: dict(outside=0, not_nearest=0, max_margin=0.0, ne_model=0, ne_product=0) for value in SCALES}
    finite, model_flips, model_max_margin = 0, 0, 0.0
    for c in range(CHUNKS):
        i = base + c * CHUNK
        attn = (i >> 16).to(torch.int16).view(torch.bfloat16).reshape(ROWS, WIDTH)  # the high 16 bits: every attn bit pattern, 64 per call
        gate = (i & 0xFFFF).to(torch.int16).view(torch.bfloat16).reshape(ROWS, WIDTH)  # the low 16 bits: every gate bit pattern in every call
        fin = torch.isfinite(attn.float()) & torch.isfinite(gate.float())
        finite += int(fin.sum())
        _, v, other, nearest, margin = fp64_bracket(attn, gate)
        model_v = (attn.float() / (1.0 + torch.exp(-gate.float()))).bfloat16()
        mb, nb = _bits(model_v), _bits(nearest)
        flip = fin & (mb != nb) & ~(((mb & 0x7FFF) == 0) & ((nb & 0x7FFF) == 0))
        model_flips += int(flip.sum())
        if bool(flip.any()):
            model_max_margin = max(model_max_margin, float(margin[flip].max()))
        for value, scale in scales.items():
            k(attn, gate, scale, out)
            got = out.view(torch.uint8)
            d = tot[value]
            d["outside"] += int((fin & (got != quantize(v, scale)) & (got != quantize(other, scale))).sum())
            off = fin & (got != quantize(nearest, scale))
            d["not_nearest"] += int(off.sum())
            if bool(off.any()):
                d["max_margin"] = max(d["max_margin"], float(margin[off].max()))
            d["ne_model"] += int((fin & (got != device_model(attn, gate, scale))).sum())
            d["ne_product"] += int((fin & (got != product_form_model(attn, gate, scale))).sum())
    print(
        f"\n[census] finite pairs per scale {finite}; torch-model bf16 value != true nearest on {model_flips} pairs, at most {model_max_margin:.4f} fp32 units from the midpoint"
    )
    for value, d in tot.items():
        print(f"[census] scale {value!r}: {d}")
    assert finite == 4261478400, finite
    for value, d in tot.items():
        assert d["outside"] == 0, f"scale {value}: {d['outside']} bytes outside the bf16 bracket of the fp64 gated value"
        assert (
            d["max_margin"] <= MARGIN_ULP32 / 2
        ), f"scale {value}: a byte left the true nearest's code {d['max_margin']:.3f} fp32 units from the midpoint; re-derive MARGIN_ULP32"
        if DEVICE_MODEL_BITWISE:
            assert d["ne_model"] == 0, f"scale {value}: {d['ne_model']} bytes differ from the torch model of the convention"
    assert model_max_margin <= MARGIN_ULP32 / 2, f"the torch model's bf16 flips reach {model_max_margin:.3f} fp32 units; re-derive MARGIN_ULP32"
    assert (
        sum(d["ne_product"] for d in tot.values()) > 0
    ), "the product form agreed on every pair at every scale: the recorded-pair cell of the L0 suite would be vacuous"
