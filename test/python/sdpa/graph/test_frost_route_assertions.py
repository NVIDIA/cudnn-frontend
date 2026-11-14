# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A migrated D256 route assertion still detects fallback and the wrong body."""

from types import SimpleNamespace

import pytest

from sdpa.graph import test_mhas_v2 as mhas

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "served,body,requested,valid",
    [
        ("sdpa_fwd_sm100_d256_lookahead", "prefill_d256_f16", None, True),
        ("sdpa_fwd_sm100_d256_swap_ab", "decode_d256_f16", None, True),
        ("sdpa_fwd_sm100_d256_lookahead", "prefill_d256_f16", "prefill_d256_f16", True),
        ("sdpa_fwd_sm100_d256_swap_ab", "decode_d256_f16", "decode_d256_f16", True),
        ("sdpa_fwd_prefill_sm100", "prefill_d256_f16", None, False),
        (None, None, None, False),
        ("sdpa_fwd_sm100_d256_swap_ab", "decode_d256_f16", "prefill_d256_f16", False),
        ("sdpa_fwd_sm100_d256_lookahead", "decode_d256_f16", "prefill_d256_f16", False),
    ],
)
def test_d256_route_assertion_keeps_owner_and_body_strict(monkeypatch, served, body, requested, valid):
    counts = {}
    monkeypatch.setattr(mhas.frost_routing, "snapshot", lambda: dict(counts))
    monkeypatch.setattr(mhas.frost_routing, "LAST_PLAN", (served, SimpleNamespace(cga=2, split_kv=1)))

    def execute(*args, **kwargs):
        counts[f"frost:{served}" if served else "native:fp16-fwd"] = 1
        if served:
            counts[f"frost:{served}:{body}"] = 1

    monkeypatch.setattr(mhas, "exec_sdpa", execute)
    cfg = SimpleNamespace(d_qk=256, d_v=256)
    if valid:
        mhas._exec_sdpa_on_frost(cfg, None, None, template=requested, cga=2, knobs={"split_kv": 1})
    else:
        with pytest.raises(AssertionError, match="expected .*serve this graph"):
            mhas._exec_sdpa_on_frost(cfg, None, None, template=requested)
