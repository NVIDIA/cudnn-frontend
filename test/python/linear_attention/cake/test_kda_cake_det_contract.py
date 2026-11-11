# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reject contracts that need a different kernel chain."""

from types import SimpleNamespace
import cudnn
import pytest
from cudnn.linear_attention.cake.kda_det_engine import check_contract
from linear_attention.cake.test_gdn_cake_contract import facts

pytestmark = [pytest.mark.L0]


@pytest.mark.parametrize(
    "changes",
    [
        dict(h_q=8, h_k=8, h_v=8),
        dict(h_v=32),
        dict(use_qk_l2norm=True),
        dict(safe_gate=True),
        dict(n_seq=65537),
        dict(checkpoint_every_n_tokens=64),
        dict(beta_dtype=cudnn.data_type.FLOAT),
        dict(gate_channels=0),
    ],
)
def test_unsupported(changes):
    config = dict(op="KDA", h_q=16, h_k=16, h_v=16, gate_channels=128, beta_dtype=cudnn.data_type.BFLOAT16, use_qk_l2norm=False)
    with pytest.raises(NotImplementedError, match="kda_cake_det"):
        check_contract(facts(**(config | changes)), SimpleNamespace(inputs={}, outputs={}))


@pytest.mark.parametrize("mode", ["replay", "tape", "light"])
def test_native_backward_saved_rows(monkeypatch, mode):
    """Replay and tape recompute WY rows; light mode uses the saved rows."""
    pytest.importorskip("torch")
    from importlib import import_module
    from cudnn._cake import loader

    package = loader.load_package("cake_kda_det_train")
    backend = import_module(package.__name__ + ".cake_backend")
    plan = SimpleNamespace(S=1, N=1, H=16, NC_pad=1, R=64, recur="serial", light_saves_qg=True, light_saves_wu=True)
    rows = tuple(object() for _ in range(3))
    saved = None
    if mode != "replay":
        saved = SimpleNamespace(
            S=1,
            H=16,
            num_docs=1,
            version=backend.SAVED_VERSION_TAPE if mode == "tape" else backend.SAVED_VERSION,
            Aqk=object(),
            Akk=object(),
            tables=None,
            h=object(),
            v_new=object(),
            hseg=None,
            seg_state=None,
            qg=rows[0],
            w=rows[1],
            u=rows[2],
        )
    views = SimpleNamespace(h=object(), v_new=object())
    workspace = SimpleNamespace(plan=plan, save_mode="light", l2n="off", device=None, _packing=lambda _: None, views=lambda _: views)
    for name in ("_check_saved_gate", "_check_saved_l2n", "_check"):
        monkeypatch.setattr(backend, name, lambda *args: None)
    monkeypatch.setattr(backend, "_alloc", lambda *args: object())
    monkeypatch.setattr(backend, "_tables", lambda *args: (None, None, None))
    launches = []
    monkeypatch.setattr(backend, "_backward_launches", lambda *args, **kwargs: launches.append(kwargs))

    backend._backward(workspace, *(object() for _ in range(7)), saved=saved, scale=1.0, doc_lengths=None)

    assert len(launches) == 1
    assert launches[0]["path"] == mode
    assert tuple(launches[0][name] for name in ("qg", "w", "u")) == (rows if mode == "light" else (None, None, None))
