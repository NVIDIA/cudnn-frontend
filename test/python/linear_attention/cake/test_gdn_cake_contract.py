# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time metadata gates need no kernel compilation."""

from types import SimpleNamespace

import cudnn
import pytest

from cudnn.linear_attention.cake.gdn_engine import check_contract

pytestmark = [pytest.mark.L0]


def facts(**changes):
    values = dict(
        op="GDN",
        invalid=None,
        thd_layout=True,
        uniform_io=True,
        io_dtype=cudnn.data_type.BFLOAT16,
        d_qk=128,
        d_v=128,
        h_q=2,
        h_k=2,
        h_v=4,
        total_t=97,
        n_seq=3,
        cu_dtype=cudnn.data_type.INT32,
        g_dtype=cudnn.data_type.FLOAT,
        beta_dtype=cudnn.data_type.FLOAT,
        gates_at_ho=True,
        gate_channels=False,
        safe_gate=False,
        use_beta_sigmoid=False,
        allow_neg_eigval=False,
        beta_guard=False,
        gate_domain="log",
        batch_invariant=False,
        overwrite_initial_state=False,
        has_state_indices=False,
        checkpoint_every_n_tokens=0,
    )
    return SimpleNamespace(**(values | changes))


@pytest.mark.parametrize(
    "changes",
    [
        dict(d_v=64),
        dict(h_v=3),
        dict(h_k=4),
        dict(total_t=0),
        dict(total_t=2**31),
        dict(n_seq=2**30),
        dict(safe_gate=True),
        dict(checkpoint_every_n_tokens=64),
        dict(batch_invariant=True),
        dict(gate_channels=True),
        dict(beta_dtype=cudnn.data_type.BFLOAT16),
    ],
)
def test_unsupported_contract(changes):
    with pytest.raises(NotImplementedError, match="gdn_cake_det"):
        check_contract(facts(**changes), SimpleNamespace(inputs={}, outputs={}))


def test_state_port_declines():
    with pytest.raises(NotImplementedError, match="zero initial state"):
        check_contract(facts(), SimpleNamespace(inputs={"initial_state": object()}, outputs={}))


def test_strided_port_declines():
    tensor = SimpleNamespace(dim=(97, 2, 128), stride=(512, 128, 1), get_data_type=lambda: cudnn.data_type.BFLOAT16)
    with pytest.raises(NotImplementedError, match="q must be contiguous"):
        check_contract(facts(), SimpleNamespace(inputs={"q": tensor}, outputs={}))
