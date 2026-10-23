# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The python-engine plan format's graph half, without building a plan: the IR survives the
round trip exactly, a value the format cannot represent refuses, and the format is told apart
from the backend's blob. The GPU round trip lives in sdpa/frost/test_sdpa_plan_serialization.py."""

import json

import pytest

import cudnn
from cudnn import _plan_serdes as ps

pytestmark = pytest.mark.L0


def _sdpa_graph(**extra):
    g = cudnn.pygraph(io_data_type=cudnn.data_type.BFLOAT16, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q = g.tensor(dim=[2, 8, 256, 128], stride=[262144, 128, 1024, 1], name="q", uid=11)
    k = g.tensor(dim=[2, 8, 256, 128], name="k")
    v = g.tensor(dim=[2, 8, 256, 128], name="v")
    seq = g.tensor(dim=[2, 1, 1, 1], data_type=cudnn.data_type.INT32, name="seq")
    o, stats = g.sdpa(q=q, k=k, v=v, attn_scale=0.125, use_causal_mask=True, generate_stats=True, seq_len_q=seq, seq_len_kv=seq, use_padding_mask=True, **extra)
    o.set_output(True)
    stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    return g


def _round_trip(g):
    restored = cudnn.pygraph()
    ps._decode_graph(restored, json.loads(json.dumps(ps._encode_graph(g))))
    return restored


def test_the_graph_survives_the_round_trip_exactly():
    g = _sdpa_graph()
    restored = _round_trip(g)
    assert restored.inspect() == g.inspect()
    assert [n.params for n in restored._nodes] == [dict(n.params) for n in g._nodes]
    for uid, t in g._tensor_by_uid.items():
        r = restored._tensor_by_uid[uid]
        for f in ps._tensor_fields():
            assert getattr(r, f) == getattr(t, f), (t.name, f)
        assert r.owner() is restored
    assert restored._next_uid == g._next_uid and restored._reserved_uids == g._reserved_uids
    assert restored.find_tensor("q").uid == 11 and restored._context == g._context
    restored.tensor(dim=[1], name="fresh")  # the uid allocator continues where the source graph stopped
    assert restored.find_tensor("fresh").uid not in g._tensor_by_uid


def test_a_frozen_graph_encodes_like_a_mutable_one():
    g = _sdpa_graph()
    before = ps._encode_graph(g)
    g._freeze()
    after = ps._encode_graph(g)
    for t in after["tensors"]:  # frozen dims/strides are tuples; decode restores lists either way
        for f in ("dim", "stride"):
            if isinstance(t[f], dict):
                t[f] = t[f]["$tuple"]
    assert after == before


def test_a_callback_param_refuses_instead_of_vanishing():
    g = _sdpa_graph(score_mod=lambda graph, score: score)
    with pytest.raises(ps._Unserializable, match="score_mod"):
        ps._encode_graph(g)


def test_blobs_are_told_apart():
    blob = ps.MAGIC + ps._LEN.pack(2) + b"{}"
    assert ps.is_python_plan(blob) and ps.is_python_plan(list(blob)) and ps.is_python_plan(bytearray(blob))
    for other in (b"", b"invalid serialized graph", list(b'{"json_version"'), "a json string", None):
        assert not ps.is_python_plan(other)


def test_an_unknown_format_or_a_truncated_blob_is_an_error():
    header = json.dumps({"format": ps.FORMAT + 1}).encode()
    with pytest.raises(ValueError, match="format"):
        cudnn.pygraph().deserialize(ps.MAGIC + ps._LEN.pack(len(header)) + header)
    with pytest.raises(ValueError, match="truncated"):
        cudnn.pygraph().deserialize(ps.MAGIC + ps._LEN.pack(1000) + b"{}")


def test_deserialize_of_a_python_plan_needs_an_empty_graph():
    header = json.dumps({"format": ps.FORMAT}).encode()
    with pytest.raises(RuntimeError, match="empty pygraph"):
        _sdpa_graph().deserialize(ps.MAGIC + ps._LEN.pack(len(header)) + header)
