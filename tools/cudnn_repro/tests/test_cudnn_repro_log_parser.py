# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import cudnn_repro.log_parser as log_parser


def payload(gid, tag, dtype):
    return {"context": {"io_data_type": dtype}, "gid": gid, "nodes": [{"tag": tag}], "tensors": []}


def test_iter_context_entries_prefers_execution_order_with_gid():
    payload1 = payload(11, "SDPA_FWD", "HALF")
    payload2 = payload(22, "SDPA_BWD", "BFLOAT16")
    lines = [
        json.dumps(payload1),
        "[cudnn_frontend] INFO: Executing gid 11",
        json.dumps(payload2),
        "[cudnn_frontend] INFO: Executing gid 22",
        "[cudnn_frontend] INFO: Executing gid 11",
    ]

    entries = list(log_parser.iter_context_entries(lines))

    assert [payload.get("gid") for _, payload in entries] == [11, 22, 11]
    assert [raw_line for raw_line, _ in entries] == [json.dumps(payload1), json.dumps(payload2), json.dumps(payload1)]


def test_iter_context_entries_falls_back_without_execution_markers():
    payload1 = payload(11, "SDPA_FWD", "HALF")
    payload2 = payload(22, "SDPA_BWD", "BFLOAT16")

    entries = list(log_parser.iter_context_entries([json.dumps(payload1), json.dumps(payload2)]))

    assert [payload.get("gid") for _, payload in entries] == [11, 22]


def test_iter_context_entries_does_not_reuse_tensor_dumps_across_gids():
    payload1 = payload(11, "SDPA_FWD", "HALF")
    payload2 = payload(22, "SDPA_BWD", "HALF")
    payload1["tensors"] = [{"uid": 5}]
    payload2["tensors"] = [{"uid": 5}]
    lines = [
        json.dumps(payload1),
        json.dumps(payload2),
        "[cudnn_frontend] INFO: Executing gid 11",
        "[cudnn_frontend] INFO: Tensor Dump uid: 5 Name:  Data: [13, 11]",
        "[cudnn_frontend] INFO: Executing gid 22",
    ]

    entries = list(log_parser.iter_context_entries(lines))

    assert entries[0][1]["tensors"][0]["pass_by_value"] == [13, 11]
    assert "pass_by_value" not in entries[-1][1]["tensors"][0]


def test_iter_context_entries_prefers_current_tensor_dump():
    payload1 = payload(11, "SDPA_FWD", "HALF")
    payload2 = payload(22, "SDPA_BWD", "HALF")
    payload1["tensors"] = [{"uid": 5}]
    payload2["tensors"] = [{"uid": 5}]
    lines = [
        json.dumps(payload1),
        json.dumps(payload2),
        "[cudnn_frontend] INFO: Executing gid 11",
        "[cudnn_frontend] INFO: Tensor Dump uid: 5 Name:  Data: [1]",
        "[cudnn_frontend] INFO: Executing gid 22",
        "[cudnn_frontend] INFO: Tensor Dump uid: 5 Name:  Data: [2]",
    ]

    entries = list(log_parser.iter_context_entries(lines))

    assert entries[-1][1]["tensors"][0]["pass_by_value"] == [2]


def test_iter_context_entries_attaches_dumped_ragged_offset_tensor():
    payload1 = payload(11, "SDPA_FWD", "HALF")
    payload1["tensors"] = [{"uid": 5, "ragged_offset_uid": 15}]
    lines = [
        json.dumps(payload1),
        "[cudnn_frontend] INFO: Executing gid 11",
        "[cudnn_frontend] INFO: Tensor Dump uid: 15 Name:  Data: [0, 128]",
    ]

    entries = list(log_parser.iter_context_entries(lines))

    assert entries[-1][1]["tensors"][-1] == {"uid": 15, "pass_by_value": [0, 128]}


def test_iter_context_entries_ignores_dumps_for_unknown_gid():
    payload1 = payload(22, "SDPA_BWD", "HALF")
    payload1["tensors"] = [{"uid": 5}]
    lines = [
        json.dumps(payload1),
        "[cudnn_frontend] INFO: Executing gid 11",
        "[cudnn_frontend] INFO: Tensor Dump uid: 5 Name:  Data: [1]",
        "[cudnn_frontend] INFO: Executing gid 22",
    ]

    entries = list(log_parser.iter_context_entries(lines))

    assert "pass_by_value" not in entries[-1][1]["tensors"][0]


def test_parse_tensor_dump_skips_malformed_data():
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: [1, 2, 3]") == (3, [1, 2, 3])
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: [1, 2, oops]") is None
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: [[1], [2]]") is None


def test_parse_tensor_dump_rejects_non_integer_payload_values():
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: [1.5]") is None
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: [1, 2.0]") is None
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: [true, false]") is None
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: [1e10000]") is None
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: [null]") is None
    assert log_parser._parse_tensor_dump('Tensor Dump uid: 3 Data: ["1"]') is None
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: []") == (3, [])
    assert log_parser._parse_tensor_dump("Tensor Dump uid: 3 Data: [0, -1, 7]") == (3, [0, -1, 7])


def test_iter_context_entries_ignores_malformed_tensor_dump_line():
    payload1 = payload(11, "SDPA_FWD", "HALF")
    lines = [
        json.dumps(payload1),
        "[cudnn_frontend] INFO: Executing gid 11",
        "Tensor Dump uid: 3 Data: [1, 2, oops]",
    ]

    entries = list(log_parser.iter_context_entries(lines))

    assert [p.get("gid") for _, p in entries] == [11]
