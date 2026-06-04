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
