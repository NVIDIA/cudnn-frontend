import json

import cudnn_repro.log_parser as log_parser


def test_iter_context_entries_prefers_execution_order_with_graph_uid():
    payload1 = {"context": {"io_data_type": "HALF"}, "graph_uid": 11, "nodes": [{"tag": "SDPA_FWD"}], "tensors": {}}
    payload2 = {"context": {"io_data_type": "BFLOAT16"}, "graph_uid": 22, "nodes": [{"tag": "SDPA_BWD"}], "tensors": {}}
    lines = [
        json.dumps(payload1),
        "[cudnn_frontend] INFO: Executing graph_uid 11",
        json.dumps(payload2),
        "[cudnn_frontend] INFO: Executing graph_uid 22",
        "[cudnn_frontend] INFO: Executing graph_uid 11",
    ]

    entries = list(log_parser.iter_context_entries(lines))

    assert [payload.get("graph_uid") for _, payload in entries] == [11, 22, 11]
    assert [raw_line for raw_line, _ in entries] == [json.dumps(payload1), json.dumps(payload2), json.dumps(payload1)]


def test_iter_context_entries_falls_back_without_execution_markers():
    payload1 = {"context": {"io_data_type": "HALF"}, "graph_uid": 11, "nodes": [{"tag": "SDPA_FWD"}], "tensors": {}}
    payload2 = {"context": {"io_data_type": "BFLOAT16"}, "graph_uid": 22, "nodes": [{"tag": "SDPA_BWD"}], "tensors": {}}

    entries = list(log_parser.iter_context_entries([json.dumps(payload1), json.dumps(payload2)]))

    assert [payload.get("graph_uid") for _, payload in entries] == [11, 22]
