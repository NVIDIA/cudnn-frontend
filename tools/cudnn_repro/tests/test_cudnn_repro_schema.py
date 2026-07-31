# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudnn_repro.operations as operations
import cudnn_repro.repro_command as repro_command
import cudnn_repro.sdpa_fwd as sdpa_fwd

from .helpers import tensor_list


def fwd_payload(*, gid=1, unfuse_fma=False):
    payload = {
        "json_version": "2.0",
        "gid": gid,
        "context": {"io_data_type": "FLOAT16"},
        "nodes": [
            {
                "tag": "SDPA_FWD",
                "name": "sdpa_fwd",
                "inputs": {"Q": 1, "K": 2, "V": 3},
                "outputs": {"O": 4},
                "diagonal_alignment": "BOTTOM_RIGHT",
                "implementation": "UNIFIED",
                "unfuse_fma": unfuse_fma,
                "left_bound": None,
                "right_bound": None,
                "padding_mask": False,
            }
        ],
        "tensors": tensor_list(
            {
                "1": {"uid": 1, "dim": [1, 3, 16, 64], "stride": [3072, 64, 192, 1]},
                "2": {"uid": 2, "dim": [1, 1, 16, 64], "stride": [1024, 64, 64, 1]},
                "3": {"uid": 3, "dim": [1, 1, 16, 64], "stride": [1024, 64, 64, 1]},
                "4": {"uid": 4, "name": "sdpa_fwd::O", "dim": [1, 3, 16, 64], "stride": [3072, 64, 192, 1]},
            }
        ),
    }
    return payload


def test_build_cfg_maps_causal_without_explicit_right_bound():
    payload = {
        "json_version": "2.0",
        "gid": 1,
        "context": {"io_data_type": "FLOAT16"},
        "nodes": [
            {
                "tag": "SDPA_FWD",
                "name": "sdpa_fwd",
                "inputs": {"Q": 1, "K": 2, "V": 3},
                "outputs": {"O": 4},
                "diagonal_alignment": "TOP_LEFT",
                "causal_mask": True,
                "left_bound": None,
                "right_bound": None,
            }
        ],
        "tensors": tensor_list(
            {
                "1": {"uid": 1, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
                "2": {"uid": 2, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
                "3": {"uid": 3, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
                "4": {"uid": 4, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
            }
        ),
    }

    cfg = operations.select_operation(payload).build_cfg("{}", payload)
    assert cfg["left_bound"] is None
    assert cfg["right_bound"] == 0
    assert cfg["diag_align"] == 0


def test_select_operation_rejects_old_graph_json():
    payload = fwd_payload()
    payload["json_version"] = "1.0"

    with pytest.raises(ValueError, match="Unsupported graph JSON version"):
        operations.select_operation(payload)


def test_build_cfg_preserves_logged_tensor_layout():
    payload = {
        "json_version": "2.0",
        "gid": 1,
        "context": {"io_data_type": "FLOAT16"},
        "nodes": [
            {
                "tag": "SDPA_FWD",
                "name": "sdpa_fwd",
                "inputs": {"Q": 1, "K": 2, "V": 3},
                "outputs": {"O": 4},
                "diagonal_alignment": "TOP_LEFT",
                "left_bound": None,
                "right_bound": None,
            }
        ],
        "tensors": tensor_list(
            {
                "1": {"uid": 1, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
                "2": {"uid": 2, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
                "3": {"uid": 3, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
                "4": {"uid": 4, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
            }
        ),
    }

    cfg = operations.select_operation(payload).build_cfg("{}", payload)
    assert cfg["shape_q"] == (2, 128, 4, 64)
    assert cfg["stride_q"] == (32768, 64, 8192, 1)
    assert cfg["h_q"] == 128
    assert cfg["s_q"] == 4
    assert cfg["left_bound"] is None
    assert cfg["right_bound"] is None


def test_build_cfg_detects_ragged_from_offset_inputs():
    payload = fwd_payload()
    node = payload["nodes"][0]
    node["padding_mask"] = True
    node["inputs"].update({"SEQ_LEN_Q": 5, "SEQ_LEN_KV": 6, "RAGGED_OFFSET_Q": 7, "RAGGED_OFFSET_KV": 8})
    payload["tensors"].extend(
        tensor_list(
            {
                "5": {"uid": 5, "data_type": "INT32", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1], "pass_by_value": [13]},
                "6": {"uid": 6, "data_type": "INT32", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1], "pass_by_value": [11]},
                "7": {"uid": 7, "data_type": "INT64", "dim": [2, 1, 1, 1], "stride": [1, 1, 1, 1], "pass_by_value": [0, 13]},
                "8": {"uid": 8, "data_type": "INT64", "dim": [2, 1, 1, 1], "stride": [1, 1, 1, 1], "pass_by_value": [0, 11]},
            }
        )
    )

    cfg = operations.select_operation(payload).build_cfg("{}", payload)

    assert cfg["is_ragged"] is True
    assert cfg["is_padding"] is True
    assert cfg["seq_len_q"] == [13]
    assert cfg["seq_len_kv"] == [11]


def test_build_cfg_detects_ragged_from_tensor_offset_refs():
    payload = fwd_payload()
    node = payload["nodes"][0]
    node["padding_mask"] = True
    node["inputs"].update({"SEQ_LEN_Q": 5, "SEQ_LEN_KV": 6})
    for entry, offset_uid in zip(payload["tensors"][:4], [7, 8, 8, 7]):
        entry["ragged_offset_uid"] = offset_uid
    payload["tensors"].extend(
        tensor_list(
            {
                "5": {"uid": 5, "data_type": "INT32", "dim": [1], "stride": [1], "pass_by_value": [13]},
                "6": {"uid": 6, "data_type": "INT32", "dim": [1], "stride": [1], "pass_by_value": [11]},
                "7": {"uid": 7, "pass_by_value": [0, 13]},
                "8": {"uid": 8, "pass_by_value": [0, 11]},
            }
        )
    )

    annotated = operations.select_operation(payload).extract_and_annotate("{}", payload, "")
    cfg = operations.select_operation(annotated).build_cfg("{}", annotated)

    assert cfg["is_ragged"] is True
    assert annotated["repro_metadata"]["ragged_offset_q"] == [0, 13]
    assert annotated["repro_metadata"]["ragged_offset_kv"] == [0, 11]


def test_build_command_normalizes_enum_fields():
    cfg = {
        "data_type": "torch.float16",
        "rng_data_seed": 123,
        "batches": 1,
        "h_q": 2,
        "h_k": 2,
        "h_v": 2,
        "s_q": 16,
        "s_kv": 16,
        "d_qk": 64,
        "d_v": 64,
        "shape_q": (1, 2, 16, 64),
        "stride_q": (2048, 1024, 64, 1),
        "shape_k": (1, 2, 16, 64),
        "stride_k": (2048, 1024, 64, 1),
        "shape_v": (1, 2, 16, 64),
        "stride_v": (2048, 1024, 64, 1),
        "shape_o": (1, 2, 16, 64),
        "stride_o": (2048, 1024, 64, 1),
        "seq_len_q": [],
        "seq_len_kv": [],
        "left_bound": None,
        "right_bound": 0,
        "diag_align": 0,
        "implementation": "AUTO",
    }

    command = repro_command.build_command(cfg)
    assert "cudnn.diagonal_alignment.TOP_LEFT" in command
    assert "cudnn.attention_implementation.AUTO" in command


def test_build_cfg_preserves_unfuse_fma():
    payload = fwd_payload(unfuse_fma=True)
    cfg = operations.select_operation(payload).build_cfg("{}", payload)
    command = repro_command.build_command(cfg)

    assert cfg["with_unfuse_fma"] is True
    assert "'with_unfuse_fma': True" in command


def test_build_cfg_preserves_rope():
    payload = fwd_payload()
    payload["nodes"].append({"tag": "ROPE_FWD", "name": "RoPE_Q"})

    cfg = operations.select_operation(payload).build_cfg("{}", payload)
    command = repro_command.build_command(cfg)

    assert cfg["with_rope"] is True
    assert "'with_rope': True" in command


def test_payload_ignores_unrelated_ragged_log_text():
    payload = fwd_payload(gid=9)
    log_text = "Backend Tensor named 'sdpa_fwd::O' with UID 4\n" "Id: 4\n" "raggedOffset: Enabled UID: 99\n"

    annotated_payload = sdpa_fwd.extract_and_annotate("{}", payload, log_text)
    cfg = sdpa_fwd.build_cfg("{}", annotated_payload, seed=123)

    assert annotated_payload["repro_metadata"]["ragged_tensor_names"] == []
    assert cfg["is_ragged"] is False
