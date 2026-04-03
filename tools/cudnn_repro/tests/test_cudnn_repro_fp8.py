import pytest

import cudnn_repro.routing as routing
import cudnn_repro.stage1_annotate_sdpa_fp8_bwd as stage1_fp8_bwd
import cudnn_repro.stage1_annotate_sdpa_fp8_fwd as stage1_fp8_fwd
import cudnn_repro.stage2_build_repro_sdpa_fp8_bwd as stage2_fp8_bwd
import cudnn_repro.stage2_build_repro_sdpa_fp8_fwd as stage2_fp8_fwd


def _fp8_fwd_payload(*, tag="SDPA_FP8_FWD", ragged=False, paged=False, output_dtype="HALF", mxfp8=False):
    inputs = {
        "Q": 1,
        "K": 2,
        "V": 3,
        "Descale_Q": 5,
        "Descale_K": 6,
        "Descale_V": 7,
        "Descale_S": 8,
        "Scale_S": 9,
        "Scale_O": 10,
    }
    if ragged:
        inputs["SEQ_LEN_Q"] = 13
        inputs["SEQ_LEN_KV"] = 14
    if paged:
        inputs["SEQ_LEN_Q"] = 13
        inputs["SEQ_LEN_KV"] = 14
        inputs["Page_table_K"] = 15
        inputs["Page_table_V"] = 16
    tensors = {
        "1": {"uid": 1, "data_type": "FP8_E4M3", "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
        "2": {"uid": 2, "data_type": "FP8_E4M3", "dim": [2, 2, 32, 64], "stride": [4096, 2048, 64, 1]},
        "3": {"uid": 3, "data_type": "FP8_E4M3", "dim": [2, 2, 32, 32], "stride": [2048, 1024, 32, 1]},
        "4": {"uid": 4, "data_type": output_dtype, "dim": [2, 4, 16, 32], "stride": [2048, 512, 32, 1]},
        "5": {"uid": 5, "data_type": "FP8_E8M0" if mxfp8 else "FLOAT", "reordering_type": "F8_128x4" if mxfp8 else "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
        "6": {"uid": 6, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
        "7": {"uid": 7, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
        "8": {"uid": 8, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
        "9": {"uid": 9, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
        "10": {"uid": 10, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
        "11": {"uid": 11, "data_type": "FLOAT", "dim": [2, 4, 16, 1], "stride": [64, 16, 1, 1]},
        "12": {"uid": 12, "data_type": "FLOAT", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
    }
    outputs = {"O": 4, "Stats": 11, "Amax_S": 12, "Amax_O": 12}
    if ragged or paged:
        tensors["13"] = {"uid": 13, "data_type": "INT32", "dim": [2, 1, 1, 1], "stride": [1, 1, 1, 1], "pass_by_value": [13, 11]}
        tensors["14"] = {"uid": 14, "data_type": "INT32", "dim": [2, 1, 1, 1], "stride": [1, 1, 1, 1], "pass_by_value": [19, 17]}
    if paged:
        tensors["2"] = {"uid": 2, "data_type": "FP8_E4M3", "dim": [6, 2, 4, 64], "stride": [512, 256, 64, 1]}
        tensors["3"] = {"uid": 3, "data_type": "FP8_E4M3", "dim": [6, 2, 4, 32], "stride": [256, 128, 32, 1]}
        tensors["15"] = {"uid": 15, "data_type": "INT32", "dim": [2, 1, 5, 1], "stride": [5, 5, 1, 1]}
        tensors["16"] = {"uid": 16, "data_type": "INT32", "dim": [2, 1, 5, 1], "stride": [5, 5, 1, 1]}
    if ragged:
        for key in ("1", "2", "3", "4"):
            tensors[key]["ragged_offset_uid"] = 99
    return {
        "context": {"io_data_type": "FP8_E4M3"},
        "nodes": [
            {
                "tag": tag,
                "name": "sdpa_fp8_fwd",
                "inputs": inputs,
                "outputs": outputs,
                "generate_stats": True,
                "padding_mask": ragged or paged,
                "diagonal_alignment": "TOP_LEFT",
                "implementation": "COMPOSITE",
                "left_bound": None,
                "right_bound": None,
            }
        ],
        "tensors": tensors,
        "repro_metadata": {"ragged_tensor_names": [""] if ragged else []},
    }


def _fp8_bwd_payload(*, ragged=False, output_dtype="HALF", mxfp8=False):
    inputs = {
        "Q": 1,
        "K": 2,
        "V": 3,
        "O": 4,
        "Stats": 5,
        "dO": 6,
        "Descale_Q": 7,
        "Descale_K": 8,
        "Descale_V": 9,
        "Descale_O": 10,
        "Descale_dO": 11,
        "Descale_S": 12,
        "Descale_dP": 13,
        "Scale_S": 14,
        "Scale_dQ": 15,
        "Scale_dK": 16,
        "Scale_dV": 17,
        "Scale_dP": 18,
    }
    if ragged:
        inputs["SEQ_LEN_Q"] = 19
        inputs["SEQ_LEN_KV"] = 20
    payload = {
        "context": {"io_data_type": "FP8_E4M3"},
        "nodes": [
            {
                "tag": "SDPA_FP8_BWD",
                "name": "sdpa_fp8_bwd",
                "inputs": inputs,
                "outputs": {"dQ": 21, "dK": 22, "dV": 23, "Amax_dQ": 24, "Amax_dK": 25, "Amax_dV": 26, "Amax_d": 27},
                "padding_mask": ragged,
                "diagonal_alignment": "BOTTOM_RIGHT",
                "implementation": "AUTO",
                "is_deterministic_algorithm": True,
                "left_bound": 3,
                "right_bound": 7,
            }
        ],
        "tensors": {
            "1": {"uid": 1, "data_type": "FP8_E4M3", "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "2": {"uid": 2, "data_type": "FP8_E4M3", "dim": [2, 2, 32, 64], "stride": [4096, 2048, 64, 1]},
            "3": {"uid": 3, "data_type": "FP8_E4M3", "dim": [2, 2, 32, 32], "stride": [2048, 1024, 32, 1]},
            "4": {"uid": 4, "data_type": output_dtype, "dim": [2, 4, 16, 32], "stride": [2048, 512, 32, 1]},
            "5": {"uid": 5, "data_type": "FLOAT", "dim": [2, 4, 16, 1], "stride": [64, 16, 1, 1]},
            "6": {"uid": 6, "data_type": "FP8_E4M3", "dim": [2, 4, 16, 32], "stride": [2048, 512, 32, 1]},
            "7": {"uid": 7, "data_type": "FP8_E8M0" if mxfp8 else "FLOAT", "reordering_type": "F8_128x4" if mxfp8 else "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "8": {"uid": 8, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "9": {"uid": 9, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "10": {"uid": 10, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "11": {"uid": 11, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "12": {"uid": 12, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "13": {"uid": 13, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "14": {"uid": 14, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "15": {"uid": 15, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "16": {"uid": 16, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "17": {"uid": 17, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "18": {"uid": 18, "data_type": "FLOAT", "reordering_type": "NONE", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "21": {"uid": 21, "data_type": output_dtype, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "22": {"uid": 22, "data_type": output_dtype, "dim": [2, 2, 32, 64], "stride": [4096, 2048, 64, 1]},
            "23": {"uid": 23, "data_type": output_dtype, "dim": [2, 2, 32, 32], "stride": [2048, 1024, 32, 1]},
            "24": {"uid": 24, "data_type": "FLOAT", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "25": {"uid": 25, "data_type": "FLOAT", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "26": {"uid": 26, "data_type": "FLOAT", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "27": {"uid": 27, "data_type": "FLOAT", "dim": [1, 1, 1, 1], "stride": [1, 1, 1, 1]},
        },
        "repro_metadata": {"ragged_tensor_names": [""] if ragged else []},
    }
    if ragged:
        payload["tensors"]["19"] = {"uid": 19, "data_type": "INT32", "dim": [2, 1, 1, 1], "stride": [1, 1, 1, 1], "pass_by_value": [9, 7]}
        payload["tensors"]["20"] = {"uid": 20, "data_type": "INT32", "dim": [2, 1, 1, 1], "stride": [1, 1, 1, 1], "pass_by_value": [15, 11]}
        for key in ("1", "2", "3", "4", "21", "22", "23"):
            payload["tensors"][key]["ragged_offset_uid"] = 99
    return payload


def test_routing_distinguishes_fp8_and_non_fp8_tags():
    assert routing.detect_operation_key({"nodes": [{"tag": "SDPA_FWD"}]}) == "sdpa_fwd"
    assert routing.detect_operation_key({"nodes": [{"tag": "SDPA_BWD"}]}) == "sdpa_bwd"
    assert routing.detect_operation_key({"nodes": [{"tag": "SDPA_FP8_FWD"}]}) == "sdpa_fp8_fwd"
    assert routing.detect_operation_key({"nodes": [{"tag": "SDPA_FP8_BWD"}]}) == "sdpa_fp8_bwd"


def test_build_fp8_fwd_cfg_extracts_output_type_and_stats():
    cfg = stage1_fp8_fwd.build_cfg("{}", _fp8_fwd_payload(ragged=True), seed=123)
    assert cfg["data_type"] == "torch.float8_e4m3fn"
    assert cfg["output_type"] == "torch.float16"
    assert cfg["rng_data_seed"] == 123
    assert cfg["is_infer"] is True
    assert cfg["is_ragged"] is True
    assert cfg["is_paged"] is False
    assert cfg["is_mxfp8"] is False
    assert cfg["shape_stats"] == (2, 4, 16, 1)
    assert cfg["stride_stats"] == (64, 16, 1, 1)
    assert cfg["seq_len_q"] == [13, 11]
    assert cfg["seq_len_kv"] == [19, 17]


def test_build_fp8_fwd_cfg_infers_paged_block_size():
    cfg = stage1_fp8_fwd.build_cfg("{}", _fp8_fwd_payload(paged=True), seed=123)
    assert cfg["is_paged"] is True
    assert cfg["block_size"] == 4
    assert cfg["shape_k"] == (2, 2, 19, 64)
    assert cfg["shape_v"] == (2, 2, 19, 32)
    assert cfg["stride_k"] is None
    assert cfg["stride_v"] is None


def test_build_fp8_fwd_cfg_rejects_mxfp8():
    with pytest.raises(NotImplementedError, match="MXFP8"):
        stage1_fp8_fwd.build_cfg("{}", _fp8_fwd_payload(mxfp8=True), seed=123)


def test_build_fp8_bwd_cfg_extracts_output_type_and_determinism():
    cfg = stage1_fp8_bwd.build_cfg("{}", _fp8_bwd_payload(ragged=True), seed=456)
    assert cfg["data_type"] == "torch.float8_e4m3fn"
    assert cfg["output_type"] == "torch.float16"
    assert cfg["rng_data_seed"] == 456
    assert cfg["is_infer"] is False
    assert cfg["is_determin"] is True
    assert cfg["is_ragged"] is True
    assert cfg["is_mxfp8"] is False
    assert cfg["diag_align"] == 1
    assert cfg["left_bound"] == 3
    assert cfg["right_bound"] == 7
    assert cfg["shape_stats"] == (2, 4, 16, 1)
    assert cfg["stride_stats"] == (64, 16, 1, 1)
    assert cfg["seq_len_q"] == [9, 7]
    assert cfg["seq_len_kv"] == [15, 11]


def test_build_fp8_bwd_cfg_rejects_mxfp8():
    with pytest.raises(NotImplementedError, match="MXFP8"):
        stage1_fp8_bwd.build_cfg("{}", _fp8_bwd_payload(mxfp8=True), seed=123)


def test_build_fp8_forward_command_preserves_fp8_fields():
    command = stage2_fp8_fwd.build_command(stage1_fp8_fwd.build_cfg("{}", _fp8_fwd_payload(output_dtype="FP8_E5M2"), seed=7))
    assert "test/python/test_mhas_v2.py::test_repro" in command
    assert "torch.float8_e4m3fn" in command
    assert "torch.float8_e5m2" in command
    assert "'is_mxfp8': False" in command
    assert "cudnn.attention_implementation.COMPOSITE" in command


def test_build_fp8_backward_command_preserves_bwd_fields():
    command = stage2_fp8_bwd.build_command(stage1_fp8_bwd.build_cfg("{}", _fp8_bwd_payload(output_dtype="FP8_E4M3"), seed=9))
    assert "test/python/test_mhas_v2.py::test_repro" in command
    assert "'is_infer': False" in command
    assert "torch.float8_e4m3fn" in command
    assert "'is_mxfp8': False" in command
    assert "cudnn.diagonal_alignment.BOTTOM_RIGHT" in command
