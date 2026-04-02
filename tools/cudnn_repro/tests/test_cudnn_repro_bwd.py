import pytest

import cudnn_repro.stage1_annotate_sdpa_bwd as stage1_bwd
import cudnn_repro.stage2_build_repro_sdpa_bwd as stage2_bwd


def test_build_bwd_cfg_simple_case():
    payload = {
        "context": {"io_data_type": "BFLOAT16"},
        "nodes": [
            {
                "tag": "SDPA_BWD",
                "name": "sdpa_backward",
                "inputs": {"Q": 0, "K": 1, "V": 2, "O": 3, "Stats": 4, "dO": 5},
                "outputs": {"dQ": 6, "dK": 7, "dV": 8},
                "diagonal_alignment": "TOP_LEFT",
                "is_deterministic_algorithm": False,
                "left_bound": None,
                "right_bound": None,
                "padding_mask": False,
            }
        ],
        "tensors": {
            "0": {"uid": 0, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "1": {"uid": 1, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "2": {"uid": 2, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "3": {"uid": 3, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "4": {"uid": 4, "dim": [2, 4, 16, 1], "stride": [64, 16, 1, 1]},
            "5": {"uid": 5, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "6": {"uid": 6, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "7": {"uid": 7, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "8": {"uid": 8, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
        },
    }

    cfg = stage1_bwd.build_cfg("{}", payload, seed=123)
    assert cfg["data_type"] == "torch.bfloat16"
    assert cfg["rng_data_seed"] == 123
    assert cfg["is_infer"] is False
    assert cfg["is_determin"] is False
    assert cfg["is_padding"] is False
    assert cfg["is_ragged"] is False
    assert cfg["with_sink_token"] is False
    assert cfg["shape_stats"] == (2, 4, 16, 1)
    assert cfg["stride_stats"] == (64, 16, 1, 1)
    assert cfg["seq_len_q"] == []
    assert cfg["seq_len_kv"] == []


def test_build_bwd_cfg_rejects_padding():
    payload = {
        "context": {"io_data_type": "FLOAT16"},
        "nodes": [
            {
                "tag": "SDPA_BWD",
                "name": "sdpa_backward",
                "inputs": {"Q": 0, "K": 1, "V": 2, "O": 3, "Stats": 4, "dO": 5, "RAGGED_OFFSET_Q": 9, "RAGGED_OFFSET_KV": 10},
                "outputs": {"dQ": 6, "dK": 7, "dV": 8},
                "diagonal_alignment": "TOP_LEFT",
                "padding_mask": False,
            }
        ],
        "tensors": {
            "0": {"uid": 0, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "1": {"uid": 1, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "2": {"uid": 2, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "3": {"uid": 3, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "4": {"uid": 4, "dim": [2, 4, 16, 1], "stride": [64, 16, 1, 1]},
            "5": {"uid": 5, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "6": {"uid": 6, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "7": {"uid": 7, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "8": {"uid": 8, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "9": {"uid": 9, "dim": [3, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "10": {"uid": 10, "dim": [3, 1, 1, 1], "stride": [1, 1, 1, 1]},
        },
    }

    with pytest.raises(NotImplementedError, match="ragged"):
        stage1_bwd.build_cfg("{}", payload, seed=123)


def test_build_bwd_cfg_supports_padding_sink_and_sliding_window():
    payload = {
        "context": {"io_data_type": "BFLOAT16"},
        "nodes": [
            {
                "tag": "SDPA_BWD",
                "name": "sdpa_backward",
                "inputs": {
                    "Q": 0,
                    "K": 1,
                    "V": 2,
                    "O": 3,
                    "Stats": 4,
                    "dO": 5,
                    "SEQ_LEN_Q": 9,
                    "SEQ_LEN_KV": 10,
                    "SINK_TOKEN": 11,
                },
                "outputs": {"dQ": 6, "dK": 7, "dV": 8, "DSINK_TOKEN": 12},
                "diagonal_alignment": "BOTTOM_RIGHT",
                "is_deterministic_algorithm": True,
                "left_bound": 8,
                "right_bound": 17,
                "padding_mask": True,
            }
        ],
        "tensors": {
            "0": {"uid": 0, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "1": {"uid": 1, "dim": [2, 1, 16, 64], "stride": [1024, 65536, 64, 1]},
            "2": {"uid": 2, "dim": [2, 1, 16, 64], "stride": [1024, 65536, 64, 1]},
            "3": {"uid": 3, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "4": {"uid": 4, "dim": [2, 4, 16, 1], "stride": [64, 16, 1, 1]},
            "5": {"uid": 5, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "6": {"uid": 6, "dim": [2, 4, 16, 64], "stride": [4096, 1024, 64, 1]},
            "7": {"uid": 7, "dim": [2, 1, 16, 64], "stride": [1024, 65536, 64, 1]},
            "8": {"uid": 8, "dim": [2, 1, 16, 64], "stride": [1024, 65536, 64, 1]},
            "9": {"uid": 9, "dim": [2, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "10": {"uid": 10, "dim": [2, 1, 1, 1], "stride": [1, 1, 1, 1]},
            "11": {"uid": 11, "dim": [1, 4, 1, 1], "stride": [4, 1, 1, 1]},
            "12": {"uid": 12, "dim": [1, 4, 1, 1], "stride": [4, 1, 1, 1]},
        },
    }

    cfg = stage1_bwd.build_cfg("{}", payload, seed=123)
    assert cfg["is_infer"] is False
    assert cfg["is_padding"] is True
    assert cfg["seq_len_q"] == [16, 16]
    assert cfg["seq_len_kv"] == [16, 16]
    assert cfg["with_sink_token"] is True
    assert cfg["left_bound"] == 8
    assert cfg["right_bound"] == 17
    assert cfg["diag_align"] == 1
    assert cfg["is_determin"] is True


def test_build_bwd_command_uses_test_repro():
    cfg = {
        "data_type": "torch.float16",
        "rng_data_seed": 7,
        "is_infer": False,
        "is_determin": False,
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
        "shape_stats": (1, 2, 16, 1),
        "stride_stats": (32, 16, 1, 1),
        "seq_len_q": [],
        "seq_len_kv": [],
        "left_bound": None,
        "right_bound": None,
        "diag_align": 0,
        "implementation": "AUTO",
    }

    command = stage2_bwd.build_command(cfg)
    assert "test/python/test_mhas_v2.py::test_repro" in command
    assert "'is_infer': False" in command
    assert "cudnn.diagonal_alignment.TOP_LEFT" in command
