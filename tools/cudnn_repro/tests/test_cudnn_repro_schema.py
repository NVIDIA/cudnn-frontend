import cudnn_repro as repro


def test_build_cfg_maps_causal_without_explicit_right_bound():
    payload = {
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
        "tensors": {
            "1": {"uid": 1, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
            "2": {"uid": 2, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
            "3": {"uid": 3, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
            "4": {"uid": 4, "dim": [2, 4, 128, 64], "stride": [32768, 8192, 64, 1]},
        },
    }

    cfg = repro._build_cfg("{}", payload)
    assert cfg["left_bound"] is None
    assert cfg["right_bound"] == 0
    assert cfg["diag_align"] == 0


def test_build_cfg_preserves_logged_tensor_layout():
    payload = {
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
        "tensors": {
            "1": {"uid": 1, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
            "2": {"uid": 2, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
            "3": {"uid": 3, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
            "4": {"uid": 4, "dim": [2, 128, 4, 64], "stride": [32768, 64, 8192, 1]},
        },
    }

    cfg = repro._build_cfg("{}", payload)
    assert cfg["shape_q"] == (2, 128, 4, 64)
    assert cfg["stride_q"] == (32768, 64, 8192, 1)
    assert cfg["h_q"] == 128
    assert cfg["s_q"] == 4
    assert cfg["left_bound"] is None
    assert cfg["right_bound"] is None


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

    command = repro._build_command(cfg)
    assert "cudnn.diagonal_alignment.TOP_LEFT" in command
    assert "cudnn.attention_implementation.AUTO" in command
