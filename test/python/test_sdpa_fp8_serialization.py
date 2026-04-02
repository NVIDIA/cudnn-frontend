import json
import os
import subprocess
import sys
from pathlib import Path

import cudnn
import pytest
import torch
from looseversion import LooseVersion

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    LooseVersion(cudnn.backend_version_string()) < "9.14.0",
    reason="SDPA FP8 requires cuDNN 9.14.0 or higher",
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 9,
    reason="SDPA FP8 requires Hopper or higher",
)
@pytest.mark.L0
def test_sdpa_fp8_fwd_log_serializes_fp8_tag(tmp_path):
    log_path = tmp_path / "sdpa_fp8_fwd.log"

    repro_cfg = {
        "data_type": "torch.float8_e4m3fn",
        "output_type": "torch.float8_e4m3fn",
        "rng_geom_seed": 8951755178493311270,
        "rng_data_seed": 1677995186,
        "is_alibi": None,
        "is_infer": True,
        "is_paged": False,
        "is_bias": None,
        "is_block_mask": None,
        "is_padding": False,
        "is_ragged": False,
        "is_dropout": None,
        "is_determin": None,
        "is_mxfp8": False,
        "with_score_max": False,
        "with_score_sum_exp": False,
        "with_sink_token": False,
        "diag_align": "cudnn.diagonal_alignment.TOP_LEFT",
        "left_bound": 227,
        "right_bound": 0,
        "batches": 1,
        "d_qk": 192,
        "d_v": 128,
        "s_q": 1543,
        "s_kv": 1543,
        "h_q": 7,
        "h_k": 1,
        "h_v": 1,
        "block_size": None,
        "shape_q": (1, 7, 1543, 192),
        "stride_q": (2073792, 192, 1344, 1),
        "shape_k": (1, 1, 1543, 192),
        "stride_k": (99990001, 3331333, 192, 1),
        "shape_v": (1, 1, 1543, 128),
        "stride_v": (3331333, 99990001, 128, 1),
        "shape_o": (1, 7, 1543, 128),
        "stride_o": (3331333, 128, 896, 1),
        "shape_stats": (1, 7, 1543, 1),
        "stride_stats": (10801, 1543, 1, 1),
        "seq_len_q": [],
        "seq_len_kv": [],
        "dropout_prob": 0.0,
        "implementation": "cudnn.attention_implementation.AUTO",
    }

    env = os.environ.copy()
    env["CUDNN_FRONTEND_LOG_INFO"] = "1"
    env["CUDNN_FRONTEND_LOG_FILE"] = str(log_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "--no-header",
            "--tb=short",
            "test/python/test_mhas_v2.py::test_repro",
            "--repro",
            repr(repro_cfg),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise AssertionError(f"FP8 repro failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    for line in log_path.read_text().splitlines():
        stripped = line.strip()
        if '"context"' not in stripped:
            continue
        payload = json.loads(stripped)
        assert payload["nodes"][0]["tag"] == "SDPA_FP8_FWD"
        return

    raise AssertionError(f"No serialized frontend payload found in {log_path}")
