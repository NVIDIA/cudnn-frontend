import json
import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def fwd_payload(graph_uid, diagonal_alignment):
    return {
        "context": {"io_data_type": "FLOAT16"},
        "graph_uid": graph_uid,
        "nodes": [
            {
                "tag": "SDPA_FWD",
                "name": "sdpa_fwd",
                "inputs": {"Q": 1, "K": 2, "V": 3},
                "outputs": {"O": 4},
                "diagonal_alignment": diagonal_alignment,
                "implementation": "AUTO",
                "left_bound": None,
                "right_bound": None,
                "padding_mask": False,
            }
        ],
        "tensors": {
            "1": {"uid": 1, "dim": [1, 2, 16, 64], "stride": [2048, 1024, 64, 1]},
            "2": {"uid": 2, "dim": [1, 2, 16, 64], "stride": [2048, 1024, 64, 1]},
            "3": {"uid": 3, "dim": [1, 2, 16, 64], "stride": [2048, 1024, 64, 1]},
            "4": {"uid": 4, "dim": [1, 2, 16, 64], "stride": [2048, 1024, 64, 1]},
        },
    }


def write_log(tmp_path):
    payloads = [
        fwd_payload(11, "TOP_LEFT"),
        fwd_payload(22, "BOTTOM_RIGHT"),
    ]
    log_path = tmp_path / "sdpa.log"
    log_path.write_text("\n".join(json.dumps(payload) for payload in payloads))
    return log_path


def run_cli(tmp_path, *args, debug=False):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT)
    if debug:
        env["CUDNN_DEBUG_REPRO"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "cudnn_repro", *args],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cli_emits_last_context_entry_by_default(tmp_path):
    log_path = write_log(tmp_path)

    proc = run_cli(tmp_path, str(log_path))

    assert proc.stdout.count("test/python/test_mhas_v2.py::test_repro") == 1
    assert "cudnn.diagonal_alignment.BOTTOM_RIGHT" in proc.stdout
    assert "cudnn.diagonal_alignment.TOP_LEFT" not in proc.stdout


def test_cli_all_emits_every_context_entry(tmp_path):
    log_path = write_log(tmp_path)

    proc = run_cli(tmp_path, "--all", str(log_path))

    assert proc.stdout.count("test/python/test_mhas_v2.py::test_repro") == 2
    assert "cudnn.diagonal_alignment.TOP_LEFT" in proc.stdout
    assert "cudnn.diagonal_alignment.BOTTOM_RIGHT" in proc.stdout


def test_cli_debug_writes_default_files(tmp_path):
    log_path = write_log(tmp_path)

    run_cli(tmp_path, str(log_path), debug=True)

    payload = json.loads((tmp_path / "cudnn_repro_payload.json").read_text())
    assert (tmp_path / "cudnn_repro_log.txt").read_text().splitlines() == log_path.read_text().splitlines()
    assert "test/python/test_mhas_v2.py::test_repro" in (tmp_path / "cudnn_repro_command.txt").read_text()
    assert payload["graph_uid"] == 22
    assert "repro_metadata" in payload


def test_cli_debug_all_writes_indexed_files(tmp_path):
    log_path = write_log(tmp_path)

    run_cli(tmp_path, "--all", str(log_path), debug=True)

    for idx, graph_uid in enumerate((11, 22)):
        payload = json.loads((tmp_path / f"cudnn_repro_payload_{idx}.json").read_text())
        assert (tmp_path / f"cudnn_repro_log_{idx}.txt").read_text().splitlines() == log_path.read_text().splitlines()
        assert "test/python/test_mhas_v2.py::test_repro" in (tmp_path / f"cudnn_repro_command_{idx}.txt").read_text()
        assert payload["graph_uid"] == graph_uid
