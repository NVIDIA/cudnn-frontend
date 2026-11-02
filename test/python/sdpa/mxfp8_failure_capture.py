# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in evidence for MXFP8 backward failures, without altering their reports.

Set MXFP8_CAPTURE_DIR to a retained artifact directory. Only the first failing
backward case per process is saved. Payloads are logical uint8 bytes plus original
shape/stride metadata; storage gaps, offsets and aliases are not preserved.
Load with torch.load(path, map_location="cpu", weights_only=True).
"""

import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile


def _diagnostic(message):
    try:
        print(message, file=sys.stderr)
    except Exception:
        # A closed/broken diagnostic stream must not replace the test report.
        pass


def _probe(function):
    try:
        return function()
    except Exception as error:
        return {"unavailable": str(error)}


def _tensor_bytes(tensor):
    import torch

    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "bytes": tensor.detach().contiguous().view(torch.uint8).cpu(),
    }


def _snapshot(local, nodeid, failure):
    # Never load Torch before conftest's allocator/TE setup. The harness has
    # already imported cuDNN; synthetic host controls may have no backend.
    import torch

    cudnn = sys.modules.get("cudnn")
    graphs = {}
    for name in ("graph_fwd", "graph_bwd"):
        graph = local[name]
        index = _probe(lambda: graph._plan_index)
        graphs[name] = {
            "index": index,
            "plan": _probe(lambda: graph.get_plan_name_at_index(index)),
            "engine_and_knobs": _probe(lambda: str(graph.get_engine_and_knobs_at_index(index))),
            "serialized": _probe(lambda: graph.serialize(as_bytes=True, include_kernels=False)),
        }
    return {
        "format_version": 1,
        "nodeid": nodeid,
        "failure": failure,
        "cfg": local["cfg"].serialize(),
        "reference_recipe": {key: str(local.get(key)) for key in ("torch_itype", "torch_otype", "attn_scale_ref", "left_bound", "right_bound", "diag_align")},
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "frontend": str(getattr(cudnn, "__version__", "unknown")),
        "frontend_path": str(getattr(cudnn, "__file__", "unknown")),
        "backend": _probe(lambda: cudnn.backend_version_string()),
        "cutlass": _probe(lambda: importlib.metadata.version("nvidia-cutlass-dsl")),
        "device": _probe(lambda: str(torch.cuda.get_device_properties(local["dO_f32"].device))),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": _probe(lambda: torch.backends.cuda.matmul.allow_tf32),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDNN_FRONTEND_ENABLE_FROST_ENGINES",
                "NVIDIA_TF32_OVERRIDE",
                "CUDNN_LOGLEVEL_DBG",
                "CUDNN_RESCALE_THRESHOLD",
                "MHAS_NUM_TESTS",
                "MHAS_SEED_OFFSET",
            )
        },
        "graphs": graphs,
        "variant_pack_bwd": {uid: _tensor_bytes(tensor) for uid, tensor in local["variant_pack_bwd"].items()},
        "reference_tensors": {
            key: _tensor_bytes(value) for key, value in local.items() if isinstance(value, torch.Tensor) and (key.endswith("_ref") or key.endswith("_f32"))
        },
    }


def capture_mxfp8_failure(nodeid, error):
    """Best-effort capture; a diagnostic error must not replace a test failure."""
    directory = os.environ.get("MXFP8_CAPTURE_DIR")
    if not directory or error is None:
        return
    temporary = None
    try:
        path = Path(directory) / f"mxfp8-failure-{os.getpid()}.pt"
        if path.exists():
            return
        traceback = error.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if frame.f_code.co_name == "exec_sdpa_mxfp8" and "variant_pack_bwd" in frame.f_locals:
                import torch

                bundle = _snapshot(frame.f_locals, nodeid, str(error))
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=path.stem + "-", suffix=".tmp", dir=path.parent)
                os.close(fd)
                torch.save(bundle, temporary)
                os.replace(temporary, path)
                _diagnostic(f"MXFP8 capture: {path}")
                return
            traceback = traceback.tb_next
    except Exception as capture_error:
        _diagnostic(f"MXFP8 capture failed (original test failure retained): {capture_error}")
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError as cleanup_error:
                _diagnostic(f"MXFP8 capture cleanup failed (original test failure retained): {cleanup_error}")
