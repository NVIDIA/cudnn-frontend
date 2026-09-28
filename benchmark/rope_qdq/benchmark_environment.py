# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh, process-local compiler caches; call before importing GPU providers."""

import os
from pathlib import Path
import tempfile

VARIABLES = {
    "CUDA_CACHE_PATH": "cuda",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TRITON_CACHE_DIR": "triton",
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    "TORCH_EXTENSIONS_DIR": "extensions",
    "TVM_FFI_CACHE_DIR": "tvmffi",
    "TILELANG_CACHE_DIR": "tilelang",
    "DG_JIT_CACHE_DIR": "deepgemm",
    "XDG_CACHE_HOME": "xdg",
}


def configure_caches(output):
    output = str(Path(output).resolve())
    if os.environ.get("ROPE_QDQ_CONFIGURED_OUTPUT") == output:
        root = Path(os.environ["ROPE_QDQ_CACHE_ROOT"])
        assert root.is_dir()
        assert all(os.environ[name] == str(root / leaf) for name, leaf in VARIABLES.items())
        return root
    root = Path(tempfile.mkdtemp(prefix="cudnn_rope_qdq_"))
    for name, leaf in VARIABLES.items():
        os.environ[name] = str(root / leaf)
    os.environ["ROPE_QDQ_CACHE_ROOT"] = str(root)
    os.environ["ROPE_QDQ_CONFIGURED_OUTPUT"] = output
    return root
