# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-case storage for resumable attention inference sweeps."""

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from .config_types import BenchmarkResult


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _driver_version():
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            output = subprocess.check_output([smi, "--query-gpu=driver_version", "--format=csv,noheader"], text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            # An unavailable nvidia-smi is equivalent to an empty version result.
            output = ""
        if output.strip():
            return output.splitlines()[0].strip()
    try:
        from cuda.bindings import driver

        status, version = driver.cuDriverGetVersion()
        if int(status) == 0:
            return version
    except ImportError as exc:
        raise ValueError("Cannot identify the CUDA driver for checkpointing") from exc
    raise ValueError("Cannot identify the CUDA driver for checkpointing")


def _cudnn_environment():
    try:
        import cudnn

        backend_version = cudnn.backend_version()
    except (ImportError, OSError, RuntimeError):
        return None  # Reference backends can run without cuDNN Frontend.

    # Editable installs can change dispatch or kernels without a version bump.
    root = Path(cudnn.__file__).parent
    sources = [(path.relative_to(root).as_posix(), _file_digest(path)) for path in sorted(root.rglob("*.py"))]
    binding = sys.modules.get(cudnn.backend_version.__module__)
    if binding is not None and binding.__file__:
        sources.append(("native_binding", _file_digest(Path(binding.__file__))))
    return {
        "frontend_version": cudnn.__version__,
        "backend_version": backend_version,
        "source": _digest(sources),
    }


def benchmark_environment():
    """Capture the device and software that the CUDA worker will use."""
    try:
        import torch

        # Workers use the first visible device, independent of the caller's
        # current CUDA device in this process.
        device = torch.cuda.get_device_properties(0)
    except (ImportError, RuntimeError, AssertionError) as exc:
        raise ValueError("Checkpointing requires a usable CUDA device") from exc
    uuid = getattr(device, "uuid", None)
    if not uuid:
        raise ValueError("Cannot identify the CUDA device UUID for checkpointing")

    packages = {}
    for dist in (
        "b12x",
        "cuda-bindings",
        "flash-attn",
        "flash-attn-4",
        "flash-mla",
        "flashinfer-cubin",
        "flashinfer-jit-cache",
        "flashinfer-python",
        "nvidia-cudnn-cu12",
        "nvidia-cudnn-cu13",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-cu13",
        "triton",
    ):
        try:
            packages[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    prefixes = ("CUDA_", "CUDNN_", "CUTE_", "CUTLASS_", "FLASH_ATTN_", "FLASH_MLA_", "FLASHINFER_", "B12X_", "NVIDIA_", "TORCH_")
    env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(prefixes) or key in {"LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"}
    }
    directory = Path(__file__).resolve().parent
    return {
        "gpu": {"uuid": str(uuid), "name": device.name, "memory": device.total_memory, "sm_count": device.multi_processor_count},
        "capability": [device.major, device.minor],
        "driver": _driver_version(),
        "python": [sys.executable, sys.version],
        "torch": [torch.__version__, torch.version.cuda],
        "cudnn": _cudnn_environment(),
        "packages": packages,
        "env": _digest(env),
        "benchmark": {
            name: _file_digest(directory / name)
            for name in ("runner.py", "checkpoint.py", "config_types.py", "benchmark_single_attention.py", "../attention_training/flops.py")
        },
    }


@contextmanager
def open_checkpoint(path, environment, resume):
    path = Path(path)
    if not resume:
        # Exclusive creation protects any existing data, including non-SQLite files.
        try:
            path.touch(exist_ok=False)
        except FileExistsError as exc:
            raise ValueError(f"Checkpoint already exists at {path}. Pass --resume to reuse it or choose a new path.") from exc
    elif not path.is_file():
        raise ValueError(f"Checkpoint does not exist at {path}")

    fingerprint = _digest(environment)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)) as db:
        if resume:
            if db.execute("PRAGMA user_version").fetchone() != (1,):
                raise ValueError(f"Unsupported checkpoint format at {path}")
            if db.execute("SELECT fingerprint FROM environment").fetchone() != (fingerprint,):
                raise ValueError(f"Checkpoint environment differs from the current benchmark environment: {path}")
        else:
            with db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("PRAGMA user_version = 1")
                db.execute("CREATE TABLE environment (fingerprint TEXT NOT NULL)")
                db.execute("INSERT INTO environment VALUES (?)", (fingerprint,))
                db.execute("CREATE TABLE results (case_key TEXT PRIMARY KEY, result TEXT NOT NULL)")
        yield db


def checkpoint_case(db, key, run):
    # Recheck under a write transaction so a second runner cannot duplicate a
    # case. Exiting the transaction commits this case or rolls it back on interruption.
    with db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT result FROM results WHERE case_key = ?", (key,)).fetchone()
        if row is not None:
            return BenchmarkResult(**json.loads(row[0])), True
        result = run()
        if result.success:
            db.execute("INSERT INTO results VALUES (?, ?)", (key, json.dumps(asdict(result))))
        return result, False
