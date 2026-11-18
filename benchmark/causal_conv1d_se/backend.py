# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the backend library used by the frontend comparison path."""

import hashlib
import os
from pathlib import Path


def verify_loaded_library(library, backend_version):
    """Require one mapped SE library matching the request and record its on-disk identity.

    Run after the first untimed frontend call, so lazy loading has completed.
    This Linux-only benchmark deliberately rejects ambiguous or deleted mappings.
    """
    requested = Path(library).resolve(strict=True)
    mappings = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not Path(fields[5]).name.startswith("libcudnn_ext.so"):
            continue
        path = Path(fields[5]).resolve(strict=True)
        major, minor = (int(part, 16) for part in fields[3].split(":"))
        mappings[path] = (os.makedev(major, minor), int(fields[4]))
    if len(mappings) != 1 or not requested.samefile(next(iter(mappings))):
        raise RuntimeError(f"Requested backend {requested} does not uniquely match loaded SE libraries: {list(map(str, mappings))}")
    loaded = next(iter(mappings))
    stat = loaded.stat()
    if mappings[loaded] != (stat.st_dev, stat.st_ino):
        raise RuntimeError(f"Loaded backend was replaced on disk: {loaded}")
    digest = hashlib.sha256()
    with loaded.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "requested_path": str(requested),
        "loaded_path": str(loaded),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "frontend_backend_version": backend_version,
    }
