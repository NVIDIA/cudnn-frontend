# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only build-cache regression tests: python -m unittest discover -s aot/sm120/test."""

import concurrent.futures
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("sm120_aot_build", Path(__file__).resolve().parents[1] / "build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class ExportCacheTests(unittest.TestCase):
    def test_transitive_inputs_and_partial_outputs_invalidate_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "obj").mkdir()
            (out / "include").mkdir()
            rec = {"name": "kernel"}
            calls = []

            def export(record, directory, cuobjdump):
                calls.append(record)
                (out / "obj/kernel.o").write_bytes(b"object")
                (out / "include/kernel.h").write_bytes(b"header")
                return {**record, "ok": True}

            def pool(**kwargs):
                # Isolate cache behavior from process creation; offline builds exercise real spawn pools.
                return concurrent.futures.ThreadPoolExecutor(max_workers=1)

            with patch.object(build.cf, "ProcessPoolExecutor", pool), patch.object(build, "_export_one", export):
                with patch.object(build, "source_fingerprint", return_value="source-1", create=True):
                    build.run_export([rec], out, 1, Path("/cuda"), True)
                    build.run_export([rec], out, 1, Path("/cuda"), True)
                    self.assertEqual(len(calls), 1)
                with patch.object(build, "source_fingerprint", return_value="source-2", create=True):
                    build.run_export([rec], out, 1, Path("/cuda"), True)
                    self.assertEqual(len(calls), 2)
                    (out / "include/kernel.h").write_bytes(b"interrupted export")
                    build.run_export([rec], out, 1, Path("/cuda"), True)
                    self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
