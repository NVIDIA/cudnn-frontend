# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only build-cache regression tests: python -m unittest discover -s aot/attention/test."""

import concurrent.futures
import importlib.util
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("aot_attention_build", Path(__file__).resolve().parents[1] / "build.py")
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class ExportCacheTests(unittest.TestCase):
    def test_other_editable_checkout_is_rejected(self):
        with patch.dict(sys.modules, {"cudnn": SimpleNamespace(__file__="/some/other/checkout/cudnn/__init__.py")}):
            with self.assertRaisesRegex(RuntimeError, "expected"):
                build.check_checkout()

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


class VariantDomainTests(unittest.TestCase):
    def test_query_tiles_map_to_a_prebuilt_template(self):
        """16 / 32 name the decode tile (at its stage-width KV tile); any other proposal snaps to a prefill tile."""
        c = build.V.CLASSES[0]
        with patch.object(build.V, "kv_tile", return_value=128), patch.object(build.V, "decode_kv_tile", return_value=32):
            for tile_m in (16, 32):
                with self.subTest(tile_m=tile_m):
                    self.assertTrue(build.V.in_domain(c, 1, build.V.Knobs(0, 0, 1, tile_m, 32)))
                    self.assertTrue(build.V.in_domain(c, 8, build.V.Knobs(8, 0, 4, tile_m, 32)))
                    self.assertFalse(build.V.in_domain(c, 1, build.V.Knobs(0, 2, 1, tile_m, 32)))
                    candidate = build.V.Knobs(0, 0, 1, tile_m, 128)
                    self.assertFalse(build.V.in_domain(c, 1, candidate))
                    snapped = build.V.snap(c, 1, candidate)
                    self.assertEqual((snapped.tile_m, snapped.tile_n), (tile_m, 32))
                    self.assertTrue(build.V.in_domain(c, 1, snapped))
            for tile_m in (96, 256):
                with self.subTest(tile_m=tile_m):
                    candidate = build.V.Knobs(0, 0, 1, tile_m, 128)
                    self.assertFalse(build.V.in_domain(c, 1, candidate))
                    snapped = build.V.snap(c, 1, candidate)
                    self.assertEqual(snapped.tile_m, 64 if tile_m < 128 else 128)
                    self.assertTrue(build.V.in_domain(c, 1, snapped))
            for tile_m in (64, 128):
                candidate = build.V.Knobs(0, 0, 1, tile_m, 128)
                self.assertTrue(build.V.in_domain(c, 1, candidate))


if __name__ == "__main__":
    unittest.main()
