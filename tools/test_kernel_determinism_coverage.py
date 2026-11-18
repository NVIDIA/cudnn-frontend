# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the coverage gate in temporary Git repositories without GPU dependencies."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).with_name("check_kernel_determinism_coverage.py")
MANIFEST = "tools/kernel_determinism_manifest.json"
SOURCE = "python/cudnn/example/kernels/first.py"
OTHER_SOURCE = "python/cudnn/example/kernels/second.py"
TEST = "test/python/example/test_first_determinism.py"
OTHER_TEST = "test/python/example/test_second_determinism.py"
KERNEL = "import cutlass.cute as cute\n\n@cute.kernel\ndef kernel():\n    pass\n"


class KernelCoverageGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Coverage gate tests")
        self.git("config", "user.email", "coverage-tests@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        self.write("tools/check_kernel_determinism_coverage.py", CHECKER.read_text())
        self.write(SOURCE, KERNEL)
        self.write(OTHER_SOURCE, KERNEL)
        self.write(TEST, "def test_replay():\n    pass\n")
        self.write(OTHER_TEST, "def test_replay():\n    pass\n")
        self.entries = [
            {"name": "first", "sources": [SOURCE], "tests": [TEST]},
            {"name": "second", "sources": [OTHER_SOURCE], "tests": [OTHER_TEST]},
        ]
        self.write_manifest()
        self.commit()
        self.base = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True).stdout

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def write_manifest(self):
        self.write(MANIFEST, json.dumps(self.entries))

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "Test fixture")

    def check_gate(self, *args):
        result = subprocess.run(
            [sys.executable, "tools/check_kernel_determinism_coverage.py", "--base", self.base, *args],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr

    def test_new_kernel_must_be_registered_even_with_exemption(self):
        source = "python/cudnn/example/kernels/new.py"
        self.write(source, KERNEL)
        self.commit()
        for args in ((), ("--allow-test-exemption",)):
            with self.subTest(args=args):
                code, output = self.check_gate(*args)
                self.assertNotEqual(code, 0, output)
                self.assertIn(source, output)

    def test_new_dispatch_outside_a_kernel_directory(self):
        source = "python/cudnn/example/dispatch.py"
        self.write(source, '"""Usage example: cute.compile(op)"""\nimport cutlass.cute as dsl\n')
        self.commit()
        code, output = self.check_gate()
        self.assertEqual(code, 0, output)

        self.write(source, "from cutlass.cute import compile as build\ncompiled = build(op)\n")
        self.commit()
        code, output = self.check_gate()
        self.assertNotEqual(code, 0, output)
        self.assertIn(source, output)

    def test_public_dispatch_and_vendored_kernel_pins_require_registration(self):
        api = "python/cudnn/example/ops/forward.py"
        pin = "python/cudnn/_cake/_vendor/VENDORED_EXAMPLE.json"
        self.write(api, "from ._implementation import forward\n")
        self.write(pin, "{}\n")
        self.commit()
        code, output = self.check_gate("--allow-test-exemption")
        self.assertNotEqual(code, 0, output)
        self.assertIn(api, output)
        self.assertIn(pin, output)

    def test_changed_kernel_requires_its_own_test_update(self):
        self.write(SOURCE, KERNEL + "\nVALUE = 1\n")
        self.write(OTHER_TEST, "def test_replay():\n    assert True\n")
        self.commit()
        code, output = self.check_gate()
        self.assertNotEqual(code, 0, output)
        self.assertIn(SOURCE, output)

        self.write(TEST, "def test_replay():\n    assert True\n")
        self.commit()
        code, output = self.check_gate()
        self.assertEqual(code, 0, output)

    def test_deleted_test_does_not_count_as_an_update(self):
        self.write(SOURCE, KERNEL + "\nVALUE = 1\n")
        (self.root / TEST).unlink()
        self.commit()
        for args in ((), ("--allow-test-exemption",)):
            with self.subTest(args=args):
                code, output = self.check_gate(*args)
                self.assertNotEqual(code, 0, output)
                self.assertIn(TEST, output)

    def test_label_allows_test_update_exemption_and_reports_it(self):
        self.write(SOURCE, KERNEL + "\n# Documentation-only change.\n")
        self.commit()
        code, output = self.check_gate("--allow-test-exemption")
        self.assertEqual(code, 0, output)
        self.assertIn("exempt", output.lower())
        self.assertIn(SOURCE, output)

    def test_explicit_registry_exemption_needs_no_test_update(self):
        self.entries[0] = {"name": "first", "sources": [SOURCE], "exempt_reason": "This fixture represents an unsupported deterministic direction."}
        self.write_manifest()
        self.write(SOURCE, KERNEL + "\nVALUE = 1\n")
        self.commit()
        code, output = self.check_gate()
        self.assertEqual(code, 0, output)

    def test_unrelated_edit_passes(self):
        self.write("README.md", "Documentation update.\n")
        self.commit()
        code, output = self.check_gate()
        self.assertEqual(code, 0, output)


if __name__ == "__main__":
    unittest.main()
