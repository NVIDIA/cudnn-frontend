# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPT = Path(__file__).with_name("api_index.py")
PACKAGE_ROOT = os.environ.get("CUDNN_API_INDEX_PACKAGE_ROOT")


class ApiIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("cudnn/__init__.py", "__all__ = []")
        self.write("modules.txt", "cudnn\n")

    def write(self, path, source):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(source), encoding="utf-8")

    def run_scanner(self, *args):
        return subprocess.run(
            [
                sys.executable,
                "-S",
                str(SCRIPT),
                "--package-root",
                str(self.root / "cudnn"),
                "--modules",
                str(self.root / "modules.txt"),
                "--index",
                str(self.root / "index.txt"),
                *args,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )

    def names(self):
        result = self.run_scanner("--write")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return set((self.root / "index.txt").read_text(encoding="utf-8").splitlines())

    def test_only_all_exports_from_listed_modules_are_checked(self):
        self.write("cudnn/__init__.py", "import os\n__all__ = ['run']\ndef run(): pass\ndef helper(): pass")
        self.write("cudnn/kernels/template.py", "raise RuntimeError('must not import')")
        self.write("cudnn/deep/api.py", "__all__ = ['entry']\ndef entry(): pass")
        self.assertEqual(self.names(), {"cudnn", "cudnn.run"})
        self.write("modules.txt", "cudnn\ncudnn.deep.api\n")
        self.assertEqual(self.names(), {"cudnn", "cudnn.run", "cudnn.deep.api", "cudnn.deep.api.entry"})

    def test_generated_and_inherited_class_members(self):
        self.write(
            "cudnn/__init__.py",
            """
            class Base:
                def inherited(self): pass
            class Builder(Base):
                @property
                def value(self): raise RuntimeError('must not evaluate')
                def _helper(self): pass
            Builder.generated = lambda self: None
            Alias = Builder
            __all__ = ['Builder', 'Alias']
            """,
        )
        expected = {"cudnn"}
        for name in ("Builder", "Alias"):
            expected.update(f"cudnn.{name}{suffix}" for suffix in ("", ".inherited", ".value", ".generated"))
        self.assertEqual(self.names(), expected)

    def test_lazy_exports_resolve_only_declared_names(self):
        self.write(
            "cudnn/__init__.py",
            """
            __all__ = ['Builder']
            def __getattr__(name):
                if name == 'Builder':
                    from ._impl import Builder
                    return Builder
                raise RuntimeError('unlisted export')
            """,
        )
        self.write("cudnn/_impl.py", "class Builder:\n    def build(self): pass")
        self.assertEqual(self.names(), {"cudnn", "cudnn.Builder", "cudnn.Builder.build"})

    def test_module_aliases_and_external_classes_do_not_expand_scope(self):
        self.write("cudnn/__init__.py", "import os\nfrom pathlib import Path\nfrom . import api\n__all__ = ['os', 'Path', 'api']")
        self.write("cudnn/api.py", "__all__ = ['run']\ndef run(): pass")
        self.assertEqual(self.names(), {"cudnn", "cudnn.os", "cudnn.Path", "cudnn.api"})

    def test_import_and_export_failures_preserve_baseline(self):
        self.write("index.txt", "keep this baseline\n")
        for source in (
            "import missing_api_index_dependency",
            "raise NameError('template parameter')",
            "raise SystemExit(0)",
            "def invalid_syntax(:",
            "__all__ = ['missing']",
            "__all__ = 'invalid'",
            "__all__ = [1]",
            "def unlisted(): pass",
        ):
            with self.subTest(source=source):
                self.write("cudnn/__init__.py", source)
                result = self.run_scanner("--write")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("API inspection failed", result.stderr)
                self.assertEqual((self.root / "index.txt").read_text(), "keep this baseline\n")

    def test_invalid_or_missing_scope_fails(self):
        for content in ("", "cudnn\ncudnn\n", "cudnn.missing\n", "cudnn._private\n", "os\n"):
            with self.subTest(content=content):
                self.write("modules.txt", content)
                self.assertEqual(self.run_scanner("--write").returncode, 2)
        (self.root / "modules.txt").unlink()
        self.assertEqual(self.run_scanner("--write").returncode, 2)

    def test_additions_and_removals_fail(self):
        self.assertEqual(self.run_scanner().returncode, 1)
        self.names()
        self.write("cudnn/__init__.py", "__all__ = ['new_api']\ndef new_api(): pass")
        result = self.run_scanner()
        self.assertEqual(result.returncode, 1)
        self.assertIn("+cudnn.new_api", result.stdout)
        self.names()
        self.write("cudnn/__init__.py", "__all__ = []")
        result = self.run_scanner()
        self.assertEqual(result.returncode, 1)
        self.assertIn("-cudnn.new_api", result.stdout)

    def test_utf8(self):
        self.write("cudnn/__init__.py", "__all__ = ['café']\ndef café(): pass")
        self.assertIn("cudnn.café", self.names())
        self.assertEqual(self.run_scanner().returncode, 0)


@unittest.skipUnless(PACKAGE_ROOT, "Set CUDNN_API_INDEX_PACKAGE_ROOT to the canonical built cudnn directory")
class RuntimeApiIndexTest(unittest.TestCase):
    def test_repository_api_index(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--package-root", PACKAGE_ROOT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
