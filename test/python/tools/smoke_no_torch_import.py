# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: ``cudnn`` and every submodule import with PyTorch made unavailable.

Installs a :class:`importlib.abc.MetaPathFinder` that raises
``ModuleNotFoundError`` for exactly ``torch`` and ``torch.*`` -- deliberately not
``sys.modules["torch"] = None``, which produces a different exception type and
would not exercise the real code path.

Run in an environment with the remaining ``[cutedsl]`` dependencies installed
(and a built ``cudnn._compiled_module``)::

    python test/python/tools/smoke_no_torch_import.py

Exits non-zero on the first failure.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys


class _TorchBlocker:
    """Meta path finder that makes ``torch`` look uninstalled."""

    def find_module(self, fullname, path=None):  # legacy API, kept harmless
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def main() -> int:
    if "torch" in sys.modules:
        print("FAIL: torch was already imported before the blocker was installed")
        return 1

    sys.meta_path.insert(0, _TorchBlocker())

    # Pre-verify that the optional deps' own torch probes survive the blocker,
    # so a failure below is genuinely ours and not theirs.
    for dep in ("cutlass", "tvm_ffi"):
        try:
            importlib.import_module(dep)
        except ModuleNotFoundError as e:
            if e.name == "torch" or (e.name or "").startswith("torch."):
                print(f"FAIL: {dep} cannot import while torch is blocked: {e}")
                return 1
            print(f"note: {dep} not installed; skipping its pre-check")
        except Exception as e:  # pragma: no cover - environment specific
            print(f"note: {dep} import raised {type(e).__name__}: {e}")

    import cudnn

    if "torch" in sys.modules:
        print("FAIL: `import cudnn` pulled in torch")
        return 1
    print("ok: import cudnn succeeded and did not import torch")

    if cudnn.datatypes.is_torch_available():
        print("FAIL: is_torch_available() is True under the blocker")
        return 1
    print("ok: is_torch_available() is False")

    if not issubclass(cudnn.TorchNotAvailableError, ImportError):
        print("FAIL: TorchNotAvailableError is not an ImportError subclass")
        return 1
    print("ok: cudnn.TorchNotAvailableError is exported and subclasses ImportError")

    failures = []
    count = 0
    for mod in pkgutil.walk_packages(cudnn.__path__, prefix="cudnn."):
        name = mod.name
        try:
            importlib.import_module(name)
            count += 1
        except cudnn.TorchNotAvailableError as e:
            failures.append(f"{name}: raised TorchNotAvailableError at import: {e}")
        except ModuleNotFoundError as e:
            if e.name == "torch" or (e.name or "").startswith("torch."):
                failures.append(f"{name}: hard torch import at module scope")
            else:
                print(f"skip {name}: optional dep missing ({e.name})")
        except Exception as e:  # pragma: no cover - environment specific
            print(f"skip {name}: {type(e).__name__}: {e}")

    print(f"ok: imported {count} submodules with torch blocked")
    if failures:
        print(f"FAIL: {len(failures)} module(s) still require torch to import:")
        for f in failures:
            print("  " + f)
        return 1

    # Representative entry points must raise the public error, not something else.
    checks = [
        ("cudnn.wrapper.Graph", lambda: cudnn.Graph()),
        ("cudnn.ops.causal_conv1d", lambda: cudnn.ops.causal_conv1d),
        ("cudnn.experimental.ops.scaled_dot_product_attention",
         lambda: importlib.import_module("cudnn.experimental.ops").scaled_dot_product_attention),
    ]
    for label, fn in checks:
        try:
            fn()
        except cudnn.TorchNotAvailableError as e:
            msg = str(e)
            if "requires PyTorch, but PyTorch is not installed" not in msg:
                print(f"FAIL: {label} raised TorchNotAvailableError with unexpected message: {msg}")
                return 1
            print(f"ok: {label} raised TorchNotAvailableError")
        except Exception as e:
            print(f"FAIL: {label} raised {type(e).__name__} instead of TorchNotAvailableError: {e}")
            return 1
        else:
            print(f"FAIL: {label} did not raise without PyTorch")
            return 1

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
