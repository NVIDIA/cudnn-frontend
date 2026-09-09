# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regressions separating benchmark protocols from GPU admission."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

E2E_DIR = Path(__file__).resolve().parents[1]


def load(relative_path):
    name = "device_selection_" + relative_path.replace("/", "_").replace(".", "_").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, E2E_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MATRIX = load("Qwen3.8/run_matrix.py")
RUNNERS = (MATRIX, load("Qwen-Image/run_bf16.py"), load("Qwen-Image/run_nvfp4.py"))


def gpu(name="SM100 test device", sm_count=42, capability=(10, 0)):
    return SimpleNamespace(name=name, multi_processor_count=sm_count, major=capability[0], minor=capability[1])


def fake_torch(devices):
    return SimpleNamespace(
        device=str,
        cuda=SimpleNamespace(
            is_available=lambda: bool(devices),
            device_count=lambda: len(devices),
            get_device_properties=lambda index: devices[int(str(index).removeprefix("cuda:"))],
            device=lambda _device: nullcontext(),
        ),
    )


def pick(runner, devices, mode):
    torch = fake_torch(devices)
    if runner is MATRIX:
        with patch.object(runner, "torch", torch):
            return runner._pick_device(mode)
    return runner._pick_device(torch, mode)[0]


class DeviceSelectionTest(unittest.TestCase):
    def test_both_modes_accept_supported_architecture_without_sku_or_sm_count_gate(self):
        for runner in RUNNERS:
            for mode in ("smoke", "formal"):
                for properties in (gpu(), gpu("NVIDIA B200", 42), gpu("NVIDIA B200", 148)):
                    with self.subTest(runner=runner.__name__, mode=mode, gpu=properties):
                        self.assertEqual(pick(runner, [properties], mode), "cuda:0")

    def test_product_name_cannot_override_unsupported_architecture(self):
        for runner in RUNNERS:
            for mode in ("smoke", "formal"):
                for capability in ((8, 0), (9, 0), (12, 0)):
                    with self.subTest(runner=runner.__name__, mode=mode, capability=capability):
                        with self.assertRaisesRegex(RuntimeError, "SM100"):
                            pick(runner, [gpu("NVIDIA B200", 148, capability)], mode)

    def test_selects_supported_device_in_mixed_inventory(self):
        for runner in RUNNERS:
            for mode in ("smoke", "formal"):
                with self.subTest(runner=runner.__name__, mode=mode):
                    self.assertEqual(pick(runner, [gpu(capability=(9, 0)), gpu()], mode), "cuda:1")

    def test_no_visible_gpu_has_actionable_error(self):
        for runner in RUNNERS:
            for mode in ("smoke", "formal"):
                with self.subTest(runner=runner.__name__, mode=mode):
                    with self.assertRaisesRegex(RuntimeError, "SM100"):
                        pick(runner, [], mode)

    def test_matrix_main_has_no_second_formal_hardware_policy_gate(self):
        torch = fake_torch([gpu()])
        args = SimpleNamespace(mode="formal")
        with (
            patch.object(MATRIX, "_parse_args", return_value=args),
            patch.object(MATRIX.importlib, "import_module", return_value=torch),
            patch.dict(MATRIX.os.environ, {"CUDNN_FRONTEND_ENABLE_FROST_ENGINES": "0"}),
            patch.object(MATRIX, "_load_run_model", return_value=object()),
            patch.object(MATRIX, "_run_experiment") as run,
        ):
            MATRIX.main()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[2], "cuda:0")


if __name__ == "__main__":
    unittest.main()
