# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys
import unittest

E2E_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = E2E_DIR / "Qwen-Image" / "run_model.py"
SPEC = importlib.util.spec_from_file_location("qwen_image_model_cpu_test", MODEL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODEL_PATH}")
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)


class QwenImageSpecTest(unittest.TestCase):
    def test_formal_shape_preserves_published_kernel_dimensions(self):
        shape = MODEL.resolve_shape("formal")
        self.assertEqual(shape["image_tokens"], 4096)
        self.assertEqual(shape["text_tokens"], 512)
        self.assertEqual(shape["joint_tokens"], 4608)
        self.assertEqual((shape["hidden"], shape["heads"], shape["head_dim"], shape["ffn"]), (3072, 24, 128, 12288))
        self.assertEqual(shape["layers"], 4)
        self.assertEqual(MODEL.PUBLISHED_SHAPE["full_layers"], 60)

    def test_smoke_changes_only_repeated_and_token_work(self):
        shape = MODEL.resolve_shape("smoke")
        self.assertEqual((shape["layers"], shape["image_tokens"], shape["text_tokens"]), (1, 256, 64))
        self.assertEqual((shape["hidden"], shape["heads"], shape["head_dim"], shape["ffn"]), (3072, 24, 128, 12288))

    def test_invalid_image_token_grid_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "perfect square"):
            MODEL.resolve_shape("smoke", image_tokens=255)

    def test_bf16_leaf_has_immutable_upstream_anchors(self):
        self.assertEqual(len(MODEL.OFFICIAL_MODEL["revision"]), 40)
        self.assertEqual(len(MODEL.DIFFUSERS_ANCHOR["commit"]), 40)
        self.assertEqual(len(MODEL.DIFFUSERS_ANCHOR["source_sha256"]), 64)
        self.assertEqual(MODEL.NUMERICAL_RECIPE["id"], "qwen-image-conservative-bf16-v1")
        self.assertEqual(MODEL.NUMERICAL_RECIPE["scope"], "inference_transformer_forward")


if __name__ == "__main__":
    unittest.main()
