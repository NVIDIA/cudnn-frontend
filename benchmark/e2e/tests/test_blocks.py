# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the published Qwen3.5 layer dimensions and the fields derived from them.

A wrong dimension here does not fail; it produces a plausible share table for a model nobody
ships. Every number below is from the corresponding model's `config.json`.
"""

from pathlib import Path
import sys
import unittest

E2E_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E2E_DIR))

from _blocks import compose  # noqa: E402
from _qwen_configs import ALIASES, MODELS, QWEN4EXP, get  # noqa: E402

# name -> (hidden, layers, conv_dim, linear layers, full-attn layers, q heads, kv heads)
PUBLISHED = {
    "Qwen3.5-0.8B": (1024, 24, 6144, 18, 6, 8, 2),
    "Qwen3.5-2B": (2048, 24, 6144, 18, 6, 8, 2),
    "Qwen3.5-4B": (2560, 32, 8192, 24, 8, 16, 4),
    "Qwen3.5-9B": (4096, 32, 8192, 24, 8, 16, 4),
    "Qwen3.5-27B": (5120, 64, 10240, 48, 16, 24, 4),
    "Qwen3.5-35B-A3B": (2048, 40, 8192, 30, 10, 16, 2),
    "Qwen3.5-122B-A10B": (3072, 48, 12288, 36, 12, 32, 2),
    "Qwen3.5-397B-A17B": (4096, 60, 12288, 45, 15, 32, 2),
    "Qwen3.8-2.4T-A95B": (8192, 92, 20480, 69, 23, 64, 4),
}


class TestConfigs(unittest.TestCase):
    def test_published_dimensions(self):
        for name, (hidden, layers, conv, lin, full, qh, kv) in PUBLISHED.items():
            with self.subTest(name):
                c = get(name)
                self.assertEqual((c["hidden_size"], c["num_hidden_layers"]), (hidden, layers))
                self.assertEqual(c["conv_dim"], conv)
                self.assertEqual((c["num_linear_layers"], c["num_full_attention_layers"]), (lin, full))
                self.assertEqual((c["num_attention_heads"], c["num_key_value_heads"]), (qh, kv))

    def test_every_model_is_covered(self):
        self.assertEqual(sorted(MODELS), sorted(PUBLISHED))

    def test_linear_attention_geometry_is_uniform_across_the_family(self):
        """Head dims 128 and 16 key heads everywhere; only the value-head count moves."""
        for name in MODELS:
            c = get(name)
            with self.subTest(name):
                self.assertEqual(c["linear_key_head_dim"], 128)
                self.assertEqual(c["linear_value_head_dim"], 128)
                self.assertEqual(c["linear_num_key_heads"], 16)
                self.assertEqual(c["linear_conv_kernel_dim"], 4)
                self.assertEqual(c["head_dim"], 256)
        self.assertEqual({get(n)["conv_dim"] for n in MODELS}, {6144, 8192, 10240, 12288, 20480})

    def test_attention_cannot_be_expressed_by_deriving_head_dim(self):
        """Why blocks are instantiated directly: `num_attention_heads * 256 > hidden_size`.

        A model config that derives `head_dim = hidden_size // num_attention_heads` -- FLA's
        does -- cannot reproduce the published pair for six of the eight architectures.
        """
        expressible = [n for n in MODELS if get(n)["hidden_size"] // 256 == get(n)["num_attention_heads"]]
        self.assertEqual(sorted(expressible), ["Qwen3.5-2B", "Qwen3.5-9B"], "only these two happen to satisfy num_attention_heads * 256 == hidden_size")

    def test_reused_geometries_are_aliases(self):
        """Qwen3.5-27B, Qwen3.6-27B and Qwen3.8-27B are one architecture, not three."""
        for alias in ALIASES:
            self.assertIn(get(alias)["name"], MODELS)
        self.assertEqual(get("Qwen3.6-27B")["name"], "Qwen3.5-27B")
        self.assertEqual(get("Qwen3.8-27B")["name"], "Qwen3.5-27B")
        self.assertEqual(get("Qwen3.6-35B-A3B")["name"], "Qwen3.5-35B-A3B")

    def test_qwen4exp_is_kept_out_of_the_qwen35_block_composition(self):
        """Qwen3.8-Flash-Next shares the GDN, short-conv and MoE shapes but its decoder layer
        carries a QSA indexer, a PLE layer and gated residuals as well. Composing it from the
        Qwen3.5 blocks would omit all three and still print a plausible table."""
        self.assertNotIn("Qwen3.8-Flash-Next", MODELS)
        cfg = get("Qwen3.8-Flash-Next")
        self.assertEqual(cfg["kind"], "qwen4exp")
        self.assertEqual(cfg["extra_components"], ("qsa_indexer", "ple", "gated_residual"))
        # block top-k for the sparse selection, the same family as the DSA / NSA indexers
        self.assertEqual(cfg["indexer"]["budget"] // cfg["indexer"]["compress_ratio"], 512)
        # the PLE convolution is dilated, and it is one site: layer 2 of 48
        self.assertEqual(cfg["ple"]["dilation"], 3)
        self.assertEqual(cfg["ple"]["layer_ids"], [2])

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError):
            get("Qwen3.5-999B")


class TestCompose(unittest.TestCase):
    def test_weighted_sum_and_contributions(self):
        total, contrib = compose({"gdn": (48, 111.163), "attn": (16, 113.874)}, head_ms=236.658)
        self.assertAlmostEqual(total, 236.658 + 48 * 111.163 + 16 * 113.874, places=6)
        self.assertAlmostEqual(contrib["gdn"], 48 * 111.163, places=6)


if __name__ == "__main__":
    unittest.main()
