# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""The SM120 FP8 tile choice and the knob domain it is chosen from.

The rule half needs no device: it is arithmetic over shape and SM count. The
correctness half does, because propose_plans now exposes tiles the default
never picks, and nothing else in the suite runs a kernel at those.
"""

import pytest

from cudnn.sdpa.fwd.config_sm120 import SEQ_KV_TILES, SEQ_Q_TILES, fp8_tile_choice

SMS = 188  # RTX PRO 6000 Blackwell, the machine the rule was measured on

# Arithmetic over shape and SM count: no GPU, no DSL, so no arch gate either.
pytestmark = pytest.mark.L0


class TestTileChoice:
    """No device: shape and SM count in, (q_tile, kv_tile) out."""

    @pytest.mark.parametrize("s_q,h_q,b", [(512, 16, 1), (4096, 16, 1), (1024, 16, 1), (16384, 16, 1), (2048, 32, 4)])
    @pytest.mark.parametrize("causal", [False, True])
    def test_kv_tile_is_always_64(self, s_q, h_q, b, causal):
        """Measured faster in 47 of 48 shapes; the exception by 0.15%. A change
        here is a claim that the P restage stopped dominating."""
        assert fp8_tile_choice(s_q, h_q, b, SMS, causal)[1] == 64

    def test_small_grid_takes_the_finer_q_tile(self):
        # grid = ceil(512/128) * 16 = 64, so 64 CTAs of 188: doubling them
        # still costs no extra pass over the machine.
        assert fp8_tile_choice(512, 16, 1, SMS, False)[0] == 64

    def test_causal_widens_the_window(self):
        # grid 96: under a causal mask the last Q tile does several times the
        # work of the first, so the finer tile still pays. Without it, it does
        # not -- both directions are measured (1.47x and 1.00x regret).
        assert fp8_tile_choice(768, 16, 1, SMS, True)[0] == 64
        assert fp8_tile_choice(768, 16, 1, SMS, False)[0] == 128

    @pytest.mark.parametrize("s_q", [3072, 4096, 8192, 16384])
    @pytest.mark.parametrize("causal", [False, True])
    def test_large_grid_takes_the_coarser_q_tile(self, s_q, causal):
        assert fp8_tile_choice(s_q, 16, 1, SMS, causal)[0] == 128

    def test_batch_and_heads_count_toward_the_grid(self):
        """The rule is grid, not sequence length: the same s_q flips once the
        batch supplies enough CTAs on its own."""
        assert fp8_tile_choice(512, 16, 1, SMS, False)[0] == 64
        assert fp8_tile_choice(512, 16, 16, SMS, False)[0] == 128

    def test_choice_is_always_in_the_domain(self):
        for s_q in (128, 512, 1024, 4096, 32768):
            for b in (1, 8):
                for causal in (False, True):
                    q, kv = fp8_tile_choice(s_q, 16, b, SMS, causal)
                    assert q in SEQ_Q_TILES and kv in SEQ_KV_TILES

    def test_unknown_sm_count_falls_back_without_dividing_by_zero(self):
        assert fp8_tile_choice(4096, 16, 1, 0, False) == (SEQ_Q_TILES[0], 64)
