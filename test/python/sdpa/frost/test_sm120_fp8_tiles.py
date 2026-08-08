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
    def test_kv_tile_is_always_128(self, s_q, h_q, b):
        """Fastest in all 28 shapes measured. It was 64 while P was staged
        through SMEM, because that traffic scaled with the KV tile -- a change
        here is a claim about the kernel, not about the hardware."""
        assert fp8_tile_choice(s_q, s_q, h_q, b, SMS)[1] == 128

    def test_grid_too_small_to_fill_the_machine_takes_the_finer_q_tile(self):
        # grid = ceil(512/128) * 16 = 64 CTAs of 188: doubling them still fits,
        # and it is worth 1.5x.
        assert fp8_tile_choice(512, 512, 16, 1, SMS)[0] == 64

    def test_a_short_sequence_does_not_amortize_the_finer_q_tile(self):
        # Same underfilled machine, but 8 KV tiles cannot absorb the extra
        # Q-tile loop, so the coarse tile wins by 1.22x.
        assert fp8_tile_choice(1024, 1024, 16, 1, SMS)[0] == 128

    def test_underfilled_and_long_takes_the_finer_q_tile(self):
        # grid 256 with 16 KV tiles: both conditions hold, worth 1.06-1.11x.
        assert fp8_tile_choice(2048, 2048, 16, 1, SMS)[0] == 64
        assert fp8_tile_choice(2048, 2048, 8, 2, SMS)[0] == 64

    @pytest.mark.parametrize("s_q", [4096, 8192, 16384])
    def test_a_full_machine_takes_the_coarser_q_tile(self, s_q):
        assert fp8_tile_choice(s_q, s_q, 16, 1, SMS)[0] == 128

    def test_the_grid_bound_sits_between_240_and_320_ctas(self):
        """Where the two tiles stop trading evenly on a 188-SM part: 240 CTAs
        still want the finer tile, 320 want the coarser one by 1.19x."""
        assert fp8_tile_choice(2560, 2560, 12, 1, SMS)[0] == 64  # grid 240
        assert fp8_tile_choice(2560, 2560, 16, 1, SMS)[0] == 128  # grid 320

    def test_batch_and_heads_count_toward_the_grid(self):
        """The rule reads grid, not sequence length: the same s_q flips once
        the batch supplies enough CTAs on its own."""
        assert fp8_tile_choice(2048, 2048, 16, 1, SMS)[0] == 64
        assert fp8_tile_choice(2048, 2048, 16, 4, SMS)[0] == 128

    def test_choice_is_always_in_the_domain(self):
        for s_q in (128, 512, 1024, 4096, 32768):
            for b in (1, 8):
                q, kv = fp8_tile_choice(s_q, s_q, 16, b, SMS)
                assert q in SEQ_Q_TILES and kv in SEQ_KV_TILES

    def test_unknown_sm_count_falls_back_without_dividing_by_zero(self):
        assert fp8_tile_choice(4096, 4096, 16, 1, 0) == (128, 128)
