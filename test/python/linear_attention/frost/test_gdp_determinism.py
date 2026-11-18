# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from linear_attention.determinism import LinearAttentionDeterminism, check_partitions, check_reference, pack, run, sequences


class TestGDP(LinearAttentionDeterminism):
    variant = "gdp"

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("num_householder", [1, 2], ids=["householder1", "householder2"])
    def test_householder_steps(self, backward, num_householder):
        samples = sequences(self.variant, (17, 3), num_householder=num_householder)
        inputs = pack(samples)
        check_reference(self.variant, inputs, run(self.variant, inputs, backward), backward)
        check_partitions(self.variant, samples, backward)
