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

    @pytest.mark.L0
    @pytest.mark.parametrize("backward", [False, True], ids=["fprop", "bprop"])
    @pytest.mark.parametrize("eps", [None, 1e-10], ids=["eps-default", "eps-1e-10"])
    def test_qk_l2norm(self, backward, eps):
        """The fused q/k normalization (``use_qk_l2norm_in_kernel``) at its default epsilon and at a ``qk_l2norm_eps``:
        the normalize kernel runs ahead of the main kernels in the same launch, so its rows, and the inverse norms the
        backward projects dQ / dK through, are part of the replayed and batch-invariant bytes.  Half of the q / k rows
        are scaled to squared norms around both epsilons, where the normalized row depends on the epsilon."""
        samples = sequences(self.variant, (65, 17, 3))
        for sample in samples:
            for name in ("q", "k"):
                sample[name][0::4] *= 1e-4
                sample[name][1::4] *= 1e-5
        options = dict(use_qk_l2norm_in_kernel=True) if eps is None else dict(use_qk_l2norm_in_kernel=True, qk_l2norm_eps=eps)
        inputs = pack(samples)
        check_reference(self.variant, inputs, run(self.variant, inputs, backward, **options), backward, l2norm_eps=1e-6 if eps is None else eps)
        check_partitions(self.variant, samples, backward, **options)
