# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer's flashinfer/batch_invariant/fp_interface/__init__.py (26075a1331ff).

"""The floating-point interface of the batch-invariant quantizers, in its backends.

* :mod:`.exact`: exact IEEE-754 arithmetic on the CPU (the reference model);
* :mod:`.term`: symbolic terms that record which operations were applied in which association;
* :mod:`.cute`: the CuTe-DSL primitives with explicit ``.rn`` rounding the kernels are allowed to use.

Schedule code is written once against an ``fp`` object and runs under any of them.
"""
