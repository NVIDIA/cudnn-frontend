# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel bodies and configs of the FROST attention-gate kernels.  ``gate_quant_decode_config`` imports without the CuTe DSL; the ``*_f16``
bodies import it (the host entry imports them after its version gate)."""
