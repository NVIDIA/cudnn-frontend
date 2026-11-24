# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel bodies and configs of the FROST row argmax kernels.  ``row_argmax_decode_config`` imports without the CuTe DSL and
without torch; the ``*_f16`` bodies import the DSL (the host entry imports them after its version gate)."""
