# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CAKE engine for linear attention: frozen CUDA C++ kernel bodies generated
by the CAKE pipeline, hosted as their own engine under the ``kimi_delta_attention``
op next to ``kda_frost`` and ``kda_cutile``.

``kernels/`` holds the bodies byte-for-byte as exported (see ``kernels/UPSTREAM.md``);
``compiler.py`` compiles them with NVRTC and launches through the driver API;
``kda_host.py`` is the host side of the C16 training route (planning, TMA
descriptors, workspace, launch sequence); ``kda_engine.py`` is the engine.
"""
