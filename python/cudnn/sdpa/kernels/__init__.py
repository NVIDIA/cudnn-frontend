# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Modules shared by the SDPA kernel packages of BOTH phases, ``sdpa/fwd/kernels/`` and ``sdpa/bwd/kernels/``.

Layout
------
Each phase package keeps the modules its own arch lines share at its ``kernels/`` level (``fwd/kernels/_common_blackwell.py``,
``bwd/kernels/bprop_chain_common.py``) so the directory a file lives in names its only owner
(``python/cudnn/frost/README.md``).  A module the FORWARD templates and the BACKWARD templates both import belongs to
neither phase; it sits here, one level above both, for the same reason.

- ``_mxfp8_sf.py`` -- the MXFP8 scale-factor TMA descriptors (rowwise per-tile, columnwise D-plane-major) and the cga2
  peer split of an SF slab.  The Rubin MXFP8 forwards (``fwd/kernels/sm107/prefill_d256_mxfp8.py``,
  ``fwd/kernels/sm107/prefill_d192_d128_mxfp8.py``) build theirs here; the MXFP8 d=256 backward chain and the
  stage-3 block-scale GEMM arm are the next consumers.

Nothing here is a template: these are plain Python helpers that trace inside a ``@cute.jit`` host or a ``@cute.kernel``.
"""
