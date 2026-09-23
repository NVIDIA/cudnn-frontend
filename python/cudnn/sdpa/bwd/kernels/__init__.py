# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""DSL SDPA-backward kernel templates, grouped by target architecture.

Layout
------
One package per arch line -- ``sm80/``, ``sm100/`` (SM100/SM103), ``sm107/``
(Rubin), ``sm120/`` (SM120/SM121) -- holding the kernels that arch owns, the
same shape as ``sdpa/fwd/kernels/``.  Within a package the filename encodes the rest of the
coverage matrix: ``bprop_d<dim>_<dtype-family>.py``, e.g.
``sm100/bprop_d512_f16.py`` (``f16`` covers fp16 and bf16, picked by
``TemplateParams``).  A file omits the dimension when one implementation
covers every supported head dim (``sm80/bprop_f16.py``, ``sm120/bprop_f16.py``);
``sm120/bprop_chain_f16.py`` is the SM120-only part of the launch chain around
that arch's fused main kernel (the deterministic dQ GEMM and the converts), and
the ``_common.py`` / ``_bprop_mxfp8_*.py`` modules inside a package are that
arch's private helpers.

Modules shared across arch lines stay at THIS level rather than inside one
arch's package, so the directory a file lives in always names its only owner:

- ``bprop_matmul_blackwell.py`` -- the stage-3 batched GEMM (dV / dK / dQ) of
  the large-head-dim backward chain; its codegen targets span the Blackwell
  line (SM100/SM103/SM107/SM110), which is why it does not sit in ``sm100/``.
- ``thd_helpers.py`` -- the THD/varlen metadata + setup kernels, used by
  ``sm80/`` and the sm100 chain.
- ``bprop_chain_common.py`` -- the arch-neutral launch-chain kernels (the
  ``dot`` delta preprocess, the GQA dK/dV reduce, ``dsink``), used by the
  sm120 chain (which re-exports them) and the sm100 chain.

Loading
-------
Every template specializes on its architecture's frozen ``TemplateParams`` at
import time (module global ``FROST_TEMPLATE_PARAMS``, injected by
``cudnn.frost.template_loader.load_template``). Tensor geometry remains an
input to each module's cached ``compile()`` function. The adapter's
``_SM*_KERNEL_FILE`` constants hold paths RELATIVE to this directory
(``"sm120/bprop_f16.py"``), which the loader joins onto it. Import a template
directly only for its all-defaults standalone path.

The SM100 d=256 MXFP8 kernels (``sm100/bprop_dq_d256_mxfp8``,
``sm100/bprop_dkdv_d256_mxfp8``, their shared ``sm100/_bprop_mxfp8_*`` helpers
and the ``sm100/bprop_sf_repack_mxfp8`` scale-factor repack) are the exception:
ported CuTe DSL kernel CLASSES that specialize through their constructors, so
they are ordinary importable modules with no template parameters.
"""
