# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""DSL SDPA-forward kernel templates, grouped by target architecture.

Layout
------
One package per arch line — ``sm80/``, ``sm100/`` (SM100/SM103), ``sm107/``
(Rubin), ``sm120/`` (SM120/SM121) — holding the flavor templates that arch
owns.  Within a package the filename encodes the rest of the coverage matrix:
``<phase>_d<dim>_<dtype-family>.py``, e.g. ``sm100/prefill_d512_f16.py``
(``f16`` covers fp16 and bf16, picked by ``TemplateParams``).

A file omits the dimension when one implementation covers a range of head
dims: ``sm120/prefill_f16.py`` and ``sm120/prefill_fp8.py`` do, while
``sm120/prefill_d256_f16.py`` is the d256 flavor (head dims that tile at 256 on
both sides). ``sm120/prefill_d512_f16.py`` serves both head dims in (256, 512]
at multiples of eight, with two warps per Q slab splitting the head dimension.

Modules shared across arch lines stay at THIS level rather than inside one
arch's package, so the directory a file lives in always names its only owner:

- ``_common_blackwell.py`` — the tcgen05 pipeline helpers shared by ``sm100/``
  and ``sm107/`` (cc 100-119; ``engines._BLACKWELL`` spans the same range).
- ``thd_helpers.py`` — the THD/varlen metadata + per-batch O-descriptor setup,
  used by ``sm100/``, ``sm107/`` and ``sm120/``.

Loading
-------
Every template specializes on its architecture's frozen ``TemplateParams`` at
import time (module global ``FROST_TEMPLATE_PARAMS``, injected by
``cudnn.frost.template_loader.load_template``). Tensor geometry remains an
input to each module's cached ``compile()`` function. The adapter's
``_SM*_KERNEL_FILES`` maps hold paths RELATIVE to this directory
(``"sm107/prefill_d128_fp8.py"``), which the loader joins onto it. Import a
template directly only for its all-defaults standalone path.

The SM80 templates (``sm80/prefill_f16.py``, ``sm80/prefill_d256_f16.py``, and
the ``bprop_*_sm80.py`` pair under ``bwd/kernels/``) were vendored 2026-07 from
an internal tile-kernel repository that has since been retired; they are
maintained in-tree from here on. They predate the ``TemplateParams`` loader:
they self-cache per shape and take masks/features as runtime kwargs — import
them directly and call ``forward``/``backward``.
"""
