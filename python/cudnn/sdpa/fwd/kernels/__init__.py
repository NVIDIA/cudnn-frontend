# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""DSL SDPA-forward kernel templates.

Filenames encode the coverage matrix: ``<phase>_d<dim>_<dtype-family>_sm<arch>.py``
(e.g. ``prefill_d512_f16_sm100.py`` — f16 covers fp16 and bf16, picked by TemplateParams).

``prefill_f16_sm120.py`` omits the dimension because one implementation covers
all supported head dimensions and also runs on SM121.

Every template specializes on its architecture's frozen ``TemplateParams`` at
import time (module global ``FROST_TEMPLATE_PARAMS``, injected by
``cudnn.frost.template_loader.load_template``). Tensor geometry remains an
input to each module's cached ``compile()`` function. Import a template
directly only for its all-defaults standalone path.

The SM80 templates (``prefill_f16_sm80.py``, ``prefill_d256_f16_sm80.py``,
``bprop_f16_sm80.py``, ``bprop_d64_f16_sm80.py``) were vendored 2026-07 from
an internal tile-kernel repository that has since been retired; they are
maintained in-tree from here on. They predate the ``TemplateParams`` loader:
they self-cache per shape and take masks/features as runtime kwargs — import
them directly and call ``forward``/``backward``.
"""
