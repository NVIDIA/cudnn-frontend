# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""DSL SDPA-backward kernel templates.

Filenames encode the coverage matrix: ``bprop_<dtype-family>_sm<arch>.py``.

Every template specializes on its architecture's frozen ``TemplateParams`` at
import time (module global ``FROST_TEMPLATE_PARAMS``, injected by
``cudnn.frost.template_loader.load_template``). Tensor geometry remains an
input to each module's cached ``compile()`` function. Import a template
directly only for its all-defaults standalone path.

The SM100 d=256 MXFP8 kernels (``bprop_dq_d256_mxfp8_sm100``,
``bprop_dkdv_d256_mxfp8_sm100``, their shared ``_bprop_mxfp8_*_sm100`` helpers
and the ``bprop_sf_repack_mxfp8_sm100`` scale-factor repack) are the exception:
ported CuTe DSL kernel CLASSES that specialize through their constructors, so
they are ordinary importable modules with no template parameters.
"""
