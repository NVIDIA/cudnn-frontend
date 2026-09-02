# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""DSL SDPA-backward kernel templates.

Filenames encode the coverage matrix: ``bprop_<dtype-family>_sm<arch>.py``.

Every template specializes on its architecture's frozen ``TemplateParams`` at
import time (module global ``FROST_TEMPLATE_PARAMS``, injected by
``cudnn.frost.template_loader.load_template``). Tensor geometry remains an
input to each module's cached ``compile()`` function. Import a template
directly only for its all-defaults standalone path.
"""
