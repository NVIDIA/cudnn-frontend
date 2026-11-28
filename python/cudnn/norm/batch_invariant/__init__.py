# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm forward with fixed per-row arithmetic for rollout and training.

The package separates the arithmetic from its placement on the GPU:

- ``_spec``: the numerical class, a function of the hidden size and the
  input dtype only.
- ``_knobs``: the legal placements (threads per row, rows per CTA). Every
  placement computes the same class.
- ``_heuristic``: a device-aware cost model over the placements.
- ``_schedule``: picks the class from the graph and the placement from the
  device and the row count.
- ``kernel``: the CuTe DSL kernel and its compiled launcher.
- ``engine``: the graph engine behind the ``BATCH_INVARIANT`` numerical note.
"""

from ._spec import RmsNormClass as RmsNormClass
from ._spec import class_for as class_for
