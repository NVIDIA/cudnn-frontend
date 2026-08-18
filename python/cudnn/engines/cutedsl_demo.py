# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AOT export reference family: two small CuTeDSL elementwise-add engines.

They exist to be exported, not to be fast. The samples under ``samples/aot``
and ``test/python/test_aot_export.py`` build against them because a kernel
small enough to read end-to-end is what makes the export path reviewable; the
family is opt_in so it never competes for a real pointwise graph.
"""

from __future__ import annotations

from typing import Dict, List


def CuteDslDemoEngines(ids: Dict[str, int]) -> List:
    """The demo engines the manifest asked for. ``ids`` is ``{name: engine_id}``."""
    from .cutedsl_pointwise_engine import CuteDslPointwiseAddEngine
    from .cutedsl_tma_add_engine import CuteDslTmaAddEngine

    classes = {
        "cutedsl_pointwise_add": CuteDslPointwiseAddEngine,
        "cutedsl_tma_add": CuteDslTmaAddEngine,
    }
    return [classes[name](engine_id) for name, engine_id in ids.items() if name in classes]
