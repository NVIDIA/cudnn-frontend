# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep unbenchmarked norm/block-scale composition behind backend plans."""

from cudnn.engines.base import PlanConfig
from cudnn.engines.heuristics import BACKEND


def propose(kind, facts, offered):
    plans = [PlanConfig(engine_id) for engine_id in offered.values()]
    return [BACKEND, *plans] if facts.quantize_nodes else plans
