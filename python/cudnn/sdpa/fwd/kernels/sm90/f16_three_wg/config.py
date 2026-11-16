# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned three-warpgroup specialization with the shared immutable SM90 ABI."""

from dataclasses import asdict, dataclass
from cudnn.sdpa.fwd.config_sm90 import TemplateParams as SharedTemplateParams, validate_params


@dataclass(frozen=True)
class TemplateParams(SharedTemplateParams):
    """The existing three-warpgroup geometry and field defaults."""


def from_shared(params):
    result = params if type(params) is TemplateParams else TemplateParams(**asdict(params))
    validate_params(result)
    return result
