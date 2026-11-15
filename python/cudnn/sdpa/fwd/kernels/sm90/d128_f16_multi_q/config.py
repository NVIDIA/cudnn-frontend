# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated D128 parameters over the shared immutable Hopper ABI."""

from dataclasses import asdict, dataclass
from cudnn.sdpa.fwd.config_sm90 import D128_FLAVOR, TemplateParams as SharedTemplateParams, validate_params as validate_shared_params


@dataclass(frozen=True)
class TemplateParams(SharedTemplateParams):
    """The dedicated 128 x 128 row-split pipeline's specialization."""


def validate_params(params, flavor=D128_FLAVOR):
    if flavor != D128_FLAVOR:
        raise ValueError("the dedicated D128 implementation requires its D128 flavor")
    validate_shared_params(params, flavor=D128_FLAVOR)


def from_shared(params):
    result = params if type(params) is TemplateParams else TemplateParams(**asdict(params))
    validate_params(result)
    return result
