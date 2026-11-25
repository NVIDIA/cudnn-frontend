# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owned D64 specialization, sharing only immutable SM90 parameter values."""

from dataclasses import asdict, dataclass

from cudnn.sdpa.fwd.config_sm90 import (
    TemplateParams as SharedTemplateParams,
    native_d64_mismatch,
    validate_params as validate_shared_params,
)


@dataclass(frozen=True)
class TemplateParams(SharedTemplateParams):
    """D64 record with the existing SM90 ABI and field defaults."""


def from_shared(params):
    """Freeze a shared adapter record as this implementation's specialization."""
    result = params if type(params) is TemplateParams else TemplateParams(**asdict(params))
    validate_params(result)
    return result


def validate_params(params):
    validate_shared_params(params)
    why = native_d64_mismatch(
        thd=params.thd_varlen,
        sink=params.has_sink,
        pack_gqa=params.pack_gqa,
        q_lengths=params.seq_q_lens_present,
        kv_lengths=params.seq_kv_lens_present,
        window_left=params.window_left,
        window_right=params.window_right,
        split_kv=params.split_kv,
    )
    if why is not None:
        raise ValueError(why)
    if params.page_size:
        raise ValueError("native SM90 D64 does not support paged KV")
