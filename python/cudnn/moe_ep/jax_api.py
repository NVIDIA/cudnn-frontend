# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reserved JAX expert-parallel MoE contracts; execution is temporarily disabled."""

from typing import NamedTuple, NoReturn

import jax

from ._config import MoeEpConfig


class MoeEpJaxForwardWeights(NamedTuple):
    """Prepacked native E4M3/E8M0 byte carriers.

    FC1 is physical ``[E,2*I,H]`` with gate/up interleaved in groups of 32;
    FC2 is physical ``[E,H,I]``. Scale planes are flat 128x4 blocked E8M0
    bytes per expert, matching ``MoeEpNativeForwardWeights``.
    """

    fc1: jax.Array
    fc1_scale: jax.Array
    fc2: jax.Array
    fc2_scale: jax.Array


class MoeEpJaxBackwardWeights(NamedTuple):
    """Prepacked native backward E4M3/E8M0 byte carriers.

    W2 transpose is ``[E,H,I]``; W1 transpose is ``[E,2*I,H]`` with its
    reduction axis gate/up interleaved in groups of 32. Scales use flat
    128x4 blocked planes, matching ``MoeEpNativeBackwardWeights``.
    """

    w2_transpose: jax.Array
    w2_transpose_scale: jax.Array
    w1_transpose: jax.Array
    w1_transpose_scale: jax.Array


class MoeEpJaxForwardResult(NamedTuple):
    """Output and status ``[receive_overflow, invalid_expert_id]``; zero is success."""

    output: jax.Array
    status: jax.Array


class MoeEpJaxTrainingForwardResult(NamedTuple):
    """XLA-owned forward output and saved receiver-domain state.

    Counts and padded offsets define valid saved rows; padding is unspecified.
    """

    output: jax.Array
    status: jax.Array
    fc1_preact: jax.Array
    fc1_a: jax.Array
    fc1_sfa: jax.Array
    valid_route_counts: jax.Array
    expert_offsets: jax.Array


class MoeEpJaxWgradOperands(NamedTuple):
    """Native WGrad operand bytes; ``fc2_b`` is physical ``[H,pool_rows]``.

    Scale arrays retain each producer's blocked native layout. These are
    operands for grouped weight-gradient GEMMs, not completed weight gradients.
    """

    fc1_a: jax.Array
    fc1_sfa: jax.Array
    fc1_b: jax.Array
    fc1_sfb: jax.Array
    fc2_a: jax.Array
    fc2_sfa: jax.Array
    fc2_b: jax.Array
    fc2_sfb: jax.Array
    expert_offsets: jax.Array
    valid_route_counts: jax.Array


class MoeEpJaxBackwardResult(NamedTuple):
    """Activation/routing gradients, status, and explicit grouped-WGrad operands."""

    grad_activation: jax.Array
    dprob: jax.Array
    status: jax.Array
    wgrad: MoeEpJaxWgradOperands


def _raise_unavailable() -> NoReturn:
    raise NotImplementedError(
        "JAX expert-parallel MoE is temporarily unavailable pending support for its NVSHMEM launch path in the standard CuTe DSL JAX bridge"
    )


class MoeEpJax:
    """Reserved prepared-plan interface; JAX execution is temporarily unavailable."""

    def __init__(self, resolved, bindings, device):
        _raise_unavailable()

    def close(self):
        _raise_unavailable()

    def inference(self, activation, topk_indices, topk_scores, weights, *, dependency=None):
        _raise_unavailable()

    def training_forward(self, activation, topk_indices, topk_scores, weights, *, dependency=None):
        _raise_unavailable()

    def training_backward(self, grad_output, topk_indices, topk_scores, weights, forward, *, dependency=None):
        _raise_unavailable()


def prepare_moe_ep_jax(config: MoeEpConfig, *, device: jax.Device, rank: int, world_size: int) -> MoeEpJax:
    """Decline JAX expert-parallel MoE before importing or preparing a backend.

    This entry point is reserved until the native NVSHMEM launch path can use
    the standard CuTe DSL JAX bridge. No GPU resources are initialized.
    """
    _raise_unavailable()
