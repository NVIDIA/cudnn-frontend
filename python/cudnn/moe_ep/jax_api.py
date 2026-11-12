# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit JAX inference/training over the native SM107 MegaMoE kernels."""

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._config import MoeEpConfig, MoeEpFc1WeightLayout, ResolvedMoeEpConfig, _ResolvedMoeEpTopology
from ._metadata import MoeEpNativeWeightStorageMode


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


class MoeEpJax:
    """Use a caller-initialized NVSHMEM world with stable symmetric scratch.

    Prepare and close collectively on every PE. Invoke the same phase sequence
    on all PEs with identical configuration and native weight layouts. JAX
    process count and visible-device count need not equal the EP world size.
    Every operand belongs to the selected local GPU; use one plan per PE.

    Methods work eagerly and under ``jax.jit``. Automatic differentiation and
    ``vmap`` are unsupported; training uses explicit forward/backward methods.
    Inspect each result's status before consuming its values. Invalid expert
    IDs are safely dropped and reported; capacity overflow is reported after
    the native communication protocol completes. No device-to-host read occurs.
    """

    def __init__(self, resolved, bindings, device):
        self.config = resolved.public_config
        self.rank = resolved.topology.ep_rank
        self.world_size = resolved.topology.ep_size
        self.device = device
        self._bindings = bindings

    def close(self):
        """Collectively free scratch after every PE blocks on its final JAX result."""
        self._bindings.close()

    def __enter__(self):
        if self._bindings.resources._closed:
            raise RuntimeError("the JAX MoeEP plan is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def _array(self, name, value, shape, dtype):
        if not isinstance(value, (jax.Array, jax.core.Tracer)):
            raise TypeError(f"{name} must be a JAX array")
        if value.shape != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}; got {value.shape}")
        if value.dtype != jnp.dtype(dtype):
            raise TypeError(f"{name} must have dtype {dtype}; got {value.dtype}")
        if not isinstance(value, jax.core.Tracer) and value.devices() != {self.device}:
            raise ValueError(f"{name} must be on the prepared GPU {self.device}")

    def _inputs(self, source, indices, scores, weights, *, backward):
        from ._megamoe_backend.mxfp8._jax_bindings import weight_specs

        if self._bindings.resources._closed:
            raise RuntimeError("the JAX MoeEP plan is closed")
        if not isinstance(source, (jax.Array, jax.core.Tracer)):
            raise TypeError("activation must be a JAX array")
        model = self.config.model
        if source.ndim != 2 or source.shape[1] != model.hidden_size:
            raise ValueError(f"activation must have shape [T,{model.hidden_size}]")
        tokens = source.shape[0]
        if not isinstance(tokens, Integral) or not 0 <= tokens <= self.config.parallel.max_tokens_per_rank:
            raise ValueError("activation token count must be static and within max_tokens_per_rank")
        if source.dtype not in (jnp.bfloat16, jnp.float32):
            raise TypeError("activation must have dtype bfloat16 or float32")
        self._array("activation", source, source.shape, source.dtype)
        self._array("topk_indices", indices, (tokens, model.top_k), "int32")
        self._array("topk_scores", scores, (tokens, model.top_k), "float32")
        expected = MoeEpJaxBackwardWeights if backward else MoeEpJaxForwardWeights
        if not isinstance(weights, expected):
            raise TypeError(f"weights must be {expected.__name__}")
        config = self._bindings.phases["training_backward" if backward else "training_forward"].config
        for name, value, shape in zip(weights._fields, weights, weight_specs(config, backward=backward)):
            self._array(name, value, shape, "uint8")

    def inference(self, activation, topk_indices, topk_scores, weights, *, dependency=None):
        """Return BF16 output/status; chain independent calls through ``dependency``."""
        self._inputs(activation, topk_indices, topk_scores, weights, backward=False)
        if dependency is not None:
            self._array("dependency", dependency, (2,), "int32")
        return MoeEpJaxForwardResult(*self._bindings.call("inference", activation, topk_indices, topk_scores, weights, dependency=dependency))

    def training_forward(self, activation, topk_indices, topk_scores, weights, *, dependency=None):
        """Return output and immutable saved tensors for explicit backward."""
        self._inputs(activation, topk_indices, topk_scores, weights, backward=False)
        if dependency is not None:
            self._array("dependency", dependency, (2,), "int32")
        return MoeEpJaxTrainingForwardResult(*self._bindings.call("training_forward", activation, topk_indices, topk_scores, weights, dependency=dependency))

    def training_backward(self, grad_output, topk_indices, topk_scores, weights, forward, *, dependency=None):
        """Return dactivation, dprob and WGrad operands using matching saved state.

        Routing and weights must correspond to the supplied forward result.
        The caller owns these values; subsequent forwards cannot overwrite them.
        """
        from ._megamoe_backend.mxfp8._jax_bindings import output_specs

        self._inputs(grad_output, topk_indices, topk_scores, weights, backward=True)
        if not isinstance(forward, MoeEpJaxTrainingForwardResult):
            raise TypeError("forward must be a MoeEpJaxTrainingForwardResult from the matching training_forward call")
        specs = output_specs(self._bindings.phases["training_forward"], grad_output.shape[0])
        for name, value, spec in zip(forward._fields, forward, specs):
            self._array("forward." + name, value, spec.shape, spec.dtype)
        if dependency is not None:
            self._array("dependency", dependency, (2,), "int32")
        dx, status, dprob, fc1_b, fc1_sfb, fc2_a, fc2_sfa, fc2_b, fc2_sfb = self._bindings.call(
            "training_backward", grad_output, topk_indices, topk_scores, weights, forward.fc1_preact, dependency=dependency
        )
        operands = MoeEpJaxWgradOperands(
            forward.fc1_a, forward.fc1_sfa, fc1_b, fc1_sfb, fc2_a, fc2_sfa, fc2_b, fc2_sfb, forward.expert_offsets, forward.valid_route_counts
        )
        return MoeEpJaxBackwardResult(dx, dprob, status, operands)


def prepare_moe_ep_jax(config: MoeEpConfig, *, device: jax.Device, rank: int, world_size: int) -> MoeEpJax:
    """Collectively prepare native SM107 MXFP8 MoE on a caller-owned NVSHMEM world.

    NVSHMEM must already be initialized on ``device``. Its PE rank and size
    must equal ``rank`` and ``world_size``; expert IDs use that PE order.
    ``config.parallel.ep_group`` must be None. The plan never initializes or
    finalizes NVSHMEM. All PEs must use identical geometry, tuning and capacity.
    Requires the package's ``comm`` extra and a CuTeDSL version with SM107.
    """
    from ._megamoe_backend._capability import validate_config

    if not isinstance(config, MoeEpConfig):
        raise TypeError("config must be a MoeEpConfig")
    if config.parallel.ep_group is not None:
        raise ValueError("JAX MoeEP uses an initialized NVSHMEM world; ep_group must be None")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("rank must be within [0, world_size)")
    if config.model.num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size")
    if config.training_weight_storage_mode is not MoeEpNativeWeightStorageMode.CONTIGUOUS:
        raise NotImplementedError("JAX MoeEP currently accepts contiguous native weight carriers")
    if config.data_path.fc1_weight_layout is not MoeEpFc1WeightLayout.GATE_UP_INTERLEAVED_32:
        raise ValueError("JAX MoeEP native weights require fc1_weight_layout=GATE_UP_INTERLEAVED_32")
    resolved = ResolvedMoeEpConfig(config, _ResolvedMoeEpTopology(world_size, rank, tuple(range(world_size)), config.model.num_experts // world_size))
    validate_config(resolved)
    if not isinstance(device, jax.Device) or device.platform != "gpu" or device.process_index != jax.process_index():
        raise ValueError("device must be a local JAX CUDA GPU")

    from cudnn._device import device_info
    from cudnn.frost.device import device_context
    from ._megamoe_backend.mxfp8._cutedsl import require_rubin_cutedsl

    require_rubin_cutedsl()
    from cudnn.frost.buffers import cutedsl_arch_requirement_error
    from cutlass.base_dsl.version_info import CUDA_VERSION

    requirement_error = cutedsl_arch_requirement_error((10, 7))
    if requirement_error:
        raise NotImplementedError(requirement_error)
    if (CUDA_VERSION.major, CUDA_VERSION.minor) < (13, 4):
        raise NotImplementedError(
            "JAX MegaMoE SM107 requires the CuTe DSL CUDA 13.4 compiler or newer; "
            f"found {CUDA_VERSION.major}.{CUDA_VERSION.minor}. Install nvidia-cutlass-dsl[cu13]"
        )
    ordinal = device.local_hardware_id
    if device_info(ordinal).compute_capability != (10, 7):
        raise NotImplementedError("JAX MegaMoE requires Rubin SM107 (compute capability 10.7)")
    with device_context(ordinal):
        import cutlass.utils as utils

        from ._megamoe_backend.mxfp8._config import MXFP8_CLUSTER_SHAPE_MNK
        from ._megamoe_backend.mxfp8._jax_bindings import NativeBindings

        cluster_size = MXFP8_CLUSTER_SHAPE_MNK[0] * MXFP8_CLUSTER_SHAPE_MNK[1]
        launch_clusters = int(utils.HardwareInfo(device_id=ordinal).get_max_active_clusters(cluster_size))
        if launch_clusters <= 0:
            raise RuntimeError("hardware occupancy query returned no launchable SM107 clusters")
        return MoeEpJax(resolved, NativeBindings(resolved, device=ordinal, launch_clusters=launch_clusters), device)
