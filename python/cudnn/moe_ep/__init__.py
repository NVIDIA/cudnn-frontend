# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

_CONFIG_EXPORTS = {
    "MoeEpConfig",
    "MoeEpDataPathConfig",
    "MoeEpFc1WeightLayout",
    "MoeEpModelConfig",
    "MoeEpParallelConfig",
}
_TUNING_EXPORTS = {"MoeEpAutotuneCandidateResult", "MoeEpAutotuneResult", "MoeEpTuningConfig"}
_METADATA_EXPORTS = {"MoeFormat", "MoeEpNativeWeightStorageMode"}
_API_EXPORTS = {"MoeEp", "pack_backward_weights", "pack_forward_weights"}
_JAX_EXPORTS = {
    "MoeEpJax",
    "MoeEpJaxForwardWeights",
    "MoeEpJaxBackwardWeights",
    "MoeEpJaxForwardResult",
    "MoeEpJaxTrainingForwardResult",
    "MoeEpJaxBackwardResult",
    "MoeEpJaxWgradOperands",
    "prepare_moe_ep_jax",
}


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = "._types"
    for source, names in (
        ("._config", _CONFIG_EXPORTS),
        ("._tuning", _TUNING_EXPORTS),
        ("._metadata", _METADATA_EXPORTS),
        (".api", _API_EXPORTS),
        (".jax_api", _JAX_EXPORTS),
    ):
        if name in names:
            module = source
            break
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "BlockScaledTensor",
    "MoeEp",
    "MoeEpJax",
    "MoeEpJaxForwardWeights",
    "MoeEpJaxBackwardWeights",
    "MoeEpJaxForwardResult",
    "MoeEpJaxTrainingForwardResult",
    "MoeEpJaxBackwardResult",
    "MoeEpJaxWgradOperands",
    "prepare_moe_ep_jax",
    "MoeEpConfig",
    "MoeEpDataPathConfig",
    "MoeEpFc1WeightLayout",
    "MoeEpModelConfig",
    "MoeEpParallelConfig",
    "MoeEpAutotuneCandidateResult",
    "MoeEpAutotuneResult",
    "MoeEpBackwardWeightStaging",
    "MoeEpBackwardWeights",
    "MoeEpForwardWeightStaging",
    "MoeEpForwardWeights",
    "MoeEpNativeBackwardWeights",
    "MoeEpNativeDiscreteBackwardWeights",
    "MoeEpNativeDiscreteForwardWeights",
    "MoeEpNativeDiscreteWeight",
    "MoeEpNativeForwardWeights",
    "MoeEpNativeWeight",
    "MoeEpNativeWeightLayout",
    "MoeEpNativeWeightStorageMode",
    "MoeEpTrainingBackwardOutputs",
    "MoeEpTrainingForwardOutputs",
    "MoeEpTrainingWgradOperands",
    "MoeEpTuningConfig",
    "MoeFormat",
    "MoeTensor",
    "pack_backward_weights",
    "pack_forward_weights",
]
