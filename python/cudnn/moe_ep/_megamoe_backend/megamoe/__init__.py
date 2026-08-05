# Hackable single-call MegaMoE forward (CuTe DSL mxfp8 GLU kernel) for use
# inside a PyTorch pipeline.  See README.md in this directory.
from megamoe.forward import MegaMoeMxfp8Forward, MegaMoeForwardConfig
from megamoe.weights import quantize_moe_weights_mxfp8, QuantizedExpertWeights

__all__ = [
    "MegaMoeMxfp8Forward",
    "MegaMoeForwardConfig",
    "quantize_moe_weights_mxfp8",
    "QuantizedExpertWeights",
]
