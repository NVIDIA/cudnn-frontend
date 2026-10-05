# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize the small graph envelope served by the Frost convolution engine.

The analyzer deliberately separates graph matching from kernel compilation so
new fusion patterns can be added without growing conditionals in the engine.
Currently accepted patterns are a single convolution and a convolution whose
output feeds one supported unary pointwise epilogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from cudnn import pygraph, GraphContext, Node, NodeType, Tensor

SUPPORTED_UNARY_EPILOGUES = frozenset(
    (
        "abs",
        "ceil",
        "cos",
        "elu",
        "erf",
        "exp",
        "floor",
        "gelu",
        "gelu_approx_tanh",
        "identity",
        "leaky_relu",
        "log",
        "logical_not",
        "neg",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "sin",
        "softplus",
        "sqrt",
        "swish",
        "tan",
        "tanh",
    )
)


@dataclass(frozen=True)
class BlockScaleData:
    """Extra nodes and scale operands for block-scaled convolution inputs."""

    sfa: Tensor
    sfb: Tensor
    image_dequant_node: Node
    weight_dequant_node: Node
    dequantized_image: Tensor
    dequantized_weight: Tensor


@dataclass(frozen=True)
class OutputQuantizeData:
    """The fused block-scale quantizer used to produce an NVFP4 output."""

    quantize_node: Node
    input: Tensor
    scale: Tensor
    group_offset: Any = None


@dataclass(frozen=True)
class ConvGraph:
    """The kernel-relevant tensors and specialization selected from a graph."""

    context: GraphContext  # dtypes
    conv_node: Node
    image: Tensor  # The graph inputs. For nvfp4 inputs, the nvfp4 input tensor without scaling factor. See also block_scale_data.sfa/sfb.
    weight: Tensor
    output: Tensor  # The graph output. For nvfp4 outputs, the nvfp4 tensor without scaling factor. See also output_quantize_data.scale.
    epilogue: str = "identity"
    epilogue_attrs: tuple[tuple[str, float], ...] = ()
    epilogue_node: Optional[Node] = None
    block_scale_data: BlockScaleData | None = None
    output_quantize_data: OutputQuantizeData | None = None


def _conv_operands(node: Node) -> tuple[Tensor, Tensor, Tensor]:
    unexpected = set(node.inputs) - {"image", "weight"}
    if unexpected:
        raise NotImplementedError(f"frost_conv: convolution does not support extra operands {tuple(sorted(unexpected))}")
    try:
        return node.inputs["image"], node.inputs["weight"], node.outputs["Y"]
    except KeyError as exc:
        raise NotImplementedError(f"frost_conv: missing convolution port {exc}") from exc


def _dequant_operands(node: Node) -> tuple[Any, Any, Any]:
    unexpected = set(node.inputs) - {"input", "descale"}
    if unexpected:
        raise NotImplementedError(f"frost_conv: block-scale dequantize does not support extra operands {tuple(sorted(unexpected))}")
    try:
        return node.inputs["input"], node.inputs["descale"], node.outputs["OUT_0"]
    except KeyError as exc:
        raise NotImplementedError(f"frost_conv: missing block-scale dequantize port {exc}") from exc


def _match_conv_node(graph: pygraph) -> tuple[Node, Tensor, Tensor, Tensor]:
    """Match the sinlge convolution node in the graph. Throws if not found or more than 1 are found."""
    conv_nodes = [n for n in graph.nodes if n.node_type == NodeType.CONV_FPROP]
    if len(conv_nodes) != 1:
        raise NotImplementedError(f"frost_conv: expects exactly one convolution node, but got {len(conv_nodes)}")
    conv_node = conv_nodes[0]
    image, weight, output = _conv_operands(conv_node)
    return image, weight, output, conv_node


def _match_block_scale_dequantize(graph: pygraph, output_tensor: Tensor) -> Optional[tuple]:
    """Match an optional BLOCK_SCALE_DEQUANTIZE type node whose output is output_tensor. Returns None if not found."""
    block_scale_dequantize_nodes = [n for n in graph.nodes if n.node_type == NodeType.BLOCK_SCALE_DEQUANTIZE and output_tensor in n.outputs.values()]
    if len(block_scale_dequantize_nodes) == 0:
        return None
    elif len(block_scale_dequantize_nodes) > 1:
        raise NotImplementedError(
            f"frost_conv: expects exactly one block-scale dequantize node for tensor {output_tensor.name}, but got {len(block_scale_dequantize_nodes)}"
        )

    block_scale_dequantize_node = block_scale_dequantize_nodes[0]
    input, scaling_factor, _ = _dequant_operands(block_scale_dequantize_node)
    return input, scaling_factor, block_scale_dequantize_node


def _match_pointwise_epilogue_node(graph: pygraph, tensor: Tensor, supported_epilogues=SUPPORTED_UNARY_EPILOGUES) -> Optional[tuple]:
    """Match an optional POINTWISE type epilogue node after a tensor."""
    epilogue_nodes = [n for n in graph.nodes if n.node_type == NodeType.POINTWISE and tensor in n.inputs.values()]
    if len(epilogue_nodes) == 0:
        return None
    elif len(epilogue_nodes) > 1:
        raise NotImplementedError(f"frost_conv: expects exactly one epilogue node for tensor {tensor.name}, but got {len(epilogue_nodes)}")

    epilogue_node = epilogue_nodes[0]
    epilogue = epilogue_node.params.get("mode")
    if epilogue not in supported_epilogues:
        supported = ", ".join(sorted(supported_epilogues))
        raise NotImplementedError(f"frost_conv: unsupported unary epilogue {epilogue!r}; supported modes: {supported}")

    scalar_attrs = {name: value for name, value in epilogue_node.params.items() if name != "mode" and value is not None}
    try:
        epilogue_attrs = tuple(sorted((name, float(value)) for name, value in scalar_attrs.items()))
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(f"frost_conv: unary epilogue {epilogue!r} attributes must be numeric") from exc

    epilogue_output = next(iter(epilogue_node.outputs.values()))
    return epilogue, epilogue_attrs, epilogue_node, epilogue_output


def _match_block_scale_quantize(graph: pygraph, tensor: Tensor) -> Optional[tuple]:
    """Match an optional BLOCK_SCALE_QUANTIZE type node after a tensor. Returns None if not found."""
    block_scale_quantize = [n for n in graph.nodes if n.node_type == NodeType.BLOCK_SCALE_QUANTIZE and tensor in n.inputs.values()]
    if len(block_scale_quantize) == 0:
        return None
    elif len(block_scale_quantize) > 1:
        raise NotImplementedError(f"frost_conv: expects exactly one block quantize node for tensor {tensor.name}, but got {len(block_scale_quantize)}")

    block_scale_quantize_node = block_scale_quantize[0]

    unexpected = set(block_scale_quantize_node.inputs) - {"input", "group_offset"}
    if unexpected:
        raise NotImplementedError(f"frost_conv: block-scale quantize does not support extra operands {tuple(sorted(unexpected))}")

    try:
        graph_output = block_scale_quantize_node.outputs["Y"]
        scale = block_scale_quantize_node.outputs["scale"]
    except KeyError as exc:
        raise NotImplementedError(f"frost_conv: missing block-scale quantize output port {exc}") from exc

    output_quantize_data = OutputQuantizeData(
        quantize_node=block_scale_quantize_node,
        input=block_scale_quantize_node.inputs.get("input"),
        scale=scale,
        group_offset=block_scale_quantize_node.inputs.get("group_offset"),
    )
    return output_quantize_data, graph_output


def analyze(graph: pygraph) -> ConvGraph:
    """Describe a structurally supported Frost convolution graph.

    The accepted graph grammar is one convolution, optionally followed by one
    supported unary pointwise epilogue::

        CONV_FPROP(image, weight) [-> POINTWISE] [-> BLOCK_SCALE_QUANTIZE] -> output

    Each convolution input may instead be supplied by one direct block-scale
    dequantization, but both inputs must use it together::

        BLOCK_SCALE_DEQUANTIZE(A, SFA) --+
                                          +-> CONV_FPROP [-> POINTWISE] [-> BLOCK_SCALE_QUANTIZE] -> output
        BLOCK_SCALE_DEQUANTIZE(B, SFB) --+

    No other nodes are accepted. Dequantized tensors must remain virtual and
    feed the convolution directly. When a pointwise node is present, the
    convolution result must remain virtual, the pointwise node must consume it
    directly. A terminal block-scale quantizer consumes that result and
    materializes both its data and scale outputs. No other operation results may
    be materialized in a block-scaled graph.

    The returned :class:`ConvGraph` always exposes the buffers a plan binds as
    ``image``, ``weight``, and ``output``. Dense graphs have
    ``block_scale_data=None``. Block-scaled graphs attach SFA/SFB and their
    dequantization nodes/tensors in :class:`BlockScaleData`. ``intermediate``
    identifies the virtual convolution result when a fused suffix is present,
    while ``epilogue_node`` is populated only for a pointwise epilogue. An
    optional terminal block-scale quantizer is exposed as
    ``output_quantize_data``.

    This function checks topology and fusion structure only. Device, dtype,
    shape, stride, scale layout, block size, and the narrower executable
    epilogue contract are validated later by ``FrostConvEngine.check_support``.

    Raises:
        NotImplementedError: If the graph does not match one of these forms.
    """
    image, weight, conv_output, conv_node = _match_conv_node(graph)
    matched_nodes: set[Node] = {conv_node}
    graph_output: Tensor = conv_output

    # Either both inputs are nvfp4, or neither of them are.
    block_scale_data: Optional[BlockScaleData] = None
    image_block_scale = _match_block_scale_dequantize(graph, image)
    weight_block_scale = _match_block_scale_dequantize(graph, weight)
    if image_block_scale or weight_block_scale:
        if not image_block_scale or not weight_block_scale:
            raise NotImplementedError("frost_conv: It is not supported for now that only one of image or weight is nvfp4.")

        quantized_image, sf_image, image_dequant_node = image_block_scale
        quantized_weight, sf_weight, weight_dequant_node = weight_block_scale
        block_scale_data = BlockScaleData(
            sfa=sf_image,
            sfb=sf_weight,
            image_dequant_node=image_dequant_node,
            weight_dequant_node=weight_dequant_node,
            dequantized_image=image,
            dequantized_weight=weight,
        )
        matched_nodes.add(image_dequant_node)
        matched_nodes.add(weight_dequant_node)
        image, weight = quantized_image, quantized_weight

    epilogue, epilogue_attrs, epilogue_node = "identity", tuple(), None
    if match := _match_pointwise_epilogue_node(graph, graph_output):
        epilogue, epilogue_attrs, epilogue_node, graph_output = match
        matched_nodes.add(epilogue_node)

    output_quantize_data: Optional[OutputQuantizeData] = None
    if match := _match_block_scale_quantize(graph, graph_output):
        output_quantize_data, graph_output = match
        matched_nodes.add(output_quantize_data.quantize_node)

    # Reject extra nodes that are not matched before.
    extra_nodes = [n for n in graph.nodes if n not in matched_nodes]
    if extra_nodes:
        raise NotImplementedError(f"Unrecognized node {extra_nodes}")

    # Only the initial input and final ouptut are materialized. All intermediate tensors must be virtual.
    for tensor in graph.tensors.values():
        is_initial_input = any(tensor in n.inputs.values() for n in graph.nodes) and all(tensor not in n.outputs.values() for n in graph.nodes)
        is_final_output = any(tensor in n.outputs.values() for n in graph.nodes) and all(tensor not in n.inputs.values() for n in graph.nodes)
        if is_initial_input and tensor.is_virtual:
            raise NotImplementedError(f"Initial input tensor {tensor.name} must not be virtual")
        if is_final_output and tensor.is_virtual:
            raise NotImplementedError(f"Final output tensor {tensor.name} must not be virtual")
        if (not is_initial_input and not is_final_output) and not tensor.is_virtual:
            raise NotImplementedError(f"Intermediate tensor {tensor.name} must be virtual")

    return ConvGraph(
        context=graph.context,
        conv_node=conv_node,
        image=image,
        weight=weight,
        output=graph_output,
        epilogue=epilogue,
        epilogue_attrs=epilogue_attrs,
        epilogue_node=epilogue_node,
        block_scale_data=block_scale_data,
        output_quantize_data=output_quantize_data,
    )


__all__ = ["BlockScaleData", "ConvGraph", "OutputQuantizeData", "SUPPORTED_UNARY_EPILOGUES", "analyze"]
