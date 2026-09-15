# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared graph builders and references for Frost convolution epilogues."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EpilogueCase:
    """A graph builder and its FP32 PyTorch reference."""

    name: str
    mode: str
    graph_fn: Callable
    reference_fn: Callable[[torch.Tensor], torch.Tensor]
    input_profile: str = "signed"


def _graph_unary(mode: str) -> Callable:
    def apply(graph, tensor):
        return getattr(graph, mode)(input=tensor, name=mode)

    return apply


def _relu_reference(
    tensor: torch.Tensor,
    *,
    negative_slope: float | None = None,
    lower_clip: float | None = None,
    upper_clip: float | None = None,
) -> torch.Tensor:
    if negative_slope is None:
        lower = 0.0 if lower_clip is None else lower_clip
        result = torch.clamp_min(tensor, lower)
    else:
        result = torch.nn.functional.leaky_relu(tensor, negative_slope=negative_slope)
        if lower_clip is not None:
            result = torch.clamp_min(result, lower_clip)
    if upper_clip is not None:
        result = torch.clamp_max(result, upper_clip)
    return result


_PLAIN_EPILOGUES: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], str]] = {
    "abs": (torch.abs, "signed"),
    "ceil": (torch.ceil, "signed"),
    "cos": (torch.cos, "signed"),
    "elu": (torch.nn.functional.elu, "signed"),
    "erf": (torch.erf, "signed"),
    "exp": (torch.exp, "signed"),
    "floor": (torch.floor, "signed"),
    "gelu": (torch.nn.functional.gelu, "signed"),
    "gelu_approx_tanh": (lambda x: torch.nn.functional.gelu(x, approximate="tanh"), "signed"),
    "identity": (lambda x: x, "signed"),
    "log": (torch.log, "positive"),
    "logical_not": (lambda x: torch.logical_not(x).to(x.dtype), "logical"),
    "neg": (torch.neg, "signed"),
    "reciprocal": (torch.reciprocal, "positive"),
    "rsqrt": (torch.rsqrt, "positive"),
    "sigmoid": (torch.sigmoid, "signed"),
    "sin": (torch.sin, "signed"),
    "softplus": (torch.nn.functional.softplus, "signed"),
    "sqrt": (torch.sqrt, "positive"),
    "tan": (torch.tan, "signed"),
    "tanh": (torch.tanh, "signed"),
}

EPILOGUE_CASES = tuple(
    EpilogueCase(name, name, _graph_unary(name), reference_fn, input_profile) for name, (reference_fn, input_profile) in _PLAIN_EPILOGUES.items()
) + (
    EpilogueCase(
        "relu",
        "relu",
        lambda graph, x: graph.relu(input=x, name="relu"),
        _relu_reference,
    ),
    EpilogueCase(
        "relu-clipped",
        "relu",
        lambda graph, x: graph.relu(input=x, name="relu", lower_clip=-0.05, upper_clip=0.10),
        lambda x: _relu_reference(x, lower_clip=-0.05, upper_clip=0.10),
    ),
    EpilogueCase(
        "relu-leaky-clipped",
        "relu",
        lambda graph, x: graph.relu(input=x, name="relu", negative_slope=0.2, lower_clip=-0.05, upper_clip=0.10),
        lambda x: _relu_reference(x, negative_slope=0.2, lower_clip=-0.05, upper_clip=0.10),
    ),
    EpilogueCase(
        "leaky-relu",
        "leaky_relu",
        lambda graph, x: graph.leaky_relu(input=x, name="leaky_relu", negative_slope=0.2),
        lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.2),
    ),
    EpilogueCase(
        "swish",
        "swish",
        lambda graph, x: graph.swish(input=x, name="swish"),
        torch.nn.functional.silu,
    ),
    EpilogueCase(
        "swish-beta",
        "swish",
        lambda graph, x: graph.swish(input=x, name="swish", swish_beta=2.0),
        lambda x: x * torch.sigmoid(2.0 * x),
    ),
)


__all__ = ["EPILOGUE_CASES", "EpilogueCase"]
