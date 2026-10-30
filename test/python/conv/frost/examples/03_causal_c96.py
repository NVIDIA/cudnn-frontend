# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""C96 causal Conv3D through the existing graph API, with caller-owned history.

Run with CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 on SM100 and a compatible
CuTe DSL installation (see the dense FROST runtime version gate).
"""

import argparse

import cudnn
import torch

from common import build_frost_conv_plans


def compact_channels_last(tensor):
    # Explicit allocation also canonicalizes strides when D or N is one.
    result = torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype, memory_format=torch.channels_last_3d)
    return result.copy_(tensor)


def run(output_channels, provider="frost", tiled_layout=False):
    if tiled_layout:
        from causal_c96_layout import pack, unpack_bias

    torch.manual_seed(96)
    video = torch.randn((1, 96, 9, 7, 9), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((output_channels, 96, 3, 3, 3), device="cuda", dtype=torch.bfloat16) * 0.1
    bias = torch.randn((1, output_channels, 1, 1, 1), device="cuda", dtype=torch.bfloat16)
    packed_weight = compact_channels_last(weight)  # One-time weight conversion.
    plans = {}
    history = None
    outputs = []
    start = 0
    for length in (1, 4, 4):
        current = video[:, :, start : start + length]
        h = 0 if history is None else history.shape[2]
        # In the 1+4+4 schedule, each chunk contains the full next history.
        next_history = current[:, :, -2:].clone()
        if tiled_layout:
            image = pack(current, history)
        else:
            assembled = current if history is None else torch.cat((history, current), dim=2)
            image = compact_channels_last(assembled)
        key = (tuple(image.shape), h)
        if key not in plans:
            result = torch.empty((1, output_channels, length, 7, 9), device="cuda", dtype=torch.bfloat16, memory_format=torch.channels_last_3d)
            graph = cudnn.pygraph(
                io_data_type=cudnn.data_type.BFLOAT16,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            x, w = graph.tensor_like(image), graph.tensor_like(packed_weight)
            y = graph.conv_fprop(x, w, pre_padding=(2 - h, 1, 1), post_padding=(0, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1))
            y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim(result.shape).set_stride(result.stride())
            if provider == "frost":
                build_frost_conv_plans(graph)
            else:
                graph.validate()
                graph.build_operation_graph()
                graph.create_execution_plans([cudnn.heur_mode.A])
                names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
                graph.select_plan(next(i for i, name in enumerate(names) if name != "frost_conv"))
                graph.check_support()
                graph.build_plans()
            workspace = torch.empty(max(1, graph.get_workspace_size()), device="cuda", dtype=torch.uint8)
            plans[key] = (graph, (x, w, y), result, workspace)
        graph, (x, w, y), result, workspace = plans[key]
        graph.execute({x: image, w: packed_weight, y: result}, workspace)
        # The convolution has already rounded to BF16. Add bias afterward.
        outputs.append(unpack_bias(result, bias.view(-1)) if tiled_layout else result.add_(bias).contiguous())
        history = next_history
        start += length
    actual = torch.cat(outputs, dim=2)
    with torch.backends.cudnn.flags(allow_tf32=False):
        padded = torch.nn.functional.pad(video.float(), (1, 1, 1, 1, 2, 0))
        reference = torch.nn.functional.conv3d(padded, weight.float()).to(torch.bfloat16).add_(bias)
    torch.testing.assert_close(actual.float(), reference.float(), atol=2 / 128, rtol=2 / 128)
    torch.testing.assert_close(history, video[:, :, -2:], atol=0, rtol=0)
    torch.cuda.synchronize()
    print(f"PASS C96 -> K{output_channels}: {provider}, tiled_layout={tiled_layout}, chunks 1+4+4, two-frame history")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("frost", "cudnn"), default="frost")
    parser.add_argument("--tiled-layout", action="store_true", help="use the optional Triton caller conversions")
    args = parser.parse_args()
    with torch.inference_mode():
        for output_channels in (96, 192):
            run(output_channels, args.provider, args.tiled_layout)
