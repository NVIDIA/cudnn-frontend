# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt the public causal-convolution contract to private SE kernels."""


def causal_conv1d(x, weight, bias, activation, cu_seqlens, initial_states, return_final_states, final_states_out):
    from .packed import packed_causal_conv1d

    if cu_seqlens is None:
        if initial_states is not None or return_final_states or final_states_out is not None:
            raise NotImplementedError("dense SE causal_conv1d does not support initial/final history; use packed offsets")
        from .dense import dense_causal_conv1d

        return dense_causal_conv1d(x, weight, bias, activation)
    initial = None if initial_states is None else initial_states.transpose(1, 2)
    output, final = packed_causal_conv1d(
        x[0].transpose(0, 1),
        weight,
        cu_seqlens,
        bias,
        initial,
        silu=activation == "silu",
        return_final_states=return_final_states,
        _final_states_out_guard=final_states_out,
    )
    from cudnn.ops.causal_conv1d import _match_causal_conv1d_output_layout

    output = _match_causal_conv1d_output_layout(output.unsqueeze(0).transpose(1, 2), x)
    final = None if final is None else final.transpose(1, 2)
    if final_states_out is not None:
        final_states_out.copy_(final)
        final = final_states_out
    return (output, final) if return_final_states else output
