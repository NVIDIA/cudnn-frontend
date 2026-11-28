# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine selection under the BATCH_INVARIANT numerical note."""

import pytest
import torch

import cudnn
from bi_rmsnorm_utils import adversarial, build, weight

pytestmark = pytest.mark.L0


def test_note_selects_the_engine(bi_device):
    x = adversarial(8, 1024, torch.bfloat16, 1, bi_device)
    w = weight(1024, torch.bfloat16, 2, bi_device)
    assert build(x, w)[0].selected_engine.name == "batch_invariant_rmsnorm"
    plain = build(x, w, note=False)[0]
    assert plain.selected_engine is None or plain.selected_engine.name != "batch_invariant_rmsnorm"


def _graph(x, w, *, bias=None, constant_eps=False, compute=cudnn.data_type.FLOAT):
    graph = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=compute)
    xt = graph.tensor_like(x)
    wt = graph.tensor_like(w.view(1, -1))
    bt = graph.tensor_like(bias.view(1, -1)) if bias is not None else None
    if constant_eps:
        et = graph.tensor_scalar(1e-6, cudnn.scalar_type.COMPILE_TIME_CONST).set_data_type(cudnn.data_type.FLOAT)
    else:
        et = graph.tensor_like(torch.full((1, 1), 1e-6, dtype=torch.float32), name="epsilon")
    y, _ = graph.rmsnorm(norm_forward_phase=cudnn.norm_forward_phase.INFERENCE, input=xt, scale=wt, bias=bt, epsilon=et)
    y.set_output(True).set_data_type(x.dtype)
    graph.select_numeric_notes([cudnn.numerical_note.BATCH_INVARIANT])
    return graph


@pytest.mark.parametrize("case", ["bias", "hidden-beyond-65536", "constant-epsilon", "fp64-compute"])
def test_unsupported_graph_raises_instead_of_falling_back(bi_device, case):
    hidden = 65544 if case == "hidden-beyond-65536" else 1024
    x = adversarial(8, hidden, torch.bfloat16, 3, bi_device)
    w = weight(hidden, torch.bfloat16, 4, bi_device)
    compute = cudnn.data_type.DOUBLE if case == "fp64-compute" else cudnn.data_type.FLOAT
    graph = _graph(x, w, bias=w.clone() if case == "bias" else None, constant_eps=case == "constant-epsilon", compute=compute)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        graph.build([cudnn.heur_mode.A])
