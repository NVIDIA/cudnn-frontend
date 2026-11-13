# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN 9.11 removed Maxwell, Pascal and Volta (compute capability below 7.5). A graph that targets one must
decline at plan creation and at check_support: the backend's heuristics still return plans for it that build and
only fail at execute (CUDNN_STATUS_EXECUTION_FAILED on a V100 with cuDNN 9.11.0, 9.11.1 and 9.27.0)."""

import cudnn
import pytest
import torch

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(cudnn.backend_version() < 91100, reason="cuDNN 9.10 and older still support compute capability below 7.5"),
]

_MESSAGE = "supports compute capability 7.5"


def _conv_graph(sm_version):
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        sm_version=sm_version,
    )
    x = graph.tensor(dim=[2, 8, 16, 16], stride=[2048, 1, 128, 8], data_type=cudnn.data_type.HALF)
    w = graph.tensor(dim=[16, 8, 3, 3], stride=[72, 1, 24, 8], data_type=cudnn.data_type.HALF)
    y = graph.conv_fprop(image=x, weight=w, padding=[1, 1], stride=[1, 1], dilation=[1, 1])
    y.set_output(True)
    graph.validate()
    graph.build_operation_graph()
    return graph


@pytest.mark.parametrize("sm_version", [52, 61, 70, 72])
def test_cudnn_911_declines_a_pre_turing_target(sm_version):
    graph = _conv_graph(sm_version)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match=_MESSAGE):
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match=_MESSAGE):
        graph.check_support()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to plan for")
def test_cudnn_911_keeps_a_turing_target():
    """SM75 is the oldest architecture cuDNN 9.11 supports: the gate leaves it alone."""
    graph = _conv_graph(75)
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    except cudnn.cudnnGraphNotSupportedError as error:  # heuristics for a foreign target may decline on their own
        assert _MESSAGE not in str(error)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (7, 5), reason="needs a GPU below compute capability 7.5")
def test_cudnn_911_declines_the_current_pre_turing_device():
    """Without sm_version the target is unset (-1); plan creation resolves it from the device before the gate, so a
    graph that goes straight to build_plans() is declined there too, not only by check_support()."""
    graph = _conv_graph(None)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError, match=_MESSAGE):
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
