# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-time cuDNN graph GEMMs for the native HCA execution experiment."""

import cudnn


class _HcaGemms:
    """Two compiled graph shapes, three launches, one shared caller workspace.

    The plan and its handle must not be executed concurrently from host threads.
    All three launches are ordered on the explicit stream.
    """

    def __init__(self, layout):
        self.handle = None
        self.handle = cudnn.create_handle()
        self.graphs = []
        g, m, k, d = layout.groups, layout.group_tokens * 128, layout.keys, 512
        try:
            for name, a_shape, a_stride, b_shape, b_stride, c_shape, c_dtype in (
                ("dq", (g, m, k), (m * k, k, 1), (g, k, d), (k * d, d, 1), (g, m, d), cudnn.data_type.BFLOAT16),
                ("dkdv", (g, k, m), (m * k, 1, k), (g, m, d), (m * d, d, 1), (g, k, d), cudnn.data_type.FLOAT),
            ):
                graph = cudnn.pygraph(
                    handle=self.handle,
                    io_data_type=cudnn.data_type.BFLOAT16,
                    intermediate_data_type=cudnn.data_type.FLOAT,
                    compute_data_type=cudnn.data_type.FLOAT,
                )
                a = graph.tensor(name="a", dim=a_shape, stride=a_stride, data_type=cudnn.data_type.BFLOAT16)
                b = graph.tensor(name="b", dim=b_shape, stride=b_stride, data_type=cudnn.data_type.BFLOAT16)
                c = graph.matmul(name=name, A=a, B=b, compute_data_type=cudnn.data_type.FLOAT)
                c.set_output(True).set_data_type(c_dtype).set_dim(c_shape).set_stride((c_shape[1] * d, d, 1))
                graph.validate()
                graph.build_operation_graph()
                graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
                graph.check_support()
                graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
                self.graphs.append((graph, a, b, c))
            self.workspace_size = max(graph.get_workspace_size() for graph, *_ in self.graphs)
        except Exception:
            self.close()
            raise

    def execute(self, tensors, workspace, stream):
        cudnn.set_stream(handle=self.handle, stream=stream.cuda_stream)
        for index, a_name, b_name, c_name in ((0, "ds", "packed_keys", "dq"), (1, "ds", "q", "dk"), (1, "p", "dout", "dv")):
            graph, a, b, c = self.graphs[index]
            graph.execute({a: tensors[a_name], b: tensors[b_name], c: tensors[c_name]}, workspace, handle=self.handle)

    def close(self):
        if self.handle is not None:
            cudnn.destroy_handle(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
