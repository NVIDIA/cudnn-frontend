# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cudnn


@dataclass(frozen=True)
class InputShape:
    """The input shape definition. Input X has shape (n, d, h, w, c), filter K has shape (k, t, r, s, c)."""

    n: int
    d: int
    h: int
    w: int
    c: int
    k: int
    t: int
    r: int
    s: int
    pre_padding: tuple[int, int, int] = (0, 0, 0)
    post_padding: tuple[int, int, int] = (0, 0, 0)
    stride: tuple[int, int, int] = (1, 1, 1)
    dilation: tuple[int, int, int] = (1, 1, 1)


def build_frost_conv_plans(g: cudnn.pygraph) -> None:
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    names = [g.get_plan_name_at_index(i) for i in range(len(g.plans))]
    g.select_plan(names.index("frost_conv"))
    g.check_support()
    g.build_plans()
