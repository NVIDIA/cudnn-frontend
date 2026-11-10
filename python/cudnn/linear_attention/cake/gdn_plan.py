# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared serial GDN chain. Backward recomputes its forward intermediates."""

from __future__ import annotations

import math

from cudnn.frost import buffers
from cudnn.frost.device import current_device, device_context
from cudnn.frost.workspace import WorkspaceLayout, carve_plan

from ..graph_analyzer import to_buffer_dtype
from .common import CakeLinearPlan, ffi_view


class GdnCakePlan(CakeLinearPlan):
    plan_name = "gdn_cake_det"

    def __init__(self, facts, node):
        from cudnn._cake import loader
        from cudnn._cake._vendor.cake_gdn_det_train import cake_build, cake_jit

        self.device = current_device()
        self.is_bwd = facts.is_bwd
        self.t, self.h, self.hv, self.n = facts.total_t, facts.h_q, facts.h_v, facts.n_seq
        self.nc = self.t // 64 + self.n
        self.scale = float(facts.scale) if facts.scale is not None else 128**-0.5
        self.normalize = int(facts.use_qk_l2norm)
        self.geometry = {name: (tuple(t.dim), to_buffer_dtype(t.get_data_type())) for name, t in node.inputs.items()}
        for name in node.outputs:
            source = {"O": "v", "dQ": "q", "dK": "k", "dV": "v", "dG": "g", "dBeta": "beta"}[name]
            self.geometry[name] = self.geometry[source]
        self.outputs = set(node.outputs)
        loader.install(cake_build)
        stages = ["meta", "prep", "wy_tc2", "fwd_ho"]
        if self.is_bwd:
            stages += ["dhu_tc2", "dqkwg", "wy_bwd", "finalize"]
        self.entries = {}
        for stage in stages:
            record = cake_jit.MODULES[cake_jit.stage_record(stage, "sm_100a")]
            module = cake_build.load_library(record["library"] + "_sm_100a")
            self.entries[stage] = getattr(module, record["ffi_entry"])
        layout, regions = WorkspaceLayout(), []

        def reserve(name, shape, dtype):
            offset = layout.add(math.prod(shape) * buffers.DTYPE_ITEMSIZE[dtype])
            regions.append((name, offset, dtype, shape))

        for name, count in (
            ("cs", self.nc),
            ("cl", self.nc),
            ("sc", self.n + 1),
            ("fs", self.n + 1),
            ("fi", self.n),
            ("fx", self.n + 1),
            ("bs", self.n + 1),
            ("bi", self.n),
            ("bx", self.n + 1),
        ):
            reserve(name, (count,), "int32")
        for name in ("qn", "kn"):
            reserve(name, (self.t, self.h, 128), "bfloat16")
        for name in ("rq", "rk"):
            reserve(name, (self.t, self.h), "float32")
        reserve("gc", (self.t, self.hv), "float32")
        reserve("a", (self.t, self.hv, 64), "bfloat16")
        for name in ("w", "u", "vn"):
            reserve(name, (self.t, self.hv, 128), "bfloat16")
        reserve("state", (self.nc, self.hv, 128, 128), "bfloat16")
        reserve("dead", (1,), "float32")
        if self.is_bwd:
            for name in ("o", "dv2", "dqh", "dkh", "dw", "dk2"):
                reserve(name, (self.t, self.hv, 128), "bfloat16")
            reserve("dh", (self.nc, self.hv, 128, 128), "bfloat16")
            for name in ("dg1", "dg2"):
                reserve(name, (self.t, self.hv), "float32")
        self.workspace_size = layout.size
        self.region_names = tuple(r[0] for r in regions)
        self.carve = carve_plan(self.plan_name, [r[1:] for r in regions])

    def run(self, views, workspace, stream):
        import tvm_ffi

        x = dict(zip(self.names, views))
        self._validate(x, workspace)

        x = {name: ffi_view(value, self.device) for name, value in x.items()}
        s = dict(zip(self.region_names, (ffi_view(value, self.device) for value in workspace.carve(self.carve))))
        h, hv, nc, n = self.h, self.hv, self.nc, self.n

        def launch(stage, grid, *args):
            self.entries[stage](*args, *grid)

        with device_context(self.device), tvm_ffi.use_raw_stream(tvm_ffi.device("cuda", self.device), int(stream or 0)):
            launch(
                "meta", (1, 1, 1), x["cu_seqlens"], s["cs"], s["cl"], s["sc"], s["fs"], s["fi"], s["fx"], s["bs"], s["bi"], s["bx"], n, nc, self.t, 0, 0, n, n
            )
            launch(
                "prep",
                (nc, hv, 1),
                x["q"],
                x["k"],
                x["g"],
                x["beta"],
                x["beta"],
                s["qn"],
                s["kn"],
                s["rq"],
                s["rk"],
                s["gc"],
                s["cs"],
                s["cl"],
                h,
                hv,
                self.normalize,
            )
            launch("wy_tc2", (nc, hv, 1), s["kn"], x["v"], s["gc"], x["beta"], s["a"], s["w"], s["u"], s["cs"], s["cl"], h, hv, 0)
            out = s["o"] if self.is_bwd else x["O"]
            # Serial metadata writes fi[i] = i, so fi is also the scan's identity walk.
            launch(
                "fwd_ho",
                (n * hv, 1, 1),
                s["qn"],
                s["kn"],
                s["w"],
                s["u"],
                s["gc"],
                s["dead"],
                s["dead"],
                s["state"],
                s["vn"],
                s["vn"],
                out,
                out,
                s["dead"],
                s["cs"],
                s["cl"],
                s["fs"],
                s["fi"],
                s["sc"],
                s["fi"],
                h,
                hv,
                0,
                0,
                0,
                self.scale,
            )
            if not self.is_bwd:
                return
            launch(
                "dhu_tc2",
                (n, hv, 1),
                s["qn"],
                s["kn"],
                s["w"],
                x["dO"],
                s["gc"],
                s["dead"],
                s["dh"],
                s["dv2"],
                s["dead"],
                s["dead"],
                s["cs"],
                s["cl"],
                s["bs"],
                s["bi"],
                s["sc"],
                h,
                hv,
                0,
                0,
                self.scale,
            )
            launch(
                "dqkwg",
                (nc, hv, 1),
                s["qn"],
                s["kn"],
                s["vn"],
                x["dO"],
                s["dv2"],
                s["state"],
                s["dh"],
                s["gc"],
                s["dqh"],
                s["dkh"],
                s["dw"],
                s["dg1"],
                s["cs"],
                s["cl"],
                h,
                hv,
                self.scale,
            )
            launch(
                "wy_bwd",
                (nc, hv, 1),
                s["kn"],
                x["v"],
                s["a"],
                s["dw"],
                s["dv2"],
                s["gc"],
                x["beta"],
                s["dk2"],
                x["dV"],
                x["dBeta"],
                s["dg2"],
                s["cs"],
                s["cl"],
                h,
                hv,
            )
            launch(
                "finalize",
                (nc, h, 1),
                s["qn"],
                s["kn"],
                s["rq"],
                s["rk"],
                s["dqh"],
                s["dkh"],
                s["dk2"],
                s["dg1"],
                s["dg2"],
                x["dQ"],
                x["dK"],
                x["dG"],
                s["cs"],
                s["cl"],
                h,
                hv,
                self.normalize,
            )
