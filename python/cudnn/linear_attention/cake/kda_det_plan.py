# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk-64 KDA serial plans with full backward recomputation."""

from cudnn.frost import buffers
from cudnn.frost.device import current_device, device_context, multiprocessor_count
from cudnn.frost.workspace import WorkspaceLayout, carve_plan

from ..graph_analyzer import to_buffer_dtype
from .common import CakeLinearPlan, ffi_view


class KdaCakeDetPlan(CakeLinearPlan):
    plan_name = "kda_cake_det"

    def __init__(self, facts, node):
        from cudnn._cake import loader
        from cudnn._cake._vendor.cake_kda_det_train import cake_backend, cake_build, cake_jit

        self.device = current_device()
        self.is_bwd = facts.is_bwd
        self.scale = float(facts.scale) if facts.scale is not None else 128**-0.5
        self.geometry = {name: (tuple(t.dim), to_buffer_dtype(t.get_data_type())) for name, t in node.inputs.items()}
        for name in node.outputs:
            source = {"O": "v", "dQ": "q", "dK": "k", "dV": "v", "dG": "g", "dBeta": "beta"}[name]
            self.geometry[name] = self.geometry[source]
        self.outputs = set(node.outputs)
        self.plan = p = cake_backend.plan(facts.total_t, facts.n_seq, facts.h_q, multiprocessor_count(self.device), "sm_100a")
        if p.recur != "serial":
            raise NotImplementedError("kda_cake_det: requires the serial recurrence")
        self.path = "replay" if self.is_bwd else "fwd_nosave"
        mode = "replay" if self.is_bwd else "fwd"
        loader.install(cake_build)
        self.entries = {}
        self.arguments = {}
        self.grids = {}
        for stage in p.stages(self.path):
            name = p.record(stage, self.path)
            record = cake_jit.MODULES[name]
            self.entries[stage] = getattr(cake_jit.load(name, p.arch), record["ffi_entry"])
            self.arguments[stage] = cake_jit._argument_plan(name)
            self.grids[stage] = p.grid(stage, self.path)
        layout = WorkspaceLayout()
        layout.add(p.workspace_bytes(mode))
        regions = [(r.name, r.offset, str(r.dtype).removeprefix("torch."), r.shape) for r in p.layout(mode)]
        table_offset = layout.add(p.table_words() * 4)
        regions.extend((name, table_offset + off * 4, "int32", (words,)) for name, off, words in p.table_layout())
        self.workspace_size = layout.size
        self.region_names = tuple(r[0] for r in regions)
        self.carve = carve_plan(self.plan_name, [r[1:] for r in regions])

    def run(self, views, workspace, stream):
        import tvm_ffi

        x = dict(zip(self.names, views))
        self._validate(x, workspace)
        x = {name: ffi_view(value, self.device) for name, value in x.items()}
        values = dict(zip(self.region_names, (ffi_view(value, self.device) for value in workspace.carve(self.carve))))
        p = self.plan
        values.update(x)
        values.update(
            num_seqs=p.N,
            num_chunks_pad=p.NC_pad,
            num_heads=p.H,
            group=1,
            num_segments=0,
            segments=1,
            seg_arg=p.seg_arg,
            proportional=p.proportional,
            num_pairs=p.R // 128,
            scale=self.scale,
            write_tapes=int(self.is_bwd),
            write_ext=0,
            write_int=int(p.internal_rows(self.path)),
            write_prep=int(not self.is_bwd),
        )
        values.update(
            w_t=values["w"].reshape(p.NC_pad, p.H, 128, 64),
            w_tk=values["w"].reshape(p.NC_pad * 128, p.H, 64),
            h2d=values["h"].reshape(-1, 128),
            aqk_ext=values["akk_int"],
            akk_ext=values["akk_int"],
        )
        if self.is_bwd:
            values.update(
                dh2d=values["dh"].reshape(-1, 128),
                aqk_src=values["aqk_int"],
                akk_src=values["akk_int"],
                dout=x["dO"],
                dq=x["dQ"],
                dk=x["dK"],
                dv=x["dV"],
                dg=x["dG"],
                dbeta=x["dBeta"],
            )
            stages = ("table", "bwd_prep", "fwd_intra", "wy", "fwdh", "dav", "dhu", "dqkg", "glm_intra", "gate_epilogue")
        else:
            values["out"] = x["O"]
            stages = ("table", "fwd_intra", "wy_fwd", "fwdh")
        with device_context(self.device), tvm_ffi.use_raw_stream(tvm_ffi.device("cuda", self.device), int(stream or 0)):
            for stage in stages:
                if stage in self.entries:
                    grid = self.grids[stage]
                    args = [values[name] if name is not None else grid[axis] for name, axis in self.arguments[stage]]
                    self.entries[stage](*args)
