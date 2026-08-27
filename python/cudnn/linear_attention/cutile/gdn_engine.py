# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuTile GDN engine: GDN / GDN_BWD nodes on the chunked cuTile kernels
(``kernels/gdn``)."""

from typing import TYPE_CHECKING

from cudnn import behavior_note
from cudnn.engines.base import BaseEngine, CompiledPlan, bind_ports
from cudnn.graph_types import NodeType

from cudnn.frost import buffers
from cudnn.frost.workspace import Workspace, WorkspaceLayout, carve_plan
from ..graph_analyzer import analyze, to_buffer_dtype
from .engine import check_layouts_compact, cutile_la_gate, expect_table

if TYPE_CHECKING:
    from cudnn._pygraph import pygraph


class GdnCuTilePlan(CompiledPlan):
    """Carve plan over the caller's workspace, driven from the normalized
    variant pack: the carve layout, geometry, scale and kernel module are fixed
    per node at build; between executes only the buffer addresses move."""

    takes_variant_pack = True
    plan_name = "GdnCuTileEngine"

    def __init__(self, graph):
        from .kernels import common
        from .kernels import gdn as kernels

        (node,) = graph.nodes
        self.kernels = kernels
        self.common = common
        self.is_bwd = node.node_type == NodeType.GDN_BWD

        q, v, cu = (node.inputs[p] for p in ("q", "v", "cu_seqlens"))
        total, H, K = (int(d) for d in q.dim)
        HV, V = int(v.dim[1]), int(v.dim[2])
        N = int(cu.dim[0]) - 1
        io = to_buffer_dtype(q.get_data_type())
        f32 = "float32"
        isz = buffers.DTYPE_ITEMSIZE[io]
        BT = kernels.BT_CHUNK
        NT_bound = common.cdiv(total, BT) + N
        l2norm = bool(node.params.get("use_qk_l2norm", False))

        layout = WorkspaceLayout()
        regions = [
            ("chunk_table", layout.add(NT_bound * 2 * 4), "int32", (NT_bound, 2)),
            ("chunk_count", layout.add(4), "int32", (1,)),
            ("chunk_offsets", layout.add((N + 1) * 4), "int32", (N + 1,)),
            ("dummy", layout.add(16), "int32", (4,)),
            ("g_cum", layout.add(total * HV * 4), f32, (total, HV)),
            ("A", layout.add(total * HV * BT * isz), io, (total, HV, BT)),
            ("w", layout.add(total * HV * K * isz), io, (total, HV, K)),
            ("u", layout.add(total * HV * V * isz), io, (total, HV, V)),
            ("state_checkpoints", layout.add(NT_bound * HV * K * V * isz), io, (NT_bound, HV, V, K)),
            ("v_new", layout.add(total * HV * V * isz), io, (total, HV, V)),
        ]
        if l2norm:
            regions += [
                ("q_norm", layout.add(total * H * K * isz), io, (total * H, K)),
                ("q_rstd", layout.add(total * H * 4), f32, (total * H,)),
                ("k_norm", layout.add(total * H * K * isz), io, (total * H, K)),
                ("k_rstd", layout.add(total * H * 4), f32, (total * H,)),
            ]
        if self.is_bwd:
            NK = common.cdiv(K, min(max(common.next_power_of_2(K), 16), 64))
            regions += [
                ("dv", layout.add(total * HV * V * isz), io, (total, HV, V)),
                ("dstate", layout.add(NT_bound * HV * K * V * isz), io, (NT_bound, HV, V, K)),
                ("dv2", layout.add(total * HV * V * isz), io, (total, HV, V)),
                ("dg_nk", layout.add(NK * total * HV * 4), f32, (NK, total, HV)),
                ("dw", layout.add(total * HV * K * isz), io, (total, HV, K)),
            ]
            if HV != H or l2norm:
                regions += [
                    ("dq", layout.add(total * HV * K * isz), io, (total, HV, K)),
                    ("dk", layout.add(total * HV * K * isz), io, (total, HV, K)),
                ]
            if HV != H:
                regions.append(("wy_dk_hred", layout.add(total * H * K * isz), io, (total, H, K)))
                if l2norm:
                    regions += [
                        ("dq_hred", layout.add(total * H * K * isz), io, (total, H, K)),
                        ("dk_hred", layout.add(total * H * K * isz), io, (total, H, K)),
                    ]
            regions += [
                ("dg", layout.add(total * HV * 4), f32, (total, HV)),
                ("wy_dk", layout.add(total * HV * K * isz), io, (total, HV, K)),
                ("wy_dg", layout.add(total * HV * 4), f32, (total, HV)),
            ]

        self.workspace_size = layout.size
        self.carve_names = [name for name, _off, _dtype, _shape in regions]
        self.carve = carve_plan(self.plan_name, [(off, dtype, shape) for _name, off, dtype, shape in regions])
        self.expect = expect_table(node)
        self.n_seqs = N
        self.bound = NT_bound
        self.bt_chunk = BT
        self.scale = float(node.params.get("scale") or K**-0.5)
        self.l2norm = l2norm
        self.safe_gate = bool(node.params.get("safe_gate", False))
        self.want_state = "final_state" in node.outputs

        if self.is_bwd:
            plant = [("dq_l2", "dQ"), ("dk_l2", "dK")] if l2norm else [("dq" if HV == H else "dq_hred", "dQ"), ("dk" if HV == H else "dk_hred", "dK")]
            plant += [("wy_dv", "dV"), ("dg_cum", "dG"), ("db", "dBeta")]
            if "initial_state" in node.inputs:
                plant.append(("dstate0", "d_initial_state"))
        else:
            plant = [("o", "O")] + ([("final_state", "final_state")] if self.want_state else [])
        self.plant = tuple(plant)

        self.ports = None
        self.names = None
        self.indices = None

    def get_workspace_size(self) -> int:
        return self.workspace_size

    def execute(self, graph, variant_pack, ctx) -> None:
        if self.ports is None:
            self.ports = bind_ports(graph, variant_pack)
            (slots,) = self.ports.values()
            self.names = list(slots.inputs) + list(slots.outputs)
            self.indices = list(slots.inputs.values()) + list(slots.outputs.values())
        views = variant_pack.operands(self.indices)
        check_layouts_compact(self.plan_name, self.expect, self.names, views)
        nb = dict(zip(self.names, views))
        stream = ctx.stream if ctx.stream is not None else 0
        workspace = Workspace.over(variant_pack, self.workspace_size, self.plan_name)
        region = dict(zip(self.carve_names, workspace.carve(self.carve)))
        self.common.build_chunk_table(
            region["chunk_table"],
            region["chunk_count"],
            region["chunk_offsets"],
            nb["cu_seqlens"],
            self.n_seqs,
            self.bt_chunk,
            self.bound,
            stream=stream,
        )
        for name, port in self.plant:
            region[name] = nb[port]
        if self.is_bwd:
            self.execute_bwd(nb, region, stream)
        else:
            self.execute_fwd(nb, region, stream)

    def execute_fwd(self, nb, region, stream) -> None:
        gate = dict(use_gate_in_kernel=True, A_log=nb["a_log"], dt_bias=nb["dt_bias"]) if self.safe_gate else {}
        self.kernels.chunk_gated_delta_rule(
            nb["q"],
            nb["k"],
            nb["v"],
            nb["g"],
            nb["beta"],
            scale=self.scale,
            initial_state=nb.get("initial_state"),
            output_final_state=self.want_state,
            use_qk_l2norm_in_kernel=self.l2norm,
            cu_seqlens=nb["cu_seqlens"],
            chunk_indices=region["chunk_table"],
            bufs=region,
            state_v_first=True,
            stream=stream,
            **gate,
        )

    def execute_bwd(self, nb, region, stream) -> None:
        self.kernels.chunk_gated_delta_rule_grad(
            nb["q"],
            nb["k"],
            nb["v"],
            nb["g"],
            nb["beta"],
            nb["dO"],
            dstate_in=nb.get("d_final_state"),
            scale=self.scale,
            initial_state=nb.get("initial_state"),
            use_qk_l2norm_in_kernel=self.l2norm,
            cu_seqlens=nb["cu_seqlens"],
            chunk_indices=region["chunk_table"],
            bufs=region,
            state_v_first=True,
            stream=stream,
        )


class GdnCuTileEngine(BaseEngine):
    """cuTile chunked-kernel backend for single-node GDN graphs (THD layout)."""

    name = "gdn_cutile"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)

    def check_support(self, graph: "pygraph") -> None:
        import cudnn

        try:
            from .kernels.gdn import chunk_gated_delta_rule  # noqa: F401 — availability probe: ImportError = decline
        except ImportError as exc:
            raise NotImplementedError(f"GdnCuTileEngine requires the cuda.tile runtime: {exc}") from exc

        facts = graph._facts_for(analyze)
        cutile_la_gate("GdnCuTileEngine", facts, "GDN", cudnn.data_type.FLOAT)
        if facts.is_bwd and facts.safe_gate:
            raise NotImplementedError("GdnCuTileEngine: safe_gate is forward-only")
        if facts.use_beta_sigmoid:
            raise NotImplementedError("GdnCuTileEngine: use_beta_sigmoid has no cuTile path (the FROST GDN engine serves it)")

    def build_plan(self, graph, plan, ctx=None) -> CompiledPlan:
        return GdnCuTilePlan(graph)
