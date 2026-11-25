# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the existing prefill softmax helper over an ordered symbolic algebra.

This is a deliberately small, fail-closed interpreter for the source operations
used here. TMEM supplies symbolic QK accumulators and collects P/alpha stores;
barrier operations are recorded, not proved. No production source is changed.
The independent specification is in the adjacent PrefillModel.lean.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace as NS

PROOFS = Path(__file__).resolve().parent
ROOT = PROOFS.parents[7]
KERNEL = (PROOFS.parent / "kernel.py").relative_to(ROOT).as_posix()
POINTWISE = "python/cudnn/frost/tile_dsl/pointwise.py"
REGTILE = "python/cudnn/frost/tile_dsl/regtile.py"
COMMON = "python/cudnn/sdpa/fwd/kernels/_common_blackwell.py"
MASK = "python/cudnn/frost/tile_dsl/mask.py"
CONSTANTS = "python/cudnn/frost/tile_dsl/constants.py"
CONFIG = "python/cudnn/sdpa/fwd/config_sm100.py"
CONTRACT = (PROOFS.parent / "contract.py").relative_to(ROOT).as_posix()
SOURCES = (KERNEL, POINTWISE, REGTILE, COMMON, MASK, CONSTANTS, CONFIG, CONTRACT)


class EraseIRWrappers(ast.NodeTransformer):
    """Host interpretation only; concrete geometry does not model int32 overflow."""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "ir_value" and not node.args and not node.keywords:
            return node.func.value
        return node


def extract_assignments(root, path, names, env):
    """Evaluate selected top-level assignments in source order, without imports.

    The exact admitted flags are checked below. Missing, duplicate, or newly
    dependent expressions fail; we never silently set an unknown gate to False.
    """
    remaining = set(names)
    for node in ast.parse((root / path).read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        selected = [target.id for target in targets if isinstance(target, ast.Name) and target.id in names]
        if not selected:
            continue
        if len(targets) != 1 or selected[0] not in remaining:
            raise ValueError(f"Ambiguous profile assignment in {path}")
        name = selected[0]
        env[name] = eval(compile(ast.Expression(node.value), str(root / path), "eval"), env)
        remaining.remove(name)
    if remaining:
        raise ValueError(f"Missing profile assignments in {path}: {sorted(remaining)}")


def mask_inline_ptx(code, *, write_only_types, read_only_args):
    # This is an explicitly trusted, bounded PTX primitive interpreter, not
    # a verified PTX semantics. Reject any spelling we have not implemented.
    common = "{ .reg .b32 n, ones; mov.b32 ones, 0xFFFFFFFF; max.s32 n, {$r0}, 0; "
    if write_only_types != [uint32] or len(read_only_args) != 1 or not isinstance(read_only_args[0], int):
        raise ValueError("Unexpected mask PTX operands")
    shift = max(read_only_args[0], 0)
    if code == common + "shr.u32 {$w0}, ones, n; }":
        return 0 if shift >= 32 else 0xFFFFFFFF >> shift
    if code == common + "shl.b32 {$w0}, ones, n; }":
        return 0 if shift >= 32 else uint32(0xFFFFFFFF << shift)
    raise ValueError(f"Unmodelled inline PTX: {code}")


def uint32(value):
    return int(value) & 0xFFFFFFFF


class Graph:
    def __init__(self):
        self.nodes = []
        self.index = {}

    def node(self, kind, *args):
        key = (kind, *args)
        if key not in self.index:
            self.index[key] = len(self.nodes)
            self.nodes.append(key)
        return Ref(self, self.index[key])

    def literal(self, value):
        return self.node("lit", struct.pack(">f", value).hex())

    def f32(self, value):
        if isinstance(value, Ref):
            if value.graph is not self:
                raise ValueError("Mixed symbolic graphs")
            return value
        return self.literal(float(value))

    def unary(self, op, x):
        if isinstance(x, Vector):
            return x.map(lambda e: self.unary(op, e))
        return self.node("unary", op, self.f32(x).index)

    def binary(self, op, a, b):
        return self.node("binary", op, self.f32(a).index, self.f32(b).index)

    def select(self, pred, yes, no):
        if isinstance(pred, bool):
            return yes if pred else no
        return self.node("ternary", "select", pred.index, self.f32(yes).index, self.f32(no).index)


class Ref:
    def __init__(self, graph, index):
        self.graph, self.index = graph, index

    def ir_value(self):
        return self

    def __bool__(self):
        raise TypeError("A symbolic float predicate cannot become a host branch")

    def __add__(self, other):
        return self.graph.binary("add", self, other)

    def __sub__(self, other):
        return self.graph.binary("sub", self, other)

    def __mul__(self, other):
        return self.graph.binary("mul", self, other)

    def __gt__(self, other):
        return self.graph.binary("gt", self, other)

    def __eq__(self, other):
        return self.graph.binary("eq", self, other)


class Vector:
    def __init__(self, values, dtype=None, shape=None):
        self.values = tuple(values.values if isinstance(values, Vector) else values)
        self.dtype = dtype
        self.shape = tuple(shape or (len(self.values),))
        if math.prod(self.shape) != len(self.values):
            raise ValueError("Invalid symbolic vector shape")
        self.type = NS(shape=self.shape)

    @staticmethod
    def from_elements(values, dtype):
        return Vector(values, dtype)

    def ir_value(self):
        return self

    def __getitem__(self, i):
        if len(self.shape) != 1:
            raise ValueError("Only 1-D scalar indexing is supported")
        return self.values[i]

    def map(self, fn, dtype=None):
        return Vector([fn(x) for x in self.values], self.dtype if dtype is None else dtype, self.shape)

    def to(self, dtype):
        if dtype == "bf16":
            return self.map(lambda x: x.graph.unary("cvt_bf16", x), dtype)
        raise ValueError(f"Unsupported cast {dtype}")

    def zip(self, other, op):
        rhs = other.values if isinstance(other, Vector) else [other] * len(self.values)
        if len(rhs) != len(self.values):
            raise ValueError("Vector width mismatch")
        return Vector([op(a, b) for a, b in zip(self.values, rhs)], self.dtype, self.shape)

    def __add__(self, other):
        return self.zip(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self.zip(other, lambda a, b: a * b)

    def __sub__(self, other):
        return self.zip(other, lambda a, b: a - b)


class VectorType:
    def __new__(cls, ty):
        return ty

    @staticmethod
    def get(shape, dtype):
        return tuple(shape)


class SourceImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module != "cudnn.frost.tile_dsl.regtile" or [x.name for x in node.names] != ["vec_concat"]:
            raise ValueError(f"Unmodelled source import: {ast.unparse(node)}")
        # The very same function is already extracted into this namespace.
        return ast.copy_location(ast.Pass(), node)


def extract(root, path, name, env, *, cls=False):
    tree = ast.parse((root / path).read_text())
    kind = ast.ClassDef if cls else ast.FunctionDef
    nodes = [n for n in tree.body if isinstance(n, kind) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected unique {name} in {path}")
    node = nodes[0]
    if not cls:
        node.decorator_list = []
        node.returns = None
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            arg.annotation = None
    node = SourceImports().visit(EraseIRWrappers().visit(node))
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(root / path), "exec"), env)
    return env[name]


def vector_environment(graph):
    f32 = graph.f32

    def shape_cast(shape, vec):
        return Vector(vec, vec.dtype, shape)

    def extract_vector(vec, *, dynamic_position, static_position):
        if dynamic_position or len(static_position) != 1 or len(vec.shape) != 2:
            raise ValueError("Unsupported extract")
        lo = static_position[0] * vec.shape[1]
        return Vector(vec.values[lo : lo + vec.shape[1]], vec.dtype)

    def extract_slice(shape, vec, *, offsets, sizes, strides):
        if len(offsets) != 1 or strides != [1] or tuple(sizes) != shape:
            raise ValueError("Unsupported vector slice")
        return Vector(vec.values[offsets[0] : offsets[0] + sizes[0]], vec.dtype)

    def insert_slice(src, dest, *, offsets, strides):
        if len(offsets) != 1 or strides != [1]:
            raise ValueError("Unsupported vector insertion")
        out = list(dest.values)
        out[offsets[0] : offsets[0] + len(src.values)] = src.values
        return Vector(out, dest.dtype, dest.shape)

    return dict(
        cutlass=NS(Float32=f32, Int32=int, Uint32=uint32, Boolean=bool, Float16="f16", BFloat16="bf16", Vector=Vector, const_expr=bool),
        cute=NS(
            math=NS(
                max=lambda a, b, **kw: graph.binary("max_ftz" if kw.get("ftz", False) else "max", a, b),
                min=lambda a, b: graph.binary("min", a, b),
                exp2=lambda x, *, fastmath: graph.unary("exp2_fast" if fastmath else "exp2", x),
            )
        ),
        arith=NS(select=graph.select, addf=lambda a, b: a + b),
        _ir=NS(VectorType=VectorType),
        T_=NS(f32=lambda: "f32"),
        vector=NS(
            shape_cast=shape_cast,
            extract=extract_vector,
            extract_strided_slice=extract_slice,
            insert_strided_slice=insert_slice,
            broadcast=lambda shape, x: Vector([x] * math.prod(shape), f32, shape),
        ),
    )


class Barrier:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def __getitem__(self, index):
        return Barrier(f"{self.name}[{index}]", self.events)

    def wait(self, phase, *, spin=False):
        self.events.append([self.name, "wait", phase, spin])

    def arrive(self, **kwargs):
        self.events.append([self.name, "arrive", kwargs])


def trace_softmax(root, *, sub_tile=0, limit=None):
    graph = Graph()
    env = vector_environment(graph)
    events, stores = [], {}
    for name in ("_ir_elem_ty", "_zero_scalar", "_vec_width", "vec_slice", "vec_concat"):
        extract(root, REGTILE, name, env)
    extract(root, REGTILE, "RegTile", env, cls=True)
    for name in ("row_max_reduction", "row_reduction_pair"):
        extract(root, POINTWISE, name, env)
    extract(root, COMMON, "row_max_for_exp2", env)
    extract_assignments(root, CONSTANTS, ("MASK_NONE", "MASK_PADDED", "MASK_CAUSAL", "MASK_SWA"), env)
    extract_assignments(root, MASK, ("_NEG_INF_BITS", "MASK_WORD_COLS", "MASK_BOUND_LIMIT"), env)
    env["inline_ptx"] = mask_inline_ptx
    for name in ("keep_below_word", "keep_from_word", "band_mask_words", "apply_mask_words", "apply_mask_chunk"):
        extract(root, MASK, name, env)

    layout = NS(S0_OFF=0, S1_OFF=256, P0_OFF=512, P1_OFF=768, STATS_OFF=1024, STATS_STRIDE=16)
    sbase = layout.S0_OFF if sub_tile == 0 else layout.S1_OFF
    pbase = layout.P0_OFF if sub_tile == 0 else layout.P1_OFF
    stat = layout.STATS_OFF + sub_tile * layout.STATS_STRIDE

    def load(shape, pointer, *, num):
        if shape != "32x32b" or pointer not in (sbase, sbase + 64) or num != 64:
            raise ValueError("Unexpected symbolic TMEM read")
        events.append(["tmem_load", pointer, num])
        return Vector([graph.node("var", 16 + pointer - sbase + k) for k in range(num)], env["cutlass"].Float32)

    def store(shape, pointer, values):
        if shape != "32x32b" or pointer not in (pbase, pbase + 32, stat) or pointer in stores:
            raise ValueError("Unexpected/duplicate symbolic TMEM write")
        stores[pointer] = values
        events.append(["tmem_store", pointer, len(values.values)])

    env.update(
        CFG=NS(
            N_BMM2_CHUNKS=2,
            TILE_N=128,
            RESCALE_THRESHOLD=8.0,
            BOTTOM_RIGHT=0,
            WINDOW_LEFT=0,
            MASK_FLAGS=env["MASK_CAUSAL"],
            WINDOW_RIGHT=0,
            CTA_MMA=2,
            TILE_M=128,
            TILES_Q=2,
            S_SLOTS=1,
            Q_IN_TMEM=0,
            THD_VARLEN=False,
            PAGED_KV=False,
            DTYPE_QKV=2,
            SPLIT_KV=1,
            BIAS_MODE=0,
            KV_MASK_FORM=0,
            PACK_GQA=False,
            PACK_G=1,
            PADDED_TAIL_FORM=0,
            SOFTMAX_WG_WARPS=4,
        ),
        PARAMS=NS(d_flavor=128, fused_ldtm_stat=0, exp2_fma_split=True),
        LAYOUT=layout,
        IS_TF32=False,
        STORAGE_DTYPE="bf16",
        named_barrier_fence=lambda barrier_id, threads: events.append(["publish_fence", barrier_id, threads]),
        nvvm=NS(
            make_tmem_ptr=lambda addr, dtype: addr,
            tcgen05_ld=load,
            tcgen05_st=store,
            tcgen05_wait=lambda **kw: events.append(["tmem_wait", kw]),
            barrier_cta_sync=lambda **kw: events.append(["named_barrier", kw]),
            Tcgen05Wait=NS(STORE="store"),
        ),
    )
    extract(root, CONFIG, "sm100_spin_ring_waits", env)
    flag_names = (
        "FUSED_LDTM_STAT",
        "SPIN_RING_WAITS",
        "_NB_FENCE_ALPHA",
        "_NB_FENCE_P0",
        "_E2E_DENSE_BAND",
        "_E2E_PAGED_INTERIOR",
        "_E2E_THD_INTERIOR",
        "_E2E_D64_INTERIOR",
        "_E2E_ENABLED",
        "_D64_EARLY_SOFTMAX_HANDOFF",
        "_D64_JOINT_ROW_MAX",
        "_D128_PAGED_JOINT_ROW_MAX",
        "_PADDED_TAIL_PREMASK",
    )
    extract_assignments(root, KERNEL, flag_names, env)
    for name in flag_names:
        if name not in ("SPIN_RING_WAITS", "_NB_FENCE_ALPHA", "_NB_FENCE_P0") and env[name] != 0:
            raise ValueError(f"Unqualified numerical specialization: {name}={env[name]}")
    fn = extract(root, KERNEL, "_softmax_kv_body", env)
    bars = NS(**{name: Barrier(name, events) for name in ("mb_bmm1_done", "mb_stat_full", "mb_bmm2_ready", "mb_stat_empty")})
    # A negative limit is the second KV tile of causal query 0: all cells dead.
    kv_loop, q_abs = (1, 0) if limit is not None and limit < 0 else (0, 127 if limit is None else limit)
    true_max_input = graph.node("var", 5)
    maximum, safe, sums, phase, empty_phase, true_max_output = fn(
        limit is not None,
        sub_tile,
        kv_loop,
        NS(load=lambda: 0),
        bars,
        q_abs,
        1024,
        1024,
        graph.node("var", 4),
        graph.node("var", 0),
        graph.node("var", 1),
        Vector([graph.node("var", 2), graph.node("var", 3)], env["cutlass"].Float32),
        0,
        1,
        0,
        true_max_input,
        track_true_max=False,
        tail_premask=None,
    )
    assert phase == 1 and empty_phase == 0
    assert true_max_output.index == true_max_input.index
    alpha = stores[stat].values[0]
    prob = stores[pbase].values + stores[pbase + 32].values
    assert len(prob) == 128
    fields = dict(maximum=maximum.index, safeMaximum=safe.index, evenSum=sums[0].index, oddSum=sums[1].index, alpha=alpha.index)
    return graph, fields, [x.index for x in prob], events, {name: env[name] for name in flag_names}


def lean_definition(name, graph, fields, prob):
    lines = [f"def {name} : SoftmaxStep :="]
    # Masked input nodes remain in the recorded trace but need not be emitted
    # as unused Lean lets. This only drops nodes unreachable from the outputs.
    live = set(fields.values()) | set(prob)
    for i in range(len(graph.nodes) - 1, -1, -1):
        if i in live and graph.nodes[i][0] not in ("var", "lit"):
            live.update(graph.nodes[i][2:])
    for i, node in enumerate(graph.nodes):
        if i not in live:
            continue
        kind, *args = node
        if kind == "var":
            expr = f".var {args[0]}"
        elif kind == "lit":
            expr = f".lit {json.dumps(args[0])}"
        else:
            expr = f".{kind} {json.dumps(args[0])} " + " ".join(f"n{j}" for j in args[1:])
        lines.append(f"  let n{i} : Expr := {expr}")
    lines.append("  { " + "\n    ".join(f"{k} := n{v}" for k, v in fields.items()))
    lines.append("    probability := [" + ", ".join(f"n{i}" for i in prob) + "] }")
    return "\n".join(lines)


def emit(root, output, *, case_names=None):
    identity = {}
    extract_assignments(root, CONTRACT, ("NAME", "TEMPLATE_NAME", "KERNEL_FILE"), identity)
    if (
        identity["NAME"] != "sdpa_fwd_sm100_d128_multi_q"
        or identity["TEMPLATE_NAME"] != "prefill_d128_f16"
        or identity["KERNEL_FILE"] != "sm100/d128_f16_multi_q/kernel.py"
    ):
        raise ValueError("Proof owner no longer matches the implementation contract")
    header = (PROOFS / "PrefillModel.lean").read_text()
    lines = [header, "namespace FrostPrefill", "set_option maxRecDepth 100000", "set_option maxHeartbeats 10000000"]
    cases = []
    for name, limit in (("unmasked", None), ("causalFirst", 0), ("causalHalf", 63), ("causalFull", 127), ("causalDead", -1)):
        if case_names is not None and name not in case_names:
            continue
        graph, fields, prob, events, flags = trace_softmax(root, limit=limit)
        other, other_fields, other_prob, _, other_flags = trace_softmax(root, sub_tile=1, limit=limit)
        # Both sub-tile owners must produce the same numerical trace despite addresses/barriers differing.
        assert graph.nodes == other.nodes and fields == other_fields and prob == other_prob and flags == other_flags
        lines.append(lean_definition(name, graph, fields, prob))
        visible = "fun _ => true" if limit is None else "fun _ => false" if limit < 0 else f"fun k => decide (k ≤ {limit})"
        lines.append(f"theorem {name}_matches_source_model : {name} = softmaxModel (scoreInputs ({visible})) := by decide")
        lines.append(f"#print axioms {name}_matches_source_model")
        cases.append(dict(name=name, limit=limit, nodes=len(graph.nodes), recorded_events=len(events), subtiles_checked=2, resolved_source_flags=flags))
    if not cases or (case_names is not None and set(case_names) != {case["name"] for case in cases}):
        raise ValueError("Empty or unknown proof cases")
    lines.append("end FrostPrefill")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(lines) + "\n")
    return dict(
        scope="F06 SM100 D128 BF16 dense top-left-causal plain softmax helper; 128 columns, CTA2, threshold8, no score outputs",
        owner=dict(engine_name="sdpa_fwd_sm100_d128_multi_q", engine_id=20535, catalog="F06", kernel=KERNEL, template="prefill_d128_f16"),
        profile=json.loads((PROOFS / "scope.json").read_text()),
        cases=cases,
        source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in SOURCES},
        bridge_sha256={
            (PROOFS / name).relative_to(ROOT).as_posix(): hashlib.sha256((PROOFS / name).read_bytes()).hexdigest()
            for name in ("trace_softmax.py", "check.py", "PrefillModel.lean", "scope.json")
        },
        generated_lean_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        gpu_checked=False,
        compiler_and_memory_verified=False,
        lean_checked=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--lean", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = emit(args.source_root, args.lean)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
