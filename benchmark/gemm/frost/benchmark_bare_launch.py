# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How little host time can one FROST gemm launch take? Run it and see.

    CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 python benchmark/gemm/frost/benchmark_bare_launch.py

`graph.execute()` spends ~20 us of host time on a bf16 gemm whose parameters are
three addresses. Almost none of that is the launch: it is re-deriving, per call,
facts that were settled when the plan was built -- what shape each buffer is,
which axis is innermost, how to spell a CUtensorMap, which kernel to run.

This walks that back to the driver call, one layer at a time, and checks the
result is bit-identical to `graph.execute()` at every step. Nothing here is
harvested from a captured launch: the parameter block is built from the
`SLOT_TABLE` codegen emits, the kernel comes out of the compiled cubin, and the
launch geometry is the generated module's own closed form.

THE CONTRACT
------------
Between `build_plans()` and any launch, for every bound tensor: dtype, rank,
extents, strides, the innermost axis and the base alignment are what the graph
declared, and the buffer is on the plan's device. Only the ADDRESSES may change.

That is not checked anywhere below -- checking it is most of the 20 us. It is a
promise the caller keeps, and one an inference server keeps trivially: fixed
weights, fixed head dims, a pool of same-shaped activations. `--vary-m` shows
the next rung, where the token count moves too.

SCOPE
-----
One narrow case: a dense bf16 matmul with an STG epilogue, on SM100. The design
generalises -- codegen classifies every parameter of every gemm flavor it can
emit, and refuses (`kind == 'unknown'`) any it cannot -- but this file
deliberately demonstrates one, so that what it claims can be checked in a
minute. See `docs/frost_bare_launch.md`.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import os
import re
import sys
import tempfile
import time

# Artifacts are only kept when asked, and the ask has to precede the compile.
# DUMP_DIR defaults to the working directory, which would leave a cubin behind.
os.environ.setdefault("CUTE_DSL_KEEP", "cubin")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", tempfile.mkdtemp(prefix="frost_bare_launch_"))
os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

import torch
from cuda.bindings import driver as cd

import cudnn
from cudnn.engines import is_python_engine

BF16, F32 = cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT

# CUtensorMapDataType / CUtensorMapSwizzle, for what this template can ask for.
_TMA_DTYPE = {"BFloat16": 9, "Float16": 6, "Float32": 11, "Float8E4M3FN": 3, "Float8E5M2": 3, "Int8": 0}
_SWIZZLE = {"none": 0, "s32b": 1, "s64b": 2, "s128b": 3}


def ck(res, *rest):
    if isinstance(res, tuple):
        res, rest = res[0], tuple(res[1:]) + rest
    if res != cd.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA error: {res}")
    return rest[0] if len(rest) == 1 else rest


def burst(fn, n=64, reps=25):
    """Host time per call. The kernel is async, so this measures the CPU side --
    which is the whole point: it is what a launch-bound server pays."""
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        out.append((time.perf_counter_ns() - t0) / n / 1000.0)
    torch.cuda.synchronize()
    return min(out)


# ---------------------------------------------------------------------------
# the graph, built the ordinary way
# ---------------------------------------------------------------------------


def build_graph(m, n, k):
    g = cudnn.pygraph(io_data_type=BF16, intermediate_data_type=F32, compute_data_type=F32)
    A = g.tensor(name="A", uid=1, dim=[1, m, k], stride=[m * k, k, 1], data_type=BF16)
    B = g.tensor(name="B", uid=2, dim=[1, k, n], stride=[k * n, 1, k], data_type=BF16)
    C = g.matmul(A=A, B=B, name="mm")
    C.set_output(True).set_data_type(BF16).set_uid(3)
    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    frost = [i for i, p in enumerate(g.plans) if is_python_engine(p.engine_id)]
    if not frost:
        raise SystemExit("no FROST plan for this graph -- needs SM100 and CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1")
    g.select_plan(frost[0])
    g.check_support()
    g.build_plans()
    return g


def operands(m, n, k, device="cuda"):
    return {
        1: torch.randn(1, m, k, dtype=torch.bfloat16, device=device),
        2: torch.randn(1, n, k, dtype=torch.bfloat16, device=device),
        3: torch.empty(1, m, n, dtype=torch.bfloat16, device=device),
    }


def problem_size(m, n, k, data):
    """The tuple the generated host builds: (m, n, k, batch) then a stride
    triple per operand, in the axis order the kernel names them."""
    a, b, c = data[1].permute(1, 2, 0), data[2].permute(1, 2, 0), data[3].permute(1, 2, 0)
    return (m, n, k, 1, *a.stride(), *b.stride(), *c.stride())


# ---------------------------------------------------------------------------
# what codegen now hands us
# ---------------------------------------------------------------------------


def generated_module(graph):
    """The kernel source FROST wrote, imported so its constants can be read."""
    path = graph._compiled_plans[graph._plan_index]._compiled.generated_path
    spec = importlib.util.spec_from_file_location("frost_generated_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


_encode = ctypes.CDLL("libcuda.so.1").cuTensorMapEncodeTiled
_encode.restype = ctypes.c_int


def encode_tma(dtype_name, addr, dims, strides_16b, box, swizzle_name):
    """The driver call the generated host's `create_tensor_map_tiled` lowers to.

    Two conversions the DSL hides: its `global_strides` are in 16-byte units
    where the driver wants bytes, and the innermost stride is implicit, so only
    the outer ranks are passed.
    """
    out = ctypes.create_string_buffer(128 + 64)
    aligned = (ctypes.addressof(out) + 63) & ~63
    rank = len(dims)
    rc = _encode(
        ctypes.c_void_p(aligned),
        ctypes.c_int(_TMA_DTYPE[dtype_name]),
        ctypes.c_uint32(rank),
        ctypes.c_void_p(addr),
        (ctypes.c_uint64 * rank)(*dims),
        (ctypes.c_uint64 * (rank - 1))(*[s * 16 for s in strides_16b]),
        (ctypes.c_uint32 * rank)(*box),
        (ctypes.c_uint32 * rank)(*([1] * rank)),
        ctypes.c_int(0),
        ctypes.c_int(_SWIZZLE[swizzle_name]),
        ctypes.c_int(0),
        ctypes.c_int(0),
    )
    if rc != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled -> {rc}")
    return ctypes.string_at(aligned, 128)


def _swizzle_name(value):
    return getattr(value, "name", None) or str(value).rsplit(".", 1)[-1]


def build_map(role, index, problem, addr, mod):
    """One descriptor, mirroring the generated host's `create_tensor_map_tiled`.

    The major-axis branches matter: which two of an operand's three strides a
    descriptor reads depends on them, so taking the wrong branch would encode a
    valid-looking descriptor that reads the wrong memory.
    """
    at = mod.PROBLEM_FIELDS.index
    m, n, k, batch = problem[0], problem[1], problem[2], problem[3]

    def stride(prefix, axis):
        return problem[at(f"{prefix}_stride_{axis}_{index}")]

    if role in ("a", "b"):
        width = mod.ab_dtype.width
        dtype, swizzle = mod.ab_tma_dtype.__name__, _swizzle_name(mod.ab_tma_swizzle)
        if role == "a":
            operand_batch = 1 if mod.matmul_a_batch == 1 else batch
            if mod.a_is_m_major:
                dims = [m, k, operand_batch]
                strides = [stride("a", "k") * width // 128, stride("a", "l") * width // 128]
                box = [mod.a_tma_group_elems, mod.cgrp_tile_mnk[2], 1]
            else:
                dims = [k, m, operand_batch]
                strides = [stride("a", "m") * width // 128, stride("a", "l") * width // 128]
                box = [mod.cgrp_tile_mnk[2], mod.cta_tile_mnk[0], 1]
        else:
            operand_batch = 1 if mod.matmul_b_batch == 1 else batch
            if mod.b_is_n_major:
                dims = [n, k, operand_batch]
                strides = [stride("b", "k") * width // 128, stride("b", "l") * width // 128]
                box = [mod.b_tma_group_elems, mod.cgrp_tile_mnk[2], 1]
            else:
                dims = [k, n, operand_batch]
                strides = [stride("b", "n") * width // 128, stride("b", "l") * width // 128]
                box = [mod.cgrp_tile_mnk[2], mod.cta_tile_mnk[1], 1]
    elif role == "c":
        width = mod.cd_dtype.width
        dtype = mod.cd_tma_dtype.__name__
        if mod.cd_out_is_m_major:
            dims = [m, n, batch]
            strides = [stride("out", "n") * width // 128, stride("out", "l") * width // 128]
            box = [mod.cd_mmajor_atom_m, mod.epi_tile_mn[1], 1]
            swizzle = "s128b" if mod.use_tma_store_epi else "none"
        else:
            dims = [n, m, batch]
            strides = [stride("out", "m") * width // 128, stride("out", "l") * width // 128]
            box = [mod.epi_tile_mn[1], mod.epi_tile_mn[0], 1]
            swizzle = "s64b" if mod.use_tma_store_epi else "none"
    else:
        raise SystemExit(f"this demo does not build {role!r} descriptors (scale factors are block-scale only)")
    return encode_tma(dtype, addr, dims, strides, box, swizzle)


#: Which caller buffer each descriptor / pointer slot refers to. A and B are the
#: matmul's operands; `c` and `tap` are the same output reached two ways -- a TMA
#: store describes it with a descriptor, an STG epilogue takes its address.
_ROLE_UID = {"a": 1, "b": 2, "c": 3, "tap": 3}


def operand_addresses(table, data):
    """Every address the parameter block needs, keyed the way SLOT_TABLE names it."""
    addrs = {}
    for _name, kind, _size, source in table:
        if kind in ("tma", "ptr"):
            if source[0] not in _ROLE_UID:
                raise SystemExit(f"slot source {source} is outside this demo's narrow case")
            addrs[source] = data[_ROLE_UID[source[0]]].data_ptr()
    return addrs


def build_param_block(table, problem, addrs, mod):
    """The device parameter block, slot by slot, straight off SLOT_TABLE."""
    unknown = [row[0] for row in table if row[1] == "unknown"]
    if unknown:
        raise SystemExit(f"codegen did not classify {unknown} -- refusing rather than guessing")
    blobs = []
    for _name, kind, size, source in table:
        if kind == "scalar":
            blobs.append(ctypes.create_string_buffer(int(problem[source[1]]).to_bytes(size, "little"), size))
        elif kind == "ptr":
            blobs.append(ctypes.create_string_buffer(int(addrs[source]).to_bytes(8, "little"), 8))
        elif kind == "tma":
            blobs.append(ctypes.create_string_buffer(build_map(source[0], source[1], problem, addrs[source], mod), 128))
        else:
            raise SystemExit(f"slot kind {kind!r} is not part of this narrow demo")
    return blobs


# ---------------------------------------------------------------------------
# the kernel and its geometry, out of the compiled artifacts
# ---------------------------------------------------------------------------


def load_kernel(graph):
    """The CUfunction, without guessing a mangled name.

    `kernel_info` is keyed by the symbol the DSL emitted, and the module reports
    how many functions it holds -- so the kernel is identified by being the one
    that is there, and the name only cross-checks it.
    """
    launchable = graph._compiled_plans[graph._plan_index]._compiled._launchable
    cubin = launchable.__cubin__
    if isinstance(cubin, str):
        cubin = cubin.encode("latin-1")
    if not cubin:
        raise SystemExit("no cubin on the compiled function -- set CUTE_DSL_KEEP=cubin before the first compile")
    module = ck(cd.cuModuleLoadData(cubin))
    count = int(ck(cd.cuModuleGetFunctionCount(module)))
    err, funcs = cd.cuModuleEnumerateFunctions(count, module)  # count FIRST
    ck(err)
    if count != 1:
        raise SystemExit(f"{count} kernels in the module; pick by the kernel_info name")
    err, name = cd.cuFuncGetName(funcs[0])
    ck(err)
    declared = list(launchable.kernel_info)
    return funcs[0], bytes(name).decode(errors="ignore").rstrip("\x00"), declared, cubin


def kernel_param_widths(func):
    sizes = []
    while True:
        err, _off, size = cd.cuFuncGetParamInfo(func, len(sizes))
        if err != cd.CUresult.CUDA_SUCCESS:
            return sizes
        sizes.append(int(size))


def geometry(mod, m, n, batch=1):
    """The closed form the generated host launches with."""
    grid = (
        ((m + mod.cgrp_tile_mnk[0] - 1) // mod.cgrp_tile_mnk[0]) * mod.cluster_shape_mnk[0],
        ((n + mod.cgrp_tile_mnk[1] - 1) // mod.cgrp_tile_mnk[1]) * mod.cluster_shape_mnk[1],
        batch,
    )
    return grid, (mod.threads_per_cta, 1, 1), tuple(mod.cluster_shape_mnk)


def dynamic_smem(graph):
    """The one number without a first-class source.

    All of it is dynamic, so `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` reads 0, and
    it is not a module constant -- the DSL sums the kernel's `cutlass.Array`
    declarations at compile time. It reaches the device through exactly one
    constant in the compiled host module, which is what this reads. The durable
    fix is upstream: a property next to `__cubin__`.
    """
    text = str(graph._compiled_plans[graph._plan_index]._compiled._launchable.ir_module)
    seen = {}
    for match in re.finditer(r"llvm\.mlir\.constant\((\d+) : i64\)", text):
        value = int(match.group(1))
        if 1024 <= value <= 256 * 1024:
            seen[value] = seen.get(value, 0) + 1
        del value
    unique = [v for v, n in seen.items() if n == 1]
    if len(unique) != 1:
        raise SystemExit(f"could not identify the dynamic smem constant (candidates {sorted(seen)})")
    return unique[0]


def make_config(grid, block, cluster, smem, use_pdl, stream):
    cfg = cd.CUlaunchConfig()
    cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid
    cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block
    cfg.sharedMemBytes = smem
    cfg.hStream = cd.CUstream(stream)
    dim = cd.CUlaunchAttribute()
    dim.id = cd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    dim.value.clusterDim.x, dim.value.clusterDim.y, dim.value.clusterDim.z = cluster
    pdl = cd.CUlaunchAttribute()
    pdl.id = cd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    pdl.value.programmaticStreamSerializationAllowed = 1 if use_pdl else 0
    attrs = [dim, pdl]  # the list has to outlive cfg: cuda-python stores a pointer into it
    cfg.attrs = attrs
    cfg.numAttrs = 2
    return cfg, attrs


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


class BarePlan:
    """A built launch plus where each address goes. Execute writes N words."""

    __slots__ = ("func", "cfg", "argv_addr", "blobs", "slots", "order", "mod", "table", "problem", "_keep")

    def execute(self, ptrs):
        """The whole entry point: one store per operand, then launch."""
        for slot, i in self.slots:
            slot.value = ptrs[i]
        self.launch()

    def launch(self):
        (err,) = cd.cuLaunchKernelEx(self.cfg, self.func, self.argv_addr, 0)
        if err != cd.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"launch failed: {err}")


def prepare(graph, m, n, k, data, stream):
    mod, path = generated_module(graph)
    func, name, declared, cubin = load_kernel(graph)

    problem = problem_size(m, n, k, data)
    sorted_uids = sorted(data)
    addrs = operand_addresses(mod.SLOT_TABLE, data)
    blobs = build_param_block(mod.SLOT_TABLE, problem, addrs, mod)

    widths = kernel_param_widths(func)
    if widths != [row[2] for row in mod.SLOT_TABLE]:
        raise SystemExit(f"SLOT_TABLE says {[r[2] for r in mod.SLOT_TABLE]}, the cubin says {widths}")

    grid, block, cluster = geometry(mod, m, n)
    smem = dynamic_smem(graph)
    cfg, attrs = make_config(grid, block, cluster, smem, getattr(mod, "USE_PDL", False), stream)
    ck(*cd.cuFuncSetAttribute(func, cd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem))

    argv = (ctypes.c_void_p * len(blobs))(*[ctypes.addressof(b) for b in blobs])

    plan = BarePlan()
    plan.func, plan.cfg, plan.argv_addr = func, cfg, ctypes.addressof(argv)
    plan.blobs, plan.mod, plan.table = blobs, mod, mod.SLOT_TABLE
    plan.order, plan.problem = sorted_uids, problem
    # Where each operand's address lands, resolved once: the slot carrying it,
    # and its position in the caller's array. A tensor map keeps the address in
    # its first 8 bytes, so rebinding one is a store, not a re-encode.
    binding = []
    for slot_index, (_name, kind, _size, source) in enumerate(mod.SLOT_TABLE):
        if kind in ("tma", "ptr"):
            binding.append((ctypes.c_uint64.from_buffer(blobs[slot_index]), sorted_uids.index(_ROLE_UID[source[0]])))
    plan.slots = tuple(binding)
    plan._keep = (argv, attrs, cfg, cubin)
    return plan, mod, path, name, declared, widths


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--vary-m", type=int, default=0, help="also serve this M from the same built block")
    args = ap.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise SystemExit("the FROST gemm engine claims SM100")
    torch.cuda.init()
    m, n, k = args.m, args.n, args.k
    stream = torch.cuda.current_stream().cuda_stream

    graph = build_graph(m, n, k)
    data = operands(m, n, k)
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute(data, workspace)
    torch.cuda.synchronize()
    reference = data[3].clone()

    plan, mod, path, name, declared, widths = prepare(graph, m, n, k, data, stream)

    print(f"generated kernel : {path}")
    print(f"cubin symbol     : {name[:96]}")
    print(f"  == kernel_info : {declared[:1] == [name]}")
    print(f"parameter block  : {len(widths)} slots, {sum(widths)} B, widths {widths}")
    for slot, (slot_name, kind, size, source) in enumerate(mod.SLOT_TABLE):
        print(f"  slot {slot:2d}  {slot_name:16s} {kind:7s} {size:4d} B   from {source}")
    grid, block, cluster = geometry(mod, m, n)
    print(f"geometry         : grid={grid} block={block} cluster={cluster} smem={plan.cfg.sharedMemBytes}")

    ptrs = [data[uid].data_ptr() for uid in plan.order]
    data[3].zero_()
    plan.execute(ptrs)
    torch.cuda.synchronize()
    if not torch.equal(data[3], reference):
        print(f"\nMISMATCH: max|diff| = {(data[3].float() - reference.float()).abs().max().item():.3e}")
        return 1
    print("\nbit-identical to graph.execute()  OK")

    # a second buffer set, to show the contract really is only about addresses
    second = operands(m, n, k)
    graph.execute(second, workspace)
    torch.cuda.synchronize()
    reference2 = second[3].clone()
    second[3].zero_()
    plan.execute([second[uid].data_ptr() for uid in plan.order])
    torch.cuda.synchronize()
    if not torch.equal(second[3], reference2):
        print(f"rebound to a second buffer set    MISMATCH: max|diff| = {(second[3].float() - reference2.float()).abs().max().item():.3e}")
        return 1
    print("rebound to a second buffer set    OK")

    print("\n=== host us/call (min over 25 bursts of 64) ===")
    rows = [
        ("graph.execute(dict)", lambda: graph.execute(data, workspace)),
        ("bare plan.execute(ptrs)", lambda: plan.execute(ptrs)),
        ("  building that ptr list", lambda: [data[uid].data_ptr() for uid in plan.order]),
        ("  cuLaunchKernelEx alone", plan.launch),
    ]
    for label, fn in rows:
        print(f"  {label:34s}{burst(fn):8.2f}")

    if args.vary_m and not vary_m(graph, plan, mod, args.vary_m, n, k):
        return 1
    print("\nThe kernel runs asynchronously, so these are HOST times -- which is what a")
    print("launch-bound server pays. See docs/frost_bare_launch.md for the contract.")
    return 0


def vary_m(graph, plan, mod, new_m, n, k):
    """The next rung: serve a different token count from the same built block.

    Driven by `PATCH_GROUPS`, which says per caller-supplied quantity what it
    reaches -- the slots that hold it, the descriptors it was built into, and
    the grid axis it sizes. Only what actually moved is applied, so a token
    count that leaves N and K alone never touches B's descriptor.

    Returns whether the result is bit-identical to a plan built at `new_m`.
    """
    del graph
    print(f"\n=== serving M={new_m} from the M={plan.problem[0]} block ===")
    reference_graph = build_graph(new_m, n, k)
    data = operands(new_m, n, k)
    ws = torch.empty(max(reference_graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    reference_graph.execute(data, ws)
    torch.cuda.synchronize()
    expected = data[3].clone()

    problem = problem_size(new_m, n, k, data)
    addrs = operand_addresses(plan.table, data)

    changed = [mod.PROBLEM_FIELDS[i] for i in range(len(problem)) if problem[i] != plan.problem[i]]
    remap, grid = set(), {"x": plan.cfg.gridDimX, "y": plan.cfg.gridDimY, "z": plan.cfg.gridDimZ}
    for var in changed:
        writes, maps, axis = mod.PATCH_GROUPS[var]
        value = problem[mod.PROBLEM_FIELDS.index(var)]
        for slot, offset, width in writes:
            ctypes.memmove(ctypes.addressof(plan.blobs[slot]) + offset, int(value).to_bytes(width, "little"), width)
        remap |= set(maps)
        if axis == "x":
            grid["x"] = ((value + mod.cgrp_tile_mnk[0] - 1) // mod.cgrp_tile_mnk[0]) * mod.cluster_shape_mnk[0]
        elif axis == "y":
            grid["y"] = ((value + mod.cgrp_tile_mnk[1] - 1) // mod.cgrp_tile_mnk[1]) * mod.cluster_shape_mnk[1]
    for slot in sorted(remap):
        source = plan.table[slot][3]
        ctypes.memmove(ctypes.addressof(plan.blobs[slot]), build_map(source[0], source[1], problem, addrs[source], mod), 128)
    plan.cfg.gridDimX, plan.cfg.gridDimY, plan.cfg.gridDimZ = grid["x"], grid["y"], grid["z"]
    plan.problem = problem

    print(f"  changed        : {changed}")
    print(f"  re-encoded     : descriptors {sorted(remap)} of {sorted(s for s, r in enumerate(plan.table) if r[1] == 'tma')}")
    print(f"  grid           : ({grid['x']},{grid['y']},{grid['z']})")
    ptrs = [data[uid].data_ptr() for uid in plan.order]
    data[3].zero_()
    plan.execute(ptrs)
    torch.cuda.synchronize()
    if not torch.equal(data[3], expected):
        print(f"  bit-identical to a plan built at M={new_m}    MISMATCH: max|diff| = {(data[3].float() - expected.float()).abs().max().item():.3e}")
        return False
    print(f"  bit-identical to a plan built at M={new_m}    OK")
    print(f"\n  {'bare plan.execute(ptrs) at the new M':34s}{burst(lambda: plan.execute(ptrs)):8.2f}")
    return True


if __name__ == "__main__":
    sys.exit(main())
