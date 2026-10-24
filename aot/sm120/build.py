#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build libcudnn_frontend_aot_sm120.so: FROST SM120 prefill/combine kernels compiled ahead of time from the in-repo
templates, the quantized-KV decode kernels, a CUDA runtime shim and the flat C ABI. No GPU is needed (sm_120a is
cross-compiled); the build host needs the CuTeDSL wheel (nvidia-cutlass-dsl), a CUDA toolkit and a C++17 compiler.

    PYTHONPATH=<repo>/python python aot/sm120/build.py --out <dir> [--jobs N] [--cuda /usr/local/cuda]
        [--classes d128] [--runtime shim|cute-dsl] [--reuse]

Steps (each writes under --out):
  1. selection  FROST's SM120 heuristic over variants.GRID -> per-class knob picks (selection.json)
  2. export     one object + header per (dtype, class, knob set) and per combine (obj/, include/)
  3. codegen    the dispatch registry and selection tables (gen/aot_registry.inc)
  4. compile    decode .cu (nvcc, sm_120a), C ABI + runtime shim (C++), link lib/libcudnn_frontend_aot_sm120.so
  5. manifest   manifest.json: every variant, sizes, the FE commit, the CuTeDSL version

``--runtime cute-dsl`` links the CuTeDSL runtime (libcute_dsl_runtime.so) instead of the in-tree shim; it exists only to
prove the shim reproduces it bit for bit (test_sdpa_sm120 --compare-lib).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import ctypes
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import variants as V  # noqa: E402

LIB_NAME = "libcudnn_frontend_aot_sm120.so"
# export_to_c without tvm-ffi and the --gpu-arch sm_120a cross-compile are verified on 4.8.0 (python/cudnn/AGENTS.md Rule 7).
AOT_CUTEDSL_MIN_VERSION = (4, 8)


def check_cutedsl():
    """Exit with a version-naming error before anything imports the DSL."""
    sys.path.insert(0, str(REPO / "python"))
    from cudnn.frost import buffers

    installed, version = buffers.cutedsl_state()
    floor = ".".join(map(str, AOT_CUTEDSL_MIN_VERSION))
    if not installed:
        sys.exit(f"aot/sm120/build.py needs nvidia-cutlass-dsl >= {floor} (install this checkout: pip install -e .)")
    if version and version[0] == "nvidia-cutlass-dsl":
        try:
            have = tuple(int(x) for x in version[1].split("+", 1)[0].split(".")[:2])
        except ValueError:
            return
        if have < AOT_CUTEDSL_MIN_VERSION:
            sys.exit(f"aot/sm120/build.py needs nvidia-cutlass-dsl >= {floor}; {version[1]} is installed")


# ---------------------------------------------------------------------------------------------------------------------
# 1. selection
# ---------------------------------------------------------------------------------------------------------------------


def _sel_init(repo_python: str):
    sys.path.insert(0, repo_python)


def _candidates(c: V.ShapeClass, g: int, sm: int, units: int, sq: int, skv: int):
    import cudnn
    from cudnn.sdpa.fwd import heuristics as heur
    from cudnn.sdpa.fwd.engines import ENGINE_SPECS
    from cudnn.sdpa.graph_analyzer import SdpaGraphFacts

    spec = next(s for s in ENGINE_SPECS if s.name == "sdpa_fwd_prefill_sm120")
    facts = SdpaGraphFacts(
        b=1,
        h_q=units * g,
        h_kv=units,
        s_q=sq,
        s_kv=skv,
        d_qk=c.d,
        d_v=c.d,
        dtype=cudnn.data_type.HALF,
        causal=c.causal,
        bottom_right=c.causal,
        right_bound=0 if c.causal else None,
        window_left=c.window_left,
        has_sink=c.sink,
        device_cc=(12, 0),
        device_sm_count=sm,
    )
    out = []
    for plan in heur.recommend("A", facts, {spec.name: 0}):
        k = plan.knobs
        out.append(V.Knobs(g if k.pack_gqa else 0, int(k.sched_policy), int(k.split_kv), int(k.tile_m), int(k.tile_n)))
    return out


def _select_slice(args):
    """Knob picks of one (class, SM count, G) over the whole grid, as a flat list in grid order."""
    ci, sm, g = args
    c = V.CLASSES[ci]
    picks, first_in_domain = [], 0
    for iu, isq, iskv in V.grid_points():
        cands = _candidates(c, g, sm, V.GRID_UNITS[iu], V.GRID_SQ[isq], V.GRID_SKV[iskv])
        if not cands:
            raise RuntimeError(f"heuristic proposed nothing for {c} g={g} sm={sm} cell {(iu, isq, iskv)}")
        hit = next((k for k in cands if V.in_domain(c, g, k)), None)
        first_in_domain += hit is cands[0]
        picks.append(hit if hit is not None else V.snap(c, g, cands[0]))
    return ci, sm, g, [tuple(k.__dict__.values()) for k in picks], first_in_domain


def run_selection(class_ids, jobs: int):
    work = [(ci, sm, g) for ci in class_ids for sm in V.SM_COUNTS for g in (*V.GRID_G, V.GRID_G_LARGE)]
    t0 = time.time()
    out, exact, total = {}, 0, 0
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_sel_init, initargs=(str(REPO / "python"),)) as ex:
        for ci, sm, g, picks, n_exact in ex.map(_select_slice, work):
            out[f"{ci}/{sm}/{g}"] = picks
            exact += n_exact
            total += len(picks)
    print(
        f"selection: {len(work)} slices x {total // max(len(work), 1)} cells in {time.time() - t0:.0f}s; "
        f"{exact / total:.1%} heuristic first choices compiled as-is"
    )
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 2. export
# ---------------------------------------------------------------------------------------------------------------------


def _export_init(repo_python: str):
    os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"
    sys.path.insert(0, repo_python)
    from cutlass.runtime import find_runtime_libraries

    # export_to_c on a host without the target GPU resolves the launcher's runtime symbols against the process.
    for p in find_runtime_libraries(enable_tvm_ffi=False):
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)


def _export_one(rec: dict, out: str, cuobjdump: str) -> dict:
    from cudnn.frost.tile_dsl.constants import DTYPE_BF16, DTYPE_FP16
    from cudnn.sdpa.fwd.config_sm120 import TemplateParams
    from cudnn.sdpa.fwd.kernels.sm120 import aot_host

    name, out = rec["name"], Path(out)
    t0 = time.time()
    try:
        if rec["kind"] == "prefill":
            c = V.CLASSES[rec["cls"]]
            dt = DTYPE_FP16 if rec["dtype"] == "f16" else DTYPE_BF16
            k = V.Knobs(*rec["knobs"])
            params = TemplateParams(
                dtype_qkv=dt,
                dtype_o=dt,
                window_left=c.window_left,
                window_right=0 if c.causal else None,
                bottom_right=c.causal,
                has_sink=c.sink,
                sched_policy=k.sched,
                q_tile=k.tile_m,
                kv_tile=k.tile_n,
                pack_gqa=bool(k.pack_gqa),
                split_kv=k.split,
            )
            compiled = aot_host.compile_prefill(params, c.d, k.pack_gqa or 1, name)
        else:
            compiled = aot_host.compile_combine(rec["dtype"], rec["sink"], name)
        with tempfile.TemporaryDirectory(dir=out / "obj") as tmp:
            compiled.export_to_c(tmp, name, function_prefix=name)
            shutil.move(os.path.join(tmp, name + ".h"), out / "include" / (name + ".h"))
            shutil.move(os.path.join(tmp, name + ".o"), out / "obj" / (name + ".o"))
        syms = _entry_symbols(out / "obj" / (name + ".o"), cuobjdump)
        bad = [s for s in syms if not s.startswith(name)]
        if not syms or bad:
            raise RuntimeError(f"device kernel symbols {syms} do not all start with {name}")
        return {**rec, "ok": True, "compile_s": round(time.time() - t0, 2), "obj_bytes": (out / "obj" / (name + ".o")).stat().st_size, "kernels": syms}
    except Exception as exc:  # noqa: BLE001
        return {**rec, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:3000]}


def _cubins(blob: bytes):
    """The cubin ELF images an export_to_c object embeds as data."""
    import struct

    pos = 1
    while True:
        i = blob.find(b"\x7fELF", pos)
        if i < 0:
            return
        if blob[i + 4] == 2:
            e_phoff, e_shoff = struct.unpack_from("<QQ", blob, i + 0x20)
            e_phentsize, e_phnum, e_shentsize, e_shnum = struct.unpack_from("<HHHH", blob, i + 0x36)
            (e_machine,) = struct.unpack_from("<H", blob, i + 0x12)
            end = max(e_shoff + e_shentsize * e_shnum, e_phoff + e_phentsize * e_phnum)
            if e_machine == 190 and 0 < end <= len(blob) - i:  # EM_CUDA
                yield blob[i : i + end]
                pos = i + end
                continue
        pos = i + 4


def _entry_symbols(obj: Path, cuobjdump: str):
    syms = []
    for blob in _cubins(obj.read_bytes()):
        with tempfile.NamedTemporaryFile(suffix=".cubin", dir=obj.parent) as f:
            f.write(blob)
            f.flush()
            text = subprocess.run([cuobjdump, "-symbols", f.name], capture_output=True, text=True, check=True).stdout
        syms += [line.split()[-1] for line in text.splitlines() if "STT_FUNC" in line and "STO_ENTRY" in line]
    return syms


def run_export(records, out: Path, jobs: int, cuda: Path, reuse: bool):
    cuobjdump = str(cuda / "bin" / "cuobjdump")
    todo = [r for r in records if not (reuse and (out / "obj" / (r["name"] + ".o")).exists() and (out / "include" / (r["name"] + ".h")).exists())]
    print(f"export: {len(records)} objects ({len(records) - len(todo)} reused), {jobs} jobs", flush=True)
    results = {r["name"]: {**r, "ok": True, "reused": True} for r in records if r not in todo}
    t0 = time.time()
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=jobs, initializer=_export_init, initargs=(str(REPO / "python"),), mp_context=ctx, max_tasks_per_child=16) as ex:
        futs = [ex.submit(_export_one, r, str(out), cuobjdump) for r in todo]
        for n, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            results[r["name"]] = r
            if not r["ok"] or n % 50 == 0 or n == len(futs):
                print(f"  [{n}/{len(futs)}] {'ok  ' if r['ok'] else 'FAIL'} {r['name']} {r.get('error', '')}", flush=True)
    for r in results.values():
        if r.get("reused"):
            r["obj_bytes"] = (out / "obj" / (r["name"] + ".o")).stat().st_size
    failed = [r for r in results.values() if not r["ok"]]
    print(f"export: {time.time() - t0:.0f}s, {len(failed)} failed", flush=True)
    return results, failed


# ---------------------------------------------------------------------------------------------------------------------
# 3. codegen
# ---------------------------------------------------------------------------------------------------------------------


def gen_registry(class_ids, class_knobs, selection, build_info: str) -> str:
    """The dispatch registry included by src/sdpa_sm120.cpp. Per dtype, a class's variants are contiguous and in the
    same order, so a selection table stores class-local indices shared by both dtypes."""
    L = ["// Generated by aot/sm120/build.py -- do not edit.", "// clang-format off", ""]
    prefill = []  # (name, dtype, class position, knobs)
    combines = [(V.combine_name(dt, s), dt, s) for dt in V.DTYPES for s in (False, True)]
    for dt in V.DTYPES:
        for pos, ci in enumerate(class_ids):
            for k in class_knobs[ci]:
                prefill.append((V.variant_name(dt, V.CLASSES[ci], k), dt, pos, k))
    for name, *_ in prefill:
        L.append(f'#include "{name}.h"')
    for name, *_ in combines:
        L.append(f'#include "{name}.h"')
    L.append("")
    for i, (name, dt, pos, k) in enumerate(prefill):
        c = V.CLASSES[class_ids[pos]]
        lse = "a.lse, " if k.split > 1 else ""
        sinks = "const_cast<float*>(a.sinks), " if c.sink else ""
        L += [
            f"static {name}_Kernel_Module_t g_mod_p{i};",
            f"static void load_p{i}() {{ {name}_Kernel_Module_Load(&g_mod_p{i}); }}",
            f"static int32_t launch_p{i}(const PrefillLaunch& a) {{ return cute_dsl_{name}_wrapper(&g_mod_p{i}, a.q, a.k, a.v, a.o, {lse}{sinks}"
            f"const_cast<int32_t*>(a.problem), const_cast<int64_t*>(a.q_stride), const_cast<int64_t*>(a.k_stride), const_cast<int64_t*>(a.v_stride), "
            f"const_cast<int64_t*>(a.o_stride), a.scale_log2, a.stream); }}",
        ]
    for i, (name, dt, sink) in enumerate(combines):
        sinks = "const_cast<float*>(a.sinks), " if sink else ""
        L += [
            f"static {name}_Kernel_Module_t g_mod_c{i};",
            f"static void load_c{i}() {{ {name}_Kernel_Module_Load(&g_mod_c{i}); }}",
            f"static int32_t launch_c{i}(const CombineLaunch& a) {{ return cute_dsl_{name}_wrapper(&g_mod_c{i}, a.o_partial, a.lse_partial, a.o, "
            f"const_cast<int32_t*>(a.problem), a.n_splits, const_cast<int64_t*>(a.o_stride), const_cast<int64_t*>(a.lse_stride), {sinks}a.stream); }}",
        ]
    L += ["", f'static const char kBuildInfo[] = "{build_info}";', "static const ClassInfo kClasses[] = {"]
    for ci in class_ids:
        c = V.CLASSES[ci]
        L.append(f"    {{{c.d}, {int(c.causal)}, {-1 if c.window_left is None else c.window_left}, {int(c.sink)}, {len(class_knobs[ci])}}},")
    L += ["};", f"static constexpr int kNumClasses = {len(class_ids)};", "", "static Variant g_prefill[] = {"]
    for i, (name, dt, pos, k) in enumerate(prefill):
        L.append(
            f'    {{"{name}", {V.DTYPES.index(dt)}, {pos}, {{{k.pack_gqa}, {k.sched}, {k.split}, {k.tile_m}, {k.tile_n}}}, load_p{i}, launch_p{i}, nullptr, {{}}, cudaSuccess}},'
        )
    L += ["};", "", "static Variant g_combine[] = {"]
    for i, (name, dt, sink) in enumerate(combines):
        L.append(f'    {{"{name}", {V.DTYPES.index(dt)}, {int(sink)}, {{0, 0, 0, 0, 0}}, load_c{i}, nullptr, launch_c{i}, {{}}, cudaSuccess}},')
    L += ["};", ""]
    # Class-local base offset of each (dtype, class) in g_prefill.
    base, off = [], 0
    for dt in V.DTYPES:
        row = []
        for ci in class_ids:
            row.append(off)
            off += len(class_knobs[ci])
        base.append(row)
    L.append(f"static const int kClassBase[{len(V.DTYPES)}][{len(class_ids)}] = {{" + ", ".join("{" + ", ".join(map(str, r)) + "}" for r in base) + "};")
    gs = (*V.GRID_G, V.GRID_G_LARGE)
    L += [
        f"static const int kSmCounts[] = {{{', '.join(map(str, V.SM_COUNTS))}}};",
        f"static constexpr int kNumSm = {len(V.SM_COUNTS)};",
        f"static constexpr int kNumG = {len(gs)};  // G = 1..{V.GRID_G[-1]}, then the G > {V.GRID_G[-1]} table",
        f"static constexpr int kNumUnits = {len(V.GRID_UNITS)}, kNumSq = {len(V.GRID_SQ)}, kNumSkv = {len(V.GRID_SKV)};",
        f"// [class][sm][g][units][s_q][s_kv] -> class-local variant index",
        f"static const uint8_t kSelect[{len(class_ids)}][{len(V.SM_COUNTS)}][{len(gs)}][{len(V.GRID_UNITS) * len(V.GRID_SQ) * len(V.GRID_SKV)}] = {{",
    ]
    for ci in class_ids:
        index = {k: i for i, k in enumerate(class_knobs[ci])}
        L.append("  {")
        for sm in V.SM_COUNTS:
            L.append("    {")
            for g in gs:
                vals = [index[V.Knobs(*t)] for t in selection[f"{ci}/{sm}/{g}"]]
                L.append("      {" + ",".join(map(str, vals)) + "},")
            L.append("    },")
        L.append("  },")
    L += ["};", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------------
# 4. compile + link
# ---------------------------------------------------------------------------------------------------------------------


def run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd) if len(cmd) < 40 else f"  $ {cmd[0]} ... ({len(cmd)} args)", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def compile_and_link(out: Path, objects, cuda: Path, cxx: str, runtime: str, dsl_libdir: Path):
    build = out / "build"
    build.mkdir(exist_ok=True)
    (out / "lib").mkdir(exist_ok=True)
    inc = [f"-I{REPO / 'include'}", f"-I{HERE / 'src'}", f"-I{out / 'include'}", f"-I{out / 'gen'}", f"-I{cuda / 'include'}"]
    nvcc = cuda / "bin" / "nvcc"
    run(
        [
            nvcc,
            "-O3",
            "-std=c++17",
            "-gencode",
            "arch=compute_120a,code=sm_120a",
            "-Xcompiler",
            "-fPIC,-fvisibility=hidden",
            *inc,
            "-c",
            HERE / "src" / "decode_q8_q4.cu",
            "-o",
            build / "decode_q8_q4.o",
        ]
    )
    cflags = ["-std=c++17", "-O2", "-fPIC", "-fvisibility=hidden", "-Wall", "-Wextra", *inc]
    run([cxx, *cflags, "-c", HERE / "src" / "sdpa_sm120.cpp", "-o", build / "sdpa_sm120.o"])
    objs = [build / "sdpa_sm120.o", build / "decode_q8_q4.o", *objects]
    libs = [f"-L{cuda / 'lib64'}", "-lcudart_static", "-ldl", "-lpthread", "-lrt"]
    if runtime == "shim":
        run([cxx, *cflags, "-c", HERE / "src" / "runtime_shim.cpp", "-o", build / "runtime_shim.o"])
        objs.append(build / "runtime_shim.o")
    else:
        libs = [f"-L{dsl_libdir}", "-lcute_dsl_runtime", f"-Wl,-rpath,{dsl_libdir}", *libs]
    lib = out / "lib" / LIB_NAME
    # Response file: several hundred objects.
    rsp = build / "link.rsp"
    rsp.write_text("\n".join(str(o) for o in objs))
    run([cxx, "-shared", "-o", lib, f"@{rsp}", f"-Wl,--version-script={HERE / 'src' / 'exports.map'}", "-Wl,--exclude-libs,ALL", "-Wl,-z,defs", *libs])
    return lib


# ---------------------------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "build" / "aot_sm120")
    ap.add_argument("--jobs", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--cuda", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    ap.add_argument("--classes", default=None, help="comma-separated substrings; build only classes whose key contains one (e.g. d128)")
    ap.add_argument("--runtime", choices=("shim", "cute-dsl"), default="shim")
    ap.add_argument("--reuse", action="store_true", help="reuse selection.json and exported objects already in --out")
    args = ap.parse_args()
    check_cutedsl()
    out = args.out.resolve()
    for sub in ("obj", "include", "gen", "lib"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    def key(c):
        return f"d{c.d}_{c.mask_name}{'_sink' if c.sink else ''}"

    class_ids = [i for i, c in enumerate(V.CLASSES) if not args.classes or any(s in key(c) for s in args.classes.split(","))]
    if not class_ids:
        sys.exit(f"--classes {args.classes} matches none of {[key(c) for c in V.CLASSES]}")

    sel_path = out / "selection.json"
    fingerprint = hashlib.sha256((HERE / "variants.py").read_bytes() + (REPO / "python/cudnn/sdpa/fwd/heuristics.py").read_bytes()).hexdigest()
    cached = json.loads(sel_path.read_text()) if (args.reuse and sel_path.exists()) else {}
    if cached.get("fingerprint") == fingerprint and all(f"{ci}/{V.SM_COUNTS[0]}/1" in cached["picks"] for ci in class_ids):
        selection = cached["picks"]
        print("selection: reused")
    else:
        selection = run_selection(class_ids, args.jobs)
        sel_path.write_text(json.dumps({"fingerprint": fingerprint, "picks": selection}))

    class_knobs = {}
    for ci in class_ids:
        ks = {V.Knobs(*t) for (k, picks) in selection.items() if k.startswith(f"{ci}/") for t in picks}
        ks |= set(V.extra_knobs(V.CLASSES[ci]))
        class_knobs[ci] = sorted(ks, key=lambda k: (k.pack_gqa, k.split, k.tile_m, k.sched, k.tile_n))
        if len(class_knobs[ci]) > 255:
            sys.exit(f"class {key(V.CLASSES[ci])} has {len(class_knobs[ci])} variants; the selection table stores uint8")
        print(f"  class {key(V.CLASSES[ci])}: {len(class_knobs[ci])} knob sets")

    records = [
        {"kind": "prefill", "name": V.variant_name(dt, V.CLASSES[ci], k), "dtype": dt, "cls": ci, "knobs": list(k.__dict__.values())}
        for dt in V.DTYPES
        for ci in class_ids
        for k in class_knobs[ci]
    ]
    records += [{"kind": "combine", "name": V.combine_name(dt, s), "dtype": dt, "sink": s} for dt in V.DTYPES for s in (False, True)]
    results, failed = run_export(records, out, args.jobs, args.cuda, args.reuse)
    if failed:
        (out / "failed.json").write_text(json.dumps(failed, indent=1))
        sys.exit(f"{len(failed)} exports failed; see {out / 'failed.json'}")

    import cutlass
    from cutlass.runtime import find_runtime_libraries

    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    fe_commit = git("rev-parse", "--short=12", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    n_pre = sum(1 for r in records if r["kind"] == "prefill")
    build_info = f"cudnn-frontend {fe_commit} cutlass-dsl {getattr(cutlass, '__version__', '?')} sm_120a {n_pre} prefill + {len(records) - n_pre} combine kernels, runtime {args.runtime}"
    (out / "gen" / "aot_registry.inc").write_text(gen_registry(class_ids, class_knobs, selection, build_info))

    dsl_libdir = Path(next(p for p in find_runtime_libraries(enable_tvm_ffi=False) if "cute_dsl_runtime" in p)).parent
    lib = compile_and_link(out, [out / "obj" / (r["name"] + ".o") for r in records], args.cuda, args.cxx, args.runtime, dsl_libdir)

    manifest = {
        "build_info": build_info,
        "fe_commit": git("rev-parse", "HEAD"),
        "fe_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "cutlass_dsl": getattr(cutlass, "__version__", "?"),
        "gpu_arch": "sm_120a",
        "runtime": args.runtime,
        "lib_bytes": lib.stat().st_size,
        "object_bytes": sum(r.get("obj_bytes", 0) for r in results.values()),
        "classes": [key(V.CLASSES[ci]) for ci in class_ids],
        "variants": sorted(results.values(), key=lambda r: r["name"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {lib} ({lib.stat().st_size / 2**20:.1f} MiB), {n_pre} prefill + {len(records) - n_pre} combine objects")


if __name__ == "__main__":
    main()
