"""Flow 2, build box: compile a set of graphs and ship one container.

Nothing here runs at launch time. This is the only step that needs a GPU
toolchain, cutlass, or Python at all — everything downstream just opens the file
this writes.

    python flow2_export.py [mykernels.cudnn]

Writes:
    mykernels.cudnn                 the container: name -> kernel
    mykernels.cudnn.manifest.json   shapes and uid order, for the samples only
"""

import json
import sys

import cudnn

from common import CONTAINER_KERNELS, PLAIN_KERNELS, build, build_plain

BENCH_KERNEL = "tile_add_large_f32"
PLAIN_BENCH_KERNEL = "add_small_f32"


def dump_bench_module(path, graph, bench_kernel=BENCH_KERNEL):
    """Drop the same kernel out as a plain .so, for the benchmarks.

    Only the benchmark needs this: its baseline arm calls the exported symbol
    directly, with no cuDNN in the picture, and both arms must provably run the
    same code. Not part of either flow.
    """
    plan = graph._compiled_plans[graph._plan_index]
    payload, module_bytes = plan.export_aot_payload(graph)
    with open(path + ".bench.so", "wb") as f:
        f.write(module_bytes)
    with open(path + ".bench.json", "w") as f:
        # These kernels are one launch, so the artifact has one step.
        (step,) = payload["steps"]
        args = step["args"]
        json.dump(
            {
                "kernel": bench_kernel,
                "symbol": step["function_name"],
                # Whether the entry point takes the stream POSITIONALLY. The
                # plain add reads the tvm-ffi environment stream and takes 3
                # args; the TMA kernel takes a CUstream and takes 4.
                "stream_arg": any(a.get("kind") == "STREAM" for a in args),
                # The shape the kernel was compiled for. The direct-FFI arms
                # build their own DLTensors and must match it exactly.
                "shape": [a["shape"] for a in args if a.get("kind") == "TENSOR"][0],
            },
            f,
            indent=2,
        )


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_bench = "--bench" in sys.argv
    # --plain swaps the TMA tile add for the flat 1-D add. Not a flow: it is
    # the benchmark's no-TMA control, the same arithmetic with the descriptors
    # taken away, which is the only way to attribute a cost to them.
    want_plain = "--plain" in sys.argv
    path = argv[0] if argv else ("plainkernels.cudnn" if want_plain else "mykernels.cudnn")

    kernels = PLAIN_KERNELS if want_plain else CONTAINER_KERNELS
    builder = build_plain if want_plain else build
    bench_kernel = PLAIN_BENCH_KERNEL if want_plain else BENCH_KERNEL

    graphs, tensors = [], {}
    for name, shape in kernels.items():
        graph, (a, b, c) = builder(name, shape)
        graphs.append(graph)
        tensors[name] = (a, b, c)

    # The SET goes in, ONE container comes out. Adding a kernel later means
    # calling this again with the full set. Serialization is internal: no blob,
    # envelope or format detail reaches the caller.
    cudnn.export_to_disk(graphs, path=path)

    # NOT part of the contract. The container already carries the uid order, and
    # flow2_import_execute.{py,cpp} read it from there via
    # variant_pack_uids_sorted(). This file only tells the samples which buffer
    # shapes to allocate and which uid is A / B / C, which a real caller knows
    # from its own model.
    manifest = {
        name: {
            "shape": list(shape),
            "a_uid": tensors[name][0].uid,
            "b_uid": tensors[name][1].uid,
            "c_uid": tensors[name][2].uid,
        }
        for name, shape in kernels.items()
    }
    with open(path + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if want_bench:
        dump_bench_module(path, graphs[list(kernels).index(bench_kernel)], bench_kernel)

    print(f"wrote {path}")
    for name in sorted(kernels):
        print(f"  {name}  shape={tuple(kernels[name])}")
    if want_bench:
        print(f"  + {path}.bench.so for bench_cpu_costs")


if __name__ == "__main__":
    main()
