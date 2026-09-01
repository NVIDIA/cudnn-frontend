"""Flow 2, deploy box (Python): open the container and run every kernel in it.

Nothing here compiles. Run it in a fresh interpreter — with no cutlass and no
kernel toolchain installed, if you want to see the point — against a container
flow2_export.py wrote earlier, possibly on another machine.

    python flow2_import_execute.py [mykernels.cudnn]
"""

import json
import sys

import torch

import cudnn


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "mykernels.cudnn"
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    with open(path + ".manifest.json") as f:
        manifest = json.load(f)

    handle = cudnn.create_handle()

    # Version, architecture and missing-runtime-dependency mismatches are all
    # rejected HERE, loudly. A misread artifact is never an illegal access.
    lib = cudnn.import_from_disk(path, handle)
    print(f"opened {path} holding {len(lib)} kernel(s): {sorted(lib.keys())}")

    failures = 0
    for name in sorted(lib.keys()):
        graph = lib[name]  # BY NAME, never by position
        entry = manifest[name]
        shape = tuple(entry["shape"])

        # Resolved once, at startup.
        uid_order = graph.variant_pack_uids_sorted()
        workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")

        a = torch.randn(shape, device="cuda")
        b = torch.randn(shape, device="cuda")
        out = torch.full(shape, float("nan"), device="cuda")
        by_uid = {entry["a_uid"]: a, entry["b_uid"]: b, entry["c_uid"]: out}

        # Hot path: gather in uid_order, then execute.
        graph.execute([by_uid[uid] for uid in uid_order], workspace, handle=handle)
        torch.cuda.synchronize()

        err = (out - (a + b)).abs().max().item()
        print(f"  {name:<16} shape={shape} max |err| = {err}{'' if err == 0 else '   <-- MISMATCH'}")
        failures += err != 0

    cudnn.destroy_handle(handle)

    # The claim this sample exists to demonstrate.
    compiled = any(m in sys.modules for m in ("cutlass", "cutlass.cute"))
    print(f"kernel toolchain loaded in this process: {compiled}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
