"""Flow 3, both consumers, in one process.

    A)  register_global (Python)  ->  execute (Python)
    B)  register_global (Python)  ->  execute (C++, a nanobind extension)

Case B is the one that matters. The kernel is compiled by Python and never
serialised; the C++ executor is a separate shared object in the same process and
reaches the same graph through the process-global registry, by name. Nothing is
written to disk in either case.

Run build.sh first so the nanobind module exists; without it case B is skipped
and case A still runs.

    python flow3_orchestrator.py
"""

import sys

import torch

import cudnn
from cudnn._handle import to_backend_handle

from common import gather
from flow3_global_execute import execute_from_global
from flow3_register_global import NAME, SHAPE, register


def case_a(handle, uids):
    """register_global (Python) -> execute (Python)."""
    err = execute_from_global(NAME, uids, SHAPE, handle)
    print(f"  A  python register -> python execute      max |err| = {err}")
    return err == 0


def case_b(handle, uids):
    """register_global (Python) -> execute (C++ nanobind extension)."""
    try:
        import flow3_global_execute_nanobind as cpp
    except ImportError as e:
        print(f"  B  SKIPPED: {e}. Run ./build.sh to compile the extension.")
        return None

    # The proof that the extension shares the registry rather than holding a
    # private copy: it is a different .so and it sees what Python registered.
    seen = cpp.registered_names()
    print(f"     the C++ .so sees the registry as: {seen}")
    assert NAME in seen, "the extension has its own registry -- see kernel_library.h"

    # Resolved once, through the C++ side this time.
    uid_order = cpp.variant_pack_uids_sorted(NAME)
    workspace = torch.empty(max(cpp.workspace_size(NAME), 1), dtype=torch.uint8, device="cuda")

    a = torch.randn(SHAPE, device="cuda")
    b = torch.randn(SHAPE, device="cuda")
    out = torch.full(SHAPE, float("nan"), device="cuda")
    by_uid = {uids["a"]: a, uids["b"]: b, uids["c"]: out}

    # The hot path crosses into C++ with a flat pointer array and nothing else.
    cpp.execute(NAME, [by_uid[uid].data_ptr() for uid in uid_order], workspace.data_ptr(), to_backend_handle(handle))
    torch.cuda.synchronize()

    err = (out - (a + b)).abs().max().item()
    print(f"  B  python register -> C++ execute         max |err| = {err}")
    return err == 0


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    handle = cudnn.create_handle()

    # One registration, both consumers. This is the whole point: the kernel is
    # published once and whoever is in the process can reach it by name.
    uids = register()
    print(f"registered {NAME!r}; nothing written to disk")

    results = [case_a(handle, uids), case_b(handle, uids)]

    cudnn.unregister_global(NAME)
    cudnn.destroy_handle(handle)

    ran = [r for r in results if r is not None]
    raise SystemExit(0 if ran and all(ran) else 1)


if __name__ == "__main__":
    main()
