"""Flow 3, execute step (Python): fetch a published kernel by name and run it.

Below the lookup line this is byte-identical to flow 2 — both flows hand back a
graph, so execution is one API. The only difference is where the graph came
from: get_global() instead of a container.

Importable (flow3_orchestrator.py case A drives it) and runnable on its own:

    python flow3_global_execute.py
"""

import torch

import cudnn

from common import gather
from flow3_register_global import NAME, SHAPE, register


def execute_from_global(name, uids, shape, handle):
    """Fetch by name and execute. Returns max |err| against a + b."""
    # A snapshot: it keeps running the kernel it was fetched with, so after an
    # override=True re-registration a caller that wants the new one re-fetches.
    graph = cudnn.get_global(name, handle)

    a = torch.randn(shape, device="cuda")
    b = torch.randn(shape, device="cuda")
    out = torch.full(shape, float("nan"), device="cuda")  # poison: a no-op is visible
    workspace = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")

    graph.execute(gather(graph, {uids["a"]: a, uids["b"]: b, uids["c"]: out}), workspace, handle=handle)
    torch.cuda.synchronize()

    return (out - (a + b)).abs().max().item()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    handle = cudnn.create_handle()
    uids = register()
    print("max |err|:", execute_from_global(NAME, uids, SHAPE, handle))
    cudnn.unregister_global(NAME)
    cudnn.destroy_handle(handle)


if __name__ == "__main__":
    main()
