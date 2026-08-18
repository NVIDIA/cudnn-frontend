"""Flow 3, publish step: compile a kernel and publish it under a name.

The same build as flow 2 with the last two lines deleted. Nothing is written
out, so there is nothing to pack — the name in the process table is the entire
contract. The registry holds a reference, so the compiled object cannot be
collected out from under a live handle.

Importable (flow3_orchestrator.py drives it) and runnable on its own:

    python flow3_register_global.py
"""

import cudnn

from common import build

NAME = "tile_add_f32"
SHAPE = (1024, 512)


def register(name=NAME, shape=SHAPE, override=False):
    """Compile and publish. Returns the tensor uids the caller binds buffers to."""
    graph, (a, b, c) = build(name, shape)

    # No file. The kernel was compiled in this process and its cubin is already
    # in the CUDA context, so a name is all a C++ executor in this same process
    # needs. Duplicate name is an error unless override=True (the autotuner
    # case); callers holding a get_global handle must re-fetch afterwards.
    cudnn.register_global(graph, override=override)

    return {"a": a.uid, "b": b.uid, "c": c.uid}


def main():
    uids = register()
    print("registered:", cudnn.aot.registered_global_names())
    print("uids:", uids)
    cudnn.unregister_global(NAME)
    print("after unregister:", cudnn.aot.registered_global_names())


if __name__ == "__main__":
    main()
