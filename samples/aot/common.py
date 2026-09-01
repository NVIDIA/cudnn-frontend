"""The graph every AOT sample builds, in one place.

A 2-D elementwise tile add whose operands arrive by TMA, routed to the CuTeDSL
engine. Small enough that the samples stay about the three flows rather than
about the kernel, but not a toy in the way a flat vector add is: it builds TMA
descriptors on every call, which is what a real kernel does and what dominates
the per-call host cost once a kernel has many operands.

Shapes must be a multiple of the 128x64 tile.

The plain 1-D add is kept below as ``build_plain``. It is not part of any flow
sample; it exists so bench_cpu_costs can price the SAME arithmetic with and
without TMA, which is the only way to attribute a cost to the descriptors
rather than to the kernel.
"""

import os

# The demo engines are opt_in in engines/manifest.py, so a graph only reaches
# them once this is set. Engines cannot be injected -- the manifest is the only
# way one exists -- so the gate is the sample's whole engine selection.
os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")

import cudnn  # noqa: E402 -- must follow the opt-in gate above

# One entry per kernel the flow-2 samples ship in a container.
CONTAINER_KERNELS = {
    "tile_add_small_f32": (128, 64),
    "tile_add_large_f32": (1024, 512),
}

# The no-TMA control, for the benchmark only.
PLAIN_KERNELS = {
    "add_small_f32": (4, 1024),
    "add_large_f32": (8, 4096),
}


def contiguous_stride(shape):
    stride, acc = [], 1
    for extent in reversed(shape):
        stride.append(acc)
        acc *= extent
    return list(reversed(stride))


def build(name, shape=(128, 64)):
    """Build and compile one named graph. The ordinary lifecycle, nothing AOT.

    Returns (graph, (a, b, c)) — the tensors come back so callers can key a
    variant pack by uid without looking them up by name.
    """
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a = graph.tensor(dim=list(shape), stride=contiguous_stride(shape), data_type=cudnn.data_type.FLOAT, name="A")
    b = graph.tensor(dim=list(shape), stride=contiguous_stride(shape), data_type=cudnn.data_type.FLOAT, name="B")
    c = graph.add(a, b, name="sum")
    c.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    # create_execution_plans -> check_support -> build_plans. The router picks
    # the engine; build_plans() is where the kernel materialises. (A plain
    # build() would try the cuDNN backend lowering first, which this graph does
    # not need.)
    graph.create_execution_plans()
    graph.build_plans()

    # Mandatory before either AOT flow: the name is the kernel's identity and
    # the lookup key in both.
    graph.set_name(name)
    return graph, (a, b, c)


def build_plain(name, shape=(4, 1024)):
    """The same add with a plain 1-D load: the benchmark's no-TMA control."""

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a = graph.tensor(dim=list(shape), stride=contiguous_stride(shape), data_type=cudnn.data_type.FLOAT, name="A")
    b = graph.tensor(dim=list(shape), stride=contiguous_stride(shape), data_type=cudnn.data_type.FLOAT, name="B")
    c = graph.add(a, b, name="sum")
    c.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    graph.build([cudnn.heur_mode.A])
    graph.set_name(name)
    return graph, (a, b, c)


def gather(graph, uid_to_buffer):
    """Pointers in the order execute() wants them.

    variant_pack_uids_sorted() is resolved once at startup; this gather is the
    only per-call work a caller does.
    """
    return [uid_to_buffer[uid] for uid in graph.variant_pack_uids_sorted()]
