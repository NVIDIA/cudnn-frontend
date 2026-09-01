"""Ahead-of-time kernel export and import.

Build a graph once, on a build box, and get the compiled kernel back in another
process — or another language — with no JIT, no cutlass and no compiler at the
other end.

Two destinations share one build path and one call machinery, and differ only in
where the compiled kernel lands:

* ``export_to_disk(graphs, path)`` packs the set into one container file;
* ``register_global(graph)`` writes nothing and publishes into the process table.

Both hand back a ``pygraph``, so every line below the lookup is the ordinary
execute API.

Only the CuTeDSL family engine can be exported today. A graph whose selected plan
belongs to the cuDNN backend, or to the cuda-python / CUDA C++ engines, raises a
clear "not implemented" — the retrofit is real work and is not done here.
"""

import json
from typing import Any, Dict, Iterable, List, Optional

from ._handle import to_backend_handle

__all__ = [
    "export_to_disk",
    "import_from_disk",
    "KernelLibrary",
    "register_global",
    "get_global",
    "unregister_global",
    "registered_global_names",
    "serialize_graph",
    "deserialize_graph",
]


def _compiled_plan_of(graph) -> Any:
    """The CompiledPlan that would run if this graph were executed."""
    engine = graph.selected_engine
    if engine is None:
        raise NotImplementedError(
            f"Graph {graph.get_name()!r} runs on the cuDNN backend engine, which does not support AOT export yet. "
            "Only the CuTeDSL family engine implements it today."
        )
    plan_index = graph._plan_index
    if plan_index not in graph._compiled_plans:
        raise RuntimeError(f"Graph {graph.get_name()!r} has not been built. Call build_plans() before exporting: the artifact is " "the compiled plan.")
    return graph._compiled_plans[plan_index]


def _graph_io_uids(graph) -> List[int]:
    """Every uid a caller has to supply a pointer for, from the Python graph.

    The fallback for a family with no cuDNN lowering: the non-virtual tensors
    wired to the graph's nodes ARE the variant pack.
    """
    consumed = set()
    for node in graph.nodes:
        for tensor in node.inputs.values():
            if tensor is not None:
                consumed.add(tensor.uid)

    uids = []
    for node in graph.nodes:
        for direction, ports in (("in", node.inputs), ("out", node.outputs)):
            for tensor in ports.values():
                if tensor is None:
                    continue
                # A terminal output has to land somewhere even if it was never
                # marked with set_output(): nothing downstream will produce a
                # buffer for it, so the caller must.
                terminal = direction == "out" and tensor.uid not in consumed
                if tensor.is_virtual and not terminal:
                    continue
                if tensor.uid not in uids:
                    uids.append(tensor.uid)
    return uids


def _lower_for_export(graph) -> Any:
    """Bring the C++ graph up to the point where its variant pack is known.

    Export needs what only the cuDNN lowering knows: the uid set the variant
    pack is addressed by, the pass-by-value scalars, the workspace layout and
    the frontend workspace size. That is exactly what build_operation_graph()
    computes, and it is computed the same way for every engine — which is why
    the artifact can reuse the existing serialize() blob instead of inventing a
    second description of the same graph.

    Not every exportable graph HAS a cuDNN lowering, though. The FROST
    linear-attention nodes are Python-engine-only: ``build_operation_graph()``
    on one of them fails outright, because there is no backend operation to
    build. For those the engine is the only thing that knows the buffer set, so
    the uid list is declared straight onto the C++ graph and everything
    downstream — serialize, the variant-pack template, execute — proceeds
    identically.
    """
    if graph._lowered_graph is None:
        if not graph._is_validated:
            graph.validate()
        import cudnn

        try:
            graph._lowered_graph = graph._lower_to_cpp()
            graph._lowered_graph.validate()
            graph._verify_uid_ownership()
        except cudnn.cudnnGraphNotSupportedError:
            # No backend lowering for these nodes. That is not an error here:
            # the plan being exported belongs to a python engine, which is the
            # only thing that was ever going to run them.
            lowered = cudnn._pybind_module.backend_graph()
            lowered._declare_variant_pack(_graph_io_uids(graph))
            graph._lowered_graph = lowered
            graph._cpp_bog_done = True
            return lowered
    if not graph._cpp_bog_done:
        graph._lowered_graph.build_operation_graph()
        graph._cpp_bog_done = True
        graph._sync_ir_shapes_from_backend()
    return graph._lowered_graph


def _attach_payload(graph) -> Any:
    """Ask this graph's engine for its artifact and hand it to the C++ graph."""
    name = graph.get_name()
    if not name or name == "test_graph":
        raise ValueError("set_name() is mandatory before AOT export: the name is the kernel's identity and the lookup key.")

    compiled = _compiled_plan_of(graph)
    payload, module_bytes = compiled.export_aot_payload(graph)

    lowered = _lower_for_export(graph)
    lowered.set_name(name)
    lowered._set_cutedsl_payload(json.dumps(payload), module_bytes)
    return lowered


def serialize_graph(graph) -> bytes:
    """One built CuTeDSL graph as a self-contained artifact.

    The bytes carry the graph's variant-pack description (the existing cuDNN
    serialization) plus the compiled module, and are accepted back by
    ``deserialize_graph``.
    """
    lowered = _attach_payload(graph)
    return bytes(lowered.serialize())


def deserialize_graph(data: bytes, handle: Optional[int] = None):
    """Rebuild an executable graph from ``serialize_graph`` bytes.

    Version, architecture and missing-runtime-dependency mismatches are all
    detected here, not at the first launch.
    """
    import cudnn

    graph = cudnn.pygraph()
    graph._lowered_graph = cudnn._pybind_module.backend_graph()
    backend_handle = to_backend_handle(handle)
    if backend_handle is not None:
        graph._lowered_graph.deserialize(backend_handle, list(data))
    else:
        graph._lowered_graph.deserialize(list(data))
    graph._is_built = True
    graph._cpp_graph_kwargs["name"] = graph._lowered_graph.get_name()
    return graph


def _wrap_lowered(lowered, handle: Optional[int] = None):
    """Present an imported C++ graph as the pygraph the caller executes."""
    import cudnn

    graph = cudnn.pygraph(handle=handle)
    graph._lowered_graph = lowered
    graph._is_built = True
    graph._cpp_graph_kwargs["name"] = lowered.get_name()
    return graph


class KernelLibrary:
    """The kernels in one container, addressed by name.

    Lookup is by name and never by position, so adding a kernel to a container
    cannot change what an existing caller resolves.
    """

    def __init__(self, graphs: Dict[str, Any]):
        self._graphs = dict(graphs)

    def __getitem__(self, name: str):
        try:
            return self._graphs[name]
        except KeyError:
            raise KeyError(f"no kernel named {name!r} in this library; it holds {sorted(self._graphs)}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._graphs

    def __len__(self) -> int:
        return len(self._graphs)

    def __iter__(self):
        return iter(self._graphs)

    def keys(self):
        return self._graphs.keys()

    def items(self):
        return self._graphs.items()

    def values(self):
        return self._graphs.values()

    def __repr__(self) -> str:
        return f"KernelLibrary({sorted(self._graphs)})"


def export_to_disk(graphs: Iterable[Any], path: str) -> None:
    """Pack a set of built graphs into one container.

    The whole set goes in and one container comes out, so adding a kernel later
    means calling this again with the full set. Every graph must have been named
    with ``set_name()``; a duplicate name within one call is an error.

    Nothing about the serialization reaches the caller: the container is opaque,
    and each engine stores itself inside it its own way.
    """
    import cudnn

    graphs = list(graphs)
    if not graphs:
        raise ValueError("export_to_disk() needs at least one graph")

    seen: Dict[str, int] = {}
    lowered: List[Any] = []
    for i, g in enumerate(graphs):
        name = g.get_name()
        if not name or name == "test_graph":
            raise ValueError(f"graph at position {i} has no name; set_name() is mandatory before AOT export")
        if name in seen:
            raise ValueError(f"graphs at positions {seen[name]} and {i} are both named {name!r}; names must be unique within one call")
        seen[name] = i
        lowered.append(_attach_payload(g))

    cudnn._pybind_module._aot_export_to_disk(lowered, path)


def import_from_disk(path: str, handle: Optional[int] = None) -> KernelLibrary:
    """Read a container back as name -> graph.

    Every graph in it is executable on return: version, architecture and
    missing-runtime-dependency mismatches are all raised here rather than at the
    first launch.
    """
    import cudnn

    pairs = cudnn._pybind_module._aot_import_from_disk(path, to_backend_handle(handle))
    return KernelLibrary({name: _wrap_lowered(lowered, handle) for name, lowered in pairs})


# ---------------------------------------------------------------------------
# Flow 3: register to memory. Same build path as flow 2 with the last two lines
# deleted — nothing is written out, so there is nothing to pack.
# ---------------------------------------------------------------------------


def register_global(graph, override: bool = False) -> None:
    """Publish a built graph under its ``set_name()`` name. No file is written.

    The kernel was compiled in this process and its cubin is already in the CUDA
    context, so a name is all a C++ executor in the same process needs. The
    registry holds a reference, so the compiled object cannot be collected out
    from under a live handle.

    A duplicate name is an error unless ``override=True`` (the autotuner case);
    callers holding a handle from ``get_global`` must re-fetch after an override.
    """
    import cudnn

    name = graph.get_name()
    if not name or name == "test_graph":
        raise ValueError("set_name() is mandatory before register_global(): the name is the kernel's identity.")

    compiled = _compiled_plan_of(graph)
    # The tvm-ffi table is flat and process-wide, so FE namespaces its entries.
    # The prefix is applied here and stripped in get_global; it never appears in
    # user code.
    symbol = cudnn._pybind_module._aot_global_symbol_for(name)
    payload = compiled.aot_global_payload(graph, symbol)

    lowered = _lower_for_export(graph)
    lowered.set_name(name)
    cudnn._pybind_module._aot_register_global(lowered, json.dumps(payload), override)


def get_global(name: str, handle: Optional[int] = None):
    """Fetch a graph published by ``register_global``, as a snapshot.

    The returned graph keeps running the kernel it was fetched with: after an
    ``override=True`` re-registration a caller that wants the new one must call
    ``get_global`` again.
    """
    import cudnn

    return _wrap_lowered(cudnn._pybind_module._aot_get_global(name, to_backend_handle(handle)), handle)


def unregister_global(name: str) -> None:
    """Drop a registration. Snapshots already handed out keep working.

    The tvm-ffi entries go too, or the compiled kernel they hold stays alive for
    the life of the process. A kernel registers one entry per launch step, named
    ``<prefix>.<i>``, so they are walked until one is missing.
    """
    import tvm_ffi

    import cudnn

    cudnn._pybind_module._aot_unregister_global(name)
    prefix = cudnn._pybind_module._aot_global_symbol_for(name)
    i = 0
    while tvm_ffi.get_global_func(f"{prefix}.{i}", allow_missing=True) is not None:
        tvm_ffi.remove_global_func(f"{prefix}.{i}")
        i += 1


def registered_global_names() -> List[str]:
    """Names currently published in this process."""
    import cudnn

    return list(cudnn._pybind_module._aot_registered_global_names())
