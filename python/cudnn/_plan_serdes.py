# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialized form of a ``pygraph`` whose selected plan is a python engine (FROST, ...).

The backend's blob is its execution plan. A python engine's plan is not a blob: it is a
deterministic function of the graph and ``(engine_id, knobs)``, which is exactly what
``create_execution_plan`` replays. So the blob carries that recipe and, optionally, the
compiled kernels the plan was built from:

    MAGIC | header length (u64 LE) | header (JSON, utf-8) | kernel objects

- ``graph``: every node and tensor of the python IR, tensors referenced by uid. A value the
  encoder cannot represent exactly (a score_mod callback, a framework object) refuses
  serialization; nothing is dropped silently.
- ``plan``: the selected plan's ``(engine_id, public knobs)``. Engine ids and knob values are
  append-only, so the recipe stays valid on any frontend that still ships the engine.
- ``kernels``: the ``compiled_cache`` entries the plan's build used, each with the environment
  manifest it was compiled under. Deserialize installs one only when that manifest equals the
  current process's exactly; otherwise the kernel is compiled again. A stale object is never
  loaded.

Deserializing loads native code from the blob: only deserialize blobs from a trusted source.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import importlib
import json
import struct
from typing import TYPE_CHECKING, Any, Dict, List

from .graph_types import NodeType, Tensor
from .nodes import Node

if TYPE_CHECKING:
    from ._pygraph import pygraph

MAGIC = b"CUDNN_FE_PYPLAN\0"
FORMAT = 1
_LEN = struct.Struct("<Q")
# Graph-level constructor arguments that are plain values; handle / kernel_cache / device_property are not graph content.
_GRAPH_KWARGS = ("name", "sm_count", "sm_version", "is_dynamic_shape_enabled", "is_override_shape_enabled")


def is_python_plan(data: Any) -> bool:
    """Whether ``data`` (bytes or the legacy list of byte values) is a blob this module wrote."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data[: len(MAGIC)]) == MAGIC
    if isinstance(data, (list, tuple)) and len(data) >= len(MAGIC):
        try:
            return bytes(data[: len(MAGIC)]) == MAGIC
        except (TypeError, ValueError):
            return False
    return False


class _Unserializable(TypeError):
    pass


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _type_path(t: type) -> str:
    return f"{t.__module__}:{t.__qualname__}"


def _resolve_type(path: str) -> type:
    module, _, qualname = path.partition(":")
    if module.split(".")[0] not in ("cudnn", "_compiled_module"):
        raise ValueError(f"serialized plan names a type outside cudnn: {path!r}")
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not (isinstance(obj, type) and hasattr(obj, "__members__")):
        raise ValueError(f"serialized plan names a type that is not an enum: {path!r}")
    return obj


def _encode(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Tensor):
        return {"$tensor": value.uid}
    if isinstance(value, enum.Enum):
        if isinstance(value, int):
            return {"$enum": _type_path(type(value)), "value": int(value), "name": value.name}
        return {"$enum": _type_path(type(value)), "name": value.name}
    t = type(value)
    if hasattr(t, "__members__") and hasattr(value, "name") and hasattr(value, "__int__"):  # pybind11 enum
        return {"$enum": _type_path(t), "value": int(value), "name": value.name}
    if isinstance(value, (list, tuple)):
        items = [_encode(v, f"{where}[{i}]") for i, v in enumerate(value)]
        return items if isinstance(value, list) else {"$tuple": items}
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"$dict": {k: _encode(v, f"{where}.{k}") for k, v in value.items()}}
    raise _Unserializable(f"{where} holds a {t.__module__}.{t.__qualname__}, which a serialized plan cannot represent")


def _decode(value: Any, tensors: Dict[int, Tensor]) -> Any:
    if isinstance(value, list):
        return [_decode(v, tensors) for v in value]
    if not isinstance(value, dict):
        return value
    if "$tensor" in value:
        return tensors[value["$tensor"]]
    if "$tuple" in value:
        return tuple(_decode(v, tensors) for v in value["$tuple"])
    if "$dict" in value:
        return {k: _decode(v, tensors) for k, v in value["$dict"].items()}
    if "$enum" in value:
        t = _resolve_type(value["$enum"])
        # Values are the append-only contract for backend-mirrored enums (data_type, knob_type);
        # python enums are looked up by name.
        if "value" in value and (type(value["value"]) is not int):
            raise ValueError(f"serialized enum value must be an integer: {value!r}")
        return t(value["value"]) if "value" in value else t[value["name"]]
    raise ValueError(f"serialized plan holds an unknown value encoding: {sorted(value)}")


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def _tensor_fields() -> List[str]:
    return [f.name for f in dataclasses.fields(Tensor) if f.name != "owner"]


def _encode_graph(graph: "pygraph") -> Dict[str, Any]:
    ctx = graph._context
    return {
        "context": {
            "io_data_type": _encode(ctx.io_data_type, "context.io_data_type"),
            "intermediate_data_type": _encode(ctx.intermediate_data_type, "context.intermediate_data_type"),
            "compute_data_type": _encode(ctx.compute_data_type, "context.compute_data_type"),
        },
        "graph_kwargs": {k: _encode(graph._cpp_graph_kwargs[k], f"graph.{k}") for k in _GRAPH_KWARGS if k in graph._cpp_graph_kwargs},
        "is_cuda_graph_replay_expected": graph.is_cuda_graph_replay_expected,
        "next_uid": graph._next_uid,
        "reserved_uids": sorted(graph._reserved_uids),
        "node_count": dict(graph._node_count),
        "tensors": [{f: _encode(getattr(t, f), f"tensor {t.name!r}.{f}") for f in _tensor_fields()} for t in graph._tensor_by_uid.values()],
        "nodes": [
            {
                "name": n.name,
                "node_type": n.node_type.name,
                "compute_data_type": _encode(n.compute_data_type, f"node {n.name!r}.compute_data_type"),
                "inputs": {port: t.uid for port, t in n.inputs.items()},
                "outputs": {port: t.uid for port, t in n.outputs.items()},
                "params": {k: _encode(v, f"node {n.name!r} param {k!r}") for k, v in n.params.items()},
            }
            for n in graph._nodes
        ],
    }


def _decode_graph(graph: "pygraph", data: Dict[str, Any]) -> None:
    from ._pygraph import GraphContext

    ctx = data["context"]
    graph._context = GraphContext(**{k: _decode(v, {}) for k, v in ctx.items()})
    for k, v in data["graph_kwargs"].items():
        graph._cpp_graph_kwargs[k] = _decode(v, {})
    graph.is_cuda_graph_replay_expected = bool(data["is_cuda_graph_replay_expected"])
    fields = set(_tensor_fields())
    tensors: Dict[int, Tensor] = {}
    pending = []
    for record in data["tensors"]:
        unknown = set(record) - fields
        if unknown:
            raise ValueError(f"serialized plan has tensor attributes this frontend does not know: {sorted(unknown)}")
        attrs = {k: v for k, v in record.items() if k != "ragged_offset"}
        t = Tensor(**{k: _decode(v, {}) for k, v in attrs.items()})
        for k in ("dim", "stride"):
            setattr(t, k, list(getattr(t, k) or []))
        tensors[t.uid] = t
        pending.append((t, record.get("ragged_offset")))
    for t, ragged in pending:
        t.ragged_offset = _decode(ragged, tensors)
        graph._register_tensor(t)
    for record in data["nodes"]:
        node = Node(record["name"], NodeType[record["node_type"]], _decode(record["compute_data_type"], tensors))
        node.inputs = {port: tensors[uid] for port, uid in record["inputs"].items()}
        node.outputs = {port: tensors[uid] for port, uid in record["outputs"].items()}
        node.params = {k: _decode(v, tensors) for k, v in record["params"].items()}
        graph._nodes.append(node)
    graph._next_uid = int(data["next_uid"])
    graph._reserved_uids = set(data["reserved_uids"])
    graph._node_count = dict(data["node_count"])


# ---------------------------------------------------------------------------
# Blob
# ---------------------------------------------------------------------------


def serialize(graph: "pygraph", include_kernels: bool) -> bytes:
    """The blob for ``graph``'s selected, built python-engine plan."""
    import cudnn

    from .frost import compiled_cache

    index = graph._plan_index
    engine = graph.selected_engine
    engine_id, knobs = graph.get_engine_and_knobs_at_index(index)
    try:
        header: Dict[str, Any] = {
            "format": FORMAT,
            "frontend_version": str(getattr(cudnn, "__version__", "unknown")),
            "graph": _encode_graph(graph),
            "plan": {
                "engine_id": int(engine_id),
                "engine_name": engine.name,
                "knobs": [[_encode(k, "knob type"), int(v)] for k, v in knobs.items()],
            },
            "kernels": [],
        }
    except _Unserializable as exc:
        raise cudnn_graph_not_supported(f"serialize(): {exc}") from None
    payload = bytearray()
    if include_kernels:
        for entry in graph._plan_kernel_entries.get(index, ()):
            item = compiled_cache.read_entry(entry)
            if item is None:
                continue  # pruned since the build: the deserializing process compiles it
            description, data = item
            description.update(offset=len(payload), size=len(data))
            header["kernels"].append(description)
            payload += data
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + _LEN.pack(len(text)) + text + bytes(payload)


def deserialize(graph: "pygraph", data: Any, handle: Any = None, enforce_precompiled: bool = False) -> None:
    """Rebuild a blob from :func:`serialize` into the empty ``graph`` and build its plan."""
    from .frost import compiled_cache

    blob = bytes(data)
    if graph._nodes or graph._planning_done:
        raise RuntimeError("deserialize() of a python-engine plan needs an empty pygraph (no nodes, no plans)")
    start = len(MAGIC) + _LEN.size
    if len(blob) < start:
        raise ValueError("serialized python-engine plan is truncated")
    (length,) = _LEN.unpack_from(blob, len(MAGIC))
    if len(blob) < start + length:
        raise ValueError("serialized python-engine plan is truncated")
    header = json.loads(blob[start : start + length].decode("utf-8"))
    payload = memoryview(blob)[start + length :]
    if header.get("format") != FORMAT:
        raise ValueError(f"serialized python-engine plan has format {header.get('format')!r}; this frontend reads format {FORMAT}")

    device = getattr(getattr(handle if handle is not None else graph._handle, "device", None), "ordinal", None)
    seeded = 0
    for description in header["kernels"]:
        offset, size = int(description.pop("offset")), int(description.pop("size"))
        if offset < 0 or offset + size > len(payload):
            raise ValueError("serialized python-engine plan is truncated")
        if compiled_cache.seed_entry(description, bytes(payload[offset : offset + size]), device=device):
            seeded += 1
    seeded_note = f"{seeded} of the blob's {len(header['kernels'])} kernels match this environment (frontend, CuTe DSL, tvm-ffi, driver, device)"

    # Any later failure restores the empty graph, so the caller can retry on it.
    empty = {k: copy.copy(v) if isinstance(v, (list, dict, set)) else v for k, v in vars(graph).items()}
    try:
        _rebuild(graph, header, handle, enforce_precompiled, seeded_note)
    except BaseException:
        vars(graph).clear()
        vars(graph).update(empty)
        raise


def _rebuild(graph: "pygraph", header: Dict[str, Any], handle: Any, enforce_precompiled: bool, seeded_note: str) -> None:
    from .frost import compiled_cache

    if handle is not None:
        graph._handle = handle
    _decode_graph(graph, header["graph"])
    plan = header["plan"]
    knobs = {_decode(k, {}): int(v) for k, v in plan["knobs"]}
    graph.validate()
    graph.create_execution_plan(int(plan["engine_id"]), knobs)
    graph.select_plan(len(graph._plans) - 1)
    engine = graph.selected_engine
    if engine is None or engine.name != plan["engine_name"]:
        raise ValueError(f"engine id {plan['engine_id']} is {getattr(engine, 'name', 'the backend')!r} here, not {plan['engine_name']!r}")
    if not enforce_precompiled:
        graph.build_plans()
        return
    try:
        with compiled_cache.refuse_compiles():
            graph.build_plans()
    except Exception as exc:
        cause: Any = exc
        while cause is not None and not isinstance(cause, compiled_cache.CompileRefused):
            cause = cause.__cause__ or cause.__context__
        if cause is None:
            raise
        raise cudnn_graph_not_supported(
            f"enforce_precompiled: a kernel of the plan is neither in the blob nor in the compiled cache ({seeded_note}); {cause}"
        ) from exc


def cudnn_graph_not_supported(message: str) -> Exception:
    from ._pygraph import cudnn_graph_not_supported as make

    return make(message)
