"""Make ``cudnn.pygraph`` engine-aware in place (no new class, no rename).

This augments the existing pybind ``pygraph`` class — the same sanctioned
mechanism ``__init__.py`` already uses (``pygraph.execute = _execute``) — so
``g = cudnn.pygraph(...)`` is unchanged for every existing sample, yet can
transparently route to a registered Python engine.

How it works:
  * A per-graph mirror (kept in a WeakKeyDictionary, since pybind instances
    reject arbitrary attributes) records a Node/Tensor IR alongside the real C++
    calls for a curated set of *represented* ops (matmul + common pointwise).
  * Every other op-builder is auto-wrapped to flag the graph "opaque" — the safe
    direction: it only *prevents* Python routing, never changes classic output.
  * The plan lifecycle (create_execution_plans / check_support / build_plans /
    get_workspace_size / execute / build) routes to a Python engine iff one is
    registered AND the whole graph is represented AND it supports the graph;
    otherwise it delegates to the untouched C++ path (byte-identical to before).

This is EAGER: the C++ graph is still built as ops are added. Lazy / pure-python
(no cuDNN) is a follow-up that needs a structured builder per op (multi-tensor
returns like sdpa can't be mirrored generically). Engine selection uses the flat
engine-id model in ``engines`` — cuDNN plans and Python plans share one id space.
"""

import weakref
from typing import Any, Dict

# Per-graph mirror state, keyed by the C++ pygraph instance.
_STATE: "weakref.WeakKeyDictionary[Any, Dict]" = weakref.WeakKeyDictionary()

_ORIG: Dict[str, Any] = {}
_INSTALLED = False

# Represented pointwise ops: pygraph method name -> (NativeGraph builder, arity).
# Mirroring delegates to the NativeGraph builder so the recorded op matches
# exactly what the reference/DSL engines consume (same code path as the tests).
_POINTWISE = {
    "relu": ("relu", 1),
    "gelu": ("gelu", 1),
    "sigmoid": ("sigmoid", 1),
    "tanh": ("tanh", 1),
    "add": ("add", 2),
    "mul": ("mul", 2),
    "bias": ("bias", 2),
    "scale": ("scale", 2),
}

# Method names that are lifecycle / query / config (never op-builders): left
# untouched except for the routing wraps installed explicitly below.
_LIFECYCLE = {
    "validate",
    "build",
    "build_operation_graph",
    "build_plans",
    "build_plan_at_index",
    "create_execution_plan",
    "create_execution_plans",
    "check_support",
    "execute",
    "execute_plan_at_index",
    "get_workspace_size",
    "get_workspace_size_plan_at_index",
    "get_execution_plan_count",
    "get_engine_count",
    "get_engine_and_knobs_at_index",
    "get_knobs_for_engine",
    "get_plan_name_at_index",
    "get_behavior_notes",
    "get_behavior_notes_for_plan_at_index",
    "deselect_engines",
    "deselect_numeric_notes",
    "deselect_behavior_notes",
    "deselect_workspace_greater_than",
    "select_numeric_notes",
    "select_behavior_notes",
    "serialize",
    "deserialize",
    "key",
    "populate_cuda_graph",
    "update_cuda_graph",
    "query_tensor_attributes_of_uid",
    "tensor",
    "tensor_like",
    "register_backend",
}


def _state(graph) -> Dict:
    st = _STATE.get(graph)
    if st is None:
        from .graph_native import NativeGraph

        st = {"ir": NativeGraph(), "map": {}, "opaque": False, "backends": [], "selected": None}
        _STATE[graph] = st
    return st


def _mirror_tensor(graph, cpp_t, dim, stride, data_type):
    st = _state(graph)
    ir_t = st["ir"].tensor(dim=list(dim), stride=(list(stride) if stride else None), data_type=data_type)
    st["map"][id(cpp_t)] = ir_t


def _tensor_inputs(st, args, kwargs):
    """Tensor operands, in positional-then-keyword order, mapped to IR tensors.
    Returns None if any operand is not represented (came from an opaque op)."""
    ir_inputs = []
    for v in list(args) + list(kwargs.values()):
        if id(v) in st["map"]:
            ir_inputs.append(st["map"][id(v)])
        elif _is_cudnn_tensor(v):
            return None  # a tensor operand we didn't mirror -> not representable
    return ir_inputs


def _is_cudnn_tensor(v) -> bool:
    import cudnn

    return isinstance(v, cudnn.tensor)


def _install_tensor_wraps(pygraph):
    def tensor(self, *args, **kwargs):
        out = _ORIG["tensor"](self, *args, **kwargs)
        try:
            b = dict(kwargs)
            names = ("dim", "stride", "data_type")
            for i, val in enumerate(args):
                if i < len(names):
                    b.setdefault(names[i], val)
            _mirror_tensor(self, out, b.get("dim", out.get_dim()), b.get("stride"), b.get("data_type"))
        except Exception:  # noqa: BLE001 — mirroring is best-effort; never break a real build
            _state(self)["opaque"] = True
        return out

    def tensor_like(self, *args, **kwargs):
        out = _ORIG["tensor_like"](self, *args, **kwargs)
        try:
            _mirror_tensor(self, out, out.get_dim(), out.get_stride(), out.get_data_type())
        except Exception:  # noqa: BLE001
            _state(self)["opaque"] = True
        return out

    pygraph.tensor = tensor
    pygraph.tensor_like = tensor_like


def _make_matmul_wrap():
    def matmul(self, *args, **kwargs):
        out = _ORIG["matmul"](self, *args, **kwargs)
        st = _state(self)
        try:
            ins = _tensor_inputs(st, args, kwargs)
            if ins is None or len(ins) != 2:
                st["opaque"] = True
                return out
            ir_c = st["ir"].matmul(ins[0], ins[1])
            st["map"][id(out)] = ir_c
        except Exception:  # noqa: BLE001
            st["opaque"] = True
        return out

    return matmul


def _make_pointwise_wrap(name, ir_method, arity):
    def pw(self, *args, **kwargs):
        out = _ORIG[name](self, *args, **kwargs)
        st = _state(self)
        try:
            # Scalar attributes (negative_slope / clips / ...) are not carried
            # by this mirror — routing a graph that uses them to a python
            # engine would silently compute the wrong thing. Go opaque instead.
            extras = [v for k, v in kwargs.items() if k not in ("name", "compute_data_type") and not _is_cudnn_tensor(v) and v is not None]
            extras += [v for v in args if not _is_cudnn_tensor(v)]
            if extras:
                st["opaque"] = True
                return out
            ins = _tensor_inputs(st, args, kwargs)
            if ins is None or len(ins) != arity:
                st["opaque"] = True
                return out
            ir_out = getattr(st["ir"], ir_method)(*ins)
            st["map"][id(out)] = ir_out
        except Exception:  # noqa: BLE001
            st["opaque"] = True
        return out

    return pw


def _make_opaque_wrap(name):
    orig = _ORIG[name]

    def opaque(self, *args, **kwargs):
        _state(self)["opaque"] = True  # only prevents python routing; classic output unchanged
        return orig(self, *args, **kwargs)

    return opaque


def _route(self) -> bool:
    """Pick a python engine over the represented IR, if eligible. Returns True
    iff a python engine was selected (else the classic cuDNN path is used)."""
    st = _state(self)
    if st["selected"] is not None:
        return True
    if st["opaque"] or not st["backends"] or not st["ir"]._nodes:
        return False
    ir = st["ir"]
    ir._backends = list(st["backends"])
    ir.create_execution_plans()
    st["selected"] = ir.selected_engine
    return st["selected"] is not None


def _install_lifecycle_wraps(pygraph):
    def register_backend(self, engine):
        _state(self)["backends"].append(engine)
        return self

    def create_execution_plans(self, *args, **kwargs):
        if _route(self):
            return None
        return _ORIG["create_execution_plans"](self, *args, **kwargs)

    def check_support(self, *args, **kwargs):
        if _route(self):
            return None
        return _ORIG["check_support"](self, *args, **kwargs)

    def build_plans(self, *args, **kwargs):
        if _state(self)["selected"] is not None:
            return None
        return _ORIG["build_plans"](self, *args, **kwargs)

    def get_workspace_size(self, *args, **kwargs):
        if _state(self)["selected"] is not None:
            return _state(self)["selected"].get_workspace_size()
        return _ORIG["get_workspace_size"](self, *args, **kwargs)

    def execute(self, tensor_to_device_buffer, *args, **kwargs):
        st = _state(self)
        if st["selected"] is None:
            _route(self)
        if st["selected"] is not None:
            uid_to_data = {}
            for key, buf in tensor_to_device_buffer.items():
                ir_t = st["map"].get(id(key))
                if ir_t is None:
                    raise KeyError("variant-pack key is not a represented tensor of this graph")
                uid_to_data[ir_t] = buf
            st["ir"].execute(uid_to_data)
            return None
        return _ORIG["execute"](self, tensor_to_device_buffer, *args, **kwargs)

    def build(self, *args, **kwargs):
        # Route through the wrapped steps so build() also reaches a python engine.
        if _route(self):
            return None
        return _ORIG["build"](self, *args, **kwargs)

    pygraph.register_backend = register_backend
    pygraph.create_execution_plans = create_execution_plans
    pygraph.check_support = check_support
    pygraph.build_plans = build_plans
    pygraph.get_workspace_size = get_workspace_size
    pygraph.execute = execute
    if hasattr(pygraph, "build"):
        _ORIG["build"] = pygraph.build
        pygraph.build = build


def install(pygraph) -> None:
    """Augment the pybind ``pygraph`` class in place. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    # Save + wrap tensor creation and represented ops.
    for name in ("tensor", "tensor_like", "matmul", *_POINTWISE):
        _ORIG[name] = getattr(pygraph, name)
    _install_tensor_wraps(pygraph)
    pygraph.matmul = _make_matmul_wrap()
    for name, (ir_method, arity) in _POINTWISE.items():
        setattr(pygraph, name, _make_pointwise_wrap(name, ir_method, arity))

    # Save the lifecycle originals we route, then install the routing wraps.
    for name in ("create_execution_plans", "check_support", "build_plans", "get_workspace_size", "execute"):
        _ORIG[name] = getattr(pygraph, name)
    _install_lifecycle_wraps(pygraph)

    # Auto-flag every other public op-builder as opaque (safe: only disables the
    # python path, never alters classic output).
    represented = {"matmul", *_POINTWISE}
    for name in dir(pygraph):
        if name.startswith("_") or name in _LIFECYCLE or name in represented:
            continue
        attr = getattr(pygraph, name, None)
        if not callable(attr):
            continue
        _ORIG[name] = attr
        setattr(pygraph, name, _make_opaque_wrap(name))

    _INSTALLED = True
