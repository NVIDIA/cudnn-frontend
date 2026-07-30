# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python graph representation for cuDNN Frontend.

All graph structure and attributes are kept in Python. Graph construction is
backend-agnostic; a backend is chosen at create_execution_plans() time by the
Router, and the backend-specific representation (e.g. the C++ cuDNN graph) is
generated lazily only then.

Execution flow (unification proposal):
    build ops -> create_execution_plans() -> Router -> selected backend
    (a registered native engine, or the cuDNN Graph backend by lazy lowering)

Example with a native backend (pass torch tensors directly):
    >>> graph = pygraph()
    >>> graph.register_backend(MyDslEngine())  # any BaseEngine
    >>> C = graph.matmul(a_tensor, b_tensor)  # auto-creates descriptors
    >>> graph.execute({C: c_tensor})  # routes to a supporting backend, else cuDNN
"""

from dataclasses import dataclass
import weakref
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .graph_types import NodeType, Tensor
from .nodes import Node, _row_major_stride

if TYPE_CHECKING:
    from .engines.base import BaseEngine


@dataclass
class GraphContext:
    """Graph-level configuration defaults."""

    io_data_type: Any = None
    intermediate_data_type: Any = None
    compute_data_type: Any = None

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise RuntimeError("the graph is frozen after lowering/planning — build a new graph to change its configuration")
        object.__setattr__(self, name, value)


class pygraph:
    """Pure Python graph representation.

    All graph structure and attributes are kept in Python. C++ is only
    used for execution via lazy lowering.

    Example:
        >>> graph = pygraph(io_data_type=cudnn.data_type.HALF)
        >>> A = graph.tensor(dim=[8, 64, 128], name="A")
        >>> B = graph.tensor(dim=[8, 128, 256], name="B")
        >>> C = graph.matmul(A, B, name="mm1")
        >>> C.set_output(True)  # outputs are explicit, like the classic API
        >>>
        >>> # Inspect graph
        >>> print(graph.nodes)  # [Node('mm1', MATMUL)]
        >>> print(graph.nodes[0].inputs)  # {"A": ..., "B": ...}
        >>> print(graph.nodes[0].params)  # {"padding": 0.0}
    """

    def __init__(
        self,
        # ---- classic pybind constructor, POSITIONALLY IDENTICAL (existing
        # callers pass name/handle/sm_count/... by position; guarded by
        # test_api_signature_parity) --------------------------------------
        name: str = "test_graph",
        io_data_type: Any = None,
        intermediate_data_type: Any = None,
        compute_data_type: Any = None,
        handle: Any = None,
        sm_count: Any = None,
        sm_version: Any = None,
        kernel_cache: Any = None,
        device_property: Any = None,
        is_dynamic_shape_enabled: bool = False,
        is_override_shape_enabled: bool = False,
        *,
        # ---- new (keyword-only: never shifts the classic positional order) --
        backends: Optional[List["BaseEngine"]] = None,
        router: Any = None,
        **kwargs,
    ):
        self._context = GraphContext(
            io_data_type=io_data_type,
            intermediate_data_type=intermediate_data_type or io_data_type,
            compute_data_type=compute_data_type or io_data_type,
        )
        self._handle = handle  # cuDNN handle for the cuDNN lowering path
        # Classic graph-level configuration, forwarded verbatim to the C++
        # graph at lowering (**kwargs covers future binding args).
        self._cpp_graph_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self._cpp_graph_kwargs["name"] = name
        for _k, _v in (
            ("sm_count", sm_count),
            ("sm_version", sm_version),
            ("kernel_cache", kernel_cache),
            ("device_property", device_property),
        ):
            if _v is not None:
                self._cpp_graph_kwargs[_k] = _v
        if is_dynamic_shape_enabled:
            self._cpp_graph_kwargs["is_dynamic_shape_enabled"] = True
        if is_override_shape_enabled:
            self._cpp_graph_kwargs["is_override_shape_enabled"] = True
        self._nodes: List[Node] = []
        self._tensors: Dict[str, Tensor] = {}
        self._tensor_by_uid: Dict[int, Tensor] = {}
        self._next_uid: int = 1
        self._node_count: Dict[str, int] = {}
        self._lowered_graph: Any = None
        self._is_validated: bool = False
        self._is_built: bool = False
        self._data_bindings: Dict[int, Any] = {}  # uid -> tensor data for auto-bound inputs

        # Backend routing (see engines/router.py). Graph construction is
        # backend-agnostic. At create_execution_plans() the Router builds a flat
        # ranked plan list (python engines + cuDNN) in one shared engine-id
        # space; each plan is dispatched by its id (is_python_engine -> python
        # registry, else lower to cuDNN). ``_plan_index`` selects the plan to run.
        self._backends: List["BaseEngine"] = []
        self._router = router  # None => engines.router.default_router at route time
        self._plans: List[Any] = []  # list[PlanConfig], populated by create_execution_plans()
        self._planning_done: bool = False  # create_execution_plans() ran (one-shot)
        self._frozen: bool = False  # whole-surface freeze (set by _freeze())
        self._plan_index: int = 0
        self._backend_heuristics: Optional[List] = None  # heur modes for a backend plan
        self._cpp_plans_created: bool = False  # C++ create_execution_plans ran
        self._compiled_plans: Dict[int, Any] = {}  # plan_index -> CompiledPlan (python plans)
        self._cpp_bog_done: bool = False  # C++ build_operation_graph ran
        self._cpp_tensors: Dict[int, Any] = {}  # IR uid -> lowered C++ tensor
        self._reserved_uids: set = set()  # user-specified uids _alloc_uid must skip
        self._ambiguous_names: set = set()  # duplicate labels: excluded from the name index
        for _e in backends or ():  # constructor path uses the SAME validation
            self.register_backend(_e)

    # =========================================================================
    # Backend registration & routing
    # =========================================================================

    def register_backend(self, engine: "BaseEngine") -> "pygraph":
        """Add a candidate python execution engine. It joins the plan list at
        create_execution_plans() time when its check_support() accepts the graph.

        Validated at registration (not at failure time): the engine must declare
        a stable engine_id in the reserved python region, ids must be unique per
        graph, and registration after planning is rejected (planning is
        one-shot — build a new graph)."""
        from .engines.engine_ids import is_python_engine

        eid = getattr(engine, "engine_id", None)
        if not isinstance(eid, int) or not is_python_engine(eid):
            raise ValueError(f"engine {engine!r} must declare a stable integer engine_id >= PYTHON_ENGINE_ID_BASE (got {eid!r})")
        if any(e.engine_id == eid for e in self._backends):
            raise ValueError(f"engine_id {eid} is already registered on this graph")
        if self._planning_done:
            raise RuntimeError("cannot register a backend after create_execution_plans(); planning is one-shot — build a new graph")
        self._backends.append(engine)
        return self

    def set_router(self, router: Any) -> "pygraph":
        """Override the plan-list / ranking policy for this graph. Must be set
        before create_execution_plans() (a later router cannot affect the
        already-planned list)."""
        if self._planning_done:
            raise RuntimeError("cannot set a router after create_execution_plans(); planning is one-shot — build a new graph")
        self._router = router
        return self

    def _engine_by_id(self, engine_id: int) -> "BaseEngine":
        for e in self._backends:
            if e.engine_id == engine_id:
                return e
        raise KeyError(f"no registered python engine with id {engine_id}")

    @property
    def backends(self) -> List["BaseEngine"]:
        """Registered candidate python engines."""
        return list(self._backends)

    @property
    def plans(self) -> List[Any]:
        """The ranked plan list (list[PlanConfig]) from create_execution_plans()."""
        return list(self._plans)

    @property
    def _selected_plan_config(self) -> Optional[Any]:
        from .engines.engine_ids import is_python_engine

        if not self._plans or not 0 <= self._plan_index < len(self._plans):
            return None
        cfg = self._plans[self._plan_index]
        return cfg if is_python_engine(cfg.engine_id) else None

    @property
    def selected_engine(self) -> Optional["BaseEngine"]:
        """The python engine for the currently selected top-level plan entry,
        or None for the backend path. Populated after create_execution_plans()."""
        cfg = self._selected_plan_config
        return self._engine_by_id(cfg.engine_id) if cfg is not None else None

    # =========================================================================
    # Tensor Creation
    # =========================================================================

    def tensor(
        self,
        # classic public tensor() signature, POSITIONALLY IDENTICAL (guarded by
        # test_api_signature_parity); classic unset sentinels (NOT_SET, -1,
        # reordering NONE) are normalized to None below
        dim: List[int],
        stride: Optional[List[int]] = None,
        data_type: Any = None,
        is_virtual: bool = False,
        is_pass_by_value: bool = False,
        ragged_offset: Optional[Tensor] = None,
        reordering_type: Any = None,
        name: str = "",
        uid: Optional[int] = None,
        ragged_offset_multiplier: int = 1,
        **kwargs,
    ) -> Tensor:
        """Create a tensor."""
        if not name:
            name = f"tensor_{len(self._tensors)}"
        if data_type is not None and getattr(data_type, "name", None) == "NOT_SET":
            data_type = None
        if reordering_type is not None and getattr(reordering_type, "name", None) == "NONE":
            reordering_type = None
        if uid == -1:  # classic unset sentinel
            uid = None

        if uid is not None:
            # User-owned uid, same rule as set_uid (_reuid_tensor): classic
            # tensors have no uid until assigned, so a user uid may land on an
            # eagerly auto-assigned one — the user wins, the auto holder is
            # renumbered; colliding with another USER uid is an error.
            holder = self._tensor_by_uid.get(uid)
            if holder is not None:
                if holder.uid_assigned:
                    raise ValueError(f"uid {uid} is already user-assigned to tensor {holder.name!r}")
                fresh = self._alloc_uid()
                self._tensor_by_uid[fresh] = holder
                if holder.uid in self._data_bindings:
                    self._data_bindings[fresh] = self._data_bindings.pop(holder.uid)
                holder.uid = fresh
            self._reserved_uids.add(uid)

        dim = list(dim)  # classic API accepts torch.Size / tuples
        t = Tensor(
            name=name,
            dim=dim,
            stride=list(stride) if stride else _row_major_stride(dim),
            data_type=data_type or (self._context.intermediate_data_type if is_virtual else self._context.io_data_type),
            is_virtual=is_virtual,
            is_pass_by_value=is_pass_by_value,
            ragged_offset=ragged_offset,
            reordering_type=reordering_type,
            ragged_offset_multiplier=ragged_offset_multiplier,
            uid=uid if uid is not None else self._alloc_uid(),
            uid_assigned=uid is not None,
            dim_assigned=True,  # graph inputs: the user specified the layout
            stride_assigned=stride is not None,
            **kwargs,
        )
        self._register_tensor(t)
        return t

    def tensor_like(self, template: Any, name: str = "", is_virtual: bool = False) -> Tensor:
        """Create tensor from another IR tensor or a DLPack object (e.g. torch).

        Classic parity: CPU (host) framework tensors become pass-by-value, like
        the C++ tensor_like (is_pass_by_value = device == CPU)."""
        if isinstance(template, Tensor):
            return self.tensor(
                dim=list(template.dim),
                stride=list(template.stride),
                data_type=template.data_type,
                is_virtual=is_virtual,
                is_pass_by_value=template.is_pass_by_value,
                name=name,
            )
        # Element strides via the DLPack protocol (classic tensor_like reads the
        # DLPack capsule). torch's .stride() is element units, but e.g. CuPy
        # exposes byte-unit .strides — so normalize any non-torch DLPack object
        # through torch.from_dlpack first.
        if hasattr(template, "stride") and callable(getattr(template, "stride", None)):
            dim = list(template.shape)
            stride = list(template.stride())
        else:
            try:
                import torch as _torch

                _view = _torch.from_dlpack(template)
                dim = list(_view.shape)
                stride = list(_view.stride())
            except Exception:  # noqa: BLE001 — no torch / exotic dlpack: assume dense
                dim = list(template.shape)
                stride = _row_major_stride(dim)

        data_type = None
        try:
            import cudnn.datatypes

            data_type = cudnn.datatypes._torch_to_cudnn_data_type(template.dtype)
        except Exception:
            pass

        is_pbv = bool(getattr(getattr(template, "device", None), "type", None) == "cpu")
        return self.tensor(dim=dim, stride=stride, data_type=data_type, is_virtual=is_virtual, is_pass_by_value=is_pbv, name=name)

    def tensor_scalar(self, value: Any, scalar_type: Any, name: str = "") -> Tensor:
        """Create a pass-by-value scalar tensor (classic tensor_scalar parity)."""
        if not name:
            name = f"scalar_{len(self._tensors)}"
        t = Tensor(
            name=name,
            dim=[1, 1, 1, 1],
            stride=[1, 1, 1, 1],
            is_pass_by_value=True,
            pass_by_value=value,
            scalar_type=scalar_type,
            uid=self._alloc_uid(),
        )
        self._register_tensor(t)
        return t

    def _check_mutable(self, what: str) -> None:
        if self._frozen:
            raise RuntimeError(f"cannot {what} after lowering/planning — the graph is frozen (planning is one-shot; build a new graph)")
        # a mutation while merely validated (python-engine graphs stay mutable
        # until planning) must re-validate later — never run on stale inference
        self._is_validated = False

    def _freeze(self) -> None:
        """Freeze the ENTIRE public graph surface (not just the fluent API).

        Called at lowering and at planning, whichever happens first. After
        this, every mutation path raises: fluent setters and op builders (via
        _check_mutable), attribute writes on Tensor/Node/GraphContext (their
        __setattr__ guards), dict writes on node.inputs/outputs/params
        (MappingProxy), and in-place list mutation of dim/stride (tuples).
        The inspection surface stays fully readable for engines."""
        if self._frozen:
            return
        from types import MappingProxyType

        for node in self._nodes:
            node.inputs = MappingProxyType(dict(node.inputs))
            node.outputs = MappingProxyType(dict(node.outputs))
            node.params = MappingProxyType(dict(node.params))
            node._frozen = True
        for t in self._tensor_by_uid.values():
            t.dim = tuple(t.dim) if t.dim else t.dim
            t.stride = tuple(t.stride) if t.stride else t.stride
            t._frozen = True
        self._context._frozen = True
        self._frozen = True

    def _rename_tensor(self, t: Tensor, name: str) -> None:
        """Atomic rename keeping the name index coherent. Classic parity: names
        are labels, so renaming ONTO an existing label is legal — the name just
        becomes ambiguous and leaves the unique-name index."""
        if name == t.name:
            return
        # NOT freeze-guarded: names are labels (classic allows renaming after
        # build — the lowered graph already carries the old label, and labels
        # have no execution semantics).
        if self._tensors.get(t.name) is t:
            del self._tensors[t.name]
        object.__setattr__(t, "name", name)  # label write is exempt from the freeze
        if name in self._tensors or name in self._ambiguous_names:
            self._tensors.pop(name, None)
            self._ambiguous_names.add(name)
        else:
            self._tensors[name] = t

    def _reuid_tensor(self, t: Tensor, uid: int) -> None:
        """Atomic re-uid keeping indexes/bindings coherent.

        Classic parity: classic tensors have NO uid until set_uid, while the IR
        assigns eagerly — so a user set_uid may land on an auto-assigned uid.
        The user wins: the auto holder is silently renumbered (auto uids are
        internal until lowering). Two USER-assigned uids colliding is an error.
        """
        if uid == t.uid:
            if not self._frozen:  # same-value set_uid is a no-op (classic allows it anytime)
                t.uid_assigned = True
            return
        self._check_mutable("re-uid a tensor")
        holder = self._tensor_by_uid.get(uid)
        if holder is not None:
            if holder.uid_assigned:
                raise ValueError(f"uid {uid} is already user-assigned to tensor {holder.name!r}")
            fresh = self._alloc_uid()  # renumber the auto holder
            self._tensor_by_uid[fresh] = holder
            if holder.uid in self._data_bindings:
                self._data_bindings[fresh] = self._data_bindings.pop(holder.uid)
            holder.uid = fresh
        self._tensor_by_uid.pop(t.uid, None)
        if t.uid in self._data_bindings:
            self._data_bindings[uid] = self._data_bindings.pop(t.uid)
        t.uid = uid
        t.uid_assigned = True
        self._reserved_uids.add(uid)
        self._tensor_by_uid[uid] = t

    def _alloc_uid(self) -> int:
        # Skip uids the user reserved via tensor(uid=...) — the Python IR owns
        # the whole uid namespace (see the uid-ownership note in _lower_to_cpp).
        while self._next_uid in self._reserved_uids:
            self._next_uid += 1
        uid = self._next_uid
        self._next_uid += 1
        return uid

    # Classic C++ auto-names op outputs "<node>::<OUTPUT_ENUM>" (graph_interface.h
    # output_tensor calls). Ports whose name differs from the classic enum are
    # mapped here so canonical names (wrapper.Graph lookups, JSON dumps) match.
    _CLASSIC_OUT_SUFFIX = {
        "inv_var": "INV_VARIANCE",
        "mean": "MEAN",
        "next_running_mean": "NEXT_RUNNING_MEAN",
        "next_running_var": "NEXT_RUNNING_VAR",
        "DScale": "DSCALE",
        "DBias": "DBIAS",
    }

    def _get_name(self, op: str, name: str) -> str:
        self._check_mutable(f"add a {op} op")
        if name:
            return name
        count = self._node_count.get(op, 0)
        self._node_count[op] = count + 1
        return f"{op}.{count}"

    def _make_output(self, name: str) -> Tensor:
        """Create a virtual output tensor. data_type is left unset: validate()'s
        inference assigns io/intermediate by the FINAL virtual state (classic
        semantics — a user set_output(True) without set_data_type gets io)."""
        return Tensor(
            name=name,
            is_virtual=True,
            uid=self._alloc_uid(),
        )

    def _register_tensor(self, t: Tensor) -> None:
        t.owner = weakref.ref(self)
        # Classic parity: names are debug LABELS — duplicates are legal
        # (pycudnnTest builds two 'weight' tensors). uid is the identity; the
        # name index serves only names that remain unique, and name-keyed
        # lookups on an ambiguous name raise instead of guessing.
        if t.name in self._tensors or t.name in self._ambiguous_names:
            self._tensors.pop(t.name, None)
            self._ambiguous_names.add(t.name)
        else:
            self._tensors[t.name] = t
        self._tensor_by_uid[t.uid] = t

    def _ensure_tensor(self, arg: Any, name: str = "") -> Tensor:
        """Convert arg to a Tensor descriptor if it isn't one already.

        If arg is a framework tensor (torch, jax, cupy, etc.), creates a
        descriptor via tensor_like() and stores the data binding for execute().
        """
        if isinstance(arg, Tensor):
            return arg
        desc = self.tensor_like(arg, name=name)
        self._data_bindings[desc.uid] = arg
        return desc

    # =========================================================================
    # Operations
    # =========================================================================

    def matmul(
        self,
        A: Any,
        B: Any,
        compute_data_type: Any = None,
        padding: float = 0.0,
        name: str = "",
    ) -> Tensor:
        """Matrix multiplication: C = A @ B.

        A and B can be Tensor descriptors or framework tensors (torch, jax, etc.).
        """
        name = self._get_name("matmul", name)
        A = self._ensure_tensor(A, name=f"{name}::A")
        B = self._ensure_tensor(B, name=f"{name}::B")

        node = Node(name, NodeType.MATMUL, compute_data_type or self._context.compute_data_type)
        node.inputs["A"] = A
        node.inputs["B"] = B
        node.params["padding"] = padding

        C = self._make_output(f"{name}::C")
        node.outputs["C"] = C
        self._register_tensor(C)

        self._nodes.append(node)
        return C

    # ---- Pointwise ops ------------------------------------------------------
    # ``params["mode"]`` is the op kind == the C++ pygraph method name (the
    # pointwise_mode enum is not exposed to Python, and the method name IS the
    # canonical semantic name), so lowering is a direct getattr dispatch — no
    # mode<->method mapping table to maintain. Extra scalar attributes
    # (negative_slope / clips / swish_beta / axis) live in params and are
    # forwarded at lowering; ops that take them get explicit builders below,
    # the uniform rest are generated from _POINTWISE_TENSOR_ARGS (the table
    # mirrors the pybind signatures — tensor-argument names per op — so both
    # positional and the classic keyword call styles work).

    _POINTWISE_TENSOR_ARGS: "dict[str, tuple]" = {
        # unary
        **{
            op: ("input",)
            for op in (
                "abs",
                "ceil",
                "cos",
                "elu",
                "erf",
                "exp",
                "floor",
                "gelu",
                "gelu_approx_tanh",
                "identity",
                "log",
                "logical_not",
                "neg",
                "reciprocal",
                "rsqrt",
                "sigmoid",
                "sin",
                "softplus",
                "sqrt",
                "tan",
                "tanh",
            )
        },
        # binary
        **{op: ("a", "b") for op in ("add", "add_square", "div", "logical_and", "logical_or", "mul", "sub")},
        **{op: ("input0", "input1") for op in ("max", "min", "mod", "pow")},
        **{op: ("input", "comparison") for op in ("cmp_eq", "cmp_ge", "cmp_gt", "cmp_le", "cmp_lt", "cmp_neq")},
        "bias": ("input", "bias"),
        "scale": ("input", "scale"),
        # backward (loss, input) -> dinput
        **{
            op: ("loss", "input")
            for op in (
                "elu_backward",
                "gelu_approx_tanh_backward",
                "gelu_backward",
                "sigmoid_backward",
                "softplus_backward",
                "tanh_backward",
            )
        },
        # ternary
        "binary_select": ("input0", "input1", "mask"),
    }
    # scalar attributes forwarded from params to the C++ call at lowering
    _POINTWISE_EXTRA_PARAMS = ("negative_slope", "lower_clip", "upper_clip", "swish_beta", "axis")

    def _pointwise(self, mode: str, inputs: list, name: str, compute_data_type: Any = None, extra_params: Optional[dict] = None) -> Tensor:
        """Internal helper for pointwise ops. ``mode`` == C++ pygraph method name."""
        inputs = [self._ensure_tensor(t, name=f"{name}::IN_{i}") for i, t in enumerate(inputs)]
        node = Node(name, NodeType.POINTWISE, compute_data_type or self._context.compute_data_type)
        node.params["mode"] = mode
        if extra_params:
            node.params.update({k: v for k, v in extra_params.items() if v is not None})
        for i, t in enumerate(inputs):
            node.inputs[f"IN_{i}"] = t

        out = self._make_output(f"{name}::OUT_0")
        node.outputs["OUT_0"] = out
        self._register_tensor(out)

        self._nodes.append(node)
        return out

    # Pointwise ops with extra scalar attributes: explicit builders.

    def relu(
        self, input: Any, negative_slope: Any = None, lower_clip: Any = None, upper_clip: Any = None, name: str = "", compute_data_type: Any = None
    ) -> Tensor:
        """ReLU (optionally leaky via negative_slope, and/or clipped)."""
        return self._pointwise(
            "relu", [input], self._get_name("relu", name), compute_data_type, dict(negative_slope=negative_slope, lower_clip=lower_clip, upper_clip=upper_clip)
        )

    def leaky_relu(self, input: Any, negative_slope: Any, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Leaky ReLU."""
        return self._pointwise("leaky_relu", [input], self._get_name("leaky_relu", name), compute_data_type, dict(negative_slope=negative_slope))

    def swish(self, input: Any, swish_beta: Any = None, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Swish / SiLU."""
        return self._pointwise("swish", [input], self._get_name("swish", name), compute_data_type, dict(swish_beta=swish_beta))

    def gen_index(self, input: Any, axis: int, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Generate index along an axis."""
        return self._pointwise("gen_index", [input], self._get_name("gen_index", name), compute_data_type, dict(axis=axis))

    def relu_backward(
        self, loss: Any, input: Any, negative_slope: Any = None, lower_clip: Any = None, upper_clip: Any = None, name: str = "", compute_data_type: Any = None
    ) -> Tensor:
        """ReLU backward."""
        return self._pointwise(
            "relu_backward",
            [loss, input],
            self._get_name("relu_backward", name),
            compute_data_type,
            dict(negative_slope=negative_slope, lower_clip=lower_clip, upper_clip=upper_clip),
        )

    def leaky_relu_backward(self, loss: Any, input: Any, negative_slope: Any, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Leaky ReLU backward."""
        return self._pointwise(
            "leaky_relu_backward", [loss, input], self._get_name("leaky_relu_backward", name), compute_data_type, dict(negative_slope=negative_slope)
        )

    def swish_backward(self, loss: Any, input: Any, swish_beta: Any = None, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Swish backward."""
        return self._pointwise("swish_backward", [loss, input], self._get_name("swish_backward", name), compute_data_type, dict(swish_beta=swish_beta))

    # NOTE: reduction / block-scale / MoE / conv / norms / structural ops are all
    # declared in _STRUCTURED_OPS (module tail) — one table entry per op, one
    # generic lowering branch. Only ops whose call shape doesn't fit the table
    # (matmul's positional ergonomics, sdpa's conditional kwargs) stay explicit.

    # NOTE: the sdpa family (sdpa / sdpa_backward / sdpa_fp8 / sdpa_mxfp8 /
    # sdpa_fp8_backward / sdpa_mxfp8_backward) is declared in _CAPTURED_OPS
    # (module tail): kwargs are captured generically — tensors become named
    # ports (port == C++ kwarg), scalars/enums/callbacks go to params verbatim,
    # dropout tuples are flattened per element — and lowering forwards them
    # verbatim, so the full C++ kwarg surface (~130 args across variants) is
    # supported without hand-mirroring each argument.

    # =========================================================================
    # Inspection
    # =========================================================================

    @property
    def nodes(self) -> List[Node]:
        """All nodes in the graph (a copy — the graph's own list is not a
        public mutation path)."""
        return list(self._nodes)

    @property
    def tensors(self) -> Dict[str, Tensor]:
        """All tensors by name (a copy — see nodes)."""
        return dict(self._tensors)

    @property
    def context(self) -> GraphContext:
        """Graph context."""
        return self._context

    def find_tensor(self, name_or_uid: Union[str, int]) -> Optional[Tensor]:
        """Find tensor by name or UID."""
        if isinstance(name_or_uid, int):
            return self._tensor_by_uid.get(name_or_uid)
        if name_or_uid in self._ambiguous_names:
            raise ValueError(f"tensor name {name_or_uid!r} is ambiguous (duplicate labels are legal; look up by uid or Tensor)")
        return self._tensors.get(name_or_uid)

    def get_node(self, name: str) -> Optional[Node]:
        """Find node by name."""
        return next((n for n in self._nodes if n.name == name), None)

    def get_inputs(self) -> List[Tensor]:
        """Get non-virtual input tensors."""
        produced = {t.uid for n in self._nodes for t in n.outputs.values() if t}
        return [t for t in self._tensors.values() if not t.is_virtual and t.uid not in produced]

    def get_outputs(self) -> List[Tensor]:
        """Get non-virtual output tensors."""
        return [t for t in self._tensors.values() if not t.is_virtual and any(t.uid == o.uid for n in self._nodes for o in n.outputs.values() if o)]

    def inspect(self) -> Dict[str, Any]:
        """Return graph structure for inspection."""
        return {
            "context": {
                "io_data_type": str(self._context.io_data_type),
                "compute_data_type": str(self._context.compute_data_type),
            },
            "nodes": [
                {
                    "name": n.name,
                    "type": n.node_type.name,
                    "inputs": {k: v.name for k, v in n.inputs.items()},
                    "outputs": {k: v.name for k, v in n.outputs.items()},
                    "params": n.params,
                }
                for n in self._nodes
            ],
            "tensors": {
                name: {"dim": t.dim, "stride": t.stride, "dtype": str(t.data_type), "is_virtual": t.is_virtual, "uid": t.uid}
                for name, t in self._tensors.items()
            },
        }

    # =========================================================================
    # Build & Execute
    # =========================================================================

    def validate(self) -> None:
        """Validate graph and infer properties.

        Classic parity: op outputs stay VIRTUAL unless the user marks them
        with set_output(True). A leaf output is NOT auto-marked — discarding
        an op result (e.g. the Stats of a training SDPA) is legal classic
        usage, and auto-marking it would make its uid required in the variant
        pack.
        """
        for node in self._nodes:
            node.infer_properties(self._context)
            # Table-driven shape inference, topologically: builder-time infer
            # only sees graph-input dims; chained ops (e.g. conv on a virtual
            # relu output) get their output dims here, once inputs are known.
            spec_entry = _STRUCTURED_BY_TYPE.get(node.node_type) or _CAPTURED_BY_TYPE.get(node.node_type)
            if spec_entry:
                _, spec = spec_entry
                infer = spec.get("infer", {})
                for oport, out_t in node.outputs.items():
                    if out_t is not None and not out_t.dim:
                        try:
                            d = infer.get(oport, lambda n: None)(node)
                        except Exception:  # noqa: BLE001 — best-effort
                            d = None
                        if d:
                            out_t.dim = list(d)
                            out_t.stride = _row_major_stride(out_t.dim)
            node.validate()
        for t in self._tensors.values():
            if t.dim and not t.stride:  # classic: stride optional, row-major inferred
                t.stride = _row_major_stride(t.dim)
            if not t.is_pass_by_value:
                t.validate()
        self._is_validated = True
        # Classic parity: with no python engines registered, C++ validation
        # happens HERE — tests catch cudnnGraphNotSupportedError around
        # graph.validate() (unsupported configs must skip, not fail later).
        if not self._backends and self._lowered_graph is None:
            self._lowered_graph = self._lower_to_cpp()
            self._lowered_graph.validate()
            self._verify_uid_ownership()

    def build_operation_graph(self) -> None:
        """Validate the graph; lower to C++ when no python engines are registered.

        Backend selection is deferred to create_execution_plans() (the Router
        stage). With python engines registered, nothing is lowered here (a graph
        routed to a python engine never touches C++). Without them — the classic
        sequencing — lowering happens now, so plan-configuration and query
        methods (deselect_engines, get_engine_count, ...) work between
        build_operation_graph() and create_execution_plans(), exactly as on the
        classic API (they delegate to the lowered C++ graph via __getattr__).
        """
        if not self._is_validated:
            self.validate()
        if self._lowered_graph is not None and not self._cpp_bog_done:
            self._lowered_graph.build_operation_graph()
            self._cpp_bog_done = True
            self._sync_ir_shapes_from_backend()

    def _sync_ir_shapes_from_backend(self) -> None:
        """After the backend's shape/layout inference (build_operation_graph),
        reflect the REAL dim/stride back into the IR tensors. The IR's own
        inferred strides are provisional row-major; the backend applies
        classic per-op layout inference (channels-last conv etc.), and
        consumers of the IR (wrapper.Graph buffer allocation, engines,
        introspection) must see the layout that will actually execute."""
        for ir_uid, cpp_t in self._cpp_tensors.items():
            ir = self._tensor_by_uid.get(ir_uid)
            if ir is None:
                continue
            try:
                d, st = cpp_t.get_dim(), cpp_t.get_stride()
            except Exception:  # noqa: BLE001 — some tensors have no dims (scalars)
                continue
            if d:
                object.__setattr__(ir, "dim", tuple(d))  # sealed (graph is frozen)
            if st:
                object.__setattr__(ir, "stride", tuple(st))

    def create_execution_plans(self, heuristics: Optional[List] = None) -> None:
        """Build the ranked execution-plan list (the dispatch stage).

        The Router returns one flat list of PlanConfig(engine_id, knobs) mixing
        python engines (reserved id region) and the backend side, in one shared
        engine-id space. Nothing is lowered here — a plan is built lazily when
        selected. ``_plan_index`` selects which plan runs (default 0, the
        highest-ranked); cuDNN heuristic modes are carried on the backend plan's
        knobs.

        Args:
            heuristics: cuDNN heuristic modes, carried to the backend plan.
        """
        if not self._is_validated:
            self.validate()

        from .engines.router import default_router

        # One-shot planning (classic conformance: the C++ graph never supported
        # re-planning — a second call there appends plans by accident, and no
        # user re-plans). Plan once; to plan differently, build a new graph
        # (IR construction is microseconds). Autotune re-selects WITHIN this
        # plan set via select_plan(). Explicit state flag, not an
        # is-the-list-nonempty proxy.
        if self._planning_done:
            raise RuntimeError(
                "create_execution_plans() was already called on this graph; planning is one-shot — build a new graph to re-plan, or use select_plan() to switch plans"
            )
        router = self._router or default_router
        plans = router.plan(self, self._backends)
        # Validate the FINAL router output (a custom Router must not bypass
        # registration): python entries must name registered engines; the only
        # non-python entry allowed is ONE backend delegating sentinel.
        from .engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID, is_python_engine

        registered = {e.engine_id for e in self._backends}
        if not plans:
            raise ValueError("router returned an empty plan list — there is no legal empty planning state (return the backend delegating entry at minimum)")
        n_cudnn = 0
        for cfg in plans:
            if is_python_engine(cfg.engine_id):
                if cfg.engine_id not in registered:
                    raise ValueError(f"router produced a plan for unregistered engine_id {cfg.engine_id}")
            elif cfg.engine_id == BACKEND_HEURISTIC_ENGINE_ID:
                n_cudnn += 1
            else:
                raise ValueError(f"router produced a plan with invalid engine_id {cfg.engine_id}")
        if n_cudnn > 1:
            raise ValueError("router produced more than one backend delegating entry")
        self._plans = plans
        self._planning_done = True
        self._freeze()  # plans reference the graph as-is: no mutation from here
        self._plan_index = 0
        self._backend_heuristics = heuristics  # applied when a backend plan is built
        # Classic sequencing: if the graph was already lowered (no python
        # engines -> build_operation_graph lowered eagerly) and the selected
        # plan is the backend one, create the C++ plans now.
        if self.selected_engine is None and self._lowered_graph is not None:
            self._lower_backend_plan()

    def _has_backend_plan(self) -> bool:
        from .engines.engine_ids import BACKEND_HEURISTIC_ENGINE_ID

        return any(cfg.engine_id == BACKEND_HEURISTIC_ENGINE_ID for cfg in self._plans)

    def get_execution_plan_count(self) -> int:
        """Classic passthrough, ALWAYS: the cuDNN backend's plan count for this
        graph (its plan list is discovered per graph from the lowered C++ graph
        and addressed via the classic ``build_plan_at_index`` /
        ``execute_plan_at_index`` / ``get_workspace_size_plan_at_index`` APIs).
        The semantics never depend on whether python engines are registered.

        The ROUTED plan list (the Router's entries: python plans + at most one
        backend delegating entry) is a separate index space: ``graph.plans``,
        selected with ``select_plan()``. Its indices are stable — the cuDNN
        entry is one index forever and never expands into this count.
        """
        if self._planning_done:
            if not self._has_backend_plan():
                raise RuntimeError(
                    "this graph's Router produced python plans only (no backend entry), so there are no backend plans — the routed plan list is graph.plans / select_plan()"
                )
            self._lower_backend_plan()  # backend plans exist on demand (one-shot)
            return self._lowered_graph.get_execution_plan_count()
        if self._lowered_graph is not None:
            # classic pre-planning sequencing: delegate, C++ reports its state
            return self._lowered_graph.get_execution_plan_count()
        return 0  # classic: an unplanned graph has zero plans (not an error)

    def select_plan(self, index: int) -> "pygraph":
        """Pick a ROUTED plan entry: the index is into ``graph.plans`` (the
        Router's entries — stable, never shifted by backend lowering). Backend
        sub-plans are a separate space, selected via the classic at-index APIs
        (see get_execution_plan_count)."""
        if not self._planning_done:
            raise RuntimeError("call create_execution_plans() before select_plan()")
        if not 0 <= index < len(self._plans):
            raise IndexError(f"plan index {index} out of range for {len(self._plans)} routed plan(s) (graph.plans)")
        self._plan_index = index
        self._is_built = False
        return self

    def _resolve_stream(self, handle: Any) -> Any:
        """Stream for a supplied handle (classic set_stream semantics). A failed
        query on a SUPPLIED handle is a correctness error and raises — never a
        silent fall-back to another stream. No handle -> None (the engine must
        resolve deterministically from its framework, e.g. torch current stream)."""
        if handle is None:
            return None
        import cudnn

        return cudnn.get_stream(handle)

    def _build_context(self, handle: Any = None) -> Any:
        from .engines.base import ExecutionContext

        h = handle if handle is not None else self._handle
        return ExecutionContext(handle=h, stream=self._resolve_stream(h))

    def _verify_uid_ownership(self) -> None:
        # Verify the uid-ownership invariant (see _lower_to_cpp): every C++
        # tensor must carry exactly its IR uid. An assertion — not a silent
        # translation — so a lowering path that forgets to push a uid fails
        # loudly in tests instead of mis-binding buffers (a swapped
        # multi-output pairing writes past the smaller buffer: corruption).
        for ir_uid, cpp_t in self._cpp_tensors.items():
            cpp_uid = cpp_t.get_uid()
            if cpp_uid != ir_uid:
                raise RuntimeError(f"uid ownership violated: IR tensor uid {ir_uid} lowered to C++ uid {cpp_uid} — a lowering path failed to push the uid")

    def _lower_backend_plan(self) -> None:
        """Lower to C++ (if not already) and create the backend plans (once)."""
        import cudnn

        if self._lowered_graph is None:
            self._lowered_graph = self._lower_to_cpp()
            self._lowered_graph.validate()
            self._verify_uid_ownership()
        if not self._cpp_bog_done:
            self._lowered_graph.build_operation_graph()
            self._cpp_bog_done = True
            self._sync_ir_shapes_from_backend()
        if not self._cpp_plans_created:
            heur = self._backend_heuristics or [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
            self._lowered_graph.create_execution_plans(heur)
            self._cpp_plans_created = True

    def check_support(self) -> None:
        """Check the selected plan's engine supports the graph.

        A python plan re-affirms its engine's check_support() (already passed
        when the Router included it); a backend plan lowers and checks C++ support.
        """
        eng = self.selected_engine
        if eng is not None:
            eng.check_support(self)
            return
        if self._lowered_graph is None:
            self._lower_backend_plan()
        self._lowered_graph.check_support()

    def build_plans(self, *args) -> None:
        """Finalize the selected plan. A python plan compiles HERE (once per
        graph/plan; the CompiledPlan is cached on the graph and reused across
        executions). The classic optional build_plan_policy passes through on
        the backend path."""
        eng = self.selected_engine
        if eng is not None:
            if self._plan_index not in self._compiled_plans:
                self._compiled_plans[self._plan_index] = eng.build_plan(self, self._selected_plan_config, self._build_context())
        if eng is None:
            if self._lowered_graph is None or not self._cpp_plans_created:
                self._lower_backend_plan()
            self._lowered_graph.build_plans(*args)
        self._is_built = True

    def build(self, heuristics: Optional[List] = None) -> None:
        """Convenience: validate -> build_operation_graph -> create_execution_plans
        -> check_support -> build_plans, in sequence."""
        if not self._is_validated:
            self.validate()

        self.build_operation_graph()
        if not self._planning_done:  # never silently re-plan: preserves select_plan()
            self.create_execution_plans(heuristics)
        self.check_support()
        self.build_plans()

    def get_workspace_size(self, *args, **kwargs) -> int:
        """Workspace bytes for the selected plan. Classic overloads (handle /
        dynamic-shape overrides) pass through on the backend path."""
        if not self._is_built:
            raise RuntimeError("Call build() first")

        if self.selected_engine is not None:
            if args or kwargs:
                raise NotImplementedError("dynamic workspace-query overrides are not supported by python plans")
            return self._compiled_plans[self._plan_index].get_workspace_size()

        return self._lowered_graph.get_workspace_size(*args, **kwargs)

    def execute(
        self,
        tensor_dict: Dict[Union[str, int, Tensor], Any],
        workspace: Any = None,
        handle: int = None,
        override_uids: Any = None,
        override_shapes: Any = None,
        override_strides: Any = None,
    ) -> None:
        """Execute the selected plan.

        Both python engines and the backend path write results directly into the
        caller-provided output tensors (in-place). Automatically calls build()
        if it hasn't run yet. Dispatch is a single check on the plan's engine id.

        Args:
            tensor_dict: Dict mapping tensors (by Tensor, name, or uid) to data.
                         Must include both input and output tensors.
            workspace: Workspace buffer (ignored by python engines)
            handle: cuDNN handle (ignored by python engines)
            override_uids/shapes/strides: dynamic-shape overrides (backend path)
        """
        if not self._is_built:
            # Auto-build. When a python plan will run and the caller supplied a
            # handle HERE, the JIT compile must see it — plan first, then let
            # the python branch below compile with the caller's context instead
            # of running the generic build (which only knows the graph handle).
            if not self._planning_done:
                self.create_execution_plans()
            if self.selected_engine is None:
                self.build()

        # Start with auto-bound inputs, then overlay user-provided (user wins)
        uid_to_data = dict(self._data_bindings)
        for key, data in tensor_dict.items():
            if key is None:
                continue  # classic API tolerates None keys (optional tensors)
            if isinstance(key, Tensor):
                uid = key.uid
            elif isinstance(key, str):
                if key in self._ambiguous_names:
                    raise ValueError(f"tensor name {key!r} is ambiguous (duplicate labels); key the variant pack by uid or Tensor")
                uid = self._tensors[key].uid
            elif isinstance(key, int):
                uid = key
            else:  # a lowered C++ tensor (advanced/interop) — trust its uid
                uid = key.get_uid()
            uid_to_data[uid] = data

        eng = self.selected_engine
        if eng is not None:  # python engine (plan id in the reserved region)
            from .engines.base import ExecutionContext

            h = handle if handle is not None else self._handle
            ctx = ExecutionContext(
                handle=h,
                stream=self._resolve_stream(h),
                workspace=workspace,
                override_uids=override_uids,
                override_shapes=override_shapes,
                override_strides=override_strides,
            )
            if self._plan_index not in self._compiled_plans:
                # compile with the CALLER's context (execute-supplied handle
                # and its stream reach the JIT build)
                self._compiled_plans[self._plan_index] = eng.build_plan(self, self._selected_plan_config, ctx)
                self._is_built = True
            self._compiled_plans[self._plan_index].execute(self, uid_to_data, ctx)
            return

        # cuDNN execution path (plan id < PYTHON_ENGINE_ID_BASE). Variant-pack
        # keys are IR uids — identical to the C++ uids by construction (the IR
        # owns the uid namespace and lowering pushes every uid explicitly).
        from .datatypes import _is_torch_tensor

        def _ptr(d):
            if type(d) is int:
                return d
            if _is_torch_tensor(d) or hasattr(d, "data_ptr"):
                return d.data_ptr()
            import cudnn

            return cudnn._pybind_module._get_data_ptr(d)  # dlpack fallback

        var_pack = {uid: _ptr(d) for uid, d in uid_to_data.items()}
        ws_ptr = _ptr(workspace) if workspace is not None else 0
        self._lowered_graph._execute(var_pack, ws_ptr, handle, override_uids, override_shapes, override_strides)

    def __getattr__(self, name: str):
        # Plan-configuration and query methods (deselect_engines,
        # get_engine_and_knobs_at_index, key, populate_cuda_graph, ...) operate
        # on the lowered C++ graph — delegate to it. Only reached when normal
        # attribute lookup fails, i.e. for names this class doesn't define.
        lowered = self.__dict__.get("_lowered_graph")
        if lowered is not None and not name.startswith("_") and hasattr(lowered, name):
            return getattr(lowered, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
            + ("" if lowered is not None else " (graph not lowered yet — call build_operation_graph() first)")
        )

    def __repr__(self) -> str:
        if self._lowered_graph is not None:
            return repr(self._lowered_graph)  # classic JSON dump
        import json

        return json.dumps(self.inspect(), default=str, indent=2)

    @property
    def engine(self) -> Optional["BaseEngine"]:
        """The python engine for the selected plan, or None for the backend path.
        Populated after create_execution_plans()."""
        return self.selected_engine

    def serialize(self):
        """Serialize the graph (classic passthrough).

        Returns the C++ binding's serialized form unchanged; C++
        ``deserialize`` accepts exactly this form back.
        """
        if self._lowered_graph is None:
            # Serialization is the cuDNN graph format by definition — lower on
            # demand (independent of which plan is selected for execution).
            self.validate()
            if self._lowered_graph is None:  # python engines registered
                self._lowered_graph = self._lower_to_cpp()
                self._lowered_graph.validate()
                self._verify_uid_ownership()
        return self._lowered_graph.serialize()

    def deserialize(self, *args, **kwargs) -> None:
        """Deserialize a graph (classic passthrough: (data) or (handle, data,
        enforce_precompiled=...)). Replaces this graph's lowered C++ graph."""
        if self._lowered_graph is None:
            import cudnn

            if self._nodes:  # deserializing into a built-up graph: lower it
                self.validate()
                self._lowered_graph = self._lower_to_cpp()
            else:  # fresh container (classic usage): empty C++ graph
                self._lowered_graph = cudnn._pybind_module.backend_graph()
        self._lowered_graph.deserialize(*args, **kwargs)
        self._is_built = True

    @classmethod
    def from_serialized(cls, data, handle: Optional[int] = None, **kwargs) -> "pygraph":
        """Create a pygraph from serialized data.

        This is a convenience method that creates a minimal graph and deserializes into it.

        Args:
            data: Serialized graph data (from serialize()).
            handle: Optional cuDNN handle for AoT compilation.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            pygraph: Deserialized graph ready for execution.
        """
        import cudnn

        # Create a new graph with a fresh C++ graph
        graph = cls(**kwargs)
        graph._lowered_graph = cudnn._pybind_module.backend_graph(
            io_data_type=graph._context.io_data_type,
            intermediate_data_type=graph._context.intermediate_data_type,
            compute_data_type=graph._context.compute_data_type,
        )

        if handle is not None:
            graph._lowered_graph.deserialize(handle, data)
        else:
            graph._lowered_graph.deserialize(data)
        graph._is_built = True
        return graph

    def _lower_to_cpp(self) -> Any:
        """Lower Python graph to C++ (the internal ``_pybind_module.backend_graph``)."""
        import cudnn
        from .datatypes import _library_type  # torch dtype -> cudnn enum (classic parity)

        self._freeze()  # the lowered graph mirrors the IR from here on

        # The C++ graph rejects None (wants the enum). io_data_type may be unset
        # (block-scale tensors carry their own dtypes), but intermediate/compute
        # default to FLOAT — matching cudnn.graph() — so cuDNN can infer virtual
        # (intermediate) tensor dtypes during build.
        pg_kwargs = dict(self._cpp_graph_kwargs)
        if self._context.io_data_type is not None:
            pg_kwargs["io_data_type"] = _library_type(self._context.io_data_type)
        pg_kwargs["intermediate_data_type"] = _library_type(self._context.intermediate_data_type or cudnn.data_type.FLOAT)
        pg_kwargs["compute_data_type"] = _library_type(self._context.compute_data_type or cudnn.data_type.FLOAT)
        if self._handle is not None:
            pg_kwargs["handle"] = self._handle
        graph = cudnn._pybind_module.backend_graph(**pg_kwargs)

        tensor_map: Dict[int, Any] = {}

        def lower_tensor(t: Tensor) -> Any:
            if t.uid in tensor_map:
                return tensor_map[t.uid]
            if t.pass_by_value is not None and t.scalar_type is not None:
                cpp = graph.tensor_scalar(t.pass_by_value, t.scalar_type)
                cpp.set_uid(t.uid)
                tensor_map[t.uid] = cpp
                return cpp
            mk_kwargs = dict(
                dim=t.dim,
                stride=t.stride,
                is_virtual=t.is_virtual,
                is_pass_by_value=t.is_pass_by_value,
                name=t.name,
                # Always propagate the IR uid so execute()'s variant pack (keyed
                # by IR uid) matches; otherwise cuDNN assigns its own and the
                # buffers never bind. IR uids are unique and positive.
                uid=t.uid,
            )
            if t.data_type is not None:  # else NOT_SET → cuDNN infers from the
                mk_kwargs["data_type"] = _library_type(t.data_type)  # graph intermediate default
            if t.reordering_type is not None:  # e.g. F8_128x4 for block-scale SFs
                mk_kwargs["reordering_type"] = t.reordering_type
            if t.ragged_offset is not None:
                mk_kwargs["ragged_offset"] = lower_tensor(t.ragged_offset)
                if t.ragged_offset_multiplier not in (None, 1):  # non-default only
                    mk_kwargs["ragged_offset_multiplier"] = t.ragged_offset_multiplier
            cpp = graph._make_tensor(**mk_kwargs)
            tensor_map[t.uid] = cpp
            return cpp

        def push_output_attrs(out_t: Tensor, cpp_t: Any) -> None:
            # Every attribute a user may set on an OP OUTPUT via the classic
            # setter chain must be pushed here (inputs get theirs through
            # _make_tensor kwargs in lower_tensor). Missing one corrupts
            # silently: no multiplier -> wrong GPU addresses (cudaErrorMisalignedAddress);
            # no reordering -> the backend rejects or misreads the layout.
            # dim/stride: USER-assigned only — inferred IR strides are
            # provisional row-major; the backend keeps its classic per-op
            # layout inference (channels-last conv etc.) when the user did
            # not pin one.
            # the label too: classic renames act on the SAME object the cpp
            # graph holds, so the lowered graph carries the user's name
            if out_t.name:
                cpp_t.set_name(out_t.name)
            if out_t.dim_assigned and out_t.dim:
                cpp_t.set_dim(out_t.dim)
            if out_t.stride_assigned and out_t.stride:
                cpp_t.set_stride(out_t.stride)
            if out_t.ragged_offset is not None:
                cpp_t.set_ragged_offset(lower_tensor(out_t.ragged_offset))
                if out_t.ragged_offset_multiplier not in (None, 1):
                    cpp_t.set_ragged_offset_multiplier(out_t.ragged_offset_multiplier)
            if out_t.reordering_type is not None:
                cpp_t.set_reordering_type(out_t.reordering_type)
            if not out_t.is_virtual:
                cpp_t.set_output(True)
            if out_t.data_type:
                cpp_t.set_data_type(out_t.data_type)

        for node in self._nodes:
            for t in node.inputs.values():
                if t:
                    lower_tensor(t)

            if node.node_type == NodeType.MATMUL:
                mm_kw = dict(A=tensor_map[node.inputs["A"].uid], B=tensor_map[node.inputs["B"].uid], padding=node.params.get("padding", 0.0), name=node.name)
                if node.compute_data_type is not None:
                    mm_kw["compute_data_type"] = _library_type(node.compute_data_type)
                cpp_out = graph.matmul(**mm_kw)
            elif node.node_type == NodeType.POINTWISE:
                # params["mode"] IS the C++ pygraph method name — direct
                # dispatch; scalar attributes (clips/negative_slope/...) are
                # forwarded as keywords, tensors positionally (they lead every
                # pointwise signature).
                inputs = [tensor_map[t.uid] for t in node.inputs.values()]
                extra = {k: node.params[k] for k in self._POINTWISE_EXTRA_PARAMS if k in node.params}
                if node.compute_data_type is not None:
                    extra["compute_data_type"] = _library_type(node.compute_data_type)
                cpp_out = getattr(graph, node.params["mode"])(*inputs, name=node.name, **extra)
            elif node.node_type in _CAPTURED_BY_TYPE:
                # Captured op (sdpa family): rebuild the original kwargs —
                # tensor ports (port == C++ kwarg) map through tensor_map,
                # scalar params forward verbatim, dropout reassembles from its
                # flattened elements — and call the C++ method once.
                method, spec = _CAPTURED_BY_TYPE[node.node_type]
                kw = {"name": node.name}
                if node.compute_data_type is not None:
                    kw["compute_data_type"] = _library_type(node.compute_data_type)
                for pk, pv in node.params.items():
                    if pk.startswith("_") or pk.startswith("dropout_"):
                        continue
                    # user callbacks (score_mod, ...) get a shimmed graph so
                    # closures over IR tensors keep working (see _CallbackGraphShim)
                    kw[pk] = _wrap_callback(pv, lower_tensor) if callable(pv) else pv
                for port, t in node.inputs.items():
                    if not port.startswith("dropout_"):
                        kw[port] = tensor_map[t.uid]
                for port in spec.get("out_kwargs", ()):
                    if port in node.outputs:  # classic passes these descriptors as args
                        kw[port] = lower_tensor(node.outputs[port])
                n_drop = node.params.get("_dropout_n")
                if n_drop:
                    kw["dropout"] = tuple(
                        tensor_map[node.inputs[f"dropout_{i}"].uid] if f"dropout_{i}" in node.inputs else node.params[f"dropout_{i}"] for i in range(n_drop)
                    )
                result = getattr(graph, method)(**kw)
                cpp_outs = list(result) if isinstance(result, (list, tuple)) else [result]
                for oport, cpp_t in zip(spec["outputs"], cpp_outs):
                    out_t = node.outputs.get(oport)
                    if out_t is None or cpp_t is None:
                        continue
                    tensor_map[out_t.uid] = cpp_t
                    # sdpa-family output layout is user-chosen: the C++ node
                    # REQUIRES O's dim/stride before validate (BSHD vs BHSD) —
                    # push whatever the IR carries (inferred or user-set).
                    if out_t.dim:
                        cpp_t.set_dim(out_t.dim)
                    if out_t.stride:
                        cpp_t.set_stride(out_t.stride)
                    push_output_attrs(out_t, cpp_t)
                continue
            elif node.node_type in _STRUCTURED_BY_TYPE:
                # Generic structured op (norms / reduction / block-scale / MoE /
                # conv / structural): input ports are named after the C++
                # kwargs, so lowering is kwargs assembly + one call + zipping
                # the returned tuple with the declared output ports.
                method, spec = _STRUCTURED_BY_TYPE[node.node_type]
                kw = {"name": node.name}
                if not spec.get("no_cdt") and node.compute_data_type is not None:
                    kw["compute_data_type"] = _library_type(node.compute_data_type)
                list_ports = spec.get("list_inputs", ())
                for port, t in node.inputs.items():
                    if any(port.startswith(f"{lp}_") for lp in list_ports):
                        continue  # collected below
                    kw[port] = tensor_map[t.uid]
                for lp in list_ports:
                    n = node.params.get(f"_n_{lp}", 0)
                    if n:
                        kw[lp] = [tensor_map[node.inputs[f"{lp}_{i}"].uid] for i in range(n)]
                for ak in spec.get("attrs", ()):
                    if ak in node.params:
                        kw[ak] = node.params[ak]
                result = getattr(graph, method)(**kw)
                cpp_outs = list(result) if isinstance(result, (list, tuple)) else [result]
                push_dims = spec.get("push_output_dims", False)
                for oport, cpp_t in zip(spec["outputs"], cpp_outs):
                    out_t = node.outputs.get(oport)
                    if out_t is None or cpp_t is None:
                        continue
                    tensor_map[out_t.uid] = cpp_t
                    if push_dims and out_t.dim:  # ops whose output dims cuDNN can't infer
                        cpp_t.set_dim(out_t.dim)
                        # stride only when USER-assigned: pushing the IR's
                        # provisional row-major stride into an (e.g.) NHWC
                        # graph makes the backend reject the fusion (classic
                        # infers the stride when the user sets only dims)
                        if out_t.stride_assigned and out_t.stride:
                            cpp_t.set_stride(out_t.stride)
                    push_output_attrs(out_t, cpp_t)
                continue
            else:
                continue

            # Map output
            for out_t in node.outputs.values():
                tensor_map[out_t.uid] = cpp_out
                push_output_attrs(out_t, cpp_out)

        # ---- uid ownership -------------------------------------------------
        # The Python IR owns the whole uid namespace: every IR tensor gets a uid
        # eagerly at creation (_alloc_uid, or user-specified via tensor(uid=)),
        # and lowering pushes ALL of them explicitly to C++ — inputs via
        # _make_tensor(uid=), op-created outputs/virtuals via set_uid here. The
        # C++ FE's build-time auto-assignment therefore NEVER triggers for
        # graphs built through the Python pygraph (its enumeration order is not
        # deterministic for multi-output ops, so relying on it mis-binds
        # buffers). Mixed construction — adding ops directly to the lowered C++
        # graph — is unsupported: a graph is either pure-Python or pure-C++.
        for ir_uid, cpp_t in tensor_map.items():
            cpp_t.set_uid(ir_uid)

        self._cpp_tensors = tensor_map
        return graph


def _install_pointwise_builders() -> None:
    """Generate the uniform pointwise builders from _POINTWISE_TENSOR_ARGS.

    Each builder accepts its tensors positionally OR by the classic pybind
    keyword names (e.g. ``g.bias(input=x, bias=b)``, ``g.max(input0=a,
    input1=b)``), matching the C++ pygraph API surface exactly. Ops with extra
    scalar attributes (relu / leaky_relu / swish / gen_index + backwards) have
    explicit builders on the class instead.
    """

    def make(op: str, argnames: tuple):
        def builder(self, *args, name: str = "", compute_data_type: Any = None, **kwargs):
            tensors = list(args)
            for an in argnames[len(args) :]:
                if an not in kwargs:
                    raise TypeError(f"{op}() missing tensor argument {an!r}")
                tensors.append(kwargs.pop(an))
            if len(tensors) != len(argnames) or kwargs:
                bad = kwargs or f"{len(tensors)} tensors"
                raise TypeError(f"{op}() expects tensor arguments {argnames}; got unexpected {bad}")
            return self._pointwise(op, tensors, self._get_name(op, name), compute_data_type)

        builder.__name__ = op
        builder.__qualname__ = f"pygraph.{op}"
        builder.__doc__ = f"Element-wise {op}({', '.join(argnames)})."
        return builder

    for op, argnames in pygraph._POINTWISE_TENSOR_ARGS.items():
        if not hasattr(pygraph, op):  # explicit builders (relu, ...) win
            setattr(pygraph, op, make(op, argnames))


_install_pointwise_builders()


# ---------------------------------------------------------------------------
# Structured ops, declaratively: norms, reduction, block-scale, MoE, conv, and
# the structural ops — everything except matmul (positional ergonomics) and
# sdpa (conditional kwarg assembly), which stay explicit.
#
# One table entry per op:
#   node_type          NodeType member (engines match on this)
#   inputs             ordered tensor ports == the C++ pybind kwarg names
#   list_inputs        ports taking a LIST of tensors (indexed ports + count)
#   attrs              scalar/enum/list params stored in node.params verbatim
#                      and forwarded as keywords at lowering
#   outputs            output ports, in C++ return order
#   infer              per-output IR-side shape inference (introspection; cuDNN
#                      re-infers at build) — best-effort, None on failure
#   push_output_dims   True for ops whose output dims cuDNN cannot infer
#                      (dgrad/wgrad/reduction/reshape/...): IR dims are pushed
#   no_cdt             True for bindings without a compute_data_type kwarg
#
# Builders are generated: tensors positionally or by port name, attrs by
# keyword, plus a reserved ``out_dims`` kwarg (dims list for a single output,
# or {port: dims} for several) for the ambiguous-shape ops.
# ---------------------------------------------------------------------------


def _like(port):  # output dims mirror an input port
    return lambda node: (node.inputs[port].dim if port in node.inputs else None)


def _stats_like(port, keep_axes):  # input-port dims with all but keep_axes reduced to 1
    def infer(node):
        d = node.inputs[port].dim if port in node.inputs else None
        return [x if i in keep_axes else 1 for i, x in enumerate(d)] if d else None

    return infer


def _conv_fprop_dims(node):
    x, w = node.inputs["image"].dim, node.inputs["weight"].dim
    sp = len(x) - 2
    sym = node.params.get("padding")
    pre = node.params.get("pre_padding") or sym or [0] * sp
    post = node.params.get("post_padding") or sym or [0] * sp
    stride = node.params.get("stride") or [1] * sp
    dil = node.params.get("dilation") or [1] * sp
    out = [x[0], w[0]]
    for i in range(sp):
        eff = (w[i + 2] - 1) * dil[i] + 1
        out.append((x[i + 2] + pre[i] + post[i] - eff) // stride[i] + 1)
    return out


def _conv_dgrad_dims(node):
    dy, w = node.inputs["loss"].dim, node.inputs["filter"].dim
    sp = len(dy) - 2
    sym = node.params.get("padding")
    pre = node.params.get("pre_padding") or sym or [0] * sp
    post = node.params.get("post_padding") or sym or [0] * sp
    stride = node.params.get("stride") or [1] * sp
    dil = node.params.get("dilation") or [1] * sp
    # Reverse of fprop — ambiguous for strided conv; out_dims/set_dim overrides.
    out = [dy[0], w[1]]
    for i in range(sp):
        eff = (w[i + 2] - 1) * dil[i] + 1
        out.append((dy[i + 2] - 1) * stride[i] + eff - pre[i] - post[i])
    return out


def _slice_dims(node):  # output extent of each python slice over the input dims
    d = node.inputs["input"].dim
    sls = node.params.get("slices")
    if not d or not sls:
        return None
    return [len(range(*sl.indices(int(n)))) for sl, n in zip(sls, d)]


def _moe_bwd_dweight_dims(node):
    do, tok, fto = (node.inputs[p].dim for p in ("doutput", "token", "first_token_offset"))
    return [fto[0], tok[-1], do[-1]]  # [E, H, N]


def _block_quant_scale_dims(node):
    d = list(node.inputs["input"].dim)
    bs = node.params.get("block_size")
    axis = node.params.get("axis")
    axis = len(d) - 1 if axis in (None, -1) else axis
    d[axis] = (d[axis] + bs - 1) // bs
    return d


_NORM_FWD_INFER = {"Y": _like("input"), "mean": _stats_like("input", (0,)), "inv_var": _stats_like("input", (0,))}
_NORM_BWD_INFER = {"DX": _like("input"), "DScale": _like("scale"), "DBias": _like("scale")}


def _training_phase(node):  # norm stats exist only in TRAINING forward phase
    phase = node.params.get("norm_forward_phase")
    return getattr(phase, "name", str(phase)).upper() != "INFERENCE"


_NORM_FWD_MAYBE = {"mean": _training_phase, "inv_var": _training_phase}

_STRUCTURED_OPS = {
    # ---- norms --------------------------------------------------------------
    "rmsnorm": dict(
        node_type=NodeType.RMSNORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "inv_var"),
        maybe={"inv_var": _training_phase},
        infer={"Y": _like("input"), "inv_var": _stats_like("input", (0,))},
    ),
    "rmsnorm_backward": dict(
        node_type=NodeType.RMSNORM_BWD,
        inputs=("grad", "input", "scale", "inv_variance"),
        attrs=("has_dbias",),
        outputs=("DX", "DScale", "DBias"),
        # classic rmsnorm_backward names its outputs ::Dscale/::Dbias (mixed
        # case), unlike the other norm backwards (::DSCALE/::DBIAS)
        out_suffix={"DScale": "Dscale", "DBias": "Dbias"},
        maybe={"DBias": lambda n: n.params.get("has_dbias", True) is not False},
        infer=_NORM_BWD_INFER,
    ),
    "layernorm": dict(
        node_type=NodeType.LAYERNORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "mean", "inv_var"),
        maybe=_NORM_FWD_MAYBE,
        infer=_NORM_FWD_INFER,
    ),
    "layernorm_backward": dict(
        node_type=NodeType.LAYERNORM_BWD,
        inputs=("grad", "input", "scale", "mean", "inv_variance"),
        outputs=("DX", "DScale", "DBias"),
        infer=_NORM_BWD_INFER,
    ),
    "adalayernorm": dict(
        node_type=NodeType.ADALAYERNORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "mean", "inv_var"),
        maybe=_NORM_FWD_MAYBE,
        infer=_NORM_FWD_INFER,
    ),
    "adalayernorm_backward": dict(
        node_type=NodeType.ADALAYERNORM_BWD,
        inputs=("grad", "input", "scale", "mean", "inv_variance"),
        outputs=("DX", "DScale", "DBias"),
        infer=_NORM_BWD_INFER,
    ),
    "instancenorm": dict(
        node_type=NodeType.INSTANCENORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "mean", "inv_var"),
        maybe=_NORM_FWD_MAYBE,
        infer={"Y": _like("input"), "mean": _stats_like("input", (0, 1)), "inv_var": _stats_like("input", (0, 1))},
    ),
    "instancenorm_backward": dict(
        node_type=NodeType.INSTANCENORM_BWD,
        inputs=("grad", "input", "scale", "mean", "inv_variance"),
        outputs=("DX", "DScale", "DBias"),
        infer=_NORM_BWD_INFER,
    ),
    "batchnorm": dict(
        node_type=NodeType.BATCHNORM,
        inputs=("input", "scale", "bias", "in_running_mean", "in_running_var", "epsilon", "momentum"),
        list_inputs=("peer_stats",),
        outputs=("Y", "mean", "inv_var", "next_running_mean", "next_running_var"),
        maybe={
            "next_running_mean": lambda n: "in_running_mean" in n.inputs,
            "next_running_var": lambda n: "in_running_var" in n.inputs,
        },
        infer={
            "Y": _like("input"),
            "mean": _stats_like("input", (1,)),
            "inv_var": _stats_like("input", (1,)),
            "next_running_mean": _stats_like("input", (1,)),
            "next_running_var": _stats_like("input", (1,)),
        },
    ),
    "batchnorm_inference": dict(
        node_type=NodeType.BATCHNORM_INFERENCE,
        inputs=("input", "mean", "inv_variance", "scale", "bias"),
        outputs=("Y",),
        infer={"Y": _like("input")},
    ),
    "batchnorm_backward": dict(
        node_type=NodeType.BATCHNORM_BWD,
        inputs=("grad", "input", "scale", "mean", "inv_variance"),
        list_inputs=("peer_stats",),
        outputs=("DX", "DScale", "DBias"),
        infer=_NORM_BWD_INFER,
    ),
    "genstats": dict(
        node_type=NodeType.GENSTATS,
        inputs=("input",),
        outputs=("SUM", "SQ_SUM"),
        infer={"SUM": _stats_like("input", (1,)), "SQ_SUM": _stats_like("input", (1,))},
    ),
    # ---- reduction / block-scale / MoE --------------------------------------
    "reduction": dict(
        node_type=NodeType.REDUCTION,
        inputs=("input", "group_offset"),
        attrs=("mode",),
        outputs=("OUT_0",),
        push_output_dims=True,  # cuDNN needs the reduced output dims explicitly
    ),
    "block_scale_dequantize": dict(
        node_type=NodeType.BLOCK_SCALE_DEQUANTIZE,
        inputs=("input", "descale"),
        attrs=("block_size", "is_negative_scale"),
        outputs=("OUT_0",),
        infer={"OUT_0": _like("input")},
    ),
    "block_scale_quantize": dict(
        node_type=NodeType.BLOCK_SCALE_QUANTIZE,
        inputs=("input",),
        attrs=("block_size", "axis", "transpose"),
        outputs=("Y", "scale"),
        infer={"Y": _like("input"), "scale": _block_quant_scale_dims},
    ),
    "moe_grouped_matmul": dict(
        node_type=NodeType.MOE_GROUPED_MATMUL,
        inputs=("token", "weight", "first_token_offset", "token_index", "token_ks"),
        attrs=("mode", "top_k"),
        outputs=("OUT_0",),
        infer={"OUT_0": lambda n: [1, n.inputs["token"].dim[-2], n.inputs["weight"].dim[-1]]},
    ),
    "moe_grouped_matmul_bwd": dict(
        node_type=NodeType.MOE_GROUPED_MATMUL_BWD,
        inputs=("doutput", "token", "first_token_offset"),
        outputs=("dweight",),
        infer={"dweight": _moe_bwd_dweight_dims},
        push_output_dims=True,
    ),
    # ---- convolution ---------------------------------------------------------
    "conv_fprop": dict(
        node_type=NodeType.CONV_FPROP,
        inputs=("image", "weight"),
        attrs=("padding", "pre_padding", "post_padding", "stride", "dilation", "convolution_mode"),
        outputs=("Y",),
        infer={"Y": _conv_fprop_dims},
    ),
    "conv_dgrad": dict(
        node_type=NodeType.CONV_DGRAD,
        inputs=("loss", "filter"),
        attrs=("padding", "pre_padding", "post_padding", "stride", "dilation", "convolution_mode"),
        outputs=("DX",),
        infer={"DX": _conv_dgrad_dims},
        push_output_dims=True,  # dgrad output dims are ambiguous for strided conv
    ),
    "conv_wgrad": dict(
        node_type=NodeType.CONV_WGRAD,
        inputs=("image", "loss"),
        attrs=("padding", "pre_padding", "post_padding", "stride", "dilation", "convolution_mode"),
        outputs=("DW",),
        push_output_dims=True,  # wgrad output (filter) dims are not inferable
    ),
    # ---- structural -----------------------------------------------------------
    "reshape": dict(
        node_type=NodeType.RESHAPE,
        inputs=("input",),
        attrs=("reshape_mode",),
        outputs=("OUT_0",),
        push_output_dims=True,  # target shape comes from out_dims / set_dim
        no_cdt=True,
    ),
    "slice": dict(
        node_type=NodeType.SLICE,
        inputs=("input",),
        attrs=("slices",),
        outputs=("OUT_0",),
        infer={"OUT_0": _slice_dims},
        dtype_like={"OUT_0": "input"},  # classic: slice output dtype == input's
    ),
    "transpose": dict(
        node_type=NodeType.TRANSPOSE,
        inputs=("input",),
        attrs=("permutation",),
        outputs=("OUT_0",),
        infer={"OUT_0": lambda n: ([n.inputs["input"].dim[i] for i in n.params["permutation"]] if n.inputs["input"].dim else None)},
    ),
    "concatenate": dict(
        node_type=NodeType.CONCATENATE,
        inputs=(),
        list_inputs=("inputs",),
        attrs=("axis", "in_place_index"),
        outputs=("OUT_0",),
        no_cdt=True,
    ),
    "rope": dict(
        node_type=NodeType.ROPE,
        inputs=("input", "freqs"),
        attrs=("output_scale", "rope_dim"),
        outputs=("OUT_0",),
        infer={"OUT_0": _like("input")},
    ),
    "rope_backward": dict(
        node_type=NodeType.ROPE_BWD,
        inputs=("dY", "freqs"),
        attrs=("output_scale", "rope_dim"),
        outputs=("OUT_0",),
        infer={"OUT_0": _like("dY")},
    ),
}

# node_type -> (method name, spec), for the generic lowering branch
_STRUCTURED_BY_TYPE = {spec["node_type"]: (op, spec) for op, spec in _STRUCTURED_OPS.items()}


def _install_structured_builders() -> None:
    """Generate builders for _STRUCTURED_OPS.

    Call style: tensors positionally (in declared port order) or by port name;
    attrs by keyword; ``out_dims`` sets output dims explicitly (a dims list for
    single-output ops, or {port: dims}) for shapes cuDNN cannot infer."""

    def make(op: str, spec: dict):
        input_ports = spec["inputs"]
        list_ports = spec.get("list_inputs", ())
        attr_kws = spec.get("attrs", ())
        infer = spec.get("infer", {})
        maybe = spec.get("maybe", {})

        def builder(self, *args, name: str = "", compute_data_type: Any = None, out_dims: Any = None, **kwargs):
            name_ = self._get_name(op, name)
            node = Node(name_, spec["node_type"], compute_data_type or self._context.compute_data_type)
            # classic positional order: tensor ports first, then attrs
            n_p = len(input_ports)
            if len(args) > n_p + len(attr_kws):
                raise TypeError(f"{op}() takes at most {n_p + len(attr_kws)} positional arguments ({input_ports} + {attr_kws})")
            for ak, v in zip(attr_kws, args[n_p:]):
                if ak in kwargs:
                    raise TypeError(f"{op}() got multiple values for {ak!r}")
                kwargs[ak] = v
            args = args[:n_p]
            for port, v in zip(input_ports, args):
                node.inputs[port] = self._ensure_tensor(v, name=f"{name_}::{port}")
            for port in input_ports[len(args) :]:
                v = kwargs.pop(port, None)
                if v is not None:
                    node.inputs[port] = self._ensure_tensor(v, name=f"{name_}::{port}")
            for lp in list_ports:
                vs = kwargs.pop(lp, None) or []
                for i, v in enumerate(vs):
                    node.inputs[f"{lp}_{i}"] = self._ensure_tensor(v, name=f"{name_}::{lp}_{i}")
                if vs:
                    node.params[f"_n_{lp}"] = len(vs)
            for ak in attr_kws:
                v = kwargs.pop(ak, None)
                if v is not None:
                    node.params[ak] = v
            if kwargs:
                raise TypeError(f"{op}() got unexpected arguments {sorted(kwargs)}; tensor ports are {input_ports}, attrs are {attr_kws}")
            if out_dims is not None and not isinstance(out_dims, dict):
                out_dims = {spec["outputs"][0]: out_dims}
            dtype_like = spec.get("dtype_like", {})
            outs = []
            for oport in spec["outputs"]:
                cond = maybe.get(oport)
                if cond is not None and not cond(node):
                    outs.append(None)  # classic returns None for absent outputs
                    continue
                o = self._make_output(f"{name_}::{spec.get('out_suffix', {}).get(oport) or self._CLASSIC_OUT_SUFFIX.get(oport, oport)}")
                src = dtype_like.get(oport)
                if src and src in node.inputs:
                    o.data_type = node.inputs[src].data_type
                d = (out_dims or {}).get(oport)
                if d is None:
                    try:  # best-effort IR-side inference; C++ validates at build
                        d = infer.get(oport, lambda n: None)(node)
                    except Exception:  # noqa: BLE001
                        d = None
                if d:
                    o.dim = list(d)
                    o.stride = _row_major_stride(o.dim)
                node.outputs[oport] = o
                self._register_tensor(o)
                outs.append(o)
            self._nodes.append(node)
            return outs[0] if len(outs) == 1 else list(outs)  # classic multi-output ops return a LIST

        builder.__name__ = op
        builder.__qualname__ = f"pygraph.{op}"
        builder.__doc__ = f"{op}({', '.join(input_ports)}) -> ({', '.join(spec['outputs'])})."
        return builder

    for op, spec in _STRUCTURED_OPS.items():
        setattr(pygraph, op, make(op, spec))


_install_structured_builders()


# ---------------------------------------------------------------------------
# Captured ops (the sdpa family): the kwarg surface is huge (~130 args across
# the six variants, including tensor-or-float args, dropout tuples, and
# score_mod callbacks), so builders capture ALL kwargs generically instead of
# hand-mirroring each one — tensors become named ports (port == C++ kwarg),
# everything else goes to params verbatim, dropout tuples are flattened per
# element. Lowering rebuilds the kwargs and makes one C++ call. The node stays
# first-class: engines read node.inputs["q"] / node.params["use_causal_mask"].
# ---------------------------------------------------------------------------


class _CallbackGraphShim:
    """Wraps the C++ graph handed to user callbacks (score_mod & co.) during
    lowering. Classic code passes the SAME object at build and callback time, so
    closures over user-created tensors just work; post-flip the user's closures
    capture IR Tensors while the callback receives the C++ graph. The shim
    translates any IR Tensor argument to its lowered C++ tensor at the call
    site, so existing callback code runs unchanged."""

    def __init__(self, target, lower_tensor):
        self._target = target
        self._lower = lower_tensor

    def _xlate(self, v):
        if isinstance(v, Tensor):
            # closure-captured helper tensors may not feed any node: lower on demand
            return self._lower(v)
        if isinstance(v, (list, tuple)):
            return type(v)(self._xlate(x) for x in v)
        return v

    def __getattr__(self, name):
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            return attr(*[self._xlate(a) for a in args], **{k: self._xlate(v) for k, v in kwargs.items()})

        return call


def _wrap_callback(fn, lower_tensor):
    """Wrap a user callback param (e.g. score_mod) so the C++ graph it receives
    is shimmed (see _CallbackGraphShim) and stray IR-Tensor args translate."""
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        import cudnn

        cpp_graph_t = cudnn._pybind_module.backend_graph

        def conv(v):
            if isinstance(v, cpp_graph_t):
                return _CallbackGraphShim(v, lower_tensor)
            if isinstance(v, Tensor):
                return lower_tensor(v)
            return v

        return fn(*[conv(a) for a in args], **{k: conv(v) for k, v in kwargs.items()})

    return wrapped


def _stats_expected(params):
    if params.get("generate_stats") is not None:
        return bool(params["generate_stats"])
    return not params.get("is_inference", True)


def _sdpa_o_dims(node):  # O: q dims with v's head dim
    q, v = node.inputs["q"].dim, node.inputs["v"].dim
    return list(q[:-1]) + [v[-1]]


def _sdpa_stats_dims(node):  # Stats: q dims with last dim 1
    return list(node.inputs["q"].dim[:-1]) + [1]


_AMAX = lambda node: [1, 1, 1, 1]  # noqa: E731 — fp8 amax side outputs

_CAPTURED_OPS = {
    "sdpa": dict(
        node_type=NodeType.SDPA,
        pos=("q", "k", "v"),
        outputs=("O", "Stats"),
        # kwargs whose tensors are semantically OUTPUTS of the node (the classic
        # API passes their descriptors as arguments): recorded in node.outputs so
        # engines see correct producer/consumer direction.
        out_kwargs=("rng_dump", "score_max", "score_sum_exp"),
        maybe={"Stats": _stats_expected},
        infer={"O": _sdpa_o_dims, "Stats": _sdpa_stats_dims},
    ),
    "sdpa_backward": dict(
        node_type=NodeType.SDPA_BWD,
        pos=("q", "k", "v", "o", "dO", "stats"),
        outputs=("dQ", "dK", "dV"),
        out_kwargs=("dBias", "dSink_token", "rng_dump"),
        infer={"dQ": _like("q"), "dK": _like("k"), "dV": _like("v")},
    ),
    "sdpa_fp8": dict(
        node_type=NodeType.SDPA_FP8,
        pos=("q", "k", "v", "descale_q", "descale_k", "descale_v", "descale_s", "scale_s", "scale_o"),
        outputs=("O", "Stats", "Amax_S", "Amax_O"),
        out_kwargs=("rng_dump", "score_max", "score_sum_exp"),
        maybe={"Stats": _stats_expected},
        infer={"O": _sdpa_o_dims, "Stats": _sdpa_stats_dims, "Amax_S": _AMAX, "Amax_O": _AMAX},
    ),
    "sdpa_fp8_backward": dict(
        node_type=NodeType.SDPA_FP8_BWD,
        pos=(
            "q",
            "k",
            "v",
            "o",
            "dO",
            "stats",
            "descale_q",
            "descale_k",
            "descale_v",
            "descale_o",
            "descale_dO",
            "descale_s",
            "descale_dP",
            "scale_s",
            "scale_dQ",
            "scale_dK",
            "scale_dV",
            "scale_dP",
        ),
        outputs=("dQ", "dK", "dV", "amax_dQ", "amax_dK", "amax_dV", "amax_dP"),
        out_kwargs=("dSink_token",),
        infer={"dQ": _like("q"), "dK": _like("k"), "dV": _like("v"), "amax_dQ": _AMAX, "amax_dK": _AMAX, "amax_dV": _AMAX, "amax_dP": _AMAX},
    ),
    # mxfp8 variants (schemas match the bindings exactly; output dims via
    # out_dims / set_dim where cuDNN needs them)
    "sdpa_mxfp8": dict(
        node_type=NodeType.SDPA_MXFP8,
        pos=("q", "k", "v", "descale_q", "descale_k", "descale_v"),
        outputs=("O", "Stats", "Amax_O"),
        maybe={"Stats": _stats_expected},
    ),
    "sdpa_mxfp8_backward": dict(
        node_type=NodeType.SDPA_MXFP8_BWD,
        pos=(
            "q",
            "q_T",
            "k",
            "k_T",
            "v",
            "o_f16",
            "dO_f16",
            "dO",
            "dO_T",
            "stats",
            "descale_q",
            "descale_q_T",
            "descale_k",
            "descale_k_T",
            "descale_v",
            "descale_dO",
            "descale_dO_T",
        ),
        outputs=("dQ", "dK", "dV", "amax_dQ", "amax_dK", "amax_dV"),
        out_kwargs=("dSink_token",),
    ),
}

_CAPTURED_BY_TYPE = {spec["node_type"]: (op, spec) for op, spec in _CAPTURED_OPS.items()}


def _install_captured_builders() -> None:
    """Generate the sdpa-family builders (generic kwarg capture)."""

    def _tensorish(v):
        return isinstance(v, Tensor) or hasattr(v, "__dlpack__")

    def make(op: str, spec: dict):
        pos = spec.get("pos", ())
        infer = spec.get("infer", {})
        maybe = spec.get("maybe", {})

        def builder(self, *args, name: str = "", compute_data_type: Any = None, out_dims: Any = None, **kwargs):
            name_ = self._get_name(op, name)
            node = Node(name_, spec["node_type"], compute_data_type or self._context.compute_data_type)
            if len(args) > len(pos):
                raise TypeError(f"{op}() takes at most {len(pos)} positional arguments {pos}")
            for k, v in zip(pos, args):
                if k in kwargs:
                    raise TypeError(f"{op}() got multiple values for {k!r}")
                kwargs[k] = v
            drop = kwargs.pop("dropout", None)
            out_kwargs = spec.get("out_kwargs", ())
            for k, v in kwargs.items():
                if v is None:
                    continue
                if _tensorish(v):
                    if k in out_kwargs:  # semantically an OUTPUT of this node
                        node.outputs[k] = self._ensure_tensor(v, name=f"{name_}::{k}")
                    else:
                        node.inputs[k] = self._ensure_tensor(v, name=f"{name_}::{k}")
                else:  # scalar / enum / callback — forwarded verbatim at lowering
                    node.params[k] = v
            if drop is not None:
                node.params["_dropout_n"] = len(drop)
                for i, e in enumerate(drop):
                    if _tensorish(e):
                        node.inputs[f"dropout_{i}"] = self._ensure_tensor(e, name=f"{name_}::dropout_{i}")
                    else:
                        node.params[f"dropout_{i}"] = e
            if out_dims is not None and not isinstance(out_dims, dict):
                out_dims = {spec["outputs"][0]: out_dims}
            rets = []
            for oport in spec["outputs"]:
                cond = maybe.get(oport)
                if cond is not None and not cond(node.params):
                    rets.append(None)  # e.g. Stats in inference mode (classic returns None)
                    continue
                o = self._make_output(f"{name_}::{spec.get('out_suffix', {}).get(oport) or self._CLASSIC_OUT_SUFFIX.get(oport, oport)}")
                d = (out_dims or {}).get(oport)
                if d is None:
                    try:
                        d = infer.get(oport, lambda n: None)(node)
                    except Exception:  # noqa: BLE001
                        d = None
                if d:
                    o.dim = list(d)
                    o.stride = _row_major_stride(o.dim)
                node.outputs[oport] = o
                self._register_tensor(o)
                rets.append(o)
            self._nodes.append(node)
            return list(rets)  # always full arity; classic returns a LIST

        builder.__name__ = op
        builder.__qualname__ = f"pygraph.{op}"
        builder.__doc__ = f"{op}(...) -> {spec['outputs']} (generic kwarg capture; see _CAPTURED_OPS)."
        return builder

    for op, spec in _CAPTURED_OPS.items():
        setattr(pygraph, op, make(op, spec))


_install_captured_builders()
