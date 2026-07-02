"""Pure Python graph representation for cuDNN Frontend.

All graph structure and attributes are kept in Python. Graph construction is
backend-agnostic; a backend is chosen at create_execution_plans() time by the
Router, and the backend-specific representation (e.g. the C++ cuDNN graph) is
generated lazily only then. See ``docs/python_native_graph_router.md``.

Execution flow (unification proposal):
    build ops -> create_execution_plans() -> Router -> selected backend
    (a registered native engine, or the cuDNN Graph backend by lazy lowering)

Example with a native backend (pass torch tensors directly):
    >>> graph = NativeGraph()
    >>> graph.register_backend(MatmulCuTileEngine())
    >>> C = graph.matmul(a_tensor, b_tensor)  # auto-creates descriptors
    >>> graph.execute({C: c_tensor})  # routes to a supporting backend, else cuDNN
"""

from dataclasses import dataclass
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


class NativeGraph:
    """Pure Python graph representation.

    All graph structure and attributes are kept in Python. C++ is only
    used for execution via lazy lowering.

    Example:
        >>> graph = NativeGraph(io_data_type=cudnn.data_type.HALF)
        >>> A = graph.tensor(dim=[8, 64, 128], name="A")
        >>> B = graph.tensor(dim=[8, 128, 256], name="B")
        >>> C = graph.matmul(A, B, name="mm1")
        >>> # C is auto-marked as output (leaf tensor) during validate()
        >>>
        >>> # Inspect graph
        >>> print(graph.nodes)  # [Node('mm1', MATMUL)]
        >>> print(graph.nodes[0].inputs)  # {"A": ..., "B": ...}
        >>> print(graph.nodes[0].params)  # {"padding": 0.0}
    """

    def __init__(
        self,
        io_data_type: Any = None,
        intermediate_data_type: Any = None,
        compute_data_type: Any = None,
        handle: Any = None,
        use_native: bool = False,
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
        self._backends: List["BaseEngine"] = list(backends) if backends else []
        self._router = router  # None => engines.router.default_router at route time
        self._plans: List[Any] = []  # list[PlanConfig], populated by create_execution_plans()
        self._plan_index: int = 0
        self._cudnn_heuristics: Optional[List] = None  # heur modes for a cuDNN plan
        self._cpp_tensors: Dict[int, Any] = {}  # IR uid -> lowered C++ tensor
        self._reserved_uids: set = set()  # user-specified uids _alloc_uid must skip

        # Back-compat: use_native=True registers the default native matmul engine
        # as a candidate (the Router still falls back to cuDNN if it can't run
        # this graph / hardware).
        if use_native:
            try:
                from .engines import MatmulCuTileEngine

                if MatmulCuTileEngine is not None:
                    self._backends.append(MatmulCuTileEngine())
            except Exception:  # noqa: BLE001 — optional deps; router falls back
                pass

    # =========================================================================
    # Backend registration & routing
    # =========================================================================

    def register_backend(self, engine: "BaseEngine") -> "NativeGraph":
        """Add a candidate python execution engine. It joins the plan list at
        create_execution_plans() time when its check_support() accepts the graph."""
        self._backends.append(engine)
        return self

    def set_router(self, router: Any) -> "NativeGraph":
        """Override the plan-list / ranking policy for this graph."""
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
    def selected_engine(self) -> Optional["BaseEngine"]:
        """The python engine for the currently selected plan, or None for the
        cuDNN path. Populated after create_execution_plans()."""
        if not self._plans:
            return None
        from .engines.engine_ids import is_python_engine

        eid = self._plans[self._plan_index].engine_id
        return self._engine_by_id(eid) if is_python_engine(eid) else None

    # =========================================================================
    # Tensor Creation
    # =========================================================================

    def tensor(
        self,
        dim: List[int],
        stride: Optional[List[int]] = None,
        data_type: Any = None,
        is_virtual: bool = False,
        name: str = "",
        uid: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """Create a tensor."""
        if not name:
            name = f"tensor_{len(self._tensors)}"

        if uid is not None:
            # User-owned uid: reserve it so _alloc_uid never hands it out, and
            # reject duplicates eagerly (C++ would only fail at build time).
            if uid in self._tensor_by_uid:
                raise ValueError(f"uid {uid} is already used by tensor {self._tensor_by_uid[uid].name!r}")
            self._reserved_uids.add(uid)

        t = Tensor(
            name=name,
            dim=dim,
            stride=stride or _row_major_stride(dim),
            data_type=data_type or (self._context.intermediate_data_type if is_virtual else self._context.io_data_type),
            is_virtual=is_virtual,
            uid=uid if uid is not None else self._alloc_uid(),
            uid_assigned=uid is not None,
            **kwargs,
        )
        self._tensors[name] = t
        self._tensor_by_uid[t.uid] = t
        return t

    def tensor_like(self, template: Any, name: str = "", is_virtual: bool = False) -> Tensor:
        """Create tensor from DLPack object (e.g., torch.Tensor)."""
        dim = list(template.shape)
        stride = list(template.stride()) if hasattr(template, "stride") else _row_major_stride(dim)

        data_type = None
        try:
            import cudnn.datatypes

            data_type = cudnn.datatypes._torch_to_cudnn_data_type(template.dtype)
        except Exception:
            pass

        return self.tensor(dim=dim, stride=stride, data_type=data_type, is_virtual=is_virtual, name=name)

    def _alloc_uid(self) -> int:
        # Skip uids the user reserved via tensor(uid=...) — the Python IR owns
        # the whole uid namespace (see the uid-ownership note in _lower_to_cpp).
        while self._next_uid in self._reserved_uids:
            self._next_uid += 1
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def _get_name(self, op: str, name: str) -> str:
        if name:
            return name
        count = self._node_count.get(op, 0)
        self._node_count[op] = count + 1
        return f"{op}.{count}"

    def _make_output(self, name: str) -> Tensor:
        """Create a virtual output tensor."""
        return Tensor(
            name=name,
            is_virtual=True,
            uid=self._alloc_uid(),
            data_type=self._context.intermediate_data_type,
        )

    def _register_tensor(self, t: Tensor) -> None:
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
        """All nodes in the graph."""
        return self._nodes

    @property
    def tensors(self) -> Dict[str, Tensor]:
        """All tensors by name."""
        return self._tensors

    @property
    def context(self) -> GraphContext:
        """Graph context."""
        return self._context

    def find_tensor(self, name_or_uid: Union[str, int]) -> Optional[Tensor]:
        """Find tensor by name or UID."""
        if isinstance(name_or_uid, int):
            return self._tensor_by_uid.get(name_or_uid)
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

        Automatically marks leaf output tensors (not consumed by any
        subsequent op) as non-virtual (outputs).
        """
        # Auto-mark leaf tensors as outputs
        consumed = {t.uid for node in self._nodes for t in node.inputs.values() if t}
        for node in self._nodes:
            for t in node.outputs.values():
                if t and t.is_virtual and t.uid not in consumed:
                    t.set_output(True)
                    # Fix data_type: _make_output sets intermediate, but outputs need io
                    if t.data_type == self._context.intermediate_data_type:
                        t.data_type = self._context.io_data_type

        for node in self._nodes:
            node.infer_properties(self._context)
            node.validate()
        for t in self._tensors.values():
            if not t.is_pass_by_value:
                t.validate()
        self._is_validated = True

    def build_operation_graph(self) -> None:
        """Validate the graph (backend-agnostic).

        Backend selection is deferred to create_execution_plans() (the Router
        stage), so this no longer commits to a backend or lowers to C++. It only
        ensures the Python graph is validated / properties inferred.
        """
        if not self._is_validated:
            self.validate()

    def create_execution_plans(self, heuristics: Optional[List] = None) -> None:
        """Build the ranked execution-plan list (the dispatch stage).

        The Router returns one flat list of PlanConfig(engine_id, knobs) mixing
        python engines (reserved id region) and the cuDNN side, in one shared
        engine-id space. Nothing is lowered here — a plan is built lazily when
        selected. ``_plan_index`` selects which plan runs (default 0, the
        highest-ranked); cuDNN heuristic modes are carried on the cuDNN plan's
        knobs.

        Args:
            heuristics: cuDNN heuristic modes, carried to the cuDNN plan.
        """
        if not self._is_validated:
            self.validate()

        from .engines.router import default_router

        router = self._router or default_router
        self._plans = router.plan(self, self._backends)
        self._plan_index = 0
        self._cudnn_heuristics = heuristics  # applied when a cuDNN plan is built

    def get_execution_plan_count(self) -> int:
        """Number of candidate plans (python + cuDNN) in the ranked list."""
        return len(self._plans)

    def select_plan(self, index: int) -> "NativeGraph":
        """Pick which plan in the ranked list to build/execute (for autotune)."""
        if not 0 <= index < len(self._plans):
            raise IndexError(f"plan index {index} out of range for {len(self._plans)} plan(s)")
        self._plan_index = index
        self._is_built = False
        return self

    def _lower_cudnn_plan(self) -> None:
        """Lazily lower to C++ and build the cuDNN plan (for a cuDNN-id plan)."""
        import cudnn

        if self._lowered_graph is None:
            self._lowered_graph = self._lower_to_cpp()
            self._lowered_graph.validate()
            self._lowered_graph.build_operation_graph()
            # Verify the uid-ownership invariant (see _lower_to_cpp): every C++
            # tensor must carry exactly its IR uid. An assertion — not a silent
            # translation — so a lowering path that forgets to push a uid fails
            # loudly in tests instead of mis-binding buffers (a swapped
            # multi-output pairing writes past the smaller buffer: corruption).
            for ir_uid, cpp_t in self._cpp_tensors.items():
                cpp_uid = cpp_t.get_uid()
                if cpp_uid != ir_uid:
                    raise RuntimeError(f"uid ownership violated: IR tensor uid {ir_uid} lowered to C++ uid {cpp_uid} — a lowering path failed to push the uid")
        heur = self._cudnn_heuristics or [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
        self._lowered_graph.create_execution_plans(heur)

    def check_support(self) -> None:
        """Check the selected plan's engine supports the graph.

        A python plan re-affirms its engine's check_support() (already passed
        when the Router included it); a cuDNN plan lowers and checks C++ support.
        """
        eng = self.selected_engine
        if eng is not None:
            eng.check_support(self)
            return
        if self._lowered_graph is None:
            self._lower_cudnn_plan()
        self._lowered_graph.check_support()

    def build_plans(self) -> None:
        """Finalize the selected plan.

        A python plan is a no-op (its engine executes directly); a cuDNN plan
        lowers to C++ and builds its plans.
        """
        if self.selected_engine is None:
            if self._lowered_graph is None:
                self._lower_cudnn_plan()
            self._lowered_graph.build_plans()
        self._is_built = True

    def build(self, heuristics: Optional[List] = None) -> None:
        """Convenience: validate -> build_operation_graph -> create_execution_plans
        -> check_support -> build_plans, in sequence."""
        if not self._is_validated:
            self.validate()

        self.build_operation_graph()
        self.create_execution_plans(heuristics)
        self.check_support()
        self.build_plans()

    def get_workspace_size(self) -> int:
        """Get workspace size in bytes for the selected plan."""
        if not self._is_built:
            raise RuntimeError("Call build() first")

        eng = self.selected_engine
        if eng is not None:
            return eng.get_workspace_size()

        return self._lowered_graph.get_workspace_size()

    def execute(
        self,
        tensor_dict: Dict[Union[str, int, Tensor], Any],
        workspace: Any = None,
        handle: int = None,
    ) -> None:
        """Execute the selected plan.

        Both python engines and the cuDNN path write results directly into the
        caller-provided output tensors (in-place). Automatically calls build()
        if it hasn't run yet. Dispatch is a single check on the plan's engine id.

        Args:
            tensor_dict: Dict mapping tensors (by Tensor, name, or uid) to data.
                         Must include both input and output tensors.
            workspace: Workspace buffer (ignored by python engines)
            handle: cuDNN handle (ignored by python engines)
        """
        if not self._is_built:
            self.build()

        # Start with auto-bound inputs, then overlay user-provided (user wins)
        uid_to_data = dict(self._data_bindings)
        for key, data in tensor_dict.items():
            if isinstance(key, Tensor):
                uid = key.uid
            elif isinstance(key, str):
                uid = self._tensors[key].uid
            else:
                uid = key
            uid_to_data[uid] = data

        eng = self.selected_engine
        if eng is not None:  # python engine (plan id in the reserved region)
            eng.execute(self, uid_to_data)
            return

        # cuDNN execution path (plan id < PYTHON_ENGINE_ID_BASE). Variant-pack
        # keys are IR uids — identical to the C++ uids by construction (the IR
        # owns the uid namespace and lowering pushes every uid explicitly).
        var_pack = {uid: (d.data_ptr() if hasattr(d, "data_ptr") else d) for uid, d in uid_to_data.items()}
        ws_ptr = workspace.data_ptr() if hasattr(workspace, "data_ptr") else workspace
        self._lowered_graph._execute(var_pack, ws_ptr, handle)

    @property
    def use_native(self) -> bool:
        """True iff the selected plan is a python engine (not the cuDNN path).

        Meaningful after create_execution_plans()/build(); before routing it
        reports whether any python engine is registered as a candidate.
        """
        if self._plans:
            return self.selected_engine is not None
        return bool(self._backends) and self._lowered_graph is None

    @property
    def engine(self) -> Optional["BaseEngine"]:
        """The python engine for the selected plan, or None for the cuDNN path.
        Populated after create_execution_plans()."""
        return self.selected_engine

    def serialize(self) -> bytes:
        """Serialize the graph to bytes.

        The graph must be built first. This lowers to the C++ serialization
        format to ensure compatibility with C++ deserialization.

        Returns:
            bytes: Serialized graph data.
        """
        if not self._is_built:
            raise RuntimeError("Call build() first")
        return bytes(self._lowered_graph.serialize())

    def deserialize(self, data: bytes, handle: Optional[int] = None) -> None:
        """Deserialize graph from bytes.

        This replaces the current graph with the deserialized one.
        The graph must have been lowered/built first to have a C++ graph to deserialize into.

        Args:
            data: Serialized graph data (from serialize()).
            handle: Optional cuDNN handle for AoT compilation.
        """
        if self._lowered_graph is None:
            # Need to lower first to have a C++ graph to deserialize into
            self.validate()
            self._lowered_graph = self._lower_to_cpp()

        if handle is not None:
            self._lowered_graph.deserialize(handle, data)
        else:
            self._lowered_graph.deserialize(data)
        self._is_built = True

    @classmethod
    def from_pygraph(cls, pygraph: Any, **kwargs) -> "NativeGraph":
        """Build a NativeGraph (Node/Tensor IR) from an existing ``cudnn.pygraph``.

        This is the second front-door for populating the IR: users who author on
        the classic ``cudnn.pygraph`` API get a backend-agnostic Node/Tensor
        graph that any backend can consume via ``graph.nodes`` — replacing the
        monkey-patch "recorder" approach.

        NOT IMPLEMENTED YET. The pybind ``cudnn.pygraph`` does not expose its
        node/tensor structure to Python, so this converter needs one of:
          * a proper C++/pybind reflection API that walks the built op graph, or
          * (interim) reuse the op-recording hook to emit Node/Tensor directly.
        Tracked as the 1718<->2163 integration step; see
        ``docs/python_native_graph_router.md``.
        """
        raise NotImplementedError(
            "NativeGraph.from_pygraph() is not implemented yet — cudnn.pygraph "
            "does not expose graph structure to Python. See "
            "docs/python_native_graph_router.md (interim: reuse the op-recording "
            "hook to emit Node/Tensor; long-term: a C++ reflection API)."
        )

    @classmethod
    def from_serialized(cls, data: bytes, handle: Optional[int] = None, **kwargs) -> "NativeGraph":
        """Create a NativeGraph from serialized data.

        This is a convenience method that creates a minimal graph and deserializes into it.

        Args:
            data: Serialized graph data (from serialize()).
            handle: Optional cuDNN handle for AoT compilation.
            **kwargs: Additional arguments passed to NativeGraph constructor.

        Returns:
            NativeGraph: Deserialized graph ready for execution.
        """
        import cudnn

        # Create a new NativeGraph with a fresh C++ graph
        graph = cls(**kwargs)
        graph._lowered_graph = cudnn.pygraph(
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
        """Lower Python graph to C++."""
        import cudnn

        # cudnn.pygraph rejects None (wants the enum). io_data_type may be unset
        # (block-scale tensors carry their own dtypes), but intermediate/compute
        # default to FLOAT — matching cudnn.graph() — so cuDNN can infer virtual
        # (intermediate) tensor dtypes during build.
        pg_kwargs = {}
        if self._context.io_data_type is not None:
            pg_kwargs["io_data_type"] = self._context.io_data_type
        pg_kwargs["intermediate_data_type"] = self._context.intermediate_data_type or cudnn.data_type.FLOAT
        pg_kwargs["compute_data_type"] = self._context.compute_data_type or cudnn.data_type.FLOAT
        if self._handle is not None:
            pg_kwargs["handle"] = self._handle
        graph = cudnn.pygraph(**pg_kwargs)

        tensor_map: Dict[int, Any] = {}

        def lower_tensor(t: Tensor) -> Any:
            if t.uid in tensor_map:
                return tensor_map[t.uid]
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
                mk_kwargs["data_type"] = t.data_type  # graph intermediate_data_type
            if t.reordering_type is not None:  # e.g. F8_128x4 for block-scale SFs
                mk_kwargs["reordering_type"] = t.reordering_type
            cpp = graph._make_tensor(**mk_kwargs)
            tensor_map[t.uid] = cpp
            return cpp

        for node in self._nodes:
            for t in node.inputs.values():
                if t:
                    lower_tensor(t)

            if node.node_type == NodeType.MATMUL:
                cpp_out = graph.matmul(
                    A=tensor_map[node.inputs["A"].uid],
                    B=tensor_map[node.inputs["B"].uid],
                    compute_data_type=node.compute_data_type,
                    padding=node.params.get("padding", 0.0),
                    name=node.name,
                )
            elif node.node_type == NodeType.POINTWISE:
                # params["mode"] IS the C++ pygraph method name — direct
                # dispatch; scalar attributes (clips/negative_slope/...) are
                # forwarded as keywords, tensors positionally (they lead every
                # pointwise signature).
                inputs = [tensor_map[t.uid] for t in node.inputs.values()]
                extra = {k: node.params[k] for k in self._POINTWISE_EXTRA_PARAMS if k in node.params}
                cpp_out = getattr(graph, node.params["mode"])(*inputs, compute_data_type=node.compute_data_type, name=node.name, **extra)
            elif node.node_type in _CAPTURED_BY_TYPE:
                # Captured op (sdpa family): rebuild the original kwargs —
                # tensor ports (port == C++ kwarg) map through tensor_map,
                # scalar params forward verbatim, dropout reassembles from its
                # flattened elements — and call the C++ method once.
                method, spec = _CAPTURED_BY_TYPE[node.node_type]
                kw = {"name": node.name, "compute_data_type": node.compute_data_type}
                for pk, pv in node.params.items():
                    if not pk.startswith("_") and not pk.startswith("dropout_"):
                        kw[pk] = pv
                for port, t in node.inputs.items():
                    if not port.startswith("dropout_"):
                        kw[port] = tensor_map[t.uid]
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
                    if not out_t.is_virtual:
                        cpp_t.set_output(True)
                    if out_t.data_type:
                        cpp_t.set_data_type(out_t.data_type)
                continue
            elif node.node_type in _STRUCTURED_BY_TYPE:
                # Generic structured op (norms / reduction / block-scale / MoE /
                # conv / structural): input ports are named after the C++
                # kwargs, so lowering is kwargs assembly + one call + zipping
                # the returned tuple with the declared output ports.
                method, spec = _STRUCTURED_BY_TYPE[node.node_type]
                kw = {"name": node.name}
                if not spec.get("no_cdt"):  # a few bindings take no compute_data_type
                    kw["compute_data_type"] = node.compute_data_type
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
                        if out_t.stride:
                            cpp_t.set_stride(out_t.stride)
                    if not out_t.is_virtual:
                        cpp_t.set_output(True)
                    if out_t.data_type:
                        cpp_t.set_data_type(out_t.data_type)
                continue
            else:
                continue

            # Map output
            for out_t in node.outputs.values():
                tensor_map[out_t.uid] = cpp_out
                if not out_t.is_virtual:
                    cpp_out.set_output(True)
                if out_t.data_type:
                    cpp_out.set_data_type(out_t.data_type)

        # ---- uid ownership -------------------------------------------------
        # The Python IR owns the whole uid namespace: every IR tensor gets a uid
        # eagerly at creation (_alloc_uid, or user-specified via tensor(uid=)),
        # and lowering pushes ALL of them explicitly to C++ — inputs via
        # _make_tensor(uid=), op-created outputs/virtuals via set_uid here. The
        # C++ FE's build-time auto-assignment therefore NEVER triggers for
        # graphs built through NativeGraph (its enumeration order is not
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
        builder.__qualname__ = f"NativeGraph.{op}"
        builder.__doc__ = f"Element-wise {op}({', '.join(argnames)})."
        return builder

    for op, argnames in NativeGraph._POINTWISE_TENSOR_ARGS.items():
        if not hasattr(NativeGraph, op):  # explicit builders (relu, ...) win
            setattr(NativeGraph, op, make(op, argnames))


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

_STRUCTURED_OPS = {
    # ---- norms --------------------------------------------------------------
    "rmsnorm": dict(
        node_type=NodeType.RMSNORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "inv_var"),
        infer={"Y": _like("input"), "inv_var": _stats_like("input", (0,))},
    ),
    "rmsnorm_backward": dict(
        node_type=NodeType.RMSNORM_BWD,
        inputs=("grad", "input", "scale", "inv_variance"),
        attrs=("has_dbias",),
        outputs=("DX", "DScale", "DBias"),
        infer=_NORM_BWD_INFER,
    ),
    "layernorm": dict(
        node_type=NodeType.LAYERNORM,
        inputs=("input", "scale", "bias", "epsilon"),
        attrs=("norm_forward_phase",),
        outputs=("Y", "mean", "inv_var"),
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
        inputs=("token", "weight", "first_token_offset"),
        attrs=("mode",),
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

        def builder(self, *args, name: str = "", compute_data_type: Any = None, out_dims: Any = None, **kwargs):
            name_ = self._get_name(op, name)
            node = Node(name_, spec["node_type"], compute_data_type or self._context.compute_data_type)
            if len(args) > len(input_ports):
                raise TypeError(f"{op}() takes at most {len(input_ports)} positional tensors {input_ports}")
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
            outs = []
            for oport in spec["outputs"]:
                o = self._make_output(f"{name_}::{oport}")
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
            return outs[0] if len(outs) == 1 else tuple(outs)

        builder.__name__ = op
        builder.__qualname__ = f"NativeGraph.{op}"
        builder.__doc__ = f"{op}({', '.join(input_ports)}) -> ({', '.join(spec['outputs'])})."
        return builder

    for op, spec in _STRUCTURED_OPS.items():
        setattr(NativeGraph, op, make(op, spec))


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
        maybe={"Stats": _stats_expected},
        infer={"O": _sdpa_o_dims, "Stats": _sdpa_stats_dims},
    ),
    "sdpa_backward": dict(
        node_type=NodeType.SDPA_BWD,
        pos=("q", "k", "v", "o", "dO", "stats"),
        outputs=("dQ", "dK", "dV"),
        infer={"dQ": _like("q"), "dK": _like("k"), "dV": _like("v")},
    ),
    "sdpa_fp8": dict(
        node_type=NodeType.SDPA_FP8,
        pos=("q", "k", "v"),
        outputs=("O", "Stats", "Amax_S", "Amax_O"),
        maybe={"Stats": _stats_expected},
        infer={"O": _sdpa_o_dims, "Stats": _sdpa_stats_dims, "Amax_S": _AMAX, "Amax_O": _AMAX},
    ),
    "sdpa_fp8_backward": dict(
        node_type=NodeType.SDPA_FP8_BWD,
        pos=("q", "k", "v", "o", "dO", "stats"),
        outputs=("dQ", "dK", "dV", "amax_dQ", "amax_dK", "amax_dV", "amax_dP"),
        infer={"dQ": _like("q"), "dK": _like("k"), "dV": _like("v"), "amax_dQ": _AMAX, "amax_dK": _AMAX, "amax_dV": _AMAX, "amax_dP": _AMAX},
    ),
    # mxfp8 variants: outputs are positional (see sdpa.cpp result_array); dims
    # via out_dims / set_dim where cuDNN needs them.
    "sdpa_mxfp8": dict(node_type=NodeType.SDPA_MXFP8, pos=("q", "k", "v"), outputs=("OUT_0", "OUT_1", "OUT_2")),
    "sdpa_mxfp8_backward": dict(
        node_type=NodeType.SDPA_MXFP8_BWD,
        pos=("q", "k", "v", "o", "dO", "stats"),
        outputs=("OUT_0", "OUT_1", "OUT_2", "OUT_3", "OUT_4", "OUT_5"),
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
            for k, v in kwargs.items():
                if v is None:
                    continue
                if _tensorish(v):
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
                o = self._make_output(f"{name_}::{oport}")
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
            return tuple(rets)  # always full arity, matching the classic API

        builder.__name__ = op
        builder.__qualname__ = f"NativeGraph.{op}"
        builder.__doc__ = f"{op}(...) -> {spec['outputs']} (generic kwarg capture; see _CAPTURED_OPS)."
        return builder

    for op, spec in _CAPTURED_OPS.items():
        setattr(NativeGraph, op, make(op, spec))


_install_captured_builders()
