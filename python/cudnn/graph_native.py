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

    def _pointwise(self, mode: Any, inputs: list, name: str, compute_data_type: Any = None) -> Tensor:
        """Internal helper for pointwise ops."""
        inputs = [self._ensure_tensor(t, name=f"{name}::IN_{i}") for i, t in enumerate(inputs)]
        node = Node(name, NodeType.POINTWISE, compute_data_type or self._context.compute_data_type)
        node.params["mode"] = mode
        for i, t in enumerate(inputs):
            node.inputs[f"IN_{i}"] = t

        out = self._make_output(f"{name}::OUT_0")
        node.outputs["OUT_0"] = out
        self._register_tensor(out)

        self._nodes.append(node)
        return out

    def add(self, a: Tensor, b: Tensor, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Element-wise add."""
        try:
            import cudnn

            mode = cudnn._pybind_module.pointwise_mode.ADD
        except Exception:
            mode = "ADD"
        return self._pointwise(mode, [a, b], self._get_name("add", name), compute_data_type)

    def mul(self, a: Tensor, b: Tensor, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Element-wise multiply."""
        try:
            import cudnn

            mode = cudnn._pybind_module.pointwise_mode.MUL
        except Exception:
            mode = "MUL"
        return self._pointwise(mode, [a, b], self._get_name("mul", name), compute_data_type)

    def relu(self, x: Tensor, name: str = "", compute_data_type: Any = None) -> Tensor:
        """ReLU activation."""
        try:
            import cudnn

            mode = cudnn._pybind_module.pointwise_mode.RELU_FWD
        except Exception:
            mode = "RELU_FWD"
        return self._pointwise(mode, [x], self._get_name("relu", name), compute_data_type)

    def gelu(self, x: Tensor, name: str = "", compute_data_type: Any = None) -> Tensor:
        """GELU activation."""
        try:
            import cudnn

            mode = cudnn._pybind_module.pointwise_mode.GELU_FWD
        except Exception:
            mode = "GELU_FWD"
        return self._pointwise(mode, [x], self._get_name("gelu", name), compute_data_type)

    def sigmoid(self, x: Tensor, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Sigmoid activation."""
        try:
            import cudnn

            mode = cudnn._pybind_module.pointwise_mode.SIGMOID_FWD
        except Exception:
            mode = "SIGMOID_FWD"
        return self._pointwise(mode, [x], self._get_name("sigmoid", name), compute_data_type)

    def tanh(self, x: Tensor, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Tanh activation."""
        try:
            import cudnn

            mode = cudnn._pybind_module.pointwise_mode.TANH_FWD
        except Exception:
            mode = "TANH_FWD"
        return self._pointwise(mode, [x], self._get_name("tanh", name), compute_data_type)

    def bias(self, x: Tensor, b: Tensor, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Add bias."""
        return self.add(x, b, name or "bias", compute_data_type)

    def scale(self, x: Tensor, s: Tensor, name: str = "", compute_data_type: Any = None) -> Tensor:
        """Scale."""
        return self.mul(x, s, name or "scale", compute_data_type)

    # -------------------------------------------------------------------------
    # Block-scale / MoE / reduction op builders.
    #
    # These represent the ops the CuTe-DSL GEMM fusion backend consumes (block
    # scaling, MoE grouped matmul, epilogue reductions). They populate the Node
    # IR so a backend's analyze(graph.nodes) pass can read them directly — no
    # monkey-patch recorder needed. cuDNN lowering (_lower_to_cpp) is wired for
    # all of them; per-op output-shape inference (e.g. reduced dims) is still
    # being filled in, so set output dims explicitly for now where cuDNN needs them.
    # -------------------------------------------------------------------------

    def block_scale_dequantize(self, input: Any, descale: Any, block_size: List[int], is_negative_scale: bool = False, name: str = "") -> Tensor:
        """Dequantize a narrow (FP4/FP8) tensor by a per-block scale factor."""
        name = self._get_name("block_scale_dequantize", name)
        input = self._ensure_tensor(input, name=f"{name}::input")
        descale = self._ensure_tensor(descale, name=f"{name}::descale")
        node = Node(name, NodeType.BLOCK_SCALE_DEQUANTIZE, self._context.compute_data_type)
        node.inputs["input"] = input
        node.inputs["descale"] = descale
        node.params.update(block_size=list(block_size), is_negative_scale=bool(is_negative_scale))
        out = self._make_output(f"{name}::OUT_0")
        out.dim = list(input.dim)
        out.stride = list(input.stride)
        node.outputs["OUT_0"] = out
        self._register_tensor(out)
        self._nodes.append(node)
        return out

    def block_scale_quantize(self, input: Any, block_size: int, axis: Optional[int] = None, transpose: bool = False, name: str = ""):
        """Quantize to a narrow dtype, returning (quantized, scale)."""
        name = self._get_name("block_scale_quantize", name)
        input = self._ensure_tensor(input, name=f"{name}::input")
        node = Node(name, NodeType.BLOCK_SCALE_QUANTIZE, self._context.compute_data_type)
        node.inputs["input"] = input
        node.params.update(block_size=int(block_size), axis=axis, transpose=bool(transpose))
        quantized = self._make_output(f"{name}::OUT_0")
        scale = self._make_output(f"{name}::OUT_1")
        node.outputs["OUT_0"] = quantized
        node.outputs["OUT_1"] = scale
        self._register_tensor(quantized)
        self._register_tensor(scale)
        self._nodes.append(node)
        return quantized, scale

    def moe_grouped_matmul(self, token: Any, weight: Any, first_token_offset: Any, mode: Any = None, name: str = "", **kwargs) -> Tensor:
        """MoE grouped matmul: per-group token range @ per-expert weight."""
        name = self._get_name("moe_grouped_matmul", name)
        token = self._ensure_tensor(token, name=f"{name}::token")
        weight = self._ensure_tensor(weight, name=f"{name}::weight")
        first_token_offset = self._ensure_tensor(first_token_offset, name=f"{name}::first_token_offset")
        node = Node(name, NodeType.MOE_GROUPED_MATMUL, self._context.compute_data_type)
        node.inputs.update(token=token, weight=weight, first_token_offset=first_token_offset)
        node.params["mode"] = mode
        out = self._make_output(f"{name}::OUT_0")
        node.outputs["OUT_0"] = out
        self._register_tensor(out)
        self._nodes.append(node)
        return out

    def reduction(
        self, input: Any, mode: Any, dim: Optional[List[int]] = None, group_offset: Optional[Any] = None, name: str = "", compute_data_type: Any = None
    ) -> Tensor:
        """Reduction (add/amax/max/min), optionally grouped by an offset tensor.

        ``dim`` is the reduced output shape (each axis either the input extent or
        1). cuDNN requires the reduction output dims to be set explicitly, so
        pass ``dim`` here (row-major stride is inferred)."""
        name = self._get_name("reduction", name)
        input = self._ensure_tensor(input, name=f"{name}::input")
        node = Node(name, NodeType.REDUCTION, compute_data_type or self._context.compute_data_type)
        node.inputs["input"] = input
        if group_offset is not None:
            node.inputs["group_offset"] = self._ensure_tensor(group_offset, name=f"{name}::group_offset")
        node.params["mode"] = mode
        out = self._make_output(f"{name}::OUT_0")
        if dim is not None:
            out.dim = list(dim)
            out.stride = _row_major_stride(list(dim))
        node.outputs["OUT_0"] = out
        self._register_tensor(out)
        self._nodes.append(node)
        return out

    def sdpa(
        self,
        q: Any,
        k: Any,
        v: Any,
        is_inference: bool = True,
        attn_scale: Optional[Union[float, "Tensor"]] = None,
        bias: Optional[Any] = None,
        use_alibi_mask: bool = False,
        use_padding_mask: bool = False,
        seq_len_q: Optional[Any] = None,
        seq_len_kv: Optional[Any] = None,
        use_causal_mask: bool = False,
        use_causal_mask_bottom_right: bool = False,
        sliding_window_length: Optional[int] = None,
        dropout: Optional[tuple] = None,
        compute_data_type: Any = None,
        name: str = "",
    ) -> Union[Tensor, tuple]:
        """Scaled Dot-Product Attention.

        Computes attention(Q, K, V) = softmax(Q @ K^T / scale) @ V

        Args:
            q: Query tensor [B, H, S_q, D] or [B, S_q, H, D]
            k: Key tensor [B, H, S_kv, D] or [B, S_kv, H, D]
            v: Value tensor [B, H, S_kv, D] or [B, S_kv, H, D]
            is_inference: If True, don't generate stats for backward pass
            attn_scale: Attention scale factor (default: 1/sqrt(D))
            bias: Optional attention bias tensor
            use_alibi_mask: Use ALiBi positional encoding
            use_padding_mask: Use padding mask with seq_len tensors
            seq_len_q: Sequence lengths for queries (for variable length)
            seq_len_kv: Sequence lengths for keys/values
            use_causal_mask: Apply causal (triangular) mask
            use_causal_mask_bottom_right: Causal mask aligned bottom-right
            sliding_window_length: Sliding window attention length
            dropout: Tuple of (probability, seed_tensor, offset_tensor)
            compute_data_type: Compute precision
            name: Node name

        Returns:
            Output tensor O, or (O, stats) if is_inference=False
        """
        name = self._get_name("sdpa", name)
        q = self._ensure_tensor(q, name=f"{name}::Q")
        k = self._ensure_tensor(k, name=f"{name}::K")
        v = self._ensure_tensor(v, name=f"{name}::V")

        node = Node(name, NodeType.SDPA, compute_data_type or self._context.compute_data_type)
        node.inputs["Q"] = q
        node.inputs["K"] = k
        node.inputs["V"] = v

        if bias is not None:
            bias = self._ensure_tensor(bias, name=f"{name}::bias")
            node.inputs["bias"] = bias
        if seq_len_q is not None:
            seq_len_q = self._ensure_tensor(seq_len_q, name=f"{name}::seq_len_q")
            node.inputs["seq_len_q"] = seq_len_q
        if seq_len_kv is not None:
            seq_len_kv = self._ensure_tensor(seq_len_kv, name=f"{name}::seq_len_kv")
            node.inputs["seq_len_kv"] = seq_len_kv
        if dropout is not None and len(dropout) >= 3:
            node.inputs["dropout_seed"] = dropout[1]
            node.inputs["dropout_offset"] = dropout[2]
            node.params["dropout_probability"] = dropout[0]

        node.params["is_inference"] = is_inference
        node.params["attn_scale"] = attn_scale
        node.params["use_alibi_mask"] = use_alibi_mask
        node.params["use_padding_mask"] = use_padding_mask
        node.params["use_causal_mask"] = use_causal_mask
        node.params["use_causal_mask_bottom_right"] = use_causal_mask_bottom_right
        node.params["sliding_window_length"] = sliding_window_length

        O = self._make_output(f"{name}::O")
        node.outputs["O"] = O
        self._register_tensor(O)

        self._nodes.append(node)

        if not is_inference:
            stats = self._make_output(f"{name}::stats")
            node.outputs["stats"] = stats
            self._register_tensor(stats)
            return O, stats

        return O

    def sdpa_backward(
        self,
        q: Any,
        k: Any,
        v: Any,
        o: Any,
        dO: Any,
        stats: Any,
        attn_scale: Optional[Union[float, "Tensor"]] = None,
        bias: Optional[Any] = None,
        use_alibi_mask: bool = False,
        use_padding_mask: bool = False,
        seq_len_q: Optional[Any] = None,
        seq_len_kv: Optional[Any] = None,
        use_causal_mask: bool = False,
        use_causal_mask_bottom_right: bool = False,
        sliding_window_length: Optional[int] = None,
        dropout: Optional[tuple] = None,
        compute_data_type: Any = None,
        name: str = "",
    ) -> tuple:
        """Scaled Dot-Product Attention Backward.

        Args:
            q, k, v: Forward pass inputs
            o: Forward pass output
            dO: Gradient of output
            stats: Stats from forward pass
            (other args same as sdpa)

        Returns:
            Tuple of (dQ, dK, dV)
        """
        name = self._get_name("sdpa_bwd", name)
        q = self._ensure_tensor(q, name=f"{name}::Q")
        k = self._ensure_tensor(k, name=f"{name}::K")
        v = self._ensure_tensor(v, name=f"{name}::V")
        o = self._ensure_tensor(o, name=f"{name}::O")
        dO = self._ensure_tensor(dO, name=f"{name}::dO")
        stats = self._ensure_tensor(stats, name=f"{name}::stats")

        node = Node(name, NodeType.SDPA_BWD, compute_data_type or self._context.compute_data_type)
        node.inputs["Q"] = q
        node.inputs["K"] = k
        node.inputs["V"] = v
        node.inputs["O"] = o
        node.inputs["dO"] = dO
        node.inputs["stats"] = stats

        if bias is not None:
            bias = self._ensure_tensor(bias, name=f"{name}::bias")
            node.inputs["bias"] = bias
        if seq_len_q is not None:
            seq_len_q = self._ensure_tensor(seq_len_q, name=f"{name}::seq_len_q")
            node.inputs["seq_len_q"] = seq_len_q
        if seq_len_kv is not None:
            seq_len_kv = self._ensure_tensor(seq_len_kv, name=f"{name}::seq_len_kv")
            node.inputs["seq_len_kv"] = seq_len_kv
        if dropout is not None and len(dropout) >= 3:
            node.inputs["dropout_seed"] = dropout[1]
            node.inputs["dropout_offset"] = dropout[2]
            node.params["dropout_probability"] = dropout[0]

        node.params["attn_scale"] = attn_scale
        node.params["use_alibi_mask"] = use_alibi_mask
        node.params["use_padding_mask"] = use_padding_mask
        node.params["use_causal_mask"] = use_causal_mask
        node.params["use_causal_mask_bottom_right"] = use_causal_mask_bottom_right
        node.params["sliding_window_length"] = sliding_window_length

        dQ = self._make_output(f"{name}::dQ")
        dK = self._make_output(f"{name}::dK")
        dV = self._make_output(f"{name}::dV")
        node.outputs["dQ"] = dQ
        node.outputs["dK"] = dK
        node.outputs["dV"] = dV
        self._register_tensor(dQ)
        self._register_tensor(dK)
        self._register_tensor(dV)

        self._nodes.append(node)
        return dQ, dK, dV

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

        # cuDNN execution path (plan id < PYTHON_ENGINE_ID_BASE)
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

        pg_kwargs = dict(
            io_data_type=self._context.io_data_type,
            intermediate_data_type=self._context.intermediate_data_type,
            compute_data_type=self._context.compute_data_type,
        )
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
                data_type=t.data_type,
                is_virtual=t.is_virtual,
                is_pass_by_value=t.is_pass_by_value,
                name=t.name,
                # Always propagate the IR uid so execute()'s variant pack (keyed
                # by IR uid) matches; otherwise cuDNN assigns its own and the
                # buffers never bind. IR uids are unique and positive.
                uid=t.uid,
            )
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
                # The C++ pygraph exposes named pointwise ops (relu/add/...), not a
                # generic pointwise(). Dispatch on the mode. add/mul cover bias/scale
                # too (broadcast add/mul), so no need to distinguish here.
                inputs = [tensor_map[t.uid] for t in node.inputs.values()]
                mode_name = getattr(node.params["mode"], "name", str(node.params["mode"])).upper()
                _PW_UNARY = {"RELU_FWD": "relu", "GELU_FWD": "gelu", "SIGMOID_FWD": "sigmoid", "TANH_FWD": "tanh"}
                _PW_BINARY = {"ADD": "add", "MUL": "mul", "SUB": "sub", "DIV": "div"}
                if len(inputs) == 1:
                    method = _PW_UNARY.get(mode_name)
                    if method is None:
                        raise NotImplementedError(f"pointwise lowering: unary mode {mode_name} not mapped")
                    cpp_out = getattr(graph, method)(inputs[0], compute_data_type=node.compute_data_type, name=node.name)
                else:
                    method = _PW_BINARY.get(mode_name)
                    if method is None:
                        raise NotImplementedError(f"pointwise lowering: binary mode {mode_name} not mapped")
                    cpp_out = getattr(graph, method)(inputs[0], inputs[1], compute_data_type=node.compute_data_type, name=node.name)
            elif node.node_type == NodeType.SDPA:
                sdpa_kwargs = {
                    "q": tensor_map[node.inputs["Q"].uid],
                    "k": tensor_map[node.inputs["K"].uid],
                    "v": tensor_map[node.inputs["V"].uid],
                    "is_inference": node.params.get("is_inference", True),
                    "compute_data_type": node.compute_data_type,
                    "name": node.name,
                }
                if node.params.get("attn_scale") is not None:
                    attn_scale = node.params["attn_scale"]
                    if isinstance(attn_scale, Tensor):
                        sdpa_kwargs["attn_scale"] = tensor_map[attn_scale.uid]
                    else:
                        sdpa_kwargs["attn_scale"] = attn_scale
                if "bias" in node.inputs:
                    sdpa_kwargs["bias"] = tensor_map[node.inputs["bias"].uid]
                if "seq_len_q" in node.inputs:
                    sdpa_kwargs["seq_len_q"] = tensor_map[node.inputs["seq_len_q"].uid]
                if "seq_len_kv" in node.inputs:
                    sdpa_kwargs["seq_len_kv"] = tensor_map[node.inputs["seq_len_kv"].uid]
                if node.params.get("use_alibi_mask"):
                    sdpa_kwargs["use_alibi_mask"] = True
                if node.params.get("use_padding_mask"):
                    sdpa_kwargs["use_padding_mask"] = True
                if node.params.get("use_causal_mask"):
                    sdpa_kwargs["use_causal_mask"] = True
                if node.params.get("use_causal_mask_bottom_right"):
                    sdpa_kwargs["use_causal_mask_bottom_right"] = True
                if node.params.get("sliding_window_length") is not None:
                    sdpa_kwargs["sliding_window_length"] = node.params["sliding_window_length"]
                if "dropout_seed" in node.inputs and "dropout_offset" in node.inputs:
                    sdpa_kwargs["dropout"] = (
                        node.params.get("dropout_probability", 0.0),
                        tensor_map[node.inputs["dropout_seed"].uid],
                        tensor_map[node.inputs["dropout_offset"].uid],
                    )

                result = graph.sdpa(**sdpa_kwargs)
                # sdpa returns [O, stats] as a list/array
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    cpp_out, cpp_stats = result[0], result[1]
                    tensor_map[node.outputs["O"].uid] = cpp_out
                    if "stats" in node.outputs and cpp_stats is not None:
                        tensor_map[node.outputs["stats"].uid] = cpp_stats
                else:
                    cpp_out = result
                    tensor_map[node.outputs["O"].uid] = cpp_out
                # Handle output marking and set dims/strides
                for out_key, out_t in node.outputs.items():
                    cpp_tensor = tensor_map.get(out_t.uid)
                    if cpp_tensor is not None:
                        if out_t.dim:
                            cpp_tensor.set_dim(out_t.dim)
                        if out_t.stride:
                            cpp_tensor.set_stride(out_t.stride)
                        if not out_t.is_virtual:
                            cpp_tensor.set_output(True)
                        if out_t.data_type:
                            cpp_tensor.set_data_type(out_t.data_type)
                continue
            elif node.node_type == NodeType.SDPA_BWD:
                sdpa_bwd_kwargs = {
                    "q": tensor_map[node.inputs["Q"].uid],
                    "k": tensor_map[node.inputs["K"].uid],
                    "v": tensor_map[node.inputs["V"].uid],
                    "o": tensor_map[node.inputs["O"].uid],
                    "dO": tensor_map[node.inputs["dO"].uid],
                    "stats": tensor_map[node.inputs["stats"].uid],
                    "compute_data_type": node.compute_data_type,
                    "name": node.name,
                }
                if node.params.get("attn_scale") is not None:
                    attn_scale = node.params["attn_scale"]
                    if isinstance(attn_scale, Tensor):
                        sdpa_bwd_kwargs["attn_scale"] = tensor_map[attn_scale.uid]
                    else:
                        sdpa_bwd_kwargs["attn_scale"] = attn_scale
                if "bias" in node.inputs:
                    sdpa_bwd_kwargs["bias"] = tensor_map[node.inputs["bias"].uid]
                if "seq_len_q" in node.inputs:
                    sdpa_bwd_kwargs["seq_len_q"] = tensor_map[node.inputs["seq_len_q"].uid]
                if "seq_len_kv" in node.inputs:
                    sdpa_bwd_kwargs["seq_len_kv"] = tensor_map[node.inputs["seq_len_kv"].uid]
                if node.params.get("use_alibi_mask"):
                    sdpa_bwd_kwargs["use_alibi_mask"] = True
                if node.params.get("use_padding_mask"):
                    sdpa_bwd_kwargs["use_padding_mask"] = True
                if node.params.get("use_causal_mask"):
                    sdpa_bwd_kwargs["use_causal_mask"] = True
                if node.params.get("use_causal_mask_bottom_right"):
                    sdpa_bwd_kwargs["use_causal_mask_bottom_right"] = True
                if node.params.get("sliding_window_length") is not None:
                    sdpa_bwd_kwargs["sliding_window_length"] = node.params["sliding_window_length"]
                if "dropout_seed" in node.inputs and "dropout_offset" in node.inputs:
                    sdpa_bwd_kwargs["dropout"] = (
                        node.params.get("dropout_probability", 0.0),
                        tensor_map[node.inputs["dropout_seed"].uid],
                        tensor_map[node.inputs["dropout_offset"].uid],
                    )

                result = graph.sdpa_backward(**sdpa_bwd_kwargs)
                # sdpa_backward returns [dQ, dK, dV] as a list/array
                dQ, dK, dV = result[0], result[1], result[2]
                tensor_map[node.outputs["dQ"].uid] = dQ
                tensor_map[node.outputs["dK"].uid] = dK
                tensor_map[node.outputs["dV"].uid] = dV
                # Handle output marking and set dims/strides
                for out_key, out_t in node.outputs.items():
                    cpp_tensor = tensor_map.get(out_t.uid)
                    if cpp_tensor is not None:
                        if out_t.dim:
                            cpp_tensor.set_dim(out_t.dim)
                        if out_t.stride:
                            cpp_tensor.set_stride(out_t.stride)
                        if not out_t.is_virtual:
                            cpp_tensor.set_output(True)
                        if out_t.data_type:
                            cpp_tensor.set_data_type(out_t.data_type)
                continue
            elif node.node_type == NodeType.REDUCTION:
                red_kwargs = {
                    "input": tensor_map[node.inputs["input"].uid],
                    "mode": node.params["mode"],
                    "compute_data_type": node.compute_data_type,
                    "name": node.name,
                }
                if "group_offset" in node.inputs:
                    red_kwargs["group_offset"] = tensor_map[node.inputs["group_offset"].uid]
                cpp_out = graph.reduction(**red_kwargs)
                # cuDNN needs the reduction output dims set explicitly.
                _red_out = node.outputs["OUT_0"]
                if _red_out.dim:
                    cpp_out.set_dim(_red_out.dim)
                if _red_out.stride:
                    cpp_out.set_stride(_red_out.stride)
            elif node.node_type == NodeType.BLOCK_SCALE_DEQUANTIZE:
                cpp_out = graph.block_scale_dequantize(
                    input=tensor_map[node.inputs["input"].uid],
                    descale=tensor_map[node.inputs["descale"].uid],
                    block_size=node.params["block_size"],
                    is_negative_scale=node.params.get("is_negative_scale", False),
                    compute_data_type=node.compute_data_type,
                    name=node.name,
                )
            elif node.node_type == NodeType.MOE_GROUPED_MATMUL:
                cpp_out = graph.moe_grouped_matmul(
                    token=tensor_map[node.inputs["token"].uid],
                    weight=tensor_map[node.inputs["weight"].uid],
                    first_token_offset=tensor_map[node.inputs["first_token_offset"].uid],
                    mode=node.params.get("mode"),
                    name=node.name,
                )
            elif node.node_type == NodeType.BLOCK_SCALE_QUANTIZE:
                q_kwargs = {
                    "input": tensor_map[node.inputs["input"].uid],
                    "block_size": node.params["block_size"],
                    "transpose": node.params.get("transpose", False),
                    "compute_data_type": node.compute_data_type,
                    "name": node.name,
                }
                if node.params.get("axis") is not None:
                    q_kwargs["axis"] = node.params["axis"]
                quantized, scale = graph.block_scale_quantize(**q_kwargs)
                # two outputs: OUT_0 quantized, OUT_1 scale
                for out_t, cpp_t in ((node.outputs.get("OUT_0"), quantized), (node.outputs.get("OUT_1"), scale)):
                    if out_t is None:
                        continue
                    tensor_map[out_t.uid] = cpp_t
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

        return graph
