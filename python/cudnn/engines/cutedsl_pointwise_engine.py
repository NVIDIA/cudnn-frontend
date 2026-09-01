"""A CuTeDSL engine for elementwise-add graphs, and the AOT export vehicle.

This is deliberately the smallest real CuTeDSL engine that can exist: it takes a
single-node POINTWISE ADD graph, JITs one CuTe kernel for it, and knows how to
turn that compiled kernel into an artifact the C++ side can load with no Python
and no compiler.

It exists for two reasons. It is the reference implementation of the AOT half of
the engine contract (``CompiledPlan.export_aot_payload``), which is the whole
per-engine plugin interface; and it is the vehicle the AOT round-trip tests run
on, since the graph it accepts is also one the cuDNN backend can run, so every
result has an oracle.

Real CuTeDSL engines (frost's GEMM family, the SDPA/grouped-GEMM kernels behind
``cudnn.experimental.ops``) plug into the same two methods.
"""

import sys
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from .base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig
from .cutedsl_aot import Step, base_payload, link_steps, register_steps, tensor_arg
from ..graph_types import NodeType

if TYPE_CHECKING:
    from .._pygraph import pygraph

THREADS_PER_BLOCK = 256

# cuDNN data_type -> (cute dtype attribute name, torch dtype name). Kept to the
# types this toy kernel is meaningful for; the payload schema itself is not
# limited to these.
_SUPPORTED_DTYPES = {
    "FLOAT": "Float32",
    "HALF": "Float16",
    "BFLOAT16": "BFloat16",
}


def _dtype_name(dt: Any) -> str:
    return getattr(dt, "name", str(dt)).upper().replace("DATA_TYPE.", "")


def _is_contiguous(dim: List[int], stride: List[int]) -> bool:
    expected = 1
    for d, s in zip(reversed(dim), reversed(stride)):
        if s != expected:
            return False
        expected *= d
    return True


def _num_elements(dim: List[int]) -> int:
    n = 1
    for d in dim:
        n *= d
    return n


def _build_kernel(cute, dtype):
    """The kernel and its host wrapper, built against a cute dtype."""

    @cute.kernel
    def add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        tid = bidx * THREADS_PER_BLOCK + tidx
        if tid < a.shape[0]:
            c[tid] = a[tid] + b[tid]

    @cute.jit
    def add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        n = a.shape[0]
        blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        add_kernel(a, b, c).launch(grid=(blocks, 1, 1), block=(THREADS_PER_BLOCK, 1, 1))

    _ = dtype  # the dtype is carried by the fake tensors passed to cute.compile
    return add


class CuteDslPointwiseAddCompiledPlan(CompiledPlan):
    """One JIT-compiled elementwise add, plus its AOT artifact."""

    def __init__(self, engine, plan, uids: Tuple[int, int, int], n: int, cute_dtype_name: str, cudnn_dtype: Any):
        self.engine = engine
        self.plan = plan
        self.a_uid, self.b_uid, self.c_uid = uids
        self.n = n
        self.cute_dtype_name = cute_dtype_name
        self.cudnn_dtype = cudnn_dtype
        self._compiled = None

    def _compile(self):
        if self._compiled is not None:
            return self._compiled
        import cutlass
        import cutlass.cute as cute

        dtype = getattr(cutlass, self.cute_dtype_name)
        fn = _build_kernel(cute, dtype)
        fakes = [cute.runtime.make_fake_compact_tensor(dtype, (self.n,)) for _ in range(3)]
        # --enable-tvm-ffi is what makes the compiled object callable across the
        # C ABI, and what export_to_c will emit an entry point for.
        self._compiled = cute.compile(fn, *fakes, options="--enable-tvm-ffi")
        return self._compiled

    def get_workspace_size(self) -> int:
        return 0

    def execute(self, graph, tensor_data: Dict[int, Any], ctx: ExecutionContext) -> None:
        compiled = self._compile()
        a = tensor_data[self.a_uid]
        b = tensor_data[self.b_uid]
        c = tensor_data[self.c_uid]
        compiled(a.reshape(-1), b.reshape(-1), c.reshape(-1))

    # ---------------------------------------------------------------- AOT ----

    def _steps(self):
        """One launch. The kernel was compiled against a flat 1-D view, so that
        is the shape the artifact hands it whatever rank the graph tensor has:
        the graph knows the uid order, only the engine knows the signature."""
        return [
            Step(
                "add",
                self._compile(),
                [tensor_arg(uid, self.cudnn_dtype, (self.n,)) for uid in (self.a_uid, self.b_uid, self.c_uid)],
            )
        ]

    def export_aot_payload(self, graph: "pygraph") -> Tuple[Dict[str, Any], bytes]:
        steps, module_bytes, runtime_deps = link_steps(self._steps(), graph.get_name())
        payload = base_payload(self.get_workspace_size())
        payload["steps"] = steps
        payload["runtime_deps"] = runtime_deps
        return payload, module_bytes

    def aot_global_payload(self, graph: "pygraph", symbol: str):
        payload = base_payload(self.get_workspace_size())
        payload["steps"] = register_steps(self._steps(), symbol)
        return payload


class CuteDslPointwiseAddEngine(BaseEngine):
    """CuTeDSL elementwise add: one POINTWISE ADD node, contiguous, on CUDA."""

    name = "cutedsl_pointwise_add"

    def __init__(self, engine_id: int):
        super().__init__()
        self.engine_id = engine_id

    def check_support(self, graph: "pygraph") -> None:
        if "cutlass" not in sys.modules:
            try:
                import cutlass  # noqa: F401
            except ImportError as e:
                raise NotImplementedError(f"CuteDslPointwiseAddEngine needs nvidia-cutlass-dsl: {e}")

        nodes = graph.nodes
        if len(nodes) != 1:
            raise NotImplementedError(f"CuteDslPointwiseAddEngine handles a single-node graph, got {len(nodes)}")
        node = nodes[0]
        if node.node_type != NodeType.POINTWISE or node.params.get("mode") != "add":
            raise NotImplementedError("CuteDslPointwiseAddEngine handles POINTWISE add only")
        if len(node.inputs) != 2:
            raise NotImplementedError("CuteDslPointwiseAddEngine needs exactly two inputs")

        tensors = [node.inputs["IN_0"], node.inputs["IN_1"], node.outputs["OUT_0"]]
        dtypes = {_dtype_name(t.data_type) for t in tensors}
        if len(dtypes) != 1 or next(iter(dtypes)) not in _SUPPORTED_DTYPES:
            raise NotImplementedError(f"CuteDslPointwiseAddEngine: unsupported/mixed data types {dtypes}")
        shapes = {tuple(t.dim) for t in tensors}
        if len(shapes) != 1:
            raise NotImplementedError(f"CuteDslPointwiseAddEngine needs identical shapes, got {shapes}")
        for t in tensors:
            if not _is_contiguous(list(t.dim), list(t.stride)):
                raise NotImplementedError(f"CuteDslPointwiseAddEngine needs contiguous tensors, {t.name} is not")


    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        self.check_support(graph)
        node = graph.nodes[0]
        a, b = node.inputs["IN_0"], node.inputs["IN_1"]
        c = node.outputs["OUT_0"]
        cudnn_dtype = a.data_type
        compiled = CuteDslPointwiseAddCompiledPlan(
            self,
            plan,
            (a.uid, b.uid, c.uid),
            _num_elements(list(a.dim)),
            _SUPPORTED_DTYPES[_dtype_name(cudnn_dtype)],
            cudnn_dtype,
        )
        compiled._compile()  # build_plans() is where the JIT cost belongs
        return compiled
