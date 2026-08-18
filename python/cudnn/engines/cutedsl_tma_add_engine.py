"""A CuTeDSL engine for 2-D elementwise add whose operands arrive by TMA.

The vehicle behind ``samples/aot``: the smallest engine that is still
representative. A flat vector add builds no TMA descriptors, so it understates
what a call costs and teaches nothing about the part that grows with a real
kernel. This one is the same arithmetic over 2-D tiles with the operands loaded
through TMA, which is enough to exercise descriptor construction, shared memory,
an mbarrier and an explicit stream argument, and still fit on a screen.

Descriptors are built by ``cpasync.make_tiled_tma_atom`` in the HOST half of the
jit function, so once per call, against the pointers that actually arrived --
not once at compile time. ``samples/aot/bench_cpu_costs`` prices that against
``cutedsl_pointwise_engine``, which is the same add with the descriptors taken
away.

Four things here compile clean and fail at runtime. They are commented inline
next to the code that avoids them, because each one cost real debugging time:
the TMA copy must not be predicated, ``tma_partition`` wants a CTA-tiled gmem
tensor, a host-built layout cannot enter the kernel region, and the fake tensors
used at compile time must carry the caller's stride order.
"""

import sys
from typing import TYPE_CHECKING, Any, Dict, Tuple

from .base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig
from .cutedsl_aot import Step, base_payload, link_steps, register_steps, tensor_arg
from ..graph_types import NodeType

if TYPE_CHECKING:
    from .._pygraph import pygraph

BM, BN = 128, 64
THREADS = 128


def dtype_name(dt: Any) -> str:
    return getattr(dt, "name", str(dt)).upper().replace("DATA_TYPE.", "")


def build_kernel():
    """The kernel and its host wrapper. Imports live here so the engine module
    stays importable without cutlass."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import cpasync

    @cute.kernel
    def tma_add_kernel(tma_a, tma_b, gA: cute.Tensor, gB: cute.Tensor, mC: cute.Tensor):
        smem_layout = cute.make_ordered_layout((BM, BN), order=(1, 0))
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=1024)
        sB = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=1024)
        mbar = smem.allocate_array(cutlass.Int64, 1)

        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # Partitioned outside any conditional: a value produced inside a region
        # cannot be consumed outside it.
        tAsA, tAgA = cpasync.tma_partition(
            tma_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(cute.local_tile(gA, (BM, BN), (bidx, bidy)), 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(cute.local_tile(gB, (BM, BN), (bidx, bidy)), 0, 2),
        )

        # One warp issues. expect_tx is elected to a single lane; the copies are
        # NOT inside the elect -- a TMA copy is warp-uniform and the DSL emits
        # the single issue itself. A copy inside elect_one deadlocks, because
        # the transaction count never reaches what was expected.
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, BM * BN * 4 * 2)
            cute.copy(tma_a, tAgA, tAsA, tma_bar_ptr=mbar)
            cute.copy(tma_b, tBgB, tBsB, tma_bar_ptr=mbar)
        cute.arch.mbarrier_wait(mbar, 0)

        for e in cutlass.range_constexpr(BM * BN // THREADS):
            flat = e * THREADS + tidx
            r = flat // BN
            c = flat % BN
            mC[(bidx * BM + r, bidy * BN + c)] = sA[(r, c)] + sB[(r, c)]

    @cute.jit
    def tma_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, stream: cuda.CUstream):
        smem_layout = cute.make_ordered_layout((BM, BN), order=(1, 0))
        op = cpasync.CopyBulkTensorTileG2SOp()
        # The cost under test: one host-side descriptor encode per operand, on
        # every call.
        tma_a, gA = cpasync.make_tiled_tma_atom(op, mA, smem_layout, (BM, BN))
        tma_b, gB = cpasync.make_tiled_tma_atom(op, mB, smem_layout, (BM, BN))
        m, n = mC.shape
        tma_add_kernel(tma_a, tma_b, gA, gB, mC).launch(
            grid=(m // BM, n // BN, 1), block=(THREADS, 1, 1), stream=stream
        )

    return tma_add


class CuteDslTmaAddCompiledPlan(CompiledPlan):
    """One JIT-compiled TMA tile add, plus its AOT artifact."""

    def __init__(self, engine, plan, uids, shape, cudnn_dtype):
        self.engine = engine
        self.plan = plan
        self.a_uid, self.b_uid, self.c_uid = uids
        self.shape = tuple(int(s) for s in shape)
        self.cudnn_dtype = cudnn_dtype
        self._compiled = None

    def _compile(self):
        if self._compiled is not None:
            return self._compiled
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute

        fn = build_kernel()
        # stride_order (1, 0) = row-major, matching the contiguous stride the
        # graph declares. The default is the other order, which compiles fine
        # and then rejects a row-major caller at the ABI boundary.
        fakes = [
            cute.runtime.make_fake_compact_tensor(cutlass.Float32, self.shape, stride_order=(1, 0))
            for _ in range(3)
        ]
        self._compiled = cute.compile(fn, *fakes, cuda.CUstream(0), options="--enable-tvm-ffi")
        return self._compiled

    def get_workspace_size(self) -> int:
        return 0

    def execute(self, graph, tensor_data: Dict[int, Any], ctx: ExecutionContext) -> None:
        import cuda.bindings.driver as cuda

        compiled = self._compile()
        stream = getattr(ctx, "stream", None) or 0
        compiled(tensor_data[self.a_uid], tensor_data[self.b_uid], tensor_data[self.c_uid], cuda.CUstream(int(stream)))

    # ---------------------------------------------------------------- AOT ----

    def _steps(self):
        """This kernel is one launch; the shared exporter takes a list."""
        return [
            Step(
                "tile_add",
                self._compile(),
                [
                    tensor_arg(self.a_uid, self.cudnn_dtype, self.shape),
                    tensor_arg(self.b_uid, self.cudnn_dtype, self.shape),
                    tensor_arg(self.c_uid, self.cudnn_dtype, self.shape),
                    # A positional handle, not the tvm-ffi environment stream:
                    # the jit function takes a CUstream.
                    {"kind": "STREAM"},
                ],
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


class CuteDslTmaAddEngine(BaseEngine):
    """CuTeDSL 2-D elementwise add with TMA-loaded operands."""

    name = "cutedsl_tma_add"

    def __init__(self, engine_id: int):
        super().__init__()
        self.engine_id = engine_id

    def check_support(self, graph: "pygraph") -> None:
        if "cutlass" not in sys.modules:
            try:
                import cutlass  # noqa: F401
            except ImportError as e:
                raise NotImplementedError(f"CuteDslTmaAddEngine needs nvidia-cutlass-dsl: {e}")

        nodes = graph.nodes
        if len(nodes) != 1:
            raise NotImplementedError(f"CuteDslTmaAddEngine handles a single-node graph, got {len(nodes)}")
        node = nodes[0]
        if node.node_type != NodeType.POINTWISE or node.params.get("mode") != "add":
            raise NotImplementedError("CuteDslTmaAddEngine handles POINTWISE add only")

        tensors = [node.inputs["IN_0"], node.inputs["IN_1"], node.outputs["OUT_0"]]
        for t in tensors:
            if dtype_name(t.data_type) != "FLOAT":
                raise NotImplementedError(f"CuteDslTmaAddEngine is fp32 only, {t.name} is {t.data_type}")
            if len(t.dim) != 2:
                raise NotImplementedError(f"CuteDslTmaAddEngine needs rank-2 tensors, {t.name} has rank {len(t.dim)}")
        shapes = {tuple(t.dim) for t in tensors}
        if len(shapes) != 1:
            raise NotImplementedError(f"CuteDslTmaAddEngine needs identical shapes, got {shapes}")
        m, n = next(iter(shapes))
        if m % BM or n % BN:
            raise NotImplementedError(f"CuteDslTmaAddEngine needs shapes divisible by the {BM}x{BN} tile, got {m}x{n}")


    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        self.check_support(graph)
        node = graph.nodes[0]
        a, b = node.inputs["IN_0"], node.inputs["IN_1"]
        c = node.outputs["OUT_0"]
        compiled = CuteDslTmaAddCompiledPlan(self, plan, (a.uid, b.uid, c.uid), a.dim, a.data_type)
        compiled._compile()  # build_plans() is where the JIT cost belongs
        return compiled
