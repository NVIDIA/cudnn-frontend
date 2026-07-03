"""Matmul cuTile execution engine using NVIDIA CUDA Tile.

This engine uses cuTile for high-performance matmul execution.
Requires Blackwell GPU (SM100+), CUDA Toolkit 13.1+, and cuda-tile package.

The caller must provide pre-allocated output tensors. The engine writes
results directly into these buffers (matching cuDNN's execution model).

Example Usage:
    import torch
    from cudnn import pygraph

    a = torch.randn(2, 3, 4, device="cuda")
    b = torch.randn(2, 4, 5, device="cuda")
    c = torch.empty(2, 3, 5, device="cuda")

    graph = pygraph(use_native=True)
    C = graph.matmul(a, b)  # pass torch tensors directly
    graph.execute({C: c})   # leaf outputs auto-detected, inputs auto-bound

Install:
    pip install nvidia-cudnn-frontend[cutile]
"""

from typing import TYPE_CHECKING, Any, Dict, List

try:
    import cuda.tile as ct
except ImportError:
    ct = None

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    cudart = None

from .base import BaseEngine
from .engine_ids import PYTHON_ENGINE_ID_BASE
from ..graph_types import NodeType

if TYPE_CHECKING:
    from ..pygraph import pygraph


# Tile sizes for matmul kernel
TM, TN, TK = 128, 128, 32


def _is_row_major(dim: List[int], stride: List[int]) -> bool:
    """Check if tensor has row-major contiguous layout."""
    if stride[-1] != 1:
        return False
    expected = 1
    for i in range(len(dim) - 1, -1, -1):
        if stride[i] != expected:
            return False
        expected *= dim[i]
    return True


# Kernel cache - lazy initialization
_kernel_cache: Dict[str, Any] = {}


def _get_matmul_kernel():
    """Get or create the 2D matmul kernel."""
    if "matmul" not in _kernel_cache:

        @ct.kernel
        def matmul_kernel(A, B, C, M: ct.Constant, N: ct.Constant, K: ct.Constant, tm: ct.Constant, tn: ct.Constant, tk: ct.Constant):
            """Tiled matrix multiplication kernel: C = A @ B."""
            # Simple 2D grid indexing
            tile_m = ct.bid(0)
            tile_n = ct.bid(1)
            num_tiles_k = ct.cdiv(K, tk)

            # Initialize accumulator
            accumulator = ct.full((tm, tn), 0, dtype=ct.float32)

            # Main loop over K dimension
            for k in range(num_tiles_k):
                a_tile = ct.load(A, index=(tile_m, k), shape=(tm, tk))
                b_tile = ct.load(B, index=(k, tile_n), shape=(tk, tn))
                accumulator = ct.mma(a_tile, b_tile, accumulator)

            # Store result
            ct.store(C, index=(tile_m, tile_n), tile=accumulator)

        _kernel_cache["matmul"] = matmul_kernel
    return _kernel_cache["matmul"]


def _get_batched_matmul_kernel():
    """Get or create the 3D batched matmul kernel."""
    if "batched_matmul" not in _kernel_cache:

        @ct.kernel
        def batched_matmul_kernel(
            A, B, C, batch: ct.Constant, M: ct.Constant, N: ct.Constant, K: ct.Constant, tm: ct.Constant, tn: ct.Constant, tk: ct.Constant
        ):
            """Batched tiled matrix multiplication kernel: C[b] = A[b] @ B[b]."""
            # Batch index from grid z dimension
            b = ct.bid(2)
            tile_m = ct.bid(0)
            tile_n = ct.bid(1)
            num_tiles_k = ct.cdiv(K, tk)

            # Initialize accumulator
            accumulator = ct.full((tm, tn), 0, dtype=ct.float32)

            # Main loop over K dimension
            for k in range(num_tiles_k):
                a_tile = ct.load(A, index=(b, tile_m, k), shape=(1, tm, tk))
                b_tile = ct.load(B, index=(b, k, tile_n), shape=(1, tk, tn))
                # Squeeze batch dim for mma
                a_tile = ct.reshape(a_tile, (tm, tk))
                b_tile = ct.reshape(b_tile, (tk, tn))
                accumulator = ct.mma(a_tile, b_tile, accumulator)

            # Store result
            c_tile = ct.reshape(accumulator, (1, tm, tn))
            ct.store(C, index=(b, tile_m, tile_n), tile=c_tile)

        _kernel_cache["batched_matmul"] = batched_matmul_kernel
    return _kernel_cache["batched_matmul"]


class MatmulCuTileEngine(BaseEngine):
    """cuTile engine for high-performance matmul execution.

    Uses NVIDIA CUDA Tile for tiled matrix operations with automatic
    tensor core utilization on supported hardware.

    Requirements:
        - Blackwell GPU (SM100+)
        - CUDA Toolkit 13.1+
        - cuda-tile package: pip install cuda-tile
    """

    name = "matmul_cutile"
    engine_id = PYTHON_ENGINE_ID_BASE + 1  # stable id

    def __init__(self, device: str = "cuda"):
        super().__init__()
        if ct is None:
            raise ImportError("MatmulCuTileEngine requires cuda-tile package. " "Install with: pip install nvidia-cudnn-frontend[cutile]")
        if cudart is None:
            raise ImportError("MatmulCuTileEngine requires cuda-python package. " "Install with: pip install cuda-python")
        self.device = device

    def check_support(self, graph: "pygraph") -> None:
        """Check hardware requirements and that graph only contains MATMUL nodes.

        Raises:
            RuntimeError: If GPU or driver doesn't meet requirements
            NotImplementedError: If graph contains unsupported operations
        """
        # Check GPU compute capability (need SM100+ for Blackwell). CUDA
        # runtime failures decline the engine (never proceed on garbage).
        err, device_id = cudart.cudaGetDevice()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"MatmulCuTileEngine: cudaGetDevice failed ({err})")  # runtime error, not a decline
        err, props = cudart.cudaGetDeviceProperties(device_id)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"MatmulCuTileEngine: cudaGetDeviceProperties failed ({err})")
        cc_int = props.major * 10 + props.minor
        if cc_int < 100:
            raise NotImplementedError(f"MatmulCuTileEngine requires Blackwell GPU (SM100+), got SM{cc_int}")

        # Check driver version (need r580+)
        err, driver_version = cudart.cudaDriverGetVersion()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"MatmulCuTileEngine: cudaDriverGetVersion failed ({err})")
        # Driver version format: 1000 * major + 10 * minor
        # r580 corresponds to CUDA 13.1 which is driver version 13010
        if driver_version < 13010:
            raise NotImplementedError(f"MatmulCuTileEngine requires NVIDIA driver r580+ (CUDA 13.1+), got driver version {driver_version}")

        # Check graph operations and tensor layouts
        for node in graph.nodes:
            if node.node_type != NodeType.MATMUL:
                raise NotImplementedError(f"MatmulCuTileEngine only supports MATMUL, got {node.node_type.name}")

            a_desc = node.inputs["A"]
            b_desc = node.inputs["B"]
            c_desc = node.outputs["C"]

            # cuTile kernels require row-major contiguous layout
            for name, desc in [("A", a_desc), ("B", b_desc), ("C", c_desc)]:
                if not _is_row_major(desc.dim, desc.stride):
                    raise NotImplementedError(
                        f"MatmulCuTileEngine requires row-major contiguous layout for tensor '{name}' (dim={desc.dim}, stride={desc.stride})"
                    )

    def execute(self, graph, tensor_data: Dict[int, Any], ctx=None) -> None:
        """Execute the graph using cuTile kernels.

        Writes results directly into the caller-provided output tensors.
        All output tensor UIDs must be present in tensor_data.
        """
        for node in graph.nodes:
            a = tensor_data[node.inputs["A"].uid]
            b = tensor_data[node.inputs["B"].uid]
            c = tensor_data[node.outputs["C"].uid]

            # all operands must live on the same CUDA device (multi-GPU hosts:
            # launching against a mismatched context silently corrupts results)
            devices = {getattr(t, "device", None) for t in (a, b, c)}
            if len(devices) != 1 or getattr(next(iter(devices)), "type", None) != "cuda":
                raise RuntimeError(f"MatmulCuTileEngine: operands must share one CUDA device, got {devices}")

            if ctx is not None and ctx.stream is not None:
                stream = ctx.stream  # the caller handle's stream
            else:
                # no handle supplied: resolve from the framework on the
                # OPERANDS' device — argless current_stream() is the active
                # device's stream, which can be a different GPU
                import torch

                stream = torch.cuda.current_stream(a.device).cuda_stream

            # Get dimensions and launch kernel
            if a.ndim == 2:
                M, K = a.shape
                K2, N = b.shape
                assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

                grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
                ct.launch(stream, grid, _get_matmul_kernel(), (a, b, c, M, N, K, TM, TN, TK))

            elif a.ndim == 3:
                batch, M, K = a.shape
                batch2, K2, N = b.shape
                assert batch == batch2, f"Batch sizes must match: {batch} vs {batch2}"
                assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

                grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), batch)
                ct.launch(stream, grid, _get_batched_matmul_kernel(), (a, b, c, batch, M, N, K, TM, TN, TK))
            else:
                raise ValueError(f"Unsupported tensor dimensions: {a.ndim}")
