# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import glob
import os
import sys
import sysconfig
import importlib


def is_windows():
    return sys.platform.startswith("win")


module_name = ".Release._compiled_module" if is_windows() else "._compiled_module"

_pybind_module = importlib.import_module(module_name, package=__name__)

symbols_to_import = [
    "backend_version",
    "backend_version_string",
    "get_last_error_string",
    "norm_forward_phase",
    "reduction_mode",
    "behavior_note",
    "knob_type",
    "create_kernel_cache",
    "create_device_properties",
    "numerical_note",
    "build_plan_policy",
    "data_type",
    "tensor_reordering",
    "heur_mode",
    "tensor",
    "knob",
    "cudnnGraphNotSupportedError",
    "diagonal_alignment",
    "attention_implementation",
    "moe_grouped_matmul_mode",
    "scalar_type",
    "reshape_mode",
]

for symbol_name in symbols_to_import:
    globals()[symbol_name] = getattr(_pybind_module, symbol_name)

for _optional_symbol in [
    "causal_conv1d_forward",
    "causal_conv1d_backward",
    "causal_conv1d_nwh_forward",
    "causal_conv1d_nwh_backward",
    "b2b_causal_conv1d_forward",
    "b2b_causal_conv1d_backward",
    "gnn_agg_op",
    "gnn_agg_simple_forward",
    "gnn_agg_simple_backward",
    "fft_causal_conv1d_forward",
    "fft_causal_conv1d_backward",
    "long_fft_causal_conv1d_get_buffer_sizes",
    "long_fft_causal_conv1d_forward",
    "long_fft_causal_conv1d_backward",
]:
    if hasattr(_pybind_module, _optional_symbol):
        globals()[_optional_symbol] = getattr(_pybind_module, _optional_symbol)


from ._handle import Handle, DeviceInfo

# Type alias for the annotations that reference ``cudnn.handle`` (a supplied handle
# is a cudnn.Handle, or a bare int for a framework-created foreign handle).
handle = Handle


def create_handle():
    """Create a cuDNN handle, returned as a first-class :class:`cudnn.Handle`.

    The Handle wraps the backend ``cudnnHandle_t`` and is bound to the current
    CUDA device. Anywhere the backend needs the raw ``cudnnHandle_t`` it is
    extracted explicitly via ``to_backend_handle()`` (grep it to trace every
    handoff) -- the Handle is never silently coerced to an int, so a Handle that
    reaches a binding unconverted fails loudly rather than being magically cast.
    """
    raw = _pybind_module.create_handle()
    ordinal = None
    try:
        from .frost.device import current_device

        ordinal = current_device()
    except Exception:
        ordinal = None  # no GPU visible / cuda-python absent: resolve lazily on .device
    # Seed the stream from the backend's actual stream (a fresh handle runs on
    # stream 0) so a python plan and a backend plan on this handle agree on the
    # stream, instead of the python side falling back to torch's current stream.
    return Handle(raw, ordinal, _pybind_module.get_stream(raw))


def set_stream(handle, stream):
    """Set the CUDA stream a cuDNN handle runs on (wraps the compiled ``cudnnSetStream``).

    ``cudnnSetStream`` is not free: for a non-null stream it issues several CUDA driver queries
    on every call (green-context detection, stream priority, priority range) to maintain cuDNN's
    internal per-priority stream pool, even when the stream is unchanged -- ~2.4us/call on
    Blackwell. Frameworks that call this before every ``execute`` pay it every iteration, so the
    :class:`cudnn.Handle` remembers its last stream and skips the backend call when it has not
    changed; a steady-state loop pays it once. (Assumes a Handle is not driven from two streams
    concurrently, which is the normal single-stream case; a caller that does needs its own handle
    per stream regardless.)
    """
    if not isinstance(handle, Handle):
        raise TypeError(f"cudnn.set_stream expects a cudnn.Handle (from cudnn.create_handle()), got {type(handle).__name__}")
    if handle.stream == stream:
        return
    if handle.backend_handle is not None:
        _pybind_module._raw_set_stream(handle.backend_handle, stream)
    handle.stream = stream


def get_stream(handle):
    """The CUDA stream a :class:`cudnn.Handle` runs on -- the cached ``Handle.stream``, no
    backend round-trip."""
    if not isinstance(handle, Handle):
        raise TypeError(f"cudnn.get_stream expects a cudnn.Handle (from cudnn.create_handle()), got {type(handle).__name__}")
    return handle.stream


def destroy_handle(handle):
    """Destroy a :class:`cudnn.Handle` (wraps the compiled binding). The backend handle is cleared
    after destruction so a reused Handle object cannot pass a released ``cudnnHandle_t`` back to
    C++ (a double-destroy or a later set_stream)."""
    if not isinstance(handle, Handle):
        raise TypeError(f"cudnn.destroy_handle expects a cudnn.Handle (from cudnn.create_handle()), got {type(handle).__name__}")
    backend = handle.backend_handle
    if backend is None:
        handle.stream = None
        return None
    _pybind_module._raw_destroy_handle(backend)
    handle.backend_handle = None
    handle.stream = None
    return None


from .datatypes import _library_type, _is_torch_tensor

__version__ = "1.28.0"


def _tensor(
    self,
    dim,
    stride,
    data_type=data_type.NOT_SET,
    is_virtual=False,
    is_pass_by_value=False,
    ragged_offset=None,
    reordering_type=tensor_reordering.NONE,
    name="",
    uid=-1,
    ragged_offset_multiplier=1,
):
    """
    Create a tensor.

    Args:
        dim (List[int]): The dimensions of the tensor.
        stride (List[int]): The strides of the tensor.
        data_type (cudnn.data_type): The data type of the tensor.
        is_virtual (bool): Flag indicating if the tensor is virtual.
        is_pass_by_value (bool): Flag indicating if the tensor is passed by value.
        ragged_offset (cudnn_tensor): The ragged offset tensor.
        reordering_type (cudnn.tensor_reordering): The reordering type of the tensor.
        name (str): The name of the tensor.
        ragged_offset_multiplier (int): Unit size of ragged offsets in tensor elements. A value of 1 means no multiplier.

    Returns:
        cudnn_tensor: The created tensor.
    """
    return self._make_tensor(
        dim=dim,
        stride=stride,
        data_type=_library_type(data_type),
        is_virtual=is_virtual,
        is_pass_by_value=is_pass_by_value,
        ragged_offset=ragged_offset,
        reordering_type=reordering_type,
        name=name,
        uid=uid,
        ragged_offset_multiplier=ragged_offset_multiplier,
    )


def _set_data_type(
    self,
    data_type=data_type.NOT_SET,
):
    return self._set_data_type(_library_type(data_type))


_pybind_module.tensor.set_data_type = _set_data_type
_pybind_module.backend_graph.tensor = _tensor


def load_cudnn():
    # First look at python site packages
    lib_path = glob.glob(os.path.join(sysconfig.get_path("purelib"), "nvidia/cudnn/bin/cudnn64_9.dll"))

    if lib_path:
        assert len(lib_path) == 1, f"Found {len(lib_path)} libcudnn.dll.x in nvidia-cudnn-cuXX."
        lib = ctypes.windll.LoadLibrary(lib_path[0])
    else:  # Fallback
        lib = ctypes.windll.LoadLibrary("cudnn64_9.dll")

    handle = ctypes.cast(lib._handle, ctypes.c_void_p).value
    _pybind_module._set_dlhandle_cudnn(handle)


def _dlopen_cudnn():
    # Honor the dynamic linker search path before packaged cuDNN so local backend
    # builds can override the wheel dependency during development.
    for library_dir in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not library_dir:
            continue
        for library_name in ("libcudnn.so.9", "libcudnn.so"):
            library_path = os.path.join(library_dir, library_name)
            if not os.path.exists(library_path):
                continue
            lib = ctypes.CDLL(library_path)
            handle = ctypes.cast(lib._handle, ctypes.c_void_p).value
            _pybind_module._set_dlhandle_cudnn(handle)
            return

    # Then look at python site packages
    lib_path = glob.glob(os.path.join(sysconfig.get_path("purelib"), "nvidia/cudnn/lib/libcudnn.so.*[0-9]"))

    if not lib_path:
        lib_path = glob.glob(os.path.join(sysconfig.get_path("purelib"), "nvidia/cudnn_jit/lib/libcudnn.so.*[0-9]"))

    if lib_path:
        assert len(lib_path) == 1, f"Found {len(lib_path)} libcudnn.so.x in nvidia-cudnn-cuXX."
        lib = ctypes.CDLL(lib_path[0])
    else:  # Fallback
        try:
            lib = ctypes.CDLL("libcudnn.so.9")
        except Exception:
            try:
                lib = ctypes.CDLL("libcudnn.so")
            except Exception:
                lib = None

    if lib is not None:
        handle = ctypes.cast(lib._handle, ctypes.c_void_p).value
        _pybind_module._set_dlhandle_cudnn(handle)


if is_windows():
    load_cudnn()
else:
    _dlopen_cudnn()

# The graph API: a Python-native IR with pluggable execution backends. The
# public ``cudnn.pygraph`` IS the Python class; the C++ graph builder stays
# internal at ``cudnn._pybind_module.backend_graph`` and is reached only through
# lowering (a graph is pure-Python or pure-C++, never mixed). Imported before
# .graph/.wrapper, which reference cudnn.pygraph at module load.
from .graph_types import NodeType, Tensor
from ._pygraph import pygraph, GraphContext
from .nodes import Node

from .graph import graph, jit, graph_cache

from typing import Any

_EAGER_PUBLIC_NAMES = (
    *symbols_to_import,
    *(
        symbol
        for symbol in (
            "causal_conv1d_forward",
            "causal_conv1d_backward",
            "causal_conv1d_nwh_forward",
            "causal_conv1d_nwh_backward",
            "b2b_causal_conv1d_forward",
            "b2b_causal_conv1d_backward",
            "gnn_agg_op",
            "gnn_agg_simple_forward",
            "gnn_agg_simple_backward",
        )
        if symbol in globals()
    ),
    "create_handle",
    "destroy_handle",
    "get_stream",
    "set_stream",
    "Handle",
    "DeviceInfo",
    "__version__",
    "NodeType",
    "Tensor",
    "pygraph",
    "GraphContext",
    "Node",
    "graph",
    "jit",
    "graph_cache",
)
__all__ = [*_EAGER_PUBLIC_NAMES, "Graph", "wrapper"]

_OPTIONAL_DEPENDENCY_INSTALL_HINT = "Install with 'pip install nvidia-cudnn-frontend[cutedsl]'"

_LAZY_OPTIONAL_IMPORTS = {
    "gnn": (".gnn", None),
    "BSA": (".block_sparse_attention", "BSA"),
    "block_sparse_attention_forward": (".block_sparse_attention", "block_sparse_attention_forward"),
    "block_sparse_attention_backward": (".block_sparse_attention", "block_sparse_attention_backward"),
    "DSA": (".deepseek_sparse_attention", "DSA"),
    "CSA": (".csa", "CSA"),
    "CSACompressorForward": (".csa", "CSACompressorForward"),
    "CSACompressorBackward": (".csa", "CSACompressorBackward"),
    "csa_compressor_forward_wrapper": (".csa", "csa_compressor_forward_wrapper"),
    "csa_compressor_backward_wrapper": (".csa", "csa_compressor_backward_wrapper"),
    "NSA": (".native_sparse_attention", "NSA"),
    "GemmSwigluSm100": (".gemm.cutedsl.dense.swiglu", "GemmSwigluSm100"),
    "gemm_swiglu_wrapper_sm100": (".gemm.cutedsl.dense.swiglu", "gemm_swiglu_wrapper_sm100"),
    "gemm_swiglu_jax_sm100": (".gemm.cutedsl.dense.swiglu", "gemm_swiglu_jax_sm100"),
    "gemm_srelu_jax_sm100": (".gemm.cutedsl.dense.srelu", "gemm_srelu_jax_sm100"),
    "gemm_dsrelu_jax_sm100": (".gemm.cutedsl.dense.dsrelu", "gemm_dsrelu_jax_sm100"),
    "GemmSreluSm100": (".gemm.cutedsl.dense.srelu", "GemmSreluSm100"),
    "gemm_srelu_wrapper_sm100": (".gemm.cutedsl.dense.srelu", "gemm_srelu_wrapper_sm100"),
    "GemmDsreluSm100": (".gemm.cutedsl.dense.dsrelu", "GemmDsreluSm100"),
    "gemm_dsrelu_wrapper_sm100": (".gemm.cutedsl.dense.dsrelu", "gemm_dsrelu_wrapper_sm100"),
    "GemmAmaxSm100": (".gemm.cutedsl.dense.amax", "GemmAmaxSm100"),
    "gemm_amax_wrapper_sm100": (".gemm.cutedsl.dense.amax", "gemm_amax_wrapper_sm100"),
    "gemm_amax_jax_sm100": (".gemm.cutedsl.dense.amax", "gemm_amax_jax_sm100"),
    "GemmProjRopeMxfp8Bf16InSm100": (".gemm.cutedsl.dense.proj_rope_mxfp8", "GemmProjRopeMxfp8Bf16InSm100"),
    "GemmProjRopeMxfp8Mxfp8InSm100": (".gemm.cutedsl.dense.proj_rope_mxfp8", "GemmProjRopeMxfp8Mxfp8InSm100"),
    "gemm_proj_rope_mxfp8_wrapper_sm100": (".gemm.cutedsl.dense.proj_rope_mxfp8", "gemm_proj_rope_mxfp8_wrapper_sm100"),
    "gemm_proj_rope_mxfp8_jax_sm100": (".gemm.cutedsl.dense.proj_rope_mxfp8", "gemm_proj_rope_mxfp8_jax_sm100"),
    "RmsNormRhtAmaxSm100": (".rmsnorm_rht_amax", "RmsNormRhtAmaxSm100"),
    "rmsnorm_rht_amax_wrapper_sm100": (".rmsnorm_rht_amax", "rmsnorm_rht_amax_wrapper_sm100"),
    "grouped_gemm": (".gemm.cutedsl.grouped", None),
    "GroupedGemmSm100": (".gemm.cutedsl.grouped", "GroupedGemmSm100"),
    "grouped_gemm_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_wrapper_sm100"),
    "grouped_gemm_jax_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_jax_sm100"),
    "grouped_gemm_glu_jax_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_glu_jax_sm100"),
    "grouped_gemm_dglu_jax_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_dglu_jax_sm100"),
    "grouped_gemm_dsrelu_jax_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_dsrelu_jax_sm100"),
    "grouped_gemm_wgrad_jax_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_wgrad_jax_sm100"),
    "discrete_grouped_gemm_swiglu_jax_sm100": (".gemm.cutedsl.discrete_grouped", "discrete_grouped_gemm_swiglu_jax_sm100"),
    "discrete_grouped_gemm_dswiglu_jax_sm100": (".gemm.cutedsl.discrete_grouped", "discrete_grouped_gemm_dswiglu_jax_sm100"),
    "GroupedGemmSwigluSm100": (".gemm.cutedsl.grouped", "GroupedGemmSwigluSm100"),
    "grouped_gemm_swiglu_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_swiglu_wrapper_sm100"),
    "GroupedGemmDswigluSm100": (".gemm.cutedsl.grouped", "GroupedGemmDswigluSm100"),
    "grouped_gemm_dswiglu_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_dswiglu_wrapper_sm100"),
    "GroupedGemmSreluSm100": (".gemm.cutedsl.grouped", "GroupedGemmSreluSm100"),
    "grouped_gemm_srelu_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_srelu_wrapper_sm100"),
    "GroupedGemmDsreluSm100": (".gemm.cutedsl.grouped", "GroupedGemmDsreluSm100"),
    "grouped_gemm_dsrelu_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_dsrelu_wrapper_sm100"),
    "GroupedGemmQuantSm100": (".gemm.cutedsl.grouped", "GroupedGemmQuantSm100"),
    "grouped_gemm_quant_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_quant_wrapper_sm100"),
    "GroupedGemmGluSm100": (".gemm.cutedsl.grouped", "GroupedGemmGluSm100"),
    "grouped_gemm_glu_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_glu_wrapper_sm100"),
    "GroupedGemmGluHadamardSm100": (".gemm.cutedsl.grouped", "GroupedGemmGluHadamardSm100"),
    "grouped_gemm_glu_hadamard_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_glu_hadamard_wrapper_sm100"),
    "GroupedGemmGluHadamardQuantSm100": (".gemm.cutedsl.grouped", "GroupedGemmGluHadamardQuantSm100"),
    "grouped_gemm_glu_hadamard_quant_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_glu_hadamard_quant_wrapper_sm100"),
    "GroupedGemmDgluSm100": (".gemm.cutedsl.grouped", "GroupedGemmDgluSm100"),
    "grouped_gemm_dglu_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_dglu_wrapper_sm100"),
    "GroupedGemmWgradSm100": (".gemm.cutedsl.grouped", "GroupedGemmWgradSm100"),
    "grouped_gemm_wgrad_wrapper_sm100": (".gemm.cutedsl.grouped", "grouped_gemm_wgrad_wrapper_sm100"),
    "discrete_grouped_gemm": (".gemm.cutedsl.discrete_grouped", None),
    "DiscreteGroupedGemmSwigluSm100": (".gemm.cutedsl.discrete_grouped", "DiscreteGroupedGemmSwigluSm100"),
    "discrete_grouped_gemm_swiglu_wrapper_sm100": (".gemm.cutedsl.discrete_grouped", "discrete_grouped_gemm_swiglu_wrapper_sm100"),
    "DiscreteGroupedGemmDswigluSm100": (".gemm.cutedsl.discrete_grouped", "DiscreteGroupedGemmDswigluSm100"),
    "discrete_grouped_gemm_dswiglu_wrapper_sm100": (".gemm.cutedsl.discrete_grouped", "discrete_grouped_gemm_dswiglu_wrapper_sm100"),
}


def _load_optional_symbol(name: str) -> Any:
    module_name, attr_name = _LAZY_OPTIONAL_IMPORTS[name]
    try:
        module = importlib.import_module(module_name, package=__name__)
        value = module if attr_name is None else getattr(module, attr_name)
    except Exception as e:
        raise ImportError(f"{name} requires optional dependencies. {_OPTIONAL_DEPENDENCY_INSTALL_HINT}: {e}") from e

    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in ("Graph", "wrapper"):
        _wrapper = importlib.import_module(".wrapper", __name__)
        globals()["wrapper"] = _wrapper
        globals()["Graph"] = _wrapper.Graph
        return globals()[name]

    if name == "ops":
        # Use importlib rather than "from . import ops" to avoid infinite
        # recursion. The cycle:
        #   1. cudnn.ops accessed → __getattr__("ops") fires
        #   2. "from . import ops" → _handle_fromlist(cudnn, ["ops"], ...)
        #   3. _handle_fromlist calls hasattr(cudnn, "ops")
        #   4. "ops" not in __dict__ yet → __getattr__("ops") again → goto 1
        # importlib.import_module bypasses _handle_fromlist entirely.
        _ops = importlib.import_module(".ops", __name__)
        globals()["ops"] = _ops
        return _ops

    if name == "experimental":
        from . import experimental as _experimental

        globals()["experimental"] = _experimental
        return _experimental

    if name == "jax":
        # `import cudnn; cudnn.jax.call` works like `import cudnn.jax`.
        # Deferred so torch-only users never pay the jax import (the submodule
        # itself raises a descriptive ImportError when jax >= 0.5 is missing).
        _jax = importlib.import_module(".jax", __name__)
        globals()["jax"] = _jax
        return _jax

    if name == "fla":
        # `import cudnn; cudnn.fla.accelerate_fla()` works like `import cudnn.fla`.
        # Deferred so `import cudnn` never eagerly imports torch / the FLA shim.
        _fla = importlib.import_module(".fla", __name__)
        globals()["fla"] = _fla
        return _fla

    if name in _LAZY_OPTIONAL_IMPORTS:
        return _load_optional_symbol(name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
