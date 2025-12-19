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
    "destroy_handle",
    "norm_forward_phase",
    "reduction_mode",
    "behavior_note",
    "knob_type",
    "create_handle",
    "create_kernel_cache",
    "create_device_properties",
    "get_stream",
    "numerical_note",
    "set_stream",
    "build_plan_policy",
    "data_type",
    "tensor_reordering",
    "heur_mode",
    "pygraph",
    "tensor",
    "knob",
    "cudnnGraphNotSupportedError",
    "diagonal_alignment",
    "attention_implementation",
]

for symbol_name in symbols_to_import:
    globals()[symbol_name] = getattr(_pybind_module, symbol_name)

from .datatypes import _library_type, _is_torch_tensor

__version__ = "1.17.0"


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
    )


def _set_data_type(
    self,
    data_type=data_type.NOT_SET,
):
    return self._set_data_type(_library_type(data_type))


_pybind_module.tensor.set_data_type = _set_data_type
pygraph.tensor = _tensor


def _library_device_pointer(input_tensor):
    # either pass in pointers directly
    if type(input_tensor) is int:
        return input_tensor
    # directly extract data pointer for torch tensors
    elif _is_torch_tensor(input_tensor):
        return input_tensor.data_ptr()
    # fall back to dlpack support by library
    else:
        return _pybind_module._get_data_ptr(input_tensor)


def _execute(self, tensor_to_device_buffer, workspace, handle=None):
    """
    Execute a cudnn graph.

    Args:
        tensor_to_device_buffer (dict(cudnn_tensor, Union[torch.Tensor, int, __dlpack__])): The dimensions of the tensor.
        workspace (Union[torch.Tensor, int, __dlpack__]): The name of the tensor.
        handle: cudnn_handle created with cudnn.create_handle()
    Returns:
        None
    """
    uid_to_tensor_pointer = {
        x if type(x) is int else x.get_uid(): _library_device_pointer(pointer)
        for x, pointer in tensor_to_device_buffer.items()
        if x is not None
    }

    workspace_pointer = _library_device_pointer(workspace)
    self._execute(uid_to_tensor_pointer, workspace_pointer, handle)


def _execute_plan_at_index(
    self, tensor_to_device_buffer, workspace, index, handle=None
):
    """
    Execute a cudnn graph.

    Args:
        tensor_to_device_buffer (dict(cudnn_tensor, Union[torch.Tensor, int, __dlpack__])): The dimensions of the tensor.
        workspace (Union[torch.Tensor, int, __dlpack__]): The name of the tensor.
        index(int): Location of execution plan to use.
        handle: cudnn_handle created with cudnn.create_handle()
    Returns:
        None
    """
    uid_to_tensor_pointer = {
        x if type(x) is int else x.get_uid(): _library_device_pointer(pointer)
        for x, pointer in tensor_to_device_buffer.items()
        if x is not None
    }

    workspace_pointer = _library_device_pointer(workspace)
    self._execute_plan_at_index(uid_to_tensor_pointer, workspace_pointer, index, handle)


pygraph.execute = _execute
pygraph.execute_plan_at_index = _execute_plan_at_index


def load_cudnn():
    # First look at python site packages
    lib_path = glob.glob(
        os.path.join(sysconfig.get_path("purelib"), "nvidia/cudnn/bin/cudnn64_9.dll")
    )

    if lib_path:
        assert (
            len(lib_path) == 1
        ), f"Found {len(lib_path)} libcudnn.dll.x in nvidia-cudnn-cuXX."
        lib = ctypes.windll.LoadLibrary(lib_path[0])
    else:  # Fallback
        lib = ctypes.windll.LoadLibrary("cudnn64_9.dll")

    handle = ctypes.cast(lib._handle, ctypes.c_void_p).value
    _pybind_module._set_dlhandle_cudnn(handle)


def _dlopen_cudnn():
    # First look at python site packages
    lib_path = glob.glob(
        os.path.join(
            sysconfig.get_path("purelib"), "nvidia/cudnn/lib/libcudnn.so.*[0-9]"
        )
    )

    if not lib_path:
        lib_path = glob.glob(
            os.path.join(
                sysconfig.get_path("purelib"), "nvidia/cudnn_jit/lib/libcudnn.so.*[0-9]"
            )
        )

    if lib_path:
        assert (
            len(lib_path) == 1
        ), f"Found {len(lib_path)} libcudnn.so.x in nvidia-cudnn-cuXX."
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

from .graph import graph, jit, graph_cache
from .wrapper import Graph

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "NSA":
        try:
            from .native_sparse_attention import NSA as _NSA

            return _NSA
        except Exception as e:
            raise ImportError(
                f"NSA requires optional dependencies. Install with 'pip install nvidia-cudnn-frontend[cutedsl]': {e}"
            ) from e

    elif name == "GemmSwigluSm100":
        try:
            from .gemm_swiglu import GemmSwigluSm100 as _GemmSwigluSm100

            return _GemmSwigluSm100
        except Exception as e:
            raise ImportError(
                f"GemmSwigluSm100 requires optional dependencies. Install with 'pip install nvidia-cudnn-frontend[cutedsl]': {e}"
            ) from e

    elif name == "gemm_swiglu_wrapper_sm100":
        try:
            from .gemm_swiglu import (
                gemm_swiglu_wrapper_sm100 as _gemm_swiglu_wrapper_sm100,
            )

            return _gemm_swiglu_wrapper_sm100
        except Exception as e:
            raise ImportError(
                f"gemm_swiglu_wrapper_sm100 requires optional dependencies. Install with 'pip install nvidia-cudnn-frontend[cutedsl]': {e}"
            ) from e

    elif name == "GemmAmaxSm100":
        try:
            from .gemm_amax import GemmAmaxSm100 as _GemmAmaxSm100

            return _GemmAmaxSm100
        except Exception as e:
            raise ImportError(
                f"GemmAmaxSm100 requires optional dependencies. Install with 'pip install nvidia-cudnn-frontend[cutedsl]': {e}"
            ) from e

    elif name == "gemm_amax_wrapper_sm100":
        try:
            from .gemm_amax import (
                gemm_amax_wrapper_sm100 as _gemm_amax_wrapper_sm100,
            )

            return _gemm_amax_wrapper_sm100
        except Exception as e:
            raise ImportError(
                f"gemm_amax_wrapper_sm100 requires optional dependencies. Install with 'pip install nvidia-cudnn-frontend[cutedsl]': {e}"
            ) from e
    else:
        raise AttributeError(name)
