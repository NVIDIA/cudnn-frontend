import sys
import importlib


def is_windows():
    return sys.platform.startswith("win")


module_name = ".Release._compiled_module" if is_windows() else "._compiled_module"

_pybind_module = importlib.import_module(module_name, package="cudnn")

globals()["cudnn_data_type"] = getattr(_pybind_module, "data_type")

torch_available = None
_torch_to_cudnn_data_type_dict = None

# Optional CUTLASS integration
cutlass_available = None
_torch_to_cutlass_data_type_dict = None


def is_torch_available():
    global torch_available, _torch_to_cudnn_data_type_dict
    # this condition ensures that datatype mapping is only created once
    if torch_available is None:
        try:
            import torch

            torch_available = True
            _torch_to_cudnn_data_type_dict = {
                torch.half: cudnn_data_type.HALF,
                torch.float16: cudnn_data_type.HALF,
                torch.bfloat16: cudnn_data_type.BFLOAT16,
                torch.float: cudnn_data_type.FLOAT,
                torch.float32: cudnn_data_type.FLOAT,
                torch.double: cudnn_data_type.DOUBLE,
                torch.float64: cudnn_data_type.DOUBLE,
                torch.int8: cudnn_data_type.INT8,
                torch.int32: cudnn_data_type.INT32,
                torch.int64: cudnn_data_type.INT64,
                torch.uint8: cudnn_data_type.UINT8,
                torch.bool: cudnn_data_type.BOOLEAN,
            }

            def possibly_add_type(torch_type_name, cudnn_type):
                # Only try adding the type if the version of torch being used supports it
                if hasattr(torch, torch_type_name):
                    torch_type = getattr(torch, torch_type_name)
                    _torch_to_cudnn_data_type_dict[torch_type] = cudnn_type

            possibly_add_type("float8_e4m3fn", cudnn_data_type.FP8_E4M3)
            possibly_add_type("float8_e5m2", cudnn_data_type.FP8_E5M2)

            possibly_add_type("float8_e8m0fnu", cudnn_data_type.FP8_E8M0)
            possibly_add_type("float4_e2m1fn_x2", cudnn_data_type.FP4_E2M1)

        except ImportError:
            torch_available = False
            _torch_to_cudnn_data_type_dict = {}
    return torch_available


def is_cutlass_available():
    global cutlass_available, _torch_to_cutlass_data_type_dict
    if cutlass_available is None:
        try:
            import torch
            import cutlass

            cutlass_available = True
            mapping = {
                torch.half: getattr(cutlass, "Float16", None),
                getattr(torch, "float16", torch.half): getattr(
                    cutlass, "Float16", None
                ),
                getattr(torch, "bfloat16", None): getattr(cutlass, "BFloat16", None),
                torch.float: getattr(cutlass, "Float32", None),
                getattr(torch, "float32", torch.float): getattr(
                    cutlass, "Float32", None
                ),
                torch.double: getattr(cutlass, "Float64", None),
                getattr(torch, "float64", torch.double): getattr(
                    cutlass, "Float64", None
                ),
                getattr(torch, "int8", None): getattr(cutlass, "Int8", None),
                getattr(torch, "int32", None): getattr(cutlass, "Int32", None),
                getattr(torch, "int64", None): getattr(cutlass, "Int64", None),
                getattr(torch, "uint8", None): getattr(cutlass, "Uint8", None),
                getattr(torch, "bool", None): getattr(cutlass, "Boolean", None),
                getattr(torch, "float8_e4m3fn", None): getattr(
                    cutlass, "Float8E4M3FN", None
                ),
                getattr(torch, "float8_e5m2", None): getattr(
                    cutlass, "Float8E5M2", None
                ),
                getattr(torch, "float8_e8m0fnu", None): getattr(
                    cutlass, "Float8E8M0FNU", None
                ),
                getattr(torch, "float4_e2m1fn_x2", None): getattr(
                    cutlass, "Float4E2M1FN", None
                ),
            }
            _torch_to_cutlass_data_type_dict = {
                t: c for t, c in mapping.items() if t is not None and c is not None
            }
        except ImportError:
            cutlass_available = False
            _torch_to_cutlass_data_type_dict = {}
    return cutlass_available


# Returns None in case mapping is not available
def _torch_to_cudnn_data_type(torch_data_type) -> cudnn_data_type:
    if is_torch_available():
        return _torch_to_cudnn_data_type_dict.get(torch_data_type, None)
    else:
        return None


def _torch_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2: bool = False):
    if is_cutlass_available() and is_torch_available():
        import torch

        if interpret_uint8_as_fp4x2 and data_type == torch.uint8:
            import cutlass

            return getattr(cutlass, "Float4E2M1FN", None)
        else:
            return _torch_to_cutlass_data_type_dict.get(data_type, None)
    return None


def _convert_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2: bool = False):
    if is_cutlass_available():
        import cutlass

        if isinstance(data_type, type) and issubclass(data_type, cutlass.Numeric):
            return data_type
        elif data_type is not None:
            cutlass_data_type = _torch_to_cutlass_data_type(
                data_type, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2
            )
            if cutlass_data_type is None:
                raise ValueError("Unsupported tensor data type.")
            return cutlass_data_type
        else:
            raise ValueError("None is not a valid tensor data type.")
    return None


def _cudnn_to_torch_data_type(cudnn_data_type):
    """Convert a cuDNN data type to a PyTorch data type.

    Args:
        cudnn_data_type: The cuDNN data type to convert.

    Returns:
        The PyTorch data type, or None if the conversion is not available.
    """
    if is_torch_available():
        for torch_type, cudnn_type in _torch_to_cudnn_data_type_dict.items():
            if cudnn_type == cudnn_data_type:
                return torch_type
    return None


def _library_type(input_type):
    if type(input_type) is cudnn_data_type:
        return input_type

    for cvt_fn in [
        _torch_to_cudnn_data_type,
        # Add more DL libraries to support here
    ]:
        out = cvt_fn(input_type)
        if out is not None:
            return out

    raise Exception(
        f"No available conversion from type {input_type} to a library type."
    )


def _is_torch_tensor(input_tensor) -> bool:
    if is_torch_available():
        import torch

        return isinstance(input_tensor, torch.Tensor)
    return False
