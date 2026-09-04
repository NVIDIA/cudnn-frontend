# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: Apache-2.0
"""Minimal CuTe DSL parameter helpers adapted from quack-kernels 0.5.0."""

from dataclasses import dataclass, fields
from typing import get_origin

import cutlass
from cutlass.base_dsl.tvm_ffi_builder import spec
from cutlass.cutlass_dsl import NumericMeta
import cutlass.cute._tvm_ffi_args_spec_converter as _converter_module

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))

_original_convert_single_arg = _converter_module._convert_single_arg


def _patched_convert_single_arg(arg, arg_name, arg_type, ctx):
    """Treat ``Constexpr`` annotations as compile-time TVM-FFI arguments."""

    if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
        return spec.ConstNone(arg_name)
    if isinstance(arg, tuple) and hasattr(type(arg), "_fields") and (arg_type is None or not hasattr(arg_type, "_fields")):
        return _original_convert_single_arg(arg, arg_name, type(arg), ctx)
    return _original_convert_single_arg(arg, arg_name, arg_type, ctx)


_converter_module._convert_single_arg = _patched_convert_single_arg


def _partition_fields(obj):
    """Split dataclass fields into compile-time and runtime dictionaries."""

    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {name: value for name, value in all_fields.items() if isinstance(value, StaticTypes)}
    runtime = {name: value for name, value in all_fields.items() if not isinstance(value, StaticTypes)}
    return constexpr, runtime


def _new_from_mlir_values(self, values):
    constexpr_fields, runtime_fields = _partition_fields(self)
    for (name, field), item_count in zip(runtime_fields.items(), self._values_pos):
        runtime_fields[name] = cutlass.new_from_mlir_values(field, values[:item_count])
        values = values[item_count:]
    return self.__class__(**runtime_fields, **constexpr_fields)


@dataclass
class ParamsBase:
    """Dataclass base that separates CuTe runtime values from constexprs."""

    def __extract_mlir_values__(self):
        _, runtime_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in runtime_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    __new_from_mlir_values__ = _new_from_mlir_values
