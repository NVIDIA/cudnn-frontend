# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy exports for the norm packages, so importing one does not import kernels.

The engines must be able to DECLINE on a machine whose CuTe DSL cannot serve
them. That only works if the decline runs before anything imports a kernel
module, because a kernel module imports ``cutlass.primitives`` at its top and an
install without it raises there -- out of the engine's reach, with the plan
already in the ranked list.

Keeping ``cudnn.norm`` itself lazy was not enough: the engine lives in
``cudnn.norm.fprop.engine``, and importing it imports the ``cudnn.norm.fprop``
PACKAGE first, whose ``__init__`` pulled in ``api`` and with it all twelve
kernels. 287 cutlass modules were loaded before any gate had a chance to speak.

This module must therefore stay free of CuTe DSL imports itself.
"""

from typing import Any


def dsl_unavailable(what: str):
    """Why the CuTe DSL cannot serve ``what``, or None when it can.

    Version first (cheap, metadata only), then the primitive layer -- that one
    imports the DSL, so it is asked second and only when the caller is about to
    import kernels anyway.
    """
    from cudnn.frost.buffers import cutedsl_primitives_error, cutedsl_requirement_error

    return cutedsl_requirement_error(what) or cutedsl_primitives_error(what)


def lazy_exports(package: str, exports: dict, namespace: dict):
    """A module ``__getattr__`` that gates the DSL before importing anything.

    ``exports`` maps the attribute name to ``(module, attribute)``. Resolved
    values are cached into ``namespace`` so the gate and the import happen once.
    """

    def __getattr__(name: str) -> Any:
        try:
            module_name, attr_name = exports[name]
        except KeyError:
            raise AttributeError(f"module {package!r} has no attribute {name!r}") from None
        error = dsl_unavailable(package)
        if error is not None:
            raise ImportError(error)
        import importlib

        value = getattr(importlib.import_module(module_name), attr_name)
        namespace[name] = value
        return value

    return __getattr__
