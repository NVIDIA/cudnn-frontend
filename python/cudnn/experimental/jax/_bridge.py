"""Shared JAX <- CuteDSL bridge (written ONCE; every kernel binding reuses it).

Architecture (the deliberate choice — see README.md):
  - FE owns a STABLE public JAX op. tvm-ffi is an INTERNAL transport, never the
    external contract. If tvm-ffi's ABI drifts, only this file moves; callers and
    the per-kernel `*_jax` signatures are unaffected.
  - The CuteDSL kernels are already compiled with `--enable-tvm-ffi`, so their
    entrypoint speaks the tvm-ffi ABI. jax-tvm-ffi adapts that to an XLA custom
    call; nothing new is needed on the kernel side.

Mitigations baked in here (from the observed flashinfer tvm-ffi/DSL breakages):
  - Tight version guard on apache-tvm-ffi with a HUMAN error, so a dependency
    mismatch does not surface downstream as an opaque `CONFIG_MISSING_TVM_FFI`
    that reads like a kernel bug.
  - register-once memoization keyed by target name.

Status: MVP / draft. Unverified end-to-end (needs SM100 + jax + jax-tvm-ffi).
"""

from __future__ import annotations

import importlib
import importlib.metadata as _md

# apache-tvm-ffi is a header-only ABI dependency; pin like flashinfer does.
_TVM_FFI_MIN = (0, 1, 11)
_TVM_FFI_MAX_EXCL = (0, 2)

_registered: set[str] = set()
_checked = False


def _parse_ver(v: str) -> tuple:
    out = []
    for part in v.split(".")[:3]:
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def ensure_available() -> None:
    """Raise a clear, human error if the JAX transport stack is missing/mismatched.

    Deliberately explicit so a packaging problem is not mistaken for a kernel bug.
    """
    global _checked
    if _checked:
        return
    missing = []
    for pkg in ("jax", "jax_tvm_ffi"):
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg.replace("_", "-"))
    if missing:
        raise ImportError("cudnn.experimental.jax requires " f"{', '.join(missing)}. Install with: pip install 'jax[cuda13]' jax-tvm-ffi")
    try:
        ver = _md.version("apache-tvm-ffi")
    except _md.PackageNotFoundError:
        raise ImportError(
            "cudnn.experimental.jax requires apache-tvm-ffi "
            f">={'.'.join(map(str, _TVM_FFI_MIN))},<{'.'.join(map(str, _TVM_FFI_MAX_EXCL))} "
            "(header-only ABI dep). Install: pip install apache-tvm-ffi"
        )
    pv = _parse_ver(ver)
    if not (_TVM_FFI_MIN <= pv < _TVM_FFI_MAX_EXCL):
        raise ImportError(
            f"apache-tvm-ffi {ver} is outside the supported ABI window "
            f"[{'.'.join(map(str, _TVM_FFI_MIN))}, {'.'.join(map(str, _TVM_FFI_MAX_EXCL))}). "
            "Pin it inside that range; the CuteDSL kernels are compiled against it."
        )
    _checked = True


def register_once(name: str, wrapper, arg_spec) -> str:
    """Register a tvm-ffi wrapper as an XLA FFI target with FE's standard flags.

    `wrapper(*rets, *args, **attrs)` reorders JAX's (outputs, inputs, scalars) into
    the compiled kernel's own parameter order and calls it. Returns `name`.
    """
    ensure_available()
    if name in _registered:
        return name
    import jax_tvm_ffi

    jax_tvm_ffi.register_ffi_target(
        name,
        wrapper,
        arg_spec=arg_spec,
        platform="gpu",
        allow_cuda_graph=True,
        pass_owned_tensor=True,  # wrapper receives tvm_ffi.Tensor (DLPack-backed)
    )
    _registered.add(name)
    return name
