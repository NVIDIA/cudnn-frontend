"""JAX binding for the SM100 blockscaled GEMM + fused amax CuteDSL kernel.

This is the *entire* per-kernel cost of exposing a CuteDSL kernel to JAX as a real
primitive (composes with @jax.jit; no host callback). Everything framework-neutral
lives in _bridge.py; below is only what is specific to gemm_amax.

Assumes torch-removal (the separate de-torching effort) is orthogonal and done;
here we only count the JAX boilerplate. The kernel is reused verbatim — same
compiled binary the torch path calls.

Status: MVP / draft — UNVERIFIED end-to-end. Needs SM100 + jax + jax-tvm-ffi and
the two GOTCHAs below resolved on hardware.
"""

from __future__ import annotations

from . import _bridge

# The CuteDSL kernel, reused as-is. Prefer the top-level re-export (stable across
# the #459 gemm-fusion reorg); fall back to the module paths on older trees.
try:
    from cudnn import GemmAmaxSm100
except ImportError:  # pragma: no cover
    try:
        from cudnn.gemm.cutedsl.dense.amax.api import GemmAmaxSm100  # post-#459
    except ImportError:
        from cudnn.gemm_amax.api import GemmAmaxSm100  # pre-reorg


_compiled_cache: dict = {}


def _get_compiled(spec, sample_torch):
    """Compile the CuteDSL kernel once per specialization; return the tvm-ffi callable.

    GOTCHA (stream): the kernel currently compiles with
        make_fake_stream(use_tvm_ffi_env_stream=False)   # amax/api.py
    For the JAX path it must be True so the CUDA stream is taken from the tvm-ffi
    env stream that jax-tvm-ffi sets from XLA's stream. That is a 1-flag change in
    the kernel's compile(); tracked as the only kernel-side edit this MVP needs.
    """
    if spec in _compiled_cache:
        return _compiled_cache[spec]
    a, b, sfa, sfb, c, amax = sample_torch
    k = GemmAmaxSm100(
        sample_a=a,
        sample_b=b,
        sample_sfa=sfa,
        sample_sfb=sfb,
        sample_c=c,
        sample_amax=amax,
        acc_dtype=spec.acc_dtype,
        mma_tiler_mn=spec.mma_tiler_mn,
        cluster_shape_mn=spec.cluster_shape_mn,
        sf_vec_size=spec.sf_vec_size,
    )
    assert k.check_support()
    k.compile()
    _compiled_cache[spec] = k._compiled_kernel
    return _compiled_cache[spec]


class _Spec:
    __slots__ = ("c_dtype", "c_major", "acc_dtype", "mma_tiler_mn", "cluster_shape_mn", "sf_vec_size")

    def __init__(self, c_dtype, c_major, acc_dtype, mma, cluster, sfv):
        self.c_dtype, self.c_major, self.acc_dtype = c_dtype, c_major, acc_dtype
        self.mma_tiler_mn, self.cluster_shape_mn, self.sf_vec_size = mma, cluster, sfv

    def key(self):
        return f"{self.c_dtype}|{self.c_major}|{self.mma_tiler_mn}|{self.cluster_shape_mn}|{self.sf_vec_size}"

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, o):
        return isinstance(o, _Spec) and self.key() == o.key()


_targets: dict = {}


def _ensure_target(spec, sample_torch):
    name = f"cudnn.gemm_amax.{spec.key()}"
    if name in _targets:
        return name
    compiled = _get_compiled(spec, sample_torch)

    # Reorder JAX's (rets, args) into the kernel's own order.
    # kernel: compiled(a, b, sfa, sfb, c, amax)   [+ env stream]
    # JAX:    (c, amax) then (a, b, sfa, sfb, amax_init)
    def _wrapper(c, amax, a, b, sfa, sfb, amax_init):
        compiled(a, b, sfa, sfb, c, amax)  # env stream injected by the bridge

    _bridge.register_once(name, _wrapper, arg_spec=["rets", "args"])
    _targets[name] = True
    return name


def gemm_amax(a, b, sfa, sfb, *, c_dtype, c_major="n", acc_dtype=None, mma_tiler_mn=(128, 128), cluster_shape_mn=(1, 1), sf_vec_size=32, _sample_torch):
    """Blockscaled GEMM with fused amax, callable from JAX under @jax.jit.

    a: [m, k, l]   b: [n, k, l]   sfa/sfb: atom-reshaped scale factors
    Returns (c: [m, n, l], amax: [1, 1, 1]). Pure JAX in / pure JAX out.

    `_sample_torch` is a temporary MVP shim: representative torch tensors used to
    drive the kernel's existing check_support()/compile(). Once torch-removal lands
    this collapses to plain shape/dtype descriptors (orthogonal effort).
    """
    import jax
    import jax.numpy as jnp

    m, _, l = a.shape
    n, _, _ = b.shape

    # abstract-eval: outputs are fully determined by INPUT shapes (no data-dependent
    # shape — this is why gemm_amax is the easy case, unlike grouped/MoE/sparse).
    out_specs = (
        jax.ShapeDtypeStruct((m, n, l), c_dtype),
        jax.ShapeDtypeStruct((1, 1, 1), jnp.float32),
    )

    spec = _Spec(c_dtype, c_major, acc_dtype, mma_tiler_mn, cluster_shape_mn, sf_vec_size)
    name = _ensure_target(spec, _sample_torch)

    # GOTCHA (amax init): XLA output buffers are uninitialized, but the kernel does
    # atomicMax into amax and needs it pre-filled with -inf (torch path: torch.full).
    # So amax enters as a donated input filled with -inf, input_output_aliased to
    # output #1 — the XLA-native way to express an in-out accumulator.
    amax_init = jnp.full((1, 1, 1), -jnp.inf, dtype=jnp.float32)

    # GOTCHA (layout): the compiled kernel is specialized to the sample's strides
    # (c_major); XLA hands row-major buffers by default. On hardware, declare
    # operand/result layouts on ffi_call or match input layout. Left for the PoC.
    c, amax = jax.ffi.ffi_call(
        name,
        out_specs,
        input_output_aliases={4: 1},  # arg #4 (amax_init) -> output #1 (amax)
    )(a, b, sfa, sfb, amax_init)
    return c, amax
