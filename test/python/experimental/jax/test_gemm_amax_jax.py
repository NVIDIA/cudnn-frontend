"""End-to-end smoke test for the JAX binding of gemm_amax (MVP / draft).

Skips unless the full stack is present: jax, jax-tvm-ffi, torch, and an SM100 GPU.
It reuses the existing torch test's input builder + reference so the JAX path is
checked against the exact same kernel the torch path runs.

Run:  pytest test/python/experimental/jax/test_gemm_amax_jax.py -q
"""

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("jax_tvm_ffi")
torch = pytest.importorskip("torch")
import jax.numpy as jnp


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major >= 10


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="gemm_amax needs SM100+")


def _to_jax(t: "torch.Tensor"):
    # zero-copy torch -> jax via DLPack (the framework-neutral handoff).
    return jax.dlpack.from_dlpack(t.detach().contiguous())


def test_gemm_amax_jax_matches_torch():
    from cudnn import gemm_amax_wrapper_sm100
    from cudnn.experimental.jax import gemm_amax as gemm_amax_jax
    from fe_api.gemm.test_gemm_amax_utils import allocate_input_tensors

    m, n, k, l = 256, 256, 256, 1
    ab_dtype, sf_dtype, sf_vec_size = torch.float8_e4m3fn, torch.float8_e8m0fnu, 32
    a_major, b_major, c_major = "k", "k", "n"
    c_dtype = torch.bfloat16

    a, b, sfa, sfb = allocate_input_tensors(m, n, k, l, ab_dtype, sf_dtype, sf_vec_size, a_major, b_major)

    # torch reference (same compiled kernel)
    c_ref, amax_ref = gemm_amax_wrapper_sm100(a, b, sfa, sfb, c_major=c_major, c_dtype=c_dtype, sf_vec_size=sf_vec_size).values()

    # JAX path, under jit, calling the identical kernel via the tvm-ffi bridge
    @jax.jit
    def run(a_j, b_j, sfa_j, sfb_j):
        return gemm_amax_jax(
            a_j,
            b_j,
            sfa_j,
            sfb_j,
            c_dtype=jnp.bfloat16,
            c_major=c_major,
            sf_vec_size=sf_vec_size,
            _sample_torch=(a, b, sfa, sfb, c_ref, amax_ref),
        )

    c_jax, amax_jax = run(_to_jax(a), _to_jax(b), _to_jax(sfa), _to_jax(sfb))

    import numpy as np

    np.testing.assert_allclose(
        np.asarray(c_jax.astype(jnp.float32)),
        c_ref.float().cpu().numpy(),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(amax_jax).reshape(()),
        float(amax_ref.reshape(()).cpu()),
        rtol=2e-2,
        atol=2e-2,
    )
