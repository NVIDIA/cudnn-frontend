# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic dprob for the BF16 grouped dGLU kernel (SM100 family).

By default each N-tile adds its ``dprob[token]`` partial with an fp32 atomic, so dprob depends on
tile arrival order once N spans more than one 256-wide tile. ``deterministic=True`` gives each
(token, N-tile) partial its own fp32 slot and sums the slots in a fixed order on the launch stream.

The suite covers both activations, dense and discrete weights, the static and dynamic schedulers,
ragged and empty experts, capacity past the last expert, natural-rank tensors and both GLU branch
orders. dprob and D are compared as raw bytes across replays under concurrent work and across expert
reordering, duplication and neighbor changes; the reference and the atomic path are separate
numerical checks. Only dprob is a reduction output: D is written by one tile and was deterministic
already, and dbias (an atomic BF16 reduction) is rejected under the flag.
"""

import pytest
import torch

from cuda.bindings import driver as cuda

import cudnn
from cudnn.frost.buffers import cutedsl_state, cutedsl_too_old
from determinism_utils import assert_bitwise, assert_replays
from test_utils import torch_fork_set_rng
from gemm.cutedsl.test_grouped_gemm_dglu_bf16_utils import assert_grouped_gemm_dglu_close, grouped_gemm_dglu_bf16_reference

pytestmark = pytest.mark.L0

# Unequal experts with an empty one, then capacity past the last offset the kernel never addresses.
GROUP_M = (512, 0, 256, 768)
TAIL_M = 256
N, K = 1024, 256  # four 256-wide N tiles per token
ACTIVATIONS = [("dswiglu", 0.0), ("dgeglu", 1.0)]


@pytest.fixture(autouse=True)
def require_sm100_family():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10 or cutedsl_too_old(cutedsl_state()[1]):
        pytest.skip("requires an SM100-family GPU and CuTe DSL>=4.7.0")


def make_problem(group_m=GROUP_M, tail_m=TAIL_M, n=N, k=K, seed=0):
    """BF16 dGLU inputs in the kernel-facing ranks; b_reference is (experts, n, k)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    experts = len(group_m)
    m = sum(group_m) + tail_m

    def randn(*shape, scale):
        return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16) * scale

    b_reference = randn(experts, n, k, scale=0.125)
    offsets = torch.tensor(group_m, dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    return {
        "a": randn(1, m, k, scale=0.125).permute(1, 2, 0),
        "b": b_reference.permute(1, 2, 0),
        "b_reference": b_reference,
        "b_ptrs": torch.tensor([b_reference[e].data_ptr() for e in range(experts)], dtype=torch.int64, device="cuda"),
        "c": randn(1, m, 2 * n, scale=0.5).permute(1, 2, 0),
        "offsets": offsets,
        "alpha": torch.linspace(0.75, -1.25, experts, device="cuda"),
        "beta": torch.linspace(1.5, -0.5, experts, device="cuda"),
        "prob": torch.linspace(-0.75, 0.875, m, device="cuda").view(m, 1, 1),
        "n": n,
        "experts": experts,
        "valid_m": int(offsets[-1].item()),
    }


def run(p, act_func, *, deterministic, discrete=False, use_dynamic_sched=False, dprob=None, **kwargs):
    """One wrapper call on fresh outputs; dprob is caller-owned and accumulated into."""
    weights = dict(b_ptrs=p["b_ptrs"], n=p["n"], b_dtype=torch.bfloat16) if discrete else dict(b_tensor=p["b"])
    return cudnn.grouped_gemm_dglu_wrapper_sm100(
        a_tensor=p["a"],
        c_tensor=p["c"],
        sfa_tensor=None,
        padded_offsets=p["offsets"],
        alpha_tensor=p["alpha"],
        beta_tensor=p["beta"],
        prob_tensor=p["prob"],
        dprob_tensor=torch.zeros_like(p["prob"]) if dprob is None else dprob,
        d_dtype=torch.bfloat16,
        act_func=act_func,
        use_dynamic_sched=use_dynamic_sched,
        deterministic=deterministic,
        **weights,
        **kwargs,
    )


def outputs(result, valid_m):
    """dprob in full; D up to the last expert offset (capacity past it is unspecified)."""
    return {"dprob": result["dprob_tensor"], "d_row": result["d_row_tensor"][:valid_m]}


@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("use_dynamic_sched", [False, True], ids=["static_sched", "dynamic_sched"])
@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "discrete"])
@pytest.mark.parametrize("act_func,linear_offset", ACTIVATIONS, ids=[a for a, _ in ACTIVATIONS])
def test_bf16_deterministic_dprob(act_func, linear_offset, discrete, use_dynamic_sched):
    """Bit-exact dprob over replays, same D as the atomic path, and the reference's values."""
    from cudnn.gemm.cutedsl.grouped.dglu import api as dglu_api

    p = make_problem()
    valid_m = p["valid_m"]
    launch = dict(discrete=discrete, use_dynamic_sched=use_dynamic_sched)
    atomic = {name: t.clone() for name, t in outputs(run(p, act_func, deterministic=False, **launch), valid_m).items()}
    # Eight runs on fresh outputs with concurrent work on another stream; raw bytes.
    det = assert_replays(lambda: outputs(run(p, act_func, deterministic=True, **launch), valid_m))

    plans = [o._implementation for o in dglu_api._cache_of_GroupedGemmDgluSm100Objects.values() if getattr(o._implementation, "_deterministic", False)]
    assert any(plan._dprob_slots == N // 256 for plan in plans), "the deterministic slot path did not compile"
    assert torch.count_nonzero(det["dprob"][:valid_m]).item() > 0
    # D has one writer per element; the flag must leave it byte-identical.
    assert_bitwise({"d_row": det["d_row"]}, {"d_row": atomic["d_row"]}, "deterministic vs atomic")
    # dprob equals the atomic sum up to fp32 reordering of four partials.
    torch.testing.assert_close(det["dprob"], atomic["dprob"], rtol=1e-4, atol=1e-4)
    # Nothing is added past the last expert offset.
    assert_bitwise({"tail": det["dprob"][valid_m:]}, {"tail": torch.zeros_like(det["dprob"][valid_m:])})

    ref_d, ref_dprob, _ = grouped_gemm_dglu_bf16_reference(p, act_func=act_func, linear_offset=linear_offset, generate_dbias=False)
    assert_grouped_gemm_dglu_close(det["dprob"][:valid_m], ref_dprob[:valid_m])
    assert_grouped_gemm_dglu_close(det["d_row"], ref_d[:valid_m])


@torch_fork_set_rng(seed=1)
@pytest.mark.parametrize("glu_layout", ["interleaved_32", "gate_then_up"])
@pytest.mark.parametrize("act_func", ["dswiglu", "dgeglu"])
def test_bf16_deterministic_dprob_expert_batch_invariance(act_func, glu_layout):
    """An expert's dprob and D bytes do not depend on which experts share the launch."""
    if glu_layout == "gate_then_up" and (act_func != "dswiglu" or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))):
        pytest.skip("BF16 gate_then_up dGLU is dSwiGLU on SM100/103 only")
    group_m = (512, 256, 768)
    p = make_problem(group_m=group_m, tail_m=0, seed=1)
    # Natural ranks: A (m, k), C (m, 2n), prob/dprob (m,), B (experts, n, k).
    a, c, prob = p["a"].squeeze(-1), p["c"].squeeze(-1), p["prob"].view(-1)
    if glu_layout == "gate_then_up":
        c = c.view(c.shape[0], -1, 2, 32).transpose(1, 2).reshape(c.shape).contiguous()
    saved = {"a": a.clone(), "c": c.clone(), "b": p["b_reference"].clone(), "prob": prob.clone()}
    begins = [0, *torch.tensor(group_m).cumsum(0).tolist()]

    def invoke(order, a=a, c=c):
        rows = torch.cat([torch.arange(begins[e], begins[e + 1], device="cuda") for e in order])
        sizes = torch.tensor([group_m[e] for e in order], dtype=torch.int32, device="cuda")
        result = cudnn.grouped_gemm_dglu_wrapper_sm100(
            a_tensor=a[rows].contiguous(),
            b_tensor=p["b_reference"][list(order)].contiguous(),
            c_tensor=c[rows].contiguous(),
            sfa_tensor=None,
            padded_offsets=sizes.cumsum(0, dtype=torch.int32),
            alpha_tensor=p["alpha"][list(order)].contiguous(),
            beta_tensor=p["beta"][list(order)].contiguous(),
            prob_tensor=prob[rows].contiguous(),
            dprob_tensor=torch.zeros_like(prob[rows]),
            d_dtype=torch.bfloat16,
            act_func=act_func,
            glu_layout=glu_layout,
            use_dynamic_sched=True,
            deterministic=True,
        )
        return {name: t.clone() for name, t in outputs(result, len(rows)).items()}, rows

    packed, _ = invoke((0, 1, 2))
    for order in ((2,), (2, 0, 1), (1, 1)):  # alone, reordered, duplicated
        result, rows = invoke(order)
        assert_bitwise(result, {name: t[rows] for name, t in packed.items()}, f"order {order}")
    # Neighbor isolation: change expert 1's tokens; experts 0 and 2 keep their bytes.
    neighbor_a, neighbor_c = a.clone(), c.clone()
    neighbor_a[begins[1] : begins[2]].zero_()
    neighbor_c[begins[1] : begins[2]].fill_(0.25)
    changed, _ = invoke((0, 1, 2), a=neighbor_a, c=neighbor_c)
    for e in (0, 2):
        rows = slice(begins[e], begins[e + 1])
        assert_bitwise({k: t[rows] for k, t in changed.items()}, {k: t[rows] for k, t in packed.items()}, f"expert {e}")
    for name, tensor in (("a", a), ("c", c), ("b", p["b_reference"]), ("prob", prob)):
        assert_bitwise({name: tensor}, {name: saved[name]}, "inputs mutated")


@torch_fork_set_rng(seed=2)
@pytest.mark.parametrize("busy", ["current_stream", "launch_stream"])
def test_bf16_deterministic_dprob_on_launch_stream(busy):
    """The slots are zeroed and reduced on the launch stream. A busy current stream would let the
    zero-fill land after the kernel; a busy launch stream would let the reduction run before it."""
    p = make_problem(seed=2)
    expected = run(p, "dswiglu", deterministic=True)["dprob_tensor"].clone()  # also compiles
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        dprob = torch.zeros_like(p["prob"])  # the caller orders its own dprob before the launch
        if busy == "launch_stream":
            torch.cuda._sleep(200_000_000)
    if busy == "current_stream":
        torch.cuda._sleep(200_000_000)
    actual = run(p, "dswiglu", deterministic=True, dprob=dprob, current_stream=cuda.CUstream(side.cuda_stream))["dprob_tensor"]
    torch.cuda.synchronize()
    assert_bitwise({"dprob": actual}, {"dprob": expected})


def test_bf16_deterministic_rejects_dbias():
    p = make_problem(group_m=(256,), tail_m=0, n=512)
    with pytest.raises(NotImplementedError, match="deterministic dprob is implemented only for torch tensors without dbias"):
        run(p, "dswiglu", deterministic=True, generate_dbias=True)


def test_bf16_deterministic_rejects_jax():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    p = make_problem(group_m=(256,), tail_m=0, n=512)

    def to_jax(tensor):
        host = tensor.cpu()
        return jnp.asarray(host.float().numpy()).astype(jnp.bfloat16) if host.dtype == torch.bfloat16 else jnp.asarray(host.numpy())

    arrays = {name: to_jax(p[name]) for name in ("a", "c", "offsets", "alpha", "beta", "prob")}
    weight = to_jax(p["b_reference"][0])
    # Packed uint8 pointers: JAX without x64 mode would narrow an int64 array.
    ptrs = jax.block_until_ready(jnp.asarray(torch.tensor([weight.unsafe_buffer_pointer()], dtype=torch.int64).numpy().view("uint8")))
    with pytest.raises(NotImplementedError, match="deterministic=True is implemented only with torch tensors"):
        cudnn.grouped_gemm_dglu_wrapper_sm100(
            a_tensor=arrays["a"],
            c_tensor=arrays["c"],
            sfa_tensor=None,
            padded_offsets=arrays["offsets"],
            alpha_tensor=arrays["alpha"],
            beta_tensor=arrays["beta"],
            prob_tensor=arrays["prob"],
            dprob_tensor=jnp.zeros(arrays["prob"].shape, dtype=jnp.float32),
            b_ptrs=ptrs,
            n=p["n"],
            b_dtype="bfloat16",
            d_dtype="bfloat16",
            deterministic=True,
        )


@pytest.mark.L1
@torch_fork_set_rng(seed=3)
@pytest.mark.parametrize("use_dynamic_sched", [False, True], ids=["static_sched", "dynamic_sched"])
def test_bf16_deterministic_dprob_at_scale(use_dynamic_sched):
    """DeepSeek-V3-like FC2 dgrad (N=2048, 8 experts x 1024 tokens), where the atomic dprob varies."""
    p = make_problem(group_m=(1024,) * 8, tail_m=0, n=2048, k=1024, seed=3)
    launch = dict(discrete=True, use_dynamic_sched=use_dynamic_sched)
    atomic = outputs(run(p, "dswiglu", deterministic=False, **launch), p["valid_m"])
    det = assert_replays(lambda: outputs(run(p, "dswiglu", deterministic=True, **launch), p["valid_m"]))
    assert_bitwise({"d_row": det["d_row"]}, {"d_row": atomic["d_row"]})
    torch.testing.assert_close(det["dprob"], atomic["dprob"], rtol=1e-3, atol=1e-3)
