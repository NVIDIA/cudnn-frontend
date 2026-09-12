# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-parametrized tests for the linear-attention ops (``gated_delta_net``,
``kimi_delta_attention``, ``gated_delta_net_v2``, ``gated_delta_product`` and
their summary ops) against the fp64 recurrent references.

Each test pins one backend's plan through the ops' ``plan_name``. A pinned
engine that declines a configuration waives the test
(``cudnnGraphNotSupportedError`` -> skip), as does a backend that is not
installed. Determinism, batch invariance and CUDA-graph replay run on the
FROST backend only.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import threading
import time
import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")
la_ops = pytest.importorskip("cudnn.linear_attention.ops")

import torch.nn.functional as F  # noqa: E402

from .conftest import gen_qkv  # noqa: E402
from .reference_gdn import gdn_reference, gdp_reference, rms_ratio  # noqa: E402
from .reference_gdn2 import beta_guard_reference, gdn2_reference  # noqa: E402
from .reference_kda import kda_reference  # noqa: E402

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

VARIANTS = ("gdn", "kda", "gdn2", "gdp")
CHANNEL_VARIANTS = ("kda", "gdn2")
SCALAR_GATE_VARIANTS = ("gdn", "gdp")
SCALAR_BETA_VARIANTS = ("gdn", "kda", "gdp")
HOUSEHOLDER_VARIANTS = ("gdp",)
BETA_GUARD_VARIANTS = ("gdn2",)
CUTILE_VARIANTS = ("gdn", "kda")
STATE_DTYPES = (torch.float32, torch.bfloat16)
SPLIT_T = 4096
HOUSEHOLDER = 3
CHUNK = {"gdn": 64, "kda": 16, "gdn2": 16, "gdp": 64}
LEAF_NAMES = {
    "gdn": ("q", "k", "v", "g", "beta"),
    "kda": ("q", "k", "v", "g", "beta"),
    "gdn2": ("q", "k", "v", "g", "beta", "w"),
    "gdp": ("q", "k", "v", "g", "beta"),
}
EXPANDED_LEAVES = ("k", "v", "beta")

FWD_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
STATE_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
BWD_TOL = {torch.bfloat16: 4e-2, torch.float16: 3e-2}
STATE_GRAD_TOL = 6e-2

HEAD_DIMS = [(64, 64), (64, 128), (128, 64), (128, 128)]  # (K, V) pairs every FROST family serves
WIDE_HEAD_DIMS = [(192, 128), (256, 128)]  # cuTile only

# (H, HV) pairs: H = Q/K heads, HV = V heads; gates/O/states live at HO = max.
HEAD_CONFIGS = [(1, 1), (3, 3), (1, 2), (2, 4), (16, 32), (16, 64)]
HEAD_CONFIGS_SMALL = [(1, 1), (2, 4)]
GQA_CONFIGS = [(4, 4, 1), (6, 6, 2), (4, 1, 1), (6, 2, 2), (1, 2, 2), (2, 4, 4)]

RAGGED_SEQ_LENS = [
    [256, 256],
    [96, 32, 160, 1],
    [511, 501],
    [64, 128, 512],
    [31, 63, 93, 123, 150, 500],
    [7] * 24 + [1] * 8,
    [2048],
]
EDGE_LENS = [1, 15, 16, 17, 31, 63, 64, 65, 121, 251, 257]

DETERMINISM_REPEATS = 8
STRESS_REPEATS = 4
STRESS_SHAPES = [
    dict(seq_lens=[96, 32, 160, 1]),
    dict(seq_lens=[64, 64]),
    dict(seq_lens=[255, 1]),
    dict(seq_lens=[1]),
    dict(B=8, T=192, H=64),
]
SEED = 888

DTYPE_IDS = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}
OP_NAMES = {"gdn": "gated_delta_net", "kda": "kimi_delta_attention", "gdn2": "gated_delta_net_v2", "gdp": "gated_delta_product"}


# ---------------------------------------------------------------------------
# Backend pinning
# ---------------------------------------------------------------------------


def clear_caches():
    """Empty every family op's plan caches (forward, backward and summary) so a test builds its own plans."""
    from cudnn.linear_attention.ops import gdn, gdn2, gdp, kda

    for mod in (gdn, kda, gdn2, gdp):
        mod.fprop_cache.clear()
        mod.bprop_cache.clear()
        mod.summary_cache.clear()


class Case:
    """One test configuration, holding inputs, gates, cu_seqlens and the geometry."""

    __slots__ = ("variant", "dtype", "q", "k", "v", "gates", "cu", "B", "T", "N", "H", "HK", "HV", "HO", "K", "V", "n", "varlen")

    def __init__(self, **fields):
        for name in self.__slots__:
            setattr(self, name, fields.pop(name))
        assert not fields, f"unknown Case fields: {sorted(fields)}"

    def clone(self, **overrides):
        fields = {name: getattr(self, name) for name in self.__slots__}
        fields.update(overrides)
        return Case(**fields)


class Backend:
    """The pinned backend; ``plan`` gives the plan name a graph must offer
    (``<variant>_<name>``, e.g. ``gdn_frost``)."""

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def plan(self, variant):
        return f"{variant}_{self.name}"


def pinned_op(backend, variant):
    """The variant's op with the backend's plan pinned through ``plan_name``."""
    return functools.partial(op(variant), plan_name=backend.plan(variant))


@pytest.fixture(params=("frost", "cutile"))
def backend(request):
    """One backend per run of each test; the tests pass its plan name to the
    ops. The op graph caches are cleared around each test."""
    clear_caches()
    try:
        yield Backend(request.param)
    finally:
        clear_caches()


@contextlib.contextmanager
def waive_unsupported(backend, variant):
    """A backend that offers no plan for the graph, or whose engine declines
    it, waives the test."""
    try:
        yield
    except cudnn.cudnnGraphNotSupportedError as exc:
        pytest.skip(f"{backend.name} {variant} declined: {exc}")


@contextlib.contextmanager
def waive_declined(what):
    """The same waiver for tests that pin no backend; a configuration no engine
    serves is a skip."""
    try:
        yield
    except cudnn.cudnnGraphNotSupportedError as exc:
        pytest.skip(f"{what} declined: {exc}")


# ---------------------------------------------------------------------------
# Case generation and dispatch
# ---------------------------------------------------------------------------


def set_seed(seed=SEED):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def sm_count():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def gen_gates(variant, B, T, HO, K, V, dtype, *, n=1, alpha=True, beta=True, w=True, lo=None, device="cuda"):
    """Gates at ``T`` tokens; beta lives on the expanded timeline (``T * n`` rows) when ``n > 1``."""
    if lo is None:
        lo = 0.6 if dtype == torch.float16 else 0.5
    gshape = (B, T, HO) if variant in SCALAR_GATE_VARIANTS else (B, T, HO, K)
    if alpha:
        g = torch.empty(gshape, device=device, dtype=torch.float32).uniform_(lo, 1.0).log()
    else:
        g = torch.zeros(gshape, device=device, dtype=torch.float32)
    if variant == "gdn2":
        b = (torch.rand(B, T, HO, K, device=device).sigmoid() * 2.0).to(dtype) if beta else torch.ones(B, T, HO, K, device=device, dtype=dtype)
        wt = torch.rand(B, T, HO, V, device=device).sigmoid().to(dtype) if w else torch.ones(B, T, HO, V, device=device, dtype=dtype)
        return {"g": g, "beta": b, "w": wt}
    rows = T * n
    b = torch.rand(B, rows, HO, device=device) if beta else torch.ones(B, rows, HO, device=device)
    return {"g": g, "beta": b}


def make_case(
    variant,
    dtype,
    *,
    B=1,
    T=None,
    seq_lens=None,
    H=2,
    HK=None,
    HV=None,
    K=128,
    V=128,
    n=None,
    alpha=True,
    beta=True,
    w=True,
    lo=None,
    gate_dtype=None,
    beta_dtype=None,
    cu_dtype=torch.int32,
    seed=SEED,
):
    """Dense ``(B, T)`` or packed varlen (``seq_lens``, B == 1) inputs plus the matching ``cu_seqlens``; ``HK``
    defaults to ``H``. ``n`` sub-tokens per token (``HOUSEHOLDER`` for a sub-token family, 1 otherwise) put k, v
    and beta on the expanded timeline of ``T * n`` rows while q, g and ``cu_seqlens`` stay at real tokens."""
    if n is None:
        n = HOUSEHOLDER if variant in HOUSEHOLDER_VARIANTS else 1
    assert n == 1 or variant in HOUSEHOLDER_VARIANTS, f"{variant} has no sub-token expansion"
    set_seed(seed)
    HV = H if HV is None else HV
    HK = H if HK is None else HK
    HO = max(H, HV)
    if seq_lens is not None:
        total = sum(seq_lens)
        bounds = [0]
        for sl in seq_lens:
            bounds.append(bounds[-1] + sl)
        cu = torch.tensor(bounds, dtype=cu_dtype, device="cuda")
        B, T, N, varlen = 1, total, len(seq_lens), True
    else:
        cu = torch.arange(0, B + 1, dtype=cu_dtype, device="cuda") * T
        N, varlen = B, False
    q, k, v = gen_qkv(B, T, H, HV, K, V, dtype)
    if n > 1:
        set_seed(seed + 1)
        k, v = gen_qkv(B, T * n, H, HV, K, V, dtype)[1:]
    if HK != H:
        from .conftest import multidist_randu

        k = F.normalize(multidist_randu(B * T * n * HK, K, device="cuda").reshape(B, T * n, HK, K), p=2.0, dim=-1).to(dtype).contiguous()
    gates = gen_gates(variant, B, T, HO, K, V, dtype, n=n, alpha=alpha, beta=beta, w=w, lo=lo)
    if gate_dtype is not None:
        gates["g"] = gates["g"].to(gate_dtype)
    if beta_dtype is not None:
        gates["beta"] = gates["beta"].to(beta_dtype)
    return Case(variant=variant, dtype=dtype, q=q, k=k, v=v, gates=gates, cu=cu, B=B, T=T, N=N, H=H, HK=HK, HV=HV, HO=HO, K=K, V=V, n=n, varlen=varlen)


def to_thd(x):
    return x.reshape(-1, *x.shape[2:])


def op(variant):
    return {
        "gdn": la_ops.gated_delta_net,
        "kda": la_ops.kimi_delta_attention,
        "gdn2": la_ops.gated_delta_net_v2,
        "gdp": la_ops.gated_delta_product,
    }[variant]


def case_tensors(case):
    """The op's tensor inputs by leaf name, dense ``(B, rows, ...)``."""
    tensors = {"q": case.q, "k": case.k, "v": case.v}
    tensors.update((name, case.gates[name]) for name in LEAF_NAMES[case.variant][3:])
    return tensors


def thd_tensors(case, window=None):
    """Fresh THD copies of the leaf tensors, cut to the real-token window ``(t0, t1)`` of the packed batch when given."""
    out = []
    for name, t in case_tensors(case).items():
        t = to_thd(t)
        if window is not None:
            scale = case.n if name in EXPANDED_LEAVES else 1
            t = t[window[0] * scale : window[1] * scale]
        out.append(t.detach().clone())
    return out


def op_tail(case, cu=None):
    """The positional arguments after the tensors: ``cu_seqlens`` and, for a sub-token family, ``num_householder``."""
    tail = [case.cu if cu is None else cu]
    if case.variant in HOUSEHOLDER_VARIANTS:
        tail.append(case.n)
    return tail


def op_args(case, cu=None, window=None):
    """Every positional op argument; a ``window`` cuts the packed batch to one sequence of ``t1 - t0`` tokens."""
    if window is not None:
        cu = torch.tensor([0, window[1] - window[0]], dtype=torch.int32, device="cuda")
    return thd_tensors(case, window) + op_tail(case, cu)


def window(case, t0, t1):
    """The tokens ``[t0, t1)`` of every row of a dense case (or of the packed batch) as a case of their own."""

    def cut(name, t):
        scale = case.n if name in EXPANDED_LEAVES else 1
        return t[:, t0 * scale : t1 * scale].contiguous()

    T = t1 - t0
    cu = torch.arange(0, case.B + 1, dtype=case.cu.dtype, device="cuda") * T
    gates = {name: cut(name, t) for name, t in case.gates.items()}
    return case.clone(q=cut("q", case.q), k=cut("k", case.k), v=cut("v", case.v), gates=gates, cu=cu, T=T, N=case.B)


def random_state(case, *, scale=0.05, seed=SEED + 1, dtype=torch.float32):
    """A seeded random ``[N, HO, V, K]`` state (or state gradient) of the case at ``scale``, in ``dtype``."""
    set_seed(seed)
    return (torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * scale).to(dtype)


def run_fwd(backend, case, *, cu=None, **kw):
    with waive_unsupported(backend, case.variant):
        return pinned_op(backend, case.variant)(*op_args(case, cu=cu), **kw)


REFERENCES = {"gdn": gdn_reference, "kda": kda_reference, "gdn2": gdn2_reference, "gdp": gdp_reference}


def reference_call(variant, tensors, n=1, **kwargs):
    """The family's fp64 reference on leaf tensors by name (autograd passes through); a sub-token family takes the
    scalar safe-gate transform host-side."""
    args = [tensors[name] for name in LEAF_NAMES[variant]]
    if variant in HOUSEHOLDER_VARIANTS:
        kwargs.pop("gate_lower_bound", None)
        a_log, dt_bias = kwargs.pop("a_log", None), kwargs.pop("dt_bias", None)
        if kwargs.pop("safe_gate", False):
            g = args[3].double()
            g = -F.softplus(g if dt_bias is None else g + dt_bias.double())
            args[3] = g if a_log is None else a_log.double().exp() * g
        kwargs["num_householder"] = n
    return REFERENCES[variant](*args, **kwargs)


def reference(
    case, *, scale=None, initial_state=None, l2norm=False, cu=None, beta_guard=False, v=None, safe_gate=False, a_log=None, dt_bias=None, use_beta_sigmoid=False
):
    """fp64 reference forward of a case; returns (o, final_state) with V-major states."""
    tensors = dict(case_tensors(case))
    if l2norm:
        tensors["q"] = F.normalize(case.q.float(), dim=-1)
        tensors["k"] = F.normalize(case.k.float(), dim=-1)
    if v is not None:
        tensors["v"] = v
    kwargs = dict(scale=scale, initial_state=initial_state)
    if beta_guard:
        kwargs["beta_guard"] = True
    if safe_gate:
        kwargs.update(safe_gate=True, a_log=a_log, dt_bias=dt_bias)
    if use_beta_sigmoid:
        kwargs["use_beta_sigmoid"] = True
    if case.varlen or cu is not None:
        kwargs["cu_seqlens"] = case.cu if cu is None else cu
    return reference_call(case.variant, tensors, case.n, **kwargs)


def assert_rms_close(name, out, want, tol):
    out = out.float()
    assert torch.isfinite(out).all(), f"non-finite values in {name}"
    r = rms_ratio(out.reshape(want.shape), want)
    print(f"{name}: rms ratio {r:.3e} (tol {tol:g})")
    assert r < tol, f"{name} rms ratio {r:.4g} >= {tol}"


def assert_fwd_parity(backend, case, *, scale=None, use_initial_state=False, state_dtype=torch.float32, l2norm=False, beta_guard=False, seed=SEED + 1):
    set_seed(seed)
    state0 = None
    if use_initial_state:
        state0 = (torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05).to(state_dtype)
    op_kw = dict(scale=scale, initial_state=state0, output_final_state=True, use_qk_l2norm_in_kernel=l2norm)
    if beta_guard:
        op_kw["beta_guard"] = True
    o, fs = run_fwd(backend, case, **op_kw)
    o_ref, fs_ref = reference(case, scale=scale, initial_state=state0, l2norm=l2norm, beta_guard=beta_guard)
    if state0 is not None:
        assert fs.dtype == state_dtype, f"final_state is {fs.dtype}, initial_state is {state_dtype}"
    assert_rms_close("o", o, o_ref, FWD_TOL[case.dtype])
    if fs is not None and fs.numel():
        assert_rms_close("final_state", fs, fs_ref, STATE_TOL[case.dtype])


# ---------------------------------------------------------------------------
# Backend pin seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_plan_name_pins_the_backend(backend, variant):
    """The named plan serves the graph (a decline waives); a plan name no plan
    carries raises instead of falling back to default routing."""
    case = make_case(variant, torch.bfloat16, T=4 * CHUNK[variant])
    with waive_unsupported(backend, variant):
        pinned_op(backend, variant)(*op_args(case))
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        op(variant)(*op_args(case), plan_name=f"{variant}_{backend.name}_absent")


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H,HV", HEAD_CONFIGS)
@pytest.mark.parametrize("B,T", [(1, 64), (1, 128), (2, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_basic(backend, variant, dtype, B, T, H, HV):
    if dtype == torch.float16 and (H, HV) not in HEAD_CONFIGS_SMALL:
        pytest.skip("fp16 runs the small head matrix")
    if (B, T) != (2, 256) and (H, HV) not in HEAD_CONFIGS_SMALL:
        pytest.skip("the full head matrix runs the two-sequence length")
    if backend.name == "cutile" and (H, HV) not in HEAD_CONFIGS_SMALL:
        pytest.skip("cuTile autotunes per shape; the small head matrix covers it")
    case = make_case(variant, dtype, B=B, T=T, H=H, HV=HV)
    assert_fwd_parity(backend, case)


@pytest.mark.parametrize("alpha,beta,w", [(True, False, True), (False, True, True), (True, True, False)], ids=["no_beta", "no_alpha", "no_w"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_gate_combinations(backend, variant, alpha, beta, w):
    if variant not in SCALAR_GATE_VARIANTS and not alpha:
        pytest.skip("the delta-rule inverse needs decay")
    if variant != "gdn2" and not w:
        pytest.skip("w is a GDN-2 gate")
    case = make_case(variant, torch.bfloat16, T=192, alpha=alpha, beta=beta, w=w)
    assert_fwd_parity(backend, case)


@pytest.mark.parametrize("scale", [0.5, 1.0, None], ids=["half", "one", "auto"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_scale(backend, variant, scale):
    case = make_case(variant, torch.bfloat16, T=192)
    assert_fwd_parity(backend, case, scale=scale)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_default_scale_matches_explicit(backend, variant):
    case = make_case(variant, torch.bfloat16, T=128)
    o_default, _ = run_fwd(backend, case)
    o_explicit, _ = run_fwd(backend, case, scale=1.0 / math.sqrt(case.K))
    torch.testing.assert_close(o_default, o_explicit)


@pytest.mark.parametrize("T", EDGE_LENS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_seqlen_edges(backend, variant, T):
    """Lengths straddling the kernels' chunk boundaries (16 and 64)."""
    case = make_case(variant, torch.bfloat16, T=T)
    assert_fwd_parity(backend, case)


@pytest.mark.parametrize("H,HV", [(1, 1), (2, 4)])
@pytest.mark.parametrize("seq_lens", RAGGED_SEQ_LENS, ids=lambda sl: f"{len(sl)}seqs_{sum(sl)}tok")
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_varlen_ragged(backend, variant, seq_lens, H, HV):
    if (H, HV) == (1, 1) and seq_lens not in ([96, 32, 160, 1], [7] * 24 + [1] * 8):
        pytest.skip("the single-head pair runs the two boundary-dense shapes")
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens, H=H, HV=HV)
    assert_fwd_parity(backend, case)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_many_short_sequences(backend, variant):
    """A 200-sequence packed batch matches the same sequences run one by one."""
    T = 33
    case = make_case(variant, torch.bfloat16, seq_lens=[T] * 200)
    o, fs = run_fwd(backend, case, output_final_state=True)
    for i in (0, 1, 99, 199):
        sl = slice(T * i, T * (i + 1))
        with waive_unsupported(backend, variant):
            o_i, fs_i = pinned_op(backend, variant)(*op_args(case, window=(sl.start, sl.stop)), output_final_state=True)
        assert_rms_close(f"o[seq {i}]", o[sl], o_i.float(), FWD_TOL[torch.bfloat16])
        assert_rms_close(f"final_state[seq {i}]", fs[i], fs_i[0].float(), STATE_TOL[torch.bfloat16])


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_zero_length_sequences(backend, variant):
    """Empty sequences must not perturb their neighbors; their state rows stay
    zero (or pass the initial state through when one is given)."""
    case = make_case(variant, torch.bfloat16, seq_lens=[64, 128])
    o_base, fs_base = run_fwd(backend, case, output_final_state=True)
    cu = torch.tensor([0, 64, 64, 192, 192], dtype=torch.int32, device="cuda")
    o, fs = run_fwd(backend, case, cu=cu, output_final_state=True)
    torch.testing.assert_close(o, o_base, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(fs[0], fs_base[0], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(fs[2], fs_base[1], atol=1e-3, rtol=1e-3)
    assert (fs[1] == 0).all() and (fs[3] == 0).all(), "zero-length sequence states must stay zero"
    state0 = torch.randn(4, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05
    o, fs_state0 = run_fwd(backend, case, cu=cu, initial_state=state0, output_final_state=True)
    torch.testing.assert_close(fs_state0[1], state0[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(fs_state0[3], state0[3], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("T", [128, 251])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_initial_state(backend, variant, T):
    case = make_case(variant, torch.bfloat16, T=T)
    assert_fwd_parity(backend, case, use_initial_state=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_split_initial_state(backend, variant):
    """With an initial state at a split-inducing length, the default schedule and
    the uncut batch-invariant schedule agree with each other and with the reference."""
    case = make_case(variant, torch.bfloat16, B=1, T=SPLIT_T)
    set_seed(SEED + 1)
    state0 = torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05
    o_split, fs_split = run_fwd(backend, case, initial_state=state0, output_final_state=True)
    o_uncut, fs_uncut = run_fwd(backend, case, initial_state=state0, output_final_state=True, batch_invariant=True)
    assert_rms_close("o split-vs-uncut", o_split, o_uncut.float(), FWD_TOL[torch.bfloat16])
    assert_rms_close("final_state split-vs-uncut", fs_split, fs_uncut.float(), STATE_TOL[torch.bfloat16])
    o_ref, fs_ref = reference(case, initial_state=state0)
    assert_rms_close("o vs reference", o_split, o_ref, FWD_TOL[torch.bfloat16])
    assert_rms_close("final_state vs reference", fs_split, fs_ref, STATE_TOL[torch.bfloat16])


@pytest.mark.parametrize("T1,T2", [(128, 128), (64, 192), (192, 121)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_chunked_prefill(backend, variant, T1, T2):
    """Two-phase prefill where phase 1's final state feeds phase 2; the
    concatenated output matches a single-shot reference."""
    case = make_case(variant, torch.bfloat16, B=2, T=T1 + T2)

    def run_phase(t0, t1, state0):
        return run_fwd(backend, window(case, t0, t1), initial_state=state0, output_final_state=True)

    o1, fs1 = run_phase(0, T1, None)
    o2, fs2 = run_phase(T1, T1 + T2, fs1)
    o = torch.cat([o1.reshape(case.B, T1, case.HO, case.V), o2.reshape(case.B, T2, case.HO, case.V)], dim=1)
    o_ref, fs_ref = reference(case)
    assert_rms_close("o", o, o_ref, 1.5 * FWD_TOL[case.dtype])
    assert_rms_close("final_state", fs2, fs_ref, 1.5 * STATE_TOL[case.dtype])


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_packed_matches_per_sequence(backend, variant):
    B, T = 3, 128
    case = make_case(variant, torch.bfloat16, B=B, T=T)
    o, fs = run_fwd(backend, case, output_final_state=True)
    cu1 = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    for b in range(B):
        args = [t[b] for t in case_tensors(case).values()]
        with waive_unsupported(backend, variant):
            o_b, fs_b = pinned_op(backend, variant)(*args, *op_tail(case, cu1), output_final_state=True)
        torch.testing.assert_close(o[b * T : (b + 1) * T], o_b)
        torch.testing.assert_close(fs[b], fs_b[0])


@pytest.mark.parametrize("K,V", HEAD_DIMS + WIDE_HEAD_DIMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_head_dims(backend, variant, K, V):
    """K/V head-dim variants; an engine that does not serve a combination declines."""
    case = make_case(variant, torch.bfloat16, T=192, K=K, V=V)
    assert_fwd_parity(backend, case)


@pytest.mark.parametrize("K,V", HEAD_DIMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_head_dims(backend, variant, K, V):
    """Backward over the full K x V head-dim matrix (with the in-kernel Q/K l2 norm
    in test_bwd_head_dims_states_varlen)."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=192, H=2, HV=4, K=K, V=V))


@pytest.mark.parametrize("K,V", HEAD_DIMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_head_dims_states_varlen(backend, variant, K, V):
    if backend.name == "cutile" and (K, V) != (128, 128):
        pytest.skip("cuTile autotunes per shape; one head-dim pair covers it")
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=[64, 128], K=K, V=V), use_initial_state=True, use_dfs=True, l2norm=True)


@pytest.mark.parametrize("H,HK,HV", GQA_CONFIGS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_gqa(backend, variant, H, HK, HV):
    """Grouped heads: native K at HK == HV, the expanded-k form and shared-kv GVA (HK == HV > H); gates, O and
    states live at HO = max(H, HV)."""
    case = make_case(variant, torch.bfloat16, T=192, H=H, HK=HK, HV=HV)
    assert_fwd_parity(backend, case)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_multi_tile(backend, variant):
    """B*H well above the SM count, so each CTA walks several (b, h) tiles back
    to back."""
    case = make_case(variant, torch.bfloat16, B=8, T=192, H=64)
    assert_fwd_parity(backend, case)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_qk_l2norm(backend, variant):
    """In-kernel Q/K L2 norm matches the reference on pre-normalized inputs."""
    case = make_case(variant, torch.bfloat16, T=256)
    assert_fwd_parity(backend, case, l2norm=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_strong_decay_varlen(backend, variant):
    case = make_case(variant, torch.bfloat16, seq_lens=[100, 2048, 0, 517], lo=0.1 if variant in SCALAR_GATE_VARIANTS else 0.3)
    assert_fwd_parity(backend, case)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_output_contract(backend, variant):
    """O is io-dtype at HO heads; final_state is empty unless requested."""
    case = make_case(variant, torch.bfloat16, T=128, H=2, HV=4)
    o, fs = run_fwd(backend, case)
    assert o.shape == (case.T, case.HO, case.V) and o.dtype == case.dtype
    assert fs.numel() == 0
    o, fs = run_fwd(backend, case, output_final_state=True)
    assert fs.shape == (case.N, case.HO, case.V, case.K) and fs.dtype == torch.float32
    o_ref, fs_ref = reference(case)
    assert_rms_close("o", o, o_ref, FWD_TOL[case.dtype])
    assert_rms_close("final_state", fs, fs_ref, STATE_TOL[case.dtype])


# ---------------------------------------------------------------------------
# Backward parity (oracle: fp64 autograd through the references)
# ---------------------------------------------------------------------------


def assert_bwd_parity(
    backend,
    case,
    *,
    scale=None,
    use_initial_state=False,
    state_dtype=torch.float32,
    use_dfs=False,
    l2norm=False,
    beta_guard=False,
    gate_grad_tol=None,
    seed=SEED + 1,
):
    variant, tol = case.variant, BWD_TOL[case.dtype]
    tensors = case_tensors(case)
    op_leaves = {name: to_thd(t).detach().clone().requires_grad_(True) for name, t in tensors.items()}
    ref_leaves = {name: t.detach().double().requires_grad_(True) for name, t in tensors.items()}
    set_seed(seed)
    state0_op = state0_ref = None
    if use_initial_state:
        state0 = (torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05).to(state_dtype)
        state0_op = state0.detach().clone().requires_grad_(True)
        state0_ref = state0.detach().double().requires_grad_(True)

    with waive_unsupported(backend, variant):
        args = list(op_leaves.values()) + op_tail(case)
        op_kw = dict(scale=scale, initial_state=state0_op, output_final_state=True, use_qk_l2norm_in_kernel=l2norm)
        if beta_guard:
            op_kw["beta_guard"] = True
        o, fs = pinned_op(backend, variant)(*args, **op_kw)
        dO = torch.randn_like(o)
        outputs, grad_outputs = [o], [dO]
        dFS = None
        if use_dfs:
            dFS = torch.randn_like(fs) * 0.1
            outputs.append(fs)
            grad_outputs.append(dFS)
        grad_inputs = list(op_leaves.values()) + ([state0_op] if use_initial_state else [])
        grads = torch.autograd.grad(outputs, grad_inputs, grad_outputs)

    ref_tensors = dict(ref_leaves)
    if l2norm:
        ref_tensors["q"], ref_tensors["k"] = F.normalize(ref_leaves["q"], dim=-1), F.normalize(ref_leaves["k"], dim=-1)
    ref_kwargs = dict(scale=scale, initial_state=state0_ref)
    if beta_guard:
        ref_kwargs["beta_guard"] = True
    if case.varlen:
        ref_kwargs["cu_seqlens"] = case.cu
    o_ref, fs_ref = reference_call(variant, ref_tensors, case.n, **ref_kwargs)
    assert_rms_close("o", o, o_ref, FWD_TOL[case.dtype])
    if fs is not None and fs.numel():
        assert_rms_close("final_state", fs, fs_ref, STATE_TOL[case.dtype])
    ref_outputs, ref_gos = [o_ref], [dO.double().reshape(o_ref.shape)]
    if use_dfs:
        ref_outputs.append(fs_ref)
        ref_gos.append(dFS.double().reshape(fs_ref.shape))
    ref_grads = torch.autograd.grad(ref_outputs, list(ref_leaves.values()) + ([state0_ref] if use_initial_state else []), ref_gos)

    names = list(op_leaves) + (["initial_state"] if use_initial_state else [])
    for name, got, want in zip(names, grads, ref_grads):
        if name == "initial_state":
            tol_n = STATE_GRAD_TOL
        elif name in ("g", "beta", "w") and gate_grad_tol is not None:
            tol_n = gate_grad_tol
        else:
            tol_n = tol
        assert_rms_close(f"d{name}", got, want, tol_n)


@pytest.mark.parametrize("H,HV", HEAD_CONFIGS_SMALL + [(16, 64)])
@pytest.mark.parametrize("T", [64, 128, 251])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_parity(backend, variant, dtype, T, H, HV):
    if dtype == torch.float16 and (T != 128 or (H, HV) != (1, 1)):
        pytest.skip("fp16 runs one representative backward config")
    if (H, HV) == (16, 64) and T != 128:
        pytest.skip("the large GVA config runs one length")
    if backend.name == "cutile" and T != 128:
        pytest.skip("cuTile autotunes per shape; one length covers it")
    assert_bwd_parity(backend, make_case(variant, dtype, T=T, H=H, HV=HV))


@pytest.mark.parametrize("H,HK,HV", GQA_CONFIGS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_gqa(backend, variant, H, HK, HV):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128, H=H, HK=HK, HV=HV))


@pytest.mark.parametrize("H,HK,HV,V", [(H, HK, HV, 128) for H, HK, HV in GQA_CONFIGS] + [(4, 4, 1, 64), (4, 1, 1, 64), (1, 2, 2, 64)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_gqa_qk_l2norm(backend, variant, H, HK, HV, V):
    """The fused l2norm backward under GQA head folds; V = 64 at one fold of each kind."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=192, H=H, HK=HK, HV=HV, V=V), l2norm=True)


@pytest.mark.parametrize("seq_lens", [[64, 192], [31, 63, 93, 123]], ids=["two", "ragged"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_varlen(backend, variant, seq_lens):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=seq_lens))


@pytest.mark.parametrize("l2norm", [False, True], ids=["plain", "l2norm"])
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_partial_final_chunk(backend, variant, l2norm):
    """One sequence four tokens past a 64-token chunk boundary."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=[68], H=1), l2norm=l2norm)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_zero_length_sequence(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=[64, 0, 128]))


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_initial_state(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128), use_initial_state=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_split_initial_state(backend, variant):
    """Backward over a cut work-item table with an initial state matches the
    uncut (batch-invariant) table."""
    case = make_case(variant, torch.bfloat16, B=1, T=SPLIT_T)
    set_seed(SEED + 1)
    state0 = torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05
    tensors = case_tensors(case)
    dO, grads = None, {}
    for tag, kw in (("split", {}), ("uncut", {"batch_invariant": True})):
        leaves = [to_thd(t).detach().clone().requires_grad_(True) for t in tensors.values()]
        s0 = state0.detach().clone().requires_grad_(True)
        with waive_unsupported(backend, variant):
            o, _ = pinned_op(backend, variant)(*leaves, *op_tail(case), initial_state=s0, output_final_state=True, **kw)
        if dO is None:
            dO = torch.randn_like(o)
        grads[tag] = torch.autograd.grad([o], leaves + [s0], [dO])
    for name, got, want in zip(list(tensors) + ["initial_state"], grads["split"], grads["uncut"]):
        assert_rms_close(f"d{name} split-vs-uncut", got, want.float(), BWD_TOL[torch.bfloat16])


@pytest.mark.parametrize("V", [128, 64])
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_split_d_final_state(backend, variant, V):
    """Backward over a cut work-item table with a final-state gradient matches
    the uncut (batch-invariant) table; only the piece owning the sequence end
    seeds from d_final_state."""
    case = make_case(variant, torch.bfloat16, B=1, T=SPLIT_T, V=V)
    set_seed(SEED + 1)
    state0 = torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05
    tensors = case_tensors(case)
    dO, dFinal, grads = None, None, {}
    for tag, kw in (("split", {}), ("uncut", {"batch_invariant": True})):
        leaves = [to_thd(t).detach().clone().requires_grad_(True) for t in tensors.values()]
        s0 = state0.detach().clone().requires_grad_(True)
        with waive_unsupported(backend, variant):
            o, final_state = pinned_op(backend, variant)(*leaves, *op_tail(case), initial_state=s0, output_final_state=True, **kw)
            if dO is None:
                dO = torch.randn_like(o)
                dFinal = torch.randn_like(final_state) * 0.05
            grads[tag] = torch.autograd.grad([o, final_state], leaves + [s0], [dO, dFinal])
    for name, got, want in zip(list(tensors) + ["initial_state"], grads["split"], grads["uncut"]):
        assert_rms_close(f"d{name} split-vs-uncut", got, want.float(), BWD_TOL[torch.bfloat16])


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_d_final_state(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128), use_initial_state=True, use_dfs=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_d_final_state_partial_chunk(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=251), use_dfs=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_scale(backend, variant):
    """A non-default scale reaches the backward path."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128), scale=1.0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_qk_l2norm(backend, variant):
    """dQ/dK must include the in-kernel normalization's own backward."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128), l2norm=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_no_decay_gate_grad_floor(backend, variant):
    """Alpha off; the gate gradients are compared at a loose bound and the data
    gradients at full tolerance."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=192, alpha=False), gate_grad_tol=0.3)


MODEL_GATE_LO = math.exp(-5.0)  # alpha at gate_lower_bound = -5, the model gate range


@pytest.mark.parametrize("variant", CHANNEL_VARIANTS)
def test_bwd_model_gate_range_bf16(backend, variant):
    """The per-channel gate with alpha down to exp(-5)."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128, lo=MODEL_GATE_LO))


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_with_checkpoints(backend, variant):
    """The checkpoint dump is non-differentiable and must not block backward."""
    ckpt = CHUNK[variant]
    case = make_case(variant, torch.bfloat16, T=4 * ckpt)
    args = thd_tensors(case)
    q_t, g_t = args[0].requires_grad_(True), args[3].requires_grad_(True)
    with waive_unsupported(backend, variant):
        o, fs, state_checkpoints = pinned_op(backend, variant)(*args, *op_tail(case), output_final_state=True, checkpoint_every_n_tokens=ckpt * case.n)
        assert not state_checkpoints.requires_grad
        (o.sum() + fs.sum()).backward()
    for name, t in (("q", q_t), ("g", g_t)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"bad grad for {name}"
    # the same run without checkpoints must match
    ref_args = thd_tensors(case)
    q_ref, g_ref = ref_args[0].requires_grad_(True), ref_args[3].requires_grad_(True)
    with waive_unsupported(backend, variant):
        o_ref, fs_ref = pinned_op(backend, variant)(*ref_args, *op_tail(case), output_final_state=True)
        (o_ref.sum() + fs_ref.sum()).backward()
    assert_rms_close("o", o, o_ref, FWD_TOL[case.dtype])
    assert_rms_close("final_state", fs, fs_ref, STATE_TOL[case.dtype])
    for name, t, r in (("q", q_t, q_ref), ("g", g_t, g_ref)):
        assert_rms_close(f"d{name}", t.grad.float(), r.grad.float(), BWD_TOL[case.dtype])


# ---------------------------------------------------------------------------
# Dtype surface (gate / beta / state widths, and the caches keyed on them)
# ---------------------------------------------------------------------------


def gate_dtype_pair(variant, gate_dtype, io_dtype=torch.bfloat16, **case_kw):
    """One case with a 16-bit gate and its EXACT fp32 twin (same values, wider
    storage)."""
    case = make_case(variant, io_dtype, **case_kw)
    g16 = case.gates["g"].to(gate_dtype)
    wide = case.clone(gates=dict(case.gates, g=g16.float()))
    narrow = case.clone(gates=dict(case.gates, g=g16))
    return narrow, wide


def bits(t):
    return t.contiguous().view(torch.uint8)


def assert_bitwise(name, got, want):
    assert got.dtype == want.dtype and got.shape == want.shape, f"{name}: {got.dtype} {tuple(got.shape)} vs {want.dtype} {tuple(want.shape)}"
    assert torch.equal(bits(got), bits(want)), f"{name} differs bitwise"


@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("T", [128, SPLIT_T])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_gate_16bit(backend, variant, T, gate_dtype):
    """SPLIT_T also drives the 16-bit gate through the split-K chunk scan."""
    narrow, wide = gate_dtype_pair(variant, gate_dtype, T=T, H=2)
    assert narrow.gates["g"].dtype == gate_dtype
    o16, fs16 = run_fwd(backend, narrow, output_final_state=True)
    o32, fs32 = run_fwd(backend, wide, output_final_state=True)
    assert_bitwise("o", o16, o32)
    assert_bitwise("final_state", fs16, fs32)
    assert_fwd_parity(backend, narrow)


@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("V", [128, 64])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_gate_16bit_varlen(backend, variant, gate_dtype, V):
    narrow, wide = gate_dtype_pair(variant, gate_dtype, seq_lens=[96, 32, 160, 1, 511], V=V)
    o16, _ = run_fwd(backend, narrow)
    o32, _ = run_fwd(backend, wide)
    assert_bitwise("o", o16, o32)
    assert_fwd_parity(backend, narrow)


@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_gate_16bit(backend, variant, gate_dtype):
    """dG leaves in the gate's own dtype, as the forward takes it."""
    narrow, _ = gate_dtype_pair(variant, gate_dtype, T=256, H=2)
    assert_bwd_parity(backend, narrow, gate_grad_tol=8e-2)


@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_gate_grad_dtype(backend, variant, gate_dtype):
    """The gate gradient comes back at the gate's dtype, carrying the same
    values as the exact fp32-gate twin (the gate is widened on load)."""
    case = make_case(variant, torch.bfloat16, T=128, H=2, gate_dtype=gate_dtype)
    set_seed(SEED + 6)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    grads = {}
    for arm, g in (("narrow", case.gates["g"]), ("wide", case.gates["g"].float())):
        tensors = dict(case_tensors(case), g=g)
        leaves = {name: to_thd(t).detach().clone().requires_grad_(True) for name, t in tensors.items()}
        with waive_unsupported(backend, variant):
            o, _ = pinned_op(backend, variant)(*leaves.values(), *op_tail(case), output_final_state=True)
            (grads[arm],) = torch.autograd.grad(o, [leaves["g"]], dO)
    assert grads["narrow"].dtype == gate_dtype, f"dG is {grads['narrow'].dtype}, gate is {gate_dtype}"
    assert torch.isfinite(grads["narrow"]).all()
    assert_rms_close("dG", grads["narrow"].float(), grads["wide"].float(), BWD_TOL[case.dtype])


@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_gate_16bit_safe_gate(backend, variant, gate_dtype):
    """Safe-gate takes the gate as raw logits, and its d_a_log / d_dt_bias
    reduction reads the raw gate again in the post-pass."""
    case, kw, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=256), "safe")
    g16 = case.gates["g"].to(gate_dtype)
    o16, fs16 = run_fwd(backend, case.clone(gates=dict(case.gates, g=g16)), output_final_state=True, **kw)
    o32, fs32 = run_fwd(backend, case.clone(gates=dict(case.gates, g=g16.float())), output_final_state=True, **kw)
    assert_bitwise("o", o16, o32)
    assert_bitwise("final_state", fs16, fs32)


@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_bwd_meta_dtypes_match_eager(backend, variant, gate_dtype):
    """The register_fake kernel must declare the dtypes the op really allocates,
    or AOTAutograd traces the wrong gradient dtype."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    case = make_case(variant, torch.bfloat16, T=128, H=2, gate_dtype=gate_dtype)
    name = OP_NAMES[variant]
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    args = [dO, *thd_tensors(case), *op_tail(case), float(case.K**-0.5)]
    state0 = (torch.randn(case.N, case.HO, case.V, case.K, device="cuda") * 0.05).to(torch.bfloat16)
    kw = dict(initial_state=state0, d_final_state=state0.clone())
    bwd = getattr(torch.ops.cudnn, name + "_bwd")
    with waive_declined(f"{variant} with a bf16 state"):
        eager = bwd(*args, **kw)
        with FakeTensorMode(allow_non_fake_inputs=True):
            meta = bwd(*args, **kw)
    got = [(i, e.dtype, m.dtype) for i, (e, m) in enumerate(zip(eager, meta)) if e.dtype != m.dtype]
    assert not got, f"meta/eager dtype mismatch at output indices {got}"


@pytest.mark.parametrize("io_dtype", [torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_gate_16bit_io_cross(backend, variant, gate_dtype, io_dtype):
    """The gate dtype is independent of the io dtype: both gate dtypes under fp16 io (bf16 io is test_fwd_gate_16bit)."""
    narrow, wide = gate_dtype_pair(variant, gate_dtype, io_dtype=io_dtype, T=256, H=2)
    o16, fs16 = run_fwd(backend, narrow, output_final_state=True)
    o32, fs32 = run_fwd(backend, wide, output_final_state=True)
    assert_bitwise("o", o16, o32)
    assert_bitwise("final_state", fs16, fs32)


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("V", [128, 64])
@pytest.mark.parametrize("variant", SCALAR_BETA_VARIANTS)
def test_fwd_beta_dtype(backend, variant, beta_dtype, V):
    """beta is accepted at fp32 or the io dtype and widened on the scalar load,
    so a 16-bit beta and its exact fp32 twin must agree bit for bit."""
    case = make_case(variant, torch.bfloat16 if beta_dtype == torch.float32 else beta_dtype, T=256, H=2, V=V)
    b16 = case.gates["beta"].to(beta_dtype)
    narrow = case.clone(gates=dict(case.gates, beta=b16))
    wide = case.clone(gates=dict(case.gates, beta=b16.float()))
    o16, fs16 = run_fwd(backend, narrow, output_final_state=True)
    o32, fs32 = run_fwd(backend, wide, output_final_state=True)
    assert_bitwise("o", o16, o32)
    assert_bitwise("final_state", fs16, fs32)
    assert_fwd_parity(backend, narrow)


@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", SCALAR_BETA_VARIANTS)
def test_bwd_beta_grad_dtype(backend, variant, beta_dtype):
    """dBeta leaves in beta's own dtype and carries the same values as the
    fp32-beta twin."""
    case = make_case(variant, torch.bfloat16, T=128, H=2, beta_dtype=beta_dtype)
    set_seed(SEED + 5)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    grads = {}
    for arm, beta in (("narrow", case.gates["beta"]), ("wide", case.gates["beta"].float())):
        leaves = {name: to_thd(t).detach().clone().requires_grad_(True) for name, t in (("q", case.q), ("g", case.gates["g"]), ("beta", beta))}
        args = thd_tensors(case)
        args[0], args[3], args[4] = leaves["q"], leaves["g"], leaves["beta"]
        with waive_unsupported(backend, variant):
            o, _ = pinned_op(backend, variant)(*args, *op_tail(case))
            (grads[arm],) = torch.autograd.grad(o, [leaves["beta"]], dO)
    assert grads["narrow"].dtype == beta_dtype, f"dBeta is {grads['narrow'].dtype}, beta is {beta_dtype}"
    assert torch.isfinite(grads["narrow"]).all()
    assert_rms_close("dBeta", grads["narrow"].float(), grads["wide"].float(), BWD_TOL[case.dtype])


@pytest.mark.parametrize("state_dtype", STATE_DTYPES, ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_state_dtype(backend, variant, state_dtype):
    """A bf16 initial_state; final_state comes back in initial_state's dtype and
    matches the reference."""
    assert_fwd_parity(backend, make_case(variant, torch.bfloat16, T=256), use_initial_state=True, state_dtype=state_dtype)


@pytest.mark.parametrize("state_dtype,V", [(torch.float32, 128), (torch.float32, 64), (torch.bfloat16, 128)], ids=["fp32-128", "fp32-64", "bf16-128"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_state_dtype(backend, variant, state_dtype, V):
    """A bf16 state pool drives a bf16 d_final_state into the kernel and a bf16
    d_initial_state out of it, with the reverse-scan accumulator still fp32."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128, V=V), use_initial_state=True, state_dtype=state_dtype, use_dfs=True)


@pytest.mark.parametrize("state_dtype", STATE_DTYPES, ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_state_grad_dtype(backend, variant, state_dtype):
    """The state gradients ride the state's own dtype end to end, asserted on the
    raw op rather than through autograd."""
    case = make_case(variant, torch.bfloat16, T=128, H=2)
    state0 = (torch.randn(case.N, case.HO, case.V, case.K, device="cuda") * 0.05).to(state_dtype)
    args = thd_tensors(case)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    dfs = (torch.randn(case.N, case.HO, case.V, case.K, device="cuda") * 0.1).to(state_dtype)
    with waive_unsupported(backend, variant):
        o, fs = pinned_op(backend, variant)(*args, *op_tail(case), initial_state=state0, output_final_state=True)
        assert fs.dtype == state_dtype, f"final_state is {fs.dtype}"
        out = getattr(torch.ops.cudnn, OP_NAMES[variant] + "_bwd")(dO, *args, *op_tail(case), float(case.K**-0.5), initial_state=state0, d_final_state=dfs)
    ds0 = out[-3]
    assert ds0.dtype == state_dtype, f"d_initial_state is {ds0.dtype}, initial_state is {state_dtype}"
    assert torch.isfinite(ds0).all()
    with waive_unsupported(backend, variant):
        wide = getattr(torch.ops.cudnn, OP_NAMES[variant] + "_bwd")(
            dO, *args, *op_tail(case), float(case.K**-0.5), initial_state=state0.float(), d_final_state=dfs.float()
        )
    assert_rms_close("d_initial_state", ds0.float(), wide[-3].float(), BWD_TOL[case.dtype])
    assert_rms_close("dQ", out[0].float(), wide[0].float(), BWD_TOL[case.dtype])


@pytest.mark.parametrize("state_dtype", STATE_DTYPES, ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_zero_length_sequence_state_dtype(backend, variant, state_dtype):
    """An empty sequence takes the scalar pass-through branch of the state-gradient
    store."""
    case = make_case(variant, torch.bfloat16, seq_lens=[64, 0, 128])
    assert_bwd_parity(backend, case, use_initial_state=True, state_dtype=state_dtype, use_dfs=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_state_grad_dtype_must_match_state(backend, variant):
    """A d_final_state whose dtype disagrees with initial_state is rejected at
    the op."""
    case = make_case(variant, torch.bfloat16, T=128, H=2)
    state0 = (torch.randn(case.N, case.HO, case.V, case.K, device="cuda") * 0.05).to(torch.bfloat16)
    args = thd_tensors(case)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    bwd = getattr(torch.ops.cudnn, OP_NAMES[variant] + "_bwd")
    dfs = torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.1
    with pytest.raises(TypeError, match="one state dtype per kernel"):
        bwd(dO, *args, *op_tail(case), float(case.K**-0.5), initial_state=state0, d_final_state=dfs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_do_dtype_must_match_io(backend, variant):
    """dO is declared at the io dtype on the graph, so a mismatched buffer is
    rejected at the op instead of being reinterpreted."""
    case = make_case(variant, torch.bfloat16, T=128, H=2)
    args = thd_tensors(case)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=torch.float16)
    bwd = getattr(torch.ops.cudnn, OP_NAMES[variant] + "_bwd")
    with pytest.raises(TypeError, match="dO"):
        bwd(dO, *args, *op_tail(case), float(case.K**-0.5))


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_state_grad_cache_separation(backend, variant):
    """Two state-gradient dtypes for one shape inside one process."""
    case = make_case(variant, torch.bfloat16, T=256, H=2)
    set_seed(SEED + 4)
    state = (torch.randn(case.N, case.HO, case.V, case.K, device="cuda") * 0.05).to(torch.bfloat16)
    dfs = (torch.randn(case.N, case.HO, case.V, case.K, device="cuda") * 0.1).to(torch.bfloat16)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    arms = []
    for narrow in (True, False):
        state0 = (state if narrow else state.float()).detach().clone().requires_grad_(True)
        leaves = {name: to_thd(t).detach().clone().requires_grad_(True) for name, t in case_tensors(case).items()}
        with waive_unsupported(backend, variant):
            o, fs = pinned_op(backend, variant)(*leaves.values(), *op_tail(case), initial_state=state0, output_final_state=True)
            grads = torch.autograd.grad([o, fs], [leaves["q"], state0], [dO, (dfs if narrow else dfs.float())])
        arms.append((grads[0], grads[1]))
    assert_bitwise("dQ", arms[0][0], arms[1][0])
    assert (arms[0][1].dtype, arms[1][1].dtype) == (torch.bfloat16, torch.float32)
    torch.testing.assert_close(arms[0][1].float(), arms[1][1], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("variant", VARIANTS)
def test_cu_seqlens_int64(backend, variant):
    """int64 cu_seqlens is a live cache-key element and produces the identical
    result."""
    narrow = make_case(variant, torch.bfloat16, seq_lens=[96, 32, 160, 1, 511], cu_dtype=torch.int32)
    wide = narrow.clone(cu=narrow.cu.to(torch.int64))
    o32, _ = run_fwd(backend, narrow)
    o64, _ = run_fwd(backend, wide)
    assert_bitwise("o", o64, o32)


@pytest.mark.parametrize("port", ["g", "beta", "initial_state", "cu"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_dtype_cache_separation(backend, variant, port):
    """Two dtypes for the SAME port inside one process, over values that are
    numerically identical in both (narrowed first, then widened back)."""
    if port == "beta" and variant == "gdn2":
        pytest.skip("gdn2 pins beta to the io dtype, so it has no second dtype to collide with")
    case = make_case(variant, torch.bfloat16, T=256, H=2)
    set_seed(SEED + 3)
    state = (torch.randn(case.N, case.HO, case.V, case.K, device="cuda") * 0.05).to(torch.bfloat16)
    arms = []
    for narrow in (True, False):
        gates, state0 = dict(case.gates), state.float()
        if port == "g":
            g16 = case.gates["g"].to(torch.bfloat16)
            gates["g"] = g16 if narrow else g16.float()
        elif port == "beta":
            b16 = case.gates["beta"].to(torch.bfloat16)
            gates["beta"] = b16 if narrow else b16.float()
        elif port == "initial_state":
            state0 = state if narrow else state.float()
        cu = case.cu.to(torch.int64) if (port == "cu" and narrow) else case.cu.to(torch.int32)
        arms.append(run_fwd(backend, case.clone(gates=gates, cu=cu), initial_state=state0, output_final_state=True))
    assert_bitwise("o", arms[0][0], arms[1][0])
    if port == "initial_state":
        assert (arms[0][1].dtype, arms[1][1].dtype) == (torch.bfloat16, torch.float32)
    else:
        assert_bitwise("final_state", arms[0][1], arms[1][1])


# ---------------------------------------------------------------------------
# Layout (innermost-contiguous inputs; outer strides pass straight to the kernels)
# ---------------------------------------------------------------------------


def strided_copy(t):
    """A non-contiguous copy of ``t`` in the leading columns of a buffer with a
    doubled innermost extent; every outer stride changes while stride(-1) stays 1."""
    wide = torch.empty(*t.shape[:-1], 2 * t.shape[-1], device=t.device, dtype=t.dtype)
    view = wide[..., : t.shape[-1]]
    view.copy_(t)
    assert view.stride(-1) == 1 and not view.is_contiguous()
    return view


def fused_qkv_views(case):
    """q/k/v as slices of one fused projection buffer per timeline; innermost-contiguous,
    never whole-tensor contiguous."""
    tensors = {"q": to_thd(case.q), "k": to_thd(case.k), "v": to_thd(case.v)}
    views = {}
    for rows in sorted({t.shape[0] for t in tensors.values()}):
        group = {name: t for name, t in tensors.items() if t.shape[0] == rows}
        widths = [t.shape[1] * t.shape[2] for t in group.values()]
        fused = torch.empty(rows, sum(widths) + (widths[0] if len(group) == 1 else 0), device="cuda", dtype=case.dtype)
        base = 0
        for (name, t), width in zip(group.items(), widths):
            view = fused[:, base : base + width].unflatten(-1, t.shape[1:])
            view.copy_(t)
            assert view.stride(-1) == 1 and not view.is_contiguous()
            views[name] = view
            base += width
    return [views["q"], views["k"], views["v"]]


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_innermost_contiguous_inputs(backend, variant):
    """Fused-projection q/k/v slices, strided gates and a strided initial state
    match the contiguous run bitwise."""
    case = make_case(variant, torch.bfloat16, seq_lens=[192, 251])
    set_seed(SEED + 1)
    state0 = torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05
    o_ref, fs_ref = run_fwd(backend, case, initial_state=state0, output_final_state=True)
    args = fused_qkv_views(case) + [strided_copy(to_thd(case.gates[name])) for name in LEAF_NAMES[variant][3:]]
    with waive_unsupported(backend, variant):
        o, fs = pinned_op(backend, variant)(*args, *op_tail(case), initial_state=strided_copy(state0), output_final_state=True)
    assert torch.equal(bits(o), bits(o_ref)), "strided inputs changed o"
    assert torch.equal(bits(fs), bits(fs_ref)), "strided inputs changed final_state"


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_innermost_contiguous_inputs(backend, variant):
    """Backward from strided leaves and a strided incoming dO matches the
    contiguous run bitwise for every gradient."""
    case = make_case(variant, torch.bfloat16, seq_lens=[192, 251])
    gate_names = list(LEAF_NAMES[variant][3:])

    def grads_from(leaves, dO):
        with waive_unsupported(backend, variant):
            o, _ = pinned_op(backend, variant)(*leaves, *op_tail(case))
            return torch.autograd.grad([o], leaves, [dO])

    set_seed(SEED + 3)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    contiguous = [to_thd(t).detach().clone() for t in (case.q, case.k, case.v)] + [to_thd(case.gates[n]).detach().clone() for n in gate_names]
    strided = fused_qkv_views(case) + [strided_copy(to_thd(case.gates[n])) for n in gate_names]
    grads_c = grads_from([t.requires_grad_(True) for t in contiguous], dO)
    grads_s = grads_from([t.requires_grad_(True) for t in strided], strided_copy(dO))
    for name, gc, gs in zip(["q", "k", "v", *gate_names], grads_c, grads_s):
        assert torch.equal(bits(gc), bits(gs)), f"d{name} differs between contiguous and strided inputs"


@pytest.mark.parametrize("backend", ["cutile"], indirect=True)
@pytest.mark.parametrize("variant", CUTILE_VARIANTS)
def test_cutile_rejects_strided_inputs(backend, variant):
    """The cuTile backend raises its contract error on a strided buffer instead
    of reading the padding or copying."""
    case = make_case(variant, torch.bfloat16, T=64)
    args = thd_tensors(case)
    args[0] = strided_copy(args[0])
    with waive_unsupported(backend, variant):
        with pytest.raises(ValueError, match="must be contiguous"):
            pinned_op(backend, variant)(*args, *op_tail(case))


# ---------------------------------------------------------------------------
# Checkpoints (per-chunk state series)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoints_match_prefix_final_states(backend, variant):
    """state_checkpoints[j] is the state AT token boundary j*ckpt, so row 0 is the
    state entering the sequence and the end is excluded; rows are a shape-derived
    capacity bound, valid entries pack first."""
    ckpt = CHUNK[variant]
    T = 5 * ckpt
    case = make_case(variant, torch.bfloat16, T=T)
    _, _, state_checkpoints = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=ckpt * case.n)
    valid = (T - 1) // ckpt + 1
    assert state_checkpoints.shape == (T // ckpt + 1, case.HO, case.V, case.K)
    assert state_checkpoints.dtype == case.dtype
    # row 0 is the incoming state, zero here since no initial_state was passed
    assert not state_checkpoints[0].any(), "row 0 must be the (zero) incoming state"
    for j in sorted({1, valid - 1}):
        n = j * ckpt
        with waive_unsupported(backend, variant):
            _, fs_p = pinned_op(backend, variant)(*op_args(case, window=(0, n)), output_final_state=True)
        assert_rms_close(f"state_checkpoints[{j}]", state_checkpoints[j], fs_p[0], STATE_TOL[case.dtype])


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoints_varlen(backend, variant):
    """Entries pack per sequence in order (one per ckpt tokens from the sequence
    start, end excluded); each entry matches its sequence's truncated prefix."""
    ckpt = CHUNK[variant]
    seq_lens = [3 * ckpt + 5, ckpt - 1, 0, 2 * ckpt]
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens)
    _, _, state_checkpoints = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=ckpt * case.n)
    counts = [(sl - 1) // ckpt + 1 if sl > 0 else 0 for sl in seq_lens]
    # shape[0] is a capacity bound; sum(counts) valid rows pack first, the tail is uninitialized
    assert state_checkpoints.shape[0] == max(sum(seq_lens) // ckpt + len(seq_lens), 1)
    bounds = case.cu.tolist()
    base = 0
    for n, cnt in enumerate(counts):
        for j in sorted({0, cnt - 1} if cnt else set()):
            if j == 0:
                assert not state_checkpoints[base].any(), f"seq {n} row 0 must be the (zero) incoming state"
                continue
            n0 = bounds[n]
            ntok = j * ckpt
            with waive_unsupported(backend, variant):
                _, fs_p = pinned_op(backend, variant)(*op_args(case, window=(n0, n0 + ntok)), output_final_state=True)
            assert_rms_close(f"state_checkpoints[seq {n}][{j}]", state_checkpoints[base + j], fs_p[0], STATE_TOL[case.dtype])
        base += cnt


TIGHT_VARLEN_RECIPES = {
    "pair+quarter": lambda c: [c + c // 4] * 2,
    "triple+1": lambda c: [c + 1] * 3,
    "single-2c+1": lambda c: [2 * c + 1],
}


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("recipe", sorted(TIGHT_VARLEN_RECIPES))
@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoints_varlen_tight_capacity(backend, variant, recipe):
    """Varlen lengths whose packed entries fill the capacity bound exactly; every
    entry matches the fp64 recurrence and its sequence's solo prefix run, and
    the chunk-cadence backward reuse matches the recompute path bitwise."""
    ckpt = CHUNK[variant]
    seq_lens = TIGHT_VARLEN_RECIPES[recipe](ckpt)
    counts = [(sl - 1) // ckpt + 1 if sl > 0 else 0 for sl in seq_lens]
    assert sum(counts) == max(sum(seq_lens) // ckpt + len(seq_lens), 1), "recipe must fill the capacity bound exactly"
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens)
    o, fs, state_checkpoints = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=ckpt * case.n)
    assert state_checkpoints.shape[0] == sum(counts)
    bounds = case.cu.tolist()
    base = 0
    for n, cnt in enumerate(counts):
        for j in range(cnt):
            if j == 0:
                assert not state_checkpoints[base].any(), f"seq {n} row 0 must be the (zero) incoming state"
                continue
            n0 = bounds[n]
            ntok = j * ckpt
            prefix = window(case, n0, n0 + ntok)
            _, fs_ref = reference(prefix)
            assert_rms_close(f"state_checkpoints[seq {n}][{j}] vs fp64 reference", state_checkpoints[base + j], fs_ref[0], STATE_TOL[case.dtype])
            _, fs_p = run_fwd(backend, prefix, output_final_state=True)
            assert_rms_close(f"state_checkpoints[seq {n}][{j}] vs solo prefix", state_checkpoints[base + j], fs_p[0], STATE_TOL[case.dtype])
        base += cnt
    grads_by_mode = []
    for mode_ckpt in (0, ckpt):
        args = thd_tensors(case)
        leaves = [args[0].requires_grad_(True), args[1].requires_grad_(True)]
        with waive_unsupported(backend, variant):
            out = pinned_op(backend, variant)(*args, *op_tail(case), checkpoint_every_n_tokens=mode_ckpt)
            set_seed(SEED + 5)
            dO = torch.randn_like(out[0])
            grads_by_mode.append(torch.autograd.grad([out[0]], leaves, [dO]))
    for gr, gc in zip(grads_by_mode[0], grads_by_mode[1]):
        assert torch.equal(bits(gr), bits(gc)), "checkpoint-reuse grads differ from the recompute path"


@pytest.mark.parametrize("ckpt_mult", [2, 3])
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoints_coarse_cadence(backend, variant, K, ckpt_mult):
    """Coarser cadences (multiples of the base chunk) keep the prefix contract."""
    ckpt = CHUNK[variant] * ckpt_mult
    T = 5 * ckpt
    case = make_case(variant, torch.bfloat16, T=T, K=K)
    o, fs, state_checkpoints = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=ckpt * case.n)
    valid = (T - 1) // ckpt + 1
    assert state_checkpoints.shape == (T // ckpt + 1, case.HO, case.V, case.K)
    assert state_checkpoints.dtype == case.dtype
    assert not state_checkpoints[0].any(), "row 0 must be the (zero) incoming state"
    for j in sorted({1, valid - 1}):
        n = j * ckpt
        with waive_unsupported(backend, variant):
            _, fs_p = pinned_op(backend, variant)(*op_args(case, window=(0, n)), output_final_state=True)
        assert_rms_close(f"state_checkpoints[{j}]", state_checkpoints[j], fs_p[0], STATE_TOL[case.dtype])


# ---------------------------------------------------------------------------
# Raw-logit gate modes (safe gate, in-kernel Beta sigmoid)
# ---------------------------------------------------------------------------


ABSENT_ARMS = ("no_a_log", "no_dt_bias", "no_params")
ABSENT_PARAMS = {"no_a_log": ("a_log",), "no_dt_bias": ("dt_bias",), "no_params": ("a_log", "dt_bias")}


def gate_mode_case(case, mode, *, arm=None, params="zero", gate_lower_bound=None, seed=SEED + 7):
    """(case, op kwargs, oracle kwargs) of a gate mode on a case.  "log" and "fp16" take the case as it is; "l2norm"
    and "beta_guard" (waived where the family has none) set the in-kernel flags; "safe" replaces the gate by raw logits
    with zero a_log / dt_bias (``params="random"`` draws them at 0.3; ``arm`` in ABSENT_ARMS zeroes the arm's absent
    parameters, so the op kwargs are the explicit-zero twin of the absent-parameter run) and hands a non-default
    ``gate_lower_bound`` to the channel-gate families; "sigmoid_beta" and "sigmoid_beta_fp32" replace beta by io-dtype
    or fp32 logits for the
    in-kernel sigmoid.  The oracle kwargs name the same transforms in the references' vocabulary."""
    set_seed(seed)
    if mode in ("log", "fp16"):
        return case, {}, {}
    if mode == "l2norm":
        return case, dict(use_qk_l2norm_in_kernel=True), dict(l2norm=True)
    if mode == "beta_guard":
        if case.variant not in BETA_GUARD_VARIANTS:
            pytest.skip(f"{case.variant} has no beta guard")
        return case, dict(use_qk_l2norm_in_kernel=True, beta_guard=True), dict(l2norm=True, beta_guard=True)
    if mode == "safe":
        graw = torch.randn_like(case.gates["g"])
        dt_shape = (case.HO,) if case.variant in SCALAR_GATE_VARIANTS else (case.HO, case.K)
        a_log = torch.randn(case.HO, device="cuda") * 0.3 if params == "random" else torch.zeros(case.HO, device="cuda")
        dt_bias = torch.randn(dt_shape, device="cuda") * 0.3 if params == "random" else torch.zeros(dt_shape, device="cuda")
        if arm is not None:
            a_log = torch.zeros_like(a_log) if "a_log" in ABSENT_PARAMS[arm] else a_log
            dt_bias = torch.zeros_like(dt_bias) if "dt_bias" in ABSENT_PARAMS[arm] else dt_bias
        kw = dict(safe_gate=True, a_log=a_log, dt_bias=dt_bias)
        if gate_lower_bound is not None and case.variant not in SCALAR_GATE_VARIANTS:
            kw["gate_lower_bound"] = gate_lower_bound
        return case.clone(gates=dict(case.gates, g=graw)), kw, dict(safe_gate=True, a_log=a_log, dt_bias=dt_bias)
    assert mode in ("sigmoid_beta", "sigmoid_beta_fp32"), mode
    if mode == "sigmoid_beta_fp32" and case.variant == "gdn2":
        pytest.skip("gdn2 pins beta to the io dtype, so it has no fp32 logit form")
    logits = torch.randn(case.gates["beta"].shape, device="cuda", dtype=torch.float32)
    if mode == "sigmoid_beta":
        logits = logits.to(case.dtype)
    return case.clone(gates=dict(case.gates, beta=logits)), dict(use_beta_sigmoid_in_kernel=True), dict(use_beta_sigmoid=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_safe_gate_forward_parity(backend, variant):
    """Raw logits with a_log = 0 / dt_bias = 0 match the post-activation path
    fed the host-side transform (``lb * sigmoid(g)``, or ``-softplus(g)`` for
    GDN's scalar gate)."""
    lb = -5.0
    raw_case, raw_kw, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=256), "safe")
    graw = raw_case.gates["g"]
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    raw_kw.update(kw)
    if variant in SCALAR_BETA_VARIANTS:
        braw = torch.randn_like(raw_case.gates["beta"].float()).to(raw_case.dtype)
        raw_case = raw_case.clone(gates=dict(raw_case.gates, beta=braw))
        raw_kw["use_beta_sigmoid_in_kernel"] = True
        eff_beta = braw.float().sigmoid()
    else:
        eff_beta = raw_case.gates["beta"]
    g_eff = -F.softplus(graw) if variant in SCALAR_GATE_VARIANTS else lb * torch.sigmoid(graw)
    eff_case = raw_case.clone(gates=dict(raw_case.gates, g=g_eff, beta=eff_beta))
    o_raw, fs_raw = run_fwd(backend, raw_case, **raw_kw)
    o_eff, fs_eff = run_fwd(backend, eff_case, **kw)
    assert_rms_close("o", o_raw, o_eff.double(), 2e-2)
    assert rms_ratio(fs_raw, fs_eff) < 2e-2


@pytest.mark.parametrize("variant", VARIANTS)
def test_allow_neg_eigval_parity(backend, variant):
    """The fused ``2 * sigmoid(beta)`` matches the same activation fed
    post-hoc, forward and backward. beta in (0, 2) is the regime where
    ``I - beta k k^T`` can reach negative eigenvalues."""
    case = make_case(variant, torch.bfloat16, T=128)
    logits = torch.randn_like(case.gates["beta"].float())
    activated = 2.0 * logits.to(case.dtype).float().sigmoid()
    eff_beta = activated.to(case.dtype) if variant == "gdn2" else activated
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    raw_case = case.clone(gates=dict(case.gates, beta=logits.to(case.dtype)))
    eff_case = case.clone(gates=dict(case.gates, beta=eff_beta))
    o_raw, fs_raw = run_fwd(backend, raw_case, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True, **kw)
    o_eff, fs_eff = run_fwd(backend, eff_case, **kw)
    assert_rms_close("o", o_raw, o_eff.double(), 2e-2)
    assert rms_ratio(fs_raw, fs_eff) < 2e-2


@pytest.mark.parametrize("variant", VARIANTS)
def test_beta_two_parity(backend, variant):
    """Raw beta in (0, 2) without the fused sigmoid, the regime where ``I - beta k k^T`` reaches negative
    eigenvalues; forward and backward against the reference."""
    case = make_case(variant, torch.bfloat16, T=128)
    beta = case.gates["beta"]
    case = case.clone(gates=dict(case.gates, beta=(torch.rand_like(beta.float()) * 1.95).to(beta.dtype)))
    assert_fwd_parity(backend, case)
    assert_bwd_parity(backend, case)


@pytest.mark.parametrize("variant", VARIANTS)
def test_allow_neg_eigval_requires_beta_sigmoid(backend, variant):
    """The 2x rides on the fused sigmoid; without it the op declines."""
    case = make_case(variant, torch.bfloat16, T=64)
    with pytest.raises(ValueError, match="allow_neg_eigval requires use_beta_sigmoid_in_kernel"):
        run_fwd(backend, case, allow_neg_eigval=True)


@pytest.mark.parametrize(
    "K,gate_dtype",
    [(128, torch.float32), (128, torch.bfloat16), (128, torch.float16), (64, torch.float32)],
    ids=["128-fp32", "128-bf16", "128-fp16", "64-fp32"],
)
@pytest.mark.parametrize("variant", VARIANTS)
def test_safe_gate_backward(backend, variant, K, gate_dtype):
    """dG comes back in raw-logit space and the parameter gradients satisfy
    their identities over dG (d_dt_bias = sum dg_raw; d_a_log = sum dg_raw *
    (g + dt_bias) per channel, or sum dg_raw * softplus(y) / sigmoid(y) for
    GDN's scalar gate)."""
    case, kw, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=128, K=K), "safe", params="random", seed=SEED + 9)
    a_leaf, dt_leaf = kw["a_log"].requires_grad_(True), kw["dt_bias"].requires_grad_(True)
    raw_gates = dict(case.gates, g=case.gates["g"].to(gate_dtype))
    kw["use_qk_l2norm_in_kernel"] = True
    if variant in SCALAR_BETA_VARIANTS:
        raw_gates["beta"] = torch.randn_like(case.gates["beta"].float()).to(case.dtype)
        kw["use_beta_sigmoid_in_kernel"] = True
    raw_case = case.clone(gates=raw_gates)
    g_leaf = to_thd(raw_gates["g"]).detach().clone().requires_grad_(True)
    beta_leaf = to_thd(raw_gates["beta"]).detach().clone().requires_grad_(True)
    args = thd_tensors(raw_case)
    args[0].requires_grad_(True)
    args[3], args[4] = g_leaf, beta_leaf
    with waive_unsupported(backend, variant):
        o, _ = pinned_op(backend, variant)(*args, *op_tail(case), **kw)
        o.sum().backward()
    assert g_leaf.grad.dtype == gate_dtype, f"dG is {g_leaf.grad.dtype}, gate is {gate_dtype}"
    dg_raw = g_leaf.grad.double()
    ddt_id = dg_raw.sum(0)
    if variant in SCALAR_GATE_VARIANTS:
        y = g_leaf.detach().double() + dt_leaf.detach().double()[None]
        da_id = (dg_raw * (F.softplus(y) / torch.sigmoid(y))).sum(0)
    else:
        da_id = (dg_raw * (g_leaf.detach().double() + dt_leaf.detach().double()[None])).sum(dim=(0, 2))
    # the identities are evaluated over the returned dG, so they hold to the gate dtype's precision
    ident_tol = 1e-4 if gate_dtype == torch.float32 else 5e-2
    for name, got, ident in (("d_dt_bias", dt_leaf.grad.double(), ddt_id), ("d_a_log", a_leaf.grad.double(), da_id)):
        scale = max(ident.abs().max().item(), 1e-6)
        assert (got - ident).abs().max().item() / scale < ident_tol, name
    for name, leaf in (("dq", args[0]), ("dbeta", beta_leaf)):
        assert leaf.grad is not None and bool(torch.isfinite(leaf.grad).all()), name


@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_safe_gate_param_16bit(backend, variant, param_dtype):
    """16-bit a_log/dt_bias match their exact fp32 twins bitwise."""
    raw_case, kw, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=256), "safe", params="random", seed=SEED + 12)
    a16, dt16 = kw["a_log"].to(param_dtype), kw["dt_bias"].to(param_dtype)
    o16, fs16 = run_fwd(backend, raw_case, **dict(kw, a_log=a16, dt_bias=dt16, output_final_state=True))
    o32, fs32 = run_fwd(backend, raw_case, **dict(kw, a_log=a16.float(), dt_bias=dt16.float(), output_final_state=True))
    assert_bitwise("o", o16, o32)
    assert_bitwise("final_state", fs16, fs32)


@pytest.mark.parametrize("K,param_dtype", [(128, torch.bfloat16), (128, torch.float16), (64, torch.bfloat16)], ids=["128-bf16", "128-fp16", "64-bf16"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_safe_gate_backward_param_16bit(backend, variant, K, param_dtype):
    """d_a_log/d_dt_bias come back in the parameter dtype and satisfy the dG
    identities to that dtype's precision."""
    case, kw, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=128, K=K), "safe", params="random", seed=SEED + 9)
    a_leaf = kw["a_log"].to(param_dtype).requires_grad_(True)
    dt_leaf = kw["dt_bias"].to(param_dtype).requires_grad_(True)
    kw.update(a_log=a_leaf, dt_bias=dt_leaf, use_qk_l2norm_in_kernel=True)
    g_leaf = to_thd(case.gates["g"]).detach().clone().requires_grad_(True)
    args = thd_tensors(case)
    args[0].requires_grad_(True)
    args[3] = g_leaf
    with waive_unsupported(backend, variant):
        o, _ = pinned_op(backend, variant)(*args, *op_tail(case), **kw)
        o.sum().backward()
    assert a_leaf.grad.dtype == param_dtype, f"d_a_log is {a_leaf.grad.dtype}, a_log is {param_dtype}"
    assert dt_leaf.grad.dtype == param_dtype, f"d_dt_bias is {dt_leaf.grad.dtype}, dt_bias is {param_dtype}"
    dg_raw = g_leaf.grad.double()
    ddt_id = dg_raw.sum(0)
    if variant in SCALAR_GATE_VARIANTS:
        y = g_leaf.detach().double() + dt_leaf.detach().double()[None]
        da_id = (dg_raw * (F.softplus(y) / torch.sigmoid(y))).sum(0)
    else:
        da_id = (dg_raw * (g_leaf.detach().double() + dt_leaf.detach().double()[None])).sum(dim=(0, 2))
    for name, got, ident in (("d_dt_bias", dt_leaf.grad.double(), ddt_id), ("d_a_log", a_leaf.grad.double(), da_id)):
        scale = max(ident.abs().max().item(), 1e-6)
        assert (got - ident).abs().max().item() / scale < 5e-2, name


@pytest.mark.parametrize("arm", ABSENT_ARMS)
@pytest.mark.parametrize("T", [256, SPLIT_T])
@pytest.mark.parametrize("variant", VARIANTS)
def test_safe_gate_absent_params_fwd_bitwise(backend, variant, T, arm):
    """An absent a_log is unit amplitude and an absent dt_bias is zero bias; o
    and final_state are bitwise the explicit-zero-tensor run (SPLIT_T also
    drives the split-K chunk scan, whose plan key carries the params)."""
    raw_case, kw_ref, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=T), "safe", params="random", arm=arm, seed=SEED + 21)
    kw_abs = dict(kw_ref, **{name: None for name in ABSENT_PARAMS[arm]})
    o_ref, fs_ref = run_fwd(backend, raw_case, output_final_state=True, **kw_ref)
    o_abs, fs_abs = run_fwd(backend, raw_case, output_final_state=True, **kw_abs)
    assert_bitwise("o", o_abs, o_ref)
    assert_bitwise("final_state", fs_abs, fs_ref)


@pytest.mark.parametrize("arm", ABSENT_ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_safe_gate_absent_params_checkpoints_bitwise(backend, variant, arm):
    """The per-chunk state_checkpoints series of the absent-parameter run is
    bitwise the explicit-zero-tensor run."""
    ckpt = CHUNK[variant] * 2
    case, kw_ref, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=4 * ckpt), "safe", params="random", arm=arm, seed=SEED + 23)
    kw_abs = dict(kw_ref, **{name: None for name in ABSENT_PARAMS[arm]})
    kw = dict(output_final_state=True, checkpoint_every_n_tokens=ckpt * case.n)
    o_ref, fs_ref, ck_ref = run_fwd(backend, case, **kw, **kw_ref)
    o_abs, fs_abs, ck_abs = run_fwd(backend, case, **kw, **kw_abs)
    valid = case.N * ((case.T - 1) // ckpt + 1)
    assert_bitwise("o", o_abs, o_ref)
    assert_bitwise("final_state", fs_abs, fs_ref)
    assert_bitwise("state_checkpoints", ck_abs[:valid], ck_ref[:valid])


@pytest.mark.parametrize("K,arm", [(128, arm) for arm in ABSENT_ARMS] + [(64, "no_params")])
@pytest.mark.parametrize("variant", VARIANTS)
def test_safe_gate_absent_params_bwd_bitwise(backend, variant, K, arm):
    """Every gradient of the absent-parameter run, the initial-state gradient
    included, is bitwise the explicit-zero-tensor run; the absent parameter has
    no leaf, so autograd has nothing to return for it (its zero twin does get a
    finite grad)."""
    case, kw_ref, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=128, K=K), "safe", params="random", arm=arm, seed=SEED + 22)
    kw_abs = dict(kw_ref, **{name: None for name in ABSENT_PARAMS[arm]})
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    state0 = torch.randn(case.N, case.HO, case.V, case.K, device="cuda")
    d_final = torch.randn_like(state0)

    def run(kw):
        tensors = case_tensors(case)
        leaves = {name: to_thd(t).detach().clone().requires_grad_(True) for name, t in tensors.items()}
        leaves["initial_state"] = state0.detach().clone().requires_grad_(True)
        params = {name: (None if kw[name] is None else kw[name].detach().clone().requires_grad_(True)) for name in ("a_log", "dt_bias")}
        with waive_unsupported(backend, variant):
            o, fs = pinned_op(backend, variant)(
                *[leaves[n] for n in tensors],
                *op_tail(case),
                initial_state=leaves["initial_state"],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                **dict(kw, **params),
            )
            torch.autograd.backward([o, fs], [dO, d_final])
        return o, leaves, params

    o_ref, leaves_ref, params_ref = run(kw_ref)
    o_abs, leaves_abs, params_abs = run(kw_abs)
    assert_bitwise("o", o_abs, o_ref)
    for name in leaves_ref:
        assert_bitwise("d" + name, leaves_abs[name].grad, leaves_ref[name].grad)
    for name in ("a_log", "dt_bias"):
        assert params_ref[name].grad is not None and bool(torch.isfinite(params_ref[name].grad).all()), name
        if params_abs[name] is not None:
            assert_bitwise("d_" + name, params_abs[name].grad, params_ref[name].grad)


@pytest.mark.parametrize("arm", ABSENT_ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_safe_gate_absent_params_bwd_op_outputs(backend, variant, arm):
    """The raw bwd op returns an EMPTY d_a_log / d_dt_bias for an absent
    parameter, and register_fake declares the same shapes and dtypes."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    case, kw, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=128), "safe", params="random", arm=arm, seed=SEED + 24)
    kw.update({name: None for name in ABSENT_PARAMS[arm]})
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    args = [dO, *thd_tensors(case), *op_tail(case), float(case.K**-0.5)]
    bwd = getattr(torch.ops.cudnn, OP_NAMES[variant] + "_bwd")
    with waive_declined(f"{variant} safe_gate {arm}"):
        eager = bwd(*args, **kw)
        with FakeTensorMode(allow_non_fake_inputs=True):
            meta = bwd(*args, **kw)
    d_a_log, d_dt_bias = eager[-2], eager[-1]
    assert (d_a_log.numel() == 0) == (kw["a_log"] is None), "d_a_log must be empty iff a_log is absent"
    assert (d_dt_bias.numel() == 0) == (kw["dt_bias"] is None), "d_dt_bias must be empty iff dt_bias is absent"
    pairs = enumerate(zip(eager, meta))
    got = [(i, e.dtype, tuple(e.shape), m.dtype, tuple(m.shape)) for i, (e, m) in pairs if e.dtype != m.dtype or tuple(e.shape) != tuple(m.shape)]
    assert not got, f"meta/eager mismatch at output indices {got}"


# ---------------------------------------------------------------------------
# Gate domain (alpha in (0, 1] versus ln(alpha))
# ---------------------------------------------------------------------------


def gate_domain_pair(case, gate_dtype):
    """(log arm, linear arm) of a case: alpha = exp(g) rounded to the gate
    dtype for the linear arm, and ln(alpha) at the same dtype for the log
    arm, so both arms see the same rounded alpha."""
    alpha = case.gates["g"].float().exp().to(gate_dtype)
    g_log = alpha.float().log().to(gate_dtype)
    return case.clone(gates=dict(case.gates, g=g_log)), case.clone(gates=dict(case.gates, g=alpha))


@pytest.mark.parametrize(
    "varlen,gate_dtype",
    [(False, torch.float32), (False, torch.bfloat16), (False, torch.float16), (True, torch.float32)],
    ids=["dense-fp32", "dense-bf16", "dense-fp16", "varlen-fp32"],
)
@pytest.mark.parametrize("variant", VARIANTS)
def test_gate_domain_linear_forward_parity(backend, variant, gate_dtype, varlen):
    """gate_domain="linear" fed alpha matches the log path fed ln(alpha) in o
    and final_state (the two paths round the log2 decay differently)."""
    case = make_case(variant, torch.bfloat16, seq_lens=[96, 32, 127, 1]) if varlen else make_case(variant, torch.bfloat16, T=256)
    log_case, lin_case = gate_domain_pair(case, gate_dtype)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_log, fs_log = run_fwd(backend, log_case, **kw)
    o_lin, fs_lin = run_fwd(backend, lin_case, gate_domain="linear", **kw)
    assert_rms_close("o", o_lin, o_log.double(), 2e-2)
    assert rms_ratio(fs_lin, fs_log) < 2e-2


@pytest.mark.parametrize("K,gate_dtype", [(128, torch.float32), (128, torch.bfloat16), (64, torch.float32)], ids=["128-fp32", "128-bf16", "64-fp32"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_gate_domain_linear_backward(backend, variant, K, gate_dtype):
    """Under gate_domain="linear" dG is the gradient with respect to alpha,
    dG_log / (alpha + 1e-10), in the gate dtype; the other input gradients
    and d_initial_state match the log arm."""
    case = make_case(variant, torch.bfloat16, T=128, K=K)
    log_case, lin_case = gate_domain_pair(case, gate_dtype)
    set_seed(SEED + 9)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)
    state0 = torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05
    names = LEAF_NAMES[variant]
    grads = {}
    for arm, arm_case, arm_kw in (("log", log_case, {}), ("linear", lin_case, dict(gate_domain="linear"))):
        leaves = [t.requires_grad_(True) for t in thd_tensors(arm_case)]
        s0 = state0.detach().clone().requires_grad_(True)
        with waive_unsupported(backend, variant):
            o, _ = pinned_op(backend, variant)(*leaves, *op_tail(case), initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True, **arm_kw)
            got = torch.autograd.grad(o, leaves + [s0], dO)
        grads[arm] = dict(zip(names + ("initial_state",), got))
    dg_lin, dg_log = grads["linear"].pop("g"), grads["log"].pop("g")
    assert dg_lin.dtype == gate_dtype, f"dG is {dg_lin.dtype}, gate is {gate_dtype}"
    alpha = to_thd(lin_case.gates["g"]).double()
    assert rms_ratio(dg_lin, dg_log.double() / (alpha + 1e-10)) < (2e-2 if gate_dtype == torch.float32 else 5e-2), "dG"
    for name in grads["log"]:
        assert rms_ratio(grads["linear"][name], grads["log"][name]) < 2e-2, f"d{name}"


@pytest.mark.parametrize("V", [128, 64])
@pytest.mark.parametrize("variant", SCALAR_BETA_VARIANTS)
@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16], ids=DTYPE_IDS.get)
def test_beta_sigmoid_in_kernel(backend, variant, beta_dtype, V):
    """float32 or io-dtype beta logits with the in-kernel sigmoid match the
    post-activation fp32 path."""
    case = make_case(variant, torch.bfloat16, T=256, V=V)
    set_seed(SEED + 11)
    braw = torch.randn_like(case.gates["beta"].float()).to(beta_dtype)
    raw_case = case.clone(gates=dict(case.gates, beta=braw))
    eff_case = case.clone(gates=dict(case.gates, beta=braw.float().sigmoid()))
    o_raw, fs_raw = run_fwd(backend, raw_case, output_final_state=True, use_beta_sigmoid_in_kernel=True)
    o_eff, fs_eff = run_fwd(backend, eff_case, output_final_state=True)
    assert_rms_close("o", o_raw, o_eff.double(), 2e-2)
    assert rms_ratio(fs_raw, fs_eff) < 2e-2


@pytest.mark.parametrize("V", [128, 64])
@pytest.mark.parametrize(
    "variant,beta_dtype",
    [(variant, torch.bfloat16) for variant in VARIANTS] + [(variant, torch.float32) for variant in SCALAR_BETA_VARIANTS],
    ids=[*(f"{variant}-bf16" for variant in VARIANTS), *(f"{variant}-fp32" for variant in SCALAR_BETA_VARIANTS)],
)
def test_beta_sigmoid_backward(backend, variant, beta_dtype, V):
    """The in-kernel Beta sigmoid returns the gradient wrt the raw logit, so
    dbeta must equal the post-activation path's dbeta times s * (1 - s) at the
    s the forward stores (io-rounded for io-dtype logits, exact for fp32)."""
    if V == 64 and (variant, beta_dtype) != ("gdn", torch.bfloat16):
        pytest.skip("V = 64 runs one representative")
    case = make_case(variant, torch.bfloat16, T=256, V=V)
    set_seed(SEED + 13)
    braw = torch.randn_like(case.gates["beta"].float()).to(beta_dtype)
    s_io = torch.sigmoid(braw.float()).to(case.dtype)

    def dbeta(beta, **kw):
        leaf = to_thd(beta).detach().clone().requires_grad_(True)
        args = thd_tensors(case)
        args[4] = leaf
        with waive_unsupported(backend, variant):
            o, _ = pinned_op(backend, variant)(*args, *op_tail(case), **kw)
            o.sum().backward()
        assert leaf.grad.dtype == beta.dtype
        return leaf.grad.double()

    got = dbeta(braw, use_beta_sigmoid_in_kernel=True)
    s = to_thd(s_io).double()
    ident = dbeta(s_io.to(case.gates["beta"].dtype)) * s * (1 - s)
    scale = ident.abs().max().item()
    assert scale > 1e-3, "dbeta is ~0, the comparison would be vacuous"
    assert (got - ident).abs().max().item() / scale < 2e-2


# ---------------------------------------------------------------------------
# Beta guard (erase-side safeguard)
# ---------------------------------------------------------------------------


def beta_guard_trip_fraction(case):
    """Reference sensor trip/fallback fractions on a case with H == HV == HO
    (no head expansion) and log-space gates (no safe_gate)."""
    kn = F.normalize(case.k.float(), dim=-1).double()
    _, unsafe, fallback = beta_guard_reference(kn, case.gates["beta"].double(), case.gates["g"].double().exp(), case.dtype)
    return unsafe.double().mean().item(), fallback.double().mean().item()


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_fwd(backend, variant):
    """Guard on, parity against the fp64 guarded reference; the sensor must
    fire on this data or the parity is vacuous."""
    case = make_case(variant, torch.bfloat16, T=256)
    trip, _ = beta_guard_trip_fraction(case)
    assert trip > 0.01, f"beta guard sensor never fires on this case (trip={trip:.4f})"
    assert_fwd_parity(backend, case, l2norm=True, beta_guard=True)


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_fwd_mixed_headroom(backend, variant):
    """Tokens with real decay headroom must pass through untouched next to
    tripping tokens (exercises the safe path and the per-token gate recovery
    at chunk rows 0 and interior rows)."""
    case = make_case(variant, torch.bfloat16, T=256)
    g = case.gates["g"].clone()
    g[:, ::2] += math.log(0.5)
    case = case.clone(gates=dict(case.gates, g=g))
    trip, _ = beta_guard_trip_fraction(case)
    assert 0.01 < trip < 0.99, f"want a mixed safe/unsafe population, got trip={trip:.4f}"
    assert_fwd_parity(backend, case, l2norm=True, beta_guard=True)


@pytest.mark.parametrize("seq_lens", [[64, 192], [31, 63, 93, 123]], ids=["two", "ragged"])
@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_fwd_varlen(backend, variant, seq_lens):
    assert_fwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=seq_lens), l2norm=True, beta_guard=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_bwd(backend, variant, dtype):
    assert_bwd_parity(backend, make_case(variant, dtype, T=128), l2norm=True, beta_guard=True)


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_bwd_varlen(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=[31, 63, 93, 123]), l2norm=True, beta_guard=True)


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_bwd_initial_state(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128), l2norm=True, beta_guard=True, use_initial_state=True)


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_recompute_matches_checkpoints(backend, variant):
    """Prefill (checkpoint dump) and recompute apply the same guard; the gradient
    gap between the checkpoint-reuse and recompute backward paths with the
    guard on stays at the scale of the guard-off gap."""
    case = make_case(variant, torch.bfloat16, T=256)
    tensors = case_tensors(case).values()

    def path_gap(beta_guard):
        grads = {}
        dO = None
        for ckpt in (CHUNK[variant], 0):
            leaves = [to_thd(t).detach().clone().requires_grad_(True) for t in tensors]
            kw = dict(use_qk_l2norm_in_kernel=True, checkpoint_every_n_tokens=ckpt * case.n)
            if beta_guard:
                kw["beta_guard"] = True
            with waive_unsupported(backend, variant):
                out = pinned_op(backend, variant)(*leaves, *op_tail(case), **kw)
            o = out[0]
            if dO is None:
                set_seed(SEED + 23)
                dO = torch.randn_like(o)
            grads[ckpt] = torch.autograd.grad([o], leaves, [dO])
        return max(rms_ratio(a, b.float()) for a, b in zip(grads[16], grads[0]))

    assert path_gap(True) <= max(4.0 * path_gap(False), 1.0e-3)


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_with_sigmoid_fwd(backend, variant):
    """Guard on top of the in-kernel sigmoid; io-dtype logits match the
    post-activation io beta path."""
    case = make_case(variant, torch.bfloat16, T=256)
    set_seed(SEED + 17)
    braw = torch.randn_like(case.gates["beta"].float()).to(case.dtype)
    raw_case = case.clone(gates=dict(case.gates, beta=braw))
    eff_case = case.clone(gates=dict(case.gates, beta=torch.sigmoid(braw.float()).to(case.dtype)))
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True, beta_guard=True)
    o_raw, fs_raw = run_fwd(backend, raw_case, use_beta_sigmoid_in_kernel=True, **kw)
    o_eff, fs_eff = run_fwd(backend, eff_case, **kw)
    assert_rms_close("o", o_raw, o_eff.double(), 2e-2)
    assert rms_ratio(fs_raw, fs_eff) < 2e-2


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_with_sigmoid_backward(backend, variant):
    """Under the in-kernel sigmoid, dbeta wrt the logits equals the
    post-activation path's dbeta times s*(1-s) at the io-rounded s."""
    case = make_case(variant, torch.bfloat16, T=256)
    set_seed(SEED + 19)
    braw = torch.randn_like(case.gates["beta"].float()).to(case.dtype)
    s_io = torch.sigmoid(braw.float()).to(case.dtype)

    def dbeta(beta, **kw):
        leaf = to_thd(beta).detach().clone().requires_grad_(True)
        args = thd_tensors(case)
        args[4] = leaf
        with waive_unsupported(backend, variant):
            o, _ = pinned_op(backend, variant)(*args, *op_tail(case), use_qk_l2norm_in_kernel=True, beta_guard=True, **kw)
            o.sum().backward()
        return leaf.grad.double()

    got = dbeta(braw, use_beta_sigmoid_in_kernel=True)
    s = to_thd(s_io).double()
    ident = dbeta(s_io) * s * (1 - s)
    scale = ident.abs().max().item()
    assert scale > 1e-3, "dbeta is ~0, the comparison would be vacuous"
    assert (got - ident).abs().max().item() / scale < 2e-2


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_multi_tile(backend, variant):
    """B*H well above the SM count with the guard on, several (b, h) tiles per
    CTA."""
    case = make_case(variant, torch.bfloat16, B=8, T=192, H=64)
    assert_fwd_parity(backend, case, l2norm=True, beta_guard=True)


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_bwd_determinism(backend, variant):
    """Guard-on backward is bitwise repeatable."""
    case = make_case(variant, torch.bfloat16, seq_lens=[96, 32, 160, 1])
    tensors = case_tensors(case).values()
    dO = None
    baseline = None
    for _ in range(4):
        leaves = [to_thd(t).detach().clone().requires_grad_(True) for t in tensors]
        with waive_unsupported(backend, variant):
            o, _ = pinned_op(backend, variant)(*leaves, *op_tail(case), use_qk_l2norm_in_kernel=True, beta_guard=True)
        if dO is None:
            set_seed(SEED + 29)
            dO = torch.randn_like(o)
        grads = torch.autograd.grad([o], leaves, [dO])
        if baseline is None:
            baseline = grads
        else:
            for name, a, b in zip(LEAF_NAMES[variant], baseline, grads):
                assert torch.equal(a, b), f"d{name} not bitwise repeatable under beta_guard"


@pytest.mark.parametrize("variant", BETA_GUARD_VARIANTS)
def test_beta_guard_requires_l2norm(backend, variant):
    """The engine must decline beta_guard without the in-kernel l2 norm (and serve it with the norm)."""
    case = make_case(variant, torch.bfloat16, T=64)
    with waive_unsupported(backend, variant):
        pinned_op(backend, variant)(*op_args(case), beta_guard=True, use_qk_l2norm_in_kernel=True)
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        pinned_op(backend, variant)(*op_args(case), beta_guard=True)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("H", (40, 160))
@pytest.mark.parametrize("variant", SCALAR_GATE_VARIANTS)
def test_scalar_gate_head_tiling(backend, variant, H):
    """The scalar gate-parameter reduction tiles heads; the dA_log / ddt_bias
    identities must hold past one tile and past 128 heads."""
    case, kw, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=128, H=H), "safe", params="random", seed=SEED + 15)
    a_leaf, dt_leaf = kw["a_log"].requires_grad_(True), kw["dt_bias"].requires_grad_(True)
    g_leaf = to_thd(case.gates["g"]).detach().clone().requires_grad_(True)
    args = thd_tensors(case)
    args[3] = g_leaf
    with waive_unsupported(backend, variant):
        o, _ = pinned_op(backend, variant)(*args, *op_tail(case), use_qk_l2norm_in_kernel=True, **kw)
        o.sum().backward()
    dg_raw = g_leaf.grad.double()
    y = g_leaf.detach().double() + dt_leaf.detach().double()[None]
    for name, got, ident in (
        ("d_dt_bias", dt_leaf.grad.double(), dg_raw.sum(0)),
        ("d_a_log", a_leaf.grad.double(), (dg_raw * (F.softplus(y) / torch.sigmoid(y))).sum(0)),
    ):
        scale = max(ident.abs().max().item(), 1e-6)
        assert (got - ident).abs().max().item() / scale < 1e-4, name


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_torch_compile_forward(backend, variant):
    case = make_case(variant, torch.bfloat16, T=128)
    with waive_unsupported(backend, variant):
        o_eager, _ = pinned_op(backend, variant)(*op_args(case))
        compiled = torch.compile(pinned_op(backend, variant), fullgraph=True)
        o_comp, _ = compiled(*op_args(case))
    torch.testing.assert_close(o_eager, o_comp)


# ---------------------------------------------------------------------------
# Argument validation (op-level contract; raises before engine selection)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_invalid_rank_raises(backend, variant):
    case = make_case(variant, torch.bfloat16, T=64)
    args = op_args(case)
    args[0] = case.q
    with pytest.raises(ValueError, match="THD"):
        op(variant)(*args)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_invalid_qk_head_mismatch_raises(backend, variant):
    case = make_case(variant, torch.bfloat16, T=64, H=2)
    args = thd_tensors(case)
    args[1] = args[1][:, :1].contiguous()
    with pytest.raises(ValueError, match="head count"):
        op(variant)(*args, *op_tail(case))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_invalid_gate_dtype_raises(backend, variant):
    """fp64 is rejected everywhere; fp32/bf16/fp16 gates are accepted everywhere."""
    case = make_case(variant, torch.bfloat16, T=64)

    def call(gate_dtype):
        args = thd_tensors(case)
        args[3] = args[3].to(gate_dtype)
        return op(variant)(*args, *op_tail(case))

    with pytest.raises((TypeError, KeyError)):
        call(torch.float64)
    for gate_dtype in (torch.float32, torch.bfloat16, torch.float16):
        with waive_declined(f"{variant} with a {gate_dtype} gate"):
            assert call(gate_dtype)[0].dtype == case.dtype


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_invalid_initial_state_count_raises(backend, variant):
    case = make_case(variant, torch.bfloat16, T=64)
    state0 = torch.zeros(3, case.HO, case.V, case.K, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="initial"):
        op(variant)(*op_args(case), initial_state=state0)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_invalid_safe_gate_args_raise(backend, variant):
    """safe_gate alone is valid (unit amplitude, zero bias); a gate parameter
    without safe_gate is not."""
    case = make_case(variant, torch.bfloat16, T=64)
    dt_shape = (case.HO,) if variant in SCALAR_GATE_VARIANTS else (case.HO, case.K)
    with pytest.raises(ValueError, match="require safe_gate=True"):
        op(variant)(*op_args(case), a_log=torch.zeros(case.HO, device="cuda"))
    with pytest.raises(ValueError, match="require safe_gate=True"):
        op(variant)(*op_args(case), dt_bias=torch.zeros(dt_shape, device="cuda"))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_gate_domain_invalid_args_raise(backend, variant):
    """gate_domain is "log" or "linear", and "linear" does not combine with
    safe_gate (the safe-gate transform takes raw logits)."""
    case = make_case(variant, torch.bfloat16, T=64)
    with pytest.raises(ValueError, match="gate_domain"):
        op(variant)(*op_args(case), gate_domain="foo")
    with pytest.raises(ValueError, match="gate_domain"):
        op(variant)(*op_args(case), gate_domain="linear", safe_gate=True)


# ---------------------------------------------------------------------------
# Determinism (contract held by the FROST backend)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_fwd(backend, variant):
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    state0 = torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05

    def launch():
        o, fs = run_fwd(backend, case, initial_state=state0, output_final_state=True)
        return o, fs

    runs = [launch() for _ in range(DETERMINISM_REPEATS)]
    torch.cuda.synchronize()
    for out in runs[0]:
        assert torch.isfinite(out.float()).all(), f"{variant} fwd: non-finite output in run 0"
    for r, outs in enumerate(runs[1:], start=1):
        for i, (a, b) in enumerate(zip(runs[0], outs)):
            assert_bitwise(f"{variant} fwd output {i} run {r}", b, a)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_bwd(backend, variant):
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    args = thd_tensors(case)
    leaves = [args[0].requires_grad_(True), args[1].requires_grad_(True)]
    with waive_unsupported(backend, variant):
        o, fs = pinned_op(backend, variant)(*args, *op_tail(case))
        dO = torch.randn_like(o)

        def launch():
            return torch.autograd.grad([o], leaves, [dO], retain_graph=True)

        runs = [launch() for _ in range(DETERMINISM_REPEATS)]
        torch.cuda.synchronize()
        for out in runs[0]:
            assert torch.isfinite(out.float()).all(), f"{variant} bwd: non-finite output in run 0"
        for r, outs in enumerate(runs[1:], start=1):
            for i, (a, b) in enumerate(zip(runs[0], outs)):
                assert_bitwise(f"{variant} bwd output {i} run {r}", b, a)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_multi_tile_fwd(backend, variant):
    """Multi-tile grid (B*H >> SM count) with an initial state is bitwise
    repeatable."""
    case = make_case(variant, torch.bfloat16, B=8, T=192, H=64)
    state0 = torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.05

    def launch():
        o, fs = run_fwd(backend, case, initial_state=state0, output_final_state=True)
        return o, fs

    runs = [launch() for _ in range(DETERMINISM_REPEATS)]
    torch.cuda.synchronize()
    for out in runs[0]:
        assert torch.isfinite(out.float()).all(), f"{variant} multi-tile fwd: non-finite output in run 0"
    for r, outs in enumerate(runs[1:], start=1):
        for i, (a, b) in enumerate(zip(runs[0], outs)):
            assert_bitwise(f"{variant} multi-tile fwd output {i} run {r}", b, a)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_multi_tile_bwd(backend, variant):
    case = make_case(variant, torch.bfloat16, B=8, T=192, H=64)
    args = thd_tensors(case)
    leaves = [args[0].requires_grad_(True), args[1].requires_grad_(True)]
    with waive_unsupported(backend, variant):
        o, fs = pinned_op(backend, variant)(*args, *op_tail(case))
        dO = torch.randn_like(o)

        def launch():
            return torch.autograd.grad([o], leaves, [dO], retain_graph=True)

        runs = [launch() for _ in range(DETERMINISM_REPEATS)]
        torch.cuda.synchronize()
        for out in runs[0]:
            assert torch.isfinite(out.float()).all(), f"{variant} multi-tile bwd: non-finite output in run 0"
        for r, outs in enumerate(runs[1:], start=1):
            for i, (a, b) in enumerate(zip(runs[0], outs)):
                assert_bitwise(f"{variant} multi-tile bwd output {i} run {r}", b, a)


# ---------------------------------------------------------------------------
# Stress: repeated execution, and several live graphs, in one process
# ---------------------------------------------------------------------------


def assert_stress_stable(backend, variant, shapes):
    """Round robin over ``shapes`` for STRESS_REPEATS rounds; every shape must
    reproduce its first round bitwise."""
    ref = {}
    for rnd in range(STRESS_REPEATS):
        for si, shape in enumerate(shapes):
            got = stress_round(backend, variant, shape)
            if si not in ref:
                ref[si] = [t.detach().clone() for t in got]
                continue
            for name, want, have in zip(("o", "final_state", "dq", "dk"), ref[si], got):
                assert torch.equal(want, have), f"shape {si} round {rnd}: {name} drifted between executions"


def stress_round(backend, variant, shape):
    """One fwd+bwd on freshly allocated operands; returns the outputs."""
    case = make_case(variant, torch.bfloat16, **shape)
    args = thd_tensors(case)
    leaves = [args[0].requires_grad_(True), args[1].requires_grad_(True)]
    o, fs = pinned_op(backend, variant)(*args, *op_tail(case), output_final_state=True)
    # ones rather than randn so the grad seed does not depend on global RNG ordering
    grads = torch.autograd.grad([o, fs], leaves, [torch.ones_like(o), torch.ones_like(fs)])
    torch.cuda.synchronize()
    return [o, fs, *grads]


@pytest.mark.parametrize("variant", VARIANTS)
def test_replay_stress(backend, variant):
    """Re-execute one cached plan many times in a single process on freshly
    allocated operands each round."""
    with waive_unsupported(backend, variant):
        assert_stress_stable(backend, variant, STRESS_SHAPES[:1])


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_multi_graph_stress(backend, variant):
    """Several distinct LA graphs live at once, executed round robin; every
    shape reproduces its first round bitwise."""
    with waive_unsupported(backend, variant):
        assert_stress_stable(backend, variant, STRESS_SHAPES)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_two_streams(backend, variant):
    """Two concurrent instances on separate streams; every repeat matches its
    own single-stream baseline bitwise."""
    case_a = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253], seed=SEED)
    case_b = make_case(variant, torch.bfloat16, B=2, T=512, seed=SEED + 1)
    launch_a = lambda: run_fwd(backend, case_a, output_final_state=True)  # noqa: E731
    launch_b = lambda: run_fwd(backend, case_b, output_final_state=True)  # noqa: E731
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    # order the side streams behind the input generation (default stream)
    torch.cuda.synchronize()

    with torch.cuda.stream(s1):
        base_a = launch_a()
    torch.cuda.synchronize()
    with torch.cuda.stream(s2):
        base_b = launch_b()
    torch.cuda.synchronize()
    for r in range(DETERMINISM_REPEATS):
        with torch.cuda.stream(s1):
            out_a = launch_a()
        with torch.cuda.stream(s2):
            out_b = launch_b()
        torch.cuda.synchronize()
        for label, base, outs in (("A", base_a, out_a), ("B", base_b, out_b)):
            for i, (x, y) in enumerate(zip(base, outs)):
                assert torch.equal(bits(x), bits(y)), f"stream {label} output {i} differs on concurrent run {r}"


# ---------------------------------------------------------------------------
# Batch invariance (whole-sequence work items; packed == solo, bitwise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_batch_invariance_fwd(backend, variant):
    """Under batch_invariant=True each packed sequence matches its solo B = 1 run bitwise."""
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    o, fs = run_fwd(backend, case, output_final_state=True, batch_invariant=True)
    bounds = case.cu.tolist()
    for n in range(case.N):
        s, e = bounds[n], bounds[n + 1]
        if s == e:
            continue
        with waive_unsupported(backend, variant):
            o_solo, fs_solo = pinned_op(backend, variant)(*op_args(case, window=(s, e)), output_final_state=True, batch_invariant=True)
        assert torch.equal(bits(o[s:e]), bits(o_solo)), f"seq {n}: packed o differs from solo"
        assert torch.equal(bits(fs[n]), bits(fs_solo[0])), f"seq {n}: packed final state differs from solo"


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_batch_invariance_bwd(backend, variant):
    """Under batch_invariant=True a sequence's grads match its solo B = 1 run bitwise."""
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    args = thd_tensors(case)
    leaves = [args[0].requires_grad_(True), args[1].requires_grad_(True)]
    s, e = 0, 497
    with waive_unsupported(backend, variant):
        o, fs = pinned_op(backend, variant)(*args, *op_tail(case), batch_invariant=True)
        dO = torch.randn_like(o)
        grads_packed = torch.autograd.grad([o], leaves, [dO], retain_graph=True)
        solo_args = op_args(case, window=(s, e))
        solo_leaves = [solo_args[0].requires_grad_(True), solo_args[1].requires_grad_(True)]
        o_solo, fs_solo = pinned_op(backend, variant)(*solo_args, batch_invariant=True)
        grads_solo = torch.autograd.grad([o_solo], solo_leaves, [dO[s:e].clone()])
    for name, gp, gs in zip(("q", "k"), grads_packed, grads_solo):
        scale = case.n if name in EXPANDED_LEAVES else 1
        rows = slice(s * scale, e * scale)
        assert torch.equal(bits(gp[rows]), bits(gs)), f"packed d{name} slice differs from solo"


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("K", [64, 128])
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_checkpoint_reuse(backend, variant, K):
    """With chunk-cadence checkpoints (the chunk in kernel rows: for a sub-token family that is
    64 expanded rows, not a real-token boundary) the backward consumes the forward's series
    instead of recomputing it; the grads match the recompute path bitwise."""
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253], K=K)
    grads_by_mode = []
    for ckpt in (0, CHUNK[variant]):
        args = thd_tensors(case)
        leaves = [args[0].requires_grad_(True), args[1].requires_grad_(True)]
        with waive_unsupported(backend, variant):
            out = pinned_op(backend, variant)(*args, *op_tail(case), checkpoint_every_n_tokens=ckpt)
            o = out[0]
            set_seed(SEED + 5)
            dO = torch.randn_like(o)
            grads_by_mode.append(torch.autograd.grad([o], leaves, [dO]))
    for gr, gc in zip(grads_by_mode[0], grads_by_mode[1]):
        assert torch.equal(bits(gr), bits(gc)), "checkpoint-reuse grads differ from the recompute path"


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("K,ckpt_mult", [(128, 2), (128, 3), (64, 2)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_coarse_checkpoint_seeded_recompute(backend, variant, ckpt_mult, K):
    """With a coarse checkpoint cadence the backward reconstructs the dense
    series by checkpoint-seeded recompute; the grads track the zero-seed
    recompute path within 2e-3, both arms batch-invariant."""
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253], K=K)
    grads_by_mode = []
    for ckpt in (0, CHUNK[variant] * ckpt_mult):
        args = thd_tensors(case)
        leaves = [args[0].requires_grad_(True), args[1].requires_grad_(True)]
        with waive_unsupported(backend, variant):
            out = pinned_op(backend, variant)(*args, *op_tail(case), checkpoint_every_n_tokens=ckpt * case.n, batch_invariant=True)
            o = out[0]
            set_seed(SEED + 5)
            dO = torch.randn_like(o)
            grads_by_mode.append(torch.autograd.grad([o], leaves, [dO]))
    for name, gr, gc in zip(("dq", "dk"), grads_by_mode[0], grads_by_mode[1]):
        assert torch.isfinite(gc.float()).all(), f"coarse-checkpoint {name} is not finite"
        assert_rms_close(f"coarse-checkpoint {name}", gc, gr, 2e-3)


@pytest.mark.parametrize("backend", ["cutile"], indirect=True)
@pytest.mark.parametrize("variant", CUTILE_VARIANTS)
def test_batch_invariance_cutile(backend, variant):
    """cuTile is batch-invariant by construction; the flag must hold there too."""
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    o, fs = run_fwd(backend, case, output_final_state=True, batch_invariant=True)
    bounds = case.cu.tolist()
    for n in range(case.N):
        s, e = bounds[n], bounds[n + 1]
        if s == e:
            continue
        with waive_unsupported(backend, variant):
            o_solo, fs_solo = pinned_op(backend, variant)(*op_args(case, window=(s, e)), output_final_state=True, batch_invariant=True)
        assert torch.equal(bits(o[s:e]), bits(o_solo)), f"seq {n}: packed o differs from solo"
        assert torch.equal(bits(fs[n]), bits(fs_solo[0])), f"seq {n}: packed final state differs from solo"


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_batch_invariance_with_coarse_checkpoints(backend, variant):
    """batch_invariant=True composes with a coarser checkpoint cadence."""
    ckpt = CHUNK[variant] * 2
    T = 4 * ckpt
    case = make_case(variant, torch.bfloat16, T=T)
    o, fs, state_checkpoints = run_fwd(backend, case, output_final_state=True, batch_invariant=True, checkpoint_every_n_tokens=ckpt * case.n)
    assert state_checkpoints.shape == (T // ckpt + 1, case.HO, case.V, case.K)
    assert not state_checkpoints[0].any(), "row 0 must be the (zero) incoming state"
    n = ckpt
    with waive_unsupported(backend, variant):
        _, fs_p = pinned_op(backend, variant)(*op_args(case, window=(0, n)), output_final_state=True)
    assert_rms_close("state_checkpoints[1]", state_checkpoints[1], fs_p[0], STATE_TOL[case.dtype])


@pytest.mark.parametrize("variant", VARIANTS)
def test_execute_from_a_thread_with_no_cuda_context(backend, variant):
    """Forward and backward from a fresh thread with no bound CUDA context match
    the warm-thread run bitwise."""
    from cuda.bindings import driver as drv

    case = make_case(variant, torch.bfloat16, T=256)
    args = thd_tensors(case)
    leaves = [args[0].requires_grad_(True), args[1].requires_grad_(True)]
    seen = {}

    set_seed(SEED + 7)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)

    def run_on_cold_thread():
        seen["before"] = int(drv.cuCtxGetCurrent()[1])
        try:
            # batch_invariant skips the split-K table launch that would bind a context first
            o, _ = pinned_op(backend, variant)(*args, *op_tail(case), batch_invariant=True)
            seen["o"] = o.detach().clone()
            seen["grads"] = torch.autograd.grad([o], leaves, [dO])
        except BaseException as exc:  # noqa: BLE001
            seen["exc"] = exc
        seen["after"] = int(drv.cuCtxGetCurrent()[1])

    worker = threading.Thread(target=run_on_cold_thread)
    with waive_unsupported(backend, variant):
        worker.start()
        worker.join()
        if "exc" in seen:
            raise seen["exc"]
    assert seen["before"] == 0, "the worker thread was already bound, so this no longer covers the cold path"
    assert seen["after"] != 0, "execute left the calling thread with no CUDA context"
    # the cold thread must not just survive, it must compute what the warm one does
    with waive_unsupported(backend, variant):
        o_warm, _ = pinned_op(backend, variant)(*args, *op_tail(case), batch_invariant=True)
        grads_warm = torch.autograd.grad([o_warm], leaves, [dO])
    assert_bitwise("o", seen["o"], o_warm)
    for name, cold, warm in zip(("dQ", "dK"), seen["grads"], grads_warm):
        assert_bitwise(name, cold, warm)


# ---------------------------------------------------------------------------
# CUDA-graph replay (contract held by the FROST backend)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_cuda_graph_replay_fwd(backend, variant):
    case = make_case(variant, torch.bfloat16, B=2, T=256)

    def launch():
        return pinned_op(backend, variant)(*op_args(case), output_final_state=True)

    with waive_unsupported(backend, variant):
        eager = launch()
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = launch()
        graph.replay()
        torch.cuda.synchronize()
    for i, (a, b) in enumerate(zip(eager, captured)):
        assert torch.equal(bits(a), bits(b)), f"replayed output {i} differs from eager"


# ---------------------------------------------------------------------------
# Hang regression (mbarrier parity-aperture class)
# ---------------------------------------------------------------------------

HANG_STRESS_ITERS = int(os.environ.get("CUDNN_LA_HANG_STRESS_ITERS", "400"))
HANG_STRESS_TIMEOUT = float(os.environ.get("CUDNN_LA_HANG_STRESS_TIMEOUT", "120"))
HANG_STRESS_COMPILE_TIMEOUT = float(os.environ.get("CUDNN_LA_HANG_STRESS_COMPILE_TIMEOUT", "900"))


@contextlib.contextmanager
def wedge_watchdog(label, heartbeat, capfd=None, timeout=None):
    timeout = HANG_STRESS_TIMEOUT if timeout is None else timeout
    done = threading.Event()

    def watch():
        while not done.wait(5.0):
            if time.monotonic() - heartbeat[0] > timeout:
                message = f"\nHANG: {label} made no progress for {timeout:.0f}s; " "GPU kernel wedge (mbarrier parity-aperture class); aborting process"
                with contextlib.suppress(Exception):
                    with capfd.disabled() if capfd is not None else contextlib.nullcontext():
                        print(message, flush=True)
                os._exit(70)

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()


def run_hang_stress(backend, case, *, use_initial_state=False, fwd_each_iter=False, iters=None, label="", capfd=None):
    """Repeated backward over a retained graph of a boundary-dense case under a
    watchdog; the previous grads are NaN-filled and freed before the next
    launch, and the last iteration must be finite and bitwise the first."""
    iters = HANG_STRESS_ITERS if iters is None else iters

    def build_inputs(c):
        leaves = [t.requires_grad_(True) for t in thd_tensors(c)]
        state0 = None
        if use_initial_state:
            state0 = (torch.randn(c.N, c.HO, c.V, c.K, device="cuda", dtype=torch.float32) * 0.05).requires_grad_(True)
        return leaves, state0

    heartbeat = [time.monotonic()]
    with waive_unsupported(backend, case.variant):
        fn = pinned_op(backend, case.variant)
        # compile on a single-wave case (one tile per CTA cannot wedge) so the real case runs under the tight watchdog
        tiny = make_case(case.variant, case.dtype, T=64, H=16)
        tiny_leaves, tiny_state0 = build_inputs(tiny)
        with wedge_watchdog(f"{label} (compile warmup)", heartbeat, capfd=capfd, timeout=HANG_STRESS_COMPILE_TIMEOUT):
            o, fs = fn(*tiny_leaves, *op_tail(tiny), initial_state=tiny_state0, output_final_state=True)
            torch.autograd.grad([o], tiny_leaves + ([tiny_state0] if use_initial_state else []), [torch.randn_like(o)])
            torch.cuda.synchronize()
        del o, fs, tiny_leaves, tiny_state0
        leaves, state0 = build_inputs(case)
        grad_inputs = leaves + ([state0] if use_initial_state else [])
        heartbeat[0] = time.monotonic()
        with wedge_watchdog(label, heartbeat, capfd=capfd):
            o, fs = fn(*leaves, *op_tail(case), initial_state=state0, output_final_state=True)
            dO = torch.randn_like(o)
            grads = torch.autograd.grad([o], grad_inputs, [dO], retain_graph=True)
            torch.cuda.synchronize()
            want = [g.detach().clone() for g in grads]
            heartbeat[0] = time.monotonic()
            for _ in range(iters):
                for g in grads:
                    g.fill_(float("nan"))
                del grads
                if fwd_each_iter:
                    o, fs = fn(*leaves, *op_tail(case), initial_state=state0, output_final_state=True)
                grads = torch.autograd.grad([o], grad_inputs, [dO], retain_graph=not fwd_each_iter)
                torch.cuda.synchronize()
                heartbeat[0] = time.monotonic()
        assert torch.isfinite(o.float()).all(), "non-finite forward output after stress"
        names = list(LEAF_NAMES[case.variant]) + (["initial_state"] if use_initial_state else [])
        for name, g, w in zip(names, grads, want):
            assert torch.isfinite(g.float()).all(), f"non-finite d{name} after stress"
            assert torch.equal(g, w), f"d{name} drifted across stress iterations (same inputs, same dO)"


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_hang_stress_tile_boundary_pipeline(backend, variant, capfd):
    """Hang stress with many short tiles per CTA."""
    case = make_case(variant, torch.bfloat16, seq_lens=[128] * 148, H=16)
    run_hang_stress(backend, case, label=f"tile_boundary_pipeline[{variant}]", capfd=capfd)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_hang_stress_zero_length_tiles(backend, variant, capfd):
    """Hang stress with chunked tiles alternating with zero-length work items."""
    case = make_case(variant, torch.bfloat16, seq_lens=[64, 0] * 96, H=16)
    run_hang_stress(backend, case, label=f"zero_length_tiles[{variant}]", capfd=capfd)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_hang_stress_initial_state_boundaries(backend, variant, capfd):
    """Hang stress with initial-state builds over boundary-dense varlen, forward
    each iteration."""
    case = make_case(variant, torch.bfloat16, seq_lens=[64, 0, 128, 0, 64] * 24, H=16)
    run_hang_stress(backend, case, use_initial_state=True, fwd_each_iter=True, label=f"initial_state_boundaries[{variant}]", capfd=capfd)


# ---------------------------------------------------------------------------
# Sub-token expansion (num_householder)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", (1, 2, 3))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", HOUSEHOLDER_VARIANTS)
def test_fwd_householder_count(backend, variant, dtype, n):
    """Every sub-token count, n = 1 included, matches the reference."""
    assert_fwd_parity(backend, make_case(variant, dtype, T=256, n=n))


@pytest.mark.parametrize("l2norm", [False, True], ids=["plain", "l2norm"])
@pytest.mark.parametrize("n", (2, 3, 4))
@pytest.mark.parametrize("V", [128, 64])
@pytest.mark.parametrize("variant", HOUSEHOLDER_VARIANTS)
def test_bwd_householder_count(backend, variant, V, n, l2norm):
    """n = 4 divides the 64-row chunk, so the final-slot run lands at a different phase than n = 3."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128, V=V, n=n), l2norm=l2norm)


@pytest.mark.parametrize("n", (2, 3))
@pytest.mark.parametrize("V", [128, 64])
@pytest.mark.parametrize("variant", HOUSEHOLDER_VARIANTS)
def test_fwd_householder_varlen_l2norm_state(backend, variant, V, n):
    """Ragged sequences with the in-kernel l2 norm and an initial state at two sub-token counts."""
    assert_fwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=[100, 156], V=V, n=n), l2norm=True, use_initial_state=True)


@pytest.mark.parametrize("variant", HOUSEHOLDER_VARIANTS)
def test_bwd_householder_ragged_l2norm(backend, variant):
    """Ragged final chunks on both timelines drive the fused-l2norm inv_q / inv_k tails."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=[68, 100], V=64), l2norm=True)


@pytest.mark.L0
@pytest.mark.parametrize("l2norm", [False, True], ids=["plain", "l2norm"])
@pytest.mark.parametrize("n", (2, 3, 4))
@pytest.mark.parametrize("V", [128, 64])
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_gdp_summary_bwd_compact_qdo_matches_port(backend, V, n, l2norm):
    """The GDP bprop summary reads q and dO on the compact token timeline (one 64-token tile per block of n
    chunks, the tile rows masked to the tokens whose readout sub-token lies in the chunk) and matches the
    expanded-timeline port's d_initial_state at bf16-ulp scale: ragged sequences with partial last blocks, with
    and without a d_final_state seed."""
    case = make_case("gdp", torch.bfloat16, seq_lens=[100, 333, 64], V=V, n=n)
    set_seed(SEED + 16)
    d_o = to_thd(torch.randn(case.B, case.T, case.HO, case.V, device="cuda", dtype=torch.float32) * 0.1).to(case.v.dtype)
    d0 = torch.randn(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32) * 0.1
    kw = dict(use_qk_l2norm_in_kernel=l2norm)
    for d_final in (None, d0):
        g_kernel = run_summary_bwd(case, d_o, d_final_state=d_final, **kw)
        _, g_port = grads_via_port(case, d_o, d_final, **kw)
        seeded = "seeded" if d_final is not None else "unseeded"
        assert rms_ratio(g_kernel, g_port.double()) < 1e-4, f"compact q/dO summary vs the expanded port ({seeded})"


@pytest.mark.L0
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("l2norm", [False, True], ids=["plain", "l2norm"])
@pytest.mark.parametrize("n", (2, 3))
def test_gdp_piece_chain_bwd_compact_summary(backend, l2norm, n):
    """The GDP backward chain at d_v = 128 (the main bprop on the expanded pack, its G summary on compact q / dO,
    an expanded l2norm q read through the strided descriptor) matches the batch run whole."""
    case = make_case("gdp", torch.bfloat16, seq_lens=[2048, 1000], H=2, n=n, lo=0.99)
    state0, d_final = random_state(case), random_state(case, scale=0.1, seed=SEED + 3)
    kw = dict(initial_state=state0, d_final_state=d_final, use_qk_l2norm_in_kernel=l2norm)
    _, _, grads, dO, _ = chain_grads(backend, case, **kw)
    _, _, grads_uncut, _, _ = chain_grads(backend, case, batch_invariant=True, dO=dO, **kw)
    assert_chain_grads_close(grads, grads_uncut)


@pytest.mark.parametrize("variant", HOUSEHOLDER_VARIANTS)
def test_checkpoints_count_expanded_rows(backend, variant):
    """The checkpoint cadence counts expanded rows: the chunk cadence and n times it (every checkpoint on a
    real-token boundary) both size the series."""
    T = 256
    case = make_case(variant, torch.bfloat16, T=T)
    rows = CHUNK[variant]
    out = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=rows)
    assert out[2].shape[0] == T * case.n // rows + case.N
    out = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=rows * case.n)
    assert out[2].shape[0] == T // rows + case.N


@pytest.mark.parametrize("variant", HOUSEHOLDER_VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_invalid_householder_rows_raise(backend, variant):
    """k on the real-token timeline, or a non-positive count, is rejected at the op."""
    case = make_case(variant, torch.bfloat16, T=64)
    args = op_args(case)
    args[1] = args[1][: case.T]
    with pytest.raises(ValueError, match="rows"):
        op(variant)(*args)
    args = op_args(case)
    args[-1] = 0
    with pytest.raises(ValueError, match="positive"):
        op(variant)(*args)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", [variant for variant in VARIANTS if variant != "gdn"])
def test_fwd_reduces_to_gdn(backend, variant):
    """With its extra freedom switched off every family is GDN: one sub-token (bitwise the gdn op), or
    channel-constant gates with a write gate equal to beta (against the gdn reference)."""
    gdn = make_case("gdn", torch.bfloat16, T=256)
    if variant in HOUSEHOLDER_VARIANTS:
        o, fs = run_fwd(backend, gdn.clone(variant=variant, n=1), output_final_state=True)
        o_gdn, fs_gdn = run_fwd(backend, gdn, output_final_state=True)
        assert_bitwise("o", o, o_gdn)
        assert_bitwise("final_state", fs, fs_gdn)
        return
    gates = dict(gdn.gates, g=gdn.gates["g"][..., None].expand(-1, -1, -1, gdn.K).contiguous())
    if variant == "gdn2":
        beta = gdn.gates["beta"].to(gdn.dtype)
        gates["beta"] = beta[..., None].expand(-1, -1, -1, gdn.K).contiguous()
        gates["w"] = beta[..., None].expand(-1, -1, -1, gdn.V).contiguous()
        gdn = gdn.clone(gates=dict(gdn.gates, beta=beta.float()))
    o, fs = run_fwd(backend, gdn.clone(variant=variant, gates=gates), output_final_state=True)
    o_ref, fs_ref = reference(gdn)
    assert_rms_close("o", o, o_ref, FWD_TOL[gdn.dtype])
    assert_rms_close("final_state", fs, fs_ref, STATE_TOL[gdn.dtype])


# ---------------------------------------------------------------------------
# Summaries for context parallelism (H, M, G)
# ---------------------------------------------------------------------------
# V-major state buffers hold S^T, so X_final = X_init @ M_buf + X_H and X_dh0 = X_dht @ M_buf^T + X_G

GATE_MODES = ("log", "safe", "sigmoid_beta", "sigmoid_beta_fp32", "fp16")
PRODUCERS = ("prefill", "summary")

SUMMARY_OPS = {
    "gdn": "gated_delta_net_summary",
    "kda": "kimi_delta_attention_summary",
    "gdn2": "gated_delta_net_v2_summary",
    "gdp": "gated_delta_product_summary",
}

SUMMARY_BWD_OPS = {
    "gdn": "gated_delta_net_summary_bwd",
    "kda": "kimi_delta_attention_summary_bwd",
    "gdn2": "gated_delta_net_v2_summary_bwd",
    "gdp": "gated_delta_product_summary_bwd",
}

M_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
H_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
AFFINE_KAPPA = 3.0
SUPERPOSE_TOL = 4e-2
G_TOL = 6e-2
NORM_EPS = 1e-8
KDA_GATE_LOWER_BOUND = -5.0


def run_frost(case, **kw):
    """The family op on a case, pinned to the FROST engine; a decline waives the test."""
    kw.setdefault("plan_name", f"{case.variant}_frost")
    try:
        return op(case.variant)(*op_args(case), **kw)
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"{case.variant} frost engine declined: {e}")


def m_twin(case):
    """The M producer twin of a case: v = 0 with V == K (square state) and, for gdn2, a unit write gate."""
    v_zero = torch.zeros(case.B, case.T * case.n, case.HV, case.K, device="cuda", dtype=case.v.dtype)
    gates = dict(case.gates)
    if "w" in gates:
        gates["w"] = torch.ones(case.B, case.T, case.HO, case.K, device="cuda", dtype=gates["w"].dtype)
    return case.clone(v=v_zero, V=case.K, gates=gates)


def run_summary(case, *, initial_state=None, **kw):
    """(H, M) through the summary op, pinned to the FROST summary engine."""
    kw.setdefault("plan_name", f"{case.variant}_summary_frost")
    fn = getattr(la_ops, SUMMARY_OPS[case.variant])
    try:
        return fn(*op_args(case)[1:], initial_state=initial_state, **kw)
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"{case.variant} frost summary engine declined: {e}")


def kernel_m(case, producer="prefill", **kw):
    """M_buf through a producer, either the prefill identity trick or the summary op."""
    if producer == "summary":
        return run_summary(case, **kw)[1]
    return run_frost(
        m_twin(case),
        initial_state=torch.eye(case.K, device="cuda", dtype=torch.float32).expand(case.N, case.HO, case.K, case.K).contiguous(),
        output_final_state=True,
        **kw,
    )[1]


def token_transitions(variant, k, g, beta, *, safe_gate=False, a_log=None, dt_bias=None, use_beta_sigmoid=False):
    """Per-token transitions ``A_t`` as an fp64 batch [tokens, HO, K, K] (math
    domain, left-acting), using the same gate transforms as the references."""
    k = k.double()
    g = g.double()
    beta = beta.double()
    if use_beta_sigmoid:
        beta = beta.sigmoid()
    if safe_gate:
        bias = dt_bias.double() if dt_bias is not None else 0.0
        if variant in SCALAR_GATE_VARIANTS:
            a_exp = a_log.double().exp() if a_log is not None else 1.0
            g = -a_exp * F.softplus(g + bias)
        else:
            a_exp = a_log.double().exp()[:, None] if a_log is not None else 1.0
            g = KDA_GATE_LOWER_BOUND * torch.sigmoid(a_exp * (g + bias))
    alpha = g.exp()
    HO = g.shape[-2] if variant in CHANNEL_VARIANTS else g.shape[-1]
    if k.shape[-2] != HO:
        k = k.repeat_interleave(HO // k.shape[-2], dim=-2)
    dim_k = k.shape[-1]
    eye = torch.eye(dim_k, device=k.device, dtype=torch.float64)
    outer_k = k.unsqueeze(-1) @ k.unsqueeze(-2)
    if variant in SCALAR_GATE_VARIANTS:
        return alpha[..., None, None] * (eye - beta[..., None, None] * outer_k)
    if variant == "kda":
        return (eye - beta[..., None, None] * outer_k) * alpha[..., None, :]
    erase = k.unsqueeze(-1) @ (beta * k).unsqueeze(-2)
    return (eye - erase) * alpha[..., None, :]


def expanded_gate(case):
    """The gate on the expanded timeline: the token's gate on sub-token 0, no decay on the others."""
    g = case.gates["g"]
    if case.n == 1:
        return g
    full = g.new_zeros(case.B, case.T, case.n, *g.shape[2:])
    full[:, :, 0] = g
    return full.reshape(case.B, case.T * case.n, *g.shape[2:])


def oracle_m(case, *, chunk=512, **transforms):
    """fp64 M_buf as the per-sequence ordered product A[L-1] @ ... @ A[0] of token transitions (pairwise tree
    reduction, at most 2048 head-tokens in flight); returned transposed (stored domain)."""
    HO, dim_k = case.HO, case.K
    chunk = max(1, min(chunk, 2048 // HO))
    k_t, g_t, beta_t = to_thd(case.k), to_thd(expanded_gate(case)), to_thd(case.gates["beta"])
    bounds = [b * case.n for b in case.cu.tolist()]
    eye = torch.eye(dim_k, device="cuda", dtype=torch.float64)
    out = []
    for n in range(len(bounds) - 1):
        start, end = bounds[n], bounds[n + 1]
        m_seq = eye.expand(HO, dim_k, dim_k).clone()
        for c0 in range(start, end, chunk):
            c1 = min(c0 + chunk, end)
            a_batch = token_transitions(case.variant, k_t[c0:c1], g_t[c0:c1], beta_t[c0:c1], **transforms)
            while a_batch.shape[0] > 1:
                even = a_batch[: a_batch.shape[0] - a_batch.shape[0] % 2]
                pairs = even[1::2] @ even[0::2]
                a_batch = torch.cat([pairs, a_batch[-1:]], dim=0) if a_batch.shape[0] % 2 else pairs
            m_seq = a_batch[0] @ m_seq
        out.append(m_seq)
    return torch.stack(out, dim=0).transpose(-1, -2).contiguous()


def slice_rel_errors(got, want):
    """Per-(sequence, head) relative Frobenius errors with an underflow guard."""
    got = got.double()
    want = want.double()
    diff = (got - want).flatten(2).norm(dim=-1)
    denom = want.flatten(2).norm(dim=-1).clamp_min(NORM_EPS)
    return diff / denom


def assert_state_close(name, got, want, tol):
    assert torch.isfinite(got.float()).all(), f"non-finite values in {name}"
    errs = slice_rel_errors(got, want)
    worst = errs.max().item()
    print(f"{name}: worst per-slice rel error {worst:.3e} (tol {tol:g})")
    assert worst < tol, f"{name} worst per-slice rel error {worst:.4g} >= {tol}"


def grads_via_port(case, d_o, d_final, *, seed=SEED + 11, **op_kw):
    """(grads dict, ds0) through the public op autograd with a zero initial
    state; ``op_kw`` reach the family op."""
    set_seed(seed)
    s0 = torch.zeros(case.N, case.HO, case.V, case.K, device="cuda", dtype=torch.float32, requires_grad=True)
    leaves = dict(zip(LEAF_NAMES[case.variant], (t.requires_grad_(True) for t in thd_tensors(case))))
    try:
        o, fs = op(case.variant)(
            *leaves.values(),
            *op_tail(case),
            initial_state=s0,
            output_final_state=True,
            plan_name=f"{case.variant}_frost",
            **op_kw,
        )
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"{case.variant} frost engine declined: {e}")
    outputs, grad_outputs = [o], [d_o]
    if d_final is not None:
        outputs.append(fs)
        grad_outputs.append(d_final)
    grads = torch.autograd.grad(outputs, list(leaves.values()) + [s0], grad_outputs)
    named = dict(zip(leaves.keys(), grads[:-1]))
    return named, grads[-1]


def soften(case, beta_scale=0.02):
    """Weaken the erase so M stays measurably nonzero over long spans."""
    gates = dict(case.gates)
    gates["beta"] = (gates["beta"].float() * beta_scale).to(gates["beta"].dtype)
    return case.clone(gates=gates)


# ---------------------------------------------------------------------------
# Oracle self-check
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_product_oracle_matches_reference(backend, variant):
    """The chunked product oracle and the sequential fp64 reference agree to
    fp64 precision (two independent constructions of M)."""
    case = make_case(variant, torch.bfloat16, T=257)
    m_prod = oracle_m(case, chunk=64)
    m_ref = reference(m_twin(case), initial_state=torch.eye(case.K, device="cuda", dtype=torch.float32).expand(case.N, case.HO, case.K, case.K).contiguous())[1]
    err = rms_ratio(m_prod, m_ref)
    assert err < 1e-9, f"fp64 oracles disagree: {err:.4g}"


# ---------------------------------------------------------------------------
# Layout pin
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_layout_pin_one_token(backend, variant):
    """One token, where the buffer holds A^T: it matches the transposed oracle elementwise, and where A is
    asymmetric (a per-channel gate) it does not match A."""
    case = make_case(variant, torch.bfloat16, T=1, H=1, K=64, V=64, seed=SEED + 3)
    m_buf = kernel_m(case)
    m_ref = oracle_m(case)
    err_t = rms_ratio(m_buf[0, 0], m_ref[0, 0])
    assert err_t < 2e-2, f"M buffer does not hold A^T (rel err {err_t:.4g})"
    if variant in CHANNEL_VARIANTS:
        err_plain = rms_ratio(m_buf[0, 0], m_ref[0, 0].t())
        assert err_plain > 5 * err_t, f"layout pin is not discriminating (A err {err_plain:.4g} vs A^T err {err_t:.4g})"


# ---------------------------------------------------------------------------
# M vs the fp64 oracle
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("length", [1, 63, 64, 65, 257, 1024])
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_m_matches_oracle_lengths(backend, variant, length, producer):
    case = make_case(variant, torch.bfloat16, T=length)
    m_buf = kernel_m(case, producer)
    m_ref = oracle_m(case)
    assert_state_close(f"M[{variant}, T={length}, {producer}]", m_buf, m_ref, M_TOL[torch.bfloat16])


@pytest.mark.L0
@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("gate_mode", GATE_MODES)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_m_matches_oracle_gate_modes(backend, variant, gate_mode, producer):
    dtype = torch.float16 if gate_mode == "fp16" else torch.bfloat16
    case, op_kw, oracle_kw = gate_mode_case(make_case(variant, dtype, T=256), gate_mode)
    m_buf = kernel_m(case, producer, **op_kw)
    m_ref = oracle_m(case, **oracle_kw)
    assert_state_close(f"M[{variant}, {gate_mode}, {producer}]", m_buf, m_ref, M_TOL[dtype])


@pytest.mark.L0
@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dim_k", [64, 128])
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_m_matches_oracle_dims(backend, variant, dim_k, producer):
    case = make_case(variant, torch.bfloat16, T=256, K=dim_k, V=dim_k)
    m_buf = kernel_m(case, producer)
    m_ref = oracle_m(case)
    assert_state_close(f"M[{variant}, K={dim_k}, {producer}]", m_buf, m_ref, M_TOL[torch.bfloat16])


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dims", [(64, 128), (128, 64)])
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_m_summary_rectangular_case(backend, variant, dims):
    """The summary op on rectangular (K, V) cases; M is (K, K) while H is (V, K)."""
    dim_k, dim_v = dims
    case = make_case(variant, torch.bfloat16, T=256, K=dim_k, V=dim_v)
    m_buf = kernel_m(case, "summary")
    m_ref = oracle_m(case)
    assert_state_close(f"M[{variant}, {dims}, summary]", m_buf, m_ref, M_TOL[torch.bfloat16])


@pytest.mark.L0
@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_m_multi_sequence_independent(backend, variant, producer):
    """Each sequence's M is independent; a zero-length sequence yields M = I exactly."""
    seq_lens = [31, 0, 93, 150]
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens)
    m_buf = kernel_m(case, producer)
    m_ref = oracle_m(case)

    N, HO, K = case.N, case.HO, case.K
    eye = torch.eye(K, device="cuda", dtype=torch.float32)
    for n, s_len in enumerate(seq_lens):
        if s_len == 0:
            torch.testing.assert_close(m_buf[n], eye.expand(HO, K, K), atol=0.0, rtol=0.0)
        else:
            assert_state_close(f"M[{variant}, seq {n}]", m_buf[n : n + 1], m_ref[n : n + 1], M_TOL[torch.bfloat16])


@pytest.mark.L1
@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("length", [8192, 32768])
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_m_matches_oracle_long(backend, variant, length, producer):
    """Long-span M accuracy (fp32 state carry in every producer path)."""
    case = soften(make_case(variant, torch.bfloat16, T=length, H=1, lo=0.9995))
    m_buf = kernel_m(case, producer)
    m_ref = oracle_m(case)
    assert_state_close(f"M[{variant}, T={length}, {producer}]", m_buf, m_ref, M_TOL[torch.bfloat16])


@pytest.mark.L1
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_forward_affine_contract_long(backend, variant):
    """At long spans the composed final state stays within kappa of the direct
    run's error against the fp64 oracle."""
    case = soften(make_case(variant, torch.bfloat16, T=8192, H=1, lo=0.9995))
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 5)
    s0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.05

    direct = run_frost(case, initial_state=s0, output_final_state=True)[1]
    h_buf, m_buf = run_summary(case)
    composed = s0.double() @ m_buf.double() + h_buf.double()

    oracle = reference(case, initial_state=s0)[1]
    err_direct = max(rms_ratio(direct, oracle), 1e-6)
    err_composed = rms_ratio(composed, oracle)
    assert (
        err_composed <= AFFINE_KAPPA * err_direct
    ), f"composed final state {err_composed:.4g} vs direct {err_direct:.4g} exceeds kappa {AFFINE_KAPPA} at T=8192"


# ---------------------------------------------------------------------------
# Summary producer gates
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dim_k", [64, 128])
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_flags_bitwise_gate(backend, variant, dim_k):
    """The GMEM-identity arm (initial_state = I, v = 0) and the summary op's
    transition output are bitwise equal."""
    case = make_case(variant, torch.bfloat16, T=257, K=dim_k, V=dim_k)
    twin = m_twin(case)
    N, HO, K = case.N, case.HO, case.K
    h_buf, m_buf = run_summary(twin, initial_state=torch.eye(K, device="cuda", dtype=torch.float32).expand(N, HO, K, K).contiguous())
    assert torch.equal(h_buf, m_buf), f"seed_identity/v_is_zero arm diverges from the GMEM-identity arm (max abs diff {(h_buf - m_buf).abs().max().item():.4g})"


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_flags_bitwise_gate_varlen(backend, variant):
    seq_lens = [31, 0, 93, 150]
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens, K=128, V=128)
    twin = m_twin(case)
    N, HO, K = case.N, case.HO, case.K
    h_buf, m_buf = run_summary(twin, initial_state=torch.eye(K, device="cuda", dtype=torch.float32).expand(N, HO, K, K).contiguous())
    assert torch.equal(h_buf, m_buf)


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_h_parity_vs_prefill(backend, variant):
    """The summary H is bitwise the prefill final_state; one (sequence, head)
    tile per SM runs every sequence whole on both sides."""
    case = make_case(variant, torch.bfloat16, T=257, H=sm_count())
    h_prefill = run_frost(case, output_final_state=True)[1]
    h_summary = run_summary(case)[0]
    assert torch.equal(
        h_summary, h_prefill.float()
    ), f"summary H diverges from prefill final_state (max abs diff {(h_summary - h_prefill.float()).abs().max().item():.4g})"


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_initial_state_parity_vs_prefill(backend, variant):
    case = make_case(variant, torch.bfloat16, T=256, H=sm_count())
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 4)
    s0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.05
    final_prefill = run_frost(case, initial_state=s0, output_final_state=True)[1]
    final_summary = run_summary(case, initial_state=s0)[0]
    assert torch.equal(
        final_summary, final_prefill.float()
    ), f"summary final(S0) diverges from prefill (max abs diff {(final_summary - final_prefill.float()).abs().max().item():.4g})"


@pytest.mark.L0
@pytest.mark.parametrize("variant", ("gdn", "gdn2", "gdp"))
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_gate_domain_summary_linear(backend, variant):
    """The summary under gate_domain="linear" fed alpha matches the log
    path fed ln(alpha) in H and M."""
    case = make_case(variant, torch.bfloat16, T=256)
    alpha = case.gates["g"].exp()

    def arm(g):
        return case.clone(gates=dict(case.gates, g=g))

    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 4)
    s0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.05
    h_log, m_log = run_summary(arm(alpha.log()), initial_state=s0)
    h_lin, m_lin = run_summary(arm(alpha), initial_state=s0, gate_domain="linear")
    assert rms_ratio(h_lin, h_log) < 2e-2, "H"
    assert rms_ratio(m_lin, m_log) < 2e-2, "M"


# ---------------------------------------------------------------------------
# Piece chain (long sequences cut into pieces)
# ---------------------------------------------------------------------------

CHAIN_RAGGED = [497, 16, 1, 480, 0, 253, 4096]
CHAIN_SHAPES = [([4096], 4), ([4096], 37), (CHAIN_RAGGED, 4)]
CHAIN_SHAPE_IDS = ["1x4096x4", "1x4096x37", "ragged"]
CHAIN_VS_UNCUT_TOL = 1e-2
CHAIN_GATE_GRAD_TOL = 2e-2
CHAIN_FWD_TOL = 2e-2
CHAIN_BWD_TOL = 4e-2
CHAIN_STATE_GRAD_TOL = 6e-2
PIECE_TOKENS = 8192
CHAIN_GATE_MODES = ("l2norm", "safe", "sigmoid_beta", "neg_eigval", "beta_guard")


def chain_case(variant, seq_lens, *, H=4, HK=None, HV=None, K=128, V=128, cu_dtype=torch.int32, seed=SEED):
    """A bf16 case with alpha in (0.99, 1), so long spans keep a live state; a sub-token family takes ``H // n``
    heads (at least 2) so its fp64 reference over the expanded rows stays within memory."""
    n = HOUSEHOLDER if variant in HOUSEHOLDER_VARIANTS else 1
    if n > 1 and HK is None and HV is None:
        H = max(2, H // n)
    return make_case(variant, torch.bfloat16, seq_lens=seq_lens, H=H, HK=HK, HV=HV, K=K, V=V, lo=0.99, cu_dtype=cu_dtype, seed=seed)


def chain_grads(backend, case, *, batch_invariant=False, checkpoint=0, initial_state=None, d_final_state=None, dO=None, cu=None, params=(), **kw):
    """(o, final_state, grads by leaf name, dO, outputs) of one pinned forward + backward on fresh leaves;
    ``params`` are extra (name, tensor) leaves handed to the op as keywords."""
    leaves = dict(zip(LEAF_NAMES[case.variant], (t.requires_grad_(True) for t in thd_tensors(case))))
    extra = {name: t.detach().clone().requires_grad_(True) for name, t in params}
    state0 = initial_state.detach().clone().requires_grad_(True) if initial_state is not None else None
    with waive_unsupported(backend, case.variant):
        out = pinned_op(backend, case.variant)(
            *leaves.values(),
            *op_tail(case, cu),
            initial_state=state0,
            output_final_state=True,
            batch_invariant=batch_invariant,
            checkpoint_every_n_tokens=checkpoint,
            **extra,
            **kw,
        )
    o, fs = out[0], out[1]
    if dO is None:
        set_seed(SEED + 7)
        dO = torch.randn_like(o)
    outputs, grad_outputs = [o], [dO]
    if d_final_state is not None:
        outputs.append(fs)
        grad_outputs.append(d_final_state)
    inputs = list(leaves.values()) + list(extra.values()) + ([state0] if state0 is not None else [])
    names = list(leaves) + list(extra) + (["initial_state"] if state0 is not None else [])
    grads = dict(zip(names, torch.autograd.grad(outputs, inputs, grad_outputs)))
    return o, fs, grads, dO, out


def assert_chain_grads_close(got, want, label="chain-vs-uncut"):
    for name, g in got.items():
        tol = CHAIN_GATE_GRAD_TOL if name in ("g", "beta", "w", "a_log", "dt_bias") else CHAIN_VS_UNCUT_TOL
        assert_rms_close(f"d{name} {label}", g, want[name].float(), tol)


def chain_reference_grads(case, state0, dO, d_final):
    """fp64 gradients of every leaf and the initial state under ``dO`` and ``d_final``."""
    ref_leaves = {name: t.detach().double().requires_grad_(True) for name, t in case_tensors(case).items()}
    state0_ref = state0.detach().double().requires_grad_(True)
    o_ref, fs_ref = reference_call(case.variant, ref_leaves, case.n, initial_state=state0_ref, cu_seqlens=case.cu)
    grads = torch.autograd.grad(
        [o_ref, fs_ref], list(ref_leaves.values()) + [state0_ref], [dO.double().reshape(o_ref.shape), d_final.double().reshape(fs_ref.shape)]
    )
    return dict(zip(list(ref_leaves) + ["initial_state"], grads))


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("state_dtype", [None, torch.float32, torch.bfloat16], ids=["no_state", "fp32_state", "bf16_state"])
@pytest.mark.parametrize("seq_lens,H", CHAIN_SHAPES, ids=CHAIN_SHAPE_IDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_fwd_matches_uncut_and_reference(backend, variant, seq_lens, H, state_dtype):
    """o and final_state of a long batch on few tiles against the batch run whole and, with an fp32 state, the fp64
    reference; an empty sequence passes its initial state through bitwise."""
    case = chain_case(variant, seq_lens, H=H)
    state0 = random_state(case, dtype=state_dtype) if state_dtype is not None else None
    o_chain, fs_chain = run_fwd(backend, case, initial_state=state0, output_final_state=True)
    o_uncut, fs_uncut = run_fwd(backend, case, initial_state=state0, output_final_state=True, batch_invariant=True)
    assert fs_chain.dtype == fs_uncut.dtype == (state_dtype or torch.float32)
    assert_rms_close("o chain-vs-uncut", o_chain, o_uncut.float(), CHAIN_VS_UNCUT_TOL)
    assert_rms_close("final_state chain-vs-uncut", fs_chain, fs_uncut.float(), CHAIN_VS_UNCUT_TOL)
    if state_dtype is torch.float32:
        o_ref, fs_ref = reference(case, initial_state=state0)
        assert_rms_close("o vs fp64", o_chain, o_ref, CHAIN_FWD_TOL)
        assert_rms_close("final_state vs fp64", fs_chain, fs_ref, CHAIN_FWD_TOL)
    for n, length in enumerate(seq_lens):
        if length == 0:
            want = state0[n] if state0 is not None else torch.zeros_like(fs_chain[n])
            assert_bitwise(f"empty sequence {n} final_state passthrough", fs_chain[n], want)


@pytest.mark.gpu_exclusive
@pytest.mark.xdist_group(name="gpu_exclusive")
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_reverse_band_matches_uncut_and_reference(backend, variant):
    """One sequence on half the SMs; o, final_state and every gradient against the batch run whole and the fp64
    reference."""
    case = chain_case(variant, [4096], H=sm_count() // 2)
    state0, d_final = random_state(case), random_state(case, scale=0.1, seed=SEED + 3)
    o, fs, grads, dO, _ = chain_grads(backend, case, initial_state=state0, d_final_state=d_final)
    o_u, fs_u, grads_u, _, _ = chain_grads(backend, case, batch_invariant=True, initial_state=state0, d_final_state=d_final, dO=dO)
    assert_rms_close("o chain-vs-uncut", o, o_u.float(), CHAIN_VS_UNCUT_TOL)
    assert_rms_close("final_state chain-vs-uncut", fs, fs_u.float(), CHAIN_VS_UNCUT_TOL)
    assert_chain_grads_close(grads, grads_u)
    for name, want in chain_reference_grads(case, state0, dO, d_final).items():
        assert_rms_close(f"d{name} vs fp64", grads[name], want, CHAIN_STATE_GRAD_TOL if name == "initial_state" else CHAIN_BWD_TOL)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_deterministic(backend, variant):
    """Forward, checkpoint series and every gradient of a cut sequence are bitwise repeatable."""
    case = chain_case(variant, [4096], H=4)
    state0 = random_state(case)
    checkpoint = CHUNK[variant]
    rows = sum((length * case.n - 1) // checkpoint + 1 for length in case.cu.diff().tolist() if length > 0)
    first = run_fwd(backend, case, initial_state=state0, output_final_state=True, checkpoint_every_n_tokens=checkpoint)
    for _ in range(2):
        again = run_fwd(backend, case, initial_state=state0, output_final_state=True, checkpoint_every_n_tokens=checkpoint)
        assert_bitwise("o run-to-run", again[0], first[0])
        assert_bitwise("final_state run-to-run", again[1], first[1])
        assert_bitwise("state_checkpoints run-to-run", again[2][:rows], first[2][:rows])
    first_grads = chain_grads(backend, case, initial_state=state0)
    for _ in range(2):
        again_grads = chain_grads(backend, case, initial_state=state0, dO=first_grads[3])
        for name in first_grads[2]:
            assert_bitwise(f"d{name} run-to-run", again_grads[2][name], first_grads[2][name])


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("state_dtype", [None, torch.float32, torch.bfloat16], ids=["no_state", "fp32_state", "bf16_state"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_fwd_batch_invariant_length_rule_bitwise(backend, variant, state_dtype):
    """Under batch_invariant a sequence's o and final_state are bitwise the same alone and inside a long batch; an
    empty sequence passes its initial state through."""
    seq_lens = [20000, 3000, 0, 17000]
    assert max(seq_lens) > 2 * PIECE_TOKENS and min(length for length in seq_lens if length) < PIECE_TOKENS
    case = chain_case(variant, seq_lens, H=2, K=64, V=64)
    state0 = random_state(case, dtype=state_dtype) if state_dtype is not None else None
    o, fs = run_fwd(backend, case, initial_state=state0, output_final_state=True, batch_invariant=True)
    for n, length in enumerate(seq_lens):
        if length == 0:
            if state0 is not None:
                assert_bitwise(f"empty sequence {n} final_state passthrough", fs[n], state0[n])
            continue
        s, e = int(case.cu[n]), int(case.cu[n + 1])
        alone = window(case, s, e)
        alone_state = state0[n : n + 1].clone() if state0 is not None else None
        clear_caches()
        o_alone, fs_alone = run_fwd(backend, alone, initial_state=alone_state, output_final_state=True, batch_invariant=True)
        assert_bitwise(f"seq {n} o", o[s:e], o_alone)
        assert_bitwise(f"seq {n} final_state", fs[n], fs_alone[0])


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("state_dtype", STATE_DTYPES, ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_bwd_batch_invariant_length_rule_bitwise(backend, variant, state_dtype):
    """Every gradient of a sequence, d_initial_state included, is bitwise the same alone and inside a long batch."""
    seq_lens = [17000, 3000]
    case = chain_case(variant, seq_lens, H=2, K=64, V=64)
    state0, d_final = random_state(case, dtype=state_dtype), random_state(case, scale=0.1, seed=SEED + 3, dtype=state_dtype)
    _, _, grads, dO, _ = chain_grads(backend, case, batch_invariant=True, initial_state=state0, d_final_state=d_final)
    for n in range(case.N):
        s, e = int(case.cu[n]), int(case.cu[n + 1])
        alone = window(case, s, e)
        clear_caches()
        _, _, grads_alone, _, _ = chain_grads(
            backend,
            alone,
            batch_invariant=True,
            initial_state=state0[n : n + 1].clone(),
            d_final_state=d_final[n : n + 1].clone(),
            dO=dO[s:e].clone(),
        )
        for name in LEAF_NAMES[variant]:
            scale = case.n if name in EXPANDED_LEAVES else 1
            rows = slice(s * scale, e * scale)
            assert_bitwise(f"seq {n} d{name}", grads[name][rows], grads_alone[name])
        assert_bitwise(f"seq {n} d_initial_state", grads["initial_state"][n], grads_alone["initial_state"][0])


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_int64_cu_seqlens(backend, variant):
    """int64 cu_seqlens is bitwise the int32 run: forward, checkpoint series and every gradient."""
    case = chain_case(variant, CHAIN_RAGGED, H=2, K=64, V=64)
    state0 = random_state(case)
    checkpoint = CHUNK[variant]
    wide = case.cu.to(torch.int64)
    o32, fs32, series32 = run_fwd(backend, case, initial_state=state0, output_final_state=True, checkpoint_every_n_tokens=checkpoint)
    o64, fs64, series64 = run_fwd(backend, case, cu=wide, initial_state=state0, output_final_state=True, checkpoint_every_n_tokens=checkpoint)
    rows = sum((length * case.n - 1) // checkpoint + 1 for length in case.cu.diff().tolist() if length > 0)
    assert_bitwise("o", o64, o32)
    assert_bitwise("final_state", fs64, fs32)
    assert_bitwise("state_checkpoints", series64[:rows], series32[:rows])
    _, _, grads32, dO, _ = chain_grads(backend, case, initial_state=state0)
    _, _, grads64, _, _ = chain_grads(backend, case, cu=wide, initial_state=state0, dO=dO)
    for name in grads32:
        assert_bitwise(f"d{name}", grads64[name], grads32[name])


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("cadence", [1, 2], ids=["chunk", "two_chunks"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_checkpoint_series_layout_matches_uncut(backend, variant, cadence):
    """The checkpoint series of a cut batch has the unsplit layout: same row count, row 0 per sequence bitwise the
    entering state, every row the state at the same token boundary, no rows for an empty sequence."""
    checkpoint = cadence * CHUNK[variant]
    case = chain_case(variant, CHAIN_RAGGED, H=4)
    state0 = random_state(case)
    _, _, series_chain = run_fwd(backend, case, initial_state=state0, output_final_state=True, checkpoint_every_n_tokens=checkpoint)
    _, _, series_uncut = run_fwd(backend, case, initial_state=state0, output_final_state=True, batch_invariant=True, checkpoint_every_n_tokens=checkpoint)
    assert series_chain.shape == series_uncut.shape
    row = 0
    for n, length in enumerate(case.cu.diff().tolist()):
        rows = (length * case.n - 1) // checkpoint + 1 if length > 0 else 0
        if rows:
            assert_bitwise(f"seq {n} row 0", series_chain[row], state0[n].to(series_chain.dtype))
            for r in range(rows):
                assert_rms_close(f"seq {n} row {r}", series_chain[row + r], series_uncut[row + r].float(), CHAIN_FWD_TOL)
        row += rows
    assert row <= series_chain.shape[0]


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("cadence", [0, 1, 3], ids=["no_series", "dense_series", "coarse_series"])
@pytest.mark.parametrize("seq_lens,H", CHAIN_SHAPES, ids=CHAIN_SHAPE_IDS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_bwd_matches_uncut_and_reference(backend, variant, seq_lens, H, cadence):
    """The three backward checkpoint cases with a nonzero initial_state and d_final_state; every gradient against
    the batch run whole, and without a series against the fp64 reference (the cadence does not enter the reference)."""
    checkpoint = cadence * CHUNK[variant]
    case = chain_case(variant, seq_lens, H=H)
    state0, d_final = random_state(case), random_state(case, scale=0.1, seed=SEED + 3)
    _, _, grads, dO, _ = chain_grads(backend, case, checkpoint=checkpoint, initial_state=state0, d_final_state=d_final)
    _, _, grads_uncut, _, _ = chain_grads(backend, case, batch_invariant=True, checkpoint=checkpoint, initial_state=state0, d_final_state=d_final, dO=dO)
    assert_chain_grads_close(grads, grads_uncut)
    if cadence == 0:
        for name, want in chain_reference_grads(case, state0, dO, d_final).items():
            assert_rms_close(f"d{name} vs fp64", grads[name], want, CHAIN_STATE_GRAD_TOL if name == "initial_state" else CHAIN_BWD_TOL)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_bwd_without_states(backend, variant):
    """No initial_state, no d_final_state, no series (the plain training step)."""
    case = chain_case(variant, CHAIN_RAGGED, H=4)
    _, _, grads, dO, _ = chain_grads(backend, case)
    _, _, grads_uncut, _, _ = chain_grads(backend, case, batch_invariant=True, dO=dO)
    assert_chain_grads_close(grads, grads_uncut)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_bwd_bf16_state_pair(backend, variant):
    """With a bf16 initial_state / d_final_state, final_state and d_initial_state come back in the state dtype and
    every gradient tracks the batch run whole."""
    case = chain_case(variant, [4096], H=4)
    state0, d_final = random_state(case, dtype=torch.bfloat16), random_state(case, scale=0.1, seed=SEED + 3, dtype=torch.bfloat16)
    _, fs, grads, dO, _ = chain_grads(backend, case, initial_state=state0, d_final_state=d_final)
    assert fs.dtype == torch.bfloat16 and grads["initial_state"].dtype == torch.bfloat16
    _, _, grads_uncut, _, _ = chain_grads(backend, case, batch_invariant=True, initial_state=state0, d_final_state=d_final, dO=dO)
    assert_chain_grads_close(grads, grads_uncut)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("l2norm", [False, True], ids=["plain", "l2norm"])
@pytest.mark.parametrize("H,HK,HV", [(4, 1, 1), (1, 2, 2), (2, 2, 4)], ids=["fold_dk_dv", "fold_dq", "gva"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_head_folds_and_l2norm(backend, variant, H, HK, HV, l2norm):
    """GQA / GVA head folds and the in-kernel qk l2norm on a cut sequence."""
    case = chain_case(variant, [2048], H=H, HK=HK, HV=HV, K=64, V=128)
    o, _, grads, dO, _ = chain_grads(backend, case, use_qk_l2norm_in_kernel=l2norm)
    o_u, _, grads_u, _, _ = chain_grads(backend, case, batch_invariant=True, dO=dO, use_qk_l2norm_in_kernel=l2norm)
    assert_rms_close("o chain-vs-uncut", o, o_u.float(), CHAIN_VS_UNCUT_TOL)
    assert_chain_grads_close(grads, grads_u)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_cuda_graph_replay(backend, variant):
    """A cut forward + backward captured into a CUDA graph replays bitwise the eager run."""
    case = chain_case(variant, [4096], H=4)
    leaves = [t.requires_grad_(True) for t in thd_tensors(case)]
    state0 = random_state(case).requires_grad_(True)
    d_final = random_state(case, scale=0.1, seed=SEED + 3)
    dO = torch.randn(case.T, case.HO, case.V, device="cuda", dtype=case.dtype)

    def launch():
        with waive_unsupported(backend, variant):
            out = pinned_op(backend, variant)(*leaves, *op_tail(case), initial_state=state0, output_final_state=True)
        o, fs = out[0], out[1]
        return [o, fs, *torch.autograd.grad([o, fs], leaves + [state0], [dO, d_final])]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        eager = launch()
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        captured = launch()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    names = ["o", "final_state"] + [f"d{name}" for name in LEAF_NAMES[variant]] + ["d_initial_state"]
    for name, a, b in zip(names, eager, captured):
        assert_bitwise(f"replayed {name}", b, a)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("dims", [(64, 128), (128, 64)], ids=["k64_v128", "k128_v64"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_rectangular(backend, variant, dims):
    """A rectangular state on a cut sequence; forward and backward against the batch run whole and the fp64
    reference."""
    K, V = dims
    case = chain_case(variant, [2048], H=4, K=K, V=V)
    state0, d_final = random_state(case), random_state(case, scale=0.1, seed=SEED + 3)
    o, fs, grads, dO, _ = chain_grads(backend, case, initial_state=state0, d_final_state=d_final)
    o_u, fs_u, grads_u, _, _ = chain_grads(backend, case, batch_invariant=True, initial_state=state0, d_final_state=d_final, dO=dO)
    assert_rms_close("o chain-vs-uncut", o, o_u.float(), CHAIN_VS_UNCUT_TOL)
    assert_rms_close("final_state chain-vs-uncut", fs, fs_u.float(), CHAIN_VS_UNCUT_TOL)
    assert_chain_grads_close(grads, grads_u)
    o_ref, fs_ref = reference(case, initial_state=state0)
    assert_rms_close("o vs fp64", o, o_ref, FWD_TOL[case.dtype])
    assert_rms_close("final_state vs fp64", fs, fs_ref, STATE_TOL[case.dtype])


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_zero_length_sequences_pass_through(backend, variant):
    """Empty sequences inside a cut batch: final_state = initial_state and d_initial_state = d_final_state bitwise,
    the other outputs track the batch run whole."""
    case = chain_case(variant, [0, 4000, 0, 100, 0], H=2, K=64, V=64)
    state0, d_final = random_state(case), random_state(case, scale=0.1, seed=SEED + 3)
    o, fs, grads, dO, _ = chain_grads(backend, case, initial_state=state0, d_final_state=d_final)
    for n in (0, 2, 4):
        assert_bitwise(f"empty seq {n} final_state", fs[n], state0[n])
        assert_bitwise(f"empty seq {n} d_initial_state", grads["initial_state"][n], d_final[n])
    _, fs_uncut, grads_uncut, _, _ = chain_grads(backend, case, batch_invariant=True, initial_state=state0, d_final_state=d_final, dO=dO)
    assert_rms_close("final_state chain-vs-uncut", fs, fs_uncut.float(), CHAIN_VS_UNCUT_TOL)
    assert_chain_grads_close(grads, grads_uncut)


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("mode", CHAIN_GATE_MODES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_piece_chain_gate_modes_match_uncut(backend, variant, mode):
    """The fused l2norm, raw gates through the safe gate with a_log / dt_bias, beta logits through the in-kernel
    sigmoid (with and without allow_neg_eigval) and the beta guard on a cut sequence; outputs and every gradient,
    the parameter gradients included, track the batch run whole."""
    case, kw, _ = gate_mode_case(
        chain_case(variant, [2048], H=4), "sigmoid_beta" if mode == "neg_eigval" else mode, params="random", gate_lower_bound=-0.02, seed=SEED + 5
    )
    if mode == "neg_eigval":
        kw["allow_neg_eigval"] = True
    params = tuple((name, kw.pop(name)) for name in ("a_log", "dt_bias") if name in kw)
    o, fs, grads, dO, _ = chain_grads(backend, case, params=params, **kw)
    o_u, fs_u, grads_u, _, _ = chain_grads(backend, case, batch_invariant=True, dO=dO, params=params, **kw)
    assert_rms_close(f"o {mode} chain-vs-uncut", o, o_u.float(), CHAIN_VS_UNCUT_TOL)
    assert_rms_close(f"final_state {mode} chain-vs-uncut", fs, fs_u.float(), CHAIN_VS_UNCUT_TOL)
    assert_chain_grads_close(grads, grads_u, label=f"{mode} chain-vs-uncut")


# ---------------------------------------------------------------------------
# H vs the fp64 oracle
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dims", [(64, 64), (64, 128), (128, 64), (128, 128)])
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_h_matches_oracle(backend, variant, dims, producer):
    dim_k, dim_v = dims
    case = make_case(variant, torch.bfloat16, T=256, K=dim_k, V=dim_v)
    if producer == "summary":
        h_buf = run_summary(case)[0]
    else:
        h_buf = run_frost(case, output_final_state=True)[1]
    h_ref = reference(case)[1]
    assert_state_close(f"H[{variant}, {dims}, {producer}]", h_buf, h_ref, H_TOL[torch.bfloat16])


@pytest.mark.L1
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_h_matches_oracle_long(backend, variant):
    case = soften(make_case(variant, torch.bfloat16, T=8192, H=1, lo=0.9995))
    out = run_frost(case, output_final_state=True)
    h_ref = reference(case)[1]
    assert_state_close(f"H[{variant}, T=8192]", out[1], h_ref, H_TOL[torch.bfloat16])


# ---------------------------------------------------------------------------
# Forward affine contract
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_forward_affine_contract(backend, variant):
    """final_state(S0) == X0 @ M_buf + H_buf, two-sided against the fp64 oracle."""
    case = make_case(variant, torch.bfloat16, T=256)
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 5)
    s0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.05

    direct = run_frost(case, initial_state=s0, output_final_state=True)[1]
    h_buf = run_frost(case, output_final_state=True)[1]
    m_buf = kernel_m(case)
    composed = s0.double() @ m_buf.double() + h_buf.double()

    oracle = reference(case, initial_state=s0)[1]
    err_direct = max(rms_ratio(direct, oracle), 1e-6)
    err_composed = rms_ratio(composed, oracle)
    assert err_composed <= AFFINE_KAPPA * err_direct, f"composed final state {err_composed:.4g} vs direct {err_direct:.4g} exceeds kappa {AFFINE_KAPPA}"
    assert_state_close(f"affine[{variant}]", composed.float(), direct.double(), 2 * H_TOL[torch.bfloat16])


# ---------------------------------------------------------------------------
# Backward affine contract
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_backward_affine_contract(backend, variant):
    """d_initial_state(dht = D0) == X_dht @ M_buf^T + X_G (transpose flip vs forward)."""
    case = make_case(variant, torch.bfloat16, T=256)
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 6)
    d_o = to_thd(torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1).to(case.v.dtype)
    d0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.1

    _, dh0_full = grads_via_port(case, d_o, d0)
    _, g_buf = grads_via_port(case, d_o, None)
    m_buf = kernel_m(case)

    composed = d0.double() @ m_buf.double().transpose(-1, -2) + g_buf.double()
    assert_state_close(f"bwd affine[{variant}]", dh0_full.double(), composed, G_TOL)


# ---------------------------------------------------------------------------
# VJP superposition
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_vjp_superposition(backend, variant):
    """grads(dO, dht) == grads(dO, 0) + grads(0, dht) for every input gradient."""
    case = make_case(variant, torch.bfloat16, T=256)
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 8)
    d_o = to_thd(torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1).to(case.v.dtype)
    d0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.1
    zero_o = torch.zeros_like(d_o)

    joint, ds0_joint = grads_via_port(case, d_o, d0)
    o_only, ds0_o = grads_via_port(case, d_o, None)
    dht_only, ds0_dht = grads_via_port(case, zero_o, d0)

    for name in joint:
        want = joint[name].double()
        got = o_only[name].double() + dht_only[name].double()
        denom = want.pow(2).mean().sqrt().clamp_min(1e-8)
        err = ((got - want).pow(2).mean().sqrt() / denom).item()
        assert err < SUPERPOSE_TOL, f"d{name} superposition rel err {err:.4g} >= {SUPERPOSE_TOL}"
    err = rms_ratio(ds0_o.double() + ds0_dht.double(), ds0_joint)
    assert err < SUPERPOSE_TOL, f"d_initial_state superposition rel err {err:.4g}"


# ---------------------------------------------------------------------------
# G vs the fp64 adjoint oracle
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_g_matches_oracle(backend, variant):
    case = make_case(variant, torch.bfloat16, T=256)
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 9)
    d_o_full = torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1
    d_o = to_thd(d_o_full).to(case.v.dtype)

    _, g_buf = grads_via_port(case, d_o, None)

    s0_ref = torch.zeros(N, HO, V, K, device="cuda", dtype=torch.float64, requires_grad=True)
    o_ref, _ = reference(case, initial_state=s0_ref)
    (g_ref,) = torch.autograd.grad([o_ref], [s0_ref], [d_o_full.double().reshape(o_ref.shape)])
    assert_state_close(f"G[{variant}]", g_buf.double(), g_ref, G_TOL)


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_g_multi_sequence(backend, variant):
    seq_lens = [31, 93, 150]
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens)
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 10)
    d_o_full = torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1
    d_o = to_thd(d_o_full).to(case.v.dtype)

    _, g_buf = grads_via_port(case, d_o, None)

    s0_ref = torch.zeros(N, HO, V, K, device="cuda", dtype=torch.float64, requires_grad=True)
    o_ref, _ = reference(case, initial_state=s0_ref)
    (g_ref,) = torch.autograd.grad([o_ref], [s0_ref], [d_o_full.double().reshape(o_ref.shape)])
    assert_state_close(f"G[{variant}, varlen]", g_buf.double(), g_ref, G_TOL)


# ---------------------------------------------------------------------------
# Summary backward
# ---------------------------------------------------------------------------


# the GDP summary reads compact q / dO (masked tile rows in place of the port's zero-filled expanded rows), so its
# fp32 reductions reassociate against the expanded port: close at bf16-ulp scale, not bitwise
PORT_CLOSE_ONLY = {("gdp", 64), ("gdp", 128)}


def run_summary_bwd(case, d_o, *, d_final_state=None, **kw):
    """d_initial_state through the summary-bwd op (bprop_summary kernel)."""
    kw.setdefault("plan_name", f"{case.variant}_summary_frost")
    fn = getattr(la_ops, SUMMARY_BWD_OPS[case.variant])
    args = op_args(case)
    tail = args[len(LEAF_NAMES[case.variant]) :]
    try:
        return fn(args[0], args[1], args[3], args[4], d_o, *tail, d_final_state=d_final_state, **kw)
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"{case.variant} frost summary engine declined: {e}")


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dim_k", [64, 128])
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_bitwise_vs_port(backend, variant, dim_k):
    """G from the backward summary equals the full-backward port's
    d_initial_state bitwise; one (sequence, head) tile per SM runs every
    sequence whole on both sides."""
    case = make_case(variant, torch.bfloat16, T=257, K=dim_k, V=dim_k, H=sm_count())
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 12)
    d_o = to_thd(torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1).to(case.v.dtype)

    g_kernel = run_summary_bwd(case, d_o)
    _, g_port = grads_via_port(case, d_o, None)
    if (variant, dim_k) in PORT_CLOSE_ONLY:
        assert rms_ratio(g_kernel, g_port.double()) < 1e-4
        return
    assert torch.equal(g_kernel, g_port), f"backward summary diverges from the port path (max abs diff {(g_kernel - g_port).abs().max().item():.4g})"


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_bitwise_vs_port_with_dfs(backend, variant):
    case = make_case(variant, torch.bfloat16, T=257, H=sm_count())
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 14)
    d_o = to_thd(torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1).to(case.v.dtype)
    d0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.1

    g_kernel = run_summary_bwd(case, d_o, d_final_state=d0)
    _, g_port = grads_via_port(case, d_o, d0)
    if (variant, case.K) in PORT_CLOSE_ONLY:
        assert rms_ratio(g_kernel, g_port.double()) < 1e-4
        return
    assert torch.equal(g_kernel, g_port)


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_varlen_and_zero_length(backend, variant):
    """Per-sequence independence and the zero-length passthrough
    (d_initial_state = d_final_state, or zero without one)."""
    seq_lens = [31, 0, 93, 150]
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens)
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 15)
    d_o = to_thd(torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1).to(case.v.dtype)
    d0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.1

    g_kernel = run_summary_bwd(case, d_o, d_final_state=d0)
    _, g_port = grads_via_port(case, d_o, d0)
    if (variant, case.K) in PORT_CLOSE_ONLY:
        assert rms_ratio(g_kernel, g_port.double()) < 1e-4
    else:
        assert torch.equal(g_kernel, g_port)
    torch.testing.assert_close(g_kernel[1], d0[1], atol=0.0, rtol=0.0)

    g_no_dfs = run_summary_bwd(case, d_o)
    assert not g_no_dfs[1].any(), "zero-length sequence without d_final_state must give zero d_initial_state"


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_matches_adjoint_oracle(backend, variant):
    """G through the backward summary against the fp64 adjoint."""
    case = make_case(variant, torch.bfloat16, T=256)
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 9)
    d_o_full = torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1
    d_o = to_thd(d_o_full).to(case.v.dtype)

    g_kernel = run_summary_bwd(case, d_o)

    s0_ref = torch.zeros(N, HO, V, K, device="cuda", dtype=torch.float64, requires_grad=True)
    o_ref, _ = reference(case, initial_state=s0_ref)
    (g_ref,) = torch.autograd.grad([o_ref], [s0_ref], [d_o_full.double().reshape(o_ref.shape)])
    assert_state_close(f"G[{variant}, gradient op]", g_kernel.double(), g_ref, G_TOL)


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_backward_affine_contract_gradient_op(backend, variant):
    """dh0(dht = D0) == X_dht @ M_buf^T + X_G with both legs from the backward
    summary."""
    case = make_case(variant, torch.bfloat16, T=256)
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 6)
    d_o = to_thd(torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1).to(case.v.dtype)
    d0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.1

    dh0_full = run_summary_bwd(case, d_o, d_final_state=d0)
    g_buf = run_summary_bwd(case, d_o)
    m_buf = kernel_m(case, "summary")

    composed = d0.double() @ m_buf.double().transpose(-1, -2) + g_buf.double()
    assert_state_close(f"bwd affine[{variant}, gradient op]", dh0_full.double(), composed, G_TOL)


# ---------------------------------------------------------------------------
# Optional gate parameters and fp32 beta logits
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.parametrize("arm", ABSENT_ARMS)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_absent_params_bitwise(backend, variant, arm):
    """An absent a_log (unit amplitude) / dt_bias (zero bias) is bitwise the
    explicit zero tensor in H, M and G; the explicit arm's G is bitwise the
    full-backward port under the same safe gate, and the fp64 M oracle reads
    an absent parameter the same way."""
    case, kw_ref, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=256, H=sm_count()), "safe", params="random", arm=arm, seed=SEED + 26)
    kw_abs = dict(kw_ref, **{name: None for name in ABSENT_PARAMS[arm]})
    N, HO, K, V = case.N, case.HO, case.K, case.V
    d_o = to_thd(torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1).to(case.v.dtype)
    d0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.1

    def run(kw):
        h_buf, m_buf = run_summary(case, **kw)
        return h_buf, m_buf, run_summary_bwd(case, d_o, d_final_state=d0, **kw)

    ref = run(kw_ref)
    absent = run(kw_abs)
    for name, got, want in zip(("H", "M", "G"), absent, ref):
        assert_bitwise(name, got, want)
    _, g_port = grads_via_port(case, d_o, d0, **kw_ref)
    if (variant, case.K) in PORT_CLOSE_ONLY:
        assert rms_ratio(ref[2], g_port.double()) < 1e-4, "backward summary under the safe gate diverges from the port path"
    else:
        assert torch.equal(ref[2], g_port), "backward summary under the safe gate diverges from the port path"
    m_oracle = oracle_m(case, **kw_abs)
    assert_state_close(f"M[{variant}, {arm}]", absent[1], m_oracle, M_TOL[torch.bfloat16])


@pytest.mark.L0
@pytest.mark.parametrize("beta_dtype", [torch.float32, torch.bfloat16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_sigmoid_beta_bitwise_vs_port(backend, variant, beta_dtype):
    """Raw beta logits under the in-kernel sigmoid, float32 or the io dtype; the
    backward summary stays bitwise the full-backward port."""
    gate_mode = "sigmoid_beta_fp32" if beta_dtype == torch.float32 else "sigmoid_beta"
    case, op_kw, _ = gate_mode_case(make_case(variant, torch.bfloat16, T=257, H=sm_count()), gate_mode)
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(SEED + 27)
    d_o = to_thd(torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1).to(case.v.dtype)
    d0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.1
    g_kernel = run_summary_bwd(case, d_o, d_final_state=d0, **op_kw)
    _, g_port = grads_via_port(case, d_o, d0, **op_kw)
    if (variant, case.K) in PORT_CLOSE_ONLY:
        assert rms_ratio(g_kernel, g_port.double()) < 1e-4
        return
    assert torch.equal(g_kernel, g_port)


# ---------------------------------------------------------------------------
# Piece chain: summary
# ---------------------------------------------------------------------------

STATE_CHAIN_TOL = 1e-2
STATE_CHAIN_RAGGED = [497, 16, 1, 480, 0, 253, 4096]


@pytest.fixture()
def summary_caches():
    """The op caches, emptied around a test that pins no backend."""
    clear_caches()
    yield
    clear_caches()


@pytest.mark.L0
@pytest.mark.parametrize("with_state", [False, True], ids=["zero_seed", "initial_state"])
@pytest.mark.parametrize("seq_lens", [[4096], STATE_CHAIN_RAGGED], ids=["1x4096", "ragged"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_chain_matches_oracle(backend, variant, seq_lens, with_state, summary_caches):
    """H and M of a long batch on few tiles against the fp64 oracles (M through
    the product oracle, H through the sequential reference); an empty sequence
    holds H = seed and M = I."""
    case = soften(make_case(variant, torch.bfloat16, seq_lens=seq_lens, lo=0.99))
    s0 = random_state(case, seed=SEED + 5) if with_state else None
    h_chain, m_chain = run_summary(case, initial_state=s0)
    m_ref = oracle_m(case)
    h_ref = reference(case, initial_state=s0)[1]
    assert_state_close(f"H[{variant}]", h_chain, h_ref, H_TOL[torch.bfloat16])
    eye = torch.eye(case.K, device="cuda", dtype=torch.float32).expand(case.HO, case.K, case.K)
    for n, s_len in enumerate(seq_lens):
        if s_len == 0:
            assert torch.equal(m_chain[n], eye), f"empty sequence {n} must hold M = I"
            want = s0[n] if with_state else torch.zeros_like(h_chain[n])
            assert torch.equal(h_chain[n], want), f"empty sequence {n} must pass its seed through"
        else:
            assert_state_close(f"M[{variant}] seq {n}", m_chain[n : n + 1], m_ref[n : n + 1], M_TOL[torch.bfloat16])


@pytest.mark.L0
@pytest.mark.parametrize("with_state", [False, True], ids=["zero_seed", "initial_state"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_chain_batch_invariant_length_rule_bitwise(backend, variant, with_state, summary_caches):
    """Under batch_invariant a sequence's H and M are bitwise the same alone and in a batch."""
    seq_lens = [20000, 3000]
    case = soften(make_case(variant, torch.bfloat16, seq_lens=seq_lens, H=1, lo=0.99))
    s0 = random_state(case, seed=SEED + 5) if with_state else None
    h, m = run_summary(case, initial_state=s0, batch_invariant=True)
    for n in range(len(seq_lens)):
        alone = window(case, int(case.cu[n]), int(case.cu[n + 1]))
        clear_caches()
        h_alone, m_alone = run_summary(alone, initial_state=s0[n : n + 1].clone() if with_state else None, batch_invariant=True)
        assert torch.equal(h[n], h_alone[0]), f"seq {n} H is not bitwise its own run"
        assert torch.equal(m[n], m_alone[0]), f"seq {n} M is not bitwise its own run"


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_chain_affine_contract(backend, variant, summary_caches):
    """final_state(S0) == X0 @ M_buf + H_buf on a cut batch, two-sided against the fp64 oracle."""
    case = soften(make_case(variant, torch.bfloat16, T=2048, lo=0.99))
    s0 = random_state(case, seed=SEED + 5)
    direct = run_summary(case, initial_state=s0)[0]
    h_buf, m_buf = run_summary(case)
    composed = s0.double() @ m_buf.double() + h_buf.double()
    oracle = reference(case, initial_state=s0)[1]
    err_direct = max(rms_ratio(direct, oracle), 1e-6)
    err_composed = rms_ratio(composed, oracle)
    assert err_composed <= AFFINE_KAPPA * err_direct, f"composed final state {err_composed:.4g} vs direct {err_direct:.4g} exceeds kappa {AFFINE_KAPPA}"
    assert_state_close(f"affine[{variant}]", composed.float(), direct.double(), 2 * H_TOL[torch.bfloat16])


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_chain_cuda_graph_replay(backend, variant, summary_caches):
    """The summary of a cut batch captured into a CUDA graph replays bitwise the eager run."""
    case = soften(make_case(variant, torch.bfloat16, T=2048, lo=0.99))
    s0 = random_state(case, seed=SEED + 5)

    def launch():
        return run_summary(case, initial_state=s0)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        eager = launch()
        for _ in range(2):
            launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        captured = launch()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured[0], eager[0]), "replayed H differs from the eager run"
        assert torch.equal(captured[1], eager[1]), "replayed M differs from the eager run"


@pytest.mark.L0
@pytest.mark.parametrize("dims", [(64, 128), (128, 64)], ids=["k64_v128", "k128_v64"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_chain_rectangular(backend, variant, dims, summary_caches):
    """A rectangular state on a cut batch; M is (K, K) while H is (V, K), both against the oracles."""
    dim_k, dim_v = dims
    case = soften(make_case(variant, torch.bfloat16, T=2048, K=dim_k, V=dim_v, lo=0.99))
    s0 = random_state(case, seed=SEED + 5)
    h_chain, m_chain = run_summary(case, initial_state=s0)
    assert h_chain.shape[-2:] == (dim_v, dim_k) and m_chain.shape[-2:] == (dim_k, dim_k)
    assert_state_close(f"M[{variant}, {dims}]", m_chain, oracle_m(case), M_TOL[torch.bfloat16])
    assert_state_close(f"H[{variant}, {dims}]", h_chain, reference(case, initial_state=s0)[1], H_TOL[torch.bfloat16])


# ---------------------------------------------------------------------------
# Piece chain: summary bwd
# ---------------------------------------------------------------------------


def gradient_chain_inputs(case, seed=SEED + 31):
    """(d_o in the io dtype, its fp32 (B, T, HO, V) source, d_final_state) for a summary-bwd case."""
    N, HO, K, V = case.N, case.HO, case.K, case.V
    set_seed(seed)
    d_o_full = torch.randn(case.B, case.T, HO, V, device="cuda", dtype=torch.float32) * 0.1
    d_o = to_thd(d_o_full).to(case.v.dtype)
    d0 = torch.randn(N, HO, V, K, device="cuda", dtype=torch.float32) * 0.1
    return d_o, d_o_full, d0


def adjoint_d_initial_state(case, d_o_full, d_final_state=None):
    """fp64 d_initial_state through autograd of the sequential reference, with d_o on the outputs
    and, when given, d_final_state on the final state."""
    N, HO, K, V = case.N, case.HO, case.K, case.V
    s0_ref = torch.zeros(N, HO, V, K, device="cuda", dtype=torch.float64, requires_grad=True)
    o_ref, fs_ref = reference(case, initial_state=s0_ref)
    outputs, cotangents = [o_ref], [d_o_full.double().reshape(o_ref.shape)]
    if d_final_state is not None:
        outputs.append(fs_ref)
        cotangents.append(d_final_state.double())
    (x_ref,) = torch.autograd.grad(outputs, [s0_ref], cotangents)
    return x_ref


@pytest.mark.L0
@pytest.mark.parametrize("with_seed", [False, True], ids=["zero_seed", "d_final_state"])
@pytest.mark.parametrize("seq_lens", [[4096], STATE_CHAIN_RAGGED], ids=["1x4096", "ragged"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_chain_matches_adjoint(backend, variant, seq_lens, with_seed, summary_caches):
    """d_initial_state of a long batch on few tiles against the fp64 adjoint; an
    empty sequence passes d_final_state through (zero without one)."""
    case = soften(make_case(variant, torch.bfloat16, seq_lens=seq_lens, lo=0.99))
    d_o, d_o_full, d0 = gradient_chain_inputs(case)
    seed = d0 if with_seed else None
    x_chain = run_summary_bwd(case, d_o, d_final_state=seed)
    for n, s_len in enumerate(seq_lens):
        if s_len == 0:
            expect = d0[n] if with_seed else torch.zeros_like(x_chain[n])
            assert torch.equal(x_chain[n], expect), f"empty sequence {n} must pass d_final_state through"
    x_ref = adjoint_d_initial_state(case, d_o_full, seed)
    assert_state_close(f"dh0[{variant}] chain vs adjoint", x_chain.double(), x_ref, G_TOL)


@pytest.mark.L0
@pytest.mark.parametrize("with_seed", [False, True], ids=["zero_seed", "d_final_state"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_chain_batch_invariant_length_rule_bitwise(backend, variant, with_seed, summary_caches):
    """Under batch_invariant a sequence's d_initial_state is bitwise the same alone and in a batch."""
    seq_lens = [20000, 3000]
    case = soften(make_case(variant, torch.bfloat16, seq_lens=seq_lens, H=1, lo=0.99))
    d_o, _, d0 = gradient_chain_inputs(case)
    seed = d0 if with_seed else None
    x = run_summary_bwd(case, d_o, d_final_state=seed, batch_invariant=True)
    for n in range(len(seq_lens)):
        s, e = int(case.cu[n]), int(case.cu[n + 1])
        alone = window(case, s, e)
        clear_caches()
        x_alone = run_summary_bwd(alone, d_o[s:e].contiguous(), d_final_state=d0[n : n + 1].clone() if with_seed else None, batch_invariant=True)
        assert torch.equal(x[n], x_alone[0]), f"seq {n} d_initial_state is not bitwise its own run"


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_chain_affine_contract(backend, variant, summary_caches):
    """dh0(dht = D0) == D0 @ M_buf^T + G_buf with all three legs on a cut batch,
    two-sided against the fp64 adjoint."""
    case = soften(make_case(variant, torch.bfloat16, T=2048, lo=0.99))
    d_o, d_o_full, d0 = gradient_chain_inputs(case)
    direct = run_summary_bwd(case, d_o, d_final_state=d0)
    g_buf = run_summary_bwd(case, d_o)
    m_buf = run_summary(case)[1]
    composed = d0.double() @ m_buf.double().transpose(-1, -2) + g_buf.double()
    oracle = adjoint_d_initial_state(case, d_o_full, d0)
    err_direct = max(rms_ratio(direct, oracle), 1e-6)
    err_composed = rms_ratio(composed, oracle)
    assert err_composed <= AFFINE_KAPPA * err_direct, f"composed d_initial_state {err_composed:.4g} vs direct {err_direct:.4g} exceeds kappa {AFFINE_KAPPA}"
    assert_state_close(f"bwd affine[{variant}]", composed.float(), direct.double(), STATE_CHAIN_TOL)


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_chain_cuda_graph_replay(backend, variant, summary_caches):
    """The backward summary of a cut batch captured into a CUDA graph replays bitwise the eager run."""
    case = soften(make_case(variant, torch.bfloat16, T=2048, lo=0.99))
    d_o, _, d0 = gradient_chain_inputs(case)

    def launch():
        return run_summary_bwd(case, d_o, d_final_state=d0)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        eager = launch()
        for _ in range(2):
            launch()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        captured = launch()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager), "replayed d_initial_state differs from the eager run"


# ---------------------------------------------------------------------------
# Graph API: state nodes
# ---------------------------------------------------------------------------

CUDNN_DTYPE = {
    torch.bfloat16: cudnn.data_type.BFLOAT16,
    torch.float16: cudnn.data_type.HALF,
    torch.float32: cudnn.data_type.FLOAT,
    torch.int32: cudnn.data_type.INT32,
}


def build_and_pin(graph, plan_name):
    """validate, lower, plan, pin ``plan_name`` and build; declines waive the test."""
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    names = [graph.get_plan_name_at_index(i) for i in range(len(graph.plans))]
    if plan_name not in names:
        pytest.skip(f"no {plan_name} plan for this graph (offered: {names})")
    graph.select_plan(names.index(plan_name))
    try:
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as exc:
        pytest.skip(f"{plan_name} declined: {exc}")
    graph.build_plans()


@pytest.mark.L0
@pytest.mark.parametrize("with_state", [False, True], ids=["zero_seed", "initial_state"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_graph_api_summary_node_matches_op(backend, variant, with_state, summary_caches):
    """The family's summary node built through cudnn.pygraph returns
    final_state and transition bitwise the summary op's."""
    case = soften(make_case(variant, torch.bfloat16, T=1024, lo=0.99))
    s0 = random_state(case, seed=SEED + 5) if with_state else None
    h_op, m_op = run_summary(case, initial_state=s0)
    args = op_args(case)[1:]
    if variant == "gdn2":
        k, v, g, beta, w, cu = args
        inputs = dict(k=k, v=v, g=g, beta=beta, w=w, cu_seqlens=cu)
    else:
        k, v, g, beta, cu = args[:5]
        inputs = dict(k=k, v=v, g=g, beta=beta, cu_seqlens=cu)
    if with_state:
        inputs["initial_state"] = s0
    graph = cudnn.pygraph()
    ports = {name: graph.tensor(list(tensor.shape), data_type=CUDNN_DTYPE[tensor.dtype], name=name) for name, tensor in inputs.items()}
    attrs = dict(output_transition=True)
    if variant in HOUSEHOLDER_VARIANTS:
        attrs["num_householder"] = case.n
    fs_t, transition_t = getattr(graph, f"{variant}_summary")(**ports, **attrs, name="summary")
    fs_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    transition_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    build_and_pin(graph, f"{variant}_summary_frost")
    final_state = torch.empty_like(h_op)
    transition = torch.empty_like(m_op)
    pack = {ports[name]: tensor for name, tensor in inputs.items()}
    pack[fs_t] = final_state
    pack[transition_t] = transition
    graph.execute(pack, torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    assert torch.equal(final_state, h_op), "graph-API final_state differs from the op"
    assert torch.equal(transition, m_op), "graph-API transition differs from the op"


@pytest.mark.L0
@pytest.mark.parametrize("with_seed", [False, True], ids=["zero_seed", "d_final_state"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_graph_api_summary_bwd_node_matches_op(backend, variant, with_seed, summary_caches):
    """The family's summary-bwd node built through cudnn.pygraph returns
    d_initial_state and transition bitwise the summary-bwd op's."""
    case = soften(make_case(variant, torch.bfloat16, T=1024, lo=0.99))
    d_o, _, d0 = gradient_chain_inputs(case)
    seed = d0 if with_seed else None
    x_op, transition_op = run_summary_bwd(case, d_o, d_final_state=seed, output_transition=True)
    args = op_args(case)
    inputs = dict(q=args[0], k=args[1], g=args[3], beta=args[4], cu_seqlens=case.cu, dO=d_o)
    K = case.K
    if with_seed:
        inputs["d_final_state"] = d0
    graph = cudnn.pygraph()
    ports = {name: graph.tensor(list(tensor.shape), data_type=CUDNN_DTYPE[tensor.dtype], name=name) for name, tensor in inputs.items()}
    attrs = dict(scale=1.0 / math.sqrt(K), output_transition=True)
    if variant in HOUSEHOLDER_VARIANTS:
        attrs["num_householder"] = case.n
    x_t, transition_t = getattr(graph, f"{variant}_summary_bwd")(**ports, **attrs, name="summary_bwd")
    x_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    transition_t.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    build_and_pin(graph, f"{variant}_summary_frost")
    d_initial_state = torch.empty_like(x_op)
    transition = torch.empty_like(transition_op)
    pack = {ports[name]: tensor for name, tensor in inputs.items()}
    pack[x_t] = d_initial_state
    pack[transition_t] = transition
    graph.execute(pack, torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    assert torch.equal(d_initial_state, x_op), "graph-API d_initial_state differs from the op"
    assert torch.equal(transition, transition_op), "graph-API transition differs from the op"


# ---------------------------------------------------------------------------
# Two-tier context parallelism
# ---------------------------------------------------------------------------

TRANSITION_FP32_TOL = 1e-4
TWO_TIER_TOL = 2e-2
TWO_TIER_STATE_TOL = 6e-2
TWO_TIER_BOUNDS = [0, 512, 1024, 1536, 2048]


def family_forward_backward(case, s0, d_o, d_final):
    """(o, final_state, gradients by leaf name with ``initial_state``) of the family op under autograd, seeded with
    ``s0`` and driven by ``d_o`` on o and ``d_final`` on final_state."""
    seed = s0.detach().clone().requires_grad_(True)
    leaves = dict(zip(LEAF_NAMES[case.variant], (t.requires_grad_(True) for t in thd_tensors(case))))
    try:
        o, fs = op(case.variant)(*leaves.values(), *op_tail(case), initial_state=seed, output_final_state=True, plan_name=f"{case.variant}_frost")
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"{case.variant} frost engine declined: {e}")
    grads = torch.autograd.grad([o, fs], list(leaves.values()) + [seed], [d_o, d_final])
    named = dict(zip(list(leaves) + ["initial_state"], grads))
    return o.detach(), fs.detach(), named


def compose_spans(seed, h, m):
    """Incoming state of every span and the state after the last, ``X_{j+1} = X_j @ M_j + H_j`` in fp32."""
    x = [seed]
    for j in range(h.shape[0]):
        x.append(x[-1] @ m[j] + h[j])
    return x


@pytest.mark.L0
@pytest.mark.parametrize("cut", ["whole", "one_piece", "chain"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_transition_orientation(backend, variant, cut, summary_caches):
    """The gradient op's transition is the forward summary's transition transposed: bitwise where
    one piece is walked (a span short enough to walk whole, and a one-piece batch-invariant chain),
    fp32-close over a many-piece chain (the products associate in opposite order)."""
    if cut == "whole":
        case, kw = soften(make_case(variant, torch.bfloat16, T=64, lo=0.99)), {}
    elif cut == "one_piece":
        case, kw = soften(make_case(variant, torch.bfloat16, T=1024, lo=0.99)), dict(batch_invariant=True)
    else:
        case, kw = soften(make_case(variant, torch.bfloat16, T=4096, lo=0.99)), {}
    d_o, _, _ = gradient_chain_inputs(case)
    m_fwd = run_summary(case, **kw)[1]
    _, t_bwd = run_summary_bwd(case, d_o, output_transition=True, **kw)
    want = m_fwd.transpose(-1, -2)
    assert t_bwd.shape == want.shape and t_bwd.dtype == torch.float32
    if cut == "chain":
        assert_state_close(f"transition[{variant}, {cut}]", t_bwd, want, TRANSITION_FP32_TOL)
    else:
        assert torch.equal(t_bwd, want), f"transition[{variant}, {cut}] is not bitwise the forward transition transposed"


@pytest.mark.L0
@pytest.mark.parametrize("dims", HEAD_DIMS, ids=lambda d: f"k{d[0]}_v{d[1]}")
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_summary_bwd_transition_orientation_dims(backend, variant, dims, summary_caches):
    """The transposed-transition identity at every (K, V) on a whole-walked span."""
    dim_k, dim_v = dims
    case = soften(make_case(variant, torch.bfloat16, T=64, K=dim_k, V=dim_v, lo=0.99))
    d_o, _, _ = gradient_chain_inputs(case)
    m_fwd = run_summary(case)[1]
    _, t_bwd = run_summary_bwd(case, d_o, output_transition=True)
    assert t_bwd.shape[-2:] == (dim_k, dim_k)
    assert torch.equal(t_bwd, m_fwd.transpose(-1, -2)), f"transition[{variant}, {dims}] is not bitwise the forward transition transposed"


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_two_tier_forward_assembles_whole_sequence(backend, variant, summary_caches):
    """Per-span H and M composed in fp32 seed each span's forward; o and the last final_state
    reproduce the whole-sequence forward at the family bound."""
    case = soften(make_case(variant, torch.bfloat16, T=TWO_TIER_BOUNDS[-1], lo=0.99))
    spans = case.clone(cu=torch.tensor(TWO_TIER_BOUNDS, dtype=case.cu.dtype, device="cuda"), N=len(TWO_TIER_BOUNDS) - 1, varlen=True)
    s0 = random_state(case, seed=SEED + 5)
    o_whole, fs_whole = run_frost(case, initial_state=s0, output_final_state=True)[:2]
    h, m = run_summary(spans)
    x = compose_spans(s0[0], h, m)
    o_spans, fs_spans = run_frost(spans, initial_state=torch.stack(x[:-1]), output_final_state=True)[:2]
    assert_rms_close(f"o[{variant}, two-tier]", o_spans, o_whole, CHAIN_FWD_TOL)
    assert_state_close(f"final_state[{variant}, two-tier]", fs_spans[-1:], fs_whole, H_TOL[torch.bfloat16])
    assert_state_close(f"composed final_state[{variant}, two-tier]", x[-1][None], fs_whole, H_TOL[torch.bfloat16])


@pytest.mark.L0
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("backend", ["frost"], indirect=True)
def test_two_tier_backward_assembles_whole_sequence(backend, variant, summary_caches):
    """Per-span G and transition composed in reverse in fp32 give each span its outgoing state
    gradient; the spans' backwards, seeded with their incoming states, reproduce every gradient of
    the whole-sequence backward at the family bound."""
    case = soften(make_case(variant, torch.bfloat16, T=TWO_TIER_BOUNDS[-1], lo=0.99))
    spans = case.clone(cu=torch.tensor(TWO_TIER_BOUNDS, dtype=case.cu.dtype, device="cuda"), N=len(TWO_TIER_BOUNDS) - 1, varlen=True)
    s0 = random_state(case, seed=SEED + 5)
    d_o, _, d_final = gradient_chain_inputs(case)
    _, _, grads_whole = family_forward_backward(case, s0, d_o, d_final)
    h, m = run_summary(spans)
    x = compose_spans(s0[0], h, m)
    g_spans, t_spans = run_summary_bwd(spans, d_o, output_transition=True)
    dx = [d_final[0]]
    for j in reversed(range(len(TWO_TIER_BOUNDS) - 1)):
        dx.append(dx[-1] @ t_spans[j] + g_spans[j])
    dx = dx[::-1]
    _, _, grads_spans = family_forward_backward(spans, torch.stack(x[:-1]), d_o, torch.stack(dx[1:]))
    for name, whole in grads_whole.items():
        if name == "initial_state":
            assert_state_close(f"d_initial_state[{variant}, two-tier]", grads_spans[name][:1], whole, TWO_TIER_STATE_TOL)
        else:
            assert_rms_close(f"d{name}[{variant}, two-tier]", grads_spans[name], whole, TWO_TIER_TOL)
    assert_state_close(f"composed span state gradients[{variant}]", torch.stack(dx[:-1]), grads_spans["initial_state"], STATE_CHAIN_TOL)
