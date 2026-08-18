# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-parametrized test suite for the linear-attention ops.

One suite for the public API (``gated_delta_net`` / ``kimi_delta_attention`` /
``gated_delta_net_v2``), parametrized over the python engine backends that can
serve it. Each test pins one backend's plan onto the ops' graphs (plan-API
``select_plan`` by name) and validates against the fp64 recurrent references. A pinned engine that declines a
configuration waives the test (``cudnnGraphNotSupportedError`` -> skip), so
the support surface is owned by the engines' ``check_support``, not by a
suite-side matrix; a backend that is not installed skips the same way.

Determinism soaks and CUDA-graph replay run on the backends that promise
those contracts (currently FROST only).
"""

from __future__ import annotations

import contextlib
import functools
import math
import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")
la_ops = pytest.importorskip("cudnn.linear_attention.ops")

import torch.nn.functional as F  # noqa: E402

from .conftest import gen_qkv  # noqa: E402
from .reference_gdn import gdn_reference, rms_ratio  # noqa: E402
from .reference_gdn2 import gdn2_reference  # noqa: E402
from .reference_kda import kda_reference  # noqa: E402

pytestmark = [
    pytest.mark.L0,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

VARIANTS = ("gdn", "kda", "gdn2")
CHUNK = {"gdn": 64, "kda": 16, "gdn2": 16}

FWD_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
STATE_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
BWD_TOL = {torch.bfloat16: 4e-2, torch.float16: 3e-2}
STATE_GRAD_TOL = 6e-2

# (H, HV) pairs: H = Q/K heads, HV = V heads; gates/O/states live at HO = max.
HEAD_CONFIGS = [(1, 1), (3, 3), (1, 2), (2, 4), (16, 32), (16, 64)]
HEAD_CONFIGS_SMALL = [(1, 1), (2, 4)]
GQA_CONFIGS = [(4, 4, 1), (6, 6, 2), (4, 1, 1), (6, 2, 2), (1, 2, 2), (2, 4, 4)]

RAGGED_SEQ_LENS = [
    [256, 256],
    [511, 501],
    [64, 128, 512],
    [31, 63, 93, 123, 150, 500],
    [7] * 24 + [1] * 8,
    [2048],
]
EDGE_LENS = [1, 15, 16, 17, 31, 63, 64, 65, 121, 251, 257]

DETERMINISM_REPEATS = 8
SEED = 888

DTYPE_IDS = {torch.bfloat16: "bf16", torch.float16: "fp16"}


# ---------------------------------------------------------------------------
# Backend pinning
# ---------------------------------------------------------------------------


def op_modules():
    from cudnn.linear_attention.ops import gdn, gdn2, kda

    return {"gdn": gdn, "kda": kda, "gdn2": gdn2}


def family_engines(backend_name):
    """The backend's engine instances per family, ids assigned by the manifest."""
    from cudnn.engines import manifest

    suffix = "_" + backend_name
    families = {}
    for family in manifest.MANIFEST:
        if family.name in VARIANTS:
            engines = manifest.instantiate(family, family.offered_ids())
            families[family.name] = [e for e in engines if e.name.endswith(suffix)]
    return families


def clear_op_caches():
    for mod in op_modules().values():
        mod._fprop_cache.clear()
        mod._bprop_cache.clear()


class Case:
    """One test configuration: inputs, gates, cu_seqlens and the geometry."""

    __slots__ = ("variant", "dtype", "q", "k", "v", "gates", "cu", "B", "T", "N", "H", "HK", "HV", "HO", "K", "V", "varlen")

    def __init__(self, **fields):
        for name in self.__slots__:
            setattr(self, name, fields.pop(name))
        assert not fields, f"unknown Case fields: {sorted(fields)}"

    def clone(self, **overrides):
        fields = {name: getattr(self, name) for name in self.__slots__}
        fields.update(overrides)
        return Case(**fields)


class Backend:
    """The pinned backend: its name, its engine instances per family, and the
    plan name a graph must offer (``<variant>_<name>``, e.g. ``gdn_frost``)."""

    __slots__ = ("name", "engines")

    def __init__(self, name, engines):
        self.name = name
        self.engines = engines

    def plan(self, variant):
        return f"{variant}_{self.name}"


def pinned_op(backend, variant):
    """The variant's op with the backend's plan pinned (ops-level
    ``plan_name``, the examples' ``select_plan`` paradigm)."""
    return functools.partial(op(variant), plan_name=backend.plan(variant))


@pytest.fixture(params=("frost", "cutile"))
def backend(request):
    """One backend per run of each test; the tests pass its plan name to the
    ops. The op graph caches are cleared around each test so the pin
    assertions only ever see this test's graphs."""
    name = request.param
    families = family_engines(name)
    missing = [v for v in VARIANTS if v not in families]
    if missing:
        pytest.fail(f"the engine manifest offers no {missing} families — a stale installed cudnn package is likely shadowing the source tree")
    clear_op_caches()
    try:
        yield Backend(name, families)
    finally:
        clear_op_caches()


@contextlib.contextmanager
def waive_unsupported(backend, variant):
    """A backend with no engine for the family, or an engine decline (no plan
    offered / check_support raise), waives the test; the engines own the
    support surface."""
    if not backend.engines[variant]:
        pytest.skip(f"the {backend.name} backend has no {variant} engine")
    try:
        yield
    except cudnn.cudnnGraphNotSupportedError as exc:
        pytest.skip(f"{backend.name} {variant} declined: {exc}")


# ---------------------------------------------------------------------------
# Case generation and dispatch
# ---------------------------------------------------------------------------


def set_seed(seed=SEED):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def gate_lo(dtype):
    return 0.6 if dtype == torch.float16 else 0.5


def gen_gates(variant, B, T, HO, K, V, dtype, *, alpha=True, beta=True, w=True, lo=None, device="cuda"):
    if lo is None:
        lo = gate_lo(dtype)
    gshape = (B, T, HO) if variant == "gdn" else (B, T, HO, K)
    if alpha:
        g = torch.empty(gshape, device=device, dtype=torch.float32).uniform_(lo, 1.0).log()
    else:
        g = torch.zeros(gshape, device=device, dtype=torch.float32)
    if variant == "gdn2":
        b = (torch.rand(B, T, HO, K, device=device).sigmoid() * 2.0).to(dtype) if beta else torch.ones(B, T, HO, K, device=device, dtype=dtype)
        wt = torch.rand(B, T, HO, V, device=device).sigmoid().to(dtype) if w else torch.ones(B, T, HO, V, device=device, dtype=dtype)
        return {"g": g, "beta": b, "w": wt}
    b = torch.rand(B, T, HO, device=device) if beta else torch.ones(B, T, HO, device=device)
    return {"g": g, "beta": b}


def make_case(variant, dtype, *, B=1, T=None, seq_lens=None, H=2, HK=None, HV=None, K=128, V=128, alpha=True, beta=True, w=True, lo=None, seed=SEED):
    """Dense ``(B, T)`` or packed varlen (``seq_lens``, B == 1) inputs plus the
    matching ``cu_seqlens``. ``HK`` defaults to ``H``; ``HK == HV < H`` is
    canonical (native grouped K) GQA."""
    set_seed(seed)
    HV = H if HV is None else HV
    HK = H if HK is None else HK
    HO = max(H, HV)
    if seq_lens is not None:
        total = sum(seq_lens)
        bounds = [0]
        for sl in seq_lens:
            bounds.append(bounds[-1] + sl)
        cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
        B, T, N, varlen = 1, total, len(seq_lens), True
    else:
        cu = torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * T
        N, varlen = B, False
    q, k, v = gen_qkv(B, T, H, HV, K, V, dtype)
    if HK != H:
        from .conftest import multidist_randu

        k = F.normalize(multidist_randu(B * T * HK, K, device="cuda").reshape(B, T, HK, K), p=2.0, dim=-1).to(dtype).contiguous()
    gates = gen_gates(variant, B, T, HO, K, V, dtype, alpha=alpha, beta=beta, w=w, lo=lo)
    return Case(variant=variant, dtype=dtype, q=q, k=k, v=v, gates=gates, cu=cu, B=B, T=T, N=N, H=H, HK=HK, HV=HV, HO=HO, K=K, V=V, varlen=varlen)


def to_thd(x):
    return x.reshape(-1, *x.shape[2:])


def op(variant):
    return {"gdn": la_ops.gated_delta_net, "kda": la_ops.kimi_delta_attention, "gdn2": la_ops.gated_delta_net_v2}[variant]


def op_args(case, cu=None):
    args = [to_thd(case.q), to_thd(case.k), to_thd(case.v), to_thd(case.gates["g"]), to_thd(case.gates["beta"])]
    if case.variant == "gdn2":
        args.append(to_thd(case.gates["w"]))
    args.append(case.cu if cu is None else cu)
    return args


def run_fwd(backend, case, *, cu=None, **kw):
    with waive_unsupported(backend, case.variant):
        return pinned_op(backend, case.variant)(*op_args(case, cu=cu), **kw)


def reference(case, *, scale=None, initial_state=None, l2norm=False, cu=None):
    fn = {"gdn": gdn_reference, "kda": kda_reference, "gdn2": gdn2_reference}[case.variant]
    q, k = case.q, case.k
    if l2norm:
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
    args = [q, k, case.v, case.gates["g"], case.gates["beta"]]
    if case.variant == "gdn2":
        args.append(case.gates["w"])
    kwargs = dict(scale=scale, initial_state=initial_state)
    if case.varlen or cu is not None:
        kwargs["cu_seqlens"] = case.cu if cu is None else cu
    with torch.no_grad():
        return fn(*args, **kwargs)


def check(name, out, ref, tol):
    out = out.float()
    assert torch.isfinite(out).all(), f"non-finite values in {name}"
    r = rms_ratio(out.reshape(ref.shape), ref)
    assert r < tol, f"{name} rms ratio {r:.4g} >= {tol}"


def check_fwd(case, o, fs, *, scale=None, initial_state=None, l2norm=False, tol_mult=1.0):
    o_ref, fs_ref = reference(case, scale=scale, initial_state=initial_state, l2norm=l2norm)
    check("o", o, o_ref, tol_mult * FWD_TOL[case.dtype])
    if fs is not None and fs.numel():
        check("final_state", fs, fs_ref, tol_mult * STATE_TOL[case.dtype])


# ---------------------------------------------------------------------------
# Backend pin seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_backend_pin_selects_engine(backend, variant):
    """The pinned engine actually serves the graph — a dead pin would silently
    validate whatever the default routing picks."""
    case = make_case(variant, torch.bfloat16, T=4 * CHUNK[variant])
    with waive_unsupported(backend, variant):
        pinned_op(backend, variant)(*op_args(case))
    mod = op_modules()[variant]
    names = {g.selected_engine.name for g, entry in mod._fprop_cache.values() if g.selected_engine is not None}
    assert names == {f"{variant}_{backend.name}"}, f"expected only {variant}_{backend.name} to serve, got {names}"


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
    case = make_case(variant, dtype, B=B, T=T, H=H, HV=HV)
    o, fs = run_fwd(backend, case, output_final_state=True)
    check_fwd(case, o, fs)


@pytest.mark.parametrize("alpha,beta,w", [(True, False, True), (False, True, True), (True, True, False)], ids=["no_beta", "no_alpha", "no_w"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_gate_combinations(backend, variant, alpha, beta, w):
    if variant != "gdn" and not alpha:
        pytest.skip("the delta-rule inverse needs decay (matches FI's use_g=False skip)")
    if variant != "gdn2" and not w:
        pytest.skip("w is a GDN-2 gate")
    case = make_case(variant, torch.bfloat16, T=192, alpha=alpha, beta=beta, w=w)
    o, fs = run_fwd(backend, case, output_final_state=True)
    check_fwd(case, o, fs)


@pytest.mark.parametrize("scale", [0.5, 1.0, None], ids=["half", "one", "auto"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_scale(backend, variant, scale):
    case = make_case(variant, torch.bfloat16, T=192)
    o, fs = run_fwd(backend, case, scale=scale, output_final_state=True)
    check_fwd(case, o, fs, scale=scale)


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
    o, fs = run_fwd(backend, case, output_final_state=True)
    check_fwd(case, o, fs)


@pytest.mark.parametrize("H,HV", [(1, 1), (2, 4)])
@pytest.mark.parametrize("seq_lens", RAGGED_SEQ_LENS, ids=lambda sl: f"{len(sl)}seqs_{sum(sl)}tok")
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_varlen_ragged(backend, variant, seq_lens, H, HV):
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens, H=H, HV=HV)
    o, fs = run_fwd(backend, case, output_final_state=True)
    check_fwd(case, o, fs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_many_short_sequences(backend, variant):
    """A 200-sequence packed batch matches the same sequences run one by one."""
    T = 33
    case = make_case(variant, torch.bfloat16, seq_lens=[T] * 200)
    o, fs = run_fwd(backend, case, output_final_state=True)
    cu1 = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    for n in (0, 1, 99, 199):
        # clone: sliced views can start at non-16B-aligned offsets, which the
        # kernels' buffer contract rejects
        sl = slice(T * n, T * (n + 1))
        args = [
            to_thd(case.q)[sl].clone(),
            to_thd(case.k)[sl].clone(),
            to_thd(case.v)[sl].clone(),
            to_thd(case.gates["g"])[sl].clone(),
            to_thd(case.gates["beta"])[sl].clone(),
        ]
        if case.variant == "gdn2":
            args.append(to_thd(case.gates["w"])[sl].clone())
        with waive_unsupported(backend, variant):
            o_n, fs_n = pinned_op(backend, variant)(*args, cu1, output_final_state=True)
        check(f"o[seq {n}]", o[sl], o_n.float(), FWD_TOL[torch.bfloat16])
        check(f"final_state[seq {n}]", fs[n], fs_n[0].float(), STATE_TOL[torch.bfloat16])


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
    state0 = torch.randn(4, case.HO, case.K, case.V, device="cuda", dtype=torch.float32) * 0.05
    o, fs_state0 = run_fwd(backend, case, cu=cu, initial_state=state0, output_final_state=True)
    torch.testing.assert_close(fs_state0[1], state0[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(fs_state0[3], state0[3], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("T", [128, 251])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_initial_state(backend, variant, T):
    case = make_case(variant, torch.bfloat16, T=T)
    state0 = torch.randn(case.N, case.HO, case.K, case.V, device="cuda", dtype=torch.float32) * 0.05
    o, fs = run_fwd(backend, case, initial_state=state0, output_final_state=True)
    check_fwd(case, o, fs, initial_state=state0)


@pytest.mark.parametrize("T1,T2", [(128, 128), (64, 192), (192, 121)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_chunked_prefill(backend, variant, T1, T2):
    """Two-phase prefill: part 1's final state feeds part 2; the concatenated
    output matches a single-shot reference (state round-trips through fp32)."""
    case = make_case(variant, torch.bfloat16, B=2, T=T1 + T2)

    def part(t0, t1, state0):
        sub = case.clone()
        sub.q, sub.k, sub.v = (x[:, t0:t1].contiguous() for x in (case.q, case.k, case.v))
        sub.gates = {n: g[:, t0:t1].contiguous() for n, g in case.gates.items()}
        sub.T = t1 - t0
        sub.cu = torch.arange(0, case.B + 1, dtype=torch.int32, device="cuda") * sub.T
        return run_fwd(backend, sub, initial_state=state0, output_final_state=True)

    o1, fs1 = part(0, T1, None)
    o2, fs2 = part(T1, T1 + T2, fs1)
    o = torch.cat([o1.reshape(case.B, T1, case.HO, case.V), o2.reshape(case.B, T2, case.HO, case.V)], dim=1)
    o_ref, fs_ref = reference(case)
    check("o", o, o_ref, 1.5 * FWD_TOL[case.dtype])
    check("final_state", fs2, fs_ref, 1.5 * STATE_TOL[case.dtype])


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_packed_matches_per_sequence(backend, variant):
    B, T = 3, 128
    case = make_case(variant, torch.bfloat16, B=B, T=T)
    o, fs = run_fwd(backend, case, output_final_state=True)
    cu1 = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    for b in range(B):
        args = [case.q[b], case.k[b], case.v[b], case.gates["g"][b], case.gates["beta"][b]]
        if variant == "gdn2":
            args.append(case.gates["w"][b])
        with waive_unsupported(backend, variant):
            o_b, fs_b = pinned_op(backend, variant)(*args, cu1, output_final_state=True)
        torch.testing.assert_close(o[b * T : (b + 1) * T], o_b)
        torch.testing.assert_close(fs[b], fs_b[0])


@pytest.mark.parametrize(
    "variant,K,V",
    [("gdn", 64, 64), ("gdn", 64, 128), ("gdn", 128, 128), ("gdn", 256, 128), ("kda", 64, 64), ("kda", 64, 128), ("kda", 128, 128), ("gdn2", 128, 128)],
)
def test_fwd_head_dims(backend, variant, K, V):
    """K/V head-dim variants; engines that only serve K = V = 128 decline."""
    case = make_case(variant, torch.bfloat16, T=192, K=K, V=V)
    o, fs = run_fwd(backend, case, output_final_state=True)
    check_fwd(case, o, fs)


@pytest.mark.parametrize("H,HK,HV", GQA_CONFIGS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_gqa(backend, variant, H, HK, HV):
    """Grouped heads: canonical GQA (native K at HK == HV), the expanded-k
    form, and shared-kv GVA (HK == HV > H); gates/O/states live at HO = max(H, HV)."""
    case = make_case(variant, torch.bfloat16, T=192, H=H, HK=HK, HV=HV)
    o, fs = run_fwd(backend, case, output_final_state=True)
    check_fwd(case, o, fs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_multi_tile(backend, variant):
    """B*H well above the SM count: each CTA walks several (b, h) tiles back
    to back, exercising the inter-tile state drain -> seed ordering the
    single-tile cases never reach."""
    case = make_case(variant, torch.bfloat16, B=8, T=192, H=64)
    o, fs = run_fwd(backend, case, output_final_state=True)
    check_fwd(case, o, fs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_qk_l2norm(backend, variant):
    """In-kernel Q/K L2 norm matches the reference on pre-normalized inputs."""
    case = make_case(variant, torch.bfloat16, T=256)
    o, fs = run_fwd(backend, case, output_final_state=True, use_qk_l2norm_in_kernel=True)
    check_fwd(case, o, fs, l2norm=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_strong_decay_varlen(backend, variant):
    case = make_case(variant, torch.bfloat16, seq_lens=[100, 2048, 0, 517], lo=0.1 if variant == "gdn" else 0.3)
    o, fs = run_fwd(backend, case, output_final_state=True)
    check_fwd(case, o, fs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_fwd_output_contract(backend, variant):
    """O is io-dtype at HO heads; final_state is fp32 and empty unless requested."""
    case = make_case(variant, torch.bfloat16, T=128, H=2, HV=4)
    o, fs = run_fwd(backend, case)
    assert o.shape == (case.T, case.HO, case.V) and o.dtype == case.dtype
    assert fs.numel() == 0
    o, fs = run_fwd(backend, case, output_final_state=True)
    assert fs.shape == (case.N, case.HO, case.K, case.V) and fs.dtype == torch.float32


# ---------------------------------------------------------------------------
# Backward parity (oracle: fp64 autograd through the references)
# ---------------------------------------------------------------------------


def assert_bwd_parity(backend, case, *, scale=None, use_initial_state=False, use_dfs=False, l2norm=False, gate_grad_tol=None, seed=SEED + 1):
    variant, tol = case.variant, BWD_TOL[case.dtype]
    tensors = {"q": case.q, "k": case.k, "v": case.v, "g": case.gates["g"], "beta": case.gates["beta"]}
    if variant == "gdn2":
        tensors["w"] = case.gates["w"]
    op_leaves = {n: to_thd(t).detach().clone().requires_grad_(True) for n, t in tensors.items()}
    ref_leaves = {n: t.detach().double().requires_grad_(True) for n, t in tensors.items()}
    set_seed(seed)
    state0_op = state0_ref = None
    if use_initial_state:
        state0 = torch.randn(case.N, case.HO, case.K, case.V, device="cuda", dtype=torch.float32) * 0.05
        state0_op = state0.detach().clone().requires_grad_(True)
        state0_ref = state0.detach().double().requires_grad_(True)

    with waive_unsupported(backend, variant):
        args = [op_leaves["q"], op_leaves["k"], op_leaves["v"], op_leaves["g"], op_leaves["beta"]]
        if variant == "gdn2":
            args.append(op_leaves["w"])
        args.append(case.cu)
        o, fs = pinned_op(backend, variant)(*args, scale=scale, initial_state=state0_op, output_final_state=True, use_qk_l2norm_in_kernel=l2norm)
        dO = torch.randn_like(o)
        outputs, grad_outputs = [o], [dO]
        dFS = None
        if use_dfs:
            dFS = torch.randn_like(fs) * 0.1
            outputs.append(fs)
            grad_outputs.append(dFS)
        grad_inputs = list(op_leaves.values()) + ([state0_op] if use_initial_state else [])
        grads = torch.autograd.grad(outputs, grad_inputs, grad_outputs)

    qd, kd = ref_leaves["q"], ref_leaves["k"]
    if l2norm:
        qd, kd = F.normalize(qd, dim=-1), F.normalize(kd, dim=-1)
    ref_fn = {"gdn": gdn_reference, "kda": kda_reference, "gdn2": gdn2_reference}[variant]
    ref_args = [qd, kd, ref_leaves["v"], ref_leaves["g"], ref_leaves["beta"]]
    if variant == "gdn2":
        ref_args.append(ref_leaves["w"])
    ref_kwargs = dict(scale=scale, initial_state=state0_ref)
    if case.varlen:
        ref_kwargs["cu_seqlens"] = case.cu
    o_ref, fs_ref = ref_fn(*ref_args, **ref_kwargs)
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
        check(f"d{name}", got, want, tol_n)


@pytest.mark.parametrize("H,HV", HEAD_CONFIGS_SMALL + [(16, 64)])
@pytest.mark.parametrize("T", [64, 128, 251])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=DTYPE_IDS.get)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_parity(backend, variant, dtype, T, H, HV):
    if dtype == torch.float16 and (T != 128 or (H, HV) != (1, 1)):
        pytest.skip("fp16 runs one representative backward config")
    if (H, HV) == (16, 64) and T != 128:
        pytest.skip("the large GVA config runs one length")
    assert_bwd_parity(backend, make_case(variant, dtype, T=T, H=H, HV=HV))


@pytest.mark.parametrize("H,HK,HV", GQA_CONFIGS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_gqa(backend, variant, H, HK, HV):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128, H=H, HK=HK, HV=HV))


@pytest.mark.parametrize("seq_lens", [[64, 192], [31, 63, 93, 123]], ids=["two", "ragged"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_varlen(backend, variant, seq_lens):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=seq_lens))


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_zero_length_sequence(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, seq_lens=[64, 0, 128]))


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_initial_state(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128), use_initial_state=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_d_final_state(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128), use_initial_state=True, use_dfs=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_d_final_state_partial_chunk(backend, variant):
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=251), use_dfs=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_scale(backend, variant):
    """A non-default scale must reach the backward path (the engines carry an
    independent 1/sqrt(K) default that would mask a dropped plumb)."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128), scale=1.0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_qk_l2norm(backend, variant):
    """dQ/dK must include the in-kernel normalization's own backward."""
    assert_bwd_parity(backend, make_case(variant, torch.bfloat16, T=128), l2norm=True)


def test_bwd_no_decay_gate_grad_floor(backend):
    """GDN with alpha off: dGate/dBeta are cancelling-reduction noise floors;
    the data grads stay at full tolerance."""
    assert_bwd_parity(backend, make_case("gdn", torch.bfloat16, T=192, alpha=False), gate_grad_tol=0.3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_with_checkpoints(backend, variant):
    """The checkpoint dump is non-differentiable and must not block backward."""
    ckpt = CHUNK[variant]
    case = make_case(variant, torch.bfloat16, T=4 * ckpt)
    q_t = to_thd(case.q).detach().clone().requires_grad_(True)
    g_t = to_thd(case.gates["g"]).detach().clone().requires_grad_(True)
    args = [q_t, to_thd(case.k), to_thd(case.v), g_t, to_thd(case.gates["beta"])]
    if variant == "gdn2":
        args.append(to_thd(case.gates["w"]))
    with waive_unsupported(backend, variant):
        o, fs, state_checkpoints = pinned_op(backend, variant)(*args, case.cu, output_final_state=True, checkpoint_every_n_tokens=ckpt)
        assert not state_checkpoints.requires_grad
        (o.sum() + fs.sum()).backward()
    for name, t in (("q", q_t), ("g", g_t)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), f"bad grad for {name}"


# ---------------------------------------------------------------------------
# Checkpoints (per-chunk state series)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoints_match_prefix_final_states(backend, variant):
    """state_checkpoints[j] is the state after (j+1)*ckpt tokens, strictly before the
    end; rows are a shape-derived capacity bound, valid entries pack first."""
    ckpt = CHUNK[variant]
    T = 5 * ckpt
    case = make_case(variant, torch.bfloat16, T=T)
    o, fs, state_checkpoints = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=ckpt)
    valid = (T - 1) // ckpt
    assert state_checkpoints.shape == (T // ckpt, case.HO, case.K, case.V)
    assert state_checkpoints.dtype == case.dtype
    for j in (0, valid - 1):
        n = (j + 1) * ckpt
        args = [to_thd(case.q)[:n], to_thd(case.k)[:n], to_thd(case.v)[:n], to_thd(case.gates["g"])[:n], to_thd(case.gates["beta"])[:n]]
        if variant == "gdn2":
            args.append(to_thd(case.gates["w"])[:n])
        cu_n = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        with waive_unsupported(backend, variant):
            o, fs_p = pinned_op(backend, variant)(*args, cu_n, output_final_state=True)
        check(f"state_checkpoints[{j}]", state_checkpoints[j], fs_p[0], STATE_TOL[case.dtype])


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoints_varlen(backend, variant):
    """Entries pack per sequence in order (one per ckpt tokens strictly before
    each sequence end); each entry matches its sequence's truncated prefix."""
    ckpt = CHUNK[variant]
    seq_lens = [3 * ckpt + 5, ckpt - 1, 0, 2 * ckpt]
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens)
    o, fs, state_checkpoints = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=ckpt)
    counts = [max(sl - 1, 0) // ckpt for sl in seq_lens]
    # shape[0] is the shape-derived capacity bound; the packed prefix holds
    # sum(counts) real rows (per-sequence, in order), the tail is uninitialized
    assert state_checkpoints.shape[0] == max(sum(seq_lens) // ckpt, 1)
    bounds = case.cu.tolist()
    base = 0
    for n, cnt in enumerate(counts):
        for j in sorted({0, cnt - 1} if cnt else set()):
            n0 = bounds[n]
            ntok = (j + 1) * ckpt
            args = [to_thd(t)[n0 : n0 + ntok].clone() for t in (case.q, case.k, case.v, case.gates["g"], case.gates["beta"])]
            if variant == "gdn2":
                args.append(to_thd(case.gates["w"])[n0 : n0 + ntok].clone())
            cu_n = torch.tensor([0, ntok], dtype=torch.int32, device="cuda")
            with waive_unsupported(backend, variant):
                o_p, fs_p = pinned_op(backend, variant)(*args, cu_n, output_final_state=True)
            check(f"state_checkpoints[seq {n}][{j}]", state_checkpoints[base + j], fs_p[0], STATE_TOL[case.dtype])
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
    """Non-multiple varlen lengths where the packed entries fill the
    host-computable capacity exactly (sum over seqs of ceil(T/ckpt) - 1 ==
    total // ckpt): every entry lands in bounds and matches the fp64
    recurrence and its sequence's solo prefix run; the chunk-cadence bwd
    reuse matches the recompute path bitwise."""
    ckpt = CHUNK[variant]
    seq_lens = TIGHT_VARLEN_RECIPES[recipe](ckpt)
    counts = [max(sl - 1, 0) // ckpt for sl in seq_lens]
    assert sum(counts) == max(sum(seq_lens) // ckpt, 1), "recipe must fill the capacity bound exactly"
    case = make_case(variant, torch.bfloat16, seq_lens=seq_lens)
    o, fs, state_checkpoints = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=ckpt)
    assert state_checkpoints.shape[0] == sum(counts)
    ref_fn = {"gdn": gdn_reference, "kda": kda_reference, "gdn2": gdn2_reference}[variant]
    bounds = case.cu.tolist()
    base = 0
    for n, cnt in enumerate(counts):
        for j in range(cnt):
            n0 = bounds[n]
            ntok = (j + 1) * ckpt
            ref_args = [t[:, n0 : n0 + ntok] for t in (case.q, case.k, case.v, case.gates["g"], case.gates["beta"])]
            args = [to_thd(t)[n0 : n0 + ntok].clone() for t in (case.q, case.k, case.v, case.gates["g"], case.gates["beta"])]
            if variant == "gdn2":
                ref_args.append(case.gates["w"][:, n0 : n0 + ntok])
                args.append(to_thd(case.gates["w"])[n0 : n0 + ntok].clone())
            with torch.no_grad():
                _, fs_ref = ref_fn(*ref_args)
            check(f"state_checkpoints[seq {n}][{j}] vs fp64 reference", state_checkpoints[base + j], fs_ref[0], STATE_TOL[case.dtype])
            cu_n = torch.tensor([0, ntok], dtype=torch.int32, device="cuda")
            with waive_unsupported(backend, variant):
                o_p, fs_p = pinned_op(backend, variant)(*args, cu_n, output_final_state=True)
            check(f"state_checkpoints[seq {n}][{j}] vs solo prefix", state_checkpoints[base + j], fs_p[0], STATE_TOL[case.dtype])
        base += cnt
    grads_by_mode = []
    for mode_ckpt in (0, ckpt):
        leaves = [to_thd(case.q).detach().clone().requires_grad_(True), to_thd(case.k).detach().clone().requires_grad_(True)]
        args = [leaves[0], leaves[1], to_thd(case.v), to_thd(case.gates["g"]), to_thd(case.gates["beta"])]
        if variant == "gdn2":
            args.append(to_thd(case.gates["w"]))
        with waive_unsupported(backend, variant):
            out = pinned_op(backend, variant)(*args, case.cu, checkpoint_every_n_tokens=mode_ckpt)
            set_seed(SEED + 5)
            dO = torch.randn_like(out[0])
            grads_by_mode.append(torch.autograd.grad([out[0]], leaves, [dO]))
    for gr, gc in zip(grads_by_mode[0], grads_by_mode[1]):
        assert torch.equal(bits(gr), bits(gc)), "checkpoint-reuse grads differ from the recompute path"


@pytest.mark.parametrize("ckpt_mult", [2, 3])
@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoints_coarse_cadence(backend, variant, ckpt_mult):
    """Coarser cadences (multiples of the base chunk) keep the prefix contract."""
    ckpt = CHUNK[variant] * ckpt_mult
    T = 5 * ckpt
    case = make_case(variant, torch.bfloat16, T=T)
    o, fs, state_checkpoints = run_fwd(backend, case, output_final_state=True, checkpoint_every_n_tokens=ckpt)
    valid = (T - 1) // ckpt
    assert state_checkpoints.shape == (T // ckpt, case.HO, case.K, case.V)
    assert state_checkpoints.dtype == case.dtype
    for j in (0, valid - 1):
        n = (j + 1) * ckpt
        args = [to_thd(case.q)[:n], to_thd(case.k)[:n], to_thd(case.v)[:n], to_thd(case.gates["g"])[:n], to_thd(case.gates["beta"])[:n]]
        if variant == "gdn2":
            args.append(to_thd(case.gates["w"])[:n])
        cu_n = torch.tensor([0, n], dtype=torch.int32, device="cuda")
        with waive_unsupported(backend, variant):
            o, fs_p = pinned_op(backend, variant)(*args, cu_n, output_final_state=True)
        check(f"state_checkpoints[{j}]", state_checkpoints[j], fs_p[0], STATE_TOL[case.dtype])


# ---------------------------------------------------------------------------
# Raw-logit gate modes (safe gate, in-kernel Beta sigmoid)
# ---------------------------------------------------------------------------


def safe_gate_case(variant, T=256, H=2, K=128, V=128, seed=SEED + 7):
    case = make_case(variant, torch.bfloat16, T=T, H=H, K=K, V=V, seed=seed)
    set_seed(seed + 1)
    if variant == "gdn":
        # scalar per-head gate: -exp(a_log[h]) * softplus(g + dt_bias[h])
        graw = torch.randn(1, T, case.HO, device="cuda", dtype=torch.float32)
        dt_bias = torch.zeros(case.HO, dtype=torch.float32, device="cuda")
    else:
        graw = torch.randn(1, T, case.HO, K, device="cuda", dtype=torch.float32)
        dt_bias = torch.zeros(case.HO, K, dtype=torch.float32, device="cuda")
    a_log = torch.zeros(case.HO, dtype=torch.float32, device="cuda")
    return case, graw, a_log, dt_bias


@pytest.mark.parametrize("variant", VARIANTS)
def test_safe_gate_forward_parity(backend, variant):
    """Raw logits with a_log = 0 / dt_bias = 0 match the post-activation path
    fed the host-side transform (``lb * sigmoid(g)``, or ``-softplus(g)`` for
    GDN's scalar gate)."""
    lb = -5.0
    case, graw, a_log, dt_bias = safe_gate_case(variant)
    kw = dict(output_final_state=True, use_qk_l2norm_in_kernel=True)
    raw_kw = dict(kw, safe_gate=True, a_log=a_log, dt_bias=dt_bias)
    if variant != "gdn":
        raw_kw["gate_lower_bound"] = lb
    raw_gates = dict(case.gates, g=graw)
    if variant in ("gdn", "kda"):
        braw = torch.randn(1, case.T, case.HO, device="cuda").to(case.dtype)
        raw_gates["beta"] = braw
        raw_kw["use_beta_sigmoid_in_kernel"] = True
        eff_beta = braw.float().sigmoid()
    else:
        eff_beta = case.gates["beta"]
    g_eff = -F.softplus(graw) if variant == "gdn" else lb * torch.sigmoid(graw)
    raw_case = case.clone(gates=raw_gates)
    eff_case = case.clone(gates=dict(case.gates, g=g_eff, beta=eff_beta))
    o_raw, fs_raw = run_fwd(backend, raw_case, **raw_kw)
    o_eff, fs_eff = run_fwd(backend, eff_case, **kw)
    check("o", o_raw, o_eff.double(), 2e-2)
    assert rms_ratio(fs_raw, fs_eff) < 2e-2


@pytest.mark.parametrize("variant", VARIANTS)
def test_safe_gate_backward(backend, variant):
    """Fused-gate training: dG comes back in raw-logit space and the
    parameter gradients satisfy their exact identities over dG
    (d_dt_bias = sum dg_raw; d_a_log = sum dg_raw * (g + dt_bias) per-channel,
    or sum dg_raw * softplus(y) / sigmoid(y) for GDN's scalar gate)."""
    lb = -5.0
    case, graw, a_log, dt_bias = safe_gate_case(variant, T=128)
    set_seed(SEED + 9)
    a_leaf = (torch.randn_like(a_log) * 0.3).requires_grad_(True)
    dt_leaf = (torch.randn_like(dt_bias) * 0.3).requires_grad_(True)
    raw_gates = dict(case.gates, g=graw)
    kw = dict(safe_gate=True, a_log=a_leaf, dt_bias=dt_leaf, use_qk_l2norm_in_kernel=True)
    if variant != "gdn":
        kw["gate_lower_bound"] = lb
    if variant in ("gdn", "kda"):
        raw_gates["beta"] = torch.randn(1, case.T, case.HO, device="cuda").to(case.dtype)
        kw["use_beta_sigmoid_in_kernel"] = True
    raw_case = case.clone(gates=raw_gates)
    g_leaf = to_thd(raw_gates["g"]).detach().clone().requires_grad_(True)
    beta_leaf = to_thd(raw_gates["beta"]).detach().clone().requires_grad_(True)
    args = [to_thd(raw_case.q).detach().clone().requires_grad_(True), to_thd(raw_case.k), to_thd(raw_case.v), g_leaf, beta_leaf]
    if variant == "gdn2":
        args.append(to_thd(raw_gates["w"]))
    with waive_unsupported(backend, variant):
        o, _ = pinned_op(backend, variant)(*args, case.cu, **kw)
        o.sum().backward()
    dg_raw = g_leaf.grad.double()
    ddt_id = dg_raw.sum(0)
    if variant == "gdn":
        y = g_leaf.detach().double() + dt_leaf.detach().double()[None]
        da_id = (dg_raw * (F.softplus(y) / torch.sigmoid(y))).sum(0)
    else:
        da_id = (dg_raw * (g_leaf.detach().double() + dt_leaf.detach().double()[None])).sum(dim=(0, 2))
    for name, got, ident in (("d_dt_bias", dt_leaf.grad.double(), ddt_id), ("d_a_log", a_leaf.grad.double(), da_id)):
        scale = max(ident.abs().max().item(), 1e-6)
        assert (got - ident).abs().max().item() / scale < 1e-4, name
    for name, leaf in (("dq", args[0]), ("dbeta", beta_leaf)):
        assert leaf.grad is not None and bool(torch.isfinite(leaf.grad).all()), name


def test_beta_sigmoid_in_kernel(backend):
    """KDA: io-dtype Beta logits with the in-kernel sigmoid match the
    post-activation fp32 path."""
    case = make_case("kda", torch.bfloat16, T=256)
    set_seed(SEED + 11)
    braw = torch.randn(1, case.T, case.HO, device="cuda").to(case.dtype)
    raw_case = case.clone(gates=dict(case.gates, beta=braw))
    eff_case = case.clone(gates=dict(case.gates, beta=braw.float().sigmoid()))
    o_raw, fs_raw = run_fwd(backend, raw_case, output_final_state=True, use_beta_sigmoid_in_kernel=True)
    o_eff, fs_eff = run_fwd(backend, eff_case, output_final_state=True)
    check("o", o_raw, o_eff.double(), 2e-2)
    assert rms_ratio(fs_raw, fs_eff) < 2e-2


@pytest.mark.parametrize("variant", VARIANTS)
def test_beta_sigmoid_backward(backend, variant):
    """The in-kernel Beta sigmoid returns the gradient wrt the raw logit, so
    dbeta must equal the post-activation path's dbeta times s * (1 - s) at the
    io-rounded s the forward stores."""
    case = make_case(variant, torch.bfloat16, T=256)
    set_seed(SEED + 13)
    braw = torch.randn_like(case.gates["beta"].float()).to(case.dtype)
    s_io = torch.sigmoid(braw.float()).to(case.dtype)

    def dbeta(beta, **kw):
        leaf = to_thd(beta).detach().clone().requires_grad_(True)
        args = [to_thd(case.q), to_thd(case.k), to_thd(case.v), to_thd(case.gates["g"]), leaf]
        if variant == "gdn2":
            args.append(to_thd(case.gates["w"]))
        with waive_unsupported(backend, variant):
            o, _ = pinned_op(backend, variant)(*args, case.cu, **kw)
            o.sum().backward()
        return leaf.grad.double()

    got = dbeta(braw, use_beta_sigmoid_in_kernel=True)
    s = to_thd(s_io).double()
    ident = dbeta(s_io.to(case.gates["beta"].dtype)) * s * (1 - s)
    scale = ident.abs().max().item()
    assert scale > 1e-3, "dbeta is ~0, the comparison would be vacuous"
    assert (got - ident).abs().max().item() / scale < 2e-2


@pytest.mark.parametrize("H", (40, 160))
def test_scalar_gate_head_tiling(backend, H):
    """GDN's scalar gate-parameter reduction tiles heads; the dA_log /
    ddt_bias identities must hold past one tile and past 128 heads."""
    case, graw, a_log, dt_bias = safe_gate_case("gdn", T=128, H=H)
    set_seed(SEED + 15)
    a_leaf = (torch.randn_like(a_log) * 0.3).requires_grad_(True)
    dt_leaf = (torch.randn_like(dt_bias) * 0.3).requires_grad_(True)
    g_leaf = to_thd(graw).detach().clone().requires_grad_(True)
    args = [to_thd(case.q), to_thd(case.k), to_thd(case.v), g_leaf, to_thd(case.gates["beta"])]
    with waive_unsupported(backend, "gdn"):
        o, _ = pinned_op(backend, "gdn")(*args, case.cu, safe_gate=True, a_log=a_leaf, dt_bias=dt_leaf, use_qk_l2norm_in_kernel=True)
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
def test_invalid_rank_raises(variant):
    case = make_case(variant, torch.bfloat16, T=64)
    with pytest.raises(ValueError, match="THD"):
        op(variant)(case.q, to_thd(case.k), to_thd(case.v), *[to_thd(case.gates[n]) for n in case.gates], case.cu)


@pytest.mark.parametrize("variant", VARIANTS)
def test_invalid_qk_head_mismatch_raises(variant):
    case = make_case(variant, torch.bfloat16, T=64, H=2)
    args = [to_thd(case.q), to_thd(case.k)[:, :1].contiguous(), to_thd(case.v), to_thd(case.gates["g"]), to_thd(case.gates["beta"])]
    if variant == "gdn2":
        args.append(to_thd(case.gates["w"]))
    with pytest.raises(ValueError, match="head count"):
        op(variant)(*args, case.cu)


@pytest.mark.parametrize("variant", VARIANTS)
def test_invalid_gate_dtype_raises(variant):
    case = make_case(variant, torch.bfloat16, T=64)
    args = [to_thd(case.q), to_thd(case.k), to_thd(case.v), to_thd(case.gates["g"]).to(torch.bfloat16), to_thd(case.gates["beta"])]
    if variant == "gdn2":
        args.append(to_thd(case.gates["w"]))
    with pytest.raises(TypeError, match="must be"):
        op(variant)(*args, case.cu)


@pytest.mark.parametrize("variant", VARIANTS)
def test_invalid_initial_state_count_raises(variant):
    case = make_case(variant, torch.bfloat16, T=64)
    state0 = torch.zeros(3, case.HO, case.K, case.V, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="initial"):
        op(variant)(*op_args(case), initial_state=state0)


@pytest.mark.parametrize("variant", ["kda", "gdn2"])
def test_invalid_safe_gate_args_raise(variant):
    case = make_case(variant, torch.bfloat16, T=64)
    with pytest.raises(ValueError, match="safe_gate"):
        op(variant)(*op_args(case), safe_gate=True)
    with pytest.raises(ValueError, match="safe_gate"):
        op(variant)(*op_args(case), a_log=torch.zeros(case.HO, device="cuda"))


# ---------------------------------------------------------------------------
# Determinism (contract held by the FROST backend)
# ---------------------------------------------------------------------------


def bits(t):
    return t.contiguous().view(torch.uint8)


def assert_bitwise_runs(launch, repeats=DETERMINISM_REPEATS, label=""):
    """Back-to-back launches (single sync) must match run 0 bit for bit —
    barrier/fence races are timing-dependent, so there is no tolerance."""
    runs = [launch() for _ in range(repeats)]
    torch.cuda.synchronize()
    for out in runs[0]:
        assert torch.isfinite(out.float()).all(), f"{label}: non-finite output in run 0"
    for r, outs in enumerate(runs[1:], start=1):
        for i, (a, b) in enumerate(zip(runs[0], outs)):
            assert torch.equal(bits(a), bits(b)), f"{label}: output {i} differs between run 0 and run {r}"


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_fwd(backend, variant):
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    state0 = torch.randn(case.N, case.HO, case.K, case.V, device="cuda", dtype=torch.float32) * 0.05

    def launch():
        o, fs = run_fwd(backend, case, initial_state=state0, output_final_state=True)
        return o, fs

    assert_bitwise_runs(launch, label=f"{variant} fwd")


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_bwd(backend, variant):
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    leaves = [to_thd(case.q).detach().clone().requires_grad_(True), to_thd(case.k).detach().clone().requires_grad_(True)]
    args = [leaves[0], leaves[1], to_thd(case.v), to_thd(case.gates["g"]), to_thd(case.gates["beta"])]
    if variant == "gdn2":
        args.append(to_thd(case.gates["w"]))
    with waive_unsupported(backend, variant):
        o, fs = pinned_op(backend, variant)(*args, case.cu)
        dO = torch.randn_like(o)

        def launch():
            return torch.autograd.grad([o], leaves, [dO], retain_graph=True)

        assert_bitwise_runs(launch, label=f"{variant} bwd")


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_multi_tile_fwd(backend, variant):
    """Multi-tile grid (B*H >> SM count) with an initial state: bitwise
    stability across the inter-tile drain -> seed window."""
    case = make_case(variant, torch.bfloat16, B=8, T=192, H=64)
    state0 = torch.randn(case.N, case.HO, case.K, case.V, device="cuda", dtype=torch.float32) * 0.05

    def launch():
        o, fs = run_fwd(backend, case, initial_state=state0, output_final_state=True)
        return o, fs

    assert_bitwise_runs(launch, label=f"{variant} multi-tile fwd")


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_multi_tile_bwd(backend, variant):
    case = make_case(variant, torch.bfloat16, B=8, T=192, H=64)
    leaves = [to_thd(case.q).detach().clone().requires_grad_(True), to_thd(case.k).detach().clone().requires_grad_(True)]
    args = [leaves[0], leaves[1], to_thd(case.v), to_thd(case.gates["g"]), to_thd(case.gates["beta"])]
    if variant == "gdn2":
        args.append(to_thd(case.gates["w"]))
    with waive_unsupported(backend, variant):
        o, fs = pinned_op(backend, variant)(*args, case.cu)
        dO = torch.randn_like(o)

        def launch():
            return torch.autograd.grad([o], leaves, [dO], retain_graph=True)

        assert_bitwise_runs(launch, label=f"{variant} multi-tile bwd")


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism_two_streams(backend, variant):
    """Two concurrent instances on separate streams must not perturb each
    other: every repeat matches its own single-stream baseline."""
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
    """batch_invariant=True: each packed sequence matches its solo B = 1 run bitwise."""
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    o, fs = run_fwd(backend, case, output_final_state=True, batch_invariant=True)
    bounds = case.cu.tolist()
    for n in range(case.N):
        s, e = bounds[n], bounds[n + 1]
        if s == e:
            continue
        args = [to_thd(t)[s:e].clone() for t in (case.q, case.k, case.v, case.gates["g"], case.gates["beta"])]
        if variant == "gdn2":
            args.append(to_thd(case.gates["w"])[s:e].clone())
        cu1 = torch.tensor([0, e - s], dtype=torch.int32, device="cuda")
        with waive_unsupported(backend, variant):
            o_solo, fs_solo = pinned_op(backend, variant)(*args, cu1, output_final_state=True, batch_invariant=True)
        assert torch.equal(bits(o[s:e]), bits(o_solo)), f"seq {n}: packed o differs from solo"
        assert torch.equal(bits(fs[n]), bits(fs_solo[0])), f"seq {n}: packed final state differs from solo"


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_batch_invariance_bwd(backend, variant):
    """batch_invariant=True: a sequence's grads match its solo B = 1 run bitwise."""
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    leaves = [to_thd(case.q).detach().clone().requires_grad_(True), to_thd(case.k).detach().clone().requires_grad_(True)]
    args = [leaves[0], leaves[1], to_thd(case.v), to_thd(case.gates["g"]), to_thd(case.gates["beta"])]
    if variant == "gdn2":
        args.append(to_thd(case.gates["w"]))
    s, e = 0, 497
    with waive_unsupported(backend, variant):
        o, fs = pinned_op(backend, variant)(*args, case.cu, batch_invariant=True)
        dO = torch.randn_like(o)
        grads_packed = torch.autograd.grad([o], leaves, [dO], retain_graph=True)
        solo_args = [to_thd(t)[s:e].clone() for t in (case.q, case.k, case.v, case.gates["g"], case.gates["beta"])]
        if variant == "gdn2":
            solo_args.append(to_thd(case.gates["w"])[s:e].clone())
        solo_leaves = [solo_args[0].requires_grad_(True), solo_args[1].requires_grad_(True)]
        cu1 = torch.tensor([0, e - s], dtype=torch.int32, device="cuda")
        o_solo, fs_solo = pinned_op(backend, variant)(*solo_args, cu1, batch_invariant=True)
        grads_solo = torch.autograd.grad([o_solo], solo_leaves, [dO[s:e].clone()])
    for gp, gs in zip(grads_packed, grads_solo):
        assert torch.equal(bits(gp[s:e]), bits(gs)), "packed grad slice differs from solo grad"


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_bwd_checkpoint_reuse(backend, variant):
    """Training with chunk-cadence checkpoints: the bwd consumes the fwd's
    series instead of recomputing it, and the grads match bitwise."""
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    grads_by_mode = []
    for ckpt in (0, CHUNK[variant]):
        leaves = [to_thd(case.q).detach().clone().requires_grad_(True), to_thd(case.k).detach().clone().requires_grad_(True)]
        args = [leaves[0], leaves[1], to_thd(case.v), to_thd(case.gates["g"]), to_thd(case.gates["beta"])]
        if variant == "gdn2":
            args.append(to_thd(case.gates["w"]))
        with waive_unsupported(backend, variant):
            out = pinned_op(backend, variant)(*args, case.cu, checkpoint_every_n_tokens=ckpt)
            o = out[0]
            set_seed(SEED + 5)
            dO = torch.randn_like(o)
            grads_by_mode.append(torch.autograd.grad([o], leaves, [dO]))
    for gr, gc in zip(grads_by_mode[0], grads_by_mode[1]):
        assert torch.equal(bits(gr), bits(gc)), "checkpoint-reuse grads differ from the recompute path"


@pytest.mark.parametrize("backend", ["cutile"], indirect=True)
@pytest.mark.parametrize("variant", ["gdn", "kda"])
def test_batch_invariance_cutile(backend, variant):
    """cuTile is batch-invariant by construction; the flag must hold there too."""
    case = make_case(variant, torch.bfloat16, seq_lens=[497, 16, 1, 480, 0, 253])
    o, fs = run_fwd(backend, case, output_final_state=True, batch_invariant=True)
    bounds = case.cu.tolist()
    for n in range(case.N):
        s, e = bounds[n], bounds[n + 1]
        if s == e:
            continue
        args = [to_thd(t)[s:e].clone() for t in (case.q, case.k, case.v, case.gates["g"], case.gates["beta"])]
        cu1 = torch.tensor([0, e - s], dtype=torch.int32, device="cuda")
        with waive_unsupported(backend, variant):
            o_solo, fs_solo = pinned_op(backend, variant)(*args, cu1, output_final_state=True, batch_invariant=True)
        assert torch.equal(bits(o[s:e]), bits(o_solo)), f"seq {n}: packed o differs from solo"
        assert torch.equal(bits(fs[n]), bits(fs_solo[0])), f"seq {n}: packed final state differs from solo"


@pytest.mark.parametrize("backend", ["frost"], indirect=True)
@pytest.mark.parametrize("variant", VARIANTS)
def test_batch_invariance_with_coarse_checkpoints(backend, variant):
    """batch_invariant=True composes with a coarser checkpoint cadence."""
    ckpt = CHUNK[variant] * 2
    T = 4 * ckpt
    case = make_case(variant, torch.bfloat16, T=T)
    o, fs, state_checkpoints = run_fwd(backend, case, output_final_state=True, batch_invariant=True, checkpoint_every_n_tokens=ckpt)
    assert state_checkpoints.shape == (T // ckpt, case.HO, case.K, case.V)
    n = ckpt
    args = [to_thd(case.q)[:n], to_thd(case.k)[:n], to_thd(case.v)[:n], to_thd(case.gates["g"])[:n], to_thd(case.gates["beta"])[:n]]
    if variant == "gdn2":
        args.append(to_thd(case.gates["w"])[:n])
    cu_n = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    with waive_unsupported(backend, variant):
        o_p, fs_p = pinned_op(backend, variant)(*args, cu_n, output_final_state=True)
    check("state_checkpoints[0]", state_checkpoints[0], fs_p[0], STATE_TOL[case.dtype])


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
