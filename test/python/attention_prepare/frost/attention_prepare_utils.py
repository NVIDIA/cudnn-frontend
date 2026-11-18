# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the attention preparation decode step's test modules: the form tables, the operand builders (drawn on the
HOST), the independent reference and the device gate.  Not a test module (no ``test_`` prefix); imported by name from the sibling
modules (pytest puts their directory on ``sys.path``).

THE OPERANDS are built the way a decode step holds them: ``q``, ``gate``, ``k`` and ``v`` are four VIEWS of one projection slab
``[tokens, HEADS x 2 D + KV_HEADS x 2 D]`` (per query head its row and its gate row side by side, then the key rows, then the value
rows), and ``k_pool`` / ``v_pool`` are the two halves ``[..., :D]`` / ``[..., D:]`` of one packed cache
``[pages, KV_HEADS, page_size, 2 D]`` -- nothing is contiguous but the last axis, and the two pools interleave in one allocation.

THE REFERENCE (:func:`twin`) is a PyTorch chain on the host with the kernel's ROUNDING POINTS -- fp32 products, one rounding of the
normalized value to bf16, one rounding of each rotated value to bf16 -- and none of its approximate instructions: the mean and the
root are evaluated in fp64 and rounded to fp32 once.  The kernel's root is approximate (and its sum has its own association), so
its fp32 normalized value differs from the twin's by a few parts in 10^7 and its bf16 value by at most ONE bf16 unit (a unit is
2^-8 of the value).  The statements (:func:`assert_twin`), each naming its unit -- the bf16 unit at the LARGER of the two magnitudes:

1. *a pass-through element* (index at or past ``ROTARY_DIM``): ``|out - ref| <= unit(out, ref)``;
2. *a rotated element*: ``|out - ref| <= unit(out, ref) + |cos| unit(n1) + |sin| unit(n2)`` with ``n1`` / ``n2`` the twin's two
   normalized operands of that element (each may be one unit off in the kernel) and their units taken one unit up;
3. *the gate rows, the value rows and every pool row no live slot names*: BITWISE (copies, and bytes the launch must not touch).

Nothing here is a tolerance on a product or a root: a rotated element outside its band is a finding with its magnitude."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.attention_prepare.frost import prepare_decode as entry  # noqa: E402
from cudnn.attention_prepare.frost.kernel import prepare_decode_config as kcfg  # noqa: E402

GEOMETRY = kcfg.AttentionPrepareGeometry(24, 4, 256, 64)
HEADS, KV_HEADS, D, ROTARY = GEOMETRY.HEADS, GEOMETRY.KV_HEADS, GEOMETRY.HEAD_DIM, GEOMETRY.ROTARY_DIM
HALF = ROTARY // 2
SLAB = HEADS * 2 * D + KV_HEADS * 2 * D  # 14,336: per query head [q | gate], then the key rows, then the value rows
EPS = 1e-6
SECTION = (11, 11, 10)
BF16, I16 = torch.bfloat16, torch.int16
INT = {"int32": torch.int32, "int64": torch.int64}
TABLE_ROWS = 4096
#: the canary a pool is filled with before a launch: a bf16 NaN pattern no written row can hold
CANARY_BITS = 0x7FC5
#: the sentinel of a fresh output buffer
SENTINEL_BITS = 0x7FC0

#: the forms a suite runs as ``id -> (positions, page_size, position_dtype, slot_dtype)``; the first one is the form a decode step
#: with three position rows and a 64-row page runs
FORMS = {
    "3d-p64-i64-i64": ("3d", 64, "int64", "int64"),
    "1d-p64-i64-i64": ("1d", 64, "int64", "int64"),
    "3d-p16-i32-i32": ("3d", 16, "int32", "int32"),
    "1d-p16-i64-i32": ("1d", 16, "int64", "int32"),
    "3d-p64-i32-i64": ("3d", 64, "int32", "int64"),
}
DEFAULT_FORM = "3d-p64-i64-i64"
#: the served launch arms as ``(pdl, pdl_release)``: off; wait only; release + wait
LAUNCH_ARMS = ((None, None), (True, False), (True, None))
LAUNCH_ARM_IDS = ("plain", "wait", "release-wait")


def form_kwargs(form):
    """The constructor keywords of a form id."""
    positions, page, position_dtype, slot_dtype = FORMS[form]
    return dict(positions=positions, mrope_section=SECTION if positions == "3d" else None, page_size=page, position_dtype=position_dtype, slot_dtype=slot_dtype)


def make_entry(device, form=DEFAULT_FORM, arm=(None, None), eps=EPS, geometry=GEOMETRY):
    return entry.AttentionPrepareDecode(device, geometry, eps, pdl=arm[0], pdl_release=arm[1], **form_kwargs(form))


# ---------------------------------------------------------------------------------------------------------------------------
# operands (drawn on the host: one seed gives one set of tensors on every GPU)
# ---------------------------------------------------------------------------------------------------------------------------
def rotary_table(rows=TABLE_ROWS):
    """A bf16 ``[rows, ROTARY_DIM]`` table: per position ``HALF`` cosines, then ``HALF`` sines, of ``position x 10000^(-r / HALF)``."""
    position = torch.arange(rows, dtype=torch.float64).reshape(-1, 1)
    frequency = 10000.0 ** (-torch.arange(HALF, dtype=torch.float64) / HALF).reshape(1, -1)
    angle = position * frequency
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=1).float().bfloat16()


def canary_cache(pages, page_size):
    """A packed cache ``[pages, KV_HEADS, page_size, 2 D]`` (K rows in the first half of the last axis, V rows in the second) filled
    with the canary pattern."""
    return torch.full((pages, KV_HEADS, page_size, 2 * D), CANARY_BITS, dtype=I16).view(BF16)


def make_operands(tokens, form=DEFAULT_FORM, device="cpu", seed=0, kind="random", pages=None, slots=None, table_rows=TABLE_ROWS):
    """The operands of one call as a dict, NaN-free and finite, from a CPU generator and moved to ``device``.

    ``slab`` ``[tokens, 14336]`` with ``q`` / ``gate`` / ``k`` / ``v`` its four views; ``q_weight`` / ``k_weight`` ``[D]``;
    ``cos_sin`` the rotary table; ``positions`` of the form's dtype -- ``[tokens]``, or ``[3, tokens]`` cut out of a wider tensor so
    that its row stride is not its length; ``cache`` the packed cache with ``k_pool`` / ``v_pool`` its two halves; ``slot_mapping`` of
    the form's dtype -- ``slots`` (a list, any length up to ``tokens``; ``None``: one distinct slot per token, in a shuffled order).

    ``kind``: ``random`` (``x ~ 3 N(0, 1)``, weights ``~ 0.3 N(0, 1)``); ``structured`` (the same draw with a few large entries per
    row, zero stretches, alternating-sign runs, negative zeros, and weights of ``-1`` and below); ``edge`` (token 0 all zeros, token 1
    negative zeros, token 2 one dominant element per head row, token 3 values whose squares underflow, token 4 large magnitudes; the
    other tokens random)."""
    positions_form, page_size, position_dtype, slot_dtype = FORMS[form]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    slab = (torch.randn(tokens, SLAB, generator=gen) * 3).bfloat16()
    q_weight = (torch.randn(D, generator=gen) * 0.3).bfloat16()
    k_weight = (torch.randn(D, generator=gen) * 0.3).bfloat16()
    if kind == "structured":
        big = torch.randint(0, SLAB, (tokens, 16), generator=gen)
        slab.scatter_(1, big, torch.tensor([[300.0, -700.0, 1500.0, -40.0]]).bfloat16().repeat(tokens, 4))
        slab[:, 64:96] = 0.0
        slab[:, 2 * D + 8 : 2 * D + 40] = torch.tensor([1.0, -1.0]).bfloat16().repeat(16)
        slab[:, HEADS * 2 * D + 200 : HEADS * 2 * D + 208] = -0.0
        q_weight[100:116] = -1.0  # 1 + w == 0: a stretch of zero outputs
        q_weight[4:8] = -2.5  # a negative (1 + w) inside the rotary part
        k_weight[40:48] = -1.0
        k_weight[200:204] = -3.0
    elif kind == "edge":
        assert tokens >= 5
        slab[0] = 0.0
        slab[1] = -0.0
        slab[2] = 0.0
        slab[2, ::D] = 3.0e4  # one dominant element per head row
        slab[3] = (slab[3].float() * 1e-25).bfloat16()
        slab[4] = (slab[4].float() * 1e15).bfloat16()
    elif kind != "random":
        raise ValueError(kind)
    assert bool(torch.isfinite(slab.float()).all())
    table = rotary_table(table_rows)
    limit = min(table_rows, 2000)
    if positions_form == "3d":
        wide = torch.randint(0, limit, (3, tokens + 17), generator=gen).to(INT[position_dtype])
        positions = wide[:, :tokens]  # row stride tokens + 17 elements: free, not the row length
    else:
        positions = torch.randint(0, limit, (tokens,), generator=gen).to(INT[position_dtype])
    if pages is None:
        pages = max(2, (tokens + page_size - 1) // page_size + 1)
    cache = canary_cache(pages, page_size)
    if slots is None:
        slots = torch.randperm(pages * page_size, generator=gen)[:tokens].tolist()
    slot_mapping = torch.tensor(list(slots), dtype=INT[slot_dtype])
    ops = dict(slab=slab, q_weight=q_weight, k_weight=k_weight, cos_sin=table, positions=positions, cache=cache, slot_mapping=slot_mapping)
    return bind({name: t.to(device) for name, t in ops.items()}, form)


def bind(ops, form):
    """``ops`` with the views a call takes -- ``q`` / ``gate`` / ``k`` / ``v`` of the slab, ``k_pool`` / ``v_pool`` of the cache -- and
    the form id."""
    slab, cache = ops["slab"], ops["cache"]
    tokens = slab.shape[0]
    pairs = slab[:, : HEADS * 2 * D].view(tokens, HEADS, 2 * D)
    ops.update(
        q=pairs[..., :D],
        gate=pairs[..., D:],
        k=slab[:, HEADS * 2 * D : HEADS * 2 * D + KV_HEADS * D].view(tokens, KV_HEADS, D),
        v=slab[:, HEADS * 2 * D + KV_HEADS * D :].view(tokens, KV_HEADS, D),
        k_pool=cache[..., :D],
        v_pool=cache[..., D:],
        form=form,
    )
    return ops


#: the table rows :func:`underflow_operands` fills with bf16 subnormals; every position of its call names one of them
UNDERFLOW_TABLE_ROWS = (4, 5, 6, 7)


def _subnormals(shape, gen):
    """bf16 SUBNORMALS of ``shape``: a zero exponent field, a random sign and a random non-zero mantissa."""
    raw = torch.randint(0, 1 << 16, shape, generator=gen, dtype=torch.int32)
    pattern = (raw & 0x807F) | 0x0001
    return ((pattern ^ 0x8000) - 0x8000).to(I16).view(BF16)


def underflow_operands(tokens, form=DEFAULT_FORM, device="cpu", seed=0):
    """Operands whose rotary PRODUCTS UNDERFLOW, for the cell that pins which product of a rotation is the fused one.  Every head row
    is bf16 subnormals (random signs) and every position of the call names a table row of bf16 subnormals (under three position
    rows each row names its own).  The weights keep ``|1 + weight|`` between 1 / 8 and 2, with a stretch of ``-2.5`` (a negative
    gain) in each rotary half.  Then: a row's sum of squares underflows to zero, so its root is the root of eps alone; every
    normalized element is a normal fp32 value whose SIGN is ``sign(x) ^ sign(1 + weight)``; every rotary product lies far below the
    smallest fp32 subnormal; and each rotated value is a ZERO with the sign of its FUSED product -- the rounded product is a zero
    and cannot change the sign of a non-zero exact sum (:func:`underflow_rotary_bits`)."""
    ops = make_operands(tokens, form, device="cpu", seed=seed)
    gen = torch.Generator(device="cpu").manual_seed(seed + 7919)
    ops["slab"][:] = _subnormals((tokens, SLAB), gen)
    for name in ("q_weight", "k_weight"):
        ops[name][:] = ops[name].float().clamp(-0.875, 0.875).bfloat16()
        ops[name][4:8] = -2.5
        ops[name][HALF + 8 : HALF + 12] = -2.5
    first, count = UNDERFLOW_TABLE_ROWS[0], len(UNDERFLOW_TABLE_ROWS)
    ops["cos_sin"][first : first + count] = _subnormals((count, ROTARY), gen)
    token = torch.arange(tokens)
    if FORMS[form][0] == "3d":
        for row in range(3):
            ops["positions"][row] = (first + (token + row) % count).to(ops["positions"].dtype)
    else:
        ops["positions"][:] = (first + token % count).to(ops["positions"].dtype)
    names = ("slab", "q_weight", "k_weight", "cos_sin", "positions", "cache", "slot_mapping")
    return bind({name: ops[name].to(device) for name in names}, form)


CALL_ORDER = ("q", "gate", "k", "v", "q_weight", "k_weight", "cos_sin", "positions", "k_pool", "v_pool", "slot_mapping")


def call_args(ops):
    """The eleven positional operands of a call, in call order."""
    return [ops[name] for name in CALL_ORDER]


def fresh_outputs(tokens, device, extra=0):
    """Caller-owned ``q_out`` / ``gate_out`` ``[tokens + extra, HEADS, D]`` filled with the sentinel."""
    make = lambda: torch.full((tokens + extra, HEADS, D), SENTINEL_BITS, dtype=I16, device=device).view(BF16)  # noqa: E731
    return dict(q_out=make(), gate_out=make())


# ---------------------------------------------------------------------------------------------------------------------------
# the reference (host torch math on copies; every function returns host tensors)
# ---------------------------------------------------------------------------------------------------------------------------
def bits(t):
    """The bit patterns of a bf16 tensor as int32 in ``0 .. 0xFFFF`` (a signed zero is its own value)."""
    return t.detach().cpu().contiguous().view(I16).to(torch.int32) & 0xFFFF


def unit(value):
    """The bf16 unit in the last place at the magnitude of ``value`` (fp64 tensor): ``2^(e - 7)`` for ``2^e <= |value| < 2^(e + 1)``,
    the smallest normal's unit below it.  Exact: built from ``frexp``'s exponent."""
    _, exponent = torch.frexp(value.abs().double())
    return torch.ldexp(torch.ones_like(value, dtype=torch.float64), (exponent.clamp(min=-125) - 8))


def position_rows(cfg):
    """Host long ``[HALF]``: the position row each rotary index reads (``kcfg.position_row_of``, the host twin of the kernel's select)."""
    return torch.tensor([kcfg.position_row_of(cfg, r) for r in range(cfg.HALF)], dtype=torch.long)


def twin_rows(x, weight, positions, table, cfg):
    """The twin on head rows ``x`` ``[tokens, heads, D]`` bf16 with one ``weight`` ``[D]``: ``(out, band)`` host tensors
    ``[tokens, heads, D]`` -- the bf16 reference and the fp64 half-width of each element's band (module docstring, statements 1, 2)."""
    xf = x.detach().cpu().float()
    eps32 = float(torch.tensor(cfg.EPS, dtype=torch.float32).double())
    mean = (xf.double() ** 2).sum(dim=-1, keepdim=True) / cfg.HEAD_DIM
    inv = (1.0 / torch.sqrt(mean + eps32)).float()  # the root in fp64, rounded to fp32 once
    gain = weight.detach().cpu().float().reshape(1, 1, -1) + 1.0
    normed = ((xf * inv) * gain).bfloat16()  # fp32 products, ONE rounding to bf16
    n = normed.double()
    pos = positions.detach().cpu().long().reshape(-1, xf.shape[0])  # [1 or 3, tokens]
    rows = position_rows(cfg)  # [HALF]
    selected = pos[rows].t()  # [tokens, HALF]: the position of rotary index r for each token
    tab = table.detach().cpu().float()
    column = torch.arange(cfg.HALF).reshape(1, -1)
    cos = tab[selected, column].reshape(xf.shape[0], 1, cfg.HALF)  # table[position of (token, r), r]
    sin = tab[selected, column + cfg.HALF].reshape(xf.shape[0], 1, cfg.HALF)
    n1, n2 = normed[..., : cfg.HALF].float(), normed[..., cfg.HALF : cfg.ROTARY_DIM].float()
    first = (n1.double() * cos.double() - (n2 * sin).double()).float().bfloat16()  # one fp32 product, the fused sum, one bf16 rounding
    if cfg.POSITIONS == "3d":  # which product of the second half is the fused one follows the position form (the kernel's step 10)
        second = (n2.double() * cos.double() + (n1 * sin).double()).float().bfloat16()
    else:
        second = (n1.double() * sin.double() + (n2 * cos).double()).float().bfloat16()
    out = normed.clone()
    out[..., : cfg.HALF] = first
    out[..., cfg.HALF : cfg.ROTARY_DIM] = second
    band = torch.zeros_like(n)  # the operands' share; the result's own unit is added at the comparison
    up = lambda v: unit(v.abs() + unit(v))  # noqa: E731  (the unit one unit up: the kernel's operand may be the neighbour)
    spread = cos.double().abs() * up(n[..., : cfg.HALF]) + sin.double().abs() * up(n[..., cfg.HALF : cfg.ROTARY_DIM])
    band[..., : cfg.HALF] = spread
    band[..., cfg.HALF : cfg.ROTARY_DIM] = spread
    return out, band


def twin(ops, cfg):
    """The reference of one call on host copies: ``q_out`` / ``k_rows`` with their bands, ``gate_out`` and ``v_rows`` (bit copies)."""
    q_ref, q_band = twin_rows(ops["q"], ops["q_weight"], ops["positions"], ops["cos_sin"], cfg)
    k_ref, k_band = twin_rows(ops["k"], ops["k_weight"], ops["positions"], ops["cos_sin"], cfg)
    return dict(q_out=q_ref, q_band=q_band, k_rows=k_ref, k_band=k_band, gate_out=ops["gate"].detach().cpu().clone(), v_rows=ops["v"].detach().cpu().clone())


def underflow_rotary_bits(x, weight, positions, table, cfg, positions_form=None):
    """Host int32 ``[tokens, heads, ROTARY_DIM]``: the bit patterns (``0x0000`` or ``0x8000``) of the rotated values of head rows
    ``x`` of :func:`underflow_operands`, by SIGN arithmetic alone -- no product and no root is evaluated.  ``sign(n) = sign(x) ^
    sign(1 + weight)``.  First half: ``fma(n[r], cos, -(n[r + HALF] sin))`` -> the sign of ``n[r] cos``.  Second half: the sign of
    its FUSED product -- ``n[r + HALF] cos`` under ``"3d"``, ``n[r] sin`` under ``"1d"``.  ``positions_form`` overrides the config's
    position form in that last step only: the bits the OTHER order would write."""
    order = cfg.POSITIONS if positions_form is None else positions_form
    negative = lambda t: (bits(t) >> 15).bool()  # noqa: E731
    gain = weight.detach().cpu().float() + 1.0  # exact: a bf16 weight of magnitude below 4
    assert bool((gain.abs() >= 0.125).all()), "the cell's weights keep the gain away from zero"
    sign_n = negative(x) ^ (gain < 0).reshape(1, 1, -1)  # [tokens, heads, D]
    pos = positions.detach().cpu().long().reshape(-1, x.shape[0])  # [1 or 3, tokens]
    selected = pos[position_rows(cfg)].t()  # [tokens, HALF]: the position of rotary index r for each token
    sign_table = negative(table)
    column = torch.arange(cfg.HALF).reshape(1, -1)
    sign_cos = sign_table[selected, column].reshape(x.shape[0], 1, cfg.HALF)
    sign_sin = sign_table[selected, column + cfg.HALF].reshape(x.shape[0], 1, cfg.HALF)
    n1, n2 = sign_n[..., : cfg.HALF], sign_n[..., cfg.HALF : cfg.ROTARY_DIM]
    first = n1 ^ sign_cos
    second = (n2 ^ sign_cos) if order == "3d" else (n1 ^ sign_sin)
    return torch.cat([first, second], dim=-1).to(torch.int32) << 15


def assert_band(got, ref, band, label):
    """Statements 1 and 2 on one bf16 tensor: ``|got - ref| <= unit(larger) + band`` element by element; returns the count of
    elements that are not bitwise the reference (the elements the approximate root moved by a unit)."""
    g, r = got.detach().cpu().double(), ref.double()
    assert bool(torch.isfinite(g).all()), f"{label}: a non-finite element"
    limit = unit(torch.maximum(g.abs(), r.abs())) + band
    outside = (g - r).abs() > limit
    count = int(outside.sum())
    if count:
        at = int(outside.reshape(-1).nonzero()[0])
        raise AssertionError(
            f"{label}: {count} of {g.numel()} elements outside the band of the twin (first at flat index {at}: got "
            f"{float(g.reshape(-1)[at])!r}, twin {float(r.reshape(-1)[at])!r}, band {float(limit.reshape(-1)[at])!r})"
        )
    return int((bits(got) != bits(ref)).sum())


def live_slots(slots, tokens, pool_slots):
    """``{token: slot}`` of the tokens that write a pool row: inside the slot vector, with ``0 <= slot < pool_slots``."""
    values = slots.detach().cpu().tolist()
    return {t: int(s) for t, s in enumerate(values[:tokens]) if 0 <= int(s) < pool_slots}


def pool_rows(cache, slots, page_size):
    """The K rows and the V rows ``[len(slots), KV_HEADS, D]`` of a packed cache at flat ``slots`` (host tensors)."""
    c = cache.detach().cpu()
    index = torch.tensor(list(slots), dtype=torch.long)
    rows = c[index // page_size, :, index % page_size]  # [n, KV_HEADS, 2 D]
    return rows[..., :D], rows[..., D:]


def assert_twin(ops, got, cfg, label="", cache_before=None):
    """The reference on ONE launch's bytes: ``got`` holds ``q_out`` / ``gate_out`` (the first ``tokens`` rows are compared, any rows
    behind them must hold the sentinel) and the launch has written ``ops["cache"]``.  ``cache_before`` is the cache as it was before
    the launch (default: all canary).  Returns the count of elements that differ from the twin by a unit (inside their band)."""
    tokens = ops["q"].shape[0]
    ref = twin(ops, cfg)
    moved = assert_band(got["q_out"][:tokens], ref["q_out"], ref["q_band"], f"{label}: q_out")
    assert torch.equal(bits(got["gate_out"][:tokens]), bits(ref["gate_out"])), f"{label}: gate_out is not the gate rows, bitwise"
    for name in ("q_out", "gate_out"):
        assert bool((bits(got[name][tokens:]) == SENTINEL_BITS).all()), f"{label}: {name} was written past the call's tokens"
    page_size = cfg.PAGE_SIZE
    cache = ops["cache"].detach().cpu()
    pool_slots = cache.shape[0] * page_size
    live = live_slots(ops["slot_mapping"], tokens, pool_slots)
    assert len(set(live.values())) == len(live), f"{label}: the cell hands two tokens one slot (a caller-contract violation of the cell itself)"
    before = canary_cache(cache.shape[0], page_size) if cache_before is None else cache_before.detach().cpu()
    untouched = torch.ones(cache.shape[0], page_size, dtype=torch.bool)
    if live:
        order = sorted(live)
        slots = [live[t] for t in order]
        index = torch.tensor(slots, dtype=torch.long)
        untouched[index // page_size, index % page_size] = False
        k_got, v_got = pool_rows(cache, slots, page_size)
        take = torch.tensor(order, dtype=torch.long)
        moved += assert_band(k_got, ref["k_rows"][take], ref["k_band"][take], f"{label}: the K pool rows")
        assert torch.equal(bits(v_got), bits(ref["v_rows"][take])), f"{label}: the V pool rows are not the value rows, bitwise"
    same = bits(cache.permute(0, 2, 1, 3)[untouched]) == bits(before.permute(0, 2, 1, 3)[untouched])
    assert bool(same.all()), f"{label}: {int((~same).sum())} bf16 elements of pool rows no live slot names were written"
    return moved


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the attention preparation decode step serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry._dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    return dev
