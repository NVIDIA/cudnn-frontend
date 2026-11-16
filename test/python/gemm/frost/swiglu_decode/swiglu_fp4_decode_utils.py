# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the SwiGLU FP4 decode entry's test modules: the FP4 / E4M3 code tables, a host quantiser that builds valid
block-scaled operands, the scale-atom order, the operand builders (random, structured, exact-integer cases), the fp64 reference with
its derived bound, raw-byte compares and the device gate.  Not a test module (no ``test_`` prefix); imported by name from the sibling
modules (pytest puts their directory on ``sys.path``).

THE REFERENCE AND ITS BOUND (never tuned to make a cell pass; a violation is a finding).  In fp64 on the exactly dequantised operands,
with ``K`` = the hidden size::

    g* = alpha * sum_k act_k * Wg_k        S_g = |alpha| * sum_k |act_k * Wg_k|        (u*, S_u likewise)        v* = silu(g*) * u*
    d_g = 2^-8 |g*| + (1 + 2^-8) (K + 1) 2^-23 S_g            one bf16 rounding of an fp32 sum of K exact products (d_u likewise)
    A   = 1.1 d_g (|u*| + d_u) + |silu(g*)| d_u               SiLU is 1.1-Lipschitz
    E_v = A + (2^-8 + 2^-19) (|v*| + A) + 2^-120              the product's bf16 rounding; 2^-19 covers the approximate exponential,
                                                              the approximate division and the two approximate reciprocals
    scale byte of a block, amax* = max |v*| and E_a = max E_v over its 16 channels (rounding is monotone: an INTERVAL of bytes):
        E4M3(gs * max(0, amax* - E_a) / 6 * (1 - 2^-21))  <=  byte  <=  E4M3(gs * (amax* + E_a) / 6 * (1 + 2^-21))
    codes, with s = value(the KERNEL's byte) / gs and h(c) = 0.25 for |c| < 2, 0.5 for 2 <= |c| < 4, 1 for |c| >= 4:
        |code * s - v*|  <=  h(code) * s + max(0, |v*| + E_v - 6 s) + E_v

The ``max`` term is the clip, exact for every scale including 0 (the zero rule: the block decodes to 0) and 448 (saturation).  What
the bound cannot see: a 4-bit output hides a one-unit bf16 defect behind ``h(code) * s`` -- that is what the exact cells are for.
"""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.gemm.frost.swiglu_decode import swiglu_fp4_decode as entry  # noqa: E402
from cudnn.gemm.frost.swiglu_decode.kernel import swiglu_fp4_decode_config as kcfg  # noqa: E402

G_DEFAULT = kcfg.Geometry(5120, 17408)
G_SMALL = kcfg.Geometry(5120, 128)
E4M3 = torch.float8_e4m3fn
E4M3_NAN = 0x7F  # the positive E4M3 NaN; 0xFF is the negative one
E4M3_MAX = 0x7E  # +448, the largest finite E4M3 value
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # the low three bits of a code; bit 3 is the sign
SF_VEC = kcfg.SF_VEC
ATOM_ROWS = kcfg.SF_ATOM_ROWS


# ---------------------------------------------------------------------------------------------------------------------------
# codes and layouts (host arithmetic; independent of the kernel)
# ---------------------------------------------------------------------------------------------------------------------------
def e2m1_table(device="cpu", dtype=torch.float64):
    mags = torch.tensor(E2M1_MAGNITUDES, dtype=dtype, device=device)
    return torch.cat([mags, -mags])


def unpack_fp4(packed, dtype=torch.float64):
    """``[R, K / 2]`` bytes -> ``[R, K]`` values: the low nibble is the even column."""
    table = e2m1_table(packed.device, dtype)
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    return torch.stack([table[lo], table[hi]], dim=-1).reshape(packed.shape[0], -1)


def e4m3_values(codes, dtype=torch.float64):
    """E4M3 bytes as values (exact)."""
    return codes.contiguous().view(E4M3).float().to(dtype)


def encode_e2m1(q):
    """Nearest FP4 code of each value (ties toward the smaller magnitude; saturating at 6): any valid code serves as an INPUT."""
    edges = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=q.dtype, device=q.device)
    idx = torch.bucketize(q.abs(), edges, right=False)
    return (idx | ((q < 0) & (idx > 0)).long() * 8).to(torch.uint8)


def pack_fp4(codes):
    """``[R, K]`` codes (0 .. 15) -> ``[R, K / 2]`` bytes, low nibble = even column."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def quantize_nvfp4(x, chunk=4096):
    """A valid block-scaled FP4 encoding of ``x`` ``[R, K]`` (fp32): per block of 16 columns the scale byte is E4M3(amax / 6) and the
    codes the nearest FP4 values of ``x / value(byte)``.  Returns ``(packed [R, K / 2] uint8, scales [R, K / 16] uint8)`` -- the
    scales in LINEAR order (:func:`to_atoms` gives the kernel's).  Host-style torch arithmetic on ``x``'s device."""
    packed, scales = [], []
    for start in range(0, x.shape[0], chunk):
        part = x[start : start + chunk].float()
        blocks = part.reshape(part.shape[0], -1, SF_VEC)
        byte = (blocks.abs().amax(dim=-1) / 6.0).to(E4M3)
        value = byte.float()
        q = torch.where(value[..., None] > 0, blocks / value[..., None].clamp(min=1e-30), torch.zeros_like(blocks))
        packed.append(pack_fp4(encode_e2m1(q).reshape(part.shape[0], -1)))
        scales.append(byte.view(torch.uint8))
    return torch.cat(packed), torch.cat(scales)


def to_atoms(linear):
    """``[R, B]`` scale bytes (row, block) -> the flat 128 x 4 scale-atom order of the rows padded to a multiple of 128 (pad bytes 0):
    byte ``(r, b)`` sits at ``(r // 128) * (B / 4 * 512) + (b // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + b % 4``."""
    rows, blocks = linear.shape
    bands = (rows + ATOM_ROWS - 1) // ATOM_ROWS
    padded = torch.zeros(bands * ATOM_ROWS, blocks, dtype=torch.uint8, device=linear.device)
    padded[:rows] = linear
    return padded.reshape(bands, 4, 32, blocks // 4, 4).permute(0, 3, 2, 1, 4).contiguous().reshape(-1)


def from_atoms(flat, rows, blocks):
    """The inverse of :func:`to_atoms` for the first ``rows`` rows: ``[rows, blocks]`` bytes."""
    bands = (rows + ATOM_ROWS - 1) // ATOM_ROWS
    used = flat.contiguous().view(torch.uint8).reshape(-1)[: bands * ATOM_ROWS * blocks]
    return used.reshape(bands, blocks // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(bands * ATOM_ROWS, blocks)[:rows].contiguous()


def atom_index(row, block, blocks):
    """The flat position of scale byte ``(row, block)`` (the documented byte rule, spelled once more for index-built expectations)."""
    return (row // 128) * (blocks // 4 * 512) + (block // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4


def interleave_index(n):
    """Natural weight row of every interleaved row: rows ``128 j .. + 63`` are gate channels ``64 j ..``, rows ``128 j + 64 ..`` the up
    channels of the same 64 (``n`` = 2 x intermediate; natural order = gate rows, then up rows)."""
    inter = n // 2
    j, r = torch.arange(n) // 128, torch.arange(n) % 128
    return torch.where(r < 64, 64 * j + r, inter + 64 * j + (r - 64))


# ---------------------------------------------------------------------------------------------------------------------------
# operands
# ---------------------------------------------------------------------------------------------------------------------------
def make_weight(geometry, device="cpu", seed=0, kind="random"):
    """The weight of one geometry in NATURAL row order (gate rows, then up rows): ``dict(codes [2 I, H / 2], scales [2 I, H / 16]
    linear bytes, global_scale)``.  ``random``: normal draws from a HOST generator, quantised; ``raw``: random code bytes with random
    finite scale bytes (every code and a wide range of scales)."""
    n, k = geometry.WEIGHT_ROWS, geometry.HIDDEN
    gen = torch.Generator(device="cpu").manual_seed(1000 + seed)
    if kind == "random":
        gs = 64.0
        codes, scales = [], []
        for start in range(0, n, 4096):
            x = torch.randn(min(4096, n - start), k, generator=gen) * 0.02 * gs
            c, s = quantize_nvfp4(x.to(device))
            codes.append(c)
            scales.append(s)
        return dict(codes=torch.cat(codes), scales=torch.cat(scales), global_scale=gs)
    if kind == "raw":
        codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen).to(device)
        scales = torch.randint(0x28, 0x48, (n, k // SF_VEC), dtype=torch.uint8, generator=gen).to(device)  # 2^-2 .. 2^2: finite, positive
        return dict(codes=codes, scales=scales, global_scale=8.0)
    raise ValueError(kind)


def interleaved(weight):
    """The weight as the entry takes it: ``(weight_interleaved, weight_scale_interleaved)`` through the entry's own load-time helper."""
    return entry.interleave_gate_up(weight["codes"], to_atoms(weight["scales"]))


def make_activation(geometry, rows, device="cpu", seed=0, kind="random"):
    """``dict(codes [rows, H / 2], scales [rows, H / 16] linear bytes, global_scale)``.  ``random`` / ``raw`` as :func:`make_weight`;
    ``structured``: per row a different shape -- an all-zero row, a row with one large value per block, a row of tiny values, a
    saturating row, a row with all-zero blocks; ``mixed``: the even rows of the ``random`` draw and the odd rows of the ``structured``
    one (every structured shape among ordinary rows)."""
    k = geometry.HIDDEN
    if kind == "mixed":
        a, b = make_activation(geometry, rows, device, seed, "random"), make_activation(geometry, rows, device, seed, "structured")
        odd = (torch.arange(rows, device=device) % 2 == 1)[:, None]
        return dict(codes=torch.where(odd, b["codes"], a["codes"]), scales=torch.where(odd, b["scales"], a["scales"]), global_scale=a["global_scale"])
    gen = torch.Generator(device="cpu").manual_seed(2000 + seed)
    if kind == "raw":
        codes = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, generator=gen).to(device)
        scales = torch.randint(0x28, 0x48, (rows, k // SF_VEC), dtype=torch.uint8, generator=gen).to(device)
        return dict(codes=codes, scales=scales, global_scale=8.0)
    gs = 16.0
    x = torch.randn(rows, k, generator=gen) * gs
    if kind == "structured":
        for r in range(rows):
            shape = r % 5
            if shape == 0:
                x[r] = 0.0
            elif shape == 1:
                x[r] *= 0.01
                x[r, ::SF_VEC] = 40.0 * gs
            elif shape == 2:
                x[r] *= 2.0**-9
            elif shape == 3:
                x[r] *= 300.0
            else:
                x[r].reshape(-1, SF_VEC)[::3] = 0.0
    elif kind != "random":
        raise ValueError(kind)
    codes, scales = quantize_nvfp4(x.to(device))
    return dict(codes=codes, scales=scales, global_scale=gs)


def new_outputs(geometry, rows, device, fill=0xA5):
    """Caller-owned ``(out [rows, I / 2], out_scale [one atom band])`` prefilled with a sentinel byte."""
    out = torch.full((rows, geometry.INTERMEDIATE // 2), fill, dtype=torch.uint8, device=device)
    out_scale = torch.full((ATOM_ROWS * geometry.INTERMEDIATE // SF_VEC,), fill, dtype=torch.uint8, device=device)
    return out, out_scale


def scalar(value, device):
    return torch.tensor([value], dtype=torch.float32, device=device)


def launch_operands(weight, activation, out_global_scale=1.0, alpha=None, pair=None):
    """The launch arguments of the entry (without the two outputs) for a weight and an activation of the builders: the weight
    interleaved (``pair`` = an :func:`interleaved` result to reuse), the activation's scales in one atom band, ``alpha`` = the
    reciprocal of the two global scales."""
    device = weight["codes"].device
    w, ws = interleaved(weight) if pair is None else pair
    if alpha is None:
        alpha = 1.0 / (weight["global_scale"] * activation["global_scale"])
    return dict(
        weight=w,
        weight_scale=ws,
        activation=activation["codes"],
        activation_scale=to_atoms(activation["scales"]),
        alpha=scalar(alpha, device),
        out_global_scale=scalar(out_global_scale, device),
    )


def sub_batch(ops, activation, index):
    """The launch operands of a sub-batch -- the rows ``index`` of an activation, in that order, on the same weight and scalars -- and
    the sub-activation's dict: ``(operands, activation)``."""
    index = torch.as_tensor(index, device=activation["codes"].device)
    sub = dict(codes=activation["codes"][index].contiguous(), scales=activation["scales"][index].contiguous(), global_scale=activation["global_scale"])
    return dict(ops, activation=sub["codes"], activation_scale=to_atoms(sub["scales"])), sub


def run(e, ops, out=None, out_scale=None, fill=0xA5):
    """One call of the entry on ``ops``; returns ``(out, out_scale)`` (fresh sentinel-filled buffers unless given)."""
    rows = ops["activation"].shape[0]
    if out is None:
        out, out_scale = new_outputs(e.geometry, rows, ops["weight"].device, fill)
    e(ops["weight"], ops["weight_scale"], ops["activation"], ops["activation_scale"], ops["alpha"], ops["out_global_scale"], out, out_scale)
    return out, out_scale


def live_scales(out_scale, rows, geometry):
    """The scale bytes of the live rows, gathered from their atom positions: ``[rows, I / 16]``."""
    return from_atoms(out_scale, rows, geometry.INTERMEDIATE // SF_VEC)


def live_mask(rows, geometry, device="cpu"):
    """Which bytes of one atom band of ``out_scale`` belong to the live rows (the entry writes exactly these)."""
    blocks = geometry.INTERMEDIATE // SF_VEC
    mask = torch.zeros(ATOM_ROWS, blocks, dtype=torch.uint8, device=device)
    mask[:rows] = 1
    return to_atoms(mask).bool()


def assert_same_bytes(got, want, label=""):
    assert got.shape == want.shape and got.dtype == want.dtype, f"{label}: shape or dtype changed"
    diff = got.contiguous().view(torch.uint8) != want.contiguous().view(torch.uint8)
    count = int(diff.sum())
    if count:
        first = int(torch.nonzero(diff.reshape(-1))[0])
        raise AssertionError(f"{label}: {count} of {diff.numel()} bytes differ; first flat index {first}")


# ---------------------------------------------------------------------------------------------------------------------------
# the fp64 reference with its bound (module docstring)
# ---------------------------------------------------------------------------------------------------------------------------
def reference(weight, activation, alpha, tiles):
    """``(v*, E_v)`` in fp64 for the channels of the weight tiles ``tiles`` (64 channels each): ``[rows, 64 x len(tiles)]``.  ``weight``
    / ``activation`` are the builders' dicts (natural row order); the dequantisation is exact in fp64."""
    k = activation["codes"].shape[1] * 2
    inter = weight["codes"].shape[0] // 2
    act = unpack_fp4(activation["codes"]) * e4m3_values(activation["scales"]).repeat_interleave(SF_VEC, dim=1)
    chan = torch.cat([torch.arange(64 * t, 64 * t + 64) for t in tiles]).to(weight["codes"].device)

    def rows_of(index):
        return unpack_fp4(weight["codes"][index]) * e4m3_values(weight["scales"][index]).repeat_interleave(SF_VEC, dim=1)

    wg, wu = rows_of(chan), rows_of(inter + chan)
    a = float(alpha)
    g, u = a * (act @ wg.t()), a * (act @ wu.t())
    sg, su = abs(a) * (act.abs() @ wg.abs().t()), abs(a) * (act.abs() @ wu.abs().t())
    silu = g * torch.sigmoid(g)
    v = silu * u
    d_g = 2.0**-8 * g.abs() + (1 + 2.0**-8) * (k + 1) * 2.0**-23 * sg
    d_u = 2.0**-8 * u.abs() + (1 + 2.0**-8) * (k + 1) * 2.0**-23 * su
    big_a = 1.1 * d_g * (u.abs() + d_u) + silu.abs() * d_u
    e_v = big_a + (2.0**-8 + 2.0**-19) * (v.abs() + big_a) + 2.0**-120
    return v, e_v, chan


def e4m3_round(x):
    """Non-negative fp64 values -> E4M3 bytes, round to nearest even, saturating at 448: a table search over the 127 finite
    non-negative codes (their values ascend with the byte), independent of any framework cast."""
    values = e4m3_values(torch.arange(0, E4M3_NAN, dtype=torch.uint8, device=x.device))
    x = x.clamp(max=448.0)
    hi = torch.searchsorted(values, x.contiguous()).clamp(max=E4M3_MAX)
    lo = (hi - 1).clamp(min=0)
    d_lo, d_hi = x - values[lo], values[hi] - x
    return torch.where((d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0)), hi, lo).to(torch.uint8)


def check_against_reference(out, out_scale, weight, activation, alpha, gs, tiles):
    """The reference comparison of the module docstring on the weight tiles ``tiles``.  Returns a dict of the worst ratios and counts;
    ``scale_outside`` and ``code_ratio > 1`` are failures (the caller asserts)."""
    rows = activation["codes"].shape[0]
    inter = weight["codes"].shape[0] // 2
    v, e_v, chan = reference(weight, activation, alpha, tiles)
    geometry = kcfg.Geometry(activation["codes"].shape[1] * 2, inter)
    blocks = (chan.reshape(-1, SF_VEC)[:, 0] // SF_VEC).long()
    got_byte = live_scales(out_scale, rows, geometry)[:, blocks]
    amax = v.abs().reshape(rows, -1, SF_VEC).amax(dim=-1)
    e_a = e_v.reshape(rows, -1, SF_VEC).amax(dim=-1)
    lo = e4m3_round(gs * (amax - e_a).clamp(min=0) / 6.0 * (1 - 2.0**-21))
    hi = e4m3_round(gs * (amax + e_a) / 6.0 * (1 + 2.0**-21))
    outside = int(((got_byte < lo) | (got_byte > hi)).sum())
    codes = unpack_fp4(out)[:, chan]
    s = (e4m3_values(got_byte) / gs).repeat_interleave(SF_VEC, dim=1)
    h = torch.where(codes.abs() < 2, 0.25, torch.where(codes.abs() < 4, 0.5, 1.0))
    bound = h * s + (v.abs() + e_v - 6 * s).clamp(min=0) + e_v
    err = (codes * s - v).abs()
    ratio = float((err / bound).max())
    rel_l2 = float(torch.linalg.vector_norm(codes * s - v) / torch.linalg.vector_norm(v).clamp(min=1e-300))
    return dict(
        scale_outside=outside,
        scale_ambiguous=int((lo != hi).sum()),
        blocks=int(got_byte.numel()),
        code_ratio=ratio,
        rel_l2=rel_l2,
        zero_blocks=int((got_byte == 0).sum()),
    )


def device_or_skip():
    """A cc 10.x CUDA device with a CuTe DSL the kernel can be compiled on, or a skip that names what is missing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the SwiGLU FP4 decode kernel serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry.dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    return dev


# ---------------------------------------------------------------------------------------------------------------------------
# exact cells: operands whose every intermediate is exact, so the kernel's bytes equal a host-built expectation
# ---------------------------------------------------------------------------------------------------------------------------
EXACT_BLOCKS = (0, 1, 2, 100, 200, 319)  # the 16-column blocks of the hidden axis that carry the non-zero elements (several K stages)
EXACT_GS = 2.0**-8
#: (gate terms, up terms) of the first channels: each term is one weight element (an E4M3-representable value) in one block
EXACT_CHANNELS = (
    ((256, 256, 2), (3,)),  # first rounding, tie down: 514 -> 512
    ((256, 256, 6), (1,)),  # first rounding, tie up: 518 -> 520
    ((32, 5), (14,)),  # second rounding, tie up: 37 x 14 = 518 -> 520
    ((28, 1), (18,)),  # second rounding, tie down: 29 x 18 = 522 -> 520
    ((), (5,)),  # a zero gate: silu(0) = 0, the product is +0
    ((), (-5,)),  # a zero gate under a negative up: the product is -0 and its code keeps the sign
    ((20,), (-6,)),
    ((448,), (0.5,)),
)


def exact_case(device="cpu", rows=5, seed=0):
    """Integer-valued operands at the small geometry with every partial sum exact in fp32, gate pre-activations that are 0 or >= 18
    (where the kernel's SiLU is the identity: ``2^(-x log2 e)`` is below half a unit of 1.0 and the sum rounds to 1) and a
    power-of-two output scale.  Each weight row holds one element per block of :data:`EXACT_BLOCKS`; activation row ``t`` is
    ``2^(t % 3)`` at the first column of those blocks, the last row is all zero.  Returns ``(weight, activation, alpha, gs)`` in the
    builders' form (natural row order)."""
    geometry = G_SMALL
    n, k, inter = geometry.WEIGHT_ROWS, geometry.HIDDEN, geometry.INTERMEDIATE
    gen = torch.Generator(device="cpu").manual_seed(4000 + seed)
    small = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28], dtype=torch.float64)  # E4M3-representable integers
    values = torch.zeros(n, len(EXACT_BLOCKS), dtype=torch.float64)
    values[:inter] = small[torch.randint(0, len(small), (inter, len(EXACT_BLOCKS)), generator=gen)]
    values[:inter, 0] = 32.0  # every random gate is at least 32
    signs = torch.where(torch.randint(0, 2, (inter, len(EXACT_BLOCKS)), generator=gen) == 0, 1.0, -1.0).double()
    values[inter:] = small[torch.randint(0, len(small), (inter, len(EXACT_BLOCKS)), generator=gen)] * signs
    for c, (gate, up) in enumerate(EXACT_CHANNELS):
        values[c] = 0.0
        values[inter + c] = 0.0
        values[c, : len(gate)] = torch.tensor(gate, dtype=torch.float64)
        values[inter + c, : len(up)] = torch.tensor(up, dtype=torch.float64)
    scales = torch.full((n, k // SF_VEC), 0x38, dtype=torch.uint8)  # 1.0 everywhere else
    codes = torch.zeros(n, k // 2, dtype=torch.uint8)
    for j, b in enumerate(EXACT_BLOCKS):
        byte = e4m3_round(values[:, j].abs())
        assert torch.equal(e4m3_values(byte), values[:, j].abs()), "every weight element is an E4M3 value"
        scales[:, b] = torch.where(values[:, j] == 0, torch.tensor(0x38, dtype=torch.uint8), byte)
        codes[:, b * 8] = torch.where(values[:, j] == 0, 0, torch.where(values[:, j] > 0, 0x2, 0xA)).to(torch.uint8)  # +1.0 / -1.0 in the low nibble
    a_scales = torch.full((rows, k // SF_VEC), 0x38, dtype=torch.uint8)
    a_codes = torch.zeros(rows, k // 2, dtype=torch.uint8)
    for t in range(rows - 1):
        for b in EXACT_BLOCKS:
            a_scales[t, b] = (0x38, 0x40, 0x48)[t % 3]  # 1, 2, 4
            a_codes[t, b * 8] = 0x2
    a_scales[rows - 1] = 0
    weight = dict(codes=codes.to(device), scales=scales.to(device), global_scale=1.0)
    activation = dict(codes=a_codes.to(device), scales=a_scales.to(device), global_scale=1.0)
    return weight, activation, 1.0, EXACT_GS


def exact_expectation(weight, activation, alpha, gs):
    """``(out [rows, I / 2], scale bytes [rows, I / 16])`` of an :func:`exact_case` from exact host arithmetic: the pre-activations in
    fp64 (exact), the two bf16 roundings by the host's round-to-nearest-even cast, the block scale in fp32 as the kernel orders it
    (``gs * (amax * fp32(1/6))``) and rounded by the table, the codes from the exact quotient -- after asserting that no quotient sits
    within 2^-16 (relative) of a rounding boundary, so the kernel's one approximate reciprocal cannot move a code."""
    cpu = {name: (v.cpu() if isinstance(v, torch.Tensor) else v) for name, v in weight.items()}
    act = {name: (v.cpu() if isinstance(v, torch.Tensor) else v) for name, v in activation.items()}
    inter = cpu["codes"].shape[0] // 2
    w = unpack_fp4(cpu["codes"]) * e4m3_values(cpu["scales"]).repeat_interleave(SF_VEC, dim=1)
    x = unpack_fp4(act["codes"]) * e4m3_values(act["scales"]).repeat_interleave(SF_VEC, dim=1)
    gate, up = alpha * (x @ w[:inter].t()), alpha * (x @ w[inter:].t())
    assert float(gate.abs().max()) < 2**24 and float(up.abs().max()) < 2**24 and bool((gate == gate.round()).all())
    gate, up = gate.float().to(torch.bfloat16).float(), up.float().to(torch.bfloat16).float()  # the first rounding
    assert bool(((gate == 0) | (gate >= 18)).all()), "the SiLU premise: every gate is 0 or at least 18"
    v = (gate * up).to(torch.bfloat16).float()  # silu(gate) = gate (or +0); the product is exact in fp32; the second rounding
    rows = v.shape[0]
    amax = v.abs().reshape(rows, -1, SF_VEC).amax(dim=-1)
    sixth = torch.tensor([0x3E2AAAAB], dtype=torch.int32).view(torch.float32)
    byte = e4m3_round((torch.tensor(gs, dtype=torch.float32) * (amax * sixth)).double())
    value = e4m3_values(byte).repeat_interleave(SF_VEC, dim=1)
    q = torch.where(value > 0, v.double().abs() * gs / value.clamp(min=1e-300), torch.zeros_like(value))
    edges = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float64)
    margin = ((q[..., None] - edges).abs() / edges).min()
    assert float(margin) > 2.0**-16, f"an exact cell sits {float(margin):.3e} (relative) from a rounding boundary"
    codes = torch.bucketize(q, edges).to(torch.uint8) | (torch.signbit(v).to(torch.uint8) << 3)
    return pack_fp4(codes), byte
