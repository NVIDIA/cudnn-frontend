# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the FP4 row quantization decode step's test modules: the launch arms, the global scales, the operand builders
(built on the HOST), the independent references and the device gate.  Not a test module (no ``test_`` prefix); imported by name from
the sibling modules (pytest puts their directory on ``sys.path``).

THE REFERENCE is written here from the format's definition -- torch math on the host; no kernel helper and no layout helper of the
package is imported -- and it is statements about BYTES:

1. *The block scales* (:func:`scale_bytes`): BITWISE.  Per block of sixteen elements ``amax = max |x_j|`` and
   ``byte = e4m3( gs x (amax x fp32(1/6)) )``: two fp32 multiplications in THAT association (``amax x fp32(1/6)`` first, the constant
   as the bit pattern ``0x3E2AAAAB``), then the nearest-even E4M3 rounding that saturates at 448.  No approximate operation.
2. *The position of a block scale* (:func:`scale_byte_index`): the 128 x 4 layout written out -- bands of 128 rows one after the
   other; in a band one 512-byte atom per four scale columns; in an atom 32 lines of 16 bytes, line ``row % 32``; in a line the four
   quarters of the band, quarter ``(row // 32) % 4``, four scale columns each.
3. *The codes* (:func:`assert_codes`).  The encode scale goes through two approximate reciprocals, so the reference is a bracket:
   with ``q = gs / F`` in fp64 (``F`` the fp32 value of the stored block scale) every code is ``e2m1(x q (1 - r))`` or
   ``e2m1(x q (1 + r))`` with ``r = RCP_REL_BOUND`` -- at most two ADJACENT codes, and ONE code for every element not within ``r`` of
   an E2M1 rounding midpoint.  A zero block scale encodes signed zeros.  ``RCP_REL_BOUND = 5 x 2^-23``: one unit in the last place per
   approximate reciprocal (its documented bound), doubled, plus one fp32 unit for the two multiplications.  A code outside the
   bracket is a finding with its magnitude, never a reason to widen.
4. *The codes of an exact block* (:func:`exact_codes`): BITWISE.  Where the global scale and the block's scale value are powers of
   two, both reciprocals and the multiply are exact, so the code is ``e2m1(x x gs / F)``.

THE GLOBAL SCALES.  ``SCALE_BITS`` holds eleven fp32 patterns.  Under each of the first eight (``PLAIN_SCALE_BITS``: a
non-power-of-two value near 48.65, 1, 0.25, 37, 2^10, 2^-3, 2^-20, 2^20) the association of statement 1 and three other ways to form
the same product -- ``(amax x gs) x fp32(1/6)``, ``amax x (gs x fp32(1/6))``, ``(amax / 6) x gs`` -- round to the SAME byte for every
finite bf16 block maximum: cells under those scales cannot tell the associations apart.  The last three
(``CONSTRUCTED_SCALE_BITS``) are neighbours of ``midpoint / (amax x fp32(1/6))`` for some block maxima: under them the associations
differ on tens of block maxima each, so a kernel that formed the scale another way writes other bytes
(``test_reference_separates_the_scale_associations`` pins the counts; the cell over every block maximum runs under all eleven).
"""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.row_quantize.frost import row_quantize_fp4_decode as entry  # noqa: E402

WIDTH = 5120
BLOCK = 16
BLOCKS = WIDTH // BLOCK
BF16, U8, I16, E4M3 = torch.bfloat16, torch.uint8, torch.int16, torch.float8_e4m3fn
E4M3_MAX = 448.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # the magnitudes of the codes 0 .. 7; code | 8 is the negative
E2M1_MIDPOINTS = tuple((a + b) / 2 for a, b in zip(E2M1_GRID[:-1], E2M1_GRID[1:]))
FP4_MAX_RCP_BITS = 0x3E2AAAAB  # fp32(1 / 6)
RCP_REL_BOUND = 5 * 2.0**-23
#: the finite non-negative bf16 patterns: 0 .. 0x7F7F (0x7F80 is +infinity)
BF16_FINITE = 0x7F80

PLAIN_SCALE_BITS = (0x42429B39, 0x3F800000, 0x3E800000, 0x42140000, 0x44800000, 0x3E000000, 0x35800000, 0x49800000)
CONSTRUCTED_SCALE_BITS = (0x41BFFFFF, 0x4219999A, 0x41DB6DB6)
SCALE_BITS = PLAIN_SCALE_BITS + CONSTRUCTED_SCALE_BITS
#: the global scales whose value is a power of two (both reciprocals of the encode scale are then exact for a power-of-two block scale)
POWER_OF_TWO_SCALE_BITS = (0x3F800000, 0x3E800000, 0x44800000, 0x3E000000, 0x35800000, 0x49800000)
#: the non-power-of-two scale of the first group: the golden and tie-sensitive cells run under it
ODD_SCALE_BITS = PLAIN_SCALE_BITS[0]
#: the served launch arms as ``(pdl, pdl_release)``: off; wait only; wait + release
LAUNCH_ARMS = ((None, None), (True, False), (True, None))
LAUNCH_ARM_IDS = ("plain", "wait", "wait-release")


def scale_id(bits):
    return f"gs{bits:08x}"


def make_entry(device, arm=(None, None), width=WIDTH):
    return entry.RowQuantizeFp4Decode(device, width, pdl=arm[0], pdl_release=arm[1])


def f32(bits):
    """The fp32 value of a bit pattern, as a 0-d host tensor."""
    return torch.tensor(bits, dtype=torch.int32).view(torch.float32)


def scale_tensor(bits, device, zero_d=False):
    """The global scale of a bit pattern as the operand a call takes: one fp32 element, ``[1]`` or 0-d."""
    value = f32(bits)
    return (value if zero_d else value.reshape(1)).clone().to(device)


# ---------------------------------------------------------------------------------------------------------------------------
# operands (built on the host: one seed gives one set of tensors on every GPU)
# ---------------------------------------------------------------------------------------------------------------------------
def make_operands(rows, device="cpu", seed=0, kind="random", width=WIDTH):
    """``x`` bf16 ``[rows, width]``, finite, from a CPU generator and moved to ``device``.  ``random``: normal values under a per-row
    and a per-block magnitude, each spread over twelve octaves.  ``structured``: the same draw with a ramp block, alternating-sign
    runs, zero blocks, negative zeros and, in every row, sixteen blocks whose ONE dominant element sits at each of the sixteen
    positions of a block."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(rows, width // BLOCK, BLOCK, generator=gen)
    row_mag = torch.exp2(torch.empty(rows, 1, 1).uniform_(-6.0, 6.0, generator=gen))
    block_mag = torch.exp2(torch.empty(rows, width // BLOCK, 1).uniform_(-6.0, 6.0, generator=gen))
    x = (base * block_mag * row_mag).reshape(rows, width).bfloat16()
    if kind == "structured":
        x[:, 0:16] = torch.arange(16, dtype=torch.float32).bfloat16() - 7.5  # a ramp through zero
        x[:, 16:48] = torch.tensor([1.0, -1.0]).bfloat16().repeat(16)
        x[:, 64:96] = 0.0
        x[:, 96:104] = -0.0
        for position in range(BLOCK):  # one dominant element per block, at each position of a block
            start = 128 + BLOCK * position
            x[:, start : start + BLOCK] = (x[:, start : start + BLOCK].float() * 2.0**-9).bfloat16()
            x[:, start + position] = -300.0 if position % 2 else 700.0
    elif kind != "random":
        raise ValueError(kind)
    assert bool(torch.isfinite(x.float()).all())
    return x.to(device)


def bf16_from_patterns(patterns):
    """bf16 values from their 16-bit patterns (an integer tensor of values 0 .. 0xFFFF)."""
    return (patterns.to(torch.int32) & 0xFFFF).to(I16).view(BF16)


def all_maxima_operand(seed=0, width=WIDTH):
    """bf16 ``[102, 5120]``: 32,640 blocks, block ``k`` (row ``k // 320``, block ``k % 320``) holding the finite non-negative bf16
    pattern ``k`` as its maximum magnitude at element ``k % 16``; the other fifteen elements are seeded patterns of smaller or equal
    magnitude; every element has a seeded sign.  Block 0 is a block of signed zeros.  EVERY finite bf16 magnitude is the maximum of
    exactly one block, so a call on this operand writes every scale byte the format can produce under its global scale."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.arange(BF16_FINITE, dtype=torch.int64)
    below = (torch.rand(BF16_FINITE, BLOCK, generator=gen, dtype=torch.float64) * (k + 1).unsqueeze(1)).floor().to(torch.int64)
    below = torch.minimum(below, k.unsqueeze(1))  # a bf16 magnitude is monotone in its pattern
    below[k, k % BLOCK] = k
    signs = torch.randint(0, 2, (BF16_FINITE, BLOCK), generator=gen, dtype=torch.int64) << 15
    x = bf16_from_patterns(below | signs).reshape(-1, width)
    assert x.shape[0] * (width // BLOCK) == BF16_FINITE
    return x.contiguous()


def maxima_scale_bytes(scale_bits, chain="definition", sixth_bits=FP4_MAX_RCP_BITS):
    """The scale byte of every finite non-negative bf16 block maximum (host uint8 ``[32640]``, by pattern) under one global scale and
    one way to form the product.  ``definition``: ``gs x (amax x fp32(1/6))`` -- statement 1 of the module docstring.  The others
    exist to show what the definition is NOT: ``amax_gs_first`` ``(amax x gs) x fp32(1/6)``; ``gs_sixth_first``
    ``amax x (gs x fp32(1/6))``; ``divide_by_six`` ``(amax / 6) x gs``.  ``sixth_bits`` replaces the constant (one unit off)."""
    amax = (torch.arange(BF16_FINITE, dtype=torch.int32) << 16).view(torch.float32)
    gs, sixth = f32(scale_bits), f32(sixth_bits)
    if chain == "definition":
        s = gs * (amax * sixth)
    elif chain == "amax_gs_first":
        s = (amax * gs) * sixth
    elif chain == "gs_sixth_first":
        s = amax * (gs * sixth)
    elif chain == "divide_by_six":
        s = (amax / torch.tensor(6.0)) * gs
    else:
        raise ValueError(chain)
    return s.clamp(max=E4M3_MAX).to(E4M3).view(U8)


def code_domain_operand(scale_bits, width=WIDTH):
    """bf16 ``[rows, 5120]`` that ENUMERATES the code domain of one global scale: a code is a function of the block's scale byte, the
    element and the global scale, so for every scale byte ``b`` (0 .. 0x7E, each that occurs) the blocks of ``b`` hold ``hi(b)`` --
    the largest bf16 magnitude whose scale byte is ``b`` -- at element 0 and, in elements 1 .. 15, every positive bf16 pattern from 1
    up to ``hi(b)`` in order (the last block of a byte repeats its last pattern).  The first half of the rows holds elements 1 .. 15
    positive, the second half the same rows with those elements negated; the rows of each half are filled up with zero blocks."""
    by_pattern = maxima_scale_bytes(scale_bits)
    blocks = []
    for byte in range(0x7F):
        patterns = (by_pattern == byte).nonzero().reshape(-1)
        if not patterns.numel():
            continue
        hi = int(patterns.max())
        body = torch.arange(1, hi + 1, dtype=torch.int64)
        if not body.numel():
            body = torch.zeros(1, dtype=torch.int64)
        fill = (-body.numel()) % (BLOCK - 1)
        body = torch.cat([body, body[-1:].expand(fill)]).reshape(-1, BLOCK - 1)
        blocks.append(torch.cat([torch.full((body.shape[0], 1), hi, dtype=torch.int64), body], dim=1))
    positive = torch.cat(blocks)
    per_row = width // BLOCK
    fill = (-positive.shape[0]) % per_row
    positive = torch.cat([positive, torch.zeros(fill, BLOCK, dtype=torch.int64)])
    negative = positive.clone()
    negative[:, 1:] |= 0x8000
    return bf16_from_patterns(torch.cat([positive, negative])).reshape(-1, width).contiguous()


# ---------------------------------------------------------------------------------------------------------------------------
# the references (host torch math on copies; every function returns host tensors)
# ---------------------------------------------------------------------------------------------------------------------------
def scale_bytes(x, scale):
    """Statement 1: the block-scale bytes ``[rows, width / 16]`` of a bf16 ``x`` in LOGICAL (row, block) order (host uint8).  A block
    with a NaN gets the byte of the maximum of its other elements here; such a block is outside the contract and is never compared."""
    xc = x.detach().cpu().float()
    amax = torch.nan_to_num(xc.abs(), nan=0.0).reshape(xc.shape[0], -1, BLOCK).amax(dim=2)
    gs = scale.detach().cpu().float().reshape(())
    return (gs * (amax * f32(FP4_MAX_RCP_BITS))).clamp(max=E4M3_MAX).to(E4M3).view(U8)


def scale_byte_index(rows, width=WIDTH):
    """Statement 2: host long ``[rows, width / 16]``, the byte of block scale ``(row, block)`` in a scale buffer."""
    atoms = (width // BLOCK + 3) // 4
    row = torch.arange(rows, dtype=torch.long).reshape(-1, 1)
    block = torch.arange(width // BLOCK, dtype=torch.long).reshape(1, -1)
    return ((row // 128) * atoms + block // 4) * 512 + (row % 32) * 16 + ((row // 32) % 4) * 4 + block % 4


def scale_rows(rows):
    """Rows of the scale buffer that holds ``rows`` rows: whole bands of 128."""
    return -(-rows // 128) * 128


def scale_bytes_of_rows(scales, rows, width=WIDTH):
    """The block-scale bytes of rows ``0 .. rows - 1`` in LOGICAL (row, block) order, read from a scale buffer in the 128 x 4 layout.
    Host uint8 ``[rows, width / 16]``."""
    return scales.detach().cpu().view(U8).reshape(-1)[scale_byte_index(rows, width)]


def scale_layout_mask(scale_buffer_rows, rows, width=WIDTH):
    """A host bool mask over the flat bytes of a ``[scale_buffer_rows, width / 16]`` scale buffer: True on the bytes of the LIVE rows
    ``0 .. rows - 1`` (the bytes the kernel writes), False on the padded rows' bytes (which it never writes)."""
    mask = torch.zeros(scale_buffer_rows * (width // BLOCK), dtype=torch.bool)
    mask[scale_byte_index(rows, width).reshape(-1)] = True
    return mask


def e2m1_codes(a):
    """fp32 / fp64 -> 4-bit E2M1 codes (uint8 ``0 .. 15``, sign in bit 3): nearest on the grid, a tie to the EVEN code, saturation at
    6, the sign kept (a negative value that rounds to zero is code 8).  NaN is refused."""
    if bool(torch.isnan(a).any()):
        raise ValueError("e2m1_codes: NaN is out of the quantizer's contract")
    mag = a.abs().double()
    mids = torch.tensor(E2M1_MIDPOINTS, dtype=torch.float64)
    below = torch.bucketize(mag, mids, right=False)  # a tie counts as below its midpoint
    above = torch.bucketize(mag, mids, right=True)  # a tie counts as above it
    code = torch.where(below % 2 == 0, below, above)
    return (code.to(U8) | (torch.signbit(a).to(U8) << 3)).contiguous()


def unpack_e2m1(packed):
    """Bytes ``[..., K / 2]`` -> codes ``[..., K]``: the low nibble is the even element."""
    u = packed.view(U8)
    return torch.stack([u & 0xF, u >> 4], dim=-1).flatten(-2)


def pack_e2m1(codes):
    """Codes ``[..., K]`` -> bytes ``[..., K / 2]``: the even element in the low nibble."""
    c = codes.to(U8) & 0xF
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()


def _scaled(x, block_bytes, scale):
    """``x x gs / F`` in fp64 per element (``F`` the fp32 value of the block's scale byte); the signed zero of ``x`` where ``F`` is zero."""
    xc = x.detach().cpu().double()
    f = block_bytes.view(E4M3).double().repeat_interleave(BLOCK, dim=1)
    gs = scale.detach().cpu().double().reshape(())
    q = torch.where(f != 0, gs / torch.where(f != 0, f, torch.ones_like(f)), torch.zeros_like(f))
    return xc * q


def code_bracket(x, block_bytes, scale):
    """Statement 3: ``(lo, hi)`` host uint8 E2M1 codes ``[rows, width]`` -- every code of the kernel is one of the two.
    ``block_bytes`` are the stored block scales in logical order (statement 1 pins them separately)."""
    a = _scaled(x, block_bytes, scale)
    return e2m1_codes(a * (1 - RCP_REL_BOUND)), e2m1_codes(a * (1 + RCP_REL_BOUND))


def exact_codes(x, block_bytes, scale):
    """Statement 4: the codes of blocks whose scale value and whose global scale are powers of two (host uint8 ``[rows, width]``)."""
    return e2m1_codes(_scaled(x, block_bytes, scale))


def assert_codes(codes, x, block_bytes, scale, label="", blocks=None):
    """Statement 3 on the kernel's packed codes (uint8 ``[rows, width / 2]``); returns the count of elements whose bracket holds two
    codes (the elements within the reciprocal bound of an E2M1 midpoint).  ``blocks`` (a host bool ``[rows, width / 16]``) restricts
    the statement to those blocks."""
    got = unpack_e2m1(codes.detach().cpu())
    lo, hi = code_bracket(x, block_bytes, scale)
    outside = (got != lo) & (got != hi)
    if blocks is not None:
        outside &= blocks.repeat_interleave(BLOCK, dim=1)
    n_out = int(outside.sum())
    assert n_out == 0, (
        f"{label}: {n_out} of {got.numel()} codes outside the bracket of the approximate reciprocals "
        f"(first at flat element {int(outside.reshape(-1).nonzero()[0])})"
    )
    return int((lo != hi).sum())


def assert_reference(codes, scales, x, scale, label="", padded_zero=True):
    """Statements 1 to 3 on ONE launch's bytes: the block scales bitwise through the layout, the code bracket, and (for a scale buffer
    whose padded rows the owner zero-filled) the zero bytes of the rows past the live ones.  Returns the count of elements whose
    bracket holds two codes."""
    rows, width = x.shape
    assert codes.dtype == U8 and tuple(codes.shape)[1] == width // 2 and codes.shape[0] >= rows, f"{label}: codes {tuple(codes.shape)} {codes.dtype}"
    assert scales.dtype == E4M3 and tuple(scales.shape)[1] == width // BLOCK and scales.shape[0] >= scale_rows(rows), f"{label}: scales {tuple(scales.shape)}"
    stored = scale_bytes_of_rows(scales, rows, width)
    want = scale_bytes(x, scale)
    n = int((stored != want).sum())
    assert n == 0, (
        f"{label}: {n} of {stored.numel()} block scales differ from e4m3(gs x (amax x fp32(1/6))) "
        f"(first at (row, block) {tuple(int(v) for v in (stored != want).nonzero()[0])})"
    )
    two = assert_codes(codes[:rows], x, stored, scale, label)
    if padded_zero:
        pad = ~scale_layout_mask(scale_rows(rows), rows, width)
        live_buffer = scales.detach().cpu().view(U8).reshape(-1)[: pad.numel()]
        assert bool((live_buffer[pad] == 0).all()), f"{label}: the scale buffer's padded rows are not zero"
    return two


# ---------------------------------------------------------------------------------------------------------------------------
# exact operands: every block's scale value is a power of two under a power-of-two global scale
# ---------------------------------------------------------------------------------------------------------------------------
#: the rows of :func:`exact_operands`
EXACT_ROWS = dict(ties=0, zeros=1, underflow=2, saturating=3, subnormal=4)


def exact_operands(scale_bits, width=WIDTH):
    """bf16 ``[5, width]`` for a POWER-OF-TWO global scale ``gs``, every block exact (statement 4 of the module docstring):

    * row 0: block ``b`` has the scale value ``F = 2^f`` with ``f = -9 + b % 18`` (every power of two E4M3 holds, its subnormals
      included): with ``u = F / gs`` the block holds ``6 u`` at position ``b % 16`` -- so the maximum sits at each of the sixteen
      positions -- and in the other fifteen positions the seven E2M1 rounding ties times ``u`` (0.25, 0.75, 1.25, 1.75, 2.5, 3.5,
      5), the grid values times ``u`` and zeros, with alternating signs: every tie element is an EXACT tie of the code rounding;
    * row 1: zeros of both signs (scale byte 0; the codes 0 and 8);
    * row 2: blocks whose scale underflows to the zero code although the block is not zero (zero codes, the signs kept);
    * row 3: blocks of the largest finite bf16 magnitude and zeros (the scale saturates at 448; the large elements at +-6);
    * row 4: bf16 subnormals of both signs (scale byte 0)."""
    gs = float(f32(scale_bits))
    mant, exponent = torch.frexp(torch.tensor(gs))
    assert float(mant) == 0.5, "exact_operands needs a power-of-two global scale"
    blocks = width // BLOCK
    ties = torch.tensor(E2M1_MIDPOINTS)
    fill = torch.cat([ties, torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 0.0, 0.0])])  # fifteen values below 6
    signs = torch.tensor([1.0, -1.0]).repeat(8)[:15]
    x = torch.zeros(5, blocks, BLOCK)
    for b in range(blocks):
        u = 2.0 ** (-9 + b % 18) / gs
        position = b % BLOCK
        others = fill.roll(b) * signs * u
        x[0, b] = torch.cat([others[:position], torch.tensor([6.0 * u * (-1.0 if b % 3 == 0 else 1.0)]), others[position:]])
    x[1] = torch.tensor([0.0, -0.0]).repeat(8)
    x[2] = torch.tensor([1.0, -0.5, 0.25, -1.0]).repeat(4) * (2.0**-11 / gs)  # gs x amax / 6 < 2^-10: the zero code
    x[3] = torch.tensor([1.0, -1.0, 0.0, -0.0]).repeat(4) * float(bf16_from_patterns(torch.tensor([0x7F7F])).float())
    out = x.reshape(5, width).bfloat16()
    assert bool((out.float() == x.reshape(5, width)).all()), "an exact operand is not a bf16 value"
    sub = bf16_from_patterns(torch.arange(width, dtype=torch.int64) % 0x7F + 1 + ((torch.arange(width, dtype=torch.int64) % 2) << 15))
    out[4] = sub
    return out.contiguous()


# ---------------------------------------------------------------------------------------------------------------------------
# the compiled kernel's operand wiring, read from its disassembly with the registers kept
# ---------------------------------------------------------------------------------------------------------------------------
SIXTH_TEXT = "0.16666667163372039795"  # fp32(1/6) as a disassembler prints the immediate


def sass_wiring_facts(sass_lines):
    """The operand WIRING of a compiled kernel as ``{fact: bool}``, from its disassembly (``nvdisasm -c`` lines).  The straight-line
    body is executed symbolically with the registers kept, so the facts are about which value reaches which operand -- what a count
    of instructions cannot show:

    * ``scale_convert``: the E4M3 convert's upper source and its merge source are the zero register, its lower source is the product;
    * ``scale_product``: that product is ``gs x (amax x fp32(1/6))`` -- ``gs`` the register the ONE 32-bit global load wrote, the
      constant an immediate on the multiply by the block maximum, the maximum taken over the magnitudes of all sixteen elements;
    * ``guarded_reciprocals``: the first reciprocal reads ``gs``, its result multiplies the widened scale value, the second
      reciprocal reads that product, and all three are predicated on ``value != 0``;
    * ``zero_else``: the encode scale is ``+0`` where the predicate is false (a zero move: ``IMAD.MOV.U32 r, RZ, RZ, RZ`` or
      ``HFMA2 r, -RZ, RZ, 0, 0``, by assembler);
    * ``code_order``: each E2M1 convert takes the products of elements ``(2i + 1, 2i)`` with the encode scale, the first stored word
      is elements 0 .. 7 and the second 8 .. 15;
    * ``scale_store``: the one-byte store writes the E4M3 convert's code;
    * ``no_ftz_multiply``: no multiply carries ``.FTZ``.

    ``None`` when the disassembly holds no E4M3 convert (nothing to read)."""
    import re

    def reg(token):
        token = token.strip()
        absolute = token.startswith("|")
        return token.strip("|").lstrip("-").split(".")[0], absolute

    env, pred, stores = {"RZ": ("zero",)}, {}, {}
    e4m3, loads32, ftz = None, 0, 0
    for raw in sass_lines:
        if not re.search(r"/\*[0-9a-f]{4}\*/", raw):
            continue
        ln = re.sub(r"/\*[0-9a-f]+\*/", "", raw).strip().rstrip(";").strip()
        guard = None
        m = re.match(r"^@(!?)(P\d)\s+(.*)$", ln)
        if m:
            guard, ln = (m.group(2), m.group(1) == "!"), m.group(3)
        op, _, rest = ln.partition(" ")
        ops = [o.strip() for o in rest.split(",")] if rest else []

        def val(token):
            name, absolute = reg(token)
            v = env.get(name, ("unknown", name))
            return ("abs", v) if absolute else v

        def put(dst, value):
            name, _ = reg(dst)
            env[name] = value if guard is None else ("sel", guard, value, env.get(name, ("unknown", name)))

        if op.startswith("LDG.E.128"):
            first = int(reg(ops[0])[0][1:])
            for i in range(4):
                env[f"R{first + i}"] = ("word", (4 if "+0x10" in rest else 0) + i)
        elif op == "LDG.E":
            put(ops[0], ("gs",))
            loads32 += 1
        elif op == "PRMT" and val(ops[3])[0] == "word" and ops[2] in ("0x5410", "0x7610"):
            put(ops[0], ("elem", 2 * val(ops[3])[1] + int(ops[2] == "0x7610")))
        elif op in ("FMNMX", "FMNMX3"):
            members, known = [], True
            for token in ops[1:]:
                if token.startswith("!") or token == "PT":
                    continue
                v = val(token)
                if v[0] == "abs" and v[1][0] == "elem":
                    members.append(v[1][1])
                elif v[0] == "max":
                    members.extend(v[1])
                else:
                    known = False
            put(ops[0], ("max", tuple(sorted(members))) if known else ("unknown", ln))
        elif op.startswith("FMUL"):
            ftz += int(".FTZ" in op)
            put(ops[0], ("mul", val(ops[1]), ("imm", ops[2]) if re.match(r"^-?\d", ops[2]) else val(ops[2])))
        elif op.startswith("F2FP.SATFINITE.E4M3"):
            e4m3 = ("e4m3", val(ops[1]), val(ops[2]), val(ops[3]))
            put(ops[0], e4m3)
        elif op == "LOP3.LUT" and len(ops) >= 5 and ops[2] == "0xffff" and ops[4] == "0xc0" and val(ops[1])[0] == "e4m3":
            put(ops[0], ("mask16", val(ops[1])))
        elif op.startswith("F2FP.F16.E4M3.UNPACK_B"):
            put(ops[0], ("unpack", val(ops[1])))
        elif op.startswith("HADD2.F32"):
            put(ops[0], ("widen", val(ops[2])))
        elif op == "IMAD.MOV.U32" and [reg(t)[0] for t in ops[1:]] == ["RZ", "RZ", "RZ"]:
            put(ops[0], ("zero",))
        elif op == "HFMA2" and ops[1:] == ["-RZ", "RZ", "0", "0"]:
            put(ops[0], ("zero",))  # fma(-0, 0, +0) = +0: another assembler's zero move
        elif op == "MOV" and len(ops) == 2 and re.match(r"^R\d+", ops[1]):
            put(ops[0], val(ops[1]))  # a register copy
        elif op.startswith("FSETP"):
            pred[ops[0]] = (op, val(ops[2]), val(ops[3]))
        elif op.startswith("MUFU.RCP"):
            put(ops[0], ("rcp", val(ops[1])))
        elif op.startswith("F2FP.SATFINITE.E2M1"):
            put(ops[0], ("e2m1", val(ops[1]), val(ops[2]), val(ops[3])))
        elif op.startswith("STG.E.64"):
            first = int(reg(ops[-1])[0][1:])
            stores["codes"] = (env.get(f"R{first}"), env.get(f"R{first + 1}"))
        elif op.startswith("STG.E.U8"):
            stores["scale"] = val(ops[-1])
        elif ops and re.match(r"^R\d+", ops[0]) and not op.startswith(("S2R", "LDC", "STG")):
            put(ops[0], ("unknown", ln))  # integer address arithmetic
    if e4m3 is None:
        return None

    def is_mul(v, a, b):
        return v[0] == "mul" and ((v[1] == a and v[2] == b) or (v[1] == b and v[2] == a))

    def arms(v, guard_name):
        return (v[2], v[3]) if v[0] == "sel" and v[1] == (guard_name, False) else (None, None)

    product = ("mul", ("max", tuple(range(BLOCK))), ("imm", SIXTH_TEXT))
    facts = dict(scale_convert=False, scale_product=False, guarded_reciprocals=False, zero_else=False, code_order=False)
    facts["scale_product"] = is_mul(e4m3[2], ("gs",), product) and loads32 == 1
    facts["scale_convert"] = e4m3[1] == ("zero",) and e4m3[3] == ("zero",) and facts["scale_product"]
    value = ("widen", ("unpack", ("mask16", e4m3)))
    guards = [name for name, (op, a, b) in pred.items() if op.startswith("FSETP.NEU") and a == value and b == ("zero",)]
    words = stores.get("codes")
    encode = None
    if guards and words and words[0] is not None and words[0][0] == "e2m1" and words[0][2][0] == "mul":
        low = words[0][2]  # the product of element 0
        encode = low[2] if low[1] == ("elem", 0) else low[1]
        then, other = arms(encode, guards[0])
        facts["zero_else"] = other == ("zero",)
        if then is not None and then[0] == "rcp":
            inner, _ = arms(then[1], guards[0])  # the predicated multiply; what it leaves where the predicate is false is never read
            if inner is not None and inner[0] == "mul":
                first = inner[2] if inner[1] == value else inner[1]
                facts["guarded_reciprocals"] = value in (inner[1], inner[2]) and arms(first, guards[0])[0] == ("rcp", ("gs",))
    if encode is not None and all(w is not None for w in words):
        ordered = True
        for w in range(2):
            node = words[w]
            for i in range(4):
                ordered = (
                    ordered and node[0] == "e2m1" and is_mul(node[1], ("elem", 8 * w + 2 * i + 1), encode) and is_mul(node[2], ("elem", 8 * w + 2 * i), encode)
                )
                if node[0] != "e2m1":
                    break
                node = node[3]
            ordered = ordered and node == ("zero",)
        facts["code_order"] = ordered
    facts["scale_store"] = stores.get("scale") in (("mask16", e4m3), e4m3)
    facts["no_ftz_multiply"] = ftz == 0
    return facts


def _device_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the FP4 row quantization decode step serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry._dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    return dev
