# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers of the FP8 projection decode entry's test modules: the operand builders (E4M3 codes, structured cases, exact-integer
cases), the fp64 reference with its per-element bound, the raw-byte view and the device gate.  Not a test module (no ``test_`` prefix);
imported by name from the sibling modules (pytest puts their directory on ``sys.path``)."""

import pytest

torch = pytest.importorskip("torch")
cudnn = pytest.importorskip("cudnn")

from cudnn.gemm.frost.projection_decode import fp8_projection_decode as entry  # noqa: E402
from cudnn.gemm.frost.projection_decode.kernel import fp8_projection_decode_config as kcfg  # noqa: E402

G_IN = kcfg.Geometry(4608, 8192)
G_OUT = kcfg.Geometry(8192, 2048)
G_5K = kcfg.Geometry(5120, 6144)
G_16K = kcfg.Geometry(16384, 5120)
G_14K = kcfg.Geometry(14336, 5120)
# in served order: (G_IN, 4), (G_IN, 2), (G_OUT, 2) -- the one-wave classes -- then (G_5K, 4), (G_5K, 2), (G_16K, 1), (G_14K, 1), run as written
CLASSES = list(kcfg.SERVED_CLASSES)
SM100_READING = dict(sm_count=kcfg.SM100_SM_COUNT, smem_per_sm=kcfg.SM100_SMEM_PER_SM_BYTES, smem_optin=kcfg.SM100_SMEM_OPTIN_BYTES)  # 152 / 233,472 / 232,448
E4M3 = torch.float8_e4m3fn
NAN_CODE = 0x7F  # the positive E4M3 NaN; 0xFF is the negative one
MAX_CODE = 0x7E  # +448, the largest finite E4M3 value


def cid(numerics):
    return f"n{numerics.GEOM.OUT_FEATURES}k{numerics.GEOM.IN_FEATURES}s{numerics.SPLIT_K}"


def from_codes(codes):
    """uint8 E4M3 codes as an E4M3 tensor (a view: the bytes are the operand)."""
    return codes.contiguous().view(E4M3)


def finite_codes(codes):
    """Random bytes with the two NaN codes folded onto the largest finite magnitude."""
    return torch.where((codes & 0x7F) == NAN_CODE, codes - 1, codes)


def make_operands(geometry, rows, device="cpu", seed=0, kind="random"):
    """``weight`` [N, K] and ``activation`` [rows, K] in E4M3, ``scale`` fp32 [1].  ``random``: codes below one in magnitude with random
    signs (sums of a few units); ``wide``: every finite code (magnitudes up to 448: large partial sums and cancellation); ``structured``:
    index patterns (every column and row distinct, long runs of equal signs)."""
    n, k = geometry.OUT_FEATURES, geometry.IN_FEATURES
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(lo, hi, *shape):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, generator=gen)

    if kind == "random":
        w = rnd(0, 64, n, k) | (rnd(0, 2, n, k) << 7)
        x = rnd(8, 72, rows, k) | (rnd(0, 2, rows, k) << 7)
        scale = 0.02 * 0.03
    elif kind == "wide":
        w, x = finite_codes(rnd(0, 256, n, k)), finite_codes(rnd(0, 256, rows, k))
        scale = 2.0**-14 * 1.37
    elif kind == "structured":
        ni, ki, ti = torch.arange(n)[:, None], torch.arange(k)[None, :], torch.arange(rows)[:, None]
        w = finite_codes(((ni * 7 + ki * 13) % 251).to(torch.uint8))
        x = finite_codes((((ti + 1) * 31 + ki * 5) % 251).to(torch.uint8))
        scale = 2.0**-12
    else:
        raise ValueError(kind)
    return dict(
        weight=from_codes(w).to(device),
        activation=from_codes(x).to(device),
        scale=torch.tensor([scale], dtype=torch.float32, device=device),
    )


def integer_operands(geometry, rows, device="cpu", seed=0):
    """Operands whose every partial sum is exact in fp32 under ANY summation order: integer values in -8 .. 8 (products up to 64 in
    magnitude, sums below 2^24 for K up to 8192) and a power-of-two scale.  The kernel's bytes must then equal the host's."""
    n, k = geometry.OUT_FEATURES, geometry.IN_FEATURES
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randint(-8, 9, (n, k), generator=gen).float().to(device).to(E4M3)
    x = torch.randint(-8, 9, (rows, k), generator=gen).float().to(device).to(E4M3)
    return dict(weight=w, activation=x, scale=torch.tensor([2.0**-6], dtype=torch.float32, device=device))


def new_out(geometry, rows, device, fill=float("nan")):
    return torch.full((rows, geometry.OUT_FEATURES), fill, dtype=torch.bfloat16, device=device)


def reference(ops):
    """(ref, s_abs) in fp64 from the E4M3 operands: ``ref = scale * sum_k activation * weight`` (every product of two E4M3 values is
    exact in fp64) and ``s_abs = sum_k |activation| |weight|``."""
    x, w = ops["activation"].float().double(), ops["weight"].float().double()
    scale = ops["scale"].double().reshape(())
    return scale * (x @ w.t()), x.abs() @ w.abs().t()


def bf16_half_ulp(v):
    """Half a bf16 unit in the last place at ``v`` (8 significant bits; the subnormal spacing below 2^-126)."""
    _, e = torch.frexp(v.double().abs().clamp(min=2.0**-126))  # |v| = m * 2^e, m in [0.5, 1)
    return torch.ldexp(torch.ones_like(v, dtype=torch.float64), e - 1 - 7) * 0.5


def error_over_bound(out, ops, split_k):
    """max over the finite elements of ``|out - ref| / bound`` with ``bound = half a bf16 ulp at out + |scale| * s_abs * (K + split_k + 1) *
    2^-23``: the products are exact in fp32, an any-order fp32 sum of K exact terms errs by at most ``(K - 1) * 2^-24 * s_abs`` to first
    order (doubled here to cover an adder that truncates), the ``split_k - 1`` slice adds and the one scale multiply add one unit each,
    and the cast adds half a bf16 unit at the kernel's own value.  Never widened.  Returns (ratio, non-finite mismatches)."""
    ref, s_abs = reference(ops)
    o = out.double()
    k = ops["weight"].shape[1]
    bound = bf16_half_ulp(o) + ops["scale"].double().abs().reshape(()) * s_abs * (k + split_k + 1) * 2.0**-23
    finite = torch.isfinite(o) & torch.isfinite(ref) & (ref.abs() < 3.0e38)
    ratio = float(((o - ref).abs()[finite] / bound[finite]).max()) if bool(finite.any()) else 0.0
    # where the reference leaves the bf16 range the kernel must write an infinity of the reference's sign
    over = ref.abs() >= 3.4e38
    bad = int((over & ~(torch.isinf(o) & (torch.sign(o) == torch.sign(ref)))).sum()) + int((~over & ~torch.isfinite(o)).sum())
    return ratio, bad


def raw(t):
    """The raw bits of a bf16 tensor (NaN payloads and signed zeros included)."""
    return t.contiguous().view(torch.int16)


def assert_same_bytes(got, want, label=""):
    assert got.shape == want.shape and got.dtype == want.dtype, f"{label}: shape or dtype changed"
    diff = raw(got) != raw(want)
    count = int(diff.sum())
    if count:
        first = int(torch.nonzero(diff.reshape(-1))[0])
        raise AssertionError(f"{label}: {count} of {diff.numel()} bf16 elements differ in their raw bytes; first flat index {first}")


def device_or_skip():
    """A cc 10.x CUDA device with a CuTe DSL the kernel can be compiled on, or a skip that names what is missing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    dev = torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if not entry.cc_supported(cc):
        pytest.skip(f"the FP8 projection decode kernel serves cc 10.x; this device is {cc[0]}.{cc[1]}")
    msg = entry.dsl_error(cc)
    if msg is not None:
        pytest.skip(msg)
    return dev
