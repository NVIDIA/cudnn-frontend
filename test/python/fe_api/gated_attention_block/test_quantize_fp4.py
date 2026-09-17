# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The block's fp4 quantize pass (``kernels/quantize_fp4.py``): e2m1 codes + the out-GEMM's F8_128x4 scale blob,
in NVFP4 (e4m3 scale per 16) and MXFP4 (E8M0 scale per 32).

Bit-exactness against ``gated_block_reference.fp4_quantize_rowwise_2d`` + ``mx_swizzle_sf_rowwise_padded`` is
the contract, on BOTH the packed codes (compared as uint8 -- torch 2.13 cannot cast to or from
``float4_e2m1fn_x2``) and the padded scale blob: the block-scale out projection binds the blob by
storage order and checks only its byte count, so a wrong byte order, a scale rounding corner or a
reciprocal-multiply that flips an e2m1 midpoint is numerically wrong downstream and never an error.
Hence ``torch.equal``.

Three tiers (the ``test_quantize_mxfp8.py`` shape):

* the constants both sides share, the e2m1 / e4m3 rounding corners, the blob byte-address formula
  pinned against the oracle's swizzle on random probes, the typed declines -- no GPU;
* an sm_107a trace-compile of both formats with an ``nvdisasm`` spill count -- any box whose
  cutlass-dsl knows ``sm_107a`` (>= 4.8.0.dev0) AND whose ``$CUDA_PATH/bin`` or ``$PATH`` carries an
  nvdisasm that decodes it.  Anything less is a SKIP, never a failure: the verdict must not depend
  on which toolkit ``CUDA_PATH`` names;
* numerics on a cc 10.x device (``cvt.rn.satfinite.e2m1x2.f32`` is sm_100a+); the block targets
  Rubin, so these run on the SM107 dev node.
"""

import glob
import os
import shutil
import struct
import subprocess
import sys
import textwrap
import zlib

import pytest
import torch

pytestmark = pytest.mark.L0

# The kernel module imports on any torch (its dtype constants are ``getattr`` -> None, ``check_torch_fp4_dtypes`` is
# the typed decline); the ORACLE and these tests view buffers as those dtypes, so the module SKIPS here -- before the
# oracle import, which would otherwise error at collection -- where torch lacks them.
_FP4_TORCH_DTYPE_NAMES = ("float4_e2m1fn_x2", "float8_e4m3fn", "float8_e8m0fnu")
if any(getattr(torch, n, None) is None for n in _FP4_TORCH_DTYPE_NAMES):
    pytest.skip(f"torch {torch.__version__} lacks one of {_FP4_TORCH_DTYPE_NAMES} (the fp4 codes / scale-blob storage dtypes)", allow_module_level=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # the block's oracle module (unique basename, see fa683e44)

from gated_block_reference import (  # noqa: E402
    E2M1_GRID,
    e2m1_codes,
    fp4_dequant_rowwise_2d,
    fp4_quantize_rowwise_2d,
    mx_sf_padded_dims,
    mx_swizzle_sf_rowwise_padded,
    mx_unswizzle_sf_rowwise,
    pack_e2m1,
    unpack_e2m1,
)

from cudnn.frost.tile_dsl.pointwise import E2M1_MAX_RCP_BITS, E4M3_MIN_SUBNORMAL_BITS  # noqa: E402
from cudnn.gated_attention_block import GatedAttentionBlockGeometry  # noqa: E402
from cudnn.gated_attention_block.api import _QuantizeFp4  # noqa: E402
from cudnn.gated_attention_block.kernels.proj_gemm import sf_blob_bytes, sf_padded_dims  # noqa: E402
from cudnn.gated_attention_block.kernels.quantize_fp4 import (  # noqa: E402
    COMPILE_OPTIONS,
    DEFAULT_THREADS_PER_CTA,
    ELEMS_PER_LANE,
    FMT_MXFP4,
    FMT_NVFP4,
    FORMATS,
    FP4_MAX_RCP_BITS,
    FP4_STORAGE_DTYPE,
    SF_TILE_ROWS,
    SF_TORCH_DTYPES,
    QuantizeFp4Recipe,
    check_torch_fp4_dtypes,
    compile_quantize_fp4,
    cta_sf_atoms,
    cta_sf_bytes,
    fp4_format,
    lanes_per_row,
    moved_bytes,
    n_row_tiles,
    run_quantize_fp4,
    sf_byte,
    sf_cols,
    sf_k4,
    validate_shape,
)

D = 256
H = 4  # h_q of the test geometry: K = H*D = 1024 -> 64 nvfp4 / 32 mxfp4 scale columns per row
K = H * D
_GEOM = dict(d_model=512, h_q=H, h_kv=2, d_head=D, rope_dim=64)
FORMAT_NAMES = (FMT_NVFP4, FMT_MXFP4)


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _nvdisasm_candidates():
    """Executables to TRY, most likely to decode sm_107a first; the probe verifies each and skips if none does.

    A public CUDA 13.x ``nvdisasm`` reports ``Cannot decode architecture 'SM107a'``, so the toolkit
    ``$CUDA_PATH`` names is tried first and the one on ``$PATH`` second -- both are hints, not authorities."""
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(os.path.join(os.environ["CUDA_PATH"], "bin", "nvdisasm"))
    on_path = shutil.which("nvdisasm")
    if on_path:
        cands.append(on_path)
    return [c for c in dict.fromkeys(cands) if os.path.isfile(c) and os.access(c, os.X_OK)]


def _sm107a_known_to_the_dsl() -> bool:
    """cutlass-dsl < 4.8.0.dev0 has no ``sm_107a`` (``Arch.from_string`` KeyError); FROST's floor is 4.7.0."""
    try:
        from cutlass.base_dsl.enums import Arch

        Arch.from_string("sm_107a")
        return True
    except Exception:
        return False


def _cc():
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None


requires_fp4_cvt = pytest.mark.skipif(
    _cc() is None or _cc() < (10, 0), reason=f"cvt.rn.satfinite.e2m1x2.f32 / ue8m0x2 need sm_100a+ (the block targets Rubin); found {_cc()}"
)


class _NamedFormat:
    """An enum-member stand-in: what the block's ``Fp4Format`` will hand the stage (``.name`` + ``block_size``)."""

    def __init__(self, name: str, block_size: int) -> None:
        self.name = name
        self.block_size = block_size


# ---------------------------------------------------------------------------
# Oracle plumbing
# ---------------------------------------------------------------------------


def _oracle(src: torch.Tensor, fmt: str):
    """``fp4_quantize_rowwise_2d`` over the ``[T, K]`` view of the block's ``[T, H, D]`` buffer + the padded blob.

    Returns ``(codes uint8 [T, K/2], blob uint8 flat)``."""
    _, block, _ = fp4_format(fmt)
    t = int(src.shape[0])
    codes, e = fp4_quantize_rowwise_2d(src.reshape(t, K), fmt)
    return codes, mx_swizzle_sf_rowwise_padded(e, block)


def _launch(fmt: str, src: torch.Tensor, *, as_fp4x2: bool = False):
    """Compile (cached), sentinel-fill sf (0xFF = e4m3 NaN / E8M0 NaN, which no finite input produces), run, sync."""
    r = compile_quantize_fp4(dtype_in=src.dtype, h=H, d=D, fmt=fmt)
    t = int(src.shape[0])
    dst4 = torch.full((t, K // 2), 0xFF, dtype=torch.uint8, device="cuda")
    if as_fp4x2:
        dst4 = dst4.view(FP4_STORAGE_DTYPE)
    sf = torch.full((sf_blob_bytes(t, K, r.block),), 0xFF, dtype=torch.uint8, device="cuda")
    run_quantize_fp4(r, src, dst4, sf, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return dst4.view(torch.uint8), sf


def _check_against_oracle(fmt: str, src: torch.Tensor, dst4: torch.Tensor, sf: torch.Tensor) -> None:
    ref_codes, ref_blob = _oracle(src, fmt)
    assert ref_blob.numel() == sf.numel(), (ref_blob.numel(), sf.numel())
    n_bad_c = int((dst4 != ref_codes).sum().item())
    n_bad_sf = int((sf != ref_blob).sum().item())
    assert n_bad_c == 0, f"{fmt}: {n_bad_c} / {dst4.numel()} code bytes differ from the oracle"
    assert n_bad_sf == 0, f"{fmt}: {n_bad_sf} / {sf.numel()} SF bytes differ from the oracle (byte ORDER or rounding)"
    assert int((sf == 0xFF).sum().item()) == 0, "SF sentinel survived: some scale bytes were never written"
    # The blob round-trips THROUGH the same reader the GEMM oracle uses: no NaN / inf can come out of finite data.
    # Read back in fp64: an MXFP4 block from bf16's top octave dequantizes to 4 x 2^126 = 2^128, which is finite
    # in fp64 and overflows fp32 (the same inf the GEMM's fp32 accumulator would produce from those codes).
    deq = fp4_dequant_rowwise_2d(dst4, sf, fmt, out_dtype=torch.float64)
    assert bool(torch.isfinite(deq).all())


def _tail_sf_bytes(t: int, block: int):
    """Absolute blob bytes that belong to pad rows (``row >= T`` in the last 128-row tile)."""
    offs = set()
    for row in range(t, n_row_tiles(t) * SF_TILE_ROWS):
        for k in range(0, K, block):
            offs.add(sf_byte(row, k, h=H, d=D, block=block))
    return sorted(offs)


# ---------------------------------------------------------------------------
# Tier 1: constants, rounding corners, blob formula, typed declines -- no GPU
# ---------------------------------------------------------------------------


def test_fp4_max_rcp_is_fp32_one_sixth():
    """The scale is ONE fp32 multiply by fp32(1/6) on both sides: the kernel's register constant and the oracle's tensor."""
    assert FP4_MAX_RCP_BITS == E2M1_MAX_RCP_BITS == 0x3E2AAAAB == _f32_bits(1.0 / 6.0)
    assert torch.tensor(1.0 / E2M1_GRID[-1], dtype=torch.float32).view(torch.int32).item() == 0x3E2AAAAB


def test_e2m1_rne_table_on_every_midpoint():
    """Nearest on ``{0, .5, 1, 1.5, 2, 3, 4, 6}``, ties to the EVEN code, saturate at 6, sign in bit 3."""
    mids = {0.25: 0, 0.75: 2, 1.25: 2, 1.75: 4, 2.5: 4, 3.5: 6, 5.0: 6}
    for mid, code in mids.items():
        assert e2m1_codes(torch.tensor([mid, -mid])).tolist() == [code, code | 0x8], mid
    assert e2m1_codes(torch.tensor(E2M1_GRID)).tolist() == list(range(8))
    assert e2m1_codes(torch.tensor([6.0, 6.5, 1e30, -7.0, -0.0])).tolist() == [7, 7, 7, 15, 8]
    packed = pack_e2m1(torch.arange(16, dtype=torch.uint8))
    assert packed.tolist() == [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]  # low nibble = even k
    assert torch.equal(unpack_e2m1(packed), torch.tensor(E2M1_GRID + tuple(-v for v in E2M1_GRID)))


def test_e4m3_scale_floor_corners():
    """``2^-9`` (``E4M3_MIN_SUBNORMAL_BITS``) is e4m3's smallest nonzero: one binade below rounds to ZERO -- the
    infinite-encode-scale hazard the NVFP4 floor exists for.  The oracle's zero block therefore scales at 0x01."""
    assert E4M3_MIN_SUBNORMAL_BITS == 0x3B000000 == _f32_bits(2.0**-9)
    e4m3 = torch.float8_e4m3fn
    assert torch.tensor(2.0**-10).to(e4m3).view(torch.uint8).item() == 0x00
    assert torch.tensor(2.0**-9).to(e4m3).view(torch.uint8).item() == 0x01
    codes, e = fp4_quantize_rowwise_2d(torch.zeros(2, 32), FMT_NVFP4)
    assert e.tolist() == [[0x01, 0x01], [0x01, 0x01]] and int(codes.max()) == 0
    codes, e = fp4_quantize_rowwise_2d(torch.zeros(2, 32), FMT_MXFP4)
    assert e.tolist() == [[0x00], [0x00]] and int(codes.max()) == 0


def test_formats_are_the_two_kernel_registry_pairs():
    assert FORMATS == {"nvfp4": (16, True), "mxfp4": (32, False)}
    assert SF_TORCH_DTYPES == {"nvfp4": torch.float8_e4m3fn, "mxfp4": torch.float8_e8m0fnu}
    assert fp4_format("nvfp4") == ("nvfp4", 16, True) and fp4_format("MXFP4") == ("mxfp4", 32, False)
    assert fp4_format(_NamedFormat("NVFP4", 16)) == ("nvfp4", 16, True)  # an enum member NAMED after the format
    assert fp4_format(_NamedFormat("MXFP4", 32)) == ("mxfp4", 32, False)
    with pytest.raises(ValueError, match="unknown fp4 format"):
        fp4_format("mxfp8")
    with pytest.raises(ValueError, match="unknown fp4 format"):
        fp4_format(None)
    with pytest.raises(ValueError, match="declares block_size=32"):
        fp4_format(_NamedFormat("NVFP4", 32))  # a format enum drifting from the kernel is a typed error, not a silent pick


@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_sf_blob_algebra_is_the_gemm_contract(fmt):
    """The blob is sized by ``proj_gemm.sf_blob_bytes`` -- rows padded to 128, scale columns to 4 -- and a head's
    ``D/block`` columns are ``D/(4*block)`` whole atoms, so a CTA's SF bytes are contiguous."""
    _, block, _ = fp4_format(fmt)
    assert sf_cols(H, D, block) == K // block
    assert sf_k4(H, D, block) == -(-(K // block) // 4) == sf_padded_dims(128, K, block)[1] // 4
    assert cta_sf_bytes(D, block) == 128 * D // block == cta_sf_atoms(D, block) * 512
    assert cta_sf_atoms(256, block) == (4 if block == 16 else 2)
    for t in (1, 127, 128, 129, 1000, 4096):
        assert n_row_tiles(t) == -(-t // 128)
        assert sf_blob_bytes(t, K, block) == n_row_tiles(t) * 128 * sf_k4(H, D, block) * 4
        assert sf_blob_bytes(t, K, block) == mx_sf_padded_dims(t, K, block)[0] * mx_sf_padded_dims(t, K, block)[1]
        assert moved_bytes(t, H, D, block) == t * K * 2 + t * K // 2 + sf_blob_bytes(t, K, block)
    assert sf_blob_bytes(1000, K, 16) == 2 * sf_blob_bytes(1000, K, 32)
    assert lanes_per_row(256) == 16 and lanes_per_row(512) == 32 and lanes_per_row(128) == 8
    assert ELEMS_PER_LANE == 16 and SF_TILE_ROWS == 128


@pytest.mark.parametrize("fmt", FORMAT_NAMES)
@pytest.mark.parametrize("t", [128, 384, 1000, 4096])
def test_sf_byte_formula_matches_the_oracle_blob(fmt, t):
    """``sf_byte(row, k)`` (band base + the shared atom rule) lands on the byte ``mx_swizzle_sf_rowwise_padded``
    (== ``to_blocked``) puts scale ``(row, k // block)`` at, on random probes; pad rows are 0x00 there."""
    _, block, _ = fp4_format(fmt)
    g = torch.Generator().manual_seed(t * block)
    e = torch.randint(1, 255, (t, K // block), dtype=torch.uint8, generator=g)
    blob = mx_swizzle_sf_rowwise_padded(e, block)
    assert blob.numel() == sf_blob_bytes(t, K, block)
    for _ in range(2000):
        row, k = int(torch.randint(0, t, (1,), generator=g)), int(torch.randint(0, K, (1,), generator=g))
        assert int(blob[sf_byte(row, k, h=H, d=D, block=block)]) == int(e[row, k // block]), (row, k)
    # one CTA (row tile, head) owns cta_sf_bytes CONTIGUOUS bytes at atom (row_tile * sf_k4 + head * atoms)
    for row_tile in (0, n_row_tiles(t) - 1):
        for head in range(H):
            lo = (row_tile * sf_k4(H, D, block) + head * cta_sf_atoms(D, block)) * 512
            offs = {sf_byte(row_tile * 128 + r, head * D + k, h=H, d=D, block=block) for r in range(128) for k in range(0, D, block)}
            assert offs == set(range(lo, lo + cta_sf_bytes(D, block))), (row_tile, head)
    tail = _tail_sf_bytes(t, block)
    assert len(tail) == (n_row_tiles(t) * 128 - t) * (K // block)
    if tail:
        assert int(blob[tail].max()) == 0
    assert torch.equal(mx_unswizzle_sf_rowwise(blob, t, K, block), e)


def test_validate_shape_declines_what_the_lanes_cannot_cover():
    for block in (16, 32):
        validate_shape(256, 256, block)
        validate_shape(512, 256, block)
        validate_shape(128, 256, block)
        with pytest.raises(ValueError, match="multiple of 4\\*block"):
            validate_shape(96, 256, block)
        with pytest.raises(ValueError, match="positive multiple"):
            validate_shape(0, 256, block)
        with pytest.raises(ValueError, match="positive multiple"):
            validate_shape(-256, 256, block)
        with pytest.raises(ValueError, match="multiple of 32"):
            validate_shape(256, 40, block)
        with pytest.raises(ValueError, match="up to 1024"):
            validate_shape(256, 2048, block)
        with pytest.raises(ValueError, match="rows per pass"):
            validate_shape(256, 160, block)  # 10 rows/pass does not divide 128 (160 lanes do cover the 128-lane nvfp4 burst)
        with pytest.raises(ValueError, match="burst"):
            validate_shape(256, 32, block)  # one warp cannot burst a 1-2 KiB SF tile
    validate_shape(64, 256, 16)  # 4 lanes/row, 64 rows/pass
    with pytest.raises(ValueError, match="multiple of 4\\*block = 128"):
        validate_shape(64, 256, 32)  # one head = half an atom
    with pytest.raises(ValueError, match="divide a warp"):
        validate_shape(192, 256, 16)  # 12 lanes/row
    with pytest.raises(ValueError, match="burst"):
        validate_shape(1024, 64, 16)  # 8 KiB SF tile needs 512 burst lanes
    with pytest.raises(ValueError, match="block must be 16"):
        validate_shape(256, 256, 64)


def test_compile_declines_bad_inputs_before_tracing():
    with pytest.raises(ValueError, match="bf16/f16"):
        compile_quantize_fp4(dtype_in=torch.float32, h=2, d=256, fmt=FMT_NVFP4)
    with pytest.raises(ValueError, match="unknown fp4 format"):
        compile_quantize_fp4(dtype_in=torch.bfloat16, h=2, d=256, fmt="fp4")
    with pytest.raises(ValueError, match="positive"):
        compile_quantize_fp4(dtype_in=torch.bfloat16, h=0, d=256, fmt=FMT_MXFP4)
    with pytest.raises(ValueError, match="multiple of 4\\*block"):
        compile_quantize_fp4(dtype_in=torch.bfloat16, h=2, d=64, fmt=FMT_MXFP4)
    with pytest.raises(ValueError, match="not built"):
        run_quantize_fp4(QuantizeFp4Recipe(torch.bfloat16, 2, 256, FMT_NVFP4), None, None, None, stream=0)


def test_missing_torch_storage_dtype_is_a_typed_decline(monkeypatch):
    """A torch without ``float4_e2m1fn_x2`` (or the format's scale dtype) imports the module fine and gets a
    ``ValueError`` naming the torch version and the missing dtype from compile AND run -- not an ``AttributeError``
    at import, and not an untyped escape at the launcher's ``.view``.  Simulated by blanking the module constants."""
    import cudnn.gated_attention_block.kernels.quantize_fp4 as qf

    for fmt in FORMAT_NAMES:
        check_torch_fp4_dtypes(fmt)  # this torch has them all: no raise
    monkeypatch.setattr(qf, "FP4_STORAGE_DTYPE", None)
    with pytest.raises(ValueError, match="torch .* has no torch.float4_e2m1fn_x2") as ei:
        compile_quantize_fp4(dtype_in=torch.bfloat16, h=H, d=D, fmt=FMT_NVFP4)
    assert torch.__version__ in str(ei.value) and "nvfp4" in str(ei.value)
    r = QuantizeFp4Recipe(torch.bfloat16, H, D, FMT_MXFP4, compiled=object())
    with pytest.raises(ValueError, match="torch.float4_e2m1fn_x2"):
        run_quantize_fp4(r, torch.zeros(8, H, D, dtype=torch.bfloat16), None, None, stream=0)
    monkeypatch.setattr(qf, "FP4_STORAGE_DTYPE", FP4_STORAGE_DTYPE)
    monkeypatch.setattr(qf, "SF_TORCH_DTYPES", dict(SF_TORCH_DTYPES, **{FMT_MXFP4: None}))
    with pytest.raises(ValueError, match="torch.float8_e8m0fnu"):
        compile_quantize_fp4(dtype_in=torch.bfloat16, h=H, d=D, fmt=FMT_MXFP4)
    check_torch_fp4_dtypes(FMT_NVFP4)  # the other format's dtypes are untouched


def test_recipe_derives_its_format_facts():
    r = QuantizeFp4Recipe(torch.bfloat16, H, D, FMT_NVFP4, compiled=object())
    assert (r.block, r.sf_e4m3, r.sf_torch_dtype, r.k, r.threads_per_cta) == (16, True, torch.float8_e4m3fn, K, DEFAULT_THREADS_PER_CTA)
    r = QuantizeFp4Recipe(torch.float16, H, D, FMT_MXFP4, compiled=object())
    assert (r.block, r.sf_e4m3, r.sf_torch_dtype) == (32, False, torch.float8_e8m0fnu)


def test_run_declines_a_mismatched_binding_before_the_device():
    """Every host check that guards a wild write fires on CPU tensors too (before the ``is_cuda`` check), so it is
    pinned without a device.  The recipe carries a stand-in artifact: no check below may reach it."""
    t = 256
    r = QuantizeFp4Recipe(torch.bfloat16, H, D, FMT_NVFP4, compiled=object())
    x = torch.zeros(t, H, D, dtype=torch.bfloat16)
    dst4 = torch.zeros(t, K // 2, dtype=torch.uint8)
    sf = torch.zeros(sf_blob_bytes(t, K, 16), dtype=torch.uint8)
    with pytest.raises(ValueError, match="compiled for"):
        run_quantize_fp4(r, x.to(torch.float16), dst4, sf, stream=0)
    with pytest.raises(ValueError, match="H=4"):
        run_quantize_fp4(r, torch.zeros(t, 2, D, dtype=torch.bfloat16), dst4, sf, stream=0)
    with pytest.raises(ValueError, match="compact"):
        run_quantize_fp4(r, torch.zeros(t, 2 * H, D, dtype=torch.bfloat16)[:, ::2], dst4, sf, stream=0)
    with pytest.raises(ValueError, match="uint8 or torch.float4_e2m1fn_x2"):
        run_quantize_fp4(r, x, dst4.view(torch.float8_e4m3fn), sf, stream=0)
    with pytest.raises(ValueError, match="T\\*H\\*D/2"):
        run_quantize_fp4(r, x, dst4[:, :-8], sf, stream=0)  # 8 bytes short per row, and non-contiguous
    with pytest.raises(ValueError, match="T\\*H\\*D/2"):
        run_quantize_fp4(r, x, torch.zeros(t, K, dtype=torch.uint8)[:, ::2], sf, stream=0)  # the right byte count, strided
    with pytest.raises(ValueError, match="T\\*H\\*D/2"):
        run_quantize_fp4(r, x, torch.zeros(t, K, dtype=torch.uint8), sf, stream=0)  # unpacked e4m3-sized buffer
    with pytest.raises(ValueError, match="uint8 or torch.float8_e4m3fn"):
        run_quantize_fp4(r, x, dst4, sf.view(torch.float8_e8m0fnu), stream=0)  # the OTHER format's scale dtype
    with pytest.raises(ValueError, match="sf_blob_bytes"):
        run_quantize_fp4(r, x, dst4, sf[:-1], stream=0)
    with pytest.raises(ValueError, match="sf_blob_bytes"):
        run_quantize_fp4(r, x, dst4, torch.zeros(sf_blob_bytes(t, K, 32), dtype=torch.uint8), stream=0)  # the mxfp4-sized blob
    with pytest.raises(ValueError, match="CUDA tensor"):
        run_quantize_fp4(r, x, dst4, sf, stream=0)
    r32 = QuantizeFp4Recipe(torch.bfloat16, H, D, FMT_MXFP4, compiled=object())
    with pytest.raises(ValueError, match="uint8 or torch.float8_e8m0fnu"):
        run_quantize_fp4(r32, x, dst4, sf.view(torch.float8_e4m3fn), stream=0)


def test_stage_check_support_declines_before_compile():
    """``_QuantizeFp4`` (api.py, built by no pipeline yet): the typed declines fire at declaration, never at launch."""
    geom = GatedAttentionBlockGeometry(**_GEOM)
    for fmt in (_NamedFormat("NVFP4", 16), _NamedFormat("MXFP4", 32), FMT_NVFP4):
        st = _QuantizeFp4(geom, batch=2, seq_len=500, dtype_in=torch.bfloat16, heads=H, fmt=fmt, name="quantize_fp4")
        st.check_support()
        _, block, _ = fp4_format(fmt)
        assert st.rows() == 1000 and st.code_bytes() == 1000 * K // 2
        assert st.sf_bytes() == sf_blob_bytes(1000, K, block)
        assert st.moved_bytes() == moved_bytes(1000, H, D, block)
        with pytest.raises(RuntimeError, match="compile\\(\\) before execute"):
            st.execute(None, None, None, current_stream=0)
    with pytest.raises(NotImplementedError, match="bf16/f16"):
        _QuantizeFp4(geom, batch=1, seq_len=128, dtype_in=torch.float32, heads=H, fmt=FMT_NVFP4, name="q").check_support()
    with pytest.raises(ValueError, match="unknown fp4 format"):
        _QuantizeFp4(geom, batch=1, seq_len=128, dtype_in=torch.bfloat16, heads=H, fmt="mxfp8", name="q").check_support()
    with pytest.raises(ValueError, match="declares block_size"):
        _QuantizeFp4(geom, batch=1, seq_len=128, dtype_in=torch.bfloat16, heads=H, fmt=_NamedFormat("MXFP4", 16), name="q").check_support()
    small = GatedAttentionBlockGeometry(**{**_GEOM, "d_head": 64, "rope_dim": 32})
    _QuantizeFp4(small, batch=1, seq_len=128, dtype_in=torch.bfloat16, heads=H, fmt=FMT_NVFP4, name="q").check_support()  # 64 % 64 == 0
    with pytest.raises(NotImplementedError, match="4\\*block = 128"):
        _QuantizeFp4(small, batch=1, seq_len=128, dtype_in=torch.bfloat16, heads=H, fmt=FMT_MXFP4, name="q").check_support()
    odd = GatedAttentionBlockGeometry(**{**_GEOM, "d_head": 192, "rope_dim": 64})
    with pytest.raises(ValueError, match="divide a warp"):
        _QuantizeFp4(odd, batch=1, seq_len=128, dtype_in=torch.bfloat16, heads=H, fmt=FMT_NVFP4, name="q").check_support()


# ---------------------------------------------------------------------------
# Tier 2: sm_107a trace-compile + spill count -- any box with a DSL that knows sm_107a
# ---------------------------------------------------------------------------

_SASS_PROBE = textwrap.dedent("""
    import glob, os, subprocess, sys
    fmt, dump, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump  # read once, at the first cutlass import
    import torch
    from cudnn.gated_attention_block.kernels.quantize_fp4 import COMPILE_OPTIONS, compile_quantize_fp4
    # --keep-cubin, NOT --keep-sass: the latter runs the DSL's own wheel nvdisasm, which ICEs on sm_107a.
    compile_quantize_fp4(dtype_in=torch.bfloat16, h=4, d=256, fmt=fmt, compile_options=COMPILE_OPTIONS + " --gpu-arch sm_107a --keep-cubin")
    print("COMPILED", fmt)
    cubins = glob.glob(os.path.join(dump, "*.sm_107a.cubin"))
    if not cubins:
        print("FAIL no .sm_107a.cubin landed in", dump)
        sys.exit(3)
    sass = None
    for nvd in cands:
        try:
            proc = subprocess.run([nvd, "-c", cubins[0]], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:  # vanished / not executable / wedged: the next candidate
            print("REJECT", nvd, "->", repr(exc))
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            sass = proc.stdout.splitlines()
            print("NVDISASM", nvd)
            break
        print("REJECT", nvd, "->", (proc.stderr.strip().splitlines() or [str(proc.returncode)])[-1])
    if sass is None:
        print("SKIP no nvdisasm candidate decodes sm_107a")
        sys.exit(0)
    print("SPILL", sum(1 for ln in sass if "STL" in ln or "LDL" in ln))
    print("E2M1", sum(1 for ln in sass if "F2FP" in ln and "E2M1" in ln))
    print("E4M3", sum(1 for ln in sass if "F2FP" in ln and "E4M3.F32" in ln))
    print("E8M0", sum(1 for ln in sass if "F2FP" in ln and ".E8." in ln and ".RP" in ln))
    print("FCHK", sum(1 for ln in sass if "FCHK" in ln))
    print("MUFU_RCP", sum(1 for ln in sass if "MUFU.RCP" in ln))
    print("FMNMX3", sum(1 for ln in sass if "FMNMX3" in ln))
    print("SHFL", sum(1 for ln in sass if "SHFL" in ln))
    print("LINES", len(sass))
    """)

_SASS_KEYS = ("SPILL", "E2M1", "E4M3", "E8M0", "FCHK", "MUFU_RCP", "FMNMX3", "SHFL", "LINES")
_PASSES = SF_TILE_ROWS // (DEFAULT_THREADS_PER_CTA // lanes_per_row(D))  # 8 row passes per lane at 256 threads / D=256


@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_sm107_trace_compile_has_no_spills(fmt, tmp_path):
    """Compile for Rubin here (no device match needed), decode with an nvdisasm that knows sm_107a: STL/LDL must be 0.

    Also pins the instruction shape (all LOWER bounds -- ptxas may duplicate): the codes come from the hardware
    ``F2FP...E2M1`` (8 per 16 elements per pass); NVFP4 scales are ``F2FP...E4M3.F32`` and every element is divided
    by the IEEE ``div.rn.f32`` lowering -- at least one ``FCHK`` range check per element (128 at 8 passes x 16; a
    ptxas that duplicates or merges a range check must not turn this red) and at least as many ``MUFU.RCP``
    seeds, never a bare reciprocal-multiply (which flips e2m1 midpoints); MXFP4 scales are the
    ``cvt.rp...ue8m0x2`` and there is NO division and one ``SHFL`` (the lane pair) per pass; the abs-max tree
    fuses (FMNMX3 > 0).  ``SPILL == 0`` is the load-bearing assertion.

    SKIPS (never fails) when the DSL predates ``sm_107a`` or no candidate nvdisasm decodes it -- an L0 verdict
    must not turn on which toolkit ``CUDA_PATH`` happens to name."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    dump = tmp_path / f"quantize_fp4_{fmt}"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _SASS_PROBE, fmt, str(dump), *cands], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    lines = proc.stdout.splitlines()
    assert f"COMPILED {fmt}" in lines, proc.stdout[-2000:]
    assert glob.glob(str(dump / "*.sm_107a.cubin")), "no .sm_107a.cubin landed in CUTE_DSL_DUMP_DIR"
    if not cands:
        pytest.skip("compiled; no nvdisasm executable to try for the SASS half (CUDA_PATH unset and none on PATH)")
    if any(ln.startswith("SKIP") for ln in lines):
        pytest.skip(f"compiled; SASS half skipped: {[ln for ln in lines if ln.startswith(('SKIP', 'REJECT'))]}")
    stats = {k: int(v) for k, v in (ln.split() for ln in lines if ln.split() and ln.split()[0] in _SASS_KEYS)}
    print(f"\n[{fmt}] sm_107a SASS: {stats} via {[ln for ln in lines if ln.startswith('NVDISASM')]}")
    assert stats["SPILL"] == 0, f"{fmt}: {stats['SPILL']} STL/LDL in the sm_107a cubin"
    assert stats["E2M1"] >= _PASSES * ELEMS_PER_LANE // 2, f"{fmt}: fewer e2m1 cvts than the source issues -- {stats}"
    assert stats["FMNMX3"] > 0, stats
    if fmt == FMT_NVFP4:
        assert stats["E4M3"] >= _PASSES and stats["E8M0"] == 0, stats
        assert stats["FCHK"] >= _PASSES * ELEMS_PER_LANE, f"nvfp4: expected at least one FCHK (div.rn range check) per element -- {stats}"
        assert stats["MUFU_RCP"] >= stats["FCHK"], f"nvfp4: every FCHK pairs with a MUFU.RCP seed -- {stats}"
    else:
        assert stats["E8M0"] >= _PASSES and stats["E4M3"] == 0, stats
        assert stats["FCHK"] == 0 and stats["MUFU_RCP"] == 0, f"mxfp4: a power-of-two scale needs no division -- {stats}"
        assert stats["SHFL"] >= _PASSES, f"mxfp4: one bfly(1) per pass completes the lane pair's amax -- {stats}"


# ---------------------------------------------------------------------------
# Tier 3: numerics on cc 10.x
# ---------------------------------------------------------------------------


@requires_fp4_cvt
@pytest.mark.parametrize("t", [128, 1000, 4096])
@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_bit_exact_vs_oracle(fmt, t):
    """Compact ``[T, H, D]`` bf16 in; packed e2m1 codes AND the padded F8_128x4 blob equal the oracle byte for byte.

    T=1000 is the tail (104 live rows in the last tile: pad rows' SF 0x00, no code store past T); T=4096 is
    32 row tiles x 4 heads."""
    _, block, _ = fp4_format(fmt)
    torch.manual_seed(zlib.crc32(f"{fmt}-{t}".encode()) & 0xFFFF)
    x = (torch.randn(t, H, D, device="cuda") * 3.0).to(torch.bfloat16)
    dst4, sf = _launch(fmt, x)
    _check_against_oracle(fmt, x, dst4, sf)
    tail = _tail_sf_bytes(t, block)
    if tail:
        assert int(sf[tail].max().item()) == 0, "pad-row SF bytes must be 0x00"


@requires_fp4_cvt
@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_adversarial_midpoints_are_bit_exact(fmt):
    """Blocks built so ``x / scale`` lands EXACTLY on e2m1 rounding midpoints (bf16-exact values), where a reciprocal-
    multiply would flip a whole code.  NVFP4: amax 2.8125 -> scale 0.46875 (e4m3), elements 0.46875 x {midpoints}
    (``0.5859375 / 0.46875 == 1.25``, the `_nvfp4` corner); MXFP4: amax 3 -> E8M0 2^-1, elements 0.5 x {midpoints}
    (and amax 6 -> scale 1, the midpoints verbatim).  Also asserts the case IS adversarial: an rcp-multiply
    emulation of the same inputs disagrees with the oracle on at least one code."""
    _, block, _ = fp4_format(fmt)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    # (amax, the scale it yields): amax * fp32(1/6) rounds to an e4m3 / E8M0 value whose midpoint multiples are bf16-exact.
    pairs = [(2.8125, 0.46875), (5.625, 0.9375)] if fmt == FMT_NVFP4 else [(3.0, 0.5), (6.0, 1.0)]
    t = 256
    x = torch.zeros(t, K, dtype=torch.float32)
    g = torch.Generator().manual_seed(9)
    for row in range(t):
        amax, scale = pairs[(row // 2) % 2]
        for b0 in range(0, K, block):
            blk = torch.zeros(block)
            blk[0 if block == 16 else 16 * ((row + b0 // block) % 2)] = amax if row % 2 == 0 else -amax  # mxfp4: the amax alternates lanes of the pair
            n = min(mids.numel(), block - 1)
            perm = torch.randperm(mids.numel(), generator=g)[:n]
            signs = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0)
            slots = [i for i in range(block) if blk[i] == 0][:n]
            blk[slots] = scale * mids[perm] * signs
            x[row, b0 : b0 + block] = blk
    xb = x.to(torch.bfloat16)
    assert torch.equal(xb.float(), x), "every adversarial value must be bf16-exact"
    ref_codes, ref_e = fp4_quantize_rowwise_2d(xb.reshape(t, K), fmt)
    # the case is adversarial: an rcp-multiply on the SAME fp32 inputs flips codes the true division keeps
    x2 = xb.float().reshape(t, K // block, block)
    if fmt == FMT_NVFP4:
        s = ref_e.view(torch.float8_e4m3fn).float().unsqueeze(-1)
        assert sorted(set(s.flatten().tolist())) == [p[1] for p in pairs], "the scales must be the intended e4m3 values"
        rcp_codes = pack_e2m1(e2m1_codes(x2 * (torch.tensor(1.0) / s)).reshape(t, K))
        assert int((rcp_codes != ref_codes).sum()) > 0, "expected the rcp-multiply emulation to flip at least one midpoint"
    else:
        assert sorted(set(ref_e.flatten().tolist())) == [126, 127], "the E8M0 exponents must be 2^-1 and 2^0"
    xd = xb.reshape(t, H, D).contiguous().cuda()
    dst4, sf = _launch(fmt, xd)
    _check_against_oracle(fmt, xd, dst4, sf)
    assert torch.equal(dst4.cpu(), ref_codes)


@requires_fp4_cvt
@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_zero_rows_and_tail_rows(fmt):
    """A VALID all-zero row (a dead ragged entry) quantizes to codes 0 with the format's zero-block scale (NVFP4 0x01 =
    the e4m3 floor, MXFP4 0x00); the PAD rows past T carry SF 0x00 in BOTH formats (the padded-blob contract, not
    the floor) and no code byte is written for them."""
    _, block, _ = fp4_format(fmt)
    t = 1000
    torch.manual_seed(13)
    x = (torch.randn(t, H, D, device="cuda") * 2.0).to(torch.bfloat16)
    x[100:110] = 0
    x[999] = 0
    dst4, sf = _launch(fmt, x)
    _check_against_oracle(fmt, x, dst4, sf)
    zero_sf = 0x01 if fmt == FMT_NVFP4 else 0x00
    for row in (100, 105, 109, 999):
        assert int(dst4[row].max()) == 0
        for k in range(0, K, block):
            assert int(sf[sf_byte(row, k, h=H, d=D, block=block)]) == zero_sf, (row, k)
    tail = _tail_sf_bytes(t, block)
    assert len(tail) == 24 * (K // block) and int(sf[tail].max()) == 0
    assert int(sf.ne(0).sum()) > 0.9 * (sf.numel() - len(tail) - 11 * (K // block) * (fmt == FMT_MXFP4))


@requires_fp4_cvt
@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_f16_source_and_fp4x2_destination_view(fmt):
    """An fp16 source; the destination handed over as ``float4_e2m1fn_x2`` (what the GEMM binds) -- same bytes."""
    torch.manual_seed(17)
    t = 384
    x = (torch.randn(t, H, D, device="cuda") * 40.0).to(torch.float16)
    d_u8, sf_u8 = _launch(fmt, x)
    _check_against_oracle(fmt, x, d_u8, sf_u8)
    d_x2, sf_x2 = _launch(fmt, x, as_fp4x2=True)
    assert torch.equal(d_u8, d_x2) and torch.equal(sf_u8, sf_x2)


@requires_fp4_cvt
@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_large_and_tiny_magnitudes(fmt):
    """Saturation and the small-block regime: |x| up to bf16's 3e38 (NVFP4 scale saturates at 448, MXFP4 exponent
    at its cap) and blocks around the NVFP4 2^-9 floor -- bit-exact either way, no NaN / inf out."""
    torch.manual_seed(19)
    t = 256
    x = torch.randn(t, H, D, device="cuda")
    x[:64] *= 1e30
    x[64:128] *= 2.0**-12
    x[128:192] *= 2.0**-8
    x[192, 0, 0] = 3.0e38
    x[193, 0, 0] = -3.0e38
    xb = x.to(torch.bfloat16)
    assert bool(torch.isfinite(xb).all())
    dst4, sf = _launch(fmt, xb)
    _check_against_oracle(fmt, xb, dst4, sf)


@requires_fp4_cvt
@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_two_launches_are_bitwise_identical(fmt):
    torch.manual_seed(5)
    x = torch.randn(1000, H, D, device="cuda").to(torch.bfloat16)
    d1, s1 = _launch(fmt, x)
    d2, s2 = _launch(fmt, x)
    assert torch.equal(d1, d2) and torch.equal(s1, s2)


@requires_fp4_cvt
@pytest.mark.parametrize("fmt", FORMAT_NAMES)
def test_stage_class_matches_the_direct_run(fmt):
    """``_QuantizeFp4`` compiles and launches the same artifact: its outputs equal the direct ``run_quantize_fp4``."""
    geom = GatedAttentionBlockGeometry(**_GEOM)
    st = _QuantizeFp4(geom, batch=2, seq_len=500, dtype_in=torch.bfloat16, heads=H, fmt=_NamedFormat(fmt.upper(), FORMATS[fmt][0]), name="quantize_fp4")
    st.check_support()
    st.compile()
    torch.manual_seed(23)
    x = torch.randn(1000, H, D, device="cuda").to(torch.bfloat16)
    dst4 = torch.full((st.code_bytes(),), 0xFF, dtype=torch.uint8, device="cuda").view(FP4_STORAGE_DTYPE).reshape(1000, K // 2)
    sf = torch.full((st.sf_bytes(),), 0xFF, dtype=torch.uint8, device="cuda")
    st.execute(x, dst4, sf, current_stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    d_ref, s_ref = _launch(fmt, x)
    assert torch.equal(dst4.view(torch.uint8), d_ref) and torch.equal(sf, s_ref)


@requires_fp4_cvt
def test_run_declines_a_misaligned_source_on_device():
    r = compile_quantize_fp4(dtype_in=torch.bfloat16, h=H, d=D, fmt=FMT_NVFP4)
    t = 128
    slab = torch.zeros(t * K + 8, dtype=torch.bfloat16, device="cuda")
    x = torch.as_strided(slab, (t, H, D), (K, D, 1), storage_offset=4)  # compact strides, base 8 B off a 16-B line
    assert x.is_contiguous() and x.data_ptr() % 16 == 8
    dst4 = torch.zeros(t, K // 2, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(sf_blob_bytes(t, K, 16), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="16-byte aligned"):
        run_quantize_fp4(r, x, dst4, sf, stream=torch.cuda.current_stream().cuda_stream)
