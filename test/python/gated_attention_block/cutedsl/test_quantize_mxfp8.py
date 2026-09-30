# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The block's MXFP8 quantize pass (``kernels/quantize_mxfp8.py``): e4m3 codes + F8_128x4 E8M0 scales.

Bit-exactness against ``test/python/sdpa/mxfp8_quant.py::quantize_to_mxfp8`` is the
contract, on BOTH the data and the uint8 scale-factor view: the production Rubin MXFP8
SDPA consumes exactly the oracle's layout and values, the adapter binds the SF blob by
storage order and checks only its byte count, so a wrong byte order or a rounding
corner is numerically wrong downstream and never an error.  Hence ``torch.equal``.

Three tiers:

* shape algebra + the two section-2.3 byte-address formulas pinned against the oracle's
  ``_swizzle_128x4`` / ``swizzle_sf_columnwise`` on random probes -- no GPU;
* an sm_107a trace-compile of both arms with an ``nvdisasm`` spill count -- any box whose
  cutlass-dsl knows ``sm_107a`` (>= 4.8.0.dev0) AND whose ``$CUDA_PATH/bin`` or ``$PATH``
  carries an nvdisasm that decodes it (an A100 box compiles Rubin cubins in a second).
  Anything less is a SKIP, never a failure: the verdict must not depend on which toolkit
  ``CUDA_PATH`` names;
* numerics on a cc 10.x device (``cvt.rp.satfinite.ue8m0x2.f32`` is sm_100+); the block
  targets Rubin, so these run on the SM107 dev node.
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

from cudnn.frost.buffers import cutedsl_requirement_error

requirement_error = cutedsl_requirement_error("Gated attention block tests")
if requirement_error:
    pytest.skip(requirement_error, allow_module_level=True)

pytestmark = pytest.mark.L0

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))  # test/python: the sdpa oracle package

from sdpa.mxfp8_quant import (  # noqa: E402
    _swizzle_128x4,
    e8m0_ceil,
    quantize_blocks,
    quantize_to_mxfp8,
    swizzle_sf_columnwise,
    swizzle_sf_rowwise,
)

from cudnn.frost.tile_dsl.pointwise import E8M0_RCP_E4M3_MAX_BITS  # noqa: E402
from cudnn.gated_attention_block.kernels.quantize_mxfp8 import (  # noqa: E402
    AXIS_COL,
    AXIS_ROW,
    COMPILE_OPTIONS,
    QuantizeMxfp8Recipe,
    compile_quantize_mxfp8,
    lanes_per_row,
    moved_bytes,
    n_sf_tiles,
    run_quantize_mxfp8,
    sf_byte_columnwise,
    sf_byte_rowwise,
    sf_bytes,
    sf_tile_bytes,
    validate_shape,
)

D = 256
N_QKVG = 17408  # the 397B slab width: Q [0, 4*256), K [1024, 1536), V [1536, 2048) at h_q=4 / h_kv=2
ROLES = {"q": (AXIS_ROW, 4, 0), "k": (AXIS_ROW, 2, 1024), "v": (AXIS_COL, 2, 1536)}  # role -> (axis, heads, slab column offset)


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


requires_mx_cvt = pytest.mark.skipif(
    _cc() is None or _cc() < (10, 0), reason=f"cvt.rp.satfinite.ue8m0x2.f32 needs sm_100+ (the block targets Rubin); found {_cc()}"
)


# ---------------------------------------------------------------------------
# Oracle plumbing
# ---------------------------------------------------------------------------


def _oracle(src_thd: torch.Tensor, b: int, s: int, h: int):
    """``quantize_to_mxfp8`` on the block's ``[T, H, D]`` buffer, permuted to BHSD.

    Returns ``(row_data [T,H,D] e4m3, row_sf flat uint8, col_data [T,H,D] e4m3, col_sf flat uint8)``."""
    bhsd = src_thd.reshape(b, s, h, D).permute(0, 2, 1, 3)
    row_d, _, row_sf, col_d, _, col_sf = quantize_to_mxfp8(bhsd, b, h, s, D, with_ref=False)
    to_thd = lambda x: x.permute(0, 2, 1, 3).reshape(b * s, h, D)  # noqa: E731
    return to_thd(row_d), row_sf.reshape(-1), to_thd(col_d), col_sf.reshape(-1)


def _launch(axis: str, src: torch.Tensor, *, batch: int, seq_len: int, h: int):
    """Compile (cached), sentinel-fill dst (0xFF = e4m3 NaN) and sf (0xFF = E8M0 NaN), run, sync."""
    r = compile_quantize_mxfp8(dtype_in=src.dtype, h=h, d=D, axis=axis)
    dst = torch.empty(batch * seq_len, h, D, dtype=torch.float8_e4m3fn, device="cuda")
    dst.view(torch.uint8).fill_(0xFF)
    sf = torch.full((sf_bytes(batch, h, seq_len, D),), 0xFF, dtype=torch.uint8, device="cuda")
    run_quantize_mxfp8(r, src, dst, sf, batch=batch, seq_len=seq_len, stream=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return dst, sf


def _check_against_oracle(axis: str, src: torch.Tensor, dst: torch.Tensor, sf: torch.Tensor, *, batch: int, seq_len: int, h: int) -> None:
    row_d, row_sf, col_d, col_sf = _oracle(src, batch, seq_len, h)
    ref_d, ref_sf = (row_d, row_sf) if axis == AXIS_ROW else (col_d, col_sf)
    got_d = dst.view(torch.uint8)
    assert ref_sf.numel() == sf.numel(), (ref_sf.numel(), sf.numel())
    n_bad_d = int((got_d != ref_d.view(torch.uint8)).sum().item())
    n_bad_sf = int((sf != ref_sf).sum().item())
    assert n_bad_d == 0, f"{axis}: {n_bad_d} / {got_d.numel()} e4m3 codes differ from the oracle"
    assert n_bad_sf == 0, f"{axis}: {n_bad_sf} / {sf.numel()} SF bytes differ from the oracle (byte ORDER or rounding)"
    # 100 % of both outputs written: a 0xFF sentinel is a NaN in either format and cannot come from finite data.
    assert int((got_d == 0xFF).sum().item()) == 0, "e4m3 sentinel survived: some data cells were never written"
    assert int((sf == 0xFF).sum().item()) == 0, "SF sentinel survived: some scale bytes were never written (E8M0 NaN under the SDPA's whole-tile TMA)"


def _tail_sf_bytes(axis: str, *, batch: int, h: int, seq_len: int):
    """Absolute SF byte offsets that belong to pad rows (``s >= S`` in the last tile), per section 2.3."""
    tiles = n_sf_tiles(seq_len)
    offs = set()
    for b in range(batch):
        for hh in range(h):
            if axis == AXIS_ROW:  # every pad ROW owns D/32 scales
                for s in range(seq_len, tiles * 128):
                    for d_idx in range(0, D, 32):
                        offs.add(sf_byte_rowwise(b, hh, s, d_idx, n_heads=h, n_tiles=tiles, d=D))
            else:  # only a 32-token block that is ENTIRELY padding is 0x00 (a straddling block keeps its live rows' amax)
                for s in range(-(-seq_len // 32) * 32, tiles * 128, 32):
                    for dd in range(D):
                        offs.add(sf_byte_columnwise(b, hh, s, dd, n_heads=h, n_tiles=tiles, batch=batch))
    return sorted(offs)


# ---------------------------------------------------------------------------
# Tier 1: shape algebra + layout contracts -- no GPU
# ---------------------------------------------------------------------------


def test_rcp_448_is_the_exact_fp32_rounding():
    """The scale is ONE fp32 multiply by fp32(1/448); the oracle does the same (``torch.tensor(1/448, float32)``)."""
    bits = struct.unpack("<I", struct.pack("<f", 1.0 / 448.0))[0]
    assert bits == 0x3B124925 == E8M0_RCP_E4M3_MAX_BITS
    assert torch.tensor(1.0 / 448.0, dtype=torch.float32).view(torch.int32).item() == 0x3B124925


@pytest.mark.parametrize(
    "amax, byte, rcp",
    [
        (0.0, 0x00, 2.0**127),
        (448.0, 0x7F, 1.0),
        (449.0, 0x80, 0.5),
        (896.0, 0x80, 0.5),
        (224.0, 0x7E, 2.0),
        (2.0, 0x78, 128.0),
        (1.0, 0x77, 256.0),
        (0.5, 0x76, 512.0),
    ],
)
def test_e8m0_oracle_corners(amax, byte, rcp):
    """The corners the kernel must reproduce: 448 * fp32(1/448) == 1.0 exactly (no round-up), 449 rounds UP, powers of two stay."""
    x = torch.zeros(1, 32)
    x[0, 0] = amax
    _, e = quantize_blocks(x, torch.float8_e4m3fn)
    assert int(e.item()) == byte
    assert e8m0_ceil(torch.tensor([amax * (1.0 / 448.0)], dtype=torch.float32)).item() == byte
    assert float(2.0 ** (127 - byte)) == rcp


def test_sf_algebra_at_d256():
    assert sf_tile_bytes(256) == 1024 and sf_tile_bytes(128) == 512 and sf_tile_bytes(512) == 2048
    assert n_sf_tiles(128) == 1 and n_sf_tiles(129) == 2 and n_sf_tiles(1000) == 8
    assert sf_bytes(2, 4, 1000, 256) == 2 * 4 * 8 * 1024  # == the adapter's b*h*n_tiles*SF_SMEM_SIZE
    assert lanes_per_row(256) == 16 and lanes_per_row(512) == 32
    assert moved_bytes(1000, 32, 256) == 1000 * 32 * 256 * 3 + 1000 * 32 * 256 // 32
    assert moved_bytes(7, 2, 256, 2) == 7 * 2 * 256 * 3 + 7 * 2 * 256 // 32


def test_validate_shape_declines_what_the_lanes_cannot_cover():
    for axis in (AXIS_ROW, AXIS_COL):
        validate_shape(256, 256, axis)
        validate_shape(128, 256, axis)
        validate_shape(512, 256, axis)
        with pytest.raises(ValueError, match="multiple of 128"):
            validate_shape(200, 256, axis)
        with pytest.raises(ValueError, match="multiple of 32"):
            validate_shape(256, 40, axis)
        with pytest.raises(ValueError, match="burst"):
            validate_shape(2048, 256, axis)  # 8 KiB SF tile needs 512 burst lanes
    with pytest.raises(ValueError, match="axis"):
        validate_shape(256, 256, "rows")
    for bad_d in (0, -128):  # 0 used to reach a ZeroDivisionError, -128 used to PASS (every modulus held) and reach cute.compile
        with pytest.raises(ValueError, match="positive multiple of 128"):
            validate_shape(bad_d, 256, AXIS_ROW)
        with pytest.raises(ValueError, match="positive multiple of 128"):
            validate_shape(bad_d, 256, AXIS_COL)
    with pytest.raises(ValueError, match="up to 1024"):
        validate_shape(256, 2048, AXIS_ROW)  # past the CTA cap: validation passed, the launch failed untyped
    with pytest.raises(ValueError, match="up to 1024"):
        validate_shape(256, 2048, AXIS_COL)
    with pytest.raises(ValueError, match="divide a warp"):
        validate_shape(384, 256, AXIS_ROW)  # 24 lanes/row
    with pytest.raises(ValueError, match="rows per pass"):
        validate_shape(256, 96, AXIS_ROW)  # 6 rows/pass does not divide 128
    with pytest.raises(ValueError, match="units per tile"):
        validate_shape(128, 512, AXIS_COL)  # 8 units over 16 warps


def test_compile_declines_bad_inputs_before_tracing():
    with pytest.raises(ValueError, match="bf16/f16"):
        compile_quantize_mxfp8(dtype_in=torch.float32, h=2, d=256, axis=AXIS_ROW)
    with pytest.raises(ValueError, match="axis"):
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=2, d=256, axis="column")
    with pytest.raises(ValueError, match="positive"):
        compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=0, d=256, axis=AXIS_ROW)
    with pytest.raises(ValueError, match="not built"):
        run_quantize_mxfp8(QuantizeMxfp8Recipe(torch.bfloat16, 2, 256, AXIS_ROW), None, None, None, batch=1, seq_len=1, stream=0)


def test_f8_128x4_atom_rule_is_the_quoted_one():
    """``_swizzle_128x4``: atoms (128 rows x 4 cols) row-major; inside, (r, c) at ``(r%32)*16 + (r//32)*4 + c``."""
    g = torch.Generator().manual_seed(0)
    for R, C in ((128, 8), (384, 8), (256, 32), (128, 4)):
        m = torch.randint(0, 256, (R, C), dtype=torch.uint8, generator=g)
        sw = _swizzle_128x4(m).reshape(-1)
        for _ in range(500):
            r, c = int(torch.randint(0, R, (1,), generator=g)), int(torch.randint(0, C, (1,), generator=g))
            atom = (r // 128) * (C // 4) + c // 4
            assert int(sw[atom * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4]) == int(m[r, c])


@pytest.mark.parametrize("b, h, s", [(1, 1, 128), (2, 3, 384), (1, 2, 1000), (2, 2, 256)])
def test_rowwise_sf_byte_formula_matches_the_oracle_swizzle(b, h, s):
    """Section 2.3 Q/K row: tile (b, h, s_tile) = 1024 contiguous bytes; inside ``(c//4)*512 + (s%32)*16 + ((s%128)//32)*4 + c%4``."""
    g = torch.Generator().manual_seed(1)
    tiles = n_sf_tiles(s)
    s_pad = tiles * 128
    row_e = torch.randint(0, 256, (b * h * s_pad, D // 32), dtype=torch.uint8, generator=g)  # the oracle's logical [l*s_padded, D/32]
    sw = swizzle_sf_rowwise(row_e).reshape(-1)
    assert sw.numel() == sf_bytes(b, h, s, D)
    for _ in range(2000):
        bb, hh = int(torch.randint(0, b, (1,), generator=g)), int(torch.randint(0, h, (1,), generator=g))
        ss, dd = int(torch.randint(0, s_pad, (1,), generator=g)), int(torch.randint(0, D, (1,), generator=g))
        off = sf_byte_rowwise(bb, hh, ss, dd, n_heads=h, n_tiles=tiles, d=D)
        assert int(sw[off]) == int(row_e[(bb * h + hh) * s_pad + ss, dd // 32]), (bb, hh, ss, dd)
    # the tile is contiguous and lands where the SDPA's _build_sf_desc expects it
    assert sf_byte_rowwise(b - 1, h - 1, s_pad - 1, D - 1, n_heads=h, n_tiles=tiles, d=D) == sw.numel() - 1


@pytest.mark.parametrize("b, h, s", [(1, 1, 128), (2, 3, 384), (1, 2, 1000), (2, 2, 256)])
def test_columnwise_sf_byte_formula_matches_the_oracle_swizzle(b, h, s):
    """Section 2.3 V row: D-plane-major, plane stride ``B*KH*n_tiles*512`` -- NOT the Q/K per-tile order."""
    g = torch.Generator().manual_seed(2)
    tiles = n_sf_tiles(s)
    s_pad = tiles * 128
    col_e = torch.randint(0, 256, (b * h * s_pad // 32, D), dtype=torch.uint8, generator=g)  # the oracle's TE storage [l*s_padded/32, D]
    sw = swizzle_sf_columnwise(col_e).reshape(-1)
    assert sw.numel() == sf_bytes(b, h, s, D)
    for _ in range(2000):
        bb, hh = int(torch.randint(0, b, (1,), generator=g)), int(torch.randint(0, h, (1,), generator=g))
        ss, dd = int(torch.randint(0, s_pad, (1,), generator=g)), int(torch.randint(0, D, (1,), generator=g))
        off = sf_byte_columnwise(bb, hh, ss, dd, n_heads=h, n_tiles=tiles, batch=b)
        assert int(sw[off]) == int(col_e[(bb * h + hh) * (s_pad // 32) + ss // 32, dd]), (bb, hh, ss, dd)
    # the two D-planes are a whole plane apart, and the second plane begins exactly where the first ends
    assert sf_byte_columnwise(0, 0, 0, 128, n_heads=h, n_tiles=tiles, batch=b) == b * h * tiles * 512
    if b * h * tiles > 1:
        assert sf_byte_columnwise(0, 0, 0, 128, n_heads=h, n_tiles=tiles, batch=b) != sf_byte_rowwise(0, 0, 0, 128, n_heads=h, n_tiles=tiles, d=D)


def test_oracle_pads_s_to_128_with_zero_scales():
    """The pad rows the kernel must write as 0x00 are 0x00 in the oracle too (both arms), located by the section-2.3 formulas."""
    torch.manual_seed(3)
    b, h, s = 2, 2, 900  # rows 900..1023 pad; columnwise the 32-token blocks at 928 / 960 / 992 are entirely padding
    x = torch.randn(b * s, h, D).to(torch.bfloat16)
    _, row_sf, _, col_sf = _oracle(x, b, s, h)
    for axis, sf in ((AXIS_ROW, row_sf), (AXIS_COL, col_sf)):
        offs = _tail_sf_bytes(axis, batch=b, h=h, seq_len=s)
        assert offs, axis
        assert int(sf[offs].max().item()) == 0, axis
        assert int(sf.ne(0).sum().item()) > 0.9 * (sf.numel() - len(offs)), axis  # everything else is a live scale


# ---------------------------------------------------------------------------
# Tier 2: sm_107a trace-compile + spill count -- any box with the internal toolkit
# ---------------------------------------------------------------------------

_SASS_PROBE = textwrap.dedent("""
    import glob, os, subprocess, sys
    axis, dump, cands = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.environ["CUTE_DSL_DUMP_DIR"] = dump  # read once, at the first cutlass import
    import torch
    from cudnn.gated_attention_block.kernels.quantize_mxfp8 import COMPILE_OPTIONS, compile_quantize_mxfp8
    # --keep-cubin, NOT --keep-sass: the latter runs the DSL's own wheel nvdisasm, which ICEs on sm_107a.
    compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=4, d=256, axis=axis, compile_options=COMPILE_OPTIONS + " --gpu-arch sm_107a --keep-cubin")
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
    print("E8M0CVT", sum(1 for ln in sass if "F2FP" in ln and ".E8." in ln and ".RP" in ln))
    print("E4M3CVT", sum(1 for ln in sass if "F2FP" in ln and "E4M3" in ln))
    print("FMNMX3", sum(1 for ln in sass if "FMNMX3" in ln))
    print("LINES", len(sass))
    """)


@pytest.mark.parametrize("axis, e8m0_cvts", [(AXIS_ROW, 8), (AXIS_COL, 2)])
def test_sm107_trace_compile_has_no_spills(axis, e8m0_cvts, tmp_path):
    """Compile for Rubin here (no device match needed), decode with an nvdisasm that knows sm_107a: STL/LDL must be 0.

    Also pins that the scale is produced by the hardware ``cvt.rp...ue8m0x2`` (at least one per row pass rowwise = 8
    at 256 threads / D=256; one ``e8m0_pair`` per warp unit columnwise = 2 -- a LOWER bound, ptxas may duplicate) and
    that the abs-max tree fuses (FMNMX3 > 0).  ``SPILL == 0`` is the load-bearing assertion.

    SKIPS (never fails) when the DSL predates ``sm_107a`` or no candidate nvdisasm decodes it -- an L0 verdict
    must not turn on which toolkit ``CUDA_PATH`` happens to name."""
    if not _sm107a_known_to_the_dsl():
        pytest.skip("this cutlass-dsl has no sm_107a (needs >= 4.8.0.dev0, --pre)")
    cands = _nvdisasm_candidates()
    if not cands:
        pytest.skip("no nvdisasm executable to try (CUDA_PATH unset and none on PATH)")
    dump = tmp_path / f"quantize_mxfp8_{axis}"
    dump.mkdir()
    proc = subprocess.run([sys.executable, "-c", _SASS_PROBE, axis, str(dump), *cands], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"trace-compile failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    if any(ln.startswith("SKIP") for ln in proc.stdout.splitlines()):
        pytest.skip(f"{[ln for ln in proc.stdout.splitlines() if ln.startswith(('SKIP', 'REJECT'))]}")
    stats = {
        k: int(v)
        for k, v in (ln.split() for ln in proc.stdout.splitlines() if ln.split() and ln.split()[0] in ("SPILL", "E8M0CVT", "E4M3CVT", "FMNMX3", "LINES"))
    }
    print(f"\n[{axis}] sm_107a SASS: {stats} via {[ln for ln in proc.stdout.splitlines() if ln.startswith('NVDISASM')]}")
    assert stats["SPILL"] == 0, f"{axis}: {stats['SPILL']} STL/LDL in the sm_107a cubin"
    assert stats["E8M0CVT"] >= e8m0_cvts, f"{axis}: fewer ue8m0x2 cvts than the source issues -- {stats}"
    assert stats["E4M3CVT"] > 0 and stats["FMNMX3"] > 0, stats
    assert glob.glob(str(dump / "*.sm_107a.cubin")), "no .sm_107a.cubin landed in CUTE_DSL_DUMP_DIR"


# ---------------------------------------------------------------------------
# Tier 3: numerics on cc 10.x
# ---------------------------------------------------------------------------


@requires_mx_cvt
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("s", [128, 256, 384, 512, 1000, 900])
@pytest.mark.parametrize("role", ["q", "k", "v"])
def test_compact_source_is_bit_exact_vs_oracle(role, s, b):
    """Compact ``[T, H, D]`` bf16 in; e4m3 codes AND the F8_128x4 SF blob equal the oracle byte for byte.

    H_q=4 (q) vs H_kv=2 (k, v) so the head stride of the SF tile index is exercised; S=1000 is the KV tail
    (104 live rows in the last tile; columnwise the 992..1023 block STRADDLES S), S=900 adds three 32-token
    blocks that are entirely padding (columnwise SF must be 0x00 there)."""
    axis, h, _ = ROLES[role]
    torch.manual_seed(zlib.crc32(f"{role}-{s}-{b}".encode()) & 0xFFFF)  # str hash is per-process randomized; crc32 replays
    x = (torch.randn(b * s, h, D, device="cuda") * 3.0).to(torch.bfloat16)
    dst, sf = _launch(axis, x, batch=b, seq_len=s, h=h)
    _check_against_oracle(axis, x, dst, sf, batch=b, seq_len=s, h=h)
    offs = _tail_sf_bytes(axis, batch=b, h=h, seq_len=s)
    if offs:
        assert int(sf[offs].max().item()) == 0, "pad-row SF bytes must be 0x00"


@requires_mx_cvt
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("axis", [AXIS_ROW, AXIS_COL])
def test_f16_and_bf16_sources(axis, dtype):
    torch.manual_seed(7)
    b, s, h = 2, 384, 2
    x = (torch.randn(b * s, h, D, device="cuda") * 40.0).to(dtype)
    dst, sf = _launch(axis, x, batch=b, seq_len=s, h=h)
    _check_against_oracle(axis, x, dst, sf, batch=b, seq_len=s, h=h)


@requires_mx_cvt
@pytest.mark.parametrize("role", ["q", "k", "v"])
def test_strided_slab_slice_source(role):
    """The source is a column slice of the fused ``[T, 17408]`` projection slab (token stride 17408), as the block hands it."""
    axis, h, col_off = ROLES[role]
    torch.manual_seed(11)
    b, s = 2, 1000
    slab = (torch.randn(b * s, N_QKVG, device="cuda") * 2.0).to(torch.bfloat16)
    src = torch.as_strided(slab, (b * s, h, D), (N_QKVG, D, 1), storage_offset=col_off)
    assert src.stride(0) == N_QKVG and not src.is_contiguous()
    dst, sf = _launch(axis, src, batch=b, seq_len=s, h=h)
    _check_against_oracle(axis, src.contiguous(), dst, sf, batch=b, seq_len=s, h=h)
    assert dst.is_contiguous()


@requires_mx_cvt
@pytest.mark.parametrize("axis", [AXIS_ROW, AXIS_COL])
def test_e8m0_corners_on_device(axis):
    """Hand-built blocks: all-zero (SF 0x00, code 0), amax exactly 448 (0x7F: 448 * fp32(1/448) == 1.0, no round-up),
    amax 450 (rounds UP to 0x80), exact powers of two 2.0 / 1.0 / 0.5 (0x78 / 0x77 / 0x76), 446 (0x7F from below),
    a negative-dominated block, and -448 (the sign must not touch the amax: 0x7F, code 0xFE).  Bit-exact vs the
    oracle AND vs the hand-written SF bytes; the e4m3 codes vs torch's saturating cast of ``value * 2^(127 - e)``."""
    b, s, h = 1, 384, 1  # 9 cases: rowwise rows 0..8, columnwise token blocks 0..8 (tokens 0..287 -- needs S > 256)
    cases = [
        (0, 0.0, 0x00),
        (1, 448.0, 0x7F),
        (2, 450.0, 0x80),
        (3, 2.0, 0x78),
        (4, 1.0, 0x77),
        (5, 0.5, 0x76),
        (6, 446.0, 0x7F),
        (7, -3.0, 0x78),
        (8, -448.0, 0x7F),
    ]
    x = torch.zeros(s, h, D, device="cuda", dtype=torch.float32)
    # rowwise: row r, block 0 (d 0..31); columnwise: d-column 0, token block r (s = 32r .. 32r+31).
    for r, amax, _ in cases:
        first, second = (r, r) if axis == AXIS_ROW else (32 * r, 32 * r + 1)
        if axis == AXIS_ROW:
            x[first, 0, 0] = amax
            x[first, 0, 1] = -amax / 4
        else:
            x[first, 0, 0] = amax
            x[second, 0, 0] = -amax / 4
    xb = x.to(torch.bfloat16)
    row_of = (lambda r: r) if axis == AXIS_ROW else (lambda r: 32 * r)
    assert float(xb[row_of(1), 0, 0]) == 448.0 and float(xb[row_of(2), 0, 0]) == 450.0 and float(xb[row_of(8), 0, 0]) == -448.0  # bf16-exact
    dst, sf = _launch(axis, xb, batch=b, seq_len=s, h=h)
    _check_against_oracle(axis, xb, dst, sf, batch=b, seq_len=s, h=h)
    tiles = n_sf_tiles(s)
    codes = dst.view(torch.uint8)
    for r, amax, e in cases:
        exp_code = int(torch.tensor([abs(amax) * 2.0 ** (127 - e)]).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).item()) | (
            0x80 if amax < 0 else 0
        )
        if axis == AXIS_ROW:
            off = sf_byte_rowwise(0, 0, r, 0, n_heads=h, n_tiles=tiles, d=D)
            code = int(codes[r, 0, 0].item())
        else:
            off = sf_byte_columnwise(0, 0, 32 * r, 0, n_heads=h, n_tiles=tiles, batch=b)
            code = int(codes[32 * r, 0, 0].item())
        assert int(sf[off].item()) == e, f"{axis} amax={amax}: SF 0x{int(sf[off].item()):02X} != 0x{e:02X}"
        assert code == exp_code, f"{axis} amax={amax}: code 0x{code:02X} != 0x{exp_code:02X}"


@requires_mx_cvt
def test_two_launches_are_bitwise_identical():
    torch.manual_seed(5)
    b, s, h = 2, 1000, 2
    x = torch.randn(b * s, h, D, device="cuda").to(torch.bfloat16)
    for axis in (AXIS_ROW, AXIS_COL):
        d1, s1 = _launch(axis, x, batch=b, seq_len=s, h=h)
        d2, s2 = _launch(axis, x, batch=b, seq_len=s, h=h)
        assert torch.equal(d1.view(torch.uint8), d2.view(torch.uint8)) and torch.equal(s1, s2), axis


@requires_mx_cvt
def test_run_declines_a_mismatched_binding():
    h, b, s = 2, 1, 256
    r = compile_quantize_mxfp8(dtype_in=torch.bfloat16, h=h, d=D, axis=AXIS_ROW)
    x = torch.randn(b * s, h, D, device="cuda").to(torch.bfloat16)
    dst = torch.empty(b * s, h, D, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.zeros(sf_bytes(b, h, s, D), dtype=torch.uint8, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        run_quantize_mxfp8(r, x, torch.empty_like(x), sf, batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="H=2"):
        run_quantize_mxfp8(r, torch.randn(b * s, 4, D, device="cuda").to(torch.bfloat16), dst, sf, batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="batch\\*seq_len"):
        run_quantize_mxfp8(r, x, dst, sf, batch=2, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="uint8"):
        run_quantize_mxfp8(r, x, dst, sf.view(torch.float8_e8m0fnu), batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="ceil\\(S/128\\)"):
        run_quantize_mxfp8(r, x, dst, sf[:-1], batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="compiled for"):
        run_quantize_mxfp8(r, x.to(torch.float16), dst, sf, batch=b, seq_len=s, stream=st)
    with pytest.raises(ValueError, match="CUDA tensor"):
        run_quantize_mxfp8(r, x.cpu(), dst, sf, batch=b, seq_len=s, stream=st)
    slab = torch.zeros(b * s, N_QKVG, device="cuda", dtype=torch.bfloat16)
    off_by_8_bytes = torch.as_strided(slab, (b * s, h, D), (N_QKVG, D, 1), storage_offset=4)  # strides fine, base 8 B off a 16-B line
    assert off_by_8_bytes.data_ptr() % 16 == 8
    with pytest.raises(ValueError, match="16-byte aligned"):
        run_quantize_mxfp8(r, off_by_8_bytes, dst, sf, batch=b, seq_len=s, stream=st)
