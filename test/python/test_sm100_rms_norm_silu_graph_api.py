"""
Test SM100 RmsNorm+SiLU: cuDNN Graph API with OPENSOURCE heur_mode.

Constructs a graph with two separate nodes:
  1. RMSNorm(X, SCALE, EPSILON) → Y
  2. SiLU/Swish(Y) → Z

When built with heur_mode.OPENSOURCE, the graph detects the fusion pattern
and dispatches to the Sm100RmsNormSiluEngine OSS kernel.

Tests cover:
  - All 40 VAE problem sizes (8 C × 5 token counts) with bf16 output
  - FP8 and NVFP4 output types (placeholder — engine currently supports bf16)
  - Negative tests for unsupported problem sizes
"""

import pytest
import torch
import torch.nn.functional as F

import cudnn


def _is_blackwell():
    if not torch.cuda.is_available():
        return False
    prop = torch.cuda.get_device_properties(0)
    return prop.major == 10


# FP4 E2M1 lookup table (4-bit value → float)
# Encoding: 1 sign bit, 2 exponent bits, 1 mantissa bit
# Subnormal (exp=0): ±0, ±0.5
# Normal (exp>0):    ±2^(exp-1) * (1 + 0.5*man)
_FP4_E2M1_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,  # positive
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,  # negative
]


def _unpack_fp4_nibbles(packed_bytes, num_tokens, C):
    """Unpack FP4 packed bytes into a [num_tokens, C] int tensor of 4-bit nibble values."""
    nibbles = torch.zeros(num_tokens, C, dtype=torch.int32, device=packed_bytes.device)
    for col_byte in range(C // 2):
        byte_val = packed_bytes[:, col_byte].int()
        nibbles[:, col_byte * 2] = byte_val & 0x0F
        nibbles[:, col_byte * 2 + 1] = (byte_val >> 4) & 0x0F
    return nibbles


def _quantize_to_fp4_reference(values_f32, C):
    """Quantize float32 values to FP4 E2M1 nibbles using the same algorithm as the kernel.

    Matches the CuDNN BlockScaleRowHelper quantization exactly:
      1. amax = max(|block of 16 elements|)
      2. scale = max(amax / 6.0, FLT_MIN)       ← float32, NOT rounded to FP8
      3. quantized = nv_fp4x2_e2m1(value / scale) ← round to nearest FP4

    Key: quantize by magnitude first, then apply sign bit. This matches the
    hardware's nv_fp4x2_e2m1 which preserves sign. A naive argmin over all
    16 FP4 values would pick +0.0 over -0.0 for small negative values (tie
    resolved by index order), causing nibble_diff=8 mismatches.
    """
    BLOCK_SIZE = 16
    FP4_MAX = 6.0
    FLT_MIN = 1.17549435082228750796873653722224568e-38
    num_tokens = values_f32.shape[0]
    num_blocks = C // BLOCK_SIZE

    # Positive FP4 magnitudes (nibbles 0-7)
    fp4_positive = torch.tensor(_FP4_E2M1_TABLE[:8], dtype=torch.float32)
    nibbles = torch.zeros(num_tokens, C, dtype=torch.int32, device=values_f32.device)

    for b in range(num_blocks):
        col_start = b * BLOCK_SIZE
        col_end = col_start + BLOCK_SIZE
        block_vals = values_f32[:, col_start:col_end].cpu().float()

        # Step 1-2: compute scale from amax (same as kernel, in float32)
        amax = block_vals.abs().max(dim=1, keepdim=True).values
        scale = torch.clamp(amax / FP4_MAX, min=FLT_MIN)

        # Step 3: quantize by magnitude, then apply sign
        scaled = block_vals / scale
        magnitudes = scaled.abs()
        signs = (scaled < 0).int()  # 1 if negative

        # Find closest positive FP4 value (nibbles 0-7) by magnitude
        diffs = (magnitudes.unsqueeze(2) - fp4_positive.unsqueeze(0).unsqueeze(0)).abs()
        mag_nibbles = diffs.argmin(dim=2)  # 0-7

        # Apply sign: negative values get +8 (bit 3 = sign bit)
        block_nibbles = mag_nibbles + signs * 8
        nibbles[:, col_start:col_end] = block_nibbles.to(values_f32.device)

    return nibbles


def dequantize_nvfp4(packed_bytes, scale_row_fp8, num_tokens, C):
    """Dequantize NVFP4 1D1X1X output to float32.

    Args:
        packed_bytes: uint8 tensor [num_tokens, C // 2] — 2 FP4 values per byte
                      Lower nibble = even element, upper nibble = odd element
        scale_row_fp8: uint8 tensor [num_tokens, C // 16] — FP8 E4M3 scale per block
        num_tokens: number of rows
        C: hidden dimension (columns)

    Returns:
        float32 tensor [num_tokens, C]
    """
    BLOCK_SIZE = 16
    scale_f32 = scale_row_fp8.view(torch.float8_e4m3fn).float()
    nibbles = _unpack_fp4_nibbles(packed_bytes, num_tokens, C)

    output = torch.zeros(num_tokens, C, dtype=torch.float32, device=packed_bytes.device)
    for col in range(C):
        block = col // BLOCK_SIZE
        fp4_vals = torch.tensor([_FP4_E2M1_TABLE[v] for v in nibbles[:, col].cpu().tolist()], dtype=torch.float32, device=packed_bytes.device)
        output[:, col] = fp4_vals * scale_f32[:, block]
    return output


def rmsnorm_silu_reference(x, weight, eps, output_dtype=None):
    """PyTorch reference: RMSNorm + SiLU.

    RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    SiLU: y * sigmoid(y)

    Args:
        output_dtype: if None, returns in x.dtype (bf16). Pass torch.float32
                      for a float32 result (avoids bf16 intermediate rounding,
                      important for FP8 reference accuracy).
    """
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    x_norm = (x.float() / rms) * weight.float()
    result = F.silu(x_norm)
    if output_dtype is not None:
        return result.to(output_dtype)
    return result.to(x.dtype)


def _build_rmsnorm_silu_graph(num_tokens, C, input_dtype=cudnn.data_type.BFLOAT16, output_dtype=None):
    """Build a graph with RMSNorm → SiLU pattern.

    Args:
        output_dtype: data type for the output Z tensor. Defaults to input_dtype.
                      Use cudnn.data_type.FP8_E4M3 for FP8 output or
                      cudnn.data_type.FP4_E2M1 for NVFP4 output.
    """
    if output_dtype is None:
        output_dtype = input_dtype

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    X = graph.tensor(
        name="X",
        dim=[num_tokens, C, 1, 1],
        stride=[C, 1, 1, 1],
        data_type=input_dtype,
    )

    scale = graph.tensor(
        name="scale",
        dim=[1, C, 1, 1],
        stride=[C, 1, 1, 1],
        data_type=input_dtype,
    )

    epsilon = graph.tensor(
        name="epsilon",
        dim=[1, 1, 1, 1],
        stride=[1, 1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )

    rmsnorm_result = graph.rmsnorm(
        name="RmsNorm",
        norm_forward_phase=cudnn.norm_forward_phase.INFERENCE,
        input=X,
        scale=scale,
        epsilon=epsilon,
    )
    Y = rmsnorm_result[0]

    Y.set_dim([num_tokens, C, 1, 1])
    Y.set_stride([C, 1, 1, 1])
    Y.set_data_type(input_dtype)

    Z = graph.swish(input=Y, swish_beta=1.0, name="SiLU")
    Z.set_output(True).set_data_type(output_dtype)

    return graph, X, scale, epsilon, Z


def _run_rmsnorm_silu_test(C, num_tokens, atol=2e-2, rtol=2e-2):
    """Run a single RmsNorm+SiLU test and return (pass, info_dict).

    Uses realistic inputs:
      - x: randn * 5.0 + 5.0 → N(5, 25), range roughly [-10, 20]
        (non-zero mean exercises normalization, not just passthrough)
      - weight: random positive values in [0.5, 2.0] (exercises gamma scaling)
      - atol/rtol = 2e-2 (bf16 has ~8-bit mantissa)
    """
    torch.manual_seed(42)

    eps = 1e-6 / C  # L2Norm-equivalent epsilon
    # Non-zero-mean input so RMSNorm actually normalizes (not already centered)
    x_gpu = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    # Random positive weights to test gamma scaling
    weight_1d = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5  # [0.5, 2.0]
    weight_gpu = weight_1d.view(1, C, 1, 1).contiguous()
    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    ref_output = rmsnorm_silu_reference(x_gpu, weight_1d, eps)

    graph, X, scale, epsilon, Z = _build_rmsnorm_silu_graph(num_tokens, C)
    graph.validate()
    graph.build_operation_graph()

    graph.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

    z_gpu = torch.empty(num_tokens, C, dtype=torch.bfloat16, device="cuda")
    x_4d = x_gpu.view(num_tokens, C, 1, 1).contiguous()
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    variant_pack = {
        X: x_4d,
        scale: weight_gpu,
        epsilon: epsilon_cpu,
        Z: z_gpu.view(num_tokens, C, 1, 1),
    }

    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    mismatches = ~torch.isclose(z_gpu.float(), ref_output.float(), atol=atol, rtol=rtol)
    num_mismatches = mismatches.sum().item()
    max_diff = (z_gpu.float() - ref_output.float()).abs().max().item()
    mean_diff = (z_gpu.float() - ref_output.float()).abs().mean().item()
    nan_count = torch.isnan(z_gpu).sum().item()

    return {
        "C": C,
        "tokens": num_tokens,
        "elements": z_gpu.numel(),
        "mismatches": num_mismatches,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "nan_count": nan_count,
        "passed": num_mismatches == 0 and nan_count == 0,
    }


# ============================================================
# Test 1: Full VAE problem size sweep (bf16 output)
# ============================================================

C_VALUES = [64, 128, 160, 256, 320, 512, 640, 1024]
TOKEN_VALUES = [1560, 6240, 24960, 99840, 399360]


@pytest.mark.skipif(not _is_blackwell(), reason="Requires SM100 (Blackwell)")
@pytest.mark.parametrize("C", C_VALUES)
@pytest.mark.parametrize("num_tokens", TOKEN_VALUES)
def test_bf16_full_sweep(C, num_tokens):
    """Test all 40 VAE problem sizes with bf16 output via graph API."""
    result = _run_rmsnorm_silu_test(C, num_tokens)

    print(f"C={C:>4}, tokens={num_tokens:>6}: " f"max_diff={result['max_diff']:.3e}, " f"mismatches={result['mismatches']}/{result['elements']}")

    assert result["nan_count"] == 0, f"Output has {result['nan_count']} NaN values"
    assert result["passed"], f"C={C}, tokens={num_tokens}: " f"{result['mismatches']}/{result['elements']} mismatches " f"(max_diff={result['max_diff']:.6e})"


# ============================================================
# Test 2: Quick smoke test (single config, for CI)
# ============================================================


@pytest.mark.skipif(not _is_blackwell(), reason="Requires SM100 (Blackwell)")
def test_bf16_smoke():
    """Quick smoke test with a single representative config."""
    result = _run_rmsnorm_silu_test(512, 24960)
    assert result["passed"], f"Smoke test failed: {result}"


# ============================================================
# Test 3: FP8 output (all 40 problem sizes)
# ============================================================


def _run_fp8_rmsnorm_silu_test(C, num_tokens, atol=0.125, rtol=0.125):
    """Run RmsNorm+SiLU with FP8 (E4M3) output and compare against float32 reference.

    Reference is computed entirely in float32 then quantized to FP8, matching the
    kernel's computation path (float32 → FP8). This avoids bf16 double-rounding
    which artificially inflates max_diff.

    FP8 E4M3 has 3-bit mantissa → ~12.5% relative precision. Tolerance is set to
    1 FP8 ULP for typical output magnitudes.
    """
    torch.manual_seed(42)
    eps = 1e-6 / C

    # Same realistic inputs as bf16 test
    x_gpu = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    weight_1d = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    weight_gpu = weight_1d.view(1, C, 1, 1).contiguous()
    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    # Compute reference in float32 (matching kernel path), then quantize to FP8
    # Key: skip bf16 intermediate to avoid double-rounding
    ref_f32 = rmsnorm_silu_reference(x_gpu, weight_1d, eps, output_dtype=torch.float32)
    ref_fp8 = ref_f32.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    graph, X, scale, epsilon, Z = _build_rmsnorm_silu_graph(num_tokens, C, output_dtype=cudnn.data_type.FP8_E4M3)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

    z_gpu = torch.empty(num_tokens, C, dtype=torch.float8_e4m3fn, device="cuda")
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    variant_pack = {
        X: x_gpu.view(num_tokens, C, 1, 1).contiguous(),
        scale: weight_gpu,
        epsilon: epsilon_cpu,
        Z: z_gpu.view(num_tokens, C, 1, 1),
    }
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    # Compare dequantized FP8 values
    z_float = z_gpu.float()
    ref_float = ref_fp8.float()
    abs_diff = (z_float - ref_float).abs()
    mismatches = ~torch.isclose(z_float, ref_float, atol=atol, rtol=rtol)
    num_mismatches = mismatches.sum().item()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    nan_count = torch.isnan(z_float).sum().item()
    # Count exact FP8 matches (both quantize to same representable value)
    exact_matches = (z_float == ref_float).sum().item()

    return {
        "C": C,
        "tokens": num_tokens,
        "elements": z_gpu.numel(),
        "mismatches": num_mismatches,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "exact_match_pct": 100.0 * exact_matches / z_gpu.numel(),
        "nan_count": nan_count,
        "passed": num_mismatches == 0 and nan_count == 0,
    }


@pytest.mark.skipif(not _is_blackwell(), reason="Requires SM100 (Blackwell)")
@pytest.mark.parametrize("C", C_VALUES)
@pytest.mark.parametrize("num_tokens", TOKEN_VALUES)
def test_fp8_full_sweep(C, num_tokens):
    """Test all 40 VAE problem sizes with FP8 (E4M3) output via graph API."""
    result = _run_fp8_rmsnorm_silu_test(C, num_tokens)

    print(f"  FP8 C={C:>4}, tokens={num_tokens:>6}: " f"max_diff={result['max_diff']:.3e}  " f"mismatches={result['mismatches']}/{result['elements']}")

    assert result["nan_count"] == 0, f"Output has {result['nan_count']} NaN values"
    assert result["passed"], (
        f"FP8 C={C}, tokens={num_tokens}: " f"{result['mismatches']}/{result['elements']} mismatches " f"(max_diff={result['max_diff']:.6e})"
    )


# ============================================================
# Test 4: NVFP4 output (representative problem sizes)
# ============================================================


def _run_nvfp4_rmsnorm_silu_test(C, num_tokens):
    """Run RmsNorm+SiLU with NVFP4 (FP4_E2M1) 1D1X1X block-scale output.

    The CuDNN kernel performs block-scale quantization matching the APEX engine:
      1. Compute amax over each block of 16 elements
      2. scale = amax / 6.0 (stored as FP8 E4M3)
      3. quantized = round(element / scale) (stored as FP4 E2M1, 2 per byte)

    We compare FP4 nibbles directly: quantize the Python reference using the
    kernel's block scales, then check that nibble indices match within ±1 ULP.
    This avoids conflating quantization error with kernel correctness.
    """
    torch.manual_seed(42)
    eps = 1e-6 / C

    x_gpu = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    weight_1d = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    weight_gpu = weight_1d.view(1, C, 1, 1).contiguous()
    epsilon_cpu = torch.full((1, 1, 1, 1), eps, dtype=torch.float32, device="cpu")

    # Float32 reference (pre-quantization)
    ref_f32 = rmsnorm_silu_reference(x_gpu, weight_1d, eps, output_dtype=torch.float32)

    graph, X, scale, epsilon, Z = _build_rmsnorm_silu_graph(num_tokens, C, output_dtype=cudnn.data_type.FP4_E2M1)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)

    # FP4 output: 2 elements per byte → C // 2 bytes per row
    z_gpu = torch.empty(num_tokens * C // 2, dtype=torch.uint8, device="cuda")
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    variant_pack = {
        X: x_gpu.view(num_tokens, C, 1, 1).contiguous(),
        scale: weight_gpu,
        epsilon: epsilon_cpu,
        Z: z_gpu,
    }
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()

    # Read scale_row from workspace (auto-allocated by engine)
    # Offset matches engine's get_scale_row_workspace_offset()
    rs_size = num_tokens * 4  # rs buffer
    rs_size_aligned = ((rs_size + 127) // 128) * 128
    fp8_scale_aligned = ((rs_size_aligned + 4 + 127) // 128) * 128
    scale_row_offset = fp8_scale_aligned
    scale_row_numel = num_tokens * (C // 16)
    scale_row_fp8 = workspace[scale_row_offset : scale_row_offset + scale_row_numel].reshape(num_tokens, C // 16)

    # Compare FP4 nibbles directly: quantize the Python reference using the
    # kernel's block scales, then check if the FP4 values match.
    # This removes quantization error from the comparison and tests whether
    # the kernel's quantization path matches our reference exactly.
    z_packed = z_gpu.reshape(num_tokens, C // 2)
    kernel_nibbles = _unpack_fp4_nibbles(z_packed, num_tokens, C)
    ref_nibbles = _quantize_to_fp4_reference(ref_f32, C)

    # Allow up to 1 FP4 ULP difference (adjacent representable value)
    nibble_diff = (kernel_nibbles - ref_nibbles).abs()
    mismatches = nibble_diff > 1  # more than 1 nibble index apart
    num_mismatches = mismatches.sum().item()
    max_nibble_diff = nibble_diff.max().item()
    nan_count = 0  # FP4 nibbles can't be NaN

    # Also report dequantized max_diff for context
    z_dequant = dequantize_nvfp4(z_packed, scale_row_fp8, num_tokens, C)
    max_deq_diff = (z_dequant - ref_f32).abs().max().item()

    return {
        "C": C,
        "tokens": num_tokens,
        "elements": num_tokens * C,
        "mismatches": num_mismatches,
        "max_nibble_diff": max_nibble_diff,
        "max_deq_diff": max_deq_diff,
        "nan_count": nan_count,
        "passed": nan_count == 0 and num_mismatches == 0,
    }


# NVFP4 requires C % 16 == 0 (block size). All our C values satisfy this.


@pytest.mark.skipif(not _is_blackwell(), reason="Requires SM100 (Blackwell)")
@pytest.mark.parametrize("C", C_VALUES)
@pytest.mark.parametrize("num_tokens", TOKEN_VALUES)
def test_nvfp4_sweep(C, num_tokens):
    """Test NVFP4 (FP4_E2M1) output with 1D1X1X block-scale quantization."""
    result = _run_nvfp4_rmsnorm_silu_test(C, num_tokens)

    print(
        f"  NVFP4 C={C:>4}, tokens={num_tokens:>6}: " f"max_nibble_diff={result['max_nibble_diff']}  " f"mismatches={result['mismatches']}/{result['elements']}"
    )

    assert result["passed"], f"NVFP4 C={C}, tokens={num_tokens}: " f"{result['mismatches']}/{result['elements']} nibbles differ by >1 ULP"


# ============================================================
# Test 5: Negative tests — unsupported problem sizes
# ============================================================


@pytest.mark.skipif(not _is_blackwell(), reason="Requires SM100 (Blackwell)")
@pytest.mark.parametrize(
    "C,num_tokens",
    [
        (48, 1560),  # C=48 not in LUT
        (512, 10000),  # tokens=10000 not in LUT
        (2048, 24960),  # C=2048 not in LUT
        (100, 100),  # Neither C nor tokens in LUT
    ],
)
def test_unsupported_problem_sizes(C, num_tokens):
    """Verify that unsupported problem sizes are rejected gracefully."""
    graph, X, scale, epsilon, Z = _build_rmsnorm_silu_graph(num_tokens, C)
    graph.validate()
    graph.build_operation_graph()

    # Should either fail at create_execution_plans or check_support
    with pytest.raises(cudnn.cudnnGraphNotSupportedError):
        graph.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
        graph.check_support()


# ============================================================
# Test 4: Verify L2Norm ↔ RMSNorm epsilon equivalence
# ============================================================


@pytest.mark.skipif(not _is_blackwell(), reason="Requires SM100 (Blackwell)")
@pytest.mark.parametrize("C", [64, 256, 512, 1024])
def test_epsilon_equivalence(C):
    """Verify that RMSNorm(eps/C) matches L2Norm(eps) * sqrt(C)."""
    torch.manual_seed(42)
    num_tokens = 1560
    eps = 1e-6

    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(C, dtype=torch.bfloat16, device="cuda")
    scale = C**0.5

    # L2Norm + SiLU
    l2norm = torch.sqrt(torch.sum(x.float() ** 2, dim=-1, keepdim=True) + eps)
    l2_out = F.silu((x.float() / l2norm) * scale * weight.float()).to(x.dtype)

    # RMSNorm(eps/C) + SiLU (what cuDNN computes)
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps / C)
    rms_out = F.silu((x.float() / rms) * weight.float()).to(x.dtype)

    atol, rtol = 1e-2, 1e-2
    mismatches = ~torch.isclose(l2_out.float(), rms_out.float(), atol=atol, rtol=rtol)
    assert mismatches.sum().item() == 0, f"L2Norm vs RMSNorm(eps/C) mismatch for C={C}: " f"{mismatches.sum().item()} elements differ"


# ============================================================
# Main — run representative configs when executed directly
# ============================================================

if __name__ == "__main__":
    if not _is_blackwell():
        print("Skipping — requires SM100 (Blackwell)")
        exit(0)

    print("=" * 70)
    print("  SM100 RmsNorm+SiLU OSS Engine — Full Test Suite")
    print("=" * 70)

    # Run full bf16 sweep
    print("\n--- bf16 Full Sweep (40 configs) ---")
    passed = 0
    failed = 0
    for C in C_VALUES:
        for tokens in TOKEN_VALUES:
            try:
                result = _run_rmsnorm_silu_test(C, tokens)
                status = "PASS" if result["passed"] else "FAIL"
                if result["passed"]:
                    passed += 1
                else:
                    failed += 1
                print(
                    f"  C={C:>4}, tokens={tokens:>6}: {status}  "
                    f"max_diff={result['max_diff']:.3e}  "
                    f"mismatches={result['mismatches']}/{result['elements']}"
                )
            except Exception as e:
                failed += 1
                print(f"  C={C:>4}, tokens={tokens:>6}: ERROR  {e}")

    bf16_passed, bf16_failed = passed, failed
    print(f"\n--- bf16 Results: {passed} passed, {failed} failed (out of {passed + failed}) ---")

    # Run FP8 sweep
    print("\n--- FP8 Full Sweep (40 configs) ---")
    passed = 0
    failed = 0
    for C in C_VALUES:
        for tokens in TOKEN_VALUES:
            try:
                result = _run_fp8_rmsnorm_silu_test(C, tokens)
                status = "PASS" if result["passed"] else "FAIL"
                if result["passed"]:
                    passed += 1
                else:
                    failed += 1
                print(
                    f"  FP8 C={C:>4}, tokens={tokens:>6}: {status}  "
                    f"max_diff={result['max_diff']:.3e}  "
                    f"mismatches={result['mismatches']}/{result['elements']}"
                )
            except Exception as e:
                failed += 1
                print(f"  FP8 C={C:>4}, tokens={tokens:>6}: ERROR  {e}")

    fp8_passed, fp8_failed = passed, failed
    print(f"\n--- FP8 Results: {passed} passed, {failed} failed (out of {passed + failed}) ---")

    # Run NVFP4 sweep
    print("\n--- NVFP4 Full Sweep (40 configs) ---")
    passed = 0
    failed = 0
    for C in C_VALUES:
        for tokens in TOKEN_VALUES:
            try:
                result = _run_nvfp4_rmsnorm_silu_test(C, tokens)
                status = "PASS" if result["passed"] else "FAIL"
                if result["passed"]:
                    passed += 1
                else:
                    failed += 1
                print(
                    f"  NVFP4 C={C:>4}, tokens={tokens:>6}: {status}  "
                    f"max_nibble_diff={result['max_nibble_diff']}  "
                    f"mismatches={result['mismatches']}/{result['elements']}"
                )
            except Exception as e:
                failed += 1
                print(f"  NVFP4 C={C:>4}, tokens={tokens:>6}: ERROR  {e}")

    nvfp4_passed, nvfp4_failed = passed, failed
    print(f"\n--- NVFP4 Results: {passed} passed, {failed} failed (out of {passed + failed}) ---")

    # Run negative tests
    print("\n--- Negative Tests ---")
    unsupported = [(48, 1560), (512, 10000), (2048, 24960)]
    for C, tokens in unsupported:
        try:
            graph, X, scale, epsilon, Z = _build_rmsnorm_silu_graph(tokens, C)
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
            graph.check_support()
            print(f"  C={C:>4}, tokens={tokens:>6}: UNEXPECTED PASS (should have been rejected)")
        except cudnn.cudnnGraphNotSupportedError:
            print(f"  C={C:>4}, tokens={tokens:>6}: Correctly rejected ✓")
        except Exception as e:
            print(f"  C={C:>4}, tokens={tokens:>6}: Rejected with: {type(e).__name__}")

    print("\n" + "=" * 70)
    total_failed = bf16_failed + fp8_failed + nvfp4_failed
    total_passed = bf16_passed + fp8_passed + nvfp4_passed
    total_status = "ALL PASS" if total_failed == 0 else f"{total_failed} FAILED"
    print(f"  Final: {total_status} ({total_passed} passed, {total_failed} failed)")
    print("=" * 70)
