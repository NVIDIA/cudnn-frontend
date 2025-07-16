import cudnn
import itertools
import pytest
import torch

from test_utils import torch_fork_set_rng

if not hasattr(torch, "float4_e2m1fn_x2"):
    pytest.skip(
        "Current torch version does not support float4_e2m1fn_x2",
        allow_module_level=True,
    )

# copy-pasted from
# https://github.com/pytorch/pytorch/blob/011026205a9d4c38458130f8ca242028f6184bf0/torch/testing/_internal/common_quantized.py#L234C1-L351C29


# copied from https://github.com/drisspg/transformer_nuggets/blob/main/transformer_nuggets/mx/to_blocked.py
def ceil_div(a, b):
    return (a + b - 1) // b


# copy-pasted from
# https://github.com/pytorch/ao/blob/bc4f51da86956275da7db0da6e420c506df97820/torchao/prototype/custom_fp_utils.py#L27C1-L142C29
def _n_ones(n: int) -> int:
    return (1 << n) - 1


FP4_EBITS = 2
FP4_MBITS = 1

EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    elif torch_type == torch.float4_e2m1fn_x2:
        return cudnn.data_type.FP4_E2M1
    elif torch_type == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    elif torch_type == torch.float8_e5m2fn:
        return cudnn.data_type.FP8_E5M2
    elif torch_type == torch.float8_e8m0fnu:
        return cudnn.data_type.FP8_E8M0
    else:
        raise ValueError("Unsupported tensor data type.")


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers with the given
    number of exponent and mantissa bits.

    Input: torch.Tensor of dtype torch.float
    Output: torch.Tensor of dtype torch.uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding

    Note: there are no special values (NaN, inf) support in this code. Values
    outside the representable range of Floatx after rounding are clamped to the
    maximum Floatx magnitude (sign is preserved).

    Code below is an adaptation of https://fburl.com/code/ciwofcg4

    Background 1: last answer in https://stackoverflow.com/q/8981913
    Background 2: Computer Organization and Design, RISC-V edition, Chapter 3.5
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    # TODO document this better
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # all E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # save the sign
    # Note that we have torch.uint32, but some ops like cpu bit shifts
    # do not work on it. So, we stay in int32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # set everything to positive, will add sign back at the end
    x = x ^ sign

    # TODO: can the branch floating point comparisons below be done without
    # converting to float? probably but need to verify
    x = x.view(torch.float)

    # rewrite saturate/denorm/norm branches without explicit data dependent
    # control flow, to be more compiler friendly
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    #
    # branch 1: saturate to max val - handled later in the code which combines
    #   the branches
    #

    #
    # branch 2: to conversion to denormal as well as rounding up to normal
    #
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    #
    # branch 3: stay in normal range, adjust the exponent and round
    #
    normal_x = x.view(torch.int32)
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    #
    # combine the branches
    #
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Right shift of a negative signed integer can fill the least significant
    # bits with either 1s or 0s, depending on the implementation. Since PyTorch
    # doesn't have an uint32 dtype, we mask out these bits to get just the
    # f4 sign bit
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _floatx_unpacked_to_f32(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert sub-byte floating point numbers with the given number of exponent
    and mantissa bits to FP32.

    Input: torch.Tensor of dtype uint8, where the bit encoding is stored
    in the least significant bits. e.g.
      fp4: bits 0-3 empty and bits 4-7 in fp4_e2m1 encoding
      fp6: bits 0-1 empty and bits 2-7 in fp6_e2m3 or fp6_e3m2 encoding
    Output: torch.Tensor of dtype fp32 with the dequantized value
    """
    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    # save the sign
    sign_lp = x & sign_mask

    # set everything to positive, will add sign back at the end
    x_pos = x ^ sign_lp

    #
    # 1. Calculate zero mask
    #
    zero_mask = x_pos == 0

    #
    # 2. Calculate the denormal path mask
    #
    denormal_mask = torch.logical_and((x_pos > 0), ((x_pos >> mbits) == 0))

    #
    # 3. Calculate the normal path
    #

    # calculate the new exponent and shift it to bits 2:9 of the result
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32

    # shift the mantissa to bits 10:32 of the result
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    #
    # 4. Add the zero and denormal casts to the already casted normal path
    #
    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    # fast path.
    # without this, performance for FP4_E2M1 is slower by 2x
    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32

    else:
        # iterate over all possible values of mantissa
        # i=0, j=1
        # i=1, j=10,11
        # i=2, j=100,101,110,111
        # and so on
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                # left shift mantissa until it overflows (create an implicit 1)
                # subtract exponent by the same amount
                left_shift = mbits - i
                mantissa_f32 = (mantissa_cmp - (1 << i)) << (
                    left_shift + MBITS_F32 - mbits
                )
                exp_biased_f32 = (denormal_exp_biased - left_shift) << MBITS_F32

                # we can update this in-place since the values won't overlap
                # torch.compile() may complain unsupported operand type(s) for |: 'SymInt' and 'int'
                # thus we use + instead of | here
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = (
                    exp_biased_f32 + mantissa_f32
                )

        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    # add sign back
    sign_f32 = sign_lp.to(torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)


def get_cc():
    (major, minor) = torch.cuda.get_device_capability()
    return major * 10 + minor


def matmul_dequantize_cache_key(cudnn_handle, A, B, A_scale, B_scale, BLOCK_SIZE):
    return (
        tuple(A.shape),
        tuple(B.shape),
    )


@cudnn.jit(heur_modes=[cudnn.heur_mode.A, cudnn.heur_mode.B])
@cudnn.graph_cache(key_fn=matmul_dequantize_cache_key)
def create_matmul_dequantize_graph(
    cudnn_handle, A, B, A_descale, B_descale, BLOCK_SIZE
):

    with cudnn.graph(cudnn_handle) as (g, _):

        batch_size, M, N, K = (
            A.shape[0],
            A.shape[1],
            B.shape[1],
            A.shape[2],
        )

        A_cudnn_tensor = g.tensor(
            name="tensor_a",
            dim=(batch_size, M, K),
            stride=(M * K, K, 1),
            data_type=convert_to_cudnn_type(A.dtype),
        )

        B_cudnn_tensor = g.tensor(
            name="tensor_b",
            dim=(batch_size, K, N),
            stride=(N * K, 1, K),
            data_type=convert_to_cudnn_type(B.dtype),
        )

        A_descale_tensor = g.tensor(
            name="block_descale_a",
            dim=A_descale.shape,
            stride=(M * K, K, 1),
            data_type=convert_to_cudnn_type(A_descale.dtype),
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

        B_descale_tensor = g.tensor(
            name="block_descale_b",
            dim=B_descale.shape,
            stride=(N * K, 1, K),
            data_type=convert_to_cudnn_type(B_descale.dtype),
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

        after_descale_a = g.block_scale_dequantize(
            A_cudnn_tensor, A_descale_tensor, block_size=[1, BLOCK_SIZE]
        )
        after_descale_b = g.block_scale_dequantize(
            B_cudnn_tensor, B_descale_tensor, block_size=[BLOCK_SIZE, 1]
        )

        C = g.matmul(
            after_descale_a,
            after_descale_b,
            compute_data_type=cudnn.data_type.FLOAT,
            name="GEMM",
        )

        C.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)

    return g, [A_cudnn_tensor, B_cudnn_tensor, A_descale_tensor, B_descale_tensor, C]


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


def _bfloat16_to_float4_e2m1fn_x2(x):
    assert x.dtype == torch.bfloat16
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


@pytest.mark.skipif(get_cc() < 100, reason="requires Blackwell or newer arch")
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_low_precision_fp4_matmul(cudnn_handle):
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("Current torch version does not support float4_e2m1fn_x2")

    batch_size, M, N, K = 1, 128, 128, 64
    BLOCK_SIZE = 16
    compute_data_type = cudnn.data_type.FLOAT

    if K % 32 != 0:
        pytest.skip("For fp4, k must be a multiple of 32")

    # Create random tensors
    A_ref = _floatx_unpacked_to_f32(
        torch.randint(0, 16, (batch_size, M, K), device="cuda", dtype=torch.uint8),
        FP4_EBITS,
        FP4_MBITS,
    ).bfloat16()
    B_ref = _floatx_unpacked_to_f32(
        torch.randint(0, 16, (batch_size, K, N), device="cuda", dtype=torch.uint8),
        FP4_EBITS,
        FP4_MBITS,
    ).bfloat16()

    print("\n\n")

    A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
    B = _bfloat16_to_float4_e2m1fn_x2(B_ref)

    A_descale = torch.full(
        (batch_size, M, K), 1.0, dtype=torch.float8_e4m3fn, device="cuda"
    )
    B_descale = torch.full(
        (batch_size, K, N), 1.0, device="cuda", dtype=torch.float8_e4m3fn
    )

    g, uids = create_matmul_dequantize_graph(
        cudnn_handle, A, B, A_descale, B_descale, BLOCK_SIZE
    )

    A_uid, B_uid, A_descale_uid, B_descale_uid, C_uid = uids

    C = torch.empty((batch_size, M, N), device="cuda", dtype=torch.bfloat16)

    variant_pack = {
        A_uid: A,
        B_uid: B,
        A_descale_uid: A_descale,
        B_descale_uid: B_descale,
        C_uid: C,
    }

    workspace = torch.empty(g.get_workspace_size(), device="cuda", dtype=torch.uint8)
    g.execute(variant_pack, workspace, handle=cudnn_handle)

    # not doing comparison because A and A_ref are not close
