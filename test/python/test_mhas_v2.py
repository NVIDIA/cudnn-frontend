"""
This script tests cuDNN front-end attention.
The recommended way to run tests:
> pytest -vv -s -rA test_mhas_v2.py
"""

import cudnn
import pytest
import random
import torch
import math
import os
import sys
from looseversion import LooseVersion
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass, asdict

from mha_v2_utils import (
    exec_cfg,
    INVALID_BOUND,
    RandomizationContext,
    RandomBatchSize,
    RandomBlockSize,
    RandomSequenceLength,
    RandomHiddenDimSize,
    RandomHeadGenerator,
    RandomChoice,
    SlidingWindowMaskGenerator,
    time_execution,
    profile_execution,
)

# fmt: off

if __name__ == "__main__":
    print("This is pytest script.")
    sys.exit(0)

def tlist(*, num_tests, rng_seed):
    assert num_tests >= 1 and type(num_tests) == int, "wrong input"
    rng = random.Random(rng_seed)
    return [(i+1, num_tests, rng.randint(65536, 2147483647)) for i in range(num_tests)]

def get_layout_name(string, indices):
    assert len(string) == 4 and sorted(indices) == [0, 1, 2, 3], "wrong input"
    chars = [string[i] for i in indices]
    return ''.join(chars)

def int_cli_option(org_val, request, cli_opt):
    val = request.config.getoption(cli_opt)
    return val if type(val) == int else org_val

def implementation_cli_option(org_val, request, cli_opt):
    str_val = request.config.getoption(cli_opt)
    val = getattr(cudnn.attention_implementation, str_val, None) if str_val else None
    return val if isinstance(val, cudnn.attention_implementation) else org_val

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
    else:
        assert False, "unsupported tensor data type"

def approx_equal(actual, expected, sepbuf, rawbuf, rtol, atol, tag, disp_elems):
    mismatches = torch.where(torch.isclose(actual.float(), expected, rtol=rtol, atol=atol) == False)
    mismatch_cnt = mismatches[0].numel()
    num_elements = torch.numel(actual)
    if mismatch_cnt != 0:
        percentage = 100 * mismatch_cnt / num_elements
        if disp_elems > 0:
            print(f"Comparing '{tag}' using rtol={rtol:.4e}, atol={atol:.4e}")
            combined = torch.stack(mismatches, dim=-1).tolist()
            count = 0
            for index in combined:
                diff = actual[tuple(index)] - expected[tuple(index)]
                if math.isfinite(diff):
                    print(f"idx{index}: {tag}_gpu={actual[tuple(index)]:+.6e}, {tag}_ref={expected[tuple(index)]:+.6e}, diff={diff:+.2e}")
                else:
                    print(f"idx{index}: {tag}_gpu={actual[tuple(index)]:+.6e}, {tag}_ref={expected[tuple(index)]:+.6e}")
                count += 1
                if count >= disp_elems:
                    break
            print(f"%%%% Total {mismatch_cnt:,} mismatches ({percentage:.1f}%) when validating '{tag}' results (first {count} mismatches displayed)")
        else:
            print(f"%%%% Total {mismatch_cnt:,} mismatches ({percentage:.1f}%) when validating '{tag}' results")

        num_nans       = torch.isnan(actual).sum().item()
        num_infs       = torch.isinf(actual).sum().item()
        num_zeros      = num_elements - torch.count_nonzero(actual)
        num_finites_nz = num_elements - num_nans - num_infs - num_zeros

        print(f"%%%% {tag}_gpu overview: elements={num_elements:,}, finites_nz={num_finites_nz:,}, zeros={num_zeros:,}, nans={num_nans:,}, infs={num_infs:,}")

        num_nans       = torch.isnan(expected).sum().item()
        num_infs       = torch.isinf(expected).sum().item()
        num_zeros      = num_elements - torch.count_nonzero(expected)
        num_finites_nz = num_elements - num_nans - num_infs - num_zeros

        print(f"%%%% {tag}_ref overview: elements={num_elements:,}, finites_nz={num_finites_nz:,}, zeros={num_zeros:,}, nans={num_nans:,}, infs={num_infs:,}")
    else:
        print(f"%%%% Numerical divergence of '{tag}' within limits")

    # Check if areas before and after the tensor were overwritten (treated as one numerical mismatch).
    if sepbuf is not None and not torch.all(torch.isnan(sepbuf)).item():
        print(f"%%%% Buffer '{tag}' overwritten outside its boundaries")
        print(sepbuf)
        mismatch_cnt += 1

    # Check if unused elements of the tensor were overwritten (treated as one numerical mismatch).
    # Note that this check destroys computed data (overwrites them with NaN-s).
    if rawbuf is not None:
        actual.fill_(float('nan'))
        if not torch.all(torch.isnan(rawbuf)).item():
            print(f"%%%% Unused gaps of '{tag}' tensor were overwritten")
            mismatch_cnt += 1

    return mismatch_cnt

def alloc_tensor(shape, data_type, *, elems=None, strides=None, rng=None, mean=0.0, std=1.0, margins=512):
    # Arguments elems/strides must be both specified or both None.
    if elems is None and strides is None:
        if hasattr(shape, '__iter__'):
            strides = []
            prod = 1
            for dim in reversed(shape):
                strides.insert(0, prod)
                prod *= int(dim)
            elems = prod
        else:
            elems = int(shape)
            strides = (1,)
            shape = (shape,)
    else:
        assert elems is not None and strides is not None, "wrong input"

    assert margins >= 0 and type(margins) == int, "wrong input"

    rawbuf = torch.empty(elems+2*margins, dtype=data_type, device="cuda")
    if torch.is_floating_point(rawbuf):
        rawbuf.fill_(float('nan'))
    else:
        rawbuf.fill_(-1)

    tensor = torch.as_strided(rawbuf, shape, strides, storage_offset=margins)
    sepbuf = (torch.as_strided(rawbuf, (2, margins), (elems+margins, 1), storage_offset=0) if margins > 0 else None)

    # Use this initialization for floating point types only.
    if rng is not None:
        tensor.normal_(mean=mean, std=std, generator=rng)

    # Not returning the raw buffer, if the data tensor has no gaps between valid elements.
    # If there are unused gaps, then we want to check that those gaps were not overwritten.
    if math.prod(shape) == elems:
        rawbuf = None

    return tensor, sepbuf, rawbuf

def fetch_blocked_tests(file_path, gpu_arch, cudnn_ver):
    assert type(gpu_arch) == type(cudnn_ver) == str, "expecting strings"
    blocked_tests = []
    try:
        line_number = None
        with open(file_path, 'r') as file:
            for line_number, line_buf in enumerate(file, 1):
                line_buf = line_buf.split('#', 1)[0]  # remove comments
                line_buf = "".join(line_buf.split())  # remove whitespaces
                if line_buf:
                    test,sms,libs = (line_buf+"::").split(':')[:3]
                    if not test:
                        raise ValueError("missing test name")
                    sms  = sms.split(',') if sms else None
                    libs = libs.split(',') if libs else None
                    if (test not in blocked_tests) and (sms == None or gpu_arch in sms) and (libs == None or cudnn_ver in libs):
                        blocked_tests.append(test)
    except Exception as e:
        blocked_tests = []
        if line_number != None:
            print(f"\n\nWARNING: {e} in {file_path}:{line_number}")
        else:
            print(f"\n\nWARNING: {e}")
    return blocked_tests

def show_blocked_tests(blocked_tests, gpu_arch, cudnn_ver):
    print(f"\n\nBlocked tests on {gpu_arch} and cudnn_ver={cudnn_ver}:")
    if blocked_tests:
        for index, test in enumerate(blocked_tests):
            assert type(test) == str, "test name must be string"
            print(f"{index+1:<4} : {test}")
    else:
        print("[empty]")

def is_test_blocked(test, blocked_tests):
    assert type(test) == str, "test name must be string"
    if not blocked_tests:
        return False
    return True if test in blocked_tests else False

def truncated_list(beg, end, arr):
    if len(arr) >= beg + 3 + end:
        hi = max(arr)
        lo = min(arr)
        s = [*arr[:beg], '...', *arr[beg:][-end:]]
        s = '['+', '.join(map(str, s))+'], min='+str(lo)+', max='+str(hi)
    else:
        s = '['+', '.join(map(str, arr))+']'
    return s

class knobNAR(IntEnum):
    NEVER  = 0
    ALWAYS = 1
    RANDOM = 2

class knobNA(IntEnum):
    NEVER  = 0
    ALWAYS = 1

class testConfig:
    __slots__ = ['gpu_arch', 'gpu_info', 'cudnn_ver', 'blocked_tests', 'implementation', 'cfg']

    def __init__(self, *, gpu_arch, gpu_info, cudnn_ver, blocked_tests, implementation):
        assert type(gpu_arch) == type(gpu_info) == type(cudnn_ver) == str, "expecting strings as arguments"
        assert isinstance(blocked_tests, list), "argument 'blocked_tests' must be list"

        # Initialize all attributes to None.
        for k in self.__slots__:
            setattr(self, k, None)

        self.gpu_arch      = gpu_arch
        self.gpu_info      = gpu_info
        self.cudnn_ver     = cudnn_ver
        self.blocked_tests = blocked_tests

        self.implementation = implementation

        self.cfg = exec_cfg()


    def showConfig(self, test_no, request, reg_run=True):
        if request.config.option.dryrun == 0 or request.config.option.dryrun == 1:
            if request.config.option.dryrun == 0:
                print("\n" + "=" * 90)
            else:
                print("\n" + "=" * 40 + "Dry-RUN" + "=" * 40)
            print(f"#### Test #{test_no[0]} of {test_no[1]} at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")
            print(f"test_name        = {request.node.name}")
            # print(f"geom_seed        = {self.geom_seed}")
            # print(f"data_seed        = {self.data_seed}")
            print(f"platform_info    = {self.gpu_arch} ({self.gpu_info}), cudnn_ver={self.cudnn_ver}")
            print(f"rng_data_seed    = {self.cfg.rng_data_seed}")
            # print(f"head_group       = {self.cfg.head_group}")
            # print(f"layout           = {self.in_layout}->{self.out_layout}")
            print(f"basic_dims       = [b={self.cfg.batches}, h_q={self.cfg.h_q}, h_k={self.cfg.h_k}, h_v={self.cfg.h_v}, d_qk={self.cfg.d_qk}, d_v={self.cfg.d_v}, s_q={self.cfg.s_q}, s_kv={self.cfg.s_kv}]")
            print(f"shape_q(b,h,s,d) = {self.cfg.shape_q}, strides={self.cfg.stride_q}, elems={self.cfg.elems_q}")
            print(f"shape_k(b,h,s,d) = {self.cfg.shape_k}, strides={self.cfg.stride_k}, elems={self.cfg.elems_k}")
            print(f"shape_v(b,h,s,d) = {self.cfg.shape_v}, strides={self.cfg.stride_v}, elems={self.cfg.elems_v}")
            print(f"shape_o(b,h,s,d) = {self.cfg.shape_o}, strides={self.cfg.stride_o}, elems={self.cfg.elems_o}")
            
            print(f"is_infer         = {self.cfg.is_infer}")
            print(f"is_padding       = {self.cfg.is_padding} ({'ragged' if self.cfg.is_ragged else 'no ragged'})")
            print(f"is_alibi         = {self.cfg.is_alibi}")
            print(f"is_paged         = {self.cfg.is_paged} (block_size={self.cfg.block_size})")
            print(f"is_bias          = {self.cfg.is_bias}")
            print(f"is_block_mask    = {self.cfg.is_block_mask}")
            print(f"is_dropout       = {self.cfg.is_dropout}")
            if self.cfg.is_infer == False:
                print(f"is_determin      = {self.cfg.is_determin}")
            print(f"diag_align       = {self.cfg.diag_align}")
            print(f"left_bound       = {self.cfg.left_bound}", '(NO BOUND)' if self.cfg.left_bound == INVALID_BOUND else '')
            print(f"right_bound      = {self.cfg.right_bound}", '(NO BOUND)' if self.cfg.right_bound == INVALID_BOUND else '')
            # print(f"seq_len_q        = {truncated_list(20, 3, self.seq_len_q)}")
            # print(f"seq_len_kv       = {truncated_list(20, 3, self.seq_len_kv)}")
            print(f"data_type        = {self.cfg.data_type}")
            print(f"implementation   = {self.cfg.implementation.name}")
            if reg_run:
                # Convert enums to integers and handle torch dtypes for proper serialization
                cfg_dict = asdict(self.cfg)
                # Convert enum values to integers
                if cfg_dict.get('diag_align') is not None:
                    cfg_dict['diag_align'] = cfg_dict['diag_align'].value
                if cfg_dict.get('implementation') is not None:
                    cfg_dict['implementation'] = cfg_dict['implementation'].name
                # Convert torch dtype to string
                if cfg_dict.get('data_type') is not None:
                    cfg_dict['data_type'] = str(cfg_dict['data_type'])
                print(f"repro_cmd        = pytest -vv -s -rA {request.module.__file__}::test_repro --repro \"{repr(cfg_dict)}\"")
        elif request.config.option.dryrun == 2:
            print(f"\npytest -vv -s -rA {request.module.__file__}::{request.node.name} --geom_seed {self.geom_seed} --data_seed {self.data_seed}")
        elif request.config.option.dryrun == 3:
            print(f"repro_cmd        = pytest -vv -s -rA {request.module.__file__}::{request.node.name} --geom_seed {self.geom_seed} --data_seed {self.data_seed}")

        else:
            assert False, "wrong --dryrun command line option"

        # Make sure to flush everything out.
        print(" ", flush=True)


    def avoid_invalid_configs(self, avoid_invalid_configs):
        if avoid_invalid_configs == avoid_invalid_configs.ALWAYS:
            # LIMIT: always is_determin=True in inference.
            if self.is_infer:
                self.is_determin = True

            # LIMIT: Paged attention only in inference.
            if not self.is_infer:
                self.is_paged = False

            # LIMIT: Paged caches can only be used in combination with padding mask (variable sequence length).
            if self.is_paged and not self.is_padding:
                self.is_paged = False

            # LIMIT: Paged caches cannot be used with ragged offsets (packed variable sequence lengths).
            if self.is_paged and self.is_ragged:
                self.is_paged = False
        
            # LIMIT: left and right bounds are only supported with is_dropout=False, is_bias=False.
            if self.left_bound != INVALID_BOUND and self.right_bound != INVALID_BOUND:
                self.is_dropout = False
                self.is_bias = False

            # LIMIT: when alibi mask is used, diagonal_band_right_bound needs to be exactly 0 (not INVALID_BOUND).
            if self.is_alibi and self.right_bound != 0:
                self.is_alibi = False

            # LIMIT: bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False.
            if self.diag_align == self.diag_align.BOTTOM_RIGHT and (self.left_bound != INVALID_BOUND or self.right_bound != INVALID_BOUND):
                self.is_bias    = False
                self.is_alibi   = False
                self.is_dropout = False

            # LIMIT: Left or right bounds are only supported with is_dropout=False, is_bias=False.
            if self.left_bound != INVALID_BOUND or self.right_bound != INVALID_BOUND:
                self.is_dropout = False
                self.is_bias    = False

            # LIMIT: Left bound (a.k.a sliding window) does not support s_q > s_kv
            if self.left_bound != INVALID_BOUND and self.s_q.val > self.s_kv.val:
                self.left_bound = INVALID_BOUND

            # LIMIT: Bottom right causal mask does not support s_q > s_kv. 
            if self.s_q.val > self.s_kv.val and self.diag_align == self.diag_align.BOTTOM_RIGHT and self.right_bound != INVALID_BOUND:
                self.right_bound = INVALID_BOUND
            
            if not self.is_infer:
                self.is_block_mask = False

def compute_ref(
    q,
    k,
    v,
    attn_scale=None,
    bias=None,
    block_mask=None,
    is_alibi=False,
    padding=None,
    diag_align=cudnn.diagonal_alignment.TOP_LEFT,
    left_bound=INVALID_BOUND,
    right_bound=INVALID_BOUND,
    dropout_prob=0.0,
    dropout_mask=None,
    generate_stats=False,
    device="cuda",
):
    b, h_q, s_q, d_qk = q.shape
    _, h_k, s_kv, _ = k.shape
    _, h_v, _, d_v = v.shape

    assert k.shape == (b, h_k, s_kv, d_qk)
    assert v.shape == (b, h_v, s_kv, d_v)

    # use float32 datatype and math for reference computation
    q = q.to(dtype=torch.float32, device=device)
    k = k.to(dtype=torch.float32, device=device)
    v = v.to(dtype=torch.float32, device=device)

    # expand tensors for GQA and MQA
    if h_q != h_k:
        assert h_q % h_k == 0
        k = k.unsqueeze(2)
        k = k.expand(-1, -1, h_q // h_k, -1, -1)
        k = k.reshape(k.size(0), -1, k.size(3), k.size(4))
    if h_q != h_v:
        assert h_q % h_v == 0
        v = v.unsqueeze(2)
        v = v.expand(-1, -1, h_q // h_v, -1, -1)
        v = v.reshape(v.size(0), -1, v.size(3), v.size(4))

    if left_bound != INVALID_BOUND:
        swa_mask_zero = torch.ones(1, 1, s_q, 1, dtype=torch.bool, device=device)
        swa_mask_zero[:, :, s_kv + left_bound - 1 :, :] = False
        q = q * swa_mask_zero

    # generate masks to compute reference values for padding mask (also called variable sequence length)
    if padding is not None:
        q_mask = torch.zeros(b, 1, s_q, 1, dtype=torch.bool, device=device)
        k_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        v_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        s_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        p_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        seq_len_q, seq_len_kv = padding
        for i, (m, n) in enumerate(zip(seq_len_q, seq_len_kv)):
            q_mask[i, :, m:, :] = True
            k_mask[i, :, n:, :] = True
            v_mask[i, :, n:, :] = True
            s_mask[i, :, :, n:] = True
            p_mask[i, :, m:, :] = True

        q = q.masked_fill(q_mask, 0.0)
        k = k.masked_fill(k_mask, 0.0)
        v = v.masked_fill(v_mask, 0.0)

    s = torch.einsum("bhqd,bhkd->bhqk", q, k)
    if attn_scale is not None:
        s = s * attn_scale

    # Attention masks are applied in the following order:
    # - Bias mask
    # - Alibi mask
    # - Padding mask
    # - Causal mask
    if bias is not None:
        s = s + bias
    if is_alibi:
        index_row = torch.arange(s_q, dtype=torch.float32, device=device).view(-1, 1)
        index_col = torch.arange(s_kv, dtype=torch.float32, device=device)
        distance = index_col - index_row

        # Get the closest power of 2 to `n_heads`.
        # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
        # and then add the remaining slopes.
        n = 2 ** math.floor(math.log2(h_q))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, 1 + n))

        # If `n_heads` is not a power of 2, then we add the remaining slopes.
        # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
        # And pick the slopes upto `n_heads`.
        if n < h_q:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (h_q - n), 2))
            # Concatenate the slopes with the remaining slopes.
            m = torch.cat([m, m_hat])

        # Reshape the tensor to [1, num_heads, 1, 1]
        m = m.view(1, -1, 1, 1).to(device=device)

        alibi_mask = distance.to(dtype=torch.float32) * m
        s = s + alibi_mask

    if padding is not None:
        s = s.masked_fill(s_mask, float("-inf"))

    if diag_align == diag_align.TOP_LEFT and right_bound != INVALID_BOUND:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1 + right_bound)
        s = s.masked_fill(causal_mask, float("-inf"))
    elif diag_align == diag_align.BOTTOM_RIGHT and right_bound != INVALID_BOUND:
        causal_mask_bottom_right = None
        if padding:
            causal_mask_bottom_right = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
            seq_len_q, seq_len_kv = padding
            for i in range(b):
                causal_mask_bottom_right[i, :, :, :].triu_(diagonal=seq_len_kv[i] - seq_len_q[i] + 1 + right_bound)
        else:
            causal_mask_bottom_right = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
            causal_mask_bottom_right.triu_(diagonal=s_kv - s_q + 1 + right_bound)
        s = s.masked_fill(causal_mask_bottom_right, float("-inf"))

    if left_bound != INVALID_BOUND:
        assert diag_align is not None
        if diag_align == diag_align.TOP_LEFT:
            swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
            swa_mask.tril_(diagonal=-1 * left_bound)
        elif diag_align == diag_align.BOTTOM_RIGHT:
            # BRCM + SWA for variable sequence lengths
            if padding:
                swa_mask = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
                seq_len_q, seq_len_kv = padding
                for i in range(b):
                    swa_mask[i, :, :, :].tril_(diagonal=seq_len_kv[i] - seq_len_q[i] - left_bound)
            # BRCM + SWA for fixed sequence lengths
            else:
                swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
                swa_mask.tril_(diagonal=-1 * left_bound + (s_kv - s_q))
        swa_mask &= swa_mask_zero.view(s_q, 1)
        s = s.masked_fill(swa_mask, float("-inf"))

    if block_mask is not None:
        TILE_M = 128
        TILE_N = 128

        block_mask = block_mask.to(dtype=torch.uint8, device=device)
        block_mask = ((block_mask[..., None] & (1 << torch.arange(8, device=block_mask.device))) != 0).reshape(block_mask.shape[0], block_mask.shape[1], block_mask.shape[2], block_mask.shape[3] * 8)
        block_mask = block_mask.unsqueeze(3).unsqueeze(5)
        block_mask = block_mask.repeat(1, 1, 1, TILE_M, 1, TILE_N)
        block_mask = block_mask.reshape(block_mask.shape[0], block_mask.shape[1], block_mask.shape[2] * TILE_M, block_mask.shape[4] * TILE_N)
        block_mask = block_mask[:, :, :s_q, :s_kv]
        s += torch.where(block_mask, torch.tensor(0.0), torch.tensor(float('-inf')))

    p = torch.softmax(s, dim=-1)

    if block_mask is not None:
        all_inf = torch.isneginf(s).all(dim=-1, keepdim=True)
        if torch.any(all_inf):
            p = torch.where(all_inf, torch.zeros_like(p), p)

    if left_bound != INVALID_BOUND:
        p = p * swa_mask_zero
    if padding is not None:
        p = p.masked_fill(p_mask, 0.0)

    # apply dropout mask over softmax outputs
    if dropout_prob != 0.0:
        assert dropout_mask != None, "PyTorch reference must have dropout_mask for dropout"
        p = (p * dropout_mask) / (1 - dropout_prob)

    o = torch.einsum("bhqk,bhkd->bhqd", p, v)

    # softmax stats is used for backwards computation
    if generate_stats:
        # amax (NOT absolute max) is used here to evenly distribute gradient
        row_max = torch.amax(s, -1, True)
        row_exp = torch.exp(s - row_max)
        row_sum = torch.sum(row_exp, -1, True)
        stats = row_max + torch.log(row_sum)
        return o, stats

    return o

# Compute the exclusive prefix sum for ragged sequence dimension
# input tensor has shape (B, 1, 1, 1)
# output tensor has shape (B+1, 1, 1, 1)
# example input seq_len: [2, 4, 1, 6] (along the B dimension)
# example output ragged_offset: [0, 2, 6, 7, 13] (along the B dimension)
def compute_exclusive_prefix_sum(tensor):
    assert list(tensor.size())[1:]==[1,1,1]
    # We need to provide a tuple of two tensors to torch.cat().
    return torch.cat((torch.zeros(1, 1, 1, 1, dtype=tensor.dtype, device=tensor.device), torch.cumsum(tensor, dim=0)))

def generate_ragged_offset(h_q, h_k, h_v, d_qk, d_v, seq_len_q, seq_len_kv):
    # Only for thd_thd_thd
    q_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_qk
    k_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_k * d_qk
    v_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_v * d_v
    o_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_v

    # Convert to int64 for cuDNN 9.6.0
    q_ragged_offset = q_ragged_offset.to(dtype=torch.int64)
    k_ragged_offset = k_ragged_offset.to(dtype=torch.int64)
    v_ragged_offset = v_ragged_offset.to(dtype=torch.int64)
    o_ragged_offset = o_ragged_offset.to(dtype=torch.int64)

    return q_ragged_offset, k_ragged_offset, v_ragged_offset, o_ragged_offset

def convert_ragged_to_uniform(ragged_tensor, seq_len):
    # limitations:
    # 1. tensor is bhsd dim order and bshd stride order (may be interleaved)
    # 2. ragged tensor is packed and in-order, therefore
    #    ragged offset is monatomically increasing
    assert ragged_tensor.dim() == 4
    b, h, s, d = ragged_tensor.size()
    b_stride, h_stride, s_stride, d_stride = ragged_tensor.stride()
    assert b_stride >= s_stride >= h_stride >= d_stride
    assert seq_len.dim() == 4 and (b, 1, 1, 1) == seq_len.size()

    # ragged offset is given in 4D, convert to 1D locally
    seq_len = seq_len.flatten()

    # convert bhsd to bshd and flatten
    uniform_tensor = torch.zeros(b, s, h, d).to(
        dtype=ragged_tensor.dtype, device=ragged_tensor.device
    )
    ragged_tensor_thd = torch.einsum("bhsd->bshd", ragged_tensor).reshape(b * s, h, d)

    # copy
    t = 0
    for b, s in enumerate(seq_len):
        uniform_tensor[b, 0:s, :, :] = ragged_tensor_thd[t : t + s, :, :]
        t += s

    # convert back to bshd to bhsd
    uniform_tensor = torch.einsum("bshd->bhsd", uniform_tensor)
    return uniform_tensor

def create_container_and_page_table(tensor, block_size):
    B, H, S, D = tensor.shape
    # num_blocks = math.ceil(S/block_size) * B
    blocks_per_batch = math.ceil(S/block_size)

    padding_seq = (blocks_per_batch * block_size) - S
    if padding_seq > 0:
        zeros = torch.zeros(B,H,padding_seq,D, device='cuda', dtype=tensor.dtype)
        cat_tensor = torch.cat((tensor, zeros), axis = 2)
    else:
        cat_tensor = tensor

    reshaped = torch.cat((cat_tensor.clone()).chunk(blocks_per_batch, dim=2), dim=0)

    table_size = math.ceil(S/block_size)
    page_table = torch.linspace(0, B*table_size-1, B*table_size, device='cuda', dtype=torch.int32).reshape(table_size,1,B,1)
    page_table = torch.transpose(page_table,0,2)

    return(reshaped, page_table)

def exec_sdpa(cfg, request, cudnn_handle):
    # Do not run any test when --dryrun option is provided.

    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    # # Check if the test is temporarily blocked.
    # if is_test_blocked(request.node.name, cfg.blocked_tests):
    #     print(f"\nWARNING: test '{request.node.name}' is blocked on {cfg.gpu_arch} and cuDNN {cfg.cudnn_ver}")
    #     print("@@@@ Overall result: SKIPPED, test blocked.")
    #     pytest.skip("test blocked")

    # ============================
    # Basic parameter check.
    # ============================

    if not all((x > 0 and type(x) == int) for x in (cfg.batches, cfg.d_qk, cfg.d_v, cfg.s_q, cfg.s_kv, cfg.h_q, cfg.h_k, cfg.h_v)):
       assert False, "tensor dimensions must be integer and positive"

    assert cfg.shape_q == (cfg.batches, cfg.h_q, cfg.s_q, cfg.d_qk), f"wrong shape_q={cfg.shape_q}"
    assert cfg.shape_k == (cfg.batches, cfg.h_k, cfg.s_kv, cfg.d_qk), f"wrong shape_k={cfg.shape_k}"
    assert cfg.shape_v == (cfg.batches, cfg.h_v, cfg.s_kv, cfg.d_v), f"wrong shape_v={cfg.shape_v}"
    assert cfg.shape_o == (cfg.batches, cfg.h_q, cfg.s_q, cfg.d_v), f"wrong shape_o={cfg.shape_o}"

    if not cfg.is_infer:
        assert cfg.is_paged == False and cfg.block_size == None, "paged attention not allowed in backward pass"

    if cfg.is_ragged:
        assert cfg.is_padding == True, "is_ragged=True and is_padding=False not allowed"

    assert isinstance(cfg.seq_len_q, (list, tuple)), "input 'seq_len_q' must be list or tuple"
    if cfg.is_padding:
        assert len(cfg.seq_len_q) == cfg.batches, f"wrong 'seq_len_q' length"
    else:
        assert len(cfg.seq_len_q) == 0, f"wrong 'seq_len_q' length, expecting 0"

    assert isinstance(cfg.seq_len_kv, (list, tuple)), "input 'seq_len_kv' must be list or tuple"
    if cfg.is_padding:
        assert len(cfg.seq_len_kv) == cfg.batches, f"wrong 'seq_len_kv' length, expecting {cfg.batches}"
    else:
        assert len(cfg.seq_len_kv) == 0, f"wrong 'seq_len_kv' length, expecting 0"

    assert all(x >= 0 and type(x) == int for x in cfg.seq_len_q), f"wrong seq_len_q={cfg.seq_len_q}"
    assert all(x >= 0 and type(x) == int for x in cfg.seq_len_kv), f"wrong seq_len_kv={cfg.seq_len_kv}"

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.10.0":
        print("@@@@ Overall result: WAIVED, test_mhas_v2.py supports cudnn 9.10.0 or higher.")
        pytest.skip("test_mhas_v2.py requires cudnn 9.10.0 or higher")

    if cudnn_version < "9.13.1" and cfg.implementation == cudnn.attention_implementation.UNIFIED:
        print("@@@@ Overall result: WAIVED, unified SDPA implementation requires cudnn 9.13.1 or higher.")
        pytest.skip("unified SDPA implementation requires cudnn 9.13.1 or higher")

    if cfg.s_q == cfg.s_kv == 1:
        print("@@@@ Overall result: WAIVED, skipping known issue of s_q == s_kv == 1.")
        pytest.skip("skipping known issue of s_q == s_kv == 1")

    qkv_num_elems = cfg.elems_q + cfg.elems_k + cfg.elems_v

    rng_data_gen = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)

    (q_gpu, _, _) = alloc_tensor(cfg.shape_q, cfg.data_type, elems=cfg.elems_q, strides=cfg.stride_q, rng=rng_data_gen, mean=-0.5, std=1.0)
    (k_gpu, _, _) = alloc_tensor(cfg.shape_k, cfg.data_type, elems=cfg.elems_k, strides=cfg.stride_k, rng=rng_data_gen, mean=-0.5, std=1.0)
    (v_gpu, _, _) = alloc_tensor(cfg.shape_v, cfg.data_type, elems=cfg.elems_v, strides=cfg.stride_v, rng=rng_data_gen, mean=-0.5, std=1.0)
    (bias_gpu, _, _) = (alloc_tensor((1, cfg.h_q, cfg.s_q, cfg.s_kv), cfg.data_type, rng=rng_data_gen, mean=0.0, std=1.0) if cfg.is_bias else (None, None, None))

    TILE_M = 128
    TILE_N = 128
    block_mask_gpu = torch.randint(0, 256, (cfg.batches, cfg.h_q, (cfg.s_q + TILE_M - 1) // TILE_M, ((cfg.s_kv + TILE_N - 1) // TILE_N + 7) // 8), dtype=torch.uint8, device="cuda")

    if not cfg.is_infer:
        (dQ_gpu, dQ_sep, dQ_raw) = alloc_tensor(cfg.shape_q, cfg.data_type, elems=cfg.elems_q, strides=cfg.stride_q)
        (dK_gpu, dK_sep, dK_raw) = alloc_tensor(cfg.shape_k, cfg.data_type, elems=cfg.elems_k, strides=cfg.stride_k)
        (dV_gpu, dV_sep, dV_raw) = alloc_tensor(cfg.shape_v, cfg.data_type, elems=cfg.elems_v, strides=cfg.stride_v)
        (dBias_gpu, dBias_sep, dBias_raw) = (alloc_tensor((1, cfg.h_q, cfg.s_q, cfg.s_kv), cfg.data_type) if cfg.is_bias else (None, None, None))
        (dO_gpu, dO_sep, dO_raw) = alloc_tensor(cfg.shape_o, cfg.data_type, elems=cfg.elems_o, strides=cfg.stride_o, rng=rng_data_gen, mean=0.0, std=0.1)

    # Sequence lengths for gpu, must be a four dimensional tensor.
    seq_len_q_gpu = seq_len_kv_gpu = None
    if len(cfg.seq_len_q) > 0:
        seq_len_q_gpu = torch.tensor(cfg.seq_len_q, dtype=torch.int32, device="cuda")
        seq_len_q_gpu = seq_len_q_gpu[:, None, None, None]  # batches x 1 x 1 x 1
    if len(cfg.seq_len_kv) > 0:
        seq_len_kv_gpu = torch.tensor(cfg.seq_len_kv, dtype=torch.int32, device="cuda")
        seq_len_kv_gpu = seq_len_kv_gpu[:, None, None, None]  # batches x 1 x 1 x 1

    # maxT = next_multiple_of_64(sum(seq_len))
    max_t_q = ((torch.sum(seq_len_q_gpu).item() + 63) // 64) * 64 if cfg.is_ragged else None
    max_t_kv = ((torch.sum(seq_len_kv_gpu).item() + 63) // 64) * 64 if cfg.is_ragged else None

    if cfg.is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    rng_dump_gpu = torch.zeros((cfg.batches, cfg.h_q, cfg.s_q, cfg.s_kv), dtype=torch.float32, device="cuda") if cfg.is_dropout else None

    if cfg.is_ragged:
       q_ragged_offset_gpu, k_ragged_offset_gpu, v_ragged_offset_gpu, o_ragged_offset_gpu = generate_ragged_offset(cfg.h_q, cfg.h_k, cfg.h_v, cfg.d_qk, cfg.d_v, seq_len_q_gpu, seq_len_kv_gpu)

    (o_gpu, o_sep, o_raw) = alloc_tensor(cfg.shape_o, cfg.data_type, elems=cfg.elems_o, strides=cfg.stride_o)
    (stats_gpu, stats_sep, stats_raw) = (alloc_tensor((cfg.batches, cfg.h_q, cfg.s_q, 1), torch.float32) if not cfg.is_infer else (None, None, None))

    container_k_gpu  = None
    container_v_gpu  = None
    page_table_k_gpu = None
    page_table_v_gpu = None

    if cfg.is_paged:
        container_k_gpu, page_table_k_gpu = create_container_and_page_table(k_gpu, cfg.block_size)
        container_v_gpu, page_table_v_gpu = create_container_and_page_table(v_gpu, cfg.block_size)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    # Forward cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(cfg.data_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu) if not cfg.is_paged else graph.tensor_like(container_k_gpu)
    v = graph.tensor_like(v_gpu) if not cfg.is_paged else graph.tensor_like(container_v_gpu)

    page_table_k = graph.tensor_like(page_table_k_gpu) if cfg.is_paged else None
    page_table_v = graph.tensor_like(page_table_v_gpu) if cfg.is_paged else None

    bias = graph.tensor_like(bias_gpu) if cfg.is_bias else None
    block_mask = graph.tensor_like(block_mask_gpu) if cfg.is_block_mask else None

    seq_len_q = graph.tensor_like(seq_len_q_gpu) if cfg.is_padding else None
    seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if cfg.is_padding else None

    if cfg.is_dropout:
        seed = graph.tensor_like(seed_gpu)
        offset = graph.tensor_like(offset_gpu)
        dropout_tuple = (cfg.dropout_prob, seed, offset)

    rng_dump = graph.tensor_like(rng_dump_gpu) if cfg.is_dropout else None

    q_ragged_offset = graph.tensor_like(q_ragged_offset_gpu) if cfg.is_ragged else None
    k_ragged_offset = graph.tensor_like(k_ragged_offset_gpu) if cfg.is_ragged else None
    v_ragged_offset = graph.tensor_like(v_ragged_offset_gpu) if cfg.is_ragged else None
    o_ragged_offset = graph.tensor_like(o_ragged_offset_gpu) if cfg.is_ragged else None

    if cfg.is_ragged:
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)

    attn_scale = 0.125

    o, stats = graph.sdpa(
        name="sdpa_forward",
        q=q,
        k=k,
        v=v,
        generate_stats=not cfg.is_infer,
        attn_scale=attn_scale,
        bias=bias,
        block_mask=block_mask,
        use_alibi_mask=cfg.is_alibi,
        use_padding_mask=cfg.is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        diagonal_band_left_bound=cfg.left_bound if cfg.left_bound != INVALID_BOUND else None,
        diagonal_band_right_bound=cfg.right_bound if cfg.right_bound != INVALID_BOUND else None,
        diagonal_alignment=cfg.diag_align,
        dropout=dropout_tuple if cfg.is_dropout else None,
        rng_dump=rng_dump,
        paged_attention_k_table=page_table_k,
        paged_attention_v_table=page_table_v,
        paged_attention_max_seq_len_kv=cfg.s_kv if cfg.is_paged else None,
        implementation=cfg.implementation,
    )

    o.set_output(True).set_dim(cfg.shape_o).set_stride(cfg.stride_o)
    if cfg.is_ragged:
        o.set_ragged_offset(o_ragged_offset)

    if cfg.is_infer == False:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    try:
        graph.validate()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"@@@@ Overall result: WAIVED, not supported forward graph. {e}")
        pytest.skip("not supported forward graph")
    except Exception as e:
        print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception during forward graph validate. {e}")
        pytest.fail("unexpected exception during forward graph validate", pytrace=False)

    try:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"@@@@ Overall result: WAIVED, not supported forward graph after validate. {e}")
        pytest.skip("not supported forward graph after validate")
    except Exception as e:
        print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception after forward validate. {e}")
        pytest.fail("unexpected exception after forward validate", pytrace=False)

    variant_pack = {
        q: q_gpu,
        k: k_gpu if not cfg.is_paged else container_k_gpu,
        v: v_gpu if not cfg.is_paged else container_v_gpu,
        bias: bias_gpu,
        block_mask: block_mask_gpu if cfg.is_block_mask else None,
        seq_len_q: seq_len_q_gpu,
        seq_len_kv: seq_len_kv_gpu,
        q_ragged_offset: q_ragged_offset_gpu if cfg.is_ragged else None,
        k_ragged_offset: k_ragged_offset_gpu if cfg.is_ragged else None,
        v_ragged_offset: v_ragged_offset_gpu if cfg.is_ragged else None,
        o_ragged_offset: o_ragged_offset_gpu if cfg.is_ragged else None,
        o: o_gpu,
        stats: stats_gpu,
        rng_dump: rng_dump_gpu,
        page_table_k: page_table_k_gpu,
        page_table_v: page_table_v_gpu
    }

    if cfg.is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    # Allocate workspace for the forward call.
    (workspace, ws_sep, _) = alloc_tensor(graph.get_workspace_size(), torch.uint8)

    # Display available memory.
    # torch.cuda.empty_cache()
    # free_mem, total_mem = torch.cuda.mem_get_info()
    # print(f"Free GPU memory (before forward): {free_mem / (1024**3):.4f} GB of {total_mem / (1024**3):.4f} GB")

    if request.config.getoption("--perf"):
        forward_times_ms = time_execution(graph.execute, variant_pack, workspace, cudnn_handle)
        print(f"@@@@ Forward graph.execute avg_time_ms={forward_times_ms.mean().item():.3f}")
        profile_execution(graph.execute, variant_pack, workspace, cudnn_handle)

    # Execute forward cuDNN graph
    graph.execute(variant_pack, workspace, cudnn_handle)
    torch.cuda.synchronize()

    if ws_sep is not None and not torch.all(ws_sep==-1).item():
        print("@@@@ Overall result: FAILED, forward workspace overwritten outside its boundaries.")
        print(ws_sep)
        pytest.fail("forward workspace overwritten outside boundaries", pytrace=False)

    if not cfg.is_infer:
        if cudnn_version < "8.9.6" and cfg.is_padding:
            # zero out padded region of the output and stats
            for i, m in enumerate(seq_len_q_gpu):
                o_gpu[i, :, m:, :] = 0
                stats_gpu[i, :, m:, :] = 0

        stream = torch.cuda.current_stream().cuda_stream  #2
        cudnn.set_stream(handle=cudnn_handle, stream=stream)
        sm_version = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]

        # Backward cuDNN graph
        graph = cudnn.pygraph(
            io_data_type=convert_to_cudnn_type(cfg.data_type),
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
            sm_version = sm_version
        )

        q = graph.tensor_like(q_gpu)
        k = graph.tensor_like(k_gpu)
        v = graph.tensor_like(v_gpu)
        o = graph.tensor_like(o_gpu)
        dO = graph.tensor_like(dO_gpu)
        stats = graph.tensor_like(stats_gpu)

        bias = graph.tensor_like(bias_gpu) if cfg.is_bias else None
        dBias = (graph.tensor_like(dBias_gpu).set_stride((cfg.h_q * cfg.s_q * cfg.s_kv, cfg.s_q * cfg.s_kv, cfg.s_kv, 1)) if cfg.is_bias else None)

        seq_len_q = graph.tensor_like(seq_len_q_gpu) if cfg.is_padding else None
        seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if cfg.is_padding else None

        if cfg.is_dropout:
            seed = graph.tensor_like(seed_gpu)
            offset = graph.tensor_like(offset_gpu)
            dropout_tuple = (cfg.dropout_prob, seed, offset)

        q_ragged_offset = graph.tensor_like(q_ragged_offset_gpu) if cfg.is_ragged else None
        k_ragged_offset = graph.tensor_like(k_ragged_offset_gpu) if cfg.is_ragged else None
        v_ragged_offset = graph.tensor_like(v_ragged_offset_gpu) if cfg.is_ragged else None
        o_ragged_offset = graph.tensor_like(o_ragged_offset_gpu) if cfg.is_ragged else None

        if cfg.is_ragged:
            q.set_ragged_offset(q_ragged_offset)
            k.set_ragged_offset(k_ragged_offset)
            v.set_ragged_offset(v_ragged_offset)
            o.set_ragged_offset(o_ragged_offset)
            dO.set_ragged_offset(o_ragged_offset)

        dQ, dK, dV = graph.sdpa_backward(
            name="sdpa_backward",
            q=q,
            k=k,
            v=v,
            o=o,
            dO=dO,
            stats=stats,
            attn_scale=attn_scale,
            bias=bias,
            dBias=dBias,
            use_alibi_mask=cfg.is_alibi,
            use_padding_mask=cfg.is_padding,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            max_total_seq_len_q=max_t_q,
            max_total_seq_len_kv=max_t_kv,
            diagonal_band_left_bound=cfg.left_bound if cfg.left_bound != INVALID_BOUND else None,
            diagonal_band_right_bound=cfg.right_bound if cfg.right_bound != INVALID_BOUND else None,
            diagonal_alignment=cfg.diag_align,
            dropout=dropout_tuple if cfg.is_dropout else None,
            use_deterministic_algorithm=cfg.is_determin,
        )

        dQ.set_output(True).set_dim(dQ_gpu.size()).set_stride(dQ_gpu.stride())
        dK.set_output(True).set_dim(dK_gpu.size()).set_stride(dK_gpu.stride())
        dV.set_output(True).set_dim(dV_gpu.size()).set_stride(dV_gpu.stride())
        if cfg.is_ragged:
            dQ.set_ragged_offset(q_ragged_offset)
            dK.set_ragged_offset(k_ragged_offset)
            dV.set_ragged_offset(v_ragged_offset)

        try:
            graph.validate()
        except cudnn.cudnnGraphNotSupportedError as e:
            print(f"@@@@ Overall result: WAIVED, not supported backward graph. {e}")
            pytest.skip("not supported backward graph")
        except Exception as e:
            print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception during backward graph validate. {e}")
            pytest.fail("unexpected exception during backward graph validate", pytrace=False)

        try:
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph.check_support()
            graph.build_plans()
        except cudnn.cudnnGraphNotSupportedError as e:
            print(f"@@@@ Overall result: WAIVED, not supported backward graph after validate. {e}")
            pytest.skip("not supported backward graph after validate")
        except Exception as e:
            print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception after backward validate. {e}")
            pytest.fail("unexpected exception after backward validate", pytrace=False)

        variant_pack = {
            q: q_gpu,
            k: k_gpu,
            v: v_gpu,
            o: o_gpu,
            dO: dO_gpu,
            stats: stats_gpu,
            dQ: dQ_gpu,
            dK: dK_gpu,
            dV: dV_gpu,
            bias: bias_gpu,
            dBias: dBias_gpu,
            seq_len_q: seq_len_q_gpu,
            seq_len_kv: seq_len_kv_gpu,
            q_ragged_offset: q_ragged_offset_gpu if cfg.is_ragged else None,
            k_ragged_offset: k_ragged_offset_gpu if cfg.is_ragged else None,
            v_ragged_offset: v_ragged_offset_gpu if cfg.is_ragged else None,
            o_ragged_offset: o_ragged_offset_gpu if cfg.is_ragged else None,
        }

        if cfg.is_dropout:
            variant_pack[seed] = seed_gpu
            variant_pack[offset] = offset_gpu

        # Allocate workspace for the backward call.
        (workspace, ws_sep, _) = alloc_tensor(graph.get_workspace_size(), torch.uint8)

        # Display available memory.
        # torch.cuda.empty_cache()
        # free_mem, total_mem = torch.cuda.mem_get_info()
        # print(f"Free GPU memory (before backward): {free_mem / (1024**3):.4f} GB of {total_mem / (1024**3):.4f} GB")

        if request.config.getoption("--perf"):
            backward_times_ms = time_execution(graph.execute, variant_pack, workspace, cudnn_handle)
            print(f"@@@@ Backward graph.execute avg_time_ms={backward_times_ms.mean().item():.3f}")
            profile_execution(graph.execute, variant_pack, workspace, cudnn_handle)

        # Execute backward cuDNN graph
        graph.execute(variant_pack, workspace, cudnn_handle)
        torch.cuda.synchronize()

        if ws_sep is not None and not torch.all(ws_sep==-1).item():
            print("@@@@ Overall result: FAILED, backward workspace overwritten outside its boundaries.")
            print(ws_sep)
            pytest.fail("backward workspace overwritten outside boundaries", pytrace=False)

    bias_ref = None
    rng_dump_ref = None

    if not cfg.is_infer:
        # Using torch autograd reference in the backward pass.
        q_ref  = q_gpu.detach().float().requires_grad_()
        k_ref  = k_gpu.detach().float().requires_grad_()
        v_ref  = v_gpu.detach().float().requires_grad_()
        dO_ref = dO_gpu.detach().float()
        if cfg.is_ragged:
            dO_ref = convert_ragged_to_uniform(dO_ref, seq_len_q_gpu.detach())
        if cfg.is_bias:
            bias_ref = bias_gpu.detach().float().requires_grad_()
    else:
        # No autograd in the forward pass.
        q_ref  = q_gpu.detach().float()
        k_ref  = k_gpu.detach().float()
        v_ref  = v_gpu.detach().float()
        dO_ref = None
        if cfg.is_bias:
            bias_ref = bias_gpu.detach().float()

    if cfg.is_ragged:
        q_ref  = convert_ragged_to_uniform(q_ref, seq_len_q_gpu.detach())
        k_ref  = convert_ragged_to_uniform(k_ref, seq_len_kv_gpu.detach())
        v_ref  = convert_ragged_to_uniform(v_ref, seq_len_kv_gpu.detach())

    if cfg.is_padding:
        seq_len_q_ref = seq_len_q_gpu.detach().flatten()
        seq_len_kv_ref = seq_len_kv_gpu.detach().flatten()

    if cfg.is_dropout:
        rng_dump_ref = rng_dump_gpu.detach().float()

    # Compute forward reference output.
    ret = compute_ref(
        q_ref,
        k_ref,
        v_ref,
        attn_scale=attn_scale,
        bias=bias_ref,
        block_mask=block_mask_gpu if cfg.is_block_mask else None,
        is_alibi=cfg.is_alibi,
        padding=(seq_len_q_ref, seq_len_kv_ref) if cfg.is_padding else None,
        left_bound=cfg.left_bound,
        right_bound=cfg.right_bound,
        diag_align=cfg.diag_align,
        dropout_prob=cfg.dropout_prob,
        dropout_mask=rng_dump_ref,
        generate_stats=(cfg.is_infer == False),
    )

    if not cfg.is_infer:
        o_ref, stats_ref = ret
    else:
        o_ref = ret

    if cfg.is_ragged:
        o_gpu = convert_ragged_to_uniform(o_gpu, seq_len_q_gpu.detach())

    err_count = 0

    if cfg.is_padding:
        # zero out padded region of the output for comparison
        for i, m in enumerate(seq_len_q_ref):
            o_ref[i, :, m:, :] = 0
            o_gpu[i, :, m:, :] = 0
            if cfg.is_infer == False:
                if cudnn_version < "9.14.0":
                    stats_ref[i, :, m:, :] = 0
                    stats_gpu[i, :, m:, :] = 0
                else:
                    stats_ref[i, :, m:, :] = -float("inf")

    diffs = int_cli_option(10, request, "--diffs")

    err_count += approx_equal(o_gpu, o_ref, o_sep, o_raw, atol=2e-2, rtol=2e-2, tag="o", disp_elems=diffs)

    if not cfg.is_infer:
        err_count += approx_equal(stats_gpu, stats_ref, stats_sep, stats_raw, atol=2e-2, rtol=2e-2, tag="stats", disp_elems=diffs)

        inputs_ref = [q_ref, k_ref, v_ref]
        if cfg.is_bias:
            inputs_ref.append(bias_ref)

        [dQ_ref, dK_ref, dV_ref, *opt_refs] = list(
            torch.autograd.grad(outputs=o_ref, inputs=inputs_ref, grad_outputs=dO_ref)
        )

        if cfg.is_bias:
            dBias_ref = opt_refs.pop(0)

        if cfg.is_ragged:
            dQ_gpu = convert_ragged_to_uniform(dQ_gpu, seq_len_q_gpu.detach())
            dK_gpu = convert_ragged_to_uniform(dK_gpu, seq_len_kv_gpu.detach())
            dV_gpu = convert_ragged_to_uniform(dV_gpu, seq_len_kv_gpu.detach())

        if cfg.is_padding:
            # zero out padded region of the output for comparison
            for i, (m, n) in enumerate(zip(seq_len_q_ref, seq_len_kv_ref)):
                dQ_ref[i, :, m:, :] = 0
                dQ_gpu[i, :, m:, :] = 0
                dK_ref[i, :, n:, :] = 0
                dK_gpu[i, :, n:, :] = 0
                dV_ref[i, :, n:, :] = 0
                dV_gpu[i, :, n:, :] = 0

        torch.cuda.synchronize()

        err_count += approx_equal(dQ_gpu, dQ_ref, dQ_sep, dQ_raw, atol=2e-2, rtol=2e-2, tag="dQ", disp_elems=diffs)
        err_count += approx_equal(dK_gpu, dK_ref, dK_sep, dK_raw, atol=2e-2 if cfg.data_type != torch.bfloat16 else 7e-2, rtol=2e-2, tag="dK", disp_elems=diffs)
        err_count += approx_equal(dV_gpu, dV_ref, dV_sep, dV_raw, atol=2e-2 if cfg.data_type != torch.bfloat16 else 7e-2, rtol=2e-2, tag="dV", disp_elems=diffs)
        if cfg.is_bias:
            err_count += approx_equal(dBias_gpu, dBias_ref, dBias_sep, dBias_raw, atol=2e-2, rtol=2e-2, tag="dBias", disp_elems=diffs)

    if err_count != 0:
        print("@@@@ Overall result: FAILED, disallowed mismatches")
        pytest.fail("disallowed mismatches", pytrace=False)
    else:
        print("@@@@ Overall result: PASSED, everything looks good!")
    
    del workspace
    del graph
    del variant_pack

    if cfg.is_paged:
        del container_k_gpu, container_v_gpu, page_table_k_gpu, page_table_v_gpu
    if cfg.is_ragged:
        del q_ragged_offset_gpu, k_ragged_offset_gpu, v_ragged_offset_gpu, o_ragged_offset_gpu
    if cfg.is_dropout:
        del seed_gpu, offset_gpu
        del rng_dump_gpu
        del rng_dump_ref
    if cfg.is_padding:
        del seq_len_q_gpu, seq_len_kv_gpu
        del seq_len_q_ref, seq_len_kv_ref

    del q_gpu, k_gpu, v_gpu, o_gpu
    if cfg.is_bias:
        del bias_gpu
    if not cfg.is_infer:
        del dQ_gpu, dK_gpu, dV_gpu, dO_gpu, stats_gpu
        if cfg.is_bias:
            del dBias_gpu

        del q_ref, k_ref, v_ref, dO_ref, o_ref, stats_ref
        if cfg.is_bias:
            del dBias_ref, bias_ref
        del dQ_ref, dK_ref, dV_ref
    else:
        del q_ref, k_ref, v_ref, o_ref
        if cfg.is_bias:
            del bias_ref

    del o_sep, o_raw
    if not cfg.is_infer:
        del dQ_sep, dQ_raw, dK_sep, dK_raw, dV_sep, dV_raw
        del stats_sep, stats_raw

    torch.cuda.empty_cache()

@pytest.fixture(scope="package")
def env_info(request):
    assert torch.cuda.is_available(), "no CUDA device"

    gpu_type = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    device   = torch.device('cuda:0')
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    gpu_arch     = f"SM_{gpu_type[0]}{gpu_type[1]}"
    gpu_info     = f"{sm_count} SM-s, {gpu_name}"
    cudnn_ver    = str(torch.backends.cudnn.version())
    blocked_file = str(request.path)
    blocked_file = blocked_file[:-3] + ".block"

    blocked_tests = fetch_blocked_tests(blocked_file, gpu_arch, cudnn_ver)
    show_blocked_tests(blocked_tests, gpu_arch, cudnn_ver)

    return {"gpu_arch": gpu_arch, "gpu_info": gpu_info, "cudnn_ver": cudnn_ver, "blocked_tests": blocked_tests}

# These options are common to all test lists
data_type_options      = {torch.float16 : 1, torch.bfloat16 : 2}
diag_alignment_options = [cudnn.diagonal_alignment.TOP_LEFT, cudnn.diagonal_alignment.BOTTOM_RIGHT]
implementation_options = [cudnn.attention_implementation.AUTO, cudnn.attention_implementation.COMPOSITE, cudnn.attention_implementation.UNIFIED]
implementation_names   = ['cudnn.attention_implementation.AUTO', 'cudnn.attention_implementation.COMPOSITE', 'cudnn.attention_implementation.UNIFIED']

# # ==================================
# # L0 fprop tests
# # ==================================
@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", tlist(num_tests=32, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_unified_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
    test.cfg.implementation = implementation_cli_option(cudnn.attention_implementation.UNIFIED, request, "--implementation")

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 bprop tests
# # ==================================

@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=844), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_bwd_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        is_deterministic=RandomChoice({True : 1, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 fprop tests with s_q=1
# # ==================================

@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=111), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_sq1_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":100, "s_q=s_kv":1, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", tlist(num_tests=32, rng_seed=111), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_sq1_unified_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":100, "s_q=s_kv":1, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
    test.cfg.implementation = implementation_cli_option(cudnn.attention_implementation.UNIFIED, request, "--implementation")

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # =====================================================
# # L0 lean attention, s_kv=513..2048
# # =====================================================

@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=222), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_lean_attn_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=513, s_kv_max=2048, s_q_distribution={"s_q=1":100, "s_q=s_kv":0, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=222), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_lean_attn_unified_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=513, s_kv_max=2048, s_q_distribution={"s_q=1":100, "s_q=s_kv":0, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
    test.cfg.implementation = implementation_cli_option(cudnn.attention_implementation.UNIFIED, request, "--implementation")

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)

# # ==================================
# # L0 ragged tests
# # ==================================

@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_ragged_unified_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
    test.cfg.implementation = implementation_cli_option(cudnn.attention_implementation.UNIFIED, request, "--implementation")

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_bwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 paged tests
# # ==================================

@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=64, s_kv_min=1, s_kv_max=512, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1,32,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
        test.cfg.is_paged = True
        test.cfg.implementation=cudnn.attention_implementation.COMPOSITE  # FIXNOW

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", tlist(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_unified_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=64, s_kv_min=1, s_kv_max=512, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1,32,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
        test.cfg.is_paged = True
    test.cfg.implementation = implementation_cli_option(cudnn.attention_implementation.UNIFIED, request, "--implementation")

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)

@pytest.mark.parametrize("test_no", tlist(num_tests=32, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_unified_block_mask_L0(env_info, test_no, request, cudnn_handle):

    test = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
        test.cfg.is_block_mask = True
    test.cfg.implementation = cudnn.attention_implementation.UNIFIED

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ===================
# # Single repro test
# # ===================

@pytest.mark.skipif("not config.getoption('--repro')", reason="used with '--repro' only")
@pytest.mark.L0
@pytest.mark.L1
@pytest.mark.L2
@pytest.mark.L3
@pytest.mark.L4
def test_repro(env_info, request, cudnn_handle):
    repro_str = request.config.getoption("--repro")
    cfg = testConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    print(f"repro_str: {repro_str}")

    # Parse the dictionary string and reconstruct the exec_cfg object
    import ast
    repro_dict = ast.literal_eval(repro_str)

    # Convert integer enum values back to enum objects
    if 'diag_align' in repro_dict and repro_dict['diag_align'] is not None:
        repro_dict['diag_align'] = cudnn.diagonal_alignment(repro_dict['diag_align'])
    if 'implementation' in repro_dict and repro_dict['implementation'] is not None:
        repro_dict['implementation'] = getattr(cudnn.attention_implementation, repro_dict['implementation'])
    # Convert string dtype back to torch dtype
    if 'data_type' in repro_dict and repro_dict['data_type'] is not None:
        if 'torch.float16' in repro_dict['data_type']:
            repro_dict['data_type'] = torch.float16
        elif 'torch.bfloat16' in repro_dict['data_type']:
            repro_dict['data_type'] = torch.bfloat16
        elif 'torch.float32' in repro_dict['data_type']:
            repro_dict['data_type'] = torch.float32

    cfg.cfg = exec_cfg(**repro_dict)
    print(f"cfg.cfg: {cfg.cfg}")

    cfg.showConfig((1,1), request, False)
    exec_sdpa(cfg.cfg, request, cudnn_handle)
