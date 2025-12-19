# fmt: off

import torch
import cudnn
import pytest
import argparse
from enum import IntEnum
from looseversion import LooseVersion
import math

from test_utils import torch_fork_set_rng

torch.nans = lambda *size, **kwargs: torch.full(size, float('nan'), **kwargs)

# sq1_*, sq4_*, sq32_*, sq64_*: BUG mismatches
TEST_CONFIGS_FWD = {
    "d128_f16":            {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "d64_f16":             {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "d128_f8e4m3":         {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},

    # cudnnTest replica:
    # ./cudnnTest -RgraphRunner -jsonTestName=LLM_paged_attention_fp8 -kv=dim_b:2 -kv=dim_qh:4 -kv=dim_qs:256 -kv=dim_kvs:256 -kv=dim_d:128 -kv=dim_kvh:4 -kv=Tin:CUDNN_DATA_FP8_E4M3 -kv=Tout:CUDNN_DATA_FP8_E4M3 -kv=atol:0.08 -kv=rtol:0.2 -minDevVer800 -backendEngine-1 -b -gpuRef -kv=block_size:16 -kv=table_size:16 -kv=max_block_num:31 -kv=dim_num_blocks:32 
    "d128_f8e4m3_paged":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2, "kv_block_size": 16},

    "d64_f8e4m3":          {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},
    "d128_f8e5m2":         {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.4},
    "d64_f8e5m2":          {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.4},

    "gqa_f16":             {"b": 2, "h_q": 15, "h_k": 5, "h_v": 3, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "gqa_f8e4m3":          {"b": 2, "h_q": 15, "h_k": 5, "h_v": 3, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},
    "gqa_f8e5m2":          {"b": 2, "h_q": 15, "h_k": 5, "h_v": 3, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.4},

    "sq1_skv256_f16":      {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq1_skv1024_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq1_skv256_f8e4m3":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},
    "sq1_skv1024_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},
    "sq1_skv256_f8e5m2":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq1_skv1024_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},

    "sq4_skv256_f16":      {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq4_skv1024_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq4_skv256_f8e4m3":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq4_skv1024_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq4_skv256_f8e5m2":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq4_skv1024_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},

    "sq8_skv128_f16":      {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 8,   "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq8_skv256_f16":      {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 8,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq8_skv1024_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 8,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq8_skv128_f8e4m3":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 8,   "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq8_skv256_f8e4m3":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 8,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq8_skv1024_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 8,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq8_skv128_f8e5m2":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 8,   "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq8_skv256_f8e5m2":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 8,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq8_skv1024_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 8,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},

    "sq16_skv128_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 16,  "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq16_skv256_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 16,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq16_skv1024_f16":    {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 16,  "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq16_skv128_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 16,  "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq16_skv256_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 16,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq16_skv1024_f8e4m3": {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 16,  "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq16_skv128_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 16,  "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq16_skv256_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 16,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq16_skv1024_f8e5m2": {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 16,  "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},

    "sq32_skv128_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 32,  "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq32_skv256_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 32,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq32_skv1024_f16":    {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 32,  "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq32_skv128_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 32,  "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq32_skv256_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 32,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq32_skv1024_f8e4m3": {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 32,  "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq32_skv128_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 32,  "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq32_skv256_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 32,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq32_skv1024_f8e5m2": {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 32,  "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},

    "sq64_skv128_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 64,  "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq64_skv256_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 64,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq64_skv1024_f16":    {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 64,  "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq64_skv128_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 64,  "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq64_skv256_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 64,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq64_skv1024_f8e4m3": {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 64,  "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2},
    "sq64_skv128_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 64,  "s_kv": 128,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq64_skv256_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 64,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},
    "sq64_skv1024_f8e5m2": {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 64,  "s_kv": 1024, "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2},

    "sq65_skv256_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 65,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.04, "rtol": 0.1},
    "sq65_skv256_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 65,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},
    "sq65_skv256_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 65,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e5m2", "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.4},
}

TEST_CONFIGS_BWD = {
    "d64_f8e4m3":            {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},
    "d64_f8e4m3_gqa":        {"b": 2, "h_q": 4,  "h_k": 2, "h_v": 2, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},
    "d64_f8e4m3_o-f16":      {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.08, "rtol": 0.2},
    "d64_f8e4m3_o-f16_gqa":  {"b": 2, "h_q": 4,  "h_k": 2, "h_v": 2, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.08, "rtol": 0.2},
    "d128_f8e4m3":           {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},
    "d128_f8e4m3_gqa":       {"b": 2, "h_q": 4,  "h_k": 2, "h_v": 2, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2},
    "d128_f8e4m3_o-f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.08, "rtol": 0.2},
    "d128_f8e4m3_o-f16_gqa": {"b": 2, "h_q": 4,  "h_k": 2, "h_v": 2, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "itype": "fp8_e4m3", "otype": "fp16",     "atol": 0.08, "rtol": 0.2},
}

BLOCKED_CONFIGS_FWD = [
    "sq1_skv1024_f8e5m2", # fails on prefill as well
]

BLOCKED_CONFIGS_BWD = []

def get_torch_and_cudnn_type(type_str):
    if type_str == "fp8_e4m3":
        return torch.float8_e4m3fn, cudnn.data_type.FP8_E4M3
    elif type_str == "fp8_e5m2":
        return torch.float8_e5m2, cudnn.data_type.FP8_E5M2
    elif type_str == "fp16":
        return torch.float16, cudnn.data_type.HALF
    elif type_str == "bf16":
        return torch.bfloat16, cudnn.data_type.BFLOAT16
    else:
        return None, None

def section_begin(msg, width=80):
    print(f" {msg} ".center(width, "="))

def section_end(width=80):
    print("=" * width)

def get_fp8_largest_po2(dtype: torch.dtype):
    if dtype == torch.float8_e4m3fn:
        return 128.0 # max representable value: 0x1.e00000p+7
    elif dtype == torch.float8_e5m2:
        return 32768.0 # max representable value: 0x1.c00000p+15
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def get_fp8_scale_factor(amax: float, dtype: torch.dtype, fudge_factor: float = 0.25, epsilon = 0.0625):
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return 1.0
    po2_next = 2 ** math.ceil(math.log2(max(amax, epsilon)))
    return get_fp8_largest_po2(dtype) / po2_next * fudge_factor

def get_fp8_descale_factor(amax: float, dtype: torch.dtype, fudge_factor: float = 0.25, epsilon = 0.0625):
    return 1.0 / get_fp8_scale_factor(amax, dtype, fudge_factor, epsilon)

class GraphFwdUid(IntEnum):
    q = 0
    k = 1
    v = 2

    q_descale = 5
    k_descale = 6
    v_descale = 7
    s_scale = 9
    s_descale = 8
    o_scale = 10

    o = 3
    stats = 4

    s_amax = 11
    o_amax = 12

    kv_seq_len = 13
    q_seq_len = 14

    k_block_table = 15
    v_block_table = 16

class GraphBwdUid(IntEnum):
    q = 100
    k = 101
    v = 102
    o = 103
    dO = 104
    stats = 105

    q_descale = 106
    k_descale = 107
    v_descale = 108
    o_descale = 109
    dO_descale = 110
    s_descale = 111
    dP_descale = 112
    s_scale = 113
    dQ_scale = 114
    dK_scale = 115
    dV_scale = 116
    dP_scale = 117

    dQ = 118
    dK = 119
    dV = 120

    dQ_amax = 121
    dK_amax = 122
    dV_amax = 123
    dP_amax = 124

def generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size):
    graph_fwd = cudnn.pygraph(io_data_type=cudnn_itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    # Variable sequence lenths are required for paged attention
    use_padding_mask = None
    kv_seq_len = None
    q_seq_len = None
    k_block_table = None
    v_block_table = None

    if block_size == 0:
        q = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=(s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1), data_type=cudnn_itype)
        k = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(b, h_k, s_kv, d_qk), stride=(s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1), data_type=cudnn_itype)
        v = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(b, h_v, s_kv, d_vo), stride=(s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1), data_type=cudnn_itype)
    else:
        table_size = math.ceil(s_kv / block_size)
        num_blocks = table_size * b

        q = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=(s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1), data_type=cudnn_itype)
        k = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(num_blocks, h_k, block_size, d_qk), stride=(block_size * h_k * d_qk, block_size * d_qk, d_qk, 1), data_type=cudnn_itype)
        v = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(num_blocks, h_v, block_size, d_vo), stride=(block_size * h_v * d_vo, block_size * d_vo, d_vo, 1), data_type=cudnn_itype)

        use_padding_mask = True
        kv_seq_len = graph_fwd.tensor(uid=GraphFwdUid.kv_seq_len, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        q_seq_len = graph_fwd.tensor(uid=GraphFwdUid.q_seq_len, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        k_block_table = graph_fwd.tensor(uid=GraphFwdUid.k_block_table, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)
        v_block_table = graph_fwd.tensor(uid=GraphFwdUid.v_block_table, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)

    q_descale = graph_fwd.tensor(uid=GraphFwdUid.q_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    k_descale = graph_fwd.tensor(uid=GraphFwdUid.k_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    v_descale = graph_fwd.tensor(uid=GraphFwdUid.v_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_scale = graph_fwd.tensor(uid=GraphFwdUid.s_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_descale = graph_fwd.tensor(uid=GraphFwdUid.s_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    o_scale = graph_fwd.tensor(uid=GraphFwdUid.o_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    o, stats, amax_s, amax_o = graph_fwd.sdpa_fp8(
        q=q,
        k=k,
        v=v,
        descale_q=q_descale,
        descale_k=k_descale,
        descale_v=v_descale,
        scale_s=s_scale,
        descale_s=s_descale,
        scale_o=o_scale,
        generate_stats=True,
        attn_scale=attn_scale,
        use_causal_mask=False,
        use_padding_mask=use_padding_mask,
        seq_len_kv=kv_seq_len,
        seq_len_q=q_seq_len,
        paged_attention_k_table=k_block_table,  # Block Table K: Tensor containing offsets to the container with K blocks
        paged_attention_v_table=v_block_table,  # Block Table V: Tensor containing offsets to the container with V blocks
        paged_attention_max_seq_len_kv=s_kv,  # The maximum sequence length for K caches (this is optional, but recommended)
    )

    o.set_uid(GraphFwdUid.o).set_output(True).set_dim((b, h_q, s_qo, d_vo)).set_stride((s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1)).set_data_type(cudnn_otype)
    stats.set_uid(GraphFwdUid.stats).set_output(True).set_dim((b, h_q, s_qo, 1)).set_stride((s_qo * h_q, s_qo, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_s.set_uid(GraphFwdUid.s_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_uid(GraphFwdUid.o_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_fwd

def generate_graph_bwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale):
    graph_bwd = cudnn.pygraph(io_data_type=cudnn_itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    q = graph_bwd.tensor(uid=GraphBwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=(s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1), data_type=cudnn_itype)
    k = graph_bwd.tensor(uid=GraphBwdUid.k, dim=(b, h_k, s_kv, d_qk), stride=(s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1), data_type=cudnn_itype)
    v = graph_bwd.tensor(uid=GraphBwdUid.v, dim=(b, h_v, s_kv, d_vo), stride=(s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1), data_type=cudnn_itype)
    o = graph_bwd.tensor(uid=GraphBwdUid.o, dim=(b, h_q, s_qo, d_vo), stride=(s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1), data_type=cudnn_otype)
    dO = graph_bwd.tensor(uid=GraphBwdUid.dO, dim=(b, h_q, s_qo, d_vo), stride=(s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1), data_type=cudnn_itype)
    stats = graph_bwd.tensor(uid=GraphBwdUid.stats, dim=(b, h_q, s_qo, 1), stride=(s_qo * h_q, s_qo, 1, 1), data_type=cudnn.data_type.FLOAT)

    q_descale = graph_bwd.tensor(uid=GraphBwdUid.q_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    k_descale = graph_bwd.tensor(uid=GraphBwdUid.k_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    v_descale = graph_bwd.tensor(uid=GraphBwdUid.v_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    o_descale = graph_bwd.tensor(uid=GraphBwdUid.o_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dO_descale = graph_bwd.tensor(uid=GraphBwdUid.dO_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    s_descale = graph_bwd.tensor(uid=GraphBwdUid.s_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dP_descale = graph_bwd.tensor(uid=GraphBwdUid.dP_descale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    s_scale = graph_bwd.tensor(uid=GraphBwdUid.s_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dQ_scale = graph_bwd.tensor(uid=GraphBwdUid.dQ_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dK_scale = graph_bwd.tensor(uid=GraphBwdUid.dK_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dV_scale = graph_bwd.tensor(uid=GraphBwdUid.dV_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    dP_scale = graph_bwd.tensor(uid=GraphBwdUid.dP_scale, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP = graph_bwd.sdpa_fp8_backward(
        q=q,
        k=k,
        v=v,
        o=o,
        dO=dO,
        stats=stats,
        descale_q=q_descale,
        descale_k=k_descale,
        descale_v=v_descale,
        descale_o=o_descale,
        descale_dO=dO_descale,
        descale_s=s_descale,
        descale_dP=dP_descale,
        scale_s=s_scale,
        scale_dQ=dQ_scale,
        scale_dK=dK_scale,
        scale_dV=dV_scale,
        scale_dP=dP_scale,
        attn_scale=attn_scale,
        use_padding_mask=False,
    )

    dQ.set_uid(GraphBwdUid.dQ).set_output(True).set_dim((b, h_q, s_qo, d_qk)).set_stride((s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1)).set_data_type(cudnn_itype)
    dK.set_uid(GraphBwdUid.dK).set_output(True).set_dim((b, h_k, s_kv, d_qk)).set_stride((s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)).set_data_type(cudnn_itype)
    dV.set_uid(GraphBwdUid.dV).set_output(True).set_dim((b, h_v, s_kv, d_vo)).set_stride((s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1)).set_data_type(cudnn_itype)

    amax_dQ.set_uid(GraphBwdUid.dQ_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dK.set_uid(GraphBwdUid.dK_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dV.set_uid(GraphBwdUid.dV_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_dP.set_uid(GraphBwdUid.dP_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_bwd

def create_paged_container_and_block_table(tensor, block_size):
    B, H, S, D = tensor.shape
    blocks_per_batch = math.ceil(S / block_size)

    padding_seq = blocks_per_batch * block_size - S
    if padding_seq > 0:
        zeros = torch.zeros(B, H, padding_seq, D, device="cuda", dtype=tensor.dtype)
        cat_tensor = torch.cat((tensor, zeros), dim=2)
    else:
        cat_tensor = tensor

    container = torch.cat(cat_tensor.chunk(blocks_per_batch, dim=2), dim=0)

    table_size = math.ceil(S / block_size)
    block_table_temp = torch.linspace(0, B * table_size - 1, B * table_size, device="cuda", dtype=torch.int32).reshape(table_size, 1, B, 1)
    block_table_temp = torch.transpose(block_table_temp, 0, 2)

    block_table = (torch.zeros(blocks_per_batch * B, device="cuda", dtype=torch.int32).as_strided((B, 1, blocks_per_batch, 1), (blocks_per_batch, blocks_per_batch, 1, 1)))
    block_table.copy_(block_table_temp)

    return (container, block_table)

def compute_ref(q, k, v, attn_scale=1.0, return_type="o"):
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, d_v = v.shape

    assert k.shape == (b, s_kv, h_k, d_qk)
    assert v.shape == (b, s_kv, h_v, d_v)

    if h_q != h_k:
        k = k.repeat_interleave(h_q // h_k, dim=2)
    if h_q != h_v:
        v = v.repeat_interleave(h_q // h_v, dim=2)

    s = torch.einsum("bqhd,bkhd->bhqk", q, k) * attn_scale
    p = s.softmax(dim=-1)
    o = torch.einsum("bhqk,bkhd->bqhd", p, v)

    if return_type == "o":
        return o
    if return_type == "o_stats":
        # TODO implement
        return o, torch.zeros()
    elif return_type == "amax":
        return p.abs().max().item(), o.abs().max().item()
    else:
        raise ValueError(f"Unsupported return type: {return_type}")

@pytest.mark.parametrize("name", TEST_CONFIGS_FWD.keys())
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_fwd_fp8(name):
    print()
    section_begin(f"Running {name}")
    config = TEST_CONFIGS_FWD[name]

    if name in BLOCKED_CONFIGS_FWD:
        pytest.skip("TEST WAIVED: blocked config")

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.14.0":
        pytest.skip("TEST WAIVED: SDPA FP8 fprop testing is limited to cuDNN 9.14.0 or higher")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("TEST WAIVED: SDPA FP8 fprop testing is limited to Blackwell or higher")

    torch_itype, cudnn_itype = get_torch_and_cudnn_type(config["itype"])
    torch_otype, cudnn_otype = get_torch_and_cudnn_type(config["otype"])
    assert torch_itype is not None and cudnn_itype is not None
    assert torch_otype is not None and cudnn_otype is not None

    b = config["b"]
    h_q = config["h_q"]
    h_k = config["h_k"]
    h_v = config["h_v"]
    s_qo = config["s_qo"]
    s_kv = config["s_kv"]
    d_qk = config["d_qk"]
    d_vo = config["d_vo"]

    attn_scale = 0.125
    block_size = config.get("kv_block_size", 0)

    is_paged_attention = block_size > 0

    section_begin("Building Graph")
    try:
        graph_fwd = generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size)
        graph_fwd.validate()
        graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_fwd.check_support()
        graph_fwd.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"TEST WAIVED: unsupported graph. {e}")
        pytest.skip("TEST WAIVED: unsupported graph.")
    except Exception as e:
        print(f"Error building graph: {e}")
        pytest.fail(f"Error building graph: {e}")
    section_end()

    section_begin("Allocate and Generate")
    q_gen = torch.clamp(torch.randn(b, s_qo, h_q, d_qk, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)
    k_gen = torch.clamp(torch.randn(b, s_kv, h_k, d_qk, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)
    v_gen = torch.clamp(torch.randn(b, s_kv, h_v, d_vo, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)

    q_amax = q_gen.abs().max().item()
    k_amax = k_gen.abs().max().item()
    v_amax = v_gen.abs().max().item()
    s_amax, o_amax = compute_ref(q_gen, k_gen, v_gen, attn_scale, return_type="amax")

    q_gpu = (q_gen * get_fp8_scale_factor(q_amax, torch_itype)).to(torch_itype)
    k_gpu = (k_gen * get_fp8_scale_factor(k_amax, torch_itype)).to(torch_itype)
    v_gpu = (v_gen * get_fp8_scale_factor(v_amax, torch_itype)).to(torch_itype)

    if is_paged_attention:
        k_gpu_bhsd = torch.einsum('bshd->bhsd', k_gpu).contiguous()
        v_gpu_bhsd = torch.einsum('bshd->bhsd', v_gpu).contiguous()
        container_k_gpu, k_block_table_gpu = create_paged_container_and_block_table(k_gpu_bhsd, block_size)
        container_v_gpu, v_block_table_gpu = create_paged_container_and_block_table(v_gpu_bhsd, block_size)

    kv_seq_len_gpu = torch.full((b, 1, 1, 1), s_kv, device="cuda", dtype=torch.int32)
    q_seq_len_gpu = torch.full((b, 1, 1, 1), s_qo, device="cuda", dtype=torch.int32)
    o_gpu = torch.nans(b, s_qo, h_q, d_vo, dtype=torch_otype, device="cuda")
    stats_gpu = torch.nans(b, h_q, s_qo, 1, dtype=torch.float, device="cuda")

    q_descale_gpu = torch.tensor([get_fp8_descale_factor(q_amax, torch_itype)], dtype=torch.float, device="cuda")
    k_descale_gpu = torch.tensor([get_fp8_descale_factor(k_amax, torch_itype)], dtype=torch.float, device="cuda")
    v_descale_gpu = torch.tensor([get_fp8_descale_factor(v_amax, torch_itype)], dtype=torch.float, device="cuda")
    s_scale_gpu = torch.tensor([get_fp8_scale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")
    s_descale_gpu = torch.tensor([get_fp8_descale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")
    o_scale_gpu = torch.tensor([get_fp8_scale_factor(o_amax, torch_otype)], dtype=torch.float, device="cuda")

    s_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
    o_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
    section_end()

    section_begin("Execute")
    # execute forward and backward graph
    variant_pack = {
        int(GraphFwdUid.q): q_gpu,
        int(GraphFwdUid.k): k_gpu,
        int(GraphFwdUid.v): v_gpu,

        int(GraphFwdUid.q_descale): q_descale_gpu,
        int(GraphFwdUid.k_descale): k_descale_gpu,
        int(GraphFwdUid.v_descale): v_descale_gpu,
        int(GraphFwdUid.s_descale): s_descale_gpu,
        int(GraphFwdUid.s_scale): s_scale_gpu,
        int(GraphFwdUid.o_scale): o_scale_gpu,

        int(GraphFwdUid.o): o_gpu,
        int(GraphFwdUid.stats): stats_gpu,

        int(GraphFwdUid.s_amax): s_amax_gpu,
        int(GraphFwdUid.o_amax): o_amax_gpu,
    }

    if is_paged_attention:
        variant_pack[int(GraphFwdUid.k)] = container_k_gpu
        variant_pack[int(GraphFwdUid.v)] = container_v_gpu
        variant_pack[int(GraphFwdUid.kv_seq_len)] = kv_seq_len_gpu
        variant_pack[int(GraphFwdUid.q_seq_len)] = q_seq_len_gpu
        variant_pack[int(GraphFwdUid.k_block_table)] = k_block_table_gpu
        variant_pack[int(GraphFwdUid.v_block_table)] = v_block_table_gpu

    workspace = torch.empty(graph_fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    cudnn_handle = cudnn.create_handle()
    graph_fwd.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    cudnn.destroy_handle(cudnn_handle)
    section_end()

    section_begin("Run Reference and Compare Output")
    q_ref = q_gpu.detach().float() * get_fp8_descale_factor(q_amax, torch_itype)
    k_ref = k_gpu.detach().float() * get_fp8_descale_factor(k_amax, torch_itype)
    v_ref = v_gpu.detach().float() * get_fp8_descale_factor(v_amax, torch_itype)
    o_ref = compute_ref(q_ref, k_ref, v_ref, attn_scale=attn_scale)

    o_ref_comp = o_ref
    o_gpu_comp = o_gpu.detach().float() * get_fp8_descale_factor(o_amax, torch_otype)

    print("o_ref_comp.numel()", o_ref_comp.numel())
    print("o_gpu_comp.numel()", o_gpu_comp.numel())
    print("Number of zeros in o_ref_comp:", (o_ref_comp == 0).sum().item())
    print("Number of zeros in o_gpu_comp:", (o_gpu_comp == 0).sum().item())
    print("Number of non-finite elements in o_ref_comp:", (~torch.isfinite(o_ref_comp)).sum().item())
    print("Number of non-finite elements in o_gpu_comp:", (~torch.isfinite(o_gpu_comp)).sum().item())

    for _ in range(3):
        coord = tuple(torch.randint(0, numel, (1,)).item() for numel in o_ref_comp.size())
        print(f"o_ref_comp{coord}:", float(o_ref_comp[coord].item()).hex())
        print(f"o_gpu_comp{coord}:", float(o_gpu_comp[coord].item()).hex())

    print(f"s_amax_gpu={s_amax_gpu.item()}, s_amax={s_amax}")
    print(f"o_amax_gpu={o_amax_gpu.item()}, o_amax={o_amax}")

    failed = []
    try:
        torch.testing.assert_close(o_gpu_comp, o_ref_comp, atol=config["atol"], rtol=config["rtol"])
    except Exception as e:
        print("\033[91m" + f"o_gpu: {e}" + "\033[0m\n"); failed.append("o_gpu")
    try:
        torch.testing.assert_close(s_amax_gpu.item(), s_amax, atol=0.04, rtol=0.10)
    except Exception as e:
        print("\033[91m" + f"s_amax_gpu: {e}" + "\033[0m\n"); failed.append("s_amax_gpu")
    try:
        torch.testing.assert_close(o_amax_gpu.item(), o_amax, atol=0.04, rtol=0.10)
    except Exception as e:
        print("\033[91m" + f"o_amax_gpu: {e}" + "\033[0m\n"); failed.append("o_amax_gpu")
    
    if len(failed) > 0:
        print("\033[91m" + "Failed!" + "\033[0m"); pytest.fail(f"Failed: mismatches in {', '.join(failed)}")
    print("\033[92m" + "Passed!" + "\033[0m")

    # # used to debug tolerances
    # x = o_ref_comp.abs()
    # y = o_ref_comp - o_gpu_comp
    # import plotly.express as px
    # import plotly.io as pio
    # fig = px.scatter(
    #     x=x.cpu().flatten().numpy(),
    #     y=y.cpu().flatten().numpy(),
    #     labels={"x": "Absolute value", "y": "Absolute Error"},
    #     title="Absolute value vs absolute error"
    # )
    # pio.write_html(fig, file=f"scatter_{name}.html", auto_open=False)
    # print(f"wrote scatter_{name}.html")

    section_end()
    print()

@pytest.mark.parametrize("name", TEST_CONFIGS_BWD.keys())
@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_bwd_fp8(name):
    print()
    section_begin(f"Running {name} (backward)")
    config = TEST_CONFIGS_BWD[name]

    if name in BLOCKED_CONFIGS_BWD:
        pytest.skip("TEST WAIVED: blocked config")

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.14.0":
        pytest.skip("TEST WAIVED: SDPA FP8 bprop testing is limited to cuDNN 9.14.0 or higher")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("TEST WAIVED: SDPA FP8 bprop testing is limited to Blackwell or higher")

    torch_itype, cudnn_itype = get_torch_and_cudnn_type(config["itype"])
    torch_otype, cudnn_otype = get_torch_and_cudnn_type(config["otype"])
    assert torch_itype is not None and cudnn_itype is not None
    assert torch_otype is not None and cudnn_otype is not None

    b = config["b"]
    h_q = config["h_q"]
    h_k = config["h_k"]
    h_v = config["h_v"]
    s_qo = config["s_qo"]
    s_kv = config["s_kv"]
    d_qk = config["d_qk"]
    d_vo = config["d_vo"]

    attn_scale = 0.125

    section_begin("Build Graphs")
    graph_fwd = generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, 0)
    graph_bwd = generate_graph_bwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale)

    try:
        graph_fwd.validate(); graph_fwd.build_operation_graph(); graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]); graph_fwd.check_support(); graph_fwd.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"TEST WAIVED: unsupported fwd graph. {e}")
        pytest.skip("TEST WAIVED: unsupported fwd graph.")
    try:
        graph_bwd.validate(); graph_bwd.build_operation_graph(); graph_bwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]); graph_bwd.check_support(); graph_bwd.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"TEST WAIVED: unsupported bwd graph. {e}")
        pytest.skip("TEST WAIVED: unsupported bwd graph.")

    section_end()

    section_begin("Allocate and Generate")
    q_gen = torch.clamp(torch.randn(b, s_qo, h_q, d_qk, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)
    k_gen = torch.clamp(torch.randn(b, s_kv, h_k, d_qk, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)
    v_gen = torch.clamp(torch.randn(b, s_kv, h_v, d_vo, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)
    dO_gen = torch.clamp(torch.randn(b, s_qo, h_q, d_vo, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)

    q_amax = q_gen.abs().max().item()
    k_amax = k_gen.abs().max().item()
    v_amax = v_gen.abs().max().item()
    s_amax, o_amax = compute_ref(q_gen, k_gen, v_gen, attn_scale, return_type="amax")
    dO_amax = dO_gen.abs().max().item()

    # q_gpu = (q_gen * get_fp8_scale_factor(q_amax, torch_itype)).to(torch_itype)
    # k_gpu = (k_gen * get_fp8_scale_factor(k_amax, torch_itype)).to(torch_itype)
    # v_gpu = (v_gen * get_fp8_scale_factor(v_amax, torch_itype)).to(torch_itype)
    q_gpu = q_gen.to(torch_itype)
    k_gpu = k_gen.to(torch_itype)
    v_gpu = v_gen.to(torch_itype)

    o_gpu = torch.nans(b, s_qo, h_q, d_vo, dtype=torch_otype, device="cuda")
    stats_gpu = torch.nans(b, h_q, s_qo, 1, dtype=torch.float, device="cuda")

    # dO_gpu = (dO_gen * get_fp8_scale_factor(dO_amax, torch_itype)).to(torch_itype)
    dO_gpu = dO_gen.to(torch_itype)

    q_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
    k_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
    v_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")

    s_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
    s_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
    o_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")

    s_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
    o_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")

    o_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
    dO_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
    dP_descale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")

    dQ_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
    dK_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
    dV_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")
    dP_scale_gpu = torch.tensor([1.0], dtype=torch.float, device="cuda")

    dQ_gpu = torch.nans(b, s_qo, h_q, d_qk, dtype=torch_itype, device="cuda")
    dK_gpu = torch.nans(b, s_kv, h_k, d_qk, dtype=torch_itype, device="cuda")
    dV_gpu = torch.nans(b, s_kv, h_v, d_vo, dtype=torch_itype, device="cuda")

    dQ_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
    dK_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
    dV_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
    dP_amax_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
    section_end()

    section_begin("Execute FWD")
    variant_pack_fwd = {
        int(GraphFwdUid.q): q_gpu,
        int(GraphFwdUid.k): k_gpu,
        int(GraphFwdUid.v): v_gpu,

        int(GraphFwdUid.q_descale): q_descale_gpu,
        int(GraphFwdUid.k_descale): k_descale_gpu,
        int(GraphFwdUid.v_descale): v_descale_gpu,
        int(GraphFwdUid.s_descale): s_descale_gpu,
        int(GraphFwdUid.s_scale): s_scale_gpu,
        int(GraphFwdUid.o_scale): o_scale_gpu,

        int(GraphFwdUid.o): o_gpu,
        int(GraphFwdUid.stats): stats_gpu,

        int(GraphFwdUid.s_amax): s_amax_gpu,
        int(GraphFwdUid.o_amax): o_amax_gpu,
    }

    workspace = torch.empty(graph_fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    cudnn_handle = cudnn.create_handle()
    graph_fwd.execute(variant_pack_fwd, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    cudnn.destroy_handle(cudnn_handle)
    section_end()

    section_begin("Execute BWD")
    variant_pack_bwd = {
        int(GraphBwdUid.q): q_gpu,
        int(GraphBwdUid.k): k_gpu,
        int(GraphBwdUid.v): v_gpu,
        int(GraphBwdUid.o): o_gpu,
        int(GraphBwdUid.dO): dO_gpu,
        int(GraphBwdUid.stats): stats_gpu,

        int(GraphBwdUid.q_descale): q_descale_gpu,
        int(GraphBwdUid.k_descale): k_descale_gpu,
        int(GraphBwdUid.v_descale): v_descale_gpu,
        int(GraphBwdUid.o_descale): o_descale_gpu,
        int(GraphBwdUid.dO_descale): dO_descale_gpu,
        int(GraphBwdUid.s_descale): s_descale_gpu,
        int(GraphBwdUid.s_scale): s_scale_gpu,
        int(GraphBwdUid.dP_descale): dP_descale_gpu,
        int(GraphBwdUid.dP_scale): dP_scale_gpu,
        int(GraphBwdUid.dQ_scale): dQ_scale_gpu,
        int(GraphBwdUid.dK_scale): dK_scale_gpu,
        int(GraphBwdUid.dV_scale): dV_scale_gpu,

        int(GraphBwdUid.dQ): dQ_gpu,
        int(GraphBwdUid.dK): dK_gpu,
        int(GraphBwdUid.dV): dV_gpu,

        int(GraphBwdUid.dQ_amax): dQ_amax_gpu,
        int(GraphBwdUid.dK_amax): dK_amax_gpu,
        int(GraphBwdUid.dV_amax): dV_amax_gpu,
        int(GraphBwdUid.dP_amax): dP_amax_gpu,
    }

    workspace_b = torch.empty(graph_bwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    cudnn_handle = cudnn.create_handle()
    graph_bwd.execute(variant_pack_bwd, workspace_b, handle=cudnn_handle)
    torch.cuda.synchronize()
    cudnn.destroy_handle(cudnn_handle)
    section_end()

    section_begin("Run Reference and Compare Output")
    q_ref = q_gpu.detach().float()
    k_ref = k_gpu.detach().float()
    v_ref = v_gpu.detach().float()
    # q_ref = q_gpu.detach().float() * get_fp8_descale_factor(q_amax, torch_itype)
    # k_ref = k_gpu.detach().float() * get_fp8_descale_factor(k_amax, torch_itype)
    # v_ref = v_gpu.detach().float() * get_fp8_descale_factor(v_amax, torch_itype)
    o_ref = compute_ref(q_ref, k_ref, v_ref, attn_scale=attn_scale)

    dO_ref = dO_gpu.detach().float()
    # dO_ref = dO_gpu.detach().float() * get_fp8_descale_factor(dO_amax, torch_itype)

    q_ref.requires_grad_(True)
    k_ref.requires_grad_(True)
    v_ref.requires_grad_(True)
    o_tmp = compute_ref(q_ref, k_ref, v_ref, attn_scale=attn_scale)
    dQ_ref, dK_ref, dV_ref = torch.autograd.grad(outputs=o_tmp, inputs=[q_ref, k_ref, v_ref], grad_outputs=dO_gen)

    dQ_amax_ref = dQ_ref.abs().max().item()
    dK_amax_ref = dK_ref.abs().max().item()
    dV_amax_ref = dV_ref.abs().max().item()

    dQ_out = dQ_gpu.detach().float()
    dK_out = dK_gpu.detach().float()
    dV_out = dV_gpu.detach().float()
    # dQ_out = dQ_gpu.detach().float() * get_fp8_descale_factor(dQ_amax, torch_itype)
    # dK_out = dK_gpu.detach().float() * get_fp8_descale_factor(dK_amax, torch_itype)
    # dV_out = dV_gpu.detach().float() * get_fp8_descale_factor(dV_amax, torch_itype)

    print("dQ_out.numel()", dQ_out.numel())
    print("dK_out.numel()", dK_out.numel())
    print("dV_out.numel()", dV_out.numel())
    print("Number of zeros in dQ_out:", (dQ_out == 0).sum().item())
    print("Number of zeros in dK_out:", (dK_out == 0).sum().item())
    print("Number of zeros in dV_out:", (dV_out == 0).sum().item())
    print("Number of non-finite elements in dQ_out:", (~torch.isfinite(dQ_out)).sum().item())
    print("Number of non-finite elements in dK_out:", (~torch.isfinite(dK_out)).sum().item())
    print("Number of non-finite elements in dV_out:", (~torch.isfinite(dV_out)).sum().item())

    print()
    for _ in range(3):
        coord_q = tuple(torch.randint(0, numel, (1,)).item() for numel in dQ_out.size())
        coord_k = tuple(torch.randint(0, numel, (1,)).item() for numel in dK_out.size())
        coord_v = tuple(torch.randint(0, numel, (1,)).item() for numel in dV_out.size())
        print(f"dQ_out{coord_q}:", float(dQ_out[coord_q].item()).hex())
        print(f"dQ_ref{coord_q}:", float(dQ_ref[coord_q].item()).hex())
        print(f"dK_out{coord_k}:", float(dK_out[coord_k].item()).hex())
        print(f"dK_ref{coord_k}:", float(dK_ref[coord_k].item()).hex())
        print(f"dV_out{coord_v}:", float(dV_out[coord_v].item()).hex())
        print(f"dV_ref{coord_v}:", float(dV_ref[coord_v].item()).hex())

    print(f"dQ_amax_gpu={dQ_amax_gpu.item()}, dQ_amax_ref={dQ_amax_ref}")
    print(f"dK_amax_gpu={dK_amax_gpu.item()}, dK_amax_ref={dK_amax_ref}")
    print(f"dV_amax_gpu={dV_amax_gpu.item()}, dV_amax_ref={dV_amax_ref}")
    print(f"dP_amax_gpu={dP_amax_gpu.item()}, dP_amax_ref=TODO")

    failed = []
    try:
        torch.testing.assert_close(dQ_out, dQ_ref, atol=config["atol"], rtol=config["rtol"])
    except Exception as e:
        print("\033[91m" + f"dQ: {e}" + "\033[0m\n"); failed.append("dQ")
    try:
        torch.testing.assert_close(dK_out, dK_ref, atol=config["atol"], rtol=config["rtol"])
    except Exception as e:
        print("\033[91m" + f"dK: {e}" + "\033[0m\n"); failed.append("dK")
    try:
        torch.testing.assert_close(dV_out, dV_ref, atol=config["atol"], rtol=config["rtol"])
    except Exception as e:
        print("\033[91m" + f"dV: {e}" + "\033[0m\n"); failed.append("dV")

    # disable amax due to NaNs currently
    try:
        torch.testing.assert_close(dQ_amax_gpu.item(), dQ_amax_ref, atol=0.04, rtol=0.10)
    except Exception as e:
        print("\033[91m" + f"amax_dQ: {e}" + "\033[0m\n"); failed.append("amax_dQ")
    try:
        torch.testing.assert_close(dK_amax_gpu.item(), dK_amax_ref, atol=0.04, rtol=0.10)
    except Exception as e:
        print("\033[91m" + f"amax_dK: {e}" + "\033[0m\n"); failed.append("amax_dK")
    try:
        torch.testing.assert_close(dV_amax_gpu.item(), dV_amax_ref, atol=0.04, rtol=0.10)
    except Exception as e:
        print("\033[91m" + f"amax_dV: {e}" + "\033[0m\n"); failed.append("amax_dV")

    if len(failed) > 0:
        print("\033[91m" + "Failed!" + "\033[0m"); pytest.fail(f"Failed: mismatches in {', '.join(failed)}")
    print("\033[92m" + "Passed!" + "\033[0m")

    section_end()
    print()
