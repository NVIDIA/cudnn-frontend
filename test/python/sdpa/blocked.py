# Blocked tests configuration
# Format: "test_name": {"sms": ["SM_90", "SM_100"], "cudnn_versions": ["91100"]}
# - sms: List of GPU architectures to block on (e.g., "SM_90", "SM_100")
# - cudnn_versions: List of cuDNN versions to block on (e.g., "91100")
# If a field is None or missing, the test is blocked on all values for that field.

# fmt: off

BLOCKED_TESTS = {
    # Currently empty - add blocked tests as needed
    # Example entries:
    # "test_sdpa_random_bwd[test64]": {"sms": ["SM_90", "SM_100"], "cudnn_versions": ["91100"]},
    # "test_sdpa_random_bwd[test65]": {"sms": ["SM_100"], "cudnn_versions": ["91100", "91000"]},
    # "test_sdpa_random_bwd[test66]": {"sms": ["SM_80"]},
    # "test_sdpa_random_bwd[test67]": {"cudnn_versions": ["90000"]},
    # "test_sdpa_random_bwd[test68]": {},

    # FP8 forward edge cases producing NaN - blocked until investigated
    # Original test_sdpa_fp8.py only tested: h_q=h_k=h_v=4, s_kv=256/1024, d_qk=64/128/192, d_v=64/128
    #
    # | Test    | s_q | s_kv | h_q | h_k | d_qk | d_v | dtype   | otype  | Issue                         |
    # |---------|-----|------|-----|-----|------|-----|---------|--------|-------------------------------|
    # | test14  |  89 |  569 |   8 |   1 |  128 | 128 | e5m2    | fp16   | e5m2+GQA+non-aligned s_q      |
    # | test17  | 207 |  207 |   9 |   9 |  120 | 120 | e4m3    | e4m3   | d_qk=120 not multiple of 16   |
    # | test18  | 766 |  766 |  13 |   1 |  192 | 128 | e5m2    | fp16   | e5m2+d_qk=192+GQA             |
    # | test21  |   1 |  936 |  10 |   5 |   64 |  64 | e4m3    | e5m2   | s_q=1 + GQA + mixed fp8 out   |
    # | test40  |   1 |  552 |   3 |   3 |   64 |  64 | e4m3    | e4m3   | s_q=1 + MHA                   |
    # | test41  |   1 |  225 |  11 |  11 |   64 |  64 | e4m3    | fp16   | s_q=1 + MHA                   |
    # | test42  | 896 |  896 |  13 |  13 |  192 | 128 | e4m3    | fp16   | d_qk=192 + large MHA          |
    # | test57  |   1 |  949 |   8 |   8 |   64 |  64 | e5m2    | e4m3   | s_q=1 + MHA + mixed fp8 out   |
    # | test64  |   1 |  489 |   9 |   1 |   64 |  64 | e5m2    | fp16   | s_q=1 + GQA + e5m2            |
    # | test73  |   1 |  321 |   9 |   1 |   64 |  64 | e4m3    | fp16   | s_q=1 + GQA                   |
    # | test86  |   1 |  375 |   8 |   2 |   64 |  64 | e5m2    | fp16   | s_q=1 + GQA + e5m2            |
    # | test90  |   1 |  213 |  12 |   3 |   64 |  64 | e4m3    | fp16   | s_q=1 + GQA                   |
    # | test96  |   1 |  132 |  13 |   1 |   64 |  64 | e4m3    | fp16   | s_q=1 + GQA                   |
    # | test128 |   1 |  707 |  10 |   1 |   64 |  64 | e4m3    | e5m2   | s_q=1 + GQA + mixed fp8 out   |
    "test_sdpa_fp8_fwd_L0[test14]": {},
    "test_sdpa_fp8_fwd_L0[test17]": {},
    "test_sdpa_fp8_fwd_L0[test18]": {},
    "test_sdpa_fp8_fwd_L0[test21]": {},
    "test_sdpa_fp8_fwd_L0[test40]": {},
    "test_sdpa_fp8_fwd_L0[test41]": {},
    "test_sdpa_fp8_fwd_L0[test42]": {},
    "test_sdpa_fp8_fwd_L0[test57]": {},
    "test_sdpa_fp8_fwd_L0[test64]": {},
    "test_sdpa_fp8_fwd_L0[test73]": {},
    "test_sdpa_fp8_fwd_L0[test86]": {},
    "test_sdpa_fp8_fwd_L0[test90]": {},
    "test_sdpa_fp8_fwd_L0[test96]": {},
    "test_sdpa_fp8_fwd_L0[test128]": {},

    # Ragged backward tests failing on Ampere (SM_80) - disallowed mismatches
    "test_sdpa_random_bwd_ragged_L0[test2]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test13]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test40]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test41]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test59]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test60]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test66]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test72]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test91]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test96]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test111]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test116]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test126]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test131]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test133]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test136]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test139]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test144]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test145]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test153]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test155]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test162]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test163]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test166]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test188]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test192]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test213]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test218]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test220]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test235]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test237]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test238]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test241]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test243]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test247]": {"sms": ["SM_80"]},
    "test_sdpa_random_bwd_ragged_L0[test256]": {"sms": ["SM_80"]},
}


def show_blocked_tests(blocked_tests, gpu_arch, cudnn_ver):
    print(f"\n\nBlocked tests on {gpu_arch} and cudnn_ver={cudnn_ver}:")
    if blocked_tests:
        for index, test in enumerate(blocked_tests):
            print(f"{index+1:<4} : {test}")
    else:
        print("[empty]")

def fetch_blocked_tests(gpu_arch, cudnn_ver):
    """
    Returns a list of test names that should be blocked for the given GPU architecture
    and cuDNN version.
    """
    assert type(gpu_arch) == type(cudnn_ver) == str, "expecting strings"
    blocked_tests = []
    for test, config in BLOCKED_TESTS.items():
        sms = config.get("sms")
        libs = config.get("cudnn_versions")
        if (test not in blocked_tests) and (sms is None or gpu_arch in sms) and (libs is None or cudnn_ver in libs):
            blocked_tests.append(test)
    return blocked_tests
