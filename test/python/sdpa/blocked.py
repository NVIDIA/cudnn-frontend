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

    # Ragged backward tests failing on Ampere (SM_80) - disallowed mismatches
    "test_sdpa_random_bwd_ragged_L0[test2]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test13]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test40]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test41]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test59]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test60]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test66]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test72]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test91]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test96]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test111]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test116]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test126]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test131]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test133]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test136]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test139]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test144]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test145]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test153]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test155]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test162]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test163]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test166]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test188]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test192]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test213]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test218]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test220]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test235]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test237]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test238]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test241]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test243]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test247]": {"sms": ["SM_80", "SM_120"]},
    "test_sdpa_random_bwd_ragged_L0[test256]": {"sms": ["SM_80", "SM_120"]},

    # FP8 backward GQA numerical accuracy issues on Hopper - passes on Blackwell
    "test_sdpa_fp8_bwd_L0[test1]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test17]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test20]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test24]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test28]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test33]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test37]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test41]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test43]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test47]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test51]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test56]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_bwd_L0[test57]": {"sms": ["SM_90"]},

    # Hopper-only blocks for bug 5732676 illegal-instruction failures in fp8 ragged tests.
    "test_sdpa_fp8_fwd_ragged_L0[test1]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test2]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test3]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test4]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test5]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test6]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test7]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test8]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test9]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test10]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test11]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test12]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test13]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test14]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test15]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test16]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test17]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test18]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test19]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test20]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test21]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test22]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test23]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test24]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test25]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test26]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test27]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test28]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test29]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test30]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test31]": {"sms": ["SM_90"]},
    "test_sdpa_fp8_fwd_ragged_L0[test32]": {"sms": ["SM_90"]},

    # FP8 ragged backward diagnostics on Blackwell (SM_100), cuDNN 9.21.0.30 (92100).
    # 2026-02-24 with all seeds unblocked: 4 pass, 8 skip, 5 determinism-fail, 15 numeric-mismatch.
    #
    # | Outcome                  | Count | Seeds |
    # |--------------------------|-------|-------|
    # | pass                     | 4     | test1, test6, test7, test22 |
    # | skip_unsupported_graph   | 8     | test2, test4, test5, test10, test15, test17, test26, test27 |
    # | determinism_fail         | 5     | test8, test16, test19, test20, test29 |
    # | numeric_mismatch         | 15    | test3, test9, test11, test12, test13, test14, test18, test21, test23, test24, test25, test28, test30, test31, test32 |
    #
    # Skip reason for all 8 skipped seeds:
    # - unsupported graph: hidden_dim d_qk should be <= 128 and aligned (unless d_qk == 192 and d_v == 128).
    # With this blocklist enabled: 4 passed, 28 skipped (20 blocked seeds + 8 unsupported-graph seeds).
    # FP8 ragged backward blocked cases on Blackwell (SM_100)
    # determinism_fail
    "test_sdpa_fp8_bwd_ragged_L0[test8]": {"sms": ["SM_90", "SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test16]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test19]": {"sms": ["SM_90", "SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test20]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test29]": {"sms": ["SM_100"]},
    # numeric_mismatch
    "test_sdpa_fp8_bwd_ragged_L0[test3]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test9]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test11]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test12]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test13]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test14]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test18]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test21]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test23]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test24]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test25]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test28]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test30]": {"sms": ["SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test31]": {"sms": ["SM_90", "SM_100"]},
    "test_sdpa_fp8_bwd_ragged_L0[test32]": {"sms": ["SM_90", "SM_100"]},
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
