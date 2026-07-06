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
