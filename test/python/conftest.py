import pytest
import cudnn
import torch

# fmt: off

# =================== Fixtures =====================
@pytest.fixture(scope="session", autouse=True)
def cudnn_handle():
    cudnn_handle = cudnn.create_handle()
    yield cudnn_handle
    cudnn.destroy_handle(cudnn_handle)


# =================== PyTest Hooks =====================
def pytest_load_initial_conftests(args, early_config, parser):
    if not any(arg.startswith("--tb=") for arg in args):
        args.append("--tb=short")


def pytest_configure(config):
    assert torch.cuda.is_available()

    print("===== cudnn-frontend conftest.py ====")
    print(f"cuDNN Frontend Version: {cudnn.__version__}")
    print(f"cuDNN Frontend Path: {cudnn.__file__}")
    print(f"cuDNN Backend Version: {cudnn.backend_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Path: {torch.__file__}")
    print(f"PyTorch GPU Name: {torch.cuda.get_device_name()}")
    print(f"PyTorch SM Arch Version: {torch.cuda.get_device_capability()}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"PyTorch cuDNN Version: {torch.backends.cudnn.version()}")


def pytest_addoption(parser):
    # Generic options that may be used by all scripts.
    parser.addoption(
        "--dryrun", action="store", nargs="?", const=1, type=int, default=0, help="show repro commands when 1 or 2 (use with '-s')",
    )
    parser.addoption(
        "--diffs", action="store", type=int, default=10, help="set number of numerical mismatches to display",
    )
    parser.addoption(
        "--repro", action="store", type=str, default=None, help="specify config string to run repro function",
    )
    parser.addoption(
        "--unlock", action="store_true", default=False, help="run 'flaky' tests, normally skipped"
    )
    parser.addoption(
        "--geom_seed", type=int, default=None, help="update seed of RNG generating task dimensions (geometries)",
    )
    parser.addoption(
        "--data_seed", type=int, default=None, help="update seed of RNG initializing task input data",
    )

    # MHA command line options to overwrite specific test dimensions in test_mhas.py and test_mhas_v2.py.
    parser.addoption(
        "--mha_b", default=None, help="[test_mhas.py] batch dimension"
    )
    parser.addoption(
        "--mha_s_q", default=None, help="[test_mhas.py] query sequence length"
    )
    parser.addoption(
        "--mha_s_kv", default=None, help="[test_mhas.py] key/value sequence length"
    )
    parser.addoption(
        "--mha_d_qk", default=None, help="[test_mhas.py] query/key embedding dimension per head",
    )
    parser.addoption(
        "--mha_d_v", default=None, help="[test_mhas.py] value embedding dimension per head",
    )
    parser.addoption(
        "--mha_h_q", default=None, help="[test_mhas.py] query number of heads"
    )
    parser.addoption(
        "--mha_h_k", default=None, help="[test_mhas.py] key number of heads"
    )
    parser.addoption(
        "--mha_h_v", default=None, help="[test_mhas.py] value number of heads"
    )
    parser.addoption(
        "--mha_deterministic", default=None, help="[test_mhas.py] force deterministic algorithm",
    )
    parser.addoption(
        "--mha_block_size", default=None, help="[test_mhas.py] block size for paged attention",
    )
    parser.addoption(
        "--mha_left_bound", default=None, help="[test_mhas.py] size of the window to the left of the diagonal",
    )
    parser.addoption(
        "--mha_right_bound", default=None, help="[test_mhas.py] size of the window to the right of the diagonal",
    )

    # MHA command line options to overwrite boolean 'is*' variables in test_mhas_v2.py.
    parser.addoption(
        "--mha_is_infer", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_infer",
    )
    parser.addoption(
        "--mha_is_causal", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_causal",
    )
    parser.addoption(
        "--mha_is_alibi", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_alibi",
    )
    parser.addoption(
        "--mha_is_paged", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_paged",
    )
    parser.addoption(
        "--mha_is_bias", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_bias",
    )
    parser.addoption(
        "--mha_is_padding", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_padding",
    )
    parser.addoption(
        "--mha_is_ragged", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_ragged",
    )
    parser.addoption(
        "--mha_is_causal_br", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_causal_br",
    )
    parser.addoption(
        "--mha_is_sliding_w", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_sliding_w",
    )
    parser.addoption(
        "--mha_is_dropout", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_dropout",
    )
    parser.addoption(
        "--mha_is_determin", action="store", default=None, type=str, choices=["True", "False"], help="[test_mhas_v2.py], overwrites is_determin",
    )

    # Refined command line option in test_mhas_v2.py to supersede --mha_b (the 'b' variable is too short).
    parser.addoption(
        "--mha_batches", default=None, help="[test_mhas_v2.py] update batch dimension"
    )
