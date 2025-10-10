import pytest
import cudnn
import torch
import argparse

# fmt: off

# =================== Fixtures =====================
@pytest.fixture(scope="session", autouse=True)
def cudnn_handle():
    # Create CUDA stream and graph objects
    stream = torch.cuda.Stream()
    cudnn_handle = cudnn.create_handle()
    cudnn.set_stream(handle=cudnn_handle, stream=stream.cuda_stream)
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
        "--dryrun", action="store", nargs="?", const=1, type=int, default=0, help="show repro commands when 1, 2, or 3 (use with '-s')",
    )
    parser.addoption(
        "--diffs", action="store", type=int, default=10, help="set number of numerical mismatches to display",
    )
    parser.addoption(
        "--repro", action="store", type=str, default=None, help="specify config string to run repro function",
    )


    # MHA command line options to overwrite specific test dimensions in test_mhas.py and test_mhas_v2.py.
    parser.addoption(
        "--b", default=None, type=int, help="[test_mhas.py] batch dimension"
    )
    parser.addoption(
        "--s_q", default=None, type=int, help="[test_mhas.py] query sequence length"
    )
    parser.addoption(
        "--s_kv", default=None, type=int, help="[test_mhas.py] key/value sequence length"
    )
    parser.addoption(
        "--d_qk", default=None, type=int, help="[test_mhas.py] query/key embedding dimension per head",
    )
    parser.addoption(
        "--d_v", default=None, type=int, help="[test_mhas.py] value embedding dimension per head",
    )
    parser.addoption(
        "--h_q", default=None, type=int, help="[test_mhas.py] query number of heads"
    )
    parser.addoption(
        "--h_k", default=None, type=int, help="[test_mhas.py] key number of heads"
    )
    parser.addoption(
        "--h_v", default=None, type=int, help="[test_mhas.py] value number of heads"
    )
    parser.addoption(
        "--deterministic", default=None, type=int, choices=[0, 1], help="[test_mhas.py] force deterministic algorithm",
    )
    parser.addoption(
        "--block_size", default=None, type=int, help="[test_mhas.py] block size for paged attention",
    )
    parser.addoption(
        "--left_bound", default=None, type=int, help="[test_mhas.py] size of the window to the left of the diagonal",
    )
    parser.addoption(
        "--right_bound", default=None, type=int, help="[test_mhas.py] size of the window to the right of the diagonal",
    )

    parser.addoption(
        "--implementation", action="store", default=None, type=str, choices=["AUTO", "COMPOSITE", "UNIFIED"], help="[test_mhas_v2.py], overwrites implementation",
    )
