import pytest
import cudnn
import torch


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
    parser.addoption("--mha_b", default=None, help="[test_mhas.py] batch dimension")
    parser.addoption(
        "--mha_s_q", default=None, help="[test_mhas.py] query sequence length"
    )
    parser.addoption(
        "--mha_s_kv", default=None, help="[test_mhas.py] key/value sequence length"
    )
    parser.addoption(
        "--mha_d_qk",
        default=None,
        help="[test_mhas.py] query/key embedding dimension per head",
    )
    parser.addoption(
        "--mha_d_v",
        default=None,
        help="[test_mhas.py] value embedding dimension per head",
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
        "--mha_deterministic",
        default=None,
        help="[test_mhas.py] force deterministic algorithm",
    )
    parser.addoption(
        "--mha_block_size",
        default=None,
        help="[test_mhas.py] block size for paged attention",
    )
    parser.addoption(
        "--mha_left_bound",
        default=None,
        help="[test_mhas.py] size of the window to the left of the diagonal",
    )
    parser.addoption(
        "--mha_right_bound",
        default=None,
        help="[test_mhas.py] size of the window to the right of the diagonal",
    )
