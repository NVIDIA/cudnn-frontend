import pytest
import cudnn
import torch
import argparse

# fmt: off

# =================== Fixtures =====================
@pytest.fixture(scope="session", autouse=True)
def cudnn_handle():
    try:
        _ = cudnn.backend_version()
    except Exception:
        # cuDNN not available; do not create a handle so tests not requiring it can run
        yield None
        return
    
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
    try:
        print(f"cuDNN Backend Version: {cudnn.backend_version()}")
    except Exception as e:
        print(f"cuDNN Backend not available: {e}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Path: {torch.__file__}")
    print(f"PyTorch GPU Name: {torch.cuda.get_device_name()}")
    print(f"PyTorch SM Arch Version: {torch.cuda.get_device_capability()}")
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"PyTorch cuDNN Version: {torch.backends.cudnn.version()}")

# fmt: off
def pytest_addoption(parser):
    # Generic options that may be used by all scripts.
    parser.addoption("--dryrun", action="store", nargs="?", const=1, type=int, default=0, help="show repro commands when 1, 2, or 3 (use with '-s')")
    parser.addoption("--diffs", action="store", type=int, default=10, help="set number of numerical mismatches to display")
    parser.addoption("--repro", action="store", type=str, default=None, help="specify config string to run repro function")
    parser.addoption("--perf", action="store_true", help="enable performance profiling")

    # MHA command line options to overwrite specific test dimensions in test_mhas.py and test_mhas_v2.py.
    parser.addoption("--b", default=None, type=int, help="[test_mhas.py] batch dimension")
    parser.addoption("--s_q", default=None, type=int, help="[test_mhas.py] query sequence length")
    parser.addoption("--s_kv", default=None, type=int, help="[test_mhas.py] key/value sequence length")
    parser.addoption("--d_qk", default=None, type=int, help="[test_mhas.py] query/key embedding dimension per head")
    parser.addoption("--d_v", default=None, type=int, help="[test_mhas.py] value embedding dimension per head")
    parser.addoption("--h_q", default=None, type=int, help="[test_mhas.py] query number of heads")
    parser.addoption("--h_k", default=None, type=int, help="[test_mhas.py] key number of heads")
    parser.addoption("--h_v", default=None, type=int, help="[test_mhas.py] value number of heads")
    parser.addoption("--deterministic", default=None, type=int, choices=[0, 1], help="[test_mhas.py] force deterministic algorithm")
    parser.addoption("--block_size", default=None, type=int, help="[test_mhas.py] block size for paged attention")
    parser.addoption("--left_bound", default=None, type=int, help="[test_mhas.py] size of the window to the left of the diagonal")
    parser.addoption("--right_bound", default=None, type=int, help="[test_mhas.py] size of the window to the right of the diagonal")

    parser.addoption("--implementation", action="store", default=None, type=str, choices=["AUTO", "COMPOSITE", "UNIFIED"], help="[test_mhas_v2.py], overwrites implementation")

    # NSA (Native Sparse Attention) command line options for test_NSA_selection_attention.py, test_NSA_swa.py
    parser.addoption("--nsa-batch-size", action="store", default=None, type=int, help="[test_NSA_selection_attention.py, test_NSA_swa.py] Batch size")
    parser.addoption("--nsa-seq-len", action="store", default=None, type=int, help="[test_NSA_selection_attention.py, test_NSA_swa.py] Sequence length (will be replicated for all batches)")
    parser.addoption("--nsa-num-q-heads", action="store", default=None, type=int, help="[test_NSA_selection_attention.py, test_NSA_swa.py] Number of query heads")
    parser.addoption("--nsa-num-kv-heads", action="store", default=None, type=int, help="[test_NSA_selection_attention.py, test_NSA_swa.py] Number of key/value heads")
    parser.addoption("--nsa-head-dim", action="store", default=None, type=int, help="[test_NSA_selection_attention.py, test_NSA_swa.py] Head dimension")
    parser.addoption("--nsa-value-dim", action="store", default=None, type=int, help="[test_NSA_selection_attention.py, test_NSA_swa.py] Value dimension")
    parser.addoption("--nsa-dtype", action="store", default=None, type=str, help="[test_NSA_selection_attention.py, test_NSA_swa.py] Data type (float16, bfloat16, float32)")
    parser.addoption("--nsa-acc-dtype", action="store", default=None, type=str, help="[test_NSA_selection_attention.py, test_NSA_swa.py] Accumulator data type (float16, bfloat16, float32)")
    parser.addoption("--nsa-skip-ref", action="store_true", help="[test_NSA_selection_attention.py, test_NSA_swa.py] Skip reference computation for performance testing")
    parser.addoption("--nsa-block-size", action="store", default=None, type=int, help="[test_NSA_selection_attention.py] Block size")
    parser.addoption("--nsa-topk-size", action="store", default=None, type=int, help="[test_NSA_selection_attention.py] Top-k size (will be replicated for all batches)")
    parser.addoption("--nsa-window-size", action="store", default=None, type=int, help="[test_NSA_swa.py] Window size")
    parser.addoption("--nsa-layout", action="store", default=None, type=str, help="[test_NSA_swa.py] Layout (bshd, thd)")

    # GEMM SwiGLU command line options for test_gemm_swiglu.py
    parser.addoption("--gemm-swiglu-mnkl", action="store", default=None, type=str, help="[test_gemm_swiglu.py] M,N,K,L dimensions as comma-separated values (e.g., '256,256,512,1')")
    
    parser.addoption("--gemm-swiglu-mma-tiler", action="store", default=None, type=str, help="[test_gemm_swiglu.py] MMA tiler (M,N) dimensions as comma-separated values (e.g., '128,128')")
    parser.addoption("--gemm-swiglu-cluster-shape", action="store", default=None, type=str, help="[test_gemm_swiglu.py] Cluster shape (M,N) dimensions as comma-separated values (e.g., '1,1')")
    parser.addoption("--gemm-swiglu-alpha", action="store", default=None, type=float, help="[test_gemm_swiglu.py] Alpha scaling factor")
    parser.addoption("--gemm-swiglu-skip-ref", action="store_true", help="[test_gemm_swiglu.py] Skip reference computation for performance testing")

    # GEMM Amax command line options for test_gemm_amax.py
    parser.addoption("--gemm-amax-mnkl", action="store", default=None, type=str, help="[test_gemm_amax.py] M,N,K,L dimensions as comma-separated values (e.g., '512,256,256,1')")
    parser.addoption("--gemm-amax-mma-tiler", action="store", default=None, type=str, help="[test_gemm_amax.py] MMA tiler (M,N) dimensions as comma-separated values (e.g., '128,128')")
    parser.addoption("--gemm-amax-cluster-shape", action="store", default=None, type=str, help="[test_gemm_amax.py] Cluster shape (M,N) dimensions as comma-separated values (e.g., '1,1')")
    parser.addoption("--gemm-amax-skip-ref", action="store_true", help="[test_gemm_amax.py] Skip reference computation for performance testing")
# fmt: on
