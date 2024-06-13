import pytest


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
