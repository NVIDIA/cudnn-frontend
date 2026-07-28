import pytest
import torch

from test_utils import torch_fork_set_rng

from fe_api.dsa.dsa_utils import dsa_init, with_dsa_indexer_top_k_params
from fe_api.dsa.dsa_reference import check_ref_indexer_top_k


def _allocate_inputs(cfg, next_n: int):
    """Allocate inputs with the kernel's ``n_rows == batch_size * next_n``
    invariant held: treat every row as its own batch for ``next_n=1``,
    otherwise group ``next_n`` consecutive rows per batch.
    """
    b = cfg["b"]
    s_kv = cfg["s_kv"]
    s_q = cfg["s_q"]
    dtype = cfg["dtype"]
    n_rows = b * s_q
    device = "cuda"

    assert n_rows % next_n == 0, f"n_rows={n_rows} must be divisible by next_n={next_n}"
    batch_size = n_rows // next_n

    input_values = torch.randn(n_rows, s_kv, dtype=dtype, device=device)
    # Random-but-reasonable seq_lens (each in [s_kv // 2, s_kv]).
    lo = max(1, s_kv // 2)
    seq_lens = torch.randint(
        lo,
        s_kv + 1,
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    return input_values, seq_lens


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_top_k_params
def test_DSA_indexer_top_k_compile_execute(
    dtype,
    acc_dtype,
    top_k,
    next_n,
    return_val,
    request,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        top_k=top_k,
        next_n=next_n,
        return_val=return_val,
        min_compute_capability=90,
    )
    input_values, seq_lens = _allocate_inputs(cfg, next_n=next_n)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        op = DSA.IndexerTopK(
            sample_input_values=input_values,
            sample_seq_lens=seq_lens,
            top_k=top_k,
            next_n=next_n,
            return_val=return_val,
        )
        assert op.check_support()
        op.compile()
        indices, values = op.execute(input_values, seq_lens, current_stream=stream)
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    if not cfg["skip_ref"]:
        check_ref_indexer_top_k(
            input_values,
            seq_lens,
            top_k,
            next_n,
            indices,
            values,
            return_val,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_indexer_top_k_params
def test_DSA_indexer_top_k_wrapper(
    dtype,
    acc_dtype,
    top_k,
    next_n,
    return_val,
    request,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        top_k=top_k,
        next_n=next_n,
        return_val=return_val,
        min_compute_capability=90,
    )
    input_values, seq_lens = _allocate_inputs(cfg, next_n=next_n)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    try:
        result = DSA.indexer_top_k_wrapper(
            input_values,
            seq_lens,
            top_k,
            next_n=next_n,
            return_val=return_val,
            stream=stream,
        )
    except (ValueError, NotImplementedError) as e:
        pytest.skip(f"Unsupported testcase: {e}")

    indices = result["indices"]
    values = result["values"]
    if not cfg["skip_ref"]:
        check_ref_indexer_top_k(
            input_values,
            seq_lens,
            top_k,
            next_n,
            indices,
            values,
            return_val,
        )


@pytest.mark.L0
@pytest.mark.parametrize("trigger_dist", ["identical", "distinct"])
def test_DSA_indexer_top_k_oob_tile_lanes(trigger_dist):
    """Regression test: -inf-filled OOB lanes of a row's final vector tile must
    not participate in the radix top-k.

    The kernel reads each row in fixed-width vector tiles and fills the final
    partial tile's out-of-bounds lanes with -inf. Those phantom lanes were
    counted as real elements by the histogram and candidate-collection passes.
    When a row's top-k threshold lands in the fp16 -inf coarse bin (fewer than
    top_k values above ~-65504, since to_coarse_key converts fp32 scores to
    fp16 first), the phantom lanes flood the threshold bin's candidate list and
    overflow the per-row candidate buffers in the large-occupancy compile
    (>148 rows), producing out-of-bounds shared/global writes
    (cudaErrorIllegalAddress) or silently out-of-range selected indices.

    This test's batch deterministically places row 0 in that regime: 149 rows
    (>148 -> large_occupancy), a 4310-wide matrix, and a 4122-long row whose
    values below the fp16 negative-overflow point plus the phantom lanes
    outnumber the candidate-buffer capacity.
    """
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("indexer top-k kernel requires SM90+")

    num_rows, num_cols, top_k, trigger_len, num_above = 149, 4310, 2048, 4122, 1122
    scores = torch.empty(num_rows, num_cols, dtype=torch.float32)
    scores[1:] = torch.arange(num_cols, dtype=torch.float32).expand(num_rows - 1, -1)
    row0 = torch.empty(num_cols, dtype=torch.float32)
    row0[:num_above] = torch.arange(num_above, dtype=torch.float32)
    if trigger_dist == "identical":
        row0[num_above:trigger_len] = -100000.0
    else:
        row0[num_above:trigger_len] = torch.linspace(-66000.0, -200000.0, trigger_len - num_above)
    row0[trigger_len:] = float("-inf")
    scores[0] = row0
    seq_lens = torch.full((num_rows,), num_cols, dtype=torch.int32)
    seq_lens[0] = trigger_len

    input_values = scores.cuda()
    seq_lens = seq_lens.cuda()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    result = DSA.indexer_top_k_wrapper(
        input_values,
        seq_lens,
        top_k,
        next_n=1,
        return_val=True,
        stream=stream,
    )
    torch.cuda.synchronize()

    indices = result["indices"].cpu()
    values = result["values"].cpu()
    for r in range(num_rows):
        L = int(seq_lens[r].cpu())
        k = min(top_k, L)
        valid = indices[r] >= 0
        assert int(valid.sum()) == k, f"row {r}: expected {k} valid indices, got {int(valid.sum())}"
        assert int(indices[r].max()) < L, f"row {r}: out-of-range index {int(indices[r].max())} >= seq_len {L}"
        got = values[r][valid].sort(descending=True).values
        ref = torch.topk(scores[r, :L], k).values.sort(descending=True).values
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-5)
