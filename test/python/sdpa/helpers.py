import cudnn
import torch
import math

# fmt: off

def fill_sparse_small_int(tensor, rng, sparsity=0.8, abs_max=2):
    """
    Fill tensor with sparse small integers for better low-precision testing.

    Using sparse integers instead of uniform/normal distributions:
    - Reduces averaging effect in large multiply-add operations
    - Uses exactly representable values in FP8/FP16/BF16
    - Makes numerical errors easier to diagnose

    Args:
        tensor: Tensor to fill (modified in-place)
        rng: torch.Generator for deterministic pseudo-random generation
        sparsity: Fraction of zeros (0.8 = 80% zeros). Range [0, 1).
        abs_max: Maximum absolute value. Fills with integers in [-abs_max, abs_max].

    Returns:
        The filled tensor (same object, modified in-place)
    """
    assert 0 <= sparsity < 1, f"sparsity must be in [0, 1), got {sparsity}"
    assert abs_max >= 1, f"abs_max must be >= 1, got {abs_max}"

    # Fill with random integers from [-abs_max, abs_max] inclusive
    # random_ is [low, high) so we use abs_max + 1
    tensor.random_(-abs_max, abs_max + 1, generator=rng)

    # Zero out sparsity fraction
    if sparsity > 0:
        mask = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        mask.uniform_(generator=rng)
        tensor[mask < sparsity] = 0

    return tensor

def create_sparse_int_tensor(shape, dtype, rng, *, device='cuda', sparsity=0.8, abs_max=2, memory_format=None):
    """
    Create a tensor filled with sparse small integers.

    Args:
        shape: Tensor shape (tuple or list)
        dtype: PyTorch dtype
        rng: torch.Generator for deterministic pseudo-random generation
        device: Device to create tensor on (default 'cuda')
        sparsity: Fraction of zeros (0.8 = 80% zeros). Range [0, 1).
        abs_max: Maximum absolute value. Fills with integers in [-abs_max, abs_max].
        memory_format: Optional memory format (e.g., torch.channels_last)

    Returns:
        Tensor filled with sparse small integers
    """
    tensor = torch.empty(shape, device=device, dtype=dtype)
    if memory_format is not None:
        tensor = tensor.to(memory_format=memory_format)
    fill_sparse_small_int(tensor, rng, sparsity=sparsity, abs_max=abs_max)
    return tensor

def print_tensor_stats(tensor, tag=None):
    """Print hash and statistics of a tensor's contents.

    Prints hash (for determinism verification) and element statistics
    (zeros, NaNs, +Inf, -Inf counts).

    Args:
        tensor: PyTorch tensor (can be on GPU or CPU)
        tag: Optional name/tag for the tensor (used in print output)

    Returns:
        Hash value as uint64
    """
    if tensor is None:
        return None

    t = tensor.contiguous()
    numel = t.numel()

    # Compute hash using torch.hash_tensor (fast GPU operation)
    # FP8 types not supported by hash_tensor, view as int8
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        hash_value = torch.hash_tensor(t.view(torch.int8))
    else:
        hash_value = torch.hash_tensor(t)

    # Compute statistics (all GPU operations)
    num_zeros = numel - torch.count_nonzero(t).item()
    if torch.is_floating_point(t):
        num_nans = torch.isnan(t).sum().item()
        num_pos_inf = torch.eq(t, float('inf')).sum().item()
        num_neg_inf = torch.eq(t, float('-inf')).sum().item()
    else:
        num_nans = num_pos_inf = num_neg_inf = 0

    if tag is not None:
        def fmt(count):
            pct = 100 * count / numel if numel > 0 else 0
            return f"{count:,} ({pct:.0f}%)" if count > 0 else "0"
        num_infs = num_pos_inf + num_neg_inf
        stats = f"elems={numel:,}, zeros={fmt(num_zeros)}, NaNs={fmt(num_nans)}, Infs={fmt(num_infs)}"
        print(f"%%%% {tag}: hash=0x{hash_value.item() >> 32:08X}, {stats}")

    return hash_value.item()

def compare_tensors(actual: torch.Tensor, expected: torch.Tensor,
                    rtol: float = 1e-2, atol: float = 1e-2,
                    num_diffs: int = 10):
    """Compare two tensors and return detailed comparison results.

    Args:
        actual: The tensor to check (e.g., cuDNN output)
        expected: The reference tensor (e.g., PyTorch reference)
        rtol: Relative tolerance
        atol: Absolute tolerance
        num_diffs: Number of mismatches to show in detail

    Returns:
        Tuple of (passed: bool, num_mismatches: int, message: str)
        - passed: True if all elements are within tolerance
        - num_mismatches: Number of elements that failed tolerance check
        - message: Detailed comparison info including max_diff, mean_diff, etc.
    """
    if expected.shape != actual.shape:
        return False, -1, f"Shape mismatch: actual={actual.shape}, expected={expected.shape}"

    # Convert to float32 for comparison
    actual_f = actual.to(torch.float32).contiguous()
    expected_f = expected.to(torch.float32).contiguous()

    # Compute differences
    diff = torch.abs(actual_f - expected_f)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative difference (avoid division by zero)
    denom = torch.maximum(torch.abs(expected_f), torch.tensor(1e-6, device=actual.device))
    rel_diff = diff / denom
    max_rel_diff = rel_diff.max().item()

    # Find mismatches using combined tolerance: |actual - expected| > atol + rtol * |expected|
    tol = atol + rtol * torch.abs(expected_f)
    mismatch_mask = diff > tol
    mismatch_indices = torch.nonzero(mismatch_mask)
    num_mismatches = mismatch_indices.shape[0]

    passed = num_mismatches == 0
    stats = f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, max_rel_diff={max_rel_diff:.2e}"

    if passed:
        return True, 0, stats
    else:
        total_elements = actual.numel()
        pct = 100.0 * num_mismatches / total_elements
        msg = f"MISMATCH: {num_mismatches:,} of {total_elements:,} elements ({pct:.2f}%) differ\n"
        msg += f"  {stats}\n"

        for i in range(min(num_diffs, num_mismatches)):
            idx = tuple(mismatch_indices[i].tolist())
            act_val = actual_f[idx].item()
            exp_val = expected_f[idx].item()
            d = diff[idx].item()
            t = tol[idx].item()
            msg += f"  [{idx}]: actual={act_val:+.6e}, expected={exp_val:+.6e}, diff={d:.2e}, tol={t:.2e}\n"

        return False, num_mismatches, msg

def get_fp8_largest_po2(dtype: torch.dtype):
    if dtype == torch.float8_e4m3fn:
        return 128.0
    elif dtype == torch.float8_e5m2:
        return 32768.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def get_fp8_scale_factor(amax: float, dtype: torch.dtype, fudge_factor: float = 0.25, epsilon = 0.0625):
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return 1.0
    po2_next = 2 ** math.ceil(math.log2(max(amax, epsilon)))
    return get_fp8_largest_po2(dtype) / po2_next * fudge_factor

def get_fp8_descale_factor(amax: float, dtype: torch.dtype, fudge_factor: float = 0.25, epsilon = 0.0625):
    return 1.0 / get_fp8_scale_factor(amax, dtype, fudge_factor, epsilon)

def compute_total_elems(shape, strides):
    """Compute total element count (max offset + 1) from shape and strides."""
    return sum((s - 1) * st for s, st in zip(shape, strides)) + 1

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    elif torch_type == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    elif torch_type == torch.float8_e5m2:
        return cudnn.data_type.FP8_E5M2
    else:
        assert False, "unsupported tensor data type"

def alloc_tensor(shape, data_type, *, elems=None, strides=None, rng=None, mean=0.1, std=0.1, margins=512, sparse_int=True, sparsity=0.8, abs_max=2):
    """
    Allocate a tensor with optional random initialization.

    Args:
        shape: Tensor shape
        data_type: PyTorch dtype
        elems: Number of elements (computed from shape/strides if not provided)
        strides: Custom strides (contiguous if not provided)
        rng: torch.Generator for random initialization. If None, tensor is not initialized.
        mean, std: Parameters for normal distribution (only used if sparse_int=False)
        margins: Safety margins for detecting buffer overruns
        sparse_int: If True, use sparse small integers. If False, use normal distribution.
        sparsity: Fraction of zeros when using sparse_int (default 0.8 = 80% zeros)
        abs_max: Maximum absolute value when using sparse_int (default 2, gives [-2, 2])

    Returns:
        (tensor, sepbuf, rawbuf) tuple for boundary checking
    """
    if strides is None:
        # Compute default contiguous strides
        if hasattr(shape, '__iter__'):
            strides = []
            prod = 1
            for dim in reversed(shape):
                strides.insert(0, prod)
                prod *= int(dim)
            if elems is None:
                elems = prod
        else:
            if elems is None:
                elems = int(shape)
            strides = (1,)
            shape = (shape,)
    elif elems is None:
        elems = compute_total_elems(shape, strides)

    assert margins >= 0 and type(margins) == int, "wrong input"

    rawbuf = torch.empty(elems+2*margins, dtype=data_type, device="cuda")
    if torch.is_floating_point(rawbuf):
        rawbuf.fill_(float('nan'))
    else:
        rawbuf.fill_(-1)

    tensor = torch.as_strided(rawbuf, shape, strides, storage_offset=margins)
    sepbuf = (torch.as_strided(rawbuf, (2, margins), (elems+margins, 1), storage_offset=0) if margins > 0 else None)

    if rng is not None:
        if sparse_int:
            fill_sparse_small_int(tensor, rng, sparsity=sparsity, abs_max=abs_max)
        else:
            tensor.normal_(mean=mean, std=std, generator=rng)

    if math.prod(shape) == elems:
        rawbuf = None

    return tensor, sepbuf, rawbuf

def prefix_sum(t):
    t = t.flatten()
    return torch.cat((torch.zeros(1, dtype=t.dtype, device=t.device), torch.cumsum(t, dim=0)))

def convert_packed_to_uniform(packed_tensor, seq_len, s_max, fill_value=0):
    assert packed_tensor.dim() == 3
    t, h, d = packed_tensor.size()
    seq_len = seq_len.flatten()
    b = seq_len.size(0)

    uniform_tensor = torch.full((b, s_max, h, d), fill_value, dtype=packed_tensor.dtype, device=packed_tensor.device)

    t_idx = 0
    for bi, s in enumerate(seq_len):
        uniform_tensor[bi, 0:s, :, :] = packed_tensor[t_idx : t_idx + s, :, :]
        t_idx += s

    uniform_tensor = torch.einsum("bshd->bhsd", uniform_tensor)
    return uniform_tensor

def convert_uniform_to_packed(uniform_tensor, seq_len, max_t):
    assert uniform_tensor.dim() == 4
    uniform_tensor = torch.einsum("bhsd->bshd", uniform_tensor)
    b, s, h, d = uniform_tensor.size()
    seq_len = seq_len.flatten()
    assert seq_len.size(0) == b
    packed_tensor = torch.full((max_t, h, d), float('nan'), dtype=uniform_tensor.dtype, device=uniform_tensor.device)

    t_idx = 0
    for bi, s_len in enumerate(seq_len):
        packed_tensor[t_idx : t_idx + s_len, :, :] = uniform_tensor[bi, 0:s_len, :, :]
        t_idx += s_len

    return packed_tensor

def create_container_and_page_table(tensor, block_size):
    B, H, S, D = tensor.shape
    blocks_per_batch = math.ceil(S/block_size)

    padding_seq = (blocks_per_batch * block_size) - S
    if padding_seq > 0:
        zeros = torch.zeros(B,H,padding_seq,D, device='cuda', dtype=tensor.dtype)
        cat_tensor = torch.cat((tensor, zeros), axis = 2)
    else:
        cat_tensor = tensor

    reshaped = torch.cat((cat_tensor.clone()).chunk(blocks_per_batch, dim=2), dim=0)

    table_size = math.ceil(S/block_size)
    page_table = torch.linspace(0, B*table_size-1, B*table_size, device='cuda', dtype=torch.int32).reshape(table_size,1,B,1)
    page_table = torch.transpose(page_table,0,2)

    return(reshaped, page_table)

def exact_equal(actual, expected, tag, disp_elems):
    both_nan = torch.isnan(actual) & torch.isnan(expected)
    mismatches = torch.where((actual != expected) & ~both_nan)
    mismatch_cnt = mismatches[0].numel()
    num_elements = torch.numel(actual)
    if mismatch_cnt != 0:
        percentage = 100 * mismatch_cnt / num_elements
        if disp_elems > 0:
            print(f"Comparing '{tag}' for exact (bitwise) equality")
            combined = torch.stack(mismatches, dim=-1).tolist()
            count = 0
            for index in combined:
                diff = actual[tuple(index)].float() - expected[tuple(index)].float()
                print(f"idx{index}: {tag}_run1={actual[tuple(index)]}, {tag}_run2={expected[tuple(index)]}, diff={diff:+.2e}")
                count += 1
                if count >= disp_elems:
                    break
            print(f"%%%% Total {mismatch_cnt:,} mismatches ({percentage:.1f}%) when validating '{tag}' for exact equality (first {count} mismatches displayed)")
        else:
            print(f"%%%% Total {mismatch_cnt:,} mismatches ({percentage:.1f}%) when validating '{tag}' for exact equality")
    else:
        print(f"%%%% Exact (bitwise) equality of '{tag}' verified")
    return mismatch_cnt

def approx_equal(alloc, expected, atol, rtol, tag, disp_elems):
    actual, sepbuf, rawbuf = alloc
    mismatches = torch.where(torch.isclose(actual.float(), expected, rtol=rtol, atol=atol, equal_nan=True) == False)
    mismatch_cnt = mismatches[0].numel()
    num_elements = torch.numel(actual)
    if mismatch_cnt != 0:
        percentage = 100 * mismatch_cnt / num_elements
        if disp_elems > 0:
            print(f"Comparing '{tag}' using rtol={rtol:.4e}, atol={atol:.4e}")
            combined = torch.stack(mismatches, dim=-1).tolist()
            count = 0
            for index in combined:
                diff = actual[tuple(index)] - expected[tuple(index)]
                if math.isfinite(diff):
                    print(f"idx{index}: {tag}_gpu={actual[tuple(index)]:+.6e}, {tag}_ref={expected[tuple(index)]:+.6e}, diff={diff:+.2e}")
                else:
                    print(f"idx{index}: {tag}_gpu={actual[tuple(index)]:+.6e}, {tag}_ref={expected[tuple(index)]:+.6e}")
                count += 1
                if count >= disp_elems:
                    break
            print(f"%%%% Total {mismatch_cnt:,} mismatches ({percentage:.1f}%) when validating '{tag}' results (first {count} mismatches displayed)")
        else:
            print(f"%%%% Total {mismatch_cnt:,} mismatches ({percentage:.1f}%) when validating '{tag}' results")

        num_nans       = torch.isnan(actual).sum().item()
        num_infs       = torch.isinf(actual).sum().item()
        num_zeros      = num_elements - torch.count_nonzero(actual)
        num_finites_nz = num_elements - num_nans - num_infs - num_zeros

        print(f"%%%% {tag}_gpu overview: elements={num_elements:,}, finites_nz={num_finites_nz:,}, zeros={num_zeros:,}, nans={num_nans:,}, infs={num_infs:,}")

        num_nans       = torch.isnan(expected).sum().item()
        num_infs       = torch.isinf(expected).sum().item()
        num_zeros      = num_elements - torch.count_nonzero(expected)
        num_finites_nz = num_elements - num_nans - num_infs - num_zeros

        print(f"%%%% {tag}_ref overview: elements={num_elements:,}, finites_nz={num_finites_nz:,}, zeros={num_zeros:,}, nans={num_nans:,}, infs={num_infs:,}")
    else:
        print(f"%%%% Numerical divergence of '{tag}' within limits")

    if sepbuf is not None and not torch.all(torch.isnan(sepbuf)).item():
        print(f"%%%% Buffer '{tag}' overwritten outside its boundaries")
        print(sepbuf)
        mismatch_cnt += 1

    if rawbuf is not None:
        actual.fill_(float('nan'))
        if not torch.all(torch.isnan(rawbuf)).item():
            print(f"%%%% Unused gaps of '{tag}' tensor were overwritten")
            mismatch_cnt += 1

    return mismatch_cnt

def time_execution(fn, *args, num_warmup: int = 3, num_trials: int = 10) -> torch.Tensor:
    elapsed_times = torch.zeros(num_trials, dtype=torch.float)
    for _ in range(num_warmup):
        fn(*args)
        torch.cuda.synchronize()
    for i in range(num_trials):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_times[i] = start_event.elapsed_time(end_event)
    return elapsed_times

def profile_execution(fn, *args, trace_dir=None):
    activities = [torch.profiler.ProfilerActivity.CUDA]
    if trace_dir:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=(torch.profiler.tensorboard_trace_handler(trace_dir) if trace_dir else None),
    ) as prof:
        fn(*args)
        torch.cuda.synchronize()
    print("Sorted by CUDA time:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    print()
    if torch.profiler.ProfilerActivity.CPU in activities:
        print("Sorted by CPU time:")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        print()

def print_section_begin(msg, width=80):
    print(f" {msg} ".center(width, "="))

def print_section_end(width=80):
    print("=" * width)
