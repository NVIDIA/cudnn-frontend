import cudnn
import torch
import math

# fmt: off

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

def alloc_tensor(shape, data_type, *, elems=None, strides=None, rng=None, mean=0.0, std=1.0, margins=512):
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
