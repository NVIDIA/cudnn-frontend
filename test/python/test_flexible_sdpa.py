import cudnn
import torch
import pytest
from functools import partial
import math

from test_utils import torch_fork_set_rng
from looseversion import LooseVersion


# Helper function to create a non contiguous container in blocks of block_size from a contiguous tensor
def create_container_and_page_table(tensor, block_size):
    B, H, S, D = tensor.shape
    # num_blocks = math.ceil(S/block_size) * B
    blocks_per_batch = math.ceil(S / block_size)

    # Only needed if S is not a multiple of block_size
    padding_seq = (blocks_per_batch * block_size) - S
    if padding_seq > 0:
        zeros = torch.zeros(B, H, padding_seq, D, device="cuda", dtype=tensor.dtype)
        cat_tensor = torch.cat((tensor, zeros), axis=2)
    else:
        cat_tensor = tensor

    # Create a container by splitting on the S dimension and concatenating at the block dimension
    # Its dimensions are [num_blocks, H, block_size, D] with num_blocks = B * blocks_per_batch
    container = torch.cat((cat_tensor.clone()).chunk(blocks_per_batch, dim=2), dim=0)

    # Create the page table
    table_size = math.ceil(S / block_size)
    page_table_temp = torch.linspace(
        0, B * table_size - 1, B * table_size, device="cuda", dtype=torch.int32
    ).reshape(table_size, 1, B, 1)
    page_table_temp = torch.transpose(page_table_temp, 0, 2)

    # Make batch size outer dimension (cuDNN backend requirement)
    page_table = (
        torch.randn(blocks_per_batch * B)
        .int()
        .cuda()
        .as_strided(
            (B, 1, blocks_per_batch, 1), (blocks_per_batch, blocks_per_batch, 1, 1)
        )
    )
    page_table.copy_(page_table_temp)

    return (container, page_table)


def padding_mask(sdpa_graph, q_kt_tensor, seq_len_q, seq_len_kv, neg_inf):
    row_index = sdpa_graph.gen_index(
        input=q_kt_tensor,
        axis=2,
        name="row_index",
        compute_data_type=cudnn.data_type.INT32,
    )
    row_index.set_data_type(cudnn.data_type.INT32)

    col_index = sdpa_graph.gen_index(
        input=q_kt_tensor,
        axis=3,
        name="col_index",
        compute_data_type=cudnn.data_type.INT32,
    )
    col_index.set_data_type(cudnn.data_type.INT32)

    row_mask = sdpa_graph.cmp_ge(
        input=row_index,
        comparison=seq_len_q,
        compute_data_type=cudnn.data_type.FLOAT,
        name="row_mask",
    )
    row_mask.set_data_type(cudnn.data_type.BOOLEAN)

    col_mask = sdpa_graph.cmp_ge(
        input=col_index,
        comparison=seq_len_kv,
        compute_data_type=cudnn.data_type.FLOAT,
        name="col_mask",
    )
    col_mask.set_data_type(cudnn.data_type.BOOLEAN)

    padding_mask = sdpa_graph.logical_and(
        a=row_mask,
        b=col_mask,
        compute_data_type=cudnn.data_type.FLOAT,
        name="padding_mask",
    )
    padding_mask.set_data_type(cudnn.data_type.BOOLEAN)

    out = sdpa_graph.binary_select(
        input0=q_kt_tensor, input1=neg_inf, mask=padding_mask, name="binary_select"
    )

    return out


def softcap(sdpa_graph, q_kt_tensor, softcap_tensor):

    div_out = sdpa_graph.div(a=q_kt_tensor, b=softcap_tensor)

    tanh_out = sdpa_graph.tanh(input=div_out)

    out = sdpa_graph.mul(a=tanh_out, b=softcap_tensor)

    return out


def decode_mask(
    sdpa_graph, q_kt_tensor, seq_len_kv, seq_len_q, neg_inf, softcap_tensor
):

    softcap_out = softcap(sdpa_graph, q_kt_tensor, softcap_tensor)

    out = padding_mask(sdpa_graph, softcap_out, seq_len_q, seq_len_kv, neg_inf)

    return out


def causal_mask(sdpa_graph, q_kt_tensor):

    row_index = sdpa_graph.gen_index(input=q_kt_tensor, axis=2)
    row_index.set_data_type(cudnn.data_type.INT32)

    col_index = sdpa_graph.gen_index(input=q_kt_tensor, axis=3)
    col_index.set_data_type(cudnn.data_type.INT32)

    mask = sdpa_graph.cmp_ge(
        input=row_index, comparison=col_index, compute_data_type=cudnn.data_type.BOOLEAN
    )
    mask.set_data_type(cudnn.data_type.BOOLEAN)

    out = sdpa_graph.binary_select(input0=q_kt_tensor, input1=q_kt_tensor, mask=mask)

    return out


def constant_bound_mask(score_mod_graph, index, bound):
    is_less_than_bound = score_mod_graph.cmp_lt(
        input=index, comparison=bound, compute_data_type=cudnn.data_type.BOOLEAN
    )
    is_less_than_bound.set_data_type(cudnn.data_type.INT32)

    return is_less_than_bound


def diag_bound_mask(score_mod_graph, row_index, col_index, diag_bound_0, diag_bound_1):
    row_minus_col = score_mod_graph.sub(
        a=row_index, b=col_index, compute_data_type=cudnn.data_type.INT32
    )
    row_minus_col.set_data_type(cudnn.data_type.INT32)
    is_larger_or_equal_to_diag_bound_0 = score_mod_graph.cmp_ge(
        input=row_minus_col,
        comparison=diag_bound_0,
        compute_data_type=cudnn.data_type.BOOLEAN,
    )
    is_larger_or_equal_to_diag_bound_0.set_data_type(cudnn.data_type.INT32)

    is_less_than_or_equal_to_diag_bound_1 = score_mod_graph.cmp_le(
        input=row_minus_col,
        comparison=diag_bound_1,
        compute_data_type=cudnn.data_type.BOOLEAN,
    )
    is_less_than_or_equal_to_diag_bound_1.set_data_type(cudnn.data_type.INT32)

    is_within_diag_bound = score_mod_graph.logical_and(
        is_larger_or_equal_to_diag_bound_0,
        is_less_than_or_equal_to_diag_bound_1,
        compute_data_type=cudnn.data_type.BOOLEAN,
    )
    is_within_diag_bound.set_data_type(cudnn.data_type.INT32)

    return is_within_diag_bound


def arrow_mask(
    score_mod_graph,
    q_kt_tensor,
    row_bound,
    col_bound,
    diag_bound_0,
    diag_bound_1,
    neg_inf,
):
    row_index = score_mod_graph.gen_index(input=q_kt_tensor, axis=2)
    row_index.set_data_type(cudnn.data_type.INT32)

    col_index = score_mod_graph.gen_index(input=q_kt_tensor, axis=3)
    col_index.set_data_type(cudnn.data_type.INT32)

    is_less_than_row_bound = constant_bound_mask(score_mod_graph, row_index, row_bound)
    is_less_than_col_bound = constant_bound_mask(score_mod_graph, col_index, col_bound)

    is_within_diag_bound = diag_bound_mask(
        score_mod_graph, row_index, col_index, diag_bound_0, diag_bound_1
    )

    mask = score_mod_graph.logical_or(
        is_less_than_row_bound,
        is_less_than_col_bound,
        compute_data_type=cudnn.data_type.BOOLEAN,
    )
    mask.set_data_type(cudnn.data_type.INT32)

    mask = score_mod_graph.logical_or(
        mask, is_within_diag_bound, compute_data_type=cudnn.data_type.BOOLEAN
    )
    mask.set_data_type(cudnn.data_type.INT32)

    out = score_mod_graph.binary_select(input0=q_kt_tensor, input1=neg_inf, mask=mask)

    return out


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_with_flexible_graph(cudnn_handle):

    b = 4  # batch size
    h_q = 12  # query number of heads
    h_k = 12  # key number of heads
    h_v = 12  # value number of heads
    s_q = 1  # maximum sequence length
    s_kv = 32 * 1024  # maximum sequence length
    d = 128  # embedding dimension per head

    attn_scale = 1.0 / math.sqrt(d)

    block_size_k, block_size_v = 1, 1

    softcap_scalar_value = 0.8
    neg_inf_scalar_value = -1e9

    q_dims = (b, h_q, s_q, d)
    q_strides = (s_q * h_q * d, d, h_q * d, 1)
    k_dims = (b, h_k, s_kv, d)
    k_strides = (s_kv * h_k * d, d, h_k * d, 1)
    v_dims = (b, h_v, s_kv, d)
    v_strides = (s_kv * h_v * d, d, h_v * d, 1)

    q_gpu = torch.randn(b * s_q * h_q * d).half().cuda().as_strided(q_dims, q_strides)
    k_gpu = torch.randn(b * s_kv * h_k * d).half().cuda().as_strided(k_dims, k_strides)
    v_gpu = torch.randn(b * s_kv * h_v * d).half().cuda().as_strided(v_dims, v_strides)
    o_gpu = torch.empty(b * s_q * h_q * d).half().cuda().as_strided(q_dims, q_strides)

    cudnn_version = LooseVersion(cudnn.backend_version_string())

    if cudnn_version < "9.6.0":
        pytest.skip("SDPA fprop with paged attention requires cudnn 9.6.0 or higher")

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = graph.tensor_like(q_gpu)
    container_k_gpu, page_table_k_gpu = create_container_and_page_table(
        k_gpu, block_size_k
    )
    container_v_gpu, page_table_v_gpu = create_container_and_page_table(
        v_gpu, block_size_v
    )

    container_k = graph.tensor_like(container_k_gpu)
    container_v = graph.tensor_like(container_v_gpu)
    page_table_k = graph.tensor_like(page_table_k_gpu)
    page_table_v = graph.tensor_like(page_table_v_gpu)

    seq_len_q_gpu = torch.randint(
        1, s_q + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda"
    )
    seq_len_kv_gpu = torch.randint(
        1, s_kv + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda"
    )

    seq_len_q = graph.tensor_like(seq_len_q_gpu)
    seq_len_kv = graph.tensor_like(seq_len_kv_gpu)

    softcap_tensor_cpu = torch.full((1, 1, 1, 1), softcap_scalar_value)
    neg_inf_tensor_cpu = torch.full((1, 1, 1, 1), neg_inf_scalar_value)
    softcap_tensor = graph.tensor(
        name="softcap_scalar",
        dim=softcap_tensor_cpu.size(),
        stride=softcap_tensor_cpu.stride(),
        is_pass_by_value=True,
        data_type=softcap_tensor_cpu.dtype,
    )

    neg_inf_tensor = graph.tensor(
        name="neg_inf_scalar",
        dim=neg_inf_tensor_cpu.size(),
        stride=neg_inf_tensor_cpu.stride(),
        is_pass_by_value=True,
        data_type=neg_inf_tensor_cpu.dtype,
    )

    o, _ = graph.sdpa(
        name="sdpa",
        q=q,
        k=container_k,
        v=container_v,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        is_inference=True,
        attn_scale=attn_scale,
        use_causal_mask=False,
        paged_attention_k_table=page_table_k,  # Page Table K: Tensor containing offsets to the container with K blocks
        paged_attention_v_table=page_table_v,  # Page Table V: Tensor containing offsets to the container with V blocks
        paged_attention_max_seq_len_kv=s_kv,  # The maximum sequence length for K caches (this is optional, but recommended)
        score_mod=partial(
            decode_mask,
            softcap_tensor=softcap_tensor,
            neg_inf=neg_inf_tensor,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
        ),
    )

    o.set_output(True).set_dim(q_dims).set_stride(q_strides)

    graph.build([cudnn.heur_mode.A])

    variant_pack = {
        q: q_gpu,
        container_k: container_k_gpu,
        container_v: container_v_gpu,
        page_table_k: page_table_k_gpu,
        page_table_v: page_table_v_gpu,
        o: o_gpu,
        softcap_tensor: softcap_tensor_cpu,
        neg_inf_tensor: neg_inf_tensor_cpu,
        seq_len_q: seq_len_q_gpu,
        seq_len_kv: seq_len_kv_gpu,
    }

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()


def document_mask(sdpa_graph, q_kt_tensor, document_tensor, document_tensor_t, neg_inf):

    col_index = sdpa_graph.gen_index(
        input=q_kt_tensor,
        axis=3,
        name="col_index",
        compute_data_type=cudnn.data_type.INT32,
    )
    col_index.set_data_type(cudnn.data_type.INT32)

    all_1_mask = sdpa_graph.cmp_le(
        input=col_index,
        comparison=col_index,
        compute_data_type=cudnn.data_type.FLOAT,
        name="all_1_mask",
    )
    all_1_mask.set_data_type(cudnn.data_type.INT32)

    row_bcast = sdpa_graph.scale(
        input=all_1_mask,
        scale=document_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="row_bcast",
    )
    row_bcast.set_data_type(cudnn.data_type.INT32)

    col_bcast = sdpa_graph.scale(
        input=all_1_mask,
        scale=document_tensor_t,
        compute_data_type=cudnn.data_type.FLOAT,
        name="col_bcast",
    )
    col_bcast.set_data_type(cudnn.data_type.INT32)

    document_mask = sdpa_graph.cmp_eq(
        input=row_bcast,
        comparison=col_bcast,
        compute_data_type=cudnn.data_type.FLOAT,
        name="document_mask",
    )
    document_mask.set_data_type(cudnn.data_type.INT32)

    out = sdpa_graph.binary_select(
        input0=q_kt_tensor, input1=neg_inf, mask=document_mask, name="binary_select"
    )

    return out


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_with_arrow_mask(cudnn_handle):

    b = 4  # batch size
    h_q = 12  # query number of heads
    h_k = 12  # key number of heads
    h_v = 12  # value number of heads
    s_q = 256  # maximum sequence length
    s_kv = 256  # maximum sequence length
    d = 128  # embedding dimension per head

    attn_scale = 1.0 / math.sqrt(d)

    row_bound_val = 2
    col_bound_val = 2
    diag_bound_0_val = 1
    diag_bound_1_val = 1

    neg_inf_scalar_value = -1e9

    q_dims = (b, h_q, s_q, d)
    q_strides = (s_q * h_q * d, d, h_q * d, 1)
    k_dims = (b, h_k, s_kv, d)
    k_strides = (s_kv * h_k * d, d, h_k * d, 1)
    v_dims = (b, h_v, s_kv, d)
    v_strides = (s_kv * h_v * d, d, h_v * d, 1)

    q_gpu = torch.randn(b * s_q * h_q * d).half().cuda().as_strided(q_dims, q_strides)
    k_gpu = torch.randn(b * s_kv * h_k * d).half().cuda().as_strided(k_dims, k_strides)
    v_gpu = torch.randn(b * s_kv * h_v * d).half().cuda().as_strided(v_dims, v_strides)
    o_gpu = torch.empty(b * s_q * h_q * d).half().cuda().as_strided(q_dims, q_strides)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = graph.tensor_like(q_gpu)

    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)

    neg_inf_tensor_cpu = torch.full((1, 1, 1, 1), neg_inf_scalar_value)
    row_bound_cpu = torch.full((1, 1, 1, 1), row_bound_val, dtype=torch.int32)
    col_bound_cpu = torch.full((1, 1, 1, 1), col_bound_val, dtype=torch.int32)
    diag_bound_0_cpu = torch.full((1, 1, 1, 1), diag_bound_0_val, dtype=torch.int32)
    diag_bound_1_cpu = torch.full((1, 1, 1, 1), diag_bound_1_val, dtype=torch.int32)
    row_bound = graph.tensor(
        name="row_bound",
        dim=row_bound_cpu.size(),
        stride=row_bound_cpu.stride(),
        is_pass_by_value=True,
        data_type=row_bound_cpu.dtype,
    )
    col_bound = graph.tensor(
        name="col_bound",
        dim=col_bound_cpu.size(),
        stride=col_bound_cpu.stride(),
        is_pass_by_value=True,
        data_type=col_bound_cpu.dtype,
    )
    diag_bound_0 = graph.tensor(
        name="diag_bound_0",
        dim=diag_bound_0_cpu.size(),
        stride=diag_bound_0_cpu.stride(),
        is_pass_by_value=True,
        data_type=diag_bound_0_cpu.dtype,
    )
    diag_bound_1 = graph.tensor(
        name="diag_bound_1",
        dim=diag_bound_1_cpu.size(),
        stride=diag_bound_1_cpu.stride(),
        is_pass_by_value=True,
        data_type=diag_bound_1_cpu.dtype,
    )
    neg_inf_tensor = graph.tensor(
        name="neg_inf_scalar",
        dim=neg_inf_tensor_cpu.size(),
        stride=neg_inf_tensor_cpu.stride(),
        is_pass_by_value=True,
        data_type=neg_inf_tensor_cpu.dtype,
    )

    o, _ = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        is_inference=True,
        attn_scale=attn_scale,
        score_mod=partial(
            arrow_mask,
            row_bound=row_bound,
            col_bound=col_bound,
            diag_bound_0=diag_bound_0,
            diag_bound_1=diag_bound_1,
            neg_inf=neg_inf_tensor,
        ),
    )

    o.set_output(True).set_dim(q_dims).set_stride(q_strides)

    graph.build([cudnn.heur_mode.A])

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        neg_inf_tensor: neg_inf_tensor_cpu,
        row_bound: row_bound_cpu,
        col_bound: col_bound_cpu,
        diag_bound_0: diag_bound_0_cpu,
        diag_bound_1: diag_bound_1_cpu,
    }

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_sdpa_with_document_mask(cudnn_handle):

    b = 1  # batch size
    h_q = 1  # query number of heads
    h_k = 1  # key number of heads
    h_v = 1  # value number of heads
    s_q = 16  # maximum sequence length
    s_kv = 16  # maximum sequence length
    d = 128  # embedding dimension per head

    q_dims = (b, h_q, s_q, d)
    q_strides = (s_q * h_q * d, d, h_q * d, 1)
    k_dims = (b, h_k, s_kv, d)
    k_strides = (s_kv * h_k * d, d, h_k * d, 1)
    v_dims = (b, h_v, s_kv, d)
    v_strides = (s_kv * h_v * d, d, h_v * d, 1)

    q_gpu = torch.randn(b * s_q * h_q * d).half().cuda().as_strided(q_dims, q_strides)
    k_gpu = torch.randn(b * s_kv * h_k * d).half().cuda().as_strided(k_dims, k_strides)
    v_gpu = torch.randn(b * s_kv * h_v * d).half().cuda().as_strided(v_dims, v_strides)
    o_gpu = torch.empty(b * s_q * h_q * d).half().cuda().as_strided(q_dims, q_strides)

    cudnn_version = LooseVersion(cudnn.backend_version_string())

    if cudnn_version < "9.9.0":
        pytest.skip(
            "SDPA fprop with document style mask requires cudnn 9.9.0 or higher"
        )

    document_tensor_gpu = (
        torch.randint(0, s_q, (1, 1, s_q, 1), device="cuda", dtype=torch.int32)
        .sort(dim=2)
        .values
    )
    document_tensor_gpu_t = document_tensor_gpu.reshape(1, 1, 1, s_q)

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.HALF,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu)
    v = graph.tensor_like(v_gpu)

    document_tensor = graph.tensor_like(document_tensor_gpu)
    document_tensor_t = graph.tensor_like(document_tensor_gpu_t)

    attn_scale = 1.0 / math.sqrt(d)
    neg_inf_scalar_value = -1e9
    neg_inf_tensor_cpu = torch.full((1, 1, 1, 1), neg_inf_scalar_value)

    neg_inf_tensor = graph.tensor(
        name="neg_inf_scalar",
        dim=neg_inf_tensor_cpu.size(),
        stride=neg_inf_tensor_cpu.stride(),
        is_pass_by_value=True,
        data_type=neg_inf_tensor_cpu.dtype,
    )

    o, _ = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        is_inference=True,
        attn_scale=attn_scale,
        score_mod=partial(
            document_mask,
            document_tensor=document_tensor,
            document_tensor_t=document_tensor_t,
            neg_inf=neg_inf_tensor,
        ),
    )

    o.set_output(True).set_dim(q_dims).set_stride(q_strides)

    graph.build([cudnn.heur_mode.A])

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        o: o_gpu,
        document_tensor: document_tensor_gpu,
        document_tensor_t: document_tensor_gpu_t,
        neg_inf_tensor: neg_inf_tensor_cpu,
    }

    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    graph.execute(variant_pack, workspace)
    torch.cuda.synchronize()
