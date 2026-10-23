"""Complete standalone Python sample.
API page: docs/api/python/sdpa.md
API: sdpa
Use case: Paged KV cache
Source: simplified from the paged-cache setup in `test/python/test_flexible_sdpa.py`.
"""

import sys

try:
    import cudnn
    import torch
except Exception as exc:
    print(f"SKIP: could not import cudnn or torch: {exc}")
    raise SystemExit(0)


def run_sample(sample):
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available to PyTorch")
        return
    try:
        sample()
    except Exception as exc:
        if "--strict" in sys.argv:
            raise
        print(f"SKIP: sample could not run on this installation: {exc}")


def sample():
    import math

    B, Hq, Hkv, S, D = 1, 2, 1, 16, 64
    block_size = 8
    blocks = math.ceil(S / block_size)
    q_stride = (Hq * S * D, D, Hq * D, 1)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    q = graph.tensor(name="q", dim=(B, Hq, S, D), stride=q_stride, data_type=cudnn.data_type.BFLOAT16)
    k = graph.tensor(
        name="k_cache",
        dim=(B * blocks, Hkv, block_size, D),
        stride=(Hkv * block_size * D, block_size * D, D, 1),
        data_type=cudnn.data_type.BFLOAT16,
    )
    v = graph.tensor(
        name="v_cache",
        dim=(B * blocks, Hkv, block_size, D),
        stride=(Hkv * block_size * D, block_size * D, D, 1),
        data_type=cudnn.data_type.BFLOAT16,
    )
    table_k = graph.tensor(
        name="table_k",
        dim=(B, 1, blocks, 1),
        stride=(blocks, blocks, 1, 1),
        data_type=cudnn.data_type.INT32,
    )
    table_v = graph.tensor(
        name="table_v",
        dim=(B, 1, blocks, 1),
        stride=(blocks, blocks, 1, 1),
        data_type=cudnn.data_type.INT32,
    )
    seq_q = graph.tensor(
        name="seq_len_q",
        dim=(B, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT32,
    )
    seq_kv = graph.tensor(
        name="seq_len_kv",
        dim=(B, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT32,
    )
    out, _ = graph.sdpa(
        q=q,
        k=k,
        v=v,
        attn_scale=float(D) ** -0.5,
        use_padding_mask=True,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
        paged_attention_k_table=table_k,
        paged_attention_v_table=table_v,
        paged_attention_max_seq_len_kv=S,
        generate_stats=False,
    )
    out.set_output(True).set_name("out").set_dim((B, Hq, S, D)).set_stride(q_stride).set_data_type(cudnn.data_type.BFLOAT16)
    graph.validate()
    print("sdpa paged cache graph validated")


if __name__ == "__main__":
    run_sample(sample)
