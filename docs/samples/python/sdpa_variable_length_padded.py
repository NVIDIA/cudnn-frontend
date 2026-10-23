"""Complete standalone Python sample.
API page: docs/api/python/sdpa.md
API: sdpa
Use case: Variable-length padded attention
Source: adapted from the THD layout setup in `test/python/test_sdpa_thd.py`
and `python/cudnn/native_sparse_attention/sliding_window_attention/api.py`.
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
    B, Hq, Hkv, S, D = 1, 2, 1, 16, 64
    q_stride = (Hq * S * D, D, Hq * D, 1)
    kv_stride = (Hkv * S * D, D, Hkv * D, 1)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        is_override_shape_enabled=True,
    )
    q_offsets = graph.tensor(
        name="q_offsets",
        dim=(B + 1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT64,
    )
    k_offsets = graph.tensor(
        name="k_offsets",
        dim=(B + 1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT64,
    )
    v_offsets = graph.tensor(
        name="v_offsets",
        dim=(B + 1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT64,
    )
    o_offsets = graph.tensor(
        name="o_offsets",
        dim=(B + 1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=cudnn.data_type.INT64,
    )
    q = graph.tensor(name="q", dim=(B, Hq, S, D), stride=q_stride, data_type=cudnn.data_type.BFLOAT16)
    q.set_ragged_offset(q_offsets)
    k = graph.tensor(
        name="k",
        dim=(B, Hkv, S, D),
        stride=kv_stride,
        data_type=cudnn.data_type.BFLOAT16,
    )
    k.set_ragged_offset(k_offsets)
    v = graph.tensor(
        name="v",
        dim=(B, Hkv, S, D),
        stride=kv_stride,
        data_type=cudnn.data_type.BFLOAT16,
    )
    v.set_ragged_offset(v_offsets)
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
        generate_stats=False,
    )
    out.set_output(True).set_name("out").set_dim((B, Hq, S, D)).set_stride(q_stride).set_data_type(cudnn.data_type.BFLOAT16)
    out.set_ragged_offset(o_offsets)
    graph.validate()
    print("sdpa variable-length graph validated")


if __name__ == "__main__":
    run_sample(sample)
