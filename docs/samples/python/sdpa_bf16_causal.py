"""Complete standalone causal GQA forward and backward sample.

API pages: docs/api/python/sdpa.md and docs/api/python/sdpa_backward.md
Use case: BF16 causal GQA training
Source: adapted from `samples/llm_coverage/test_gqa_b+h+s+d.py`.
"""

import cudnn
import torch
import torch.nn.functional as F

B, H_Q, H_KV, S, D = 1, 2, 1, 16, 64
Q_SHAPE = (B, H_Q, S, D)
Q_STRIDE = (H_Q * S * D, D, H_Q * D, 1)  # BSHD storage
KV_SHAPE = (B, H_KV, S, D)
KV_STRIDE = (H_KV * S * D, D, H_KV * D, 1)  # BSHD storage
STATS_SHAPE = (B, H_Q, S, 1)
STATS_STRIDE = (H_Q * S, S, 1, 1)
DO_STRIDE = (H_Q * S * D, S * D, D, 1)  # BHSD storage
ATTN_SCALE = float(D) ** -0.5


def new_graph(*, inputs, outputs):
    return cudnn.Graph(
        io_data_type=torch.bfloat16,
        intermediate_data_type=torch.float32,
        compute_data_type=torch.float32,
        inputs=inputs,
        outputs=outputs,
        handle="auto",
    )


def add_qkv_tensors(graph):
    q = graph.tensor(name="q", dim=Q_SHAPE, stride=Q_STRIDE)
    k = graph.tensor(name="k", dim=KV_SHAPE, stride=KV_STRIDE)
    v = graph.tensor(name="v", dim=KV_SHAPE, stride=KV_STRIDE)
    return q, k, v


def build_forward_graph():
    with new_graph(
        inputs=["q", "k", "v"],
        outputs=["out", "stats"],
    ) as graph:
        q, k, v = add_qkv_tensors(graph)
        out, stats = graph.sdpa(
            q=q,
            k=k,
            v=v,
            generate_stats=True,
            attn_scale=ATTN_SCALE,
            diagonal_band_right_bound=0,
            name="sdpa",
        )
        out.set_output(True).set_name("out").set_dim(Q_SHAPE).set_stride(Q_STRIDE)
        stats.set_output(True).set_name("stats").set_data_type(torch.float32)
    return graph


def build_backward_graph():
    with new_graph(
        inputs=["q", "k", "v", "out", "dout", "stats"],
        outputs=["dq", "dk", "dv"],
    ) as graph:
        q, k, v = add_qkv_tensors(graph)
        out = graph.tensor(name="out", dim=Q_SHAPE, stride=Q_STRIDE)
        dout = graph.tensor(name="dout", dim=Q_SHAPE, stride=DO_STRIDE)
        stats = graph.tensor(
            name="stats",
            dim=STATS_SHAPE,
            stride=STATS_STRIDE,
            data_type=torch.float32,
        )
        dq, dk, dv = graph.sdpa_backward(
            q=q,
            k=k,
            v=v,
            o=out,
            dO=dout,
            stats=stats,
            attn_scale=ATTN_SCALE,
            diagonal_band_right_bound=0,
            name="sdpa_backward",
        )
        dq.set_output(True).set_name("dq").set_dim(Q_SHAPE).set_stride(Q_STRIDE)
        dk.set_output(True).set_name("dk").set_dim(KV_SHAPE).set_stride(KV_STRIDE)
        dv.set_output(True).set_name("dv").set_dim(KV_SHAPE).set_stride(KV_STRIDE)
    return graph


def sample():
    # Transposing BSHD tensors produces BHSD views with BSHD physical storage.
    q = torch.randn((B, S, H_Q, D), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn((B, S, H_KV, D), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    v = torch.randn((B, S, H_KV, D), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    dout = torch.randn(Q_SHAPE, device="cuda", dtype=torch.bfloat16)

    forward_graph = build_forward_graph()
    out, stats = forward_graph(q, k, v)

    backward_graph = build_backward_graph()
    dq, dk, dv = backward_graph(q, k, v, out, dout, stats)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        is_causal=True,
        enable_gqa=True,
    )
    out_ref.backward(dout)

    tolerances = {"rtol": 5e-2, "atol": 5e-2}
    torch.testing.assert_close(out, out_ref, **tolerances)
    torch.testing.assert_close(dq, q_ref.grad, **tolerances)
    torch.testing.assert_close(dk, k_ref.grad, **tolerances)
    torch.testing.assert_close(dv, v_ref.grad, **tolerances)
    print(f"causal GQA forward and backward passed: out={tuple(out.shape)}")


if __name__ == "__main__":
    sample()
