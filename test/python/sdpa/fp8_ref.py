import torch

# fmt: off

def compute_ref(q, k, v, attn_scale=1.0, return_type="o"):
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, d_v = v.shape

    assert k.shape == (b, s_kv, h_k, d_qk)
    assert v.shape == (b, s_kv, h_v, d_v)

    if h_q != h_k:
        k = k.repeat_interleave(h_q // h_k, dim=2)
    if h_q != h_v:
        v = v.repeat_interleave(h_q // h_v, dim=2)

    s = torch.einsum("bqhd,bkhd->bhqk", q, k) * attn_scale
    p = s.softmax(dim=-1)
    o = torch.einsum("bhqk,bkhd->bqhd", p, v)

    if return_type == "o":
        return o
    if return_type == "o_stats":
        return o, torch.zeros()
    elif return_type == "amax":
        return p.abs().max().item(), o.abs().max().item()
    else:
        raise ValueError(f"Unsupported return type: {return_type}")
