"""Linear-attention torch custom operators (GDN, KDA, GDN-2).

Importing this package registers three ``cudnn::*`` ops with
``torch.library``:

- ``cudnn::gated_delta_net_fwd`` / ``_bwd``         — GDN
- ``cudnn::kimi_delta_attention_fwd`` / ``_bwd``    — KDA
- ``cudnn::gated_delta_net_v2_fwd`` / ``_bwd``      — GDN-2

The current implementations are pure-PyTorch references (chunked forward
for GDN/KDA, per-token recurrent forward for GDN-2; backward routed
through autograd on the recurrent form). They are correct but not
performance-tuned: a fused cuDNN backend can later replace the op bodies
without changing the public functional API.
"""

from .gdn import gated_delta_net
from .kda import kimi_delta_attention
from .gdn2 import gated_delta_net_v2

__all__ = [
    "gated_delta_net",
    "kimi_delta_attention",
    "gated_delta_net_v2",
]
