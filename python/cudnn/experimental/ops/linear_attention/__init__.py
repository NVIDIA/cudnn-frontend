"""Linear-attention torch custom operators (GDN).

Importing this package registers the ``cudnn::gated_delta_net_fwd`` op with
``torch.library``.

Forward and backward dispatch to the vendored cuTile chunked kernel
(``_gdn_chunk_cutile``), which requires the ``cuda.tile`` runtime.
"""

from .gdn import gated_delta_net

__all__ = [
    "gated_delta_net",
]
