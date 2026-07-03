"""Compatibility import for the shared SM100 MXFP8 indexer-score kernel."""

from ..score_recompute.indexer_score_unified_sm100_mxfp8 import (
    IndexerScoreUnifiedSm100Mxfp8 as IndexerForwardSm100Mxfp8,
)

__all__ = ["IndexerForwardSm100Mxfp8"]
