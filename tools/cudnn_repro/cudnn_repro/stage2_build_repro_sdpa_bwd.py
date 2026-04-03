"""Stage 2: Build pytest repro command for SDPA backward."""

from . import repro_command
from . import stage1_annotate_sdpa_bwd as stage1


def normalize_repro_cfg(cfg: dict) -> dict:
    """Normalize config for command generation.

    TODO: Add backward-specific normalization if needed.
    """
    return repro_command.normalize_repro_cfg(cfg)


def build_command(cfg: dict) -> str:
    """Build a simple one-line pytest command."""
    return repro_command.build_command(cfg)


def build_pretty_command(cfg: dict) -> str:
    """Build a multi-line formatted pytest command."""
    return repro_command.build_pretty_command(cfg)


def build_repro_command(raw_line: str, stage1_json: dict) -> str:
    """Stage 2: Build pytest command from stage1 JSON for backward."""
    seed = stage1_json.get("repro_metadata", {}).get("rng_data_seed")
    cfg = stage1.build_cfg(raw_line, stage1_json, seed)
    return build_pretty_command(cfg)
