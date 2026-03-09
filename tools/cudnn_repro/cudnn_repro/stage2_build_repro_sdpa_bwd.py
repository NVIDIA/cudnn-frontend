"""Stage 2: Build pytest repro command for SDPA backward."""

from . import stage1_annotate_sdpa_bwd as stage1


def normalize_repro_cfg(cfg: dict) -> dict:
    """Normalize config for command generation.

    TODO: Add backward-specific normalization if needed.
    """
    repro_cfg = dict(cfg)
    if isinstance(repro_cfg.get("diag_align"), int):
        diag_to_name = {0: "TOP_LEFT", 1: "BOTTOM_RIGHT"}
        name = diag_to_name.get(repro_cfg["diag_align"])
        if name is not None:
            repro_cfg["diag_align"] = f"cudnn.diagonal_alignment.{name}"

    impl = repro_cfg.get("implementation")
    if isinstance(impl, str) and "." not in impl:
        repro_cfg["implementation"] = f"cudnn.attention_implementation.{impl}"
    return repro_cfg


def build_command(cfg: dict) -> str:
    """Build a simple one-line pytest command."""
    repro_cfg = normalize_repro_cfg(cfg)
    # TODO: Route to backward-specific test function instead of test_repro
    return f'pytest -vv -s -rA test/python/test_mhas_v2.py::test_repro --repro "{repro_cfg}"'


def build_pretty_command(cfg: dict) -> str:
    """Build a multi-line formatted pytest command."""
    repro_cfg = normalize_repro_cfg(cfg)
    indent = " " * 4
    lines = [
        "pytest -vv -s -rA",
        f"{indent}test/python/test_mhas_v2.py::test_repro",  # TODO: Change to backward test function
        f"{indent}--repro \"",
        f"{indent}{indent}" + "{",
    ]
    items = list(repro_cfg.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        lines.append(f"{indent}{indent}{indent}'{k}': {repr(v)}{comma}")
    lines.append(f"{indent}{indent}" + "}")
    lines.append(f'{indent}"')
    max_len = max(len(line) for line in lines[:-1])
    aligned = [f"{line:<{max_len}} \\" for line in lines[:-1]]
    aligned.append(lines[-1])
    return "\n".join(aligned)


def build_repro_command(raw_line: str, stage1_json: dict) -> str:
    """Stage 2: Build pytest command from stage1 JSON for backward."""
    seed = stage1_json.get("repro_metadata", {}).get("rng_data_seed")
    cfg = stage1.build_cfg(raw_line, stage1_json, seed)
    return build_pretty_command(cfg)
