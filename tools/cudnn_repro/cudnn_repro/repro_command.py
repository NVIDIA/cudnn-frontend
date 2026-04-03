"""Shared helpers for rendering pytest repro commands."""


def normalize_repro_cfg(cfg: dict) -> dict:
    """Normalize config values for command generation."""
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
    return f'pytest -vv -s -rA test/python/test_mhas_v2.py::test_repro --repro "{repro_cfg}"'


def build_pretty_command(cfg: dict) -> str:
    """Build a multi-line formatted pytest command."""
    repro_cfg = normalize_repro_cfg(cfg)
    indent = " " * 4
    lines = [
        "pytest -vv -s -rA",
        f"{indent}test/python/test_mhas_v2.py::test_repro",
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
