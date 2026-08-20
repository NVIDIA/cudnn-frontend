"""The epilogue drain is ONE logic across all 20 kernel templates.

Source-level only -- no GPU, no render, no JIT. See CLAUDE.md
"The epilogue is ONE logic across all 20 templates".
"""

import ast
import difflib
import pathlib
import re
import textwrap

import pytest

import cudnn.gemm.frost

pytestmark = pytest.mark.L0

_MARKER = re.compile(r"^[ \t]*# *@@EPILOGUE_(SETUP|DRAIN):(BEGIN|END)@@[ \t]*$")

# Groups the region must be identical within. A template must appear in exactly
# one group per region; adding a template makes the completeness test fail until
# its group is declared here.
_PLAIN_1 = [
    "sm100_matmul_1ctamma.py",
    "sm100_matmul_1ctamma_static.py",
    "sm100_matmul_mainloop_1ctamma.py",
]
_PLAIN_2 = [
    "sm100_matmul_2ctamma.py",
    "sm100_matmul_2ctamma_static.py",
    "sm100_matmul_mainloop_2ctamma.py",
]
_BS_1 = [
    "sm100_block_scale_matmul_1ctamma.py",
    "sm100_block_scale_matmul_1ctamma_static.py",
    "sm103_block_scale_matmul_1ctamma.py",
    "sm107_block_scale_matmul_1ctamma.py",
]
_BS_2 = [
    "sm100_block_scale_matmul_2ctamma.py",
    "sm100_block_scale_matmul_2ctamma_static.py",
    "sm103_block_scale_matmul_2ctamma.py",
    "sm107_block_scale_matmul_2ctamma.py",
]
_MOE_PLAIN_1 = ["sm100_moe_grouped_matmul_fwd_1ctamma.py"]
_MOE_PLAIN_2 = ["sm100_moe_grouped_matmul_fwd_2ctamma.py"]
_MOE_BS_1 = [
    "sm100_moe_grouped_block_scale_matmul_fwd_1ctamma.py",
    "sm107_moe_grouped_block_scale_matmul_fwd_1ctamma.py",
]
_MOE_BS_2 = [
    "sm100_moe_grouped_block_scale_matmul_fwd_2ctamma.py",
    "sm107_moe_grouped_block_scale_matmul_fwd_2ctamma.py",
]

# SETUP (LDTM shape + row base + span list) depends only on cta_group and on
# whether the pipeline is block-scaled, so MoE joins its family.
_SETUP_GROUPS = {
    "1ctamma": _PLAIN_1 + _MOE_PLAIN_1,
    "2ctamma": _PLAIN_2 + _MOE_PLAIN_2,
    "block_scale_1ctamma": _BS_1 + _MOE_BS_1,
    "block_scale_2ctamma": _BS_2 + _MOE_BS_2,
}

# DRAIN additionally splits on MoE: no TMA-store half, no mixed CGA, and the
# store is bounded by the routed group rather than by M.
_DRAIN_GROUPS = {
    "1ctamma": _PLAIN_1,
    "2ctamma": _PLAIN_2,
    "block_scale_1ctamma": _BS_1,
    "block_scale_2ctamma": _BS_2,
    "moe_1ctamma": _MOE_PLAIN_1,
    "moe_2ctamma": _MOE_PLAIN_2,
    "moe_block_scale_1ctamma": _MOE_BS_1,
    "moe_block_scale_2ctamma": _MOE_BS_2,
}

_BLOCK_SCALE = set(_BS_1 + _BS_2 + _MOE_BS_1 + _MOE_BS_2)


def _template_dir():
    # kernel_templates has no __init__.py (it is exec'd per render), so go
    # through the package that does.
    return pathlib.Path(cudnn.gemm.frost.__file__).parent / "kernel_templates"


def _templates():
    return sorted(p for p in _template_dir().glob("sm*.py"))


def _region(path, name):
    """The marked region's text, dedented (MoE nests one level deeper)."""
    src = path.read_text().splitlines(keepends=True)
    begin = end = None
    for i, line in enumerate(src):
        m = _MARKER.match(line.rstrip("\n"))
        if not m or m.group(1) != name:
            continue
        if m.group(2) == "BEGIN":
            assert begin is None, f"{path.name}: duplicate {name}:BEGIN"
            begin = i
        else:
            assert begin is not None, f"{path.name}: {name}:END before BEGIN"
            assert end is None, f"{path.name}: duplicate {name}:END"
            end = i
    assert begin is not None and end is not None, f"{path.name}: missing @@EPILOGUE_{name}@@ markers"
    return textwrap.dedent("".join(src[begin + 1 : end]))


def _diff(name_a, text_a, name_b, text_b):
    return "\n".join(difflib.unified_diff(text_a.splitlines(), text_b.splitlines(), fromfile=name_a, tofile=name_b, lineterm=""))


@pytest.mark.parametrize("region,groups", [("SETUP", _SETUP_GROUPS), ("DRAIN", _DRAIN_GROUPS)])
def test_every_template_is_assigned_to_exactly_one_group(region, groups):
    declared = [f for names in groups.values() for f in names]
    assert len(declared) == len(set(declared)), f"{region}: a template is in two groups"
    on_disk = {p.name for p in _templates()}
    assert set(declared) == on_disk, (
        f"{region} group table is out of date -- declare the group of every template.\n"
        f"  missing from the table: {sorted(on_disk - set(declared))}\n"
        f"  no longer on disk:      {sorted(set(declared) - on_disk)}"
    )


@pytest.mark.parametrize(
    "region,group",
    [("SETUP", g) for g in _SETUP_GROUPS] + [("DRAIN", g) for g in _DRAIN_GROUPS],
)
def test_the_region_is_identical_within_its_group(region, group):
    names = (_SETUP_GROUPS if region == "SETUP" else _DRAIN_GROUPS)[group]
    d = _template_dir()
    ref_name = names[0]
    ref = _region(d / ref_name, region)
    assert ref.strip(), f"{ref_name}: empty {region} region"
    for name in names[1:]:
        got = _region(d / name, region)
        assert got == ref, (
            f"{region} region of {name} has drifted from its group '{group}'.\n"
            f"A new epilogue feature lands in EVERY template of the group, in the same shape.\n" + _diff(ref_name, ref, name, got)
        )


def test_the_packed_ldtm_arm_keys_on_the_hardware_mma_m():
    """foot-gun #18: a 2-CTA cluster-MMA m=128 tile also has
    epi_rows_per_mma_m == 64, but must NOT take the packed 16x32bx2 path --
    keying the LDTM shape on it is a silent miscompute, not a fault."""
    offenders = []
    for path in _templates():
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "SHAPE_16X32BX2" not in line:
                continue
            window = path.read_text().splitlines()[max(0, i - 3) : i]
            if not any("mma_inst_shape_mnk[0] == 64" in w for w in window):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, "packed LDTM shape not guarded by the HARDWARE MMA M:\n  " + "\n  ".join(offenders)


def test_block_scale_templates_carry_no_m64_path():
    """tile_config.validate_block_scale_config rejects mma_inst_m % 128 != 0, so
    the hardware-M=64 arms are provably dead there and are deleted, not gated."""
    offenders = [f"{p.name}: {tok}" for p in _templates() if p.name in _BLOCK_SCALE for tok in ("SHAPE_16X32BX2", "ld_half_off") if tok in p.read_text()]
    assert not offenders, "block-scale forbids mma_inst_m=64 -- drop the dead arm:\n  " + "\n  ".join(offenders)


def test_the_retired_fixed_width_drain_name_is_gone():
    """The drain width comes from `_epi_subtile_spans`; `t2r_inst_repx` was the
    fixed-width spelling it replaced."""
    offenders = [p.name for p in _templates() if "t2r_inst_repx" in p.read_text()]
    assert not offenders, f"use epi_spans / subtile_w, not t2r_inst_repx: {offenders}"


def test_cols_per_acc_stage_has_one_meaning():
    """It briefly named two different quantities: `num_mma_m * epi_cols_per_mma_m`
    in the plain pipeline and `epi_cols_per_mma_m` in block-scale."""
    bad_def, bad_use = [], []
    for path in _templates():
        src = path.read_text()
        if "cols_per_acc_stage" not in src:
            continue
        if path.name in _BLOCK_SCALE:
            bad_use.append(path.name)
            continue
        if "cols_per_acc_stage = num_mma_m * epi_cols_per_mma_m" not in src:
            bad_def.append(path.name)
    assert not bad_use, f"block-scale means epi_cols_per_mma_m -- say so: {bad_use}"
    assert not bad_def, f"cols_per_acc_stage must be num_mma_m * epi_cols_per_mma_m: {bad_def}"


def test_the_overlap_arm_never_indexes_the_span_list():
    """`epi_spans` is a Python list, so it takes a COMPILE-TIME index; under
    acc overlap the subtile index depends on `tile_iter % 2` and is a runtime
    value. That arm computes the column offset arithmetically instead."""
    offenders = []
    for path in _templates():
        src = path.read_text()
        if "_sub = subtile_idx" not in src:
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if "epi_spans[_sub]" in line:
                offenders.append(f"{path.name}:{i}")
    assert not offenders, "runtime index into a Python list:\n  " + "\n  ".join(offenders)


def test_the_markers_do_not_break_the_template_parse():
    for path in _templates():
        ast.parse(path.read_text())
