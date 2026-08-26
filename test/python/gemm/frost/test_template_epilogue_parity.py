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
# Entries are (template file, cta_group). A merged template serves both MMA
# modes from one file, so it appears once per mode -- the region compared is the
# one left after its @@CTA{1,2}_ONLY@@ blocks resolve.
_PLAIN_1 = [
    ("sm100_matmul.py", 1),
    ("sm100_matmul_mainloop.py", 1),
]
_PLAIN_2 = [
    ("sm100_matmul.py", 2),
    ("sm100_matmul_mainloop.py", 2),
]
_BS_1 = [
    ("sm100_block_scale_matmul.py", 1),
    ("sm103_block_scale_matmul.py", 1),
]
_BS_2 = [
    ("sm100_block_scale_matmul.py", 2),
    ("sm103_block_scale_matmul.py", 2),
]
_MOE_PLAIN_1 = [("sm100_moe_grouped_matmul_fwd.py", 1)]
_MOE_PLAIN_2 = [("sm100_moe_grouped_matmul_fwd.py", 2)]
_MOE_BS_1 = [
    ("sm100_moe_grouped_block_scale_matmul_fwd.py", 1),
]
_MOE_BS_2 = [
    ("sm100_moe_grouped_block_scale_matmul_fwd.py", 2),
]

# sm120 is warp-scoped MMA: the accumulators are already in registers, so there
# is no LDTM shape, no TMEM row base and no span list to share -- and its store
# is transposed-STG only. It shares no epilogue region with the tcgen05
# families, so it is its own family here rather than a member of one.
_SM120 = [("sm120_matmul.py", 1)]

# SETUP (LDTM shape + row base + span list) depends on the DRAIN LAYOUT, which
# the compiler hands down as `epi_packed_lanes` / `epi_dp22` -- not on the MMA
# mode. The plain pipeline is on that form, so it has ONE setup for both modes.
# MoE has not been moved onto it yet and keeps a group per mode; block-scale has
# no hardware-M=64 arm at all (validate_block_scale_config rejects
# mma_inst_m % 128 != 0), so its two modes never differed here.
_SETUP_GROUPS = {
    "plain": _PLAIN_1 + _PLAIN_2,
    "moe_1ctamma": _MOE_PLAIN_1,
    "moe_2ctamma": _MOE_PLAIN_2,
    "block_scale_1ctamma": _BS_1 + _MOE_BS_1,
    "block_scale_2ctamma": _BS_2 + _MOE_BS_2,
    "sm120": _SM120,
}

# DRAIN additionally splits on MoE: no TMA-store half, no mixed CGA, and the
# store is bounded by the routed group rather than by M.
_DRAIN_GROUPS = {
    "plain": _PLAIN_1 + _PLAIN_2,
    "block_scale_1ctamma": _BS_1,
    "block_scale_2ctamma": _BS_2,
    "moe_1ctamma": _MOE_PLAIN_1,
    "moe_2ctamma": _MOE_PLAIN_2,
    "moe_block_scale_1ctamma": _MOE_BS_1,
    "moe_block_scale_2ctamma": _MOE_BS_2,
    "sm120": _SM120,
}

_BLOCK_SCALE = {f for f, _ in _BS_1 + _BS_2 + _MOE_BS_1 + _MOE_BS_2}


def _template_dir():
    # kernel_templates has no __init__.py (it is exec'd per render), so go
    # through the package that does.
    return pathlib.Path(cudnn.gemm.frost.__file__).parent / "kernel_templates"


def _templates():
    return sorted(p for p in _template_dir().glob("sm*.py"))


_IMPLIED_GROUP = re.compile(r"_([12])ctamma\.py$")


def _template_variants():
    """(path, cta_group, source) for every MMA mode a template serves.

    A merged template renders one kernel per mode, so a per-template invariant
    has to hold for each of them separately."""
    for path in _templates():
        src = path.read_text()
        m = _IMPLIED_GROUP.search(path.name)
        if m is not None:
            groups = (int(m.group(1)),)
        else:
            # A merged template says so itself: it switches on cta_group.
            groups = (1, 2) if "cutlass.const_expr(cta_group " in src else (1,)
        for cta_group in groups:
            yield path, cta_group, src


def _region(path, name, cta_group=1):
    """The marked region's text, dedented (MoE nests one level deeper).

    ``cta_group`` selects which mode a merged template is compared as; the text
    itself is mode-agnostic, so it only picks the group the region belongs to."""
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
    declared = [entry for names in groups.values() for entry in names]
    assert len(declared) == len(set(declared)), f"{region}: a (template, cta_group) is in two groups"
    on_disk = {p.name for p in _templates()}
    declared = {f for f, _ in declared}
    assert declared == on_disk, (
        f"{region} group table is out of date -- declare the group of every template.\n"
        f"  missing from the table: {sorted(on_disk - declared)}\n"
        f"  no longer on disk:      {sorted(declared - on_disk)}"
    )


@pytest.mark.parametrize(
    "region,group",
    [("SETUP", g) for g in _SETUP_GROUPS] + [("DRAIN", g) for g in _DRAIN_GROUPS],
)
def test_the_region_is_identical_within_its_group(region, group):
    names = (_SETUP_GROUPS if region == "SETUP" else _DRAIN_GROUPS)[group]
    if {n for n, _ in names} <= _STANDALONE:
        pytest.skip(f"{group} is a family of one -- it shares no region to compare")
    d = _template_dir()
    ref_name, ref_group = names[0]
    ref = _region(d / ref_name, region, ref_group)
    assert ref.strip(), f"{ref_name}: empty {region} region"
    for name, cta_group in names[1:]:
        got = _region(d / name, region, cta_group)
        assert got == ref, (
            f"{region} region of {name} (cta_group={cta_group}) has drifted from its group '{group}'.\n"
            f"A new epilogue feature lands in EVERY template of the group, in the same shape.\n" + _diff(ref_name, ref, name, got)
        )


def test_the_packed_ldtm_arm_keys_on_the_drain_layout():
    """foot-gun #18: a 2-CTA cluster-MMA m=128 tile also has
    epi_rows_per_mma_m == 64, but must NOT take the packed 16x32bx2 path --
    keying the LDTM shape on it is a silent miscompute, not a fault.

    Two spellings are correct: `epi_packed_lanes`, the condition computed once
    host-side (compiler._epi_packed_lanes = mma_inst_m == 64 and cta_group == 1),
    and `mma_inst_shape_mnk[0] == 64`, the hardware M -- equivalent in a template
    that serves one mode. `epi_rows_per_mma_m` is the per-CTA M and is the one
    that must never appear here."""
    ok = ("epi_packed_lanes", "mma_inst_shape_mnk[0] == 64")
    offenders = []
    for path in _templates():
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if "SHAPE_16X32BX2" not in line:
                continue
            window = lines[max(0, i - 3) : i]
            if not any(tok in w for w in window for tok in ok):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, "packed LDTM shape guarded by neither epi_packed_lanes nor the hardware MMA M:\n  " + "\n  ".join(offenders)


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
    """It briefly named two different quantities: `mma_size_m * epi_cols_per_mma_m`
    in the plain pipeline and `epi_cols_per_mma_m` in block-scale."""
    bad_def, bad_use = [], []
    for path in _templates():
        src = path.read_text()
        if "cols_per_acc_stage" not in src:
            continue
        if path.name in _BLOCK_SCALE:
            bad_use.append(path.name)
            continue
        if "cols_per_acc_stage = mma_size_m * epi_cols_per_mma_m" not in src:
            bad_def.append(path.name)
    assert not bad_use, f"block-scale means epi_cols_per_mma_m -- say so: {bad_use}"
    assert not bad_def, f"cols_per_acc_stage must be mma_size_m * epi_cols_per_mma_m: {bad_def}"


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


_MIXED_CGA = {f for f, _ in _PLAIN_1 + _PLAIN_2 + _BS_1 + _BS_2}
_MOE = {f for f, _ in _MOE_PLAIN_1 + _MOE_PLAIN_2 + _MOE_BS_1 + _MOE_BS_2}
_STANDALONE = {f for f, _ in _SM120}


def _call_endswith(node, suffix):
    return isinstance(node, ast.Call) and ast.unparse(node.func).endswith(suffix)


def _assignment_names(node):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return set()
    return {name.id for target in targets for name in ast.walk(target) if isinstance(name, ast.Name)}


def test_l2_identity_fastpath_is_compile_time_and_used_by_every_mixed_cga_call():
    """A pinned width of one is the identity raster.  Keep the general
    divide/modulo mapping out of every hot path in that specialization."""

    helper_tree = ast.parse((_template_dir() / "_tile_helpers.py").read_text())
    helper = next(node for node in helper_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "l2_swizzle_tile")
    assert helper.args.args[-1].arg == "identity"
    assert len(helper.args.defaults) >= 1 and isinstance(helper.args.defaults[-1], ast.Constant) and helper.args.defaults[-1].value is False

    body = helper.body[1:] if helper.body and isinstance(helper.body[0], ast.Expr) and isinstance(helper.body[0].value, ast.Constant) else helper.body
    assert body and isinstance(body[0], ast.If), "identity guard must be the first executable helper statement"
    assert ast.dump(body[0].test, include_attributes=False) == _expr_dump("cutlass.const_expr(identity)")
    assert len(body[0].body) == 1 and isinstance(body[0].body[0], ast.Return)
    assert ast.dump(body[0].body[0].value, include_attributes=False) == _expr_dump("(raw_m, raw_n)")

    expected_identity = _expr_dump("tile_swizzle_n == 1")
    offenders = []
    for path, cta_group, src in _template_variants():
        tree = ast.parse(src)
        calls = [node for node in ast.walk(tree) if _call_endswith(node, "_l2_swizzle_tile")]
        if path.name in _STANDALONE:
            continue  # its own in-file raster; no mixed CGA to specialize for
        if path.name not in _MIXED_CGA:
            if calls:
                offenders.append(f"{path.name}: MoE must keep its separate swizzle path")
            continue

        expected_count = 4 if path.name.startswith("sm103_") else 3
        if len(calls) != expected_count:
            offenders.append(f"{path.name} (cta_group={cta_group}): expected {expected_count} L2-swizzle calls, found {len(calls)}")
        for call in calls:
            identities = [keyword.value for keyword in call.keywords if keyword.arg == "identity"]
            if len(identities) != 1 or ast.dump(identities[0], include_attributes=False) != expected_identity:
                offenders.append(f"{path.name}:{call.lineno}: identity must be tile_swizzle_n == 1")

    assert not offenders, "L2 identity fastpath is incomplete:\n  " + "\n  ".join(offenders)


def test_every_template_hoists_its_complete_smem_descriptor_inventory():
    """Descriptor metadata is invariant; only its byte address changes in the
    tile/K loops.  Pin both the full template inventory and the root shape so
    deleting a build cannot make the no-build-in-a-loop check pass vacuously."""

    assert {path.name for path in _templates()} == _MIXED_CGA | _MOE | _STANDALONE
    total_builds = 0
    total_expected = 0
    offenders = []
    for path, _cta_group, src in _template_variants():
        tree = ast.parse(src)
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        builds = [node for node in ast.walk(tree) if _call_endswith(node, "Tcgen05SmemDesc.build")]
        total_builds += len(builds)
        if path.name in _STANDALONE:
            # warp-scoped MMA reads SMEM through ldmatrix, not an SMEM descriptor.
            expected_bases = []
        elif path.name in _BLOCK_SCALE:
            expected_bases = sorted(("smem_a_list[i]", "smem_b_list[j]", "smem_sfa_list[i]", "smem_sfb_list[j]"))
        elif "mainloop" in path.name:
            expected_bases = ["smem_a", "smem_b"]
        else:
            expected_bases = ["smem_a_list[i]", "smem_b_list[j]"]
        # The MoE ctamma merge left the MMA warp arm-split -- the 2-CTA arm's
        # pair leader carries its own copy of the roots -- so those templates
        # hold one full set per MMA mode. Everything else unified to one.
        total_expected += len(expected_bases) * (2 if path.name in _MOE else 1)

        actual_bases = []
        for build in builds:
            start_address = next((keyword.value for keyword in build.keywords if keyword.arg == "start_address"), None)
            actual_bases.append(ast.unparse(start_address) if start_address is not None else "<missing>")

            cursor = build
            enclosing_assignment = None
            in_runtime_loop = False
            while cursor in parents:
                cursor = parents[cursor]
                in_runtime_loop |= isinstance(cursor, (ast.For, ast.AsyncFor, ast.While))
                if enclosing_assignment is None and isinstance(cursor, (ast.Assign, ast.AnnAssign)):
                    enclosing_assignment = cursor
            target_names = _assignment_names(enclosing_assignment)
            if in_runtime_loop or not target_names or not all("root" in name for name in target_names):
                offenders.append(f"{path.name}:{build.lineno}")

        if path.name in _MOE:
            expected_bases = sorted(expected_bases * 2)
        if sorted(actual_bases) != expected_bases:
            offenders.append(f"{path.name}: descriptor bases {sorted(actual_bases)!r}, expected {expected_bases!r}")

        if "sm103_" in path.name:
            expected_root_stages = {
                "desc_a_roots": {"slot_stages"},
                "desc_b_roots": {"slot_stages"},
                "desc_sfa_roots": {"mma_sf_stage"},
                "desc_sfb_roots": {"mma_sf_stage"},
            }
        elif path.name in _STANDALONE:
            expected_root_stages = {}
        else:
            expected_root_stages = {
                ("desc_a_root" if "mainloop" in path.name else "desc_a_roots"): {"stage"},
                ("desc_b_root" if "mainloop" in path.name else "desc_b_roots"): {"stage"},
            }
            if path.name in _BLOCK_SCALE:
                expected_root_stages.update({"desc_sfa_roots": {"stage"}, "desc_sfb_roots": {"stage"}})

        advance_calls = [
            node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "advance_start_address"
        ]
        for root_name, stage_names in expected_root_stages.items():
            advances_stage_in_loop = False
            for call in advance_calls:
                receiver_names = {node.id for node in ast.walk(call.func.value) if isinstance(node, ast.Name)}
                argument_names = {node.id for argument in call.args for node in ast.walk(argument) if isinstance(node, ast.Name)}
                cursor = call
                in_runtime_loop = False
                while cursor in parents:
                    cursor = parents[cursor]
                    in_runtime_loop |= isinstance(cursor, (ast.For, ast.AsyncFor, ast.While))
                if root_name in receiver_names and argument_names & stage_names and in_runtime_loop:
                    advances_stage_in_loop = True
                    break
            if not advances_stage_in_loop:
                offenders.append(f"{path.name}: {root_name} never advances its stage address in a runtime loop")

        if "sm103_" in path.name:
            assignment_values = {
                name: assignment.value
                for assignment in ast.walk(tree)
                if isinstance(assignment, (ast.Assign, ast.AnnAssign))
                for name in _assignment_names(assignment)
                if name
                in {
                    "desc_a_slots",
                    "desc_b_slots",
                    "desc_a_circ",
                    "desc_b_circ",
                    "desc_a_next",
                    "desc_b_next",
                }
            }
            for operand in ("a", "b"):
                slots = assignment_values.get(f"desc_{operand}_slots")
                circ = assignment_values.get(f"desc_{operand}_circ")
                next_bits = assignment_values.get(f"desc_{operand}_next")
                if slots is None or {f"desc_{operand}_roots", "slot_stages"} - {node.id for node in ast.walk(slots) if isinstance(node, ast.Name)}:
                    offenders.append(f"{path.name}: SM103 {operand.upper()} slots do not derive from staged descriptor roots")
                if circ is None or not any(_call_endswith(node, "_sm103_circular_mma_desc_base") for node in ast.walk(circ)):
                    offenders.append(f"{path.name}: SM103 {operand.upper()} circular-base chain is missing")
                if next_bits is None or not any(_call_endswith(node, "_sm103_circular_mma_next_bits") for node in ast.walk(next_bits)):
                    offenders.append(f"{path.name}: SM103 {operand.upper()} circular-next chain is missing")

            made_operands = set()
            for call in (node for node in ast.walk(tree) if _call_endswith(node, "_sm103_make_circular_mma_desc")):
                if len(call.args) != 3:
                    offenders.append(f"{path.name}:{call.lineno}: malformed SM103 circular descriptor construction")
                    continue
                current_names = {node.id for node in ast.walk(call.args[0]) if isinstance(node, ast.Name)}
                next_names = {node.id for node in ast.walk(call.args[2]) if isinstance(node, ast.Name)}
                for operand in ("a", "b"):
                    if f"desc_{operand}_circ" in current_names and f"desc_{operand}_next" in next_names:
                        made_operands.add(operand)
            if made_operands != {"a", "b"}:
                offenders.append(f"{path.name}: SM103 current/next circular descriptors do not feed both MMA operands")

    assert total_builds == total_expected, f"descriptor inventory changed: the per-template bases expect {total_expected} builds, found {total_builds}"
    assert not offenders, "build invariant descriptor roots once, outside every runtime loop:\n  " + "\n  ".join(offenders)


def _expr_dump(source):
    return ast.dump(ast.parse(source, mode="eval").body, include_attributes=False)


def _raises_not_implemented(statement):
    return any(
        isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name) and node.exc.func.id == "NotImplementedError"
        for node in ast.walk(statement)
    )


def _parents(root):
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _assignments_to(root, target):
    return [node for node in ast.walk(root) if isinstance(node, (ast.Assign, ast.AnnAssign)) and target in _assignment_names(node)]


_LIVE_FALLBACK_MISMATCH = _expr_dump("(cluster_m != cluster_shape_mnk[0]) | (cluster_n != cluster_shape_mnk[1])")


def _is_in_live_fallback_body(node, parents):
    """Whether *node* is in the true arm of the live cluster-shape test."""

    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If) and ast.dump(parent.test, include_attributes=False) == _LIVE_FALLBACK_MISMATCH:
            return child in parent.body
        child = parent
    return False


def _check_preferred_fallback_pair(
    *,
    path,
    kernel,
    parents,
    target,
    preferred,
    fallback,
    site,
    offenders,
    before_lineno=None,
):
    """Validate one source-level preferred/fallback value selection."""

    assignments = _assignments_to(kernel, target)
    preferred_dump = _expr_dump(preferred)
    fallback_dump = _expr_dump(fallback)
    preferred_assignments = [node for node in assignments if ast.dump(node.value, include_attributes=False) == preferred_dump]
    fallback_assignments = [node for node in assignments if ast.dump(node.value, include_attributes=False) == fallback_dump]

    if len(assignments) != 2 or len(preferred_assignments) != 1 or len(fallback_assignments) != 1:
        got = [f"{node.lineno}: {ast.unparse(node.value)}" for node in assignments]
        offenders.append(f"{path.name}: {site} must define {target} exactly once for preferred and fallback; got {got}")
        return

    preferred_assignment = preferred_assignments[0]
    fallback_assignment = fallback_assignments[0]
    if _is_in_live_fallback_body(preferred_assignment, parents):
        offenders.append(f"{path.name}:{preferred_assignment.lineno}: {site} preferred definition is inside fallback path")
    if not _is_in_live_fallback_body(fallback_assignment, parents):
        offenders.append(f"{path.name}:{fallback_assignment.lineno}: {site} fallback definition is not in the live mismatch true arm")
    if before_lineno is not None and (preferred_assignment.lineno >= before_lineno or fallback_assignment.lineno >= before_lineno):
        offenders.append(f"{path.name}:{before_lineno}: {site} is used before both shape alternatives are defined")


def test_mixed_cga_uses_host_constant_masks_and_shifts_at_every_use_site():
    """Pin the 10 mixed-CGA templates' source-level lowering contract.

    Preferred and fallback dimensions are guarded positive powers of two, and
    their host-derived masks/shifts are kept distinct through every rank, L2
    coordinate, and multicast-slice use site.  This deliberately makes no claim
    that the backend eliminates the selected shift SSA in PTX or SASS.
    """

    actual_mixed = {path.name for path in _templates() if "fallback_cluster_shape_mnk" in path.read_text()}
    assert actual_mixed == _MIXED_CGA

    preferred_guard = _expr_dump("any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in cluster_shape_mnk[:2])")
    fallback_guard = _expr_dump("fallback_cluster_shape_mnk is not None and any(_d <= 0 or (_d & (_d - 1)) != 0 for _d in fallback_cluster_shape_mnk[:2])")
    module_shifts = {
        "_preferred_cluster_m_shift": "cluster_shape_mnk[0].bit_length() - 1",
        "_preferred_cluster_n_shift": "cluster_shape_mnk[1].bit_length() - 1",
        "_fallback_cluster_m_shift": (
            "_preferred_cluster_m_shift if fallback_cluster_shape_mnk is None " "else fallback_cluster_shape_mnk[0].bit_length() - 1"
        ),
        "_fallback_cluster_n_shift": (
            "_preferred_cluster_n_shift if fallback_cluster_shape_mnk is None " "else fallback_cluster_shape_mnk[1].bit_length() - 1"
        ),
    }
    live_cluster_names = {"cluster_m", "cluster_n", "cluster_size", "cdim_x", "cdim_y"}
    offenders = []
    for path, cta_group, src in _template_variants():
        if path.name not in _MIXED_CGA:
            continue
        tree = ast.parse(src)
        guarded_tests = {
            ast.dump(statement.test, include_attributes=False)
            for statement in tree.body
            if isinstance(statement, ast.If) and _raises_not_implemented(statement)
        }
        if not {preferred_guard, fallback_guard} <= guarded_tests:
            offenders.append(f"{path.name}: missing exact preferred/fallback positive-pow2 guard")

        for target, expected_source in module_shifts.items():
            assignments = [
                statement for statement in tree.body if isinstance(statement, (ast.Assign, ast.AnnAssign)) and target in _assignment_names(statement)
            ]
            if len(assignments) != 1 or ast.dump(assignments[0].value, include_attributes=False) != _expr_dump(expected_source):
                got = [ast.unparse(statement.value) for statement in assignments]
                offenders.append(f"{path.name}: module-level {target} has wrong bit_length()-1 provenance: {got}")

        kernel = next(
            (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_kernel"),
            None,
        )
        if kernel is None:
            offenders.append(f"{path.name}: missing _kernel")
            continue
        parents = _parents(kernel)

        for old_shift in ("cluster_m_shift", "cluster_n_shift"):
            assignments = _assignments_to(kernel, old_shift)
            if assignments:
                offenders.append(f"{path.name}: _kernel still assigns runtime-selected {old_shift} at " f"{[node.lineno for node in assignments]}")

        _check_preferred_fallback_pair(
            path=path,
            kernel=kernel,
            parents=parents,
            target="m_rank",
            preferred="cta_rank_in_cluster & (cluster_shape_mnk[0] - 1)",
            fallback="cta_rank_in_cluster & (fallback_cluster_shape_mnk[0] - 1)",
            site="M rank",
            offenders=offenders,
        )
        _check_preferred_fallback_pair(
            path=path,
            kernel=kernel,
            parents=parents,
            target="n_rank",
            preferred="cta_rank_in_cluster >> _preferred_cluster_m_shift",
            fallback="cta_rank_in_cluster >> _fallback_cluster_m_shift",
            site="N rank",
            offenders=offenders,
        )

        l2_calls = [node for node in ast.walk(kernel) if _call_endswith(node, "_l2_swizzle_tile")]
        coordinate_specs = (
            (0, "raw M", "m"),
            (1, "raw N", "n"),
            (2, "grid M", "m"),
            (3, "grid N", "n"),
        )
        for call in l2_calls:
            if len(call.args) < 4:
                offenders.append(f"{path.name}:{call.lineno}: L2 call has fewer than four coordinates")
                continue
            for index, coordinate, axis in coordinate_specs:
                argument = call.args[index]
                expected_suffix = ("raw" if index < 2 else "nt") + f"_{axis}"
                if not isinstance(argument, ast.Name) or not argument.id.endswith(expected_suffix):
                    offenders.append(f"{path.name}:{call.lineno}: L2 {coordinate} must be a named preferred/fallback value, got {ast.unparse(argument)}")
                    continue
                target = argument.id
                if index < 2:
                    if target.startswith("init_"):
                        source = "bidx" if axis == "m" else "bidy"
                    else:
                        source = "m_idx" if axis == "m" else "n_idx"
                else:
                    source = "gridx" if axis == "m" else "gridy"
                _check_preferred_fallback_pair(
                    path=path,
                    kernel=kernel,
                    parents=parents,
                    target=target,
                    preferred=f"{source} >> _preferred_cluster_{axis}_shift",
                    fallback=f"{source} >> _fallback_cluster_{axis}_shift",
                    site=f"L2 {coordinate} at line {call.lineno}",
                    offenders=offenders,
                    before_lineno=call.lineno,
                )

        _check_preferred_fallback_pair(
            path=path,
            kernel=kernel,
            parents=parents,
            target="_a_per_cta",
            preferred="a_mcast_slices >> _preferred_cluster_n_shift",
            fallback="a_mcast_slices >> _fallback_cluster_n_shift",
            site="A multicast slices",
            offenders=offenders,
        )
        # The MMA pair consumes one factor of two of cluster_m, so B's slice count
        # shifts one less. A template that serves both modes spells that as the
        # host constant; a single-mode one folds it to a literal.
        if "_cta_group_shift" in src:
            b_shift_adjust = " - _cta_group_shift"
        else:
            b_shift_adjust = " - 1" if cta_group == 2 else ""
        _check_preferred_fallback_pair(
            path=path,
            kernel=kernel,
            parents=parents,
            target="_b_per_cta",
            preferred=f"b_mcast_slices >> (_preferred_cluster_m_shift{b_shift_adjust})",
            fallback=f"b_mcast_slices >> (_fallback_cluster_m_shift{b_shift_adjust})",
            site="B multicast slices",
            offenders=offenders,
        )

        for operation in (node for node in ast.walk(kernel) if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod))):
            denominator_names = {node.id for node in ast.walk(operation.right) if isinstance(node, ast.Name)}
            if denominator_names & live_cluster_names:
                offenders.append(f"{path.name}:{operation.lineno}: runtime cluster divisor in {ast.unparse(operation)!r}")

    assert not offenders, "mixed-CGA host-constant mask/shift fast path is incomplete:\n  " + "\n  ".join(offenders)
