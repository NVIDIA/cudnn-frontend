# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the backward SASS probe's analysis without compiling a kernel."""

import ast
from pathlib import Path
import re
import textwrap

import pytest

pytestmark = pytest.mark.L0


def _analyze_sass(sass):
    # The probe is subprocess source. Load that string without importing the GPU test module,
    # then execute its analysis with synthetic nvdisasm output instead of compiling a kernel.
    path = Path(__file__).with_name("test_sdpa_bwd_dsl_sm107.py")
    tree = ast.parse(path.read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_SASS_PROBE" for t in n.targets))
    probe = eval(compile(ast.Expression(assignment.value), str(path), "eval"), {"textwrap": textwrap})
    analysis = probe[probe.index("def cnt(") :]
    exec(compile(analysis, str(path), "exec"), {"sass": sass, "re": re})


@pytest.mark.parametrize("order", [("main",), ("main", "setup"), ("setup", "main")], ids=["main-only", "main-first", "setup-first"])
def test_main_kernel_counts_do_not_depend_on_section_order(order, capsys):
    main = [
        '    .section .text.cudnn_kernel__kernel_example,"ax",@progbits',
        "    /*0000*/ USETMAXREG.DEALLOC 0x38;",
        "    /*0010*/ USETMAXREG.ALLOC 0xe0;",
        "    /*0020*/ STL [R1], R2;",
        "    /*0030*/ LDL R2, [R1];",
    ]
    setup = [
        '    .section .text.cudnn_kernel__thd_setup_kernel_example,"ax",@progbits',
        "    /*0000*/ MEMBAR.ALL.GPU;",
        "    /*0010*/ CGAERRBAR;",
    ]
    # Unrelated code and non-code sections must not contribute to main or setup counts.
    unrelated = ['    .section .text.other_kernel,"ax",@progbits', "    /*0000*/ USETMAXREG.ALLOC 0x40;"]
    data = ['    .section .nv.constant0.cudnn_kernel__kernel_example,"a",@progbits', "    // USETMAXREG STL LDL MEMBAR.ALL.GPU CGAERRBAR"]
    sections = {"main": main + data, "setup": setup + data}
    sass = ["// disassembly header", "// .text.setup_kernel is not a section declaration"] + unrelated
    for name in order:
        sass.extend(sections[name])
    sass.extend(unrelated)
    _analyze_sass(sass)
    stats = {line.split()[1]: int(line.split()[2]) for line in capsys.readouterr().out.splitlines() if line.startswith("SASS ")}
    assert {key: stats[key] for key in ("USETMAXREG", "STL", "LDL", "MEMBAR_GPU", "CGAERRBAR")} == {
        "USETMAXREG": 2,
        "STL": 1,
        "LDL": 1,
        "MEMBAR_GPU": 0,
        "CGAERRBAR": 0,
    }
    assert stats["SETUP_KERNEL_LINES"] == (len(setup) if "setup" in order else 0)


@pytest.mark.parametrize(
    "sass", [[], ['    .section .text.cudnn_kernel__thd_setup_kernel_example,"ax",@progbits'], ['    .section .text.other_kernel,"ax",@progbits']]
)
def test_missing_main_kernel_section_fails_explicitly(sass):
    with pytest.raises(AssertionError, match="No main kernel .text section"):
        _analyze_sass(sass)
