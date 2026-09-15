# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The per-arch trees of the FROST GEMM engine and the facades that pick one.

Import- and source-level only -- no GPU needed: which tree the facades picked
follows from the GPU (or its absence) the same way on every machine.
"""

import importlib
import os
import subprocess
import sys

import pytest

import cudnn.gemm.frost
from cudnn.gemm.frost import arch_family as AF

pytestmark = pytest.mark.L0


@pytest.mark.parametrize(
    "arch, family",
    [(None, "sm100"), (90, "sm100"), (100, "sm100"), (103, "sm100"), (107, "sm100"), (120, "sm120"), (121, "sm120"), (130, "sm100")],
)
def test_family_for_arch(arch, family):
    assert AF.family_for_arch(arch) == family


def test_select_family_env_override_wins_and_is_validated():
    assert AF.select_family("sm120", 100) == "sm120"
    assert AF.select_family("sm100", 120) == "sm100"
    assert AF.select_family("", 120) == "sm120"
    assert AF.select_family(None, None) == AF.DEFAULT_FAMILY
    with pytest.raises(ValueError, match=AF.FAMILY_ENV):
        AF.select_family("sm90", None)


def test_active_family_follows_the_gpu_or_the_override():
    assert AF.active_family() in AF.FAMILIES
    forced = os.environ.get(AF.FAMILY_ENV, "").strip()
    assert AF.active_family() == (forced or AF.family_for_arch(AF.current_arch()))


def test_facades_are_the_active_family_modules():
    """``cudnn.gemm.frost.compiler`` IS ``cudnn.gemm.frost.<family>.compiler``:
    one object under both names (and on the parent package), and the epilogue
    codegen the compiler imported as its sibling is what the other facade became."""
    fam = AF.active_family()
    import cudnn.gemm.frost.compiler as C
    import cudnn.gemm.frost.epilogue_codegen as E
    from cudnn.gemm.frost import compiler as C2, epilogue_codegen as E2

    assert C is importlib.import_module(f"cudnn.gemm.frost.{fam}.compiler")
    assert E is importlib.import_module(f"cudnn.gemm.frost.{fam}.epilogue_codegen")
    assert C is C2 is cudnn.gemm.frost.compiler is sys.modules["cudnn.gemm.frost.compiler"]
    assert E is E2 is cudnn.gemm.frost.epilogue_codegen is sys.modules["cudnn.gemm.frost.epilogue_codegen"]
    assert C.__name__ == f"cudnn.gemm.frost.{fam}.compiler"
    assert C.generate is E.generate
    assert C._current_arch() == AF.current_arch()


def test_pinning_an_attribute_on_the_facade_reaches_the_module_the_registry_calls(monkeypatch):
    """The reason the facade is module identity and not a copied namespace: the
    ``monkeypatch.setattr(C, "_current_arch", ...)`` idiom of the gemm suite must
    steer ``kernel_registry``'s ``from . import compiler as C`` and the compiler's
    own internal calls alike."""
    import cudnn.gemm.frost.compiler as C
    from cudnn.gemm.frost.kernel_registry import TEMPLATES

    sm120 = next(t for t in TEMPLATES if t.pipeline == "sm120")
    sm100 = next(t for t in TEMPLATES if t.pipeline == "sm100")
    monkeypatch.setattr(C, "_current_arch", lambda: 90)
    assert sm120.arch_active_reject() and sm100.arch_active_reject()  # neither family runs on Hopper
    assert C._mixed_cga_supported() is False  # an internal caller of _current_arch sees the pin too
    monkeypatch.setattr(C, "_current_arch", lambda: 120)
    assert sm120.arch_active_reject() is None and sm100.arch_active_reject()
    assert C._mixed_cga_supported() is True


def test_both_trees_import_side_by_side_and_bind_their_own_sibling():
    """Pinning a tree by name works next to the active one, and each compiler
    copy binds the epilogue codegen of ITS OWN directory."""
    for fam in AF.FAMILIES:
        c = importlib.import_module(f"cudnn.gemm.frost.{fam}.compiler")
        e = importlib.import_module(f"cudnn.gemm.frost.{fam}.epilogue_codegen")
        assert c.__name__ == f"cudnn.gemm.frost.{fam}.compiler"
        assert c.generate is e.generate, fam
    trees = [importlib.import_module(f"cudnn.gemm.frost.{fam}.compiler") for fam in AF.FAMILIES]
    assert len({id(m) for m in trees}) == len(AF.FAMILIES)


def test_every_registered_template_ships_in_its_family_tree():
    from cudnn.gemm.frost.kernel_registry import PIPELINE_FAMILY, TEMPLATES, template_path

    for t in TEMPLATES:
        assert t.path == template_path(t.file)
        assert t.path.is_file(), t.file
        assert t.path.parent == AF.template_dir(PIPELINE_FAMILY[t.pipeline]), t.file
    # the sm*.py inventory on disk is exactly the registry: no strays in either tree
    assert set(AF.template_files()) == {t.path for t in TEMPLATES}
    assert template_path("sm103_block_scale_matmul.py").parent == AF.template_dir("sm100")
    assert template_path("sm120_matmul.py").parent == AF.template_dir("sm120")


def test_shared_template_code_sits_above_the_trees():
    """A module both families' templates import lives at kernel_templates/
    level, not inside one tree (frost/README.md: the directory names the only
    owner); the tcgen05-only helpers live in the sm100 tree and only its
    templates reach for them."""
    assert (AF.FROST_DIR / "kernel_templates" / "split_k_reduction_epilogue_fusion.py").is_file()
    assert (AF.template_dir("sm100") / "_tile_helpers.py").is_file()
    assert not (AF.template_dir("sm120") / "_tile_helpers.py").exists()
    for path in AF.template_files():
        src = path.read_text()
        if "split_k_reduction_epilogue_fusion" in src:
            assert "from cudnn.gemm.frost.kernel_templates.split_k_reduction_epilogue_fusion import" in src, path.name
        if "_tile_helpers" in src:
            assert "from cudnn.gemm.frost.sm100.kernel_templates._tile_helpers import" in src, path.name
            assert path.parent == AF.template_dir("sm100"), path.name


@pytest.mark.parametrize("family", AF.FAMILIES)
def test_env_override_pins_the_facades_in_a_fresh_process(family):
    """The one knob that steers the facades without a GPU of that family."""
    probe = (
        "import cudnn.gemm.frost.arch_family as AF, cudnn.gemm.frost.compiler as C, cudnn.gemm.frost.epilogue_codegen as E; "
        "print(AF.__file__); print(C.__name__); print(E.__name__)"
    )
    env = dict(os.environ, **{AF.FAMILY_ENV: family})
    res = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    lines = res.stdout.splitlines()
    if res.returncode != 0 or lines[:1] != [AF.__file__]:
        pytest.skip(f"a fresh interpreter does not import THIS checkout's cudnn.gemm.frost: {res.stderr.strip()[-300:]}")
    assert lines[1:] == [f"cudnn.gemm.frost.{family}.compiler", f"cudnn.gemm.frost.{family}.epilogue_codegen"]
