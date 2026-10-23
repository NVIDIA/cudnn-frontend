# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from tools.doc.build_doc_indexes import (
    render_api_availability,
    render_api_index,
    render_api_sources,
    render_candidate_contract,
    render_sample_index,
    sync_api_availability,
    sync_api_sources,
    write_or_check,
)
from tools.doc.build_doc_manifests import (
    build_sample_manifest,
    cpp_type_definitions,
    cpp_type_learning_anchor,
    discover_optional_namespace_members,
    optional_namespace_learning_anchor,
    python_export_learning_anchor,
)
from tools.doc.contracts import candidate_availability_contract
from tools.doc.check_doc_coverage import (
    CoverageError,
    _anchor_exists,
    _heading_key,
    _markdown_slug,
    candidate_has_documentation_owner,
    documentation_target,
    source_reference_contains,
    source_reference_exists,
    validate_api_manifest,
    validate_published_samples,
)
from tools.doc.run_doc_samples import (
    ManifestError,
    _requirements_met,
    fern_sample_references,
    find_inline_executable_examples,
    load_sample_manifest,
    run_sample,
    scan_markdown_code,
    snippet_language,
)


def api_entry(name: str, page: str) -> dict:
    return {
        "language": "python",
        "qualified_name": name,
        "title": name.rsplit(".", 1)[-1],
        "kind": "method",
        "summary": f"Reference for {name}.",
        "disposition": "Public",
        "stability": "Stable",
        "availability": "cuDNN 9.12 or later",
        "source_declaration": "python/cudnn/_pygraph.py:1",
        "implementation": "python/cudnn/_pygraph.py:1",
        "source_resolution": "exact",
        "source_declaration_token": "add",
        "implementation_token": "add",
        "evidence": ["python/cudnn/_pygraph.py:1"],
        "page": page,
        "learning_anchor": "tutorials/first-graph.md#build-the-graph",
        "counterpart": None,
        "aliases": [],
        "compatibility_id": "core",
        "samples": ["python-matmul"],
        "verification": "source-verified",
    }


def test_api_index_is_sorted_by_qualified_name() -> None:
    entries = [
        api_entry("cudnn.pygraph.tensor", "api/python/tensor.md"),
        api_entry("cudnn.pygraph.add", "api/python/add.md"),
    ]

    rendered = render_api_index("Python API", "python", entries)

    assert rendered.index("`cudnn.pygraph.add`") < rendered.index("`cudnn.pygraph.tensor`")
    assert "[A](#a)" in rendered
    assert "[T](#t)" in rendered


def test_api_index_links_family_owned_candidates_to_learning_anchor() -> None:
    candidate = {
        "language": "cpp",
        "qualified_name": "cudnn_frontend::graph::Tensor_attributes",
        "disposition": "Public",
        "evidence": "include/cudnn_frontend/graph_properties.h:49",
        "learning_anchor": "concepts/graphs-and-tensors.md#tensor-identity",
        "suggestion": "Add a dedicated type page if lookup demand warrants it.",
        "compatibility_id": "core",
        "constraint_status": "documented",
        "constraint_note": "Constraints are stated at the learning owner.",
        "kind": "struct",
        "summary": "Configure tensor attributes.",
        "stability": "Stable",
    }

    feature = {
        "release_status": "Stable",
        "cudnn": "9.12+",
        "cuda": "CUDA 13 tested",
        "gpu": "SM100 tested",
        "platforms": ["Linux (executed)", "Windows (not executed)"],
        "backends": ["cuDNN backend"],
        "languages": ["Python", "C++"],
        "dependencies": ["cuDNN", "CUDA"],
        "verification": "Executed",
        "evidence": ["docs/compatibility.json"],
    }
    candidate["availability_contract"] = candidate_availability_contract(candidate, feature)

    rendered = render_api_index("C++ API", "cpp", [], [candidate])

    assert "[`cudnn_frontend::graph::Tensor_attributes`](/concepts/graphs-and-tensors#tensor-identity)" in rendered
    assert "**cuDNN:** 9.12+" in rendered
    assert "**CUDA:** CUDA 13 tested" in rendered
    assert "**GPU:** SM100 tested" in rendered
    assert "**Language:** C++" in rendered
    assert "**Dependencies:** cuDNN; CUDA" in rendered
    assert render_candidate_contract(candidate) in rendered
    assert "| struct | Configure tensor attributes. | Stable |" in rendered
    assert "Notes to reviewers" not in rendered


def test_api_index_marks_candidate_without_learning_anchor_as_dual_gap() -> None:
    candidate = {
        "language": "cpp",
        "qualified_name": "cudnn_frontend::graph::UnassignedType",
        "disposition": "Public",
        "evidence": "include/example.h:1",
        "learning_anchor": "",
        "suggestion": "Add a type page.",
        "compatibility_id": "core",
        "constraint_status": "documented",
        "constraint_note": "No owner exists yet.",
        "kind": "class",
        "summary": "Use the unassigned type.",
        "stability": "Stable",
    }

    feature = {
        "release_status": "Stable",
        "cudnn": "9.12+",
        "cuda": "CUDA 13 tested",
        "gpu": "SM100 tested",
        "platforms": ["Linux"],
        "backends": ["cuDNN backend"],
        "languages": ["C++"],
        "dependencies": ["cuDNN", "CUDA"],
        "verification": "Source-verified",
        "evidence": ["docs/compatibility.json"],
    }
    candidate["availability_contract"] = candidate_availability_contract(candidate, feature)

    rendered = render_api_index("C++ API", "cpp", [], [candidate])

    assert "No learning page is available." in rendered
    assert "Notes to reviewers" not in rendered


def test_api_availability_block_is_inserted_in_notes_and_is_idempotent() -> None:
    feature = {
        "name": "Core Graph operations",
        "release_status": "Stable",
        "cudnn": "9.12+",
        "cuda": "CUDA 13 tested",
        "gpu": "SM100 tested",
        "platforms": ["Linux (executed)", "Windows (not executed)"],
        "backends": ["cuDNN backend"],
        "languages": ["Python", "C++"],
        "dependencies": ["cuDNN", "CUDA"],
        "verification": "Executed",
    }
    entry = {
        "disposition": "Public",
        "stability": "Stable",
        "language": "python",
        "availability": "cuDNN 9.12+ for this API",
    }
    block = render_api_availability(feature, entry)
    page = "# add\n\n## Notes\n\nHuman-authored details.\n\n## Examples\n"

    updated = sync_api_availability(page, block)

    assert updated.index("## Notes") < updated.index("{/* BEGIN GENERATED API AVAILABILITY */}")
    assert "**API-specific availability:** cuDNN 9.12+ for this API" in updated
    assert "**Family platforms:** Linux (executed); Windows (not executed)" in updated
    assert sync_api_availability(updated, block) == updated


def test_api_source_block_replaces_legacy_links_and_is_idempotent() -> None:
    entry = api_entry("cudnn.pygraph.add", "api/python/add.md")
    entry["source_declaration"] = "python/cudnn/_pygraph.py:517"
    entry["implementation"] = "python/cudnn/_pygraph.py:1374"
    entry["source_declaration_token"] = "add"
    entry["implementation_token"] = "_install_pointwise_builders"
    commit = "8b99eecabb0453e98393e019e7cc60aa65ca5f31"
    block = render_api_sources(entry, commit)
    page = "# add\n\nCurrent sources: [Python graph](old).\n"

    updated = sync_api_sources(page, block)

    assert f"/blob/{commit}/python/cudnn/_pygraph.py#L517" in updated
    assert f"/blob/{commit}/python/cudnn/_pygraph.py#L1374" in updated
    assert sync_api_sources(updated, block) == updated


def test_write_or_check_detects_drift(tmp_path: Path) -> None:
    target = tmp_path / "index.md"
    target.write_text("stale\n", encoding="utf-8")

    assert write_or_check(target, "current\n", check=True) is False
    assert target.read_text(encoding="utf-8") == "stale\n"
    assert write_or_check(target, "current\n", check=False) is True
    assert target.read_text(encoding="utf-8") == "current\n"


def test_sample_index_publishes_runnable_stable_id_command() -> None:
    sample = {
        "id": "cpp-add-basic",
        "path": "samples/cpp/add_basic.cc",
        "language": "cpp",
        "purpose": "Execute addition.",
        "owners": ["api/cpp/add.md"],
        "requirements": ["linux", "cuda", "nvcc"],
        "command": ["nvcc", "{path}", "-o", "{binary}"],
        "fallback_command": ["nvcc", "{path}", "-o", "{binary}"],
        "expected": "exit:0",
        "verification": "executed",
    }

    rendered = render_sample_index([sample])

    assert "from the repository root" in rendered
    assert "python tools/doc/run_doc_samples.py --id cpp-add-basic" in rendered
    assert "{path}" not in rendered
    assert "{binary}" not in rendered


def test_api_manifest_rejects_duplicate_qualified_names(tmp_path: Path) -> None:
    entry = api_entry("cudnn.pygraph.add", "api/python/add.md")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "apis": [entry, entry]}), encoding="utf-8")

    with pytest.raises(CoverageError, match="duplicate qualified_name"):
        validate_api_manifest(manifest, tmp_path)


def test_sample_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    entry = {
        "id": "python-add",
        "path": "samples/python/add.py",
        "language": "python",
        "mode": "execute",
        "owners": ["tutorials/add.md"],
        "apis": ["cudnn.pygraph.add"],
        "regions": ["main"],
        "command": ["python", "{path}"],
        "requirements": ["python"],
        "timeout_seconds": 30,
        "expected": "ok",
        "verification": "pending",
        "purpose": "Exercise addition.",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "samples": [entry, {**entry, "id": "python-add-2"}]}), encoding="utf-8")

    with pytest.raises(ManifestError, match="duplicate sample path"):
        load_sample_manifest(manifest)


def test_sample_manifest_rejects_blank_expected_result(tmp_path: Path) -> None:
    entry = {
        "id": "python-add",
        "path": "samples/python/add.py",
        "language": "python",
        "mode": "execute",
        "owners": ["tutorials/add.md"],
        "apis": ["cudnn.pygraph.add"],
        "regions": ["main"],
        "command": ["python", "{path}"],
        "requirements": ["python"],
        "timeout_seconds": 30,
        "expected": "",
        "verification": "pending",
        "purpose": "Exercise addition.",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "samples": [entry]}), encoding="utf-8")

    with pytest.raises(ManifestError, match="nonblank expected result"):
        load_sample_manifest(manifest)


def test_fusion_samples_list_the_operations_they_compose() -> None:
    samples = {sample["id"]: sample for sample in build_sample_manifest()["samples"]}

    assert samples["python-fusion"]["apis"] == [
        "cudnn.pygraph.matmul",
        "cudnn.pygraph.bias",
        "cudnn.pygraph.relu",
    ]
    assert samples["cpp-fusion"]["apis"] == [
        "cudnn_frontend::graph::Graph::matmul",
        "cudnn_frontend::graph::Graph::pointwise",
    ]


def test_broadcast_add_samples_map_to_add() -> None:
    samples = {sample["id"]: sample for sample in build_sample_manifest()["samples"]}

    assert samples["python-add-broadcast"]["apis"] == ["cudnn.pygraph.add"]
    assert samples["cpp-add-broadcast"]["apis"] == ["cudnn_frontend::graph::Graph::add"]


def test_inline_executable_scan_ignores_synopsis_but_flags_examples(tmp_path: Path) -> None:
    page = tmp_path / "api.md"
    page.write_text(
        """# add

## Python synopsis

```python
graph.add(a, b)
```

## Examples

```python
print(\"maintained inline\")
```
""",
        encoding="utf-8",
    )

    findings = find_inline_executable_examples(page)

    assert findings == [11]


def test_inline_executable_scan_allows_fern_code_component(tmp_path: Path) -> None:
    page = tmp_path / "api.md"
    page.write_text(
        """# add

## Examples

<Code src="../samples/python/add.py" language="python" maxLines={0} />
""",
        encoding="utf-8",
    )

    assert find_inline_executable_examples(page) == []
    assert scan_markdown_code(page)[1] == []


def test_learning_samples_avoid_defensive_scaffolding() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sample_paths = set()
    for section in ("tutorials", "how-to"):
        for page in (repo_root / "docs" / section).glob("*.md"):
            sample_paths.update(f"docs/{path}" for path in fern_sample_references(page))

    forbidden = {
        ".py": (
            "try:",
            "except ",
            "finally:",
            "SKIP:",
            "def run_sample",
            "raise ValueError",
            "torch.cuda.is_available",
        ),
        ".cc": (
            "\n    try {",
            "catch (",
            "SKIP: sample",
            "check_cuda",
            "ManagedBuffer",
            "cudaGetDeviceProperties",
        ),
    }
    for relative_path in sample_paths:
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        for pattern in forbidden.get(Path(relative_path).suffix, ()):
            assert pattern not in source, f"{relative_path} contains defensive scaffolding: {pattern}"


def test_markdown_code_scan_requires_language_labels(tmp_path: Path) -> None:
    page = tmp_path / "guide.md"
    page.write_text(
        """# Guide

```text
descriptive output
```

```
value
```
""",
        encoding="utf-8",
    )

    problems = scan_markdown_code(page)[1]

    assert any("has no language label" in problem for problem in problems)


def test_snippet_language_uses_canonical_fence_labels() -> None:
    assert snippet_language("docs/samples/python/add.py") == "python"
    assert snippet_language("docs/samples/cpp/add.cc:main") == "cpp"
    assert snippet_language("docs/samples/shell/run.sh") == "bash"
    assert snippet_language("docs/samples/cmake/CMakeLists.txt") == "cmake"


def test_language_synopsis_headings_share_the_template_key() -> None:
    assert _heading_key("Python synopsis") == "synopsis"
    assert _heading_key("C++ synopsis") == "synopsis"
    assert _heading_key("Appendix: Python equivalent") == "appendix"


def test_markdown_slug_collapses_punctuation_and_whitespace() -> None:
    assert _markdown_slug("Run the RMSNorm + SiLU Graph fusion") == "run-the-rmsnorm-silu-graph-fusion"


def test_available_command_requirement_is_satisfied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.doc.run_doc_samples.shutil.which", lambda command: f"/usr/bin/{command}")

    assert _requirements_met(["nvcc"]) == (True, "")


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (130500, (True, "")),
        (130201, (False, "requires cublasLt 13.5 or newer")),
    ],
)
def test_cublaslt_version_requirement(version: int, expected: tuple[bool, str], monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCublasLt:
        @staticmethod
        def cublasLtGetVersion() -> int:
            return version

    monkeypatch.setattr("tools.doc.run_doc_samples.ctypes.CDLL", lambda library: FakeCublasLt())

    assert _requirements_met(["cublaslt>=130500"]) == expected


def test_unmet_requirement_runs_declared_static_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.doc.run_doc_samples._requirements_met", lambda requirements: (False, "requires newer backend"))
    sample = {
        "id": "python-fallback",
        "path": "samples/python/fallback.py",
        "mode": "execute",
        "command": ["{python}", "-c", "raise SystemExit(1)"],
        "fallback_command": ["{python}", "-c", "print('compiled')"],
        "requirements": ["cudnn>=99900"],
        "timeout_seconds": 30,
        "expected": "exit:0",
    }

    result = run_sample(sample, tmp_path, require_execution=True)

    assert result["status"] == "compiled"
    assert "requires newer backend" in result["reason"]


def test_require_execution_fails_unmet_requirement_without_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.doc.run_doc_samples._requirements_met", lambda requirements: (False, "requires missing feature"))
    sample = {
        "id": "python-no-fallback",
        "path": "samples/python/no_fallback.py",
        "mode": "execute",
        "command": ["{python}", "-c", "print('unused')"],
        "requirements": ["missing"],
        "timeout_seconds": 30,
        "expected": "exit:0",
    }

    assert run_sample(sample, tmp_path, require_execution=True)["status"] == "failed"


def test_require_execution_rejects_compatible_compile_only_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.doc.run_doc_samples._requirements_met", lambda requirements: (True, ""))
    sample = {
        "id": "cpp-compile-only",
        "path": "samples/cpp/compile_only.cc",
        "mode": "compile",
        "command": ["nvcc", "{path}"],
        "requirements": ["nvcc"],
        "timeout_seconds": 30,
        "expected": "exit:0",
    }

    assert run_sample(sample, tmp_path, require_execution=True)["status"] == "failed"


def test_candidate_learning_anchor_is_valid_family_level_ownership() -> None:
    assert candidate_has_documentation_owner({"learning_anchor": "how-to/legacy-migration.md#map-the-construction-model"})
    assert not candidate_has_documentation_owner({"learning_anchor": ""})


def test_learning_anchor_accepts_an_explicit_markdown_heading_id(tmp_path: Path) -> None:
    (tmp_path / "tutorial.md").write_text(
        "# Tutorial\n\n## Build the expression\n",
        encoding="utf-8",
    )

    assert _anchor_exists(tmp_path, "tutorial.md#build-the-expression")


def test_device_properties_is_not_misclassified_as_rope() -> None:
    assert cpp_type_learning_anchor("DeviceProperties") == "concepts/compilation-caching.md#cache-identity"


def test_cpp_type_definitions_preserve_namespace_and_skip_forward_declarations(tmp_path: Path) -> None:
    header = tmp_path / "types.h"
    header.write_text(
        "namespace cudnn_frontend {\n"
        "class Forward;\n"
        "class KernelCache { public: int value; };\n"
        "namespace graph {\n"
        "class Graph;\n"
        "struct MatmulNode { int value; };\n"
        "}\n"
        "}\n",
        encoding="utf-8",
    )

    assert cpp_type_definitions(header) == [
        ("cudnn_frontend::KernelCache", 3),
        ("cudnn_frontend::graph::MatmulNode", 6),
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("GemmProjRopeMxfp8Sm100", "reference/oss-dense-gemm.md#gemmprojropemxfp8sm100"),
        ("GroupedGemmGluHadamardSm100", "reference/oss-grouped-gemm.md#groupedgemmgluhadamardsm100"),
        ("discrete_grouped_gemm_dswiglu_wrapper_sm100", "reference/oss-discrete-grouped-gemm.md#discrete_grouped_gemm_dswiglu_wrapper_sm100"),
        ("RmsNormRhtAmaxSm100", "reference/oss-normalization.md#rmsnormrhtamaxsm100"),
    ],
)
def test_optional_exports_map_to_api_local_learning_anchors(name: str, expected: str) -> None:
    assert python_export_learning_anchor(name, optional=True) == expected


def test_optional_namespace_members_map_to_family_sections() -> None:
    assert optional_namespace_learning_anchor("BSA", "block_sparse_attention_forward") == (
        "reference/oss-block-sparse-attention.md#block_sparse_attention_forward"
    )
    assert optional_namespace_learning_anchor("DSA", "dense_attn_score_recompute_wrapper") == ("reference/oss-deepseek-sparse-attention.md#score-recomputation")
    assert optional_namespace_learning_anchor("NSA", "SlidingWindowAttention") == ("reference/oss-native-sparse-attention.md#sliding_window_attention_wrapper")


def test_optional_namespace_members_are_discovered_from_advertised_source() -> None:
    discovered = discover_optional_namespace_members()

    assert set(discovered["BSA"]) == {
        "block_sparse_attention_forward",
        "block_sparse_attention_fp8_forward",
        "block_sparse_attention_backward",
    }
    assert "DenseIndexerBackward" in discovered["DSA"]
    assert "SlidingWindowAttention" in discovered["NSA"]


def test_documentation_target_is_stable_across_docs_only_commits() -> None:
    target = "8b99eecabb0453e98393e019e7cc60aa65ca5f31"

    assert documentation_target({"target_commit": target}, {"target_commit": target}) == target
    with pytest.raises(CoverageError, match="target commits disagree"):
        documentation_target({"target_commit": target}, {"target_commit": "0" * 40})


def test_release_check_requires_every_manifest_sample_in_fern_catalog(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"samples": [{"path": "samples/python/add.py"}, {"path": "samples/cpp/add.cc"}]}),
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    (docs / "samples" / "python").mkdir(parents=True)
    (docs / "samples" / "python" / "add.py").write_text("", encoding="utf-8")
    (docs / "samples" / "index.md").write_text(
        "https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/python/add.py\n",
        encoding="utf-8",
    )

    with pytest.raises(CoverageError, match="omit 1 manifest samples"):
        validate_published_samples(manifest, docs)

    (docs / "samples" / "cpp").mkdir()
    (docs / "samples" / "cpp" / "add.cc").write_text("", encoding="utf-8")
    with pytest.raises(CoverageError, match="Fern sample catalog omits 1"):
        validate_published_samples(manifest, docs)

    with (docs / "samples" / "index.md").open("a", encoding="utf-8") as catalog:
        catalog.write("https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/samples/cpp/add.cc\n")
    assert validate_published_samples(manifest, docs) == 2


def test_source_reference_requires_an_existing_line(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("first\nsecond\n", encoding="utf-8")

    assert source_reference_exists("source.py:2", tmp_path)
    assert not source_reference_exists("source.py:3", tmp_path)
    assert not source_reference_exists("missing.py:1", tmp_path)
    assert source_reference_contains("source.py:2", "second", tmp_path)
    assert not source_reference_contains("source.py:2", "first", tmp_path)
