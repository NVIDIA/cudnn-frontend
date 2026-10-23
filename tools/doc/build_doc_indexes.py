#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build deterministic API, sample, and compatibility indexes."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
AVAILABILITY_BEGIN = "{/* BEGIN GENERATED API AVAILABILITY */}"
AVAILABILITY_END = "{/* END GENERATED API AVAILABILITY */}"
SOURCES_BEGIN = "{/* BEGIN GENERATED API SOURCES */}"
SOURCES_END = "{/* END GENERATED API SOURCES */}"
SAMPLE_REGION_PATTERN = re.compile(r"--8<--\s+\[(start|end):([^]]+)\]")
SAMPLE_LANGUAGES = {
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cu": "cuda",
    ".py": "python",
    ".sh": "bash",
}


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load {path}: {exc}") from exc


def write_or_check(path: Path, content: str, *, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return True
    if check:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _initial(name: str) -> str:
    short_name = name.rsplit(".", 1)[-1]
    first = short_name[0].upper() if short_name else "#"
    return first if first in string.ascii_uppercase else "#"


def fern_page_url(page: str) -> str:
    """Return the published Fern URL for a documentation-relative page."""
    path = Path(page)
    if path == Path("index.md"):
        return "/frontend"
    if path == Path("verification.md"):
        return "/verification"
    if path == Path("reference/index.md"):
        return "/optional-kernels"
    if path.parent == Path("reference") and path.suffix == ".md":
        return f"/optional-kernels/{path.stem}"
    if path == Path("samples/guide.md"):
        return "/verified-samples/guide"
    if path == Path("samples/index.md"):
        return "/verified-samples/catalog"
    if path == Path("samples/python/README.md"):
        return "/verified-samples/python"
    if path == Path("samples/cpp/README.md"):
        return "/verified-samples/cpp"
    if path.name == "index.md":
        return f"/{path.parent.as_posix()}"
    if path.parent in {Path("api/python"), Path("api/cpp")}:
        slug = path.stem.replace("_", "-")
        return f"/{path.parent.as_posix()}/{slug}"
    return f"/{path.with_suffix('').as_posix()}"


def fern_anchor_url(anchor: str) -> str:
    page, separator, fragment = anchor.partition("#")
    url = fern_page_url(page)
    return f"{url}#{fragment}" if separator else url


def render_api_availability(feature: dict, entry: dict) -> str:
    dependencies = "; ".join(feature["dependencies"]) or "None"
    return "\n".join(
        [
            AVAILABILITY_BEGIN,
            f"**Availability: {feature['name']}**",
            "",
            f"- **API disposition:** {entry['disposition']}",
            f"- **API stability:** {entry['stability']}",
            f"- **API language:** {entry['language']}",
            f"- **API-specific availability:** {entry['availability']}",
            f"- **Family release status:** {feature['release_status']}",
            f"- **Family cuDNN:** {feature['cudnn']}",
            f"- **Family CUDA:** {feature['cuda']}",
            f"- **Family GPU:** {feature['gpu']}",
            f"- **Family platforms:** {'; '.join(feature['platforms'])}",
            f"- **Family backends:** {'; '.join(feature['backends'])}",
            f"- **Family languages:** {'; '.join(feature['languages'])}",
            f"- **Family optional and runtime dependencies:** {dependencies}",
            f"- **Recorded verification:** {feature['verification']}",
            "- [Compare feature availability](/getting-started/compatibility)",
            AVAILABILITY_END,
        ]
    )


def sync_api_availability(page: str, block: str) -> str:
    pattern = re.compile(
        rf"{re.escape(AVAILABILITY_BEGIN)}.*?{re.escape(AVAILABILITY_END)}",
        re.DOTALL,
    )
    if pattern.search(page):
        return pattern.sub(block, page, count=1)
    notes_heading = "## Notes\n"
    if notes_heading not in page:
        raise RuntimeError("API page has no exact '## Notes' heading")
    return page.replace(notes_heading, f"{notes_heading}\n{block}\n", 1)


def _github_source_link(commit: str, reference: str) -> str:
    path, separator, line = reference.rpartition(":")
    if not separator or not line.isdigit():
        raise RuntimeError(f"invalid source reference: {reference}")
    return f"https://github.com/NVIDIA/cudnn-frontend/blob/{commit}/{path}#L{line}"


def render_api_sources(entry: dict, commit: str) -> str:
    declaration = _github_source_link(commit, entry["source_declaration"])
    implementation = _github_source_link(commit, entry["implementation"])
    if entry["source_resolution"] == "absence-evidence":
        description = f"Current absence evidence: [inspected public surface]({declaration}) and [inspected implementation surface]({implementation})."
    else:
        description = f"Current source: [declaration]({declaration}) and [implementation]({implementation})."
    return "\n".join(
        [
            SOURCES_BEGIN,
            description,
            SOURCES_END,
        ]
    )


def sync_api_sources(page: str, block: str) -> str:
    generated = re.compile(rf"{re.escape(SOURCES_BEGIN)}.*?{re.escape(SOURCES_END)}", re.DOTALL)
    if generated.search(page):
        return generated.sub(block, page, count=1)
    legacy = re.compile(r"^Current sources?:.*$", re.MULTILINE)
    if legacy.search(page):
        return legacy.sub(block, page, count=1)
    examples_heading = "## Examples\n"
    if examples_heading not in page:
        raise RuntimeError("API page has neither a current-source line nor an Examples heading")
    return page.replace(examples_heading, f"{block}\n\n{examples_heading}", 1)


def _contract_value(value: object) -> str:
    if isinstance(value, list):
        text = "; ".join(str(item) for item in value) or "None"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_candidate_contract(candidate: dict) -> str:
    """Render one candidate's committed API-local compatibility contract."""
    contract = candidate["availability_contract"]
    fields = (
        ("Status", contract["release_status"]),
        ("Family status", contract["family_release_status"]),
        ("cuDNN", contract["cudnn"]),
        ("CUDA", contract["cuda"]),
        ("GPU", contract["gpu"]),
        ("Platforms", contract["platforms"]),
        ("Backends", contract["backends"]),
        ("Language", contract["language"]),
        ("Family languages", contract["family_languages"]),
        ("Dependencies", contract["dependencies"]),
        ("Verification", contract["verification"]),
        ("Compatibility evidence", contract["compatibility_evidence"]),
    )
    details = [f"**{label}:** {_contract_value(value)}" for label, value in fields]
    details.append(f"**Symbol-specific rules:** {candidate['constraint_note']}")
    details.append("[Compare compatibility families](/getting-started/compatibility)")
    details.append(f"**Source:** `{candidate['evidence']}`")
    return "<br />".join(details)


def render_candidate_row(candidate: dict) -> str:
    api = (
        f"[`{candidate['qualified_name']}`]({fern_anchor_url(candidate['learning_anchor'])})"
        if candidate["learning_anchor"]
        else f"`{candidate['qualified_name']}` (No learning page is available.)"
    )
    return f"| {api} | {candidate['kind']} | {candidate['summary']} | " f"{candidate['stability']} | {render_candidate_contract(candidate)} |"


def render_api_index(
    title: str,
    language: str,
    entries: Iterable[dict],
    unresolved: Iterable[dict] = (),
) -> str:
    selected = sorted(
        (entry for entry in entries if entry["language"] == language),
        key=lambda entry: entry["qualified_name"].casefold(),
    )
    candidates = sorted(
        (item for item in unresolved if item["language"] == language),
        key=lambda item: item["qualified_name"].casefold(),
    )
    present = {_initial(item["qualified_name"]) for item in [*selected, *candidates]}
    jumps = " ".join("[Other](#other)" if letter == "#" else f"[{letter}](#{letter.lower()})" for letter in ("#", *string.ascii_uppercase) if letter in present)
    lines = [
        f"# {title}",
        "",
        "Look up APIs by fully qualified name. Names with individual reference pages link to those pages; family-owned names link to their exact learning anchor and repeat their API-local availability contract in the table.",
        "",
        jumps or "No APIs are registered.",
        "",
    ]
    for letter in ("#", *string.ascii_uppercase):
        group = sorted(
            [
                *(("entry", entry) for entry in selected if _initial(entry["qualified_name"]) == letter),
                *(("candidate", candidate) for candidate in candidates if _initial(candidate["qualified_name"]) == letter),
            ],
            key=lambda tagged: tagged[1]["qualified_name"].casefold(),
        )
        if not group:
            continue
        lines.extend(
            [
                "## Other" if letter == "#" else f"## {letter}",
                "",
                "| API | Kind | Purpose | Stability | Availability |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for owner_kind, item in group:
            if owner_kind == "candidate":
                lines.append(render_candidate_row(item))
                continue
            link = fern_page_url(item["page"])
            lines.append(f"| [`{item['qualified_name']}`]({link}) | {item['kind']} | " f"{item['summary']} | {item['stability']} | {item['availability']} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_root_api_index(entries: list[dict], candidates: Iterable[dict] = ()) -> str:
    counts: dict[tuple[str, str, str], int] = {}
    for entry in entries:
        key = (entry["language"], "Individual reference page", entry["disposition"])
        counts[key] = counts.get(key, 0) + 1
    for candidate in candidates:
        key = (candidate["language"], "Family learning owner", candidate["disposition"])
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# API reference",
        "",
        "The reference is an alphabetical dictionary of exact API contracts. Start with a tutorial or how-to guide when you need a workflow.",
        "",
        "- [Python API](/api/python)",
        "- [C++ API](/api/cpp)",
        "- [Compatibility matrix](/getting-started/compatibility)",
        "",
        "## Coverage summary",
        "",
        "| Language | Documentation owner | Disposition | Entries |",
        "| --- | --- | --- | ---: |",
    ]
    for (language, owner, disposition), count in sorted(counts.items()):
        lines.append(f"| {language} | {owner} | {disposition} | {count} |")
    lines.append("")
    return "\n".join(lines)


def render_sample_index(samples: list[dict]) -> str:
    lines = [
        "# Documentation samples",
        "",
        "Every executable example displayed in the documentation is maintained here and checked by `tools/doc/run_doc_samples.py`.",
        "",
        "Run catalog commands from the repository root. The runner selects the canonical source, resolves compiler and output paths, and uses a declared static fallback when the current system does not meet an execution requirement.",
        "",
        "| ID | Language | Purpose | Owners | Requirements | Command | Expected result | Verification | Source |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sample in sorted(samples, key=lambda item: item["id"]):
        owners = ", ".join(f"[`{Path(owner).stem}`]({fern_page_url(owner)})" for owner in sample["owners"])
        requirements = ", ".join(sample["requirements"]) or "None"
        command = f"python tools/doc/run_doc_samples.py --id {sample['id']}"
        expected = "Exit status 0" if sample["expected"] == "exit:0" else f"Output contains `{sample['expected']}`"
        lines.append(
            f"| `{sample['id']}` | {sample['language']} | {sample['purpose']} | {owners} | "
            f"{requirements} | `{command}` | {expected} | {sample['verification']} | "
            f"[source](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/{sample['path']}) |"
        )
    lines.append("")
    return "\n".join(lines)


def render_fern_region_snippets() -> dict[Path, str]:
    """Render sample marker regions as reusable Fern Markdown snippets."""
    outputs: dict[Path, str] = {}
    for source in sorted((DOCS_ROOT / "samples").rglob("*")):
        language = SAMPLE_LANGUAGES.get(source.suffix)
        if language is None or not source.is_file():
            continue
        active: str | None = None
        regions: dict[str, list[str]] = {}
        for line in source.read_text(encoding="utf-8").splitlines():
            marker = SAMPLE_REGION_PATTERN.search(line)
            if marker is not None:
                action, name = marker.groups()
                if action == "start":
                    if active is not None:
                        raise RuntimeError(f"nested sample regions in {source}")
                    active = name
                    regions[name] = []
                elif active != name:
                    raise RuntimeError(f"mismatched sample region {name!r} in {source}")
                else:
                    active = None
                continue
            if active is not None:
                regions[active].append(line)
        if active is not None:
            raise RuntimeError(f"unclosed sample region {active!r} in {source}")
        relative = source.relative_to(DOCS_ROOT / "samples")
        for name, body in regions.items():
            destination = DOCS_ROOT / "fern" / "snippets" / "generated" / relative.parent / relative.stem / f"{name}.mdx"
            outputs[destination] = f"```{language} maxLines=0\n" + "\n".join(body).rstrip() + "\n```\n"
    return outputs


def render_compatibility(data: dict) -> str:
    lines = [
        "# Compatibility and feature availability",
        "",
        "This matrix centralizes comparison data. Each API reference page repeats all constraints that apply to that API.",
        "",
        "| Feature | Status | cuDNN | CUDA | GPU | Platforms | Backends | Languages | Dependencies | Verification |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in sorted(data["features"], key=lambda row: row["name"].casefold()):
        lines.append(
            f"| {item['name']} | {item['release_status']} | {item['cudnn']} | {item['cuda']} | "
            f"{item['gpu']} | {', '.join(item['platforms'])} | {', '.join(item['backends'])} | "
            f"{', '.join(item['languages'])} | {', '.join(item['dependencies']) or 'None'} | "
            f"{item['verification']} |"
        )
    lines.extend(
        [
            "",
            "## Read the labels",
            "",
            "**Stable** identifies a supported public surface. **Experimental** may change without the normal compatibility guarantees. **Unreleased** exists in this checkout but is not claimed for an earlier package release. **Legacy** remains for migration and compatibility.",
            "",
            "## Verification limits",
            "",
            "- Windows behavior and architecture-specific OSS ranges are source-verified but were not executed in the recorded Linux Blackwell environment.",
            "- The recorded environment used CUTLASS DSL 4.5.2 while `pyproject.toml` pins 4.5.0. Successful imports verify the observed environment, not the pinned extra.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report generated-file drift without writing")
    args = parser.parse_args(argv)

    api_manifest = load_json(DOCS_ROOT / "api" / "manifest.json")
    sample_manifest = load_json(DOCS_ROOT / "samples" / "manifest.json")
    compatibility = load_json(DOCS_ROOT / "compatibility.json")
    compatibility_by_id = {feature["id"]: feature for feature in compatibility["features"]}
    entries = api_manifest["apis"]
    unresolved = api_manifest.get("unresolved_candidates", [])
    outputs = {
        DOCS_ROOT / "api" / "index.md": render_root_api_index(entries, unresolved),
        DOCS_ROOT / "api" / "python" / "index.md": render_api_index("Python API", "python", entries, unresolved),
        DOCS_ROOT / "api" / "cpp" / "index.md": render_api_index("C++ API", "cpp", entries, unresolved),
        DOCS_ROOT / "samples" / "index.md": render_sample_index(sample_manifest["samples"]),
        DOCS_ROOT / "getting-started" / "compatibility.md": render_compatibility(compatibility),
    }
    region_outputs = render_fern_region_snippets()
    outputs.update(region_outputs)
    for entry in entries:
        page = DOCS_ROOT / entry["page"]
        feature = compatibility_by_id[entry["compatibility_id"]]
        with_availability = sync_api_availability(
            page.read_text(encoding="utf-8"),
            render_api_availability(feature, entry),
        )
        outputs[page] = sync_api_sources(
            with_availability,
            render_api_sources(entry, api_manifest["target_commit"]),
        )

    generated_root = DOCS_ROOT / "fern" / "snippets" / "generated"
    stale_generated = set(generated_root.rglob("*.mdx")) - set(region_outputs) if generated_root.exists() else set()
    if not args.check:
        for path in stale_generated:
            path.unlink()
    drift = [path for path, content in outputs.items() if not write_or_check(path, content, check=args.check)]
    drift.extend(sorted(stale_generated) if args.check else [])
    if drift:
        for path in drift:
            print(f"DRIFT: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1
    action = "checked" if args.check else "built"
    print(f"{action} {len(outputs)} generated documentation artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
