#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate documentation manifests, API pages, and learning anchors."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from tools.doc.build_doc_indexes import render_api_availability, render_api_sources, render_candidate_contract, render_candidate_row
    from tools.doc.contracts import CANDIDATE_AVAILABILITY_FIELDS, candidate_availability_contract
except ModuleNotFoundError:  # Direct script execution puts tools/doc on sys.path.
    from build_doc_indexes import render_api_availability, render_api_sources, render_candidate_contract, render_candidate_row
    from contracts import CANDIDATE_AVAILABILITY_FIELDS, candidate_availability_contract


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
REQUIRED_API_FIELDS = {
    "language",
    "qualified_name",
    "title",
    "kind",
    "summary",
    "disposition",
    "stability",
    "availability",
    "source_declaration",
    "implementation",
    "source_resolution",
    "source_declaration_token",
    "implementation_token",
    "evidence",
    "page",
    "learning_anchor",
    "counterpart",
    "aliases",
    "compatibility_id",
    "samples",
    "verification",
}
CALLABLE_HEADINGS = ["synopsis", "parameters", "returns", "raises", "notes", "examples", "appendix"]
TYPE_HEADINGS = ["synopsis", "members", "notes", "examples", "appendix"]
ENUM_HEADINGS = ["synopsis", "values", "notes", "examples", "appendix"]
REQUIRED_COMPATIBILITY_FIELDS = {
    "id",
    "name",
    "release_status",
    "cudnn",
    "cuda",
    "gpu",
    "platforms",
    "backends",
    "languages",
    "dependencies",
    "verification",
    "evidence",
}
REQUIRED_OLD_TOPIC_FIELDS = {"source", "disposition", "destination", "reason"}
OLD_TOPIC_DISPOSITIONS = {"retained", "merged", "excluded", "review-gap"}


class CoverageError(RuntimeError):
    pass


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageError(f"cannot load {path}: {exc}") from exc


def validate_api_manifest(path: Path, docs_root: Path) -> list[dict]:
    data = _load(path)
    entries = data.get("apis")
    if not isinstance(entries, list):
        raise CoverageError("API manifest must contain an 'apis' list")
    names: set[tuple[str, str]] = set()
    pages: set[str] = set()
    for index, entry in enumerate(entries):
        missing = REQUIRED_API_FIELDS - entry.keys()
        if missing:
            raise CoverageError(f"API row {index} is missing fields: {', '.join(sorted(missing))}")
        key = (entry["language"], entry["qualified_name"])
        if key in names:
            raise CoverageError(f"duplicate qualified_name for {entry['language']}: {entry['qualified_name']}")
        names.add(key)
        if entry["page"] in pages:
            raise CoverageError(f"duplicate canonical API page: {entry['page']}")
        pages.add(entry["page"])
    for entry in entries:
        page = docs_root / entry["page"]
        if not page.is_file():
            raise CoverageError(f"missing API page for {entry['qualified_name']}: {entry['page']}")
    return entries


def _headings(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE)]


def _heading_key(heading: str) -> str:
    normalized = heading.casefold()
    if normalized.endswith(" synopsis"):
        return "synopsis"
    if normalized.startswith("appendix:"):
        return "appendix"
    return normalized.split()[0]


def validate_page(entry: dict, docs_root: Path, commit: str) -> list[str]:
    page = docs_root / entry["page"]
    text = page.read_text(encoding="utf-8")
    problems: list[str] = []
    first = text.splitlines()[0] if text else ""
    if first != f"# {entry['title']}":
        problems.append(f"{entry['page']}: title is {first!r}, expected '# {entry['title']}'")
    headings = _headings(text)
    keys = [_heading_key(heading) for heading in headings]
    if entry["kind"] == "enum":
        expected = ENUM_HEADINGS
    elif entry["kind"] in {"class", "type"}:
        expected = TYPE_HEADINGS
    else:
        expected = list(CALLABLE_HEADINGS)
        if entry["language"] == "cpp":
            expected[3] = "errors"
    position = -1
    for required in expected:
        candidates = [i for i, key in enumerate(keys) if key == required and i > position]
        if not candidates:
            problems.append(f"{entry['page']}: missing or misplaced '{required}' section")
            continue
        position = candidates[0]
    pinned = f"https://github.com/NVIDIA/cudnn-frontend/blob/{commit}/"
    if pinned not in text:
        problems.append(f"{entry['page']}: missing pinned GitHub source link for {commit}")
    if not entry["learning_anchor"] and not entry.get("reference_only_reason"):
        problems.append(f"{entry['page']}: no learning anchor or reference-only rationale")
    return problems


def _anchor_exists(docs_root: Path, value: str) -> bool:
    if not value:
        return True
    page_name, _, anchor = value.partition("#")
    page = docs_root / page_name
    if not page.is_file():
        return False
    if not anchor:
        return True
    text = page.read_text(encoding="utf-8").casefold()
    normalized = _markdown_slug(anchor)
    headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    for heading in headings:
        explicit = re.search(r"\{#([a-z0-9_-]+)(?:\s+[^}]*)?\}\s*$", heading)
        if explicit and explicit.group(1) == normalized:
            return True
        heading = re.sub(r"\s*\{#[^}]+\}\s*$", "", heading)
        if _markdown_slug(heading) == normalized:
            return True
    return False


def _markdown_slug(value: str) -> str:
    """Normalize the ASCII headings used by this documentation like Markdown TOC."""
    normalized = re.sub(r"[^a-z0-9 _-]", "", value.casefold()).strip()
    return re.sub(r"[-\s]+", "-", normalized)


def candidate_has_documentation_owner(candidate: dict) -> bool:
    """Return whether a candidate is owned by the accepted family-level model."""
    return bool(candidate.get("learning_anchor"))


def documentation_target(api_manifest: dict, compatibility: dict) -> str:
    api_target = api_manifest.get("target_commit")
    compatibility_target = compatibility.get("target_commit")
    if api_target != compatibility_target:
        raise CoverageError("API and compatibility target commits disagree")
    if not isinstance(api_target, str) or not re.fullmatch(r"[0-9a-f]{40}", api_target):
        raise CoverageError("documentation target_commit must be a full commit SHA")
    return api_target


def source_reference_exists(reference: str, repo_root: Path) -> bool:
    path_value, separator, line_value = reference.rpartition(":")
    if not separator or not line_value.isdigit():
        return False
    path = repo_root / path_value
    if not path.resolve().is_relative_to(repo_root.resolve()) or not path.is_file():
        return False
    return 1 <= int(line_value) <= len(path.read_text(encoding="utf-8").splitlines())


def source_reference_contains(reference: str, token: str, repo_root: Path) -> bool:
    if not source_reference_exists(reference, repo_root) or not token:
        return False
    path_value, _, line_value = reference.rpartition(":")
    line = (repo_root / path_value).read_text(encoding="utf-8").splitlines()[int(line_value) - 1]
    return token in line


def validate_published_samples(manifest_path: Path, docs_root: Path) -> int:
    """Require every canonical sample to exist and appear in the Fern catalog."""
    samples = _load(manifest_path).get("samples")
    if not isinstance(samples, list):
        raise CoverageError("sample manifest must contain a 'samples' list")
    missing: list[str] = []
    uncataloged: list[str] = []
    catalog = (docs_root / "samples" / "index.md").read_text(encoding="utf-8")
    for index, sample in enumerate(samples):
        path_value = sample.get("path") if isinstance(sample, dict) else None
        if not isinstance(path_value, str) or not path_value:
            raise CoverageError(f"sample row {index} has no path")
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "samples":
            raise CoverageError(f"sample row {index} has an unsafe publication path: {path_value}")
        if not (docs_root / relative).is_file():
            missing.append(path_value)
        source_url = f"https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/{path_value}"
        if source_url not in catalog:
            uncataloged.append(path_value)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        raise CoverageError(f"documentation sources omit {len(missing)} manifest samples: {preview}{suffix}")
    if uncataloged:
        preview = ", ".join(uncataloged[:5])
        suffix = "" if len(uncataloged) <= 5 else f" (and {len(uncataloged) - 5} more)"
        raise CoverageError(f"Fern sample catalog omits {len(uncataloged)} manifest samples: {preview}{suffix}")
    return len(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("review", "release"), default="review")
    args = parser.parse_args(argv)
    try:
        entries = validate_api_manifest(DOCS_ROOT / "api" / "manifest.json", DOCS_ROOT)
        api_manifest = _load(DOCS_ROOT / "api" / "manifest.json")
        unresolved = api_manifest.get("unresolved_candidates", [])
        compatibility = _load(DOCS_ROOT / "compatibility.json")
        compatibility_features = compatibility.get("features", [])
        for index, feature in enumerate(compatibility_features):
            missing = REQUIRED_COMPATIBILITY_FIELDS - feature.keys()
            if missing:
                raise CoverageError(f"compatibility row {index} is missing fields: {', '.join(sorted(missing))}")
        compatibility_by_id = {item["id"]: item for item in compatibility_features}
        compatibility_ids = set(compatibility_by_id)
        old_topics = _load(DOCS_ROOT / "old-docs-disposition.json").get("topics", [])
        if not old_topics:
            raise CoverageError("old-docs disposition ledger is empty")
        old_topic_review_gaps = 0
        for index, topic in enumerate(old_topics):
            missing = REQUIRED_OLD_TOPIC_FIELDS - topic.keys()
            if missing:
                raise CoverageError(f"old-topic row {index} is missing fields: {', '.join(sorted(missing))}")
            if topic["disposition"] not in OLD_TOPIC_DISPOSITIONS:
                raise CoverageError(f"old-topic row {index} has invalid disposition: {topic['disposition']}")
            if not isinstance(topic["reason"], str) or not topic["reason"].strip():
                raise CoverageError(f"old-topic row {index} has no reason")
            if topic["disposition"] in {"retained", "merged", "review-gap"}:
                destination = topic["destination"]
                if not isinstance(destination, str) or not _anchor_exists(DOCS_ROOT, destination):
                    raise CoverageError(f"old-topic row {index} has an invalid destination: {destination}")
            old_topic_review_gaps += topic["disposition"] == "review-gap"
        published_samples = None
        if args.mode == "release":
            published_samples = validate_published_samples(DOCS_ROOT / "samples" / "manifest.json", DOCS_ROOT)
        problems: list[str] = []
        commit = documentation_target(api_manifest, compatibility)
        for entry in entries:
            problems.extend(validate_page(entry, DOCS_ROOT, commit))
            for field in ("source_declaration", "implementation"):
                if not source_reference_exists(entry[field], REPO_ROOT):
                    problems.append(f"{entry['qualified_name']}: invalid {field} reference {entry[field]}")
            source_checks = (
                ("source_declaration", "source_declaration_token"),
                ("implementation", "implementation_token"),
            )
            for field, token_field in source_checks:
                if not source_reference_contains(entry[field], entry[token_field], REPO_ROOT):
                    problems.append(f"{entry['qualified_name']}: {field} does not contain token {entry[token_field]!r}")
            if entry["source_resolution"] not in {"exact", "absence-evidence"}:
                problems.append(f"{entry['qualified_name']}: invalid source resolution {entry['source_resolution']}")
            if entry["source_resolution"] == "absence-evidence" and entry["disposition"] != "Not exposed":
                problems.append(f"{entry['qualified_name']}: public API uses non-exact source evidence")
            if entry["compatibility_id"] not in compatibility_ids:
                problems.append(f"{entry['qualified_name']}: unknown compatibility id {entry['compatibility_id']}")
            else:
                page_text = (DOCS_ROOT / entry["page"]).read_text(encoding="utf-8")
                expected_availability = render_api_availability(compatibility_by_id[entry["compatibility_id"]], entry)
                if expected_availability not in page_text:
                    problems.append(f"{entry['page']}: missing or contradictory generated availability block")
                expected_sources = render_api_sources(entry, commit)
                if expected_sources not in page_text:
                    problems.append(f"{entry['page']}: missing or contradictory generated source links")
            if not _anchor_exists(DOCS_ROOT, entry["learning_anchor"]):
                problems.append(f"{entry['qualified_name']}: missing learning anchor {entry['learning_anchor']}")
        candidate_constraint_gaps = 0
        candidate_indexes = {language: (DOCS_ROOT / "api" / language / "index.md").read_text(encoding="utf-8") for language in ("cpp", "python")}
        for candidate in unresolved:
            missing_fields = {
                "language",
                "qualified_name",
                "disposition",
                "evidence",
                "learning_anchor",
                "suggestion",
                "compatibility_id",
                "constraint_status",
                "constraint_note",
                "kind",
                "summary",
                "stability",
                "availability_contract",
            } - candidate.keys()
            if missing_fields:
                problems.append(f"unresolved candidate is missing fields: {', '.join(sorted(missing_fields))}")
                continue
            if not source_reference_exists(candidate["evidence"], REPO_ROOT):
                problems.append(f"{candidate['qualified_name']}: missing evidence source {candidate['evidence']}")
            evidence_token = candidate["qualified_name"].rsplit(".", 1)[-1].rsplit("::", 1)[-1]
            if not source_reference_contains(candidate["evidence"], evidence_token, REPO_ROOT):
                problems.append(f"{candidate['qualified_name']}: evidence source does not contain token {evidence_token!r}")
            if not _anchor_exists(DOCS_ROOT, candidate["learning_anchor"]):
                problems.append(f"{candidate['qualified_name']}: missing learning anchor {candidate['learning_anchor']}")
            if candidate["compatibility_id"] not in compatibility_ids:
                problems.append(f"{candidate['qualified_name']}: unknown compatibility id {candidate['compatibility_id']}")
            elif candidate["constraint_status"] == "documented":
                feature = compatibility_by_id[candidate["compatibility_id"]]
                expected_availability = candidate_availability_contract(candidate, feature)
                availability_contract = candidate["availability_contract"]
                if not isinstance(availability_contract, dict):
                    problems.append(f"{candidate['qualified_name']}: local availability contract must be an object")
                    availability_contract = {}
                if availability_contract != expected_availability:
                    problems.append(f"{candidate['qualified_name']}: local availability fields contradict its compatibility record")
                missing_availability = CANDIDATE_AVAILABILITY_FIELDS - availability_contract.keys()
                if missing_availability:
                    problems.append(
                        f"{candidate['qualified_name']}: local availability contract is missing fields: " f"{', '.join(sorted(missing_availability))}"
                    )
                if not missing_availability:
                    expected_contract = render_candidate_contract(candidate)
                    expected_row = render_candidate_row(candidate)
                    index_text = candidate_indexes.get(candidate["language"], "")
                    if expected_contract not in index_text or expected_row not in index_text:
                        problems.append(f"{candidate['qualified_name']}: missing or contradictory generated API-local availability contract")
            if not all(isinstance(candidate[field], str) and candidate[field].strip() for field in ("kind", "summary", "stability")):
                problems.append(f"{candidate['qualified_name']}: kind, summary, and stability must be nonblank strings")
            if candidate["constraint_status"] not in {"documented", "review-gap"}:
                problems.append(f"{candidate['qualified_name']}: invalid constraint status {candidate['constraint_status']}")
            if not isinstance(candidate["constraint_note"], str) or not candidate["constraint_note"].strip():
                problems.append(f"{candidate['qualified_name']}: missing constraint note")
            candidate_constraint_gaps += candidate["constraint_status"] == "review-gap"
        unowned = [candidate for candidate in unresolved if not candidate_has_documentation_owner(candidate)]
        family_owned = len(unresolved) - len(unowned)
        for problem in problems:
            print(f"GAP: {problem}")
        if family_owned:
            print(f"INFO: {family_owned} candidate APIs use family-level learning anchors")
        documented_candidates = len(unresolved) - candidate_constraint_gaps
        if documented_candidates:
            print(f"INFO: {documented_candidates} candidate APIs have generated API-local availability contracts")
        if candidate_constraint_gaps:
            print(f"GAP: {candidate_constraint_gaps} family-owned candidate APIs retain explicit constraint review gaps")
        if old_topic_review_gaps:
            print(f"GAP: {old_topic_review_gaps} old-topic contracts retain review-gap dispositions")
        if published_samples is not None:
            print(f"INFO: {published_samples} manifest samples are present in the Fern source catalog")
        if unowned:
            print(f"GAP: {len(unowned)} candidate public APIs lack both a reference page and a learning anchor")
        if args.mode == "release" and old_topic_review_gaps:
            return 1
        if args.mode == "release" and candidate_constraint_gaps:
            return 1
        if problems or unowned:
            return 0 if args.mode == "review" else 1
        if candidate_constraint_gaps:
            print(f"coverage review complete: {len(entries)} APIs, {len(old_topics)} old-topic dispositions")
            return 0
        print(f"coverage ok: {len(entries)} APIs, {len(old_topics)} old-topic dispositions")
        return 0
    except CoverageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
