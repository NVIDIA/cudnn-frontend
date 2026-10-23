# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_PREFIX = "https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/"
MARKDOWN_LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")
OVERSIZED_NAV_ROUTING_PAGES = {
    "docs/api/python/index.md",
    "docs/api/cpp/index.md",
    "docs/samples/index.md",
}
FERN_NAV = REPO_ROOT / "docs" / "fern" / "versions" / "latest.yml"
FERN_ROOT = REPO_ROOT / "docs" / "fern"


def _llms_links(text: str) -> list[str]:
    return MARKDOWN_LINK.findall(text)


def _fern_source_paths(nav: object, *, include_folder_pages: bool = False) -> set[str]:
    paths: set[str] = set()
    if isinstance(nav, list):
        for item in nav:
            paths.update(_fern_source_paths(item, include_folder_pages=include_folder_pages))
        return paths
    if isinstance(nav, dict):
        page = nav.get("path")
        if isinstance(page, str):
            resolved = (FERN_NAV.parent / page).resolve()
            paths.add(resolved.relative_to(REPO_ROOT).as_posix())
        folder = nav.get("folder")
        if isinstance(folder, str):
            resolved = (FERN_NAV.parent / folder).resolve()
            candidates = resolved.glob("*.md*") if include_folder_pages else (resolved / "index.md",)
            paths.update(path.relative_to(REPO_ROOT).as_posix() for path in candidates if path.is_file())
        for key, value in nav.items():
            if key not in {"path", "folder"}:
                paths.update(_fern_source_paths(value, include_folder_pages=include_folder_pages))
    return paths


def _frontend_guide() -> dict:
    navigation = yaml.safe_load(FERN_NAV.read_text(encoding="utf-8"))["navigation"]
    return next(item for item in navigation if item.get("section") == "Frontend Guide")


def _join_route(prefix: str, slug: str) -> str:
    parts = [part for part in (prefix.strip("/"), slug.strip("/")) if part]
    return "/" + "/".join(parts)


def _fern_routes(nav: object, prefix: str = "") -> set[str]:
    routes: set[str] = set()
    if isinstance(nav, list):
        for item in nav:
            routes.update(_fern_routes(item, prefix))
        return routes
    if not isinstance(nav, dict):
        return routes

    route = _join_route(prefix, str(nav.get("slug", "")))
    if "path" in nav:
        routes.add(route)
    if "folder" in nav:
        folder = (FERN_NAV.parent / nav["folder"]).resolve()
        routes.add(route)
        for page in folder.glob("*.md*"):
            if page.stem.casefold() != "index":
                routes.add(_join_route(route, page.stem.replace("_", "-").casefold()))
    child_prefix = prefix if nav.get("skip-slug") else route
    routes.update(_fern_routes(nav.get("contents", []), child_prefix))
    return routes


def test_llms_index_is_valid_and_fits_in_one_agent_fetch() -> None:
    text = (REPO_ROOT / "docs" / "llms.txt").read_text(encoding="utf-8")
    lines = text.splitlines()

    assert len(text) < 50_000
    assert lines[0].startswith("# ")
    assert any(line.startswith("> ") for line in lines[:5])
    assert any(line.startswith("## ") for line in lines)
    assert _llms_links(text)


def test_llms_index_links_to_existing_markdown_sources() -> None:
    text = (REPO_ROOT / "docs" / "llms.txt").read_text(encoding="utf-8")
    links = _llms_links(text)

    assert len(links) == len(set(links))
    for url in links:
        assert url.startswith(RAW_PREFIX), url
        relative_path = url.removeprefix(RAW_PREFIX)
        assert relative_path.endswith(".md")
        assert (REPO_ROOT / relative_path).is_file(), relative_path
        assert len((REPO_ROOT / relative_path).read_text(encoding="utf-8")) < 50_000


def test_llms_index_covers_fern_frontend_guide_navigation() -> None:
    nav_paths = _fern_source_paths(_frontend_guide())
    llms_paths = {url.removeprefix(RAW_PREFIX) for url in _llms_links((REPO_ROOT / "docs" / "llms.txt").read_text(encoding="utf-8"))}

    assert nav_paths <= llms_paths | OVERSIZED_NAV_ROUTING_PAGES
    assert OVERSIZED_NAV_ROUTING_PAGES <= nav_paths


def test_fern_navigation_sources_exist() -> None:
    for relative_path in _fern_source_paths(_frontend_guide(), include_folder_pages=True):
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_internal_fern_links_resolve_to_navigation_routes() -> None:
    navigation = yaml.safe_load(FERN_NAV.read_text(encoding="utf-8"))["navigation"]
    routes = _fern_routes(navigation)
    internal_link = re.compile(r"\[[^]]*\]\((/[^)\s]+)\)")
    for page in [
        *(REPO_ROOT / "docs").rglob("*.md"),
        *(REPO_ROOT / "docs").rglob("*.mdx"),
    ]:
        for destination in internal_link.findall(page.read_text(encoding="utf-8")):
            route = destination.partition("#")[0].rstrip("/") or "/"
            assert route in routes, (page.relative_to(REPO_ROOT), destination)


def test_frontend_guide_uses_fern_components() -> None:
    roots = ("getting-started", "tutorials", "how-to", "concepts", "reference", "api")
    pages = [REPO_ROOT / "docs" / "index.md"]
    for root in roots:
        pages.extend((REPO_ROOT / "docs" / root).rglob("*.md"))

    code_component = re.compile(r"<Code\b(?P<attributes>.*?)/>", re.DOTALL)
    component_attribute = re.compile(r'(?P<name>[A-Za-z][A-Za-z0-9]*)="(?P<value>[^"]*)"')
    markdown_source = re.compile(r'<Markdown\b.*?\bsrc="([^"]+)".*?/>', re.DOTALL)
    for page in pages:
        text = page.read_text(encoding="utf-8")
        relative_path = page.relative_to(REPO_ROOT)
        assert '--8<-- "docs/samples/' not in text, relative_path
        assert not re.search(r'^===\s+"', text, re.MULTILINE), relative_path
        assert "```puml" not in text, relative_path
        assert not re.search(r"\{#[A-Za-z0-9_-]+\}", text), relative_path
        assert text.count("<Tabs>") == text.count("</Tabs>"), relative_path
        assert text.count("<Tab ") == text.count("</Tab>"), relative_path
        for component in code_component.finditer(text):
            attributes = {match.group("name"): match.group("value") for match in component_attribute.finditer(component.group("attributes"))}
            source = attributes["src"]
            if source.startswith(("https://", "http://")):
                continue
            resolved = FERN_ROOT / source.lstrip("/") if source.startswith("/") else page.parent / source
            assert resolved.resolve().is_file(), (relative_path, source)
            expected_language = (
                "cmake" if resolved.name == "CMakeLists.txt" else {".cc": "cpp", ".cu": "cuda", ".py": "python", ".sh": "bash"}.get(resolved.suffix)
            )
            if expected_language is not None:
                assert attributes.get("language") == expected_language, (
                    relative_path,
                    source,
                    attributes.get("language"),
                )
        for source in markdown_source.findall(text):
            resolved = FERN_ROOT / source.lstrip("/") if source.startswith("/") else page.parent / source
            assert resolved.resolve().is_file(), (relative_path, source)


def test_documentation_omits_audience_process_callouts() -> None:
    docs_root = REPO_ROOT / "docs"
    text_suffixes = {
        ".cc",
        ".css",
        ".js",
        ".json",
        ".md",
        ".mdx",
        ".py",
        ".sh",
        ".svg",
        ".txt",
    }
    documentation_files = [path for path in docs_root.rglob("*") if path.suffix in text_suffixes]
    forbidden_phrases = (
        "for coding agents",
        "for reviewer",
        "notes to reviewer",
        "reviewer note",
        "reviewers should",
        "agent-docs-directive",
    )

    for path in documentation_files:
        page = path.read_text(encoding="utf-8").casefold()
        relative_path = path.relative_to(REPO_ROOT)
        for phrase in forbidden_phrases:
            assert phrase not in page, (relative_path, phrase)


def test_tutorials_and_how_to_guides_state_prerequisites_and_results() -> None:
    for section in ("tutorials", "how-to"):
        for path in sorted((REPO_ROOT / "docs" / section).glob("*.md")):
            if path.name == "index.md":
                continue
            page = path.read_text(encoding="utf-8")
            relative_path = path.relative_to(REPO_ROOT)
            assert "## Before you begin" in page, relative_path
            assert "## Expected result" in page, relative_path


def test_callable_api_pages_follow_the_reference_section_order() -> None:
    expected_by_language = {
        "python": (
            "Python synopsis",
            "Parameters",
            "Returns",
            "Raises",
            "Notes",
            "Examples",
            "Appendix: C++ equivalent",
        ),
        "cpp": (
            "C++ synopsis",
            "Parameters",
            "Returns",
            "Errors",
            "Notes",
            "Examples",
            "Appendix: Python equivalent",
        ),
    }

    for language, expected in expected_by_language.items():
        for path in sorted((REPO_ROOT / "docs" / "api" / language).glob("*.md")):
            if path.name == "index.md":
                continue
            headings = [line.removeprefix("## ") for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("## ")]
            relative_path = path.relative_to(REPO_ROOT)
            positions = []
            for heading in expected:
                assert heading in headings, (relative_path, heading)
                positions.append(headings.index(heading))
            assert positions == sorted(positions), (relative_path, headings)
