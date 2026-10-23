# NVIDIA cuDNN Documentation

This directory is the **source for NVIDIA cuDNN product documentation**, published with Fern Docs to `docs.nvidia.com/cudnn`.

The docs here describe installation and the graph-oriented frontend APIs (Python and C++). The lower-level C backend API published at `docs.nvidia.com/deeplearning/cudnn/backend/` is not maintained here.

## What lives here

| Area | Path | Contents |
|------|------|----------|
| Pages | `*.mdx`, `**/*.md` | Product landing pages, the Frontend Guide, developer material, and API reference |
| Verified samples | `samples/` | Canonical Python, C++, shell, and CMake examples displayed by the Frontend Guide |
| Images | `images/` | Figures referenced from pages (stored in Git LFS) |
| Fern config | `fern/` | Site settings, navigation, and generated reusable sample-region snippets |

## Layout

```text
docs/
  index.mdx                 # landing page
  quickstart.mdx
  samples.mdx
  index.md                  # Frontend Guide landing page
  getting-started/
  tutorials/
  how-to/
  concepts/
  api/{python,cpp}/
  samples/{python,cpp,...}/ # canonical executable examples
  installation/*.mdx
  developer/*.mdx
  operations/*.md
  utilities/*.md
  fe-oss-apis/**/*.md
  reference/*.mdx
  images/
  fern/
    fern.config.json
    docs.yml                # instances, theme, redirects
    versions/latest.yml     # navigation (paths are ../../<page>)
    snippets/generated/     # generated sample regions; do not edit
```

Every page must be listed in `fern/versions/latest.yml` or discovered by one
of its `folder` entries. Links between pages use published paths such as
`/tutorials/attention`, never source-file-relative paths. Fern converts
underscore-separated filenames discovered through a folder into hyphenated
slugs; for example, `api/python/sdpa_backward.md` is published at
`/api/python/sdpa-backward`.

Use Fern's `<Tabs>` and `<Tab>` components for paired Python and C++ material.
Display a complete canonical sample with `<Code src="..." />`. For a selected
sample region, add region markers to the canonical source and run the index
generator; pages include the resulting file from `/snippets/generated/` with a
`<Markdown>` component. Do not edit generated snippets or duplicate executable
examples in a page.

## Building locally

Prerequisites: Node.js 22+ and Git LFS.

```bash
git lfs pull                     # fetch images
npm install -g fern-api@5.56.3
cd docs/fern
fern check --warnings            # validate Fern configuration and navigation
```

Run the repository checks from the repository root as well:

```bash
python tools/doc/build_doc_manifests.py --check
python tools/doc/build_doc_indexes.py --check
python tools/doc/check_doc_coverage.py --mode release
python -m pytest tools/doc/tests -q
python tools/doc/run_doc_samples.py --list
```

These checks validate navigation coverage, Fern component sources, generated
regions, manifests, API-local availability contracts, and the sample catalog.
GitHub Actions runs the Fern and repository checks for documentation changes.
The tracked `old-docs-disposition.json` file is the canonical historical-topic
inventory; builds do not require a separate checkout of the former docs tree.

Optional live preview (`fern docs dev --broken-links` at
`http://localhost:3000`) performs the full MDX bundle and link check. It may
require Fern authentication and can upload documentation artifacts to Fern's
preview service, so use it only in an approved environment.

## CI and publishing

GitHub Actions run Fern from `.github/workflows/`:

- [`docs-fern.yml`](../.github/workflows/docs-fern.yml) — `fern check` and a lint for unclosed HTML image tags on pull requests and pushes
- [`docs-fern-publish.yml`](../.github/workflows/docs-fern-publish.yml) — `fern generate --docs` to staging (`develop`) and production (`main`)

Publishing uses the `FERN_TOKEN` repository secret on `develop` and `main`. Fork pull requests cannot publish; merged changes on those branches trigger the publish workflow when `docs/**` changes.
