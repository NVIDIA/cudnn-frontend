# NVIDIA cuDNN Documentation

This directory is the **source for NVIDIA cuDNN product documentation**, published with Fern Docs to `docs.nvidia.com/cudnn`.

[cuDNN](https://developer.nvidia.com/cudnn) is the CUDA Deep Neural Network library: GPU-accelerated primitives and fusion patterns for deep learning workloads. The docs here describe installation and the graph-oriented frontend APIs (Python and C++). The lower-level C backend API is maintained in the `cudnn/cudnn` repository under `docs/` and published separately at `docs.nvidia.com/deeplearning/cudnn/backend/`.

API Markdown under `operations/`, `utilities/`, and `fe-oss-apis/` is maintained in this repository with the library code.

## What lives here

| Area | Path | Contents |
|------|------|----------|
| Pages | `*.mdx`, `**/*.md` | Landing page, quick start, samples, installation, developer guide, frontend API reference (`operations/`, `utilities/`, `fe-oss-apis/`), reference material |
| Images | `images/` | Figures referenced from pages (stored in Git LFS) |
| Fern config | `fern/` | `fern.config.json`, `docs.yml` (site settings, redirects), `versions/latest.yml` (left navigation) |

## Layout

```text
docs/
  index.mdx                 # landing page
  quickstart.mdx
  samples.mdx
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
```

Every page must be listed in `fern/versions/latest.yml` to be built. Page `slug`s are the source file names, so links between pages use `/section/page` (for example `/developer/graph-api`), never file paths.

## Building locally

Prerequisites: Node.js 22+ and Git LFS.

```bash
git lfs pull                     # fetch images
npm install -g fern-api
cd docs/fern
fern check                       # validate config, navigation, and MDX (no login required)
```

`fern check` is enough to catch navigation drift and MDX issues before opening a pull request; GitHub Actions runs the same command on doc changes.

Optional live preview (`fern docs dev` at `http://localhost:3000`) uses the Fern CLI and may require `fern login --email <you@example.com>` with a Fern account that can access this docs project. Preview is not required to contribute content fixes.

## CI and publishing

GitHub Actions run Fern from `.github/workflows/`:

- [`docs-fern.yml`](../.github/workflows/docs-fern.yml) — `fern check` and a lint for unclosed HTML image tags on pull requests and pushes
- [`docs-fern-publish.yml`](../.github/workflows/docs-fern-publish.yml) — `fern generate --docs` to staging (`develop`) and production (`main`)

Publishing uses the `FERN_TOKEN` repository secret on `develop` and `main`. Fork pull requests cannot publish; merged changes on those branches trigger the publish workflow when `docs/**` changes.
