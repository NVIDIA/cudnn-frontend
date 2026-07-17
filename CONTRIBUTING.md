# Contributing to cudnn-frontend

If you are interested in contributing to cudnn-frontend, we encourage you to:
- Post PRs with your ideas and fixes
- File issues for bugs you find, things you'd like to see, or questions you have

## Development environment

Requirements: an NVIDIA GPU (Hopper/Blackwell for attention and OSS kernels), CUDA toolkit, cuDNN 9.x, CMake ≥ 3.23, a C++17 compiler, Python ≥ 3.9. If cuDNN/CUDA are not installed system-wide, point `CUDNN_PATH` and `CUDAToolkit_ROOT` at them.

```bash
git clone https://github.com/NVIDIA/cudnn-frontend.git
cd cudnn-frontend

# C++ library, samples, and tests
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)

# Python package (editable). Add ".[cutedsl]" to work on the OSS CuTeDSL kernels.
pip install -e .
```

See [AGENTS.md](AGENTS.md) for the full repo map, all build options, and per-directory guides (also useful for humans, not just coding agents).

## Running tests

```bash
./build/bin/tests                 # C++ (Catch2); run one case: ./build/bin/tests "Validate conv node"
cd test/python && pytest          # Python; defaults to the L0 smoke level
```

Python tests need `pip install -e ".[cutedsl]"` plus `pytest pytest-xdist looseversion`, and a GPU. Details and conventions: [test/AGENTS.md](test/AGENTS.md).

## Code formatting

Formatting is enforced with [pre-commit](https://pre-commit.com) (clang-format for C++/CUDA, black with line length 160 for Python and notebooks):

```bash
pip install pre-commit
git add <your changed files>
pre-commit run                    # checks staged files; first run is slow while hook environments install
```

Optionally `pre-commit install` to format on every commit.

## Code contributions

### Your first issue

1. Set up the development environment as described above.
2. Comment on the issue saying you are going to work on it and what changes you are going to make.
3. Code! Make sure to update unit tests, and docs/samples when you change public APIs.
   Adding a new frontend-only kernel API? Follow the checklist in [python/cudnn/AGENTS.md](python/cudnn/AGENTS.md).
4. Run the tests and `pre-commit run` locally, then [create your pull request](https://github.com/NVIDIA/cudnn-frontend/compare) and fill in the PR template.
5. Wait for other developers to review your code and update code as needed.
6. Once reviewed and approved, a cudnn-frontend developer will merge your pull request.
7. Merged changes ship untagged until the next frontend release, when the cudnn team assigns a release tag.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

### Branches and versions

Active development happens on the `develop` branch — base your work on `develop` and submit PRs against it. `main` tracks releases. We will update this doc as the policy changes.

### Branch naming

Branches used to create PRs should have a name of the form `<name>-issue-<issue_number>`
which conforms to the following conventions:

- Name:
    - A name to convey what is being worked on
    - Please use dashes or underscores between words as opposed to spaces.

## Attribution
Portions of contribution guide adopted from [https://github.com/rapidsai/cuml/blob/branch-24.04/CONTRIBUTING.md](https://github.com/rapidsai/cuml/blob/branch-24.04/CONTRIBUTING.md)
