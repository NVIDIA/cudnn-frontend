# Contributing to cudnn-frontend

If you are interested in contributing to cudnn-frontend, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/NVIDIA/cudnn-frontend/issues)
    describing what you encountered or what you want to see changed.
    - The cudnn team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/NVIDIA/cudnn-frontend/blob/main/README.md)
   to learn how to setup the development environment.
2. Comment on the issue saying you are going to work on it and what changes you are going to make.
3. Code! Make sure to update unit tests!
4. When done, [create your pull request](https://github.com/NVIDIA/cudnn-frontend/compare).
5. Wait for other developers to review your code and update code as needed.
6. Once reviewed and approved, a cudnn-frontend developer will merge your pull request.
7. At this time, we are accepting only small fixes, changes. Once merged to main this will be an untagged version. A release tag will be assigned along with future frontend release by cudnn team.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

## Code Formatting

Consistent code formatting is important in the cudnn-frontend project to ensure
readability, maintainability, and thus simplifies collaboration.

### Branches and Versions

The cudnn-frontend repository has one main branch. Please submit a PR to this branch. We will update the doc as the policy changes.

### Branch naming

Branches used to create PRs should have a name of the form `<name>-issue-<issue_number>`
which conforms to the following conventions:

- Name:
    - A name to convey what is being worked on
    - Please use dashes or underscores between words as opposed to spaces.

## Attribution
Portions of contribution guide adopted from [https://github.com/rapidsai/cuml/blob/branch-24.04/CONTRIBUTING.md](https://github.com/rapidsai/cuml/blob/branch-24.04/CONTRIBUTING.md)
