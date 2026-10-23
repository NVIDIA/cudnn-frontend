#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared data contracts for generated documentation artifacts."""

from __future__ import annotations

CANDIDATE_AVAILABILITY_FIELDS = {
    "release_status",
    "family_release_status",
    "cudnn",
    "cuda",
    "gpu",
    "platforms",
    "backends",
    "language",
    "family_languages",
    "dependencies",
    "verification",
    "compatibility_evidence",
}


def candidate_language(language: str) -> str:
    return {"cpp": "C++", "python": "Python"}.get(language, language)


def candidate_availability_contract(candidate: dict, feature: dict) -> dict:
    """Copy one candidate's mapped compatibility envelope into its ledger row."""
    return {
        "release_status": candidate["stability"],
        "family_release_status": feature["release_status"],
        "cudnn": feature["cudnn"],
        "cuda": feature["cuda"],
        "gpu": feature["gpu"],
        "platforms": feature["platforms"],
        "backends": feature["backends"],
        "language": candidate_language(candidate["language"]),
        "family_languages": feature["languages"],
        "dependencies": feature["dependencies"],
        "verification": feature["verification"],
        "compatibility_evidence": feature["evidence"],
    }
