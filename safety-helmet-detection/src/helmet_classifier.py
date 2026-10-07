from __future__ import annotations

import re


def normalize_label(label: str) -> str:
    normalized = re.sub(r"[_-]+", " ", str(label).casefold())
    return " ".join(normalized.split())


def classify_status(class_name: str | None) -> str:
    label = normalize_label(class_name or "")

    no_helmet_phrases = (
        "no helmet",
        "without helmet",
        "not wearing helmet",
        "no hardhat",
        "without hardhat",
        "no hard hat",
        "without hard hat",
        "without safety helmet",
        "missing helmet",
        "no hard hat",
        "without hard hat",
        "missing hardhat",
        "no_helmet",
    )
    helmet_phrases = (
        "with helmet",
        "wearing helmet",
        "helmeted",
        "safety helmet",
        "hard hat",
        "hardhat",
        "helmet",
    )

    if any(phrase in label for phrase in no_helmet_phrases):
        return "NO HELMET"
    if any(phrase in label for phrase in helmet_phrases):
        return "HELMET"
    return "UNKNOWN"
