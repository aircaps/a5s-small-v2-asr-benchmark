"""Apply the frozen, audio-identity-checked meeting reference serialization."""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files

REFERENCE_VERSION = "four-corpus-word-onset-v2"


def apply_word_onset_references(rows: list[dict]) -> list[dict]:
    payload = files("a5sv2_eval").joinpath("references/word-onset-v2.jsonl").read_text()
    references = {row["id"]: row for row in map(json.loads, payload.splitlines())}
    output = []
    for row in rows:
        reference = references[row["id"]]
        for key in ("pcm_sha256", "source_sha256", "sample_rate", "num_samples"):
            if row[key] != reference[key]:
                raise ValueError(f"Reference audio mismatch: {row['id']} {key}")
        text = reference["reference_raw"]
        output.append({
            **row,
            "reference_raw": text,
            "reference_version": REFERENCE_VERSION,
            "reference_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
    return output
