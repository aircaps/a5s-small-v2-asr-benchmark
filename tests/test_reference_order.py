import json
from importlib.resources import files

import pytest

from a5sv2_eval.reference_order import apply_word_onset_references
from a5sv2_eval.score import normalize


def references():
    return [json.loads(line) for line in files("a5sv2_eval").joinpath(
        "references/word-onset-v2.jsonl"
    ).read_text().splitlines()]


def test_frozen_reference_counts_and_non_mutating_overlay():
    rows = references()
    assert len(rows) == len({r["id"] for r in rows}) == 35
    assert {c: sum(len(normalize(r["reference_raw"]).split()) for r in rows
                   if r["condition"] == c) for c in ("ami", "dipco", "notsofar")} == {
        "ami": 32928, "dipco": 33679, "notsofar": 33492,
    }
    source = {**rows[0], "reference_raw": "old turn order", "pcm_path": "local.pcm"}
    result = apply_word_onset_references([source])[0]
    assert result["reference_raw"] == rows[0]["reference_raw"]
    assert result["pcm_path"] == "local.pcm"
    assert source["reference_raw"] == "old turn order"
    assert result["reference_version"] == "four-corpus-word-onset-v2"


@pytest.mark.parametrize("key", ["pcm_sha256", "source_sha256", "sample_rate", "num_samples"])
def test_overlay_rejects_different_audio(key):
    row = references()[0]
    row[key] = "mismatch"
    with pytest.raises(ValueError, match="Reference audio mismatch"):
        apply_word_onset_references([row])
