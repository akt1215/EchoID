"""The SHIPPED config values must sit inside their measured brackets.

Guard tests elsewhere pass thresholds explicitly, so they stay green even if
the tracked configuration template drifts. These read config.example.yaml itself.

All values are cosines in the RAW WeSpeaker space, measured over 18 hand-labeled
clusters that clear the minimum-speech gate (24 same-person pairs, 96
different-person pairs). See the Results section of
docs/superpowers/specs/2026-07-25-voiceprint-authority-design.md.

Fusing two people is unrecoverable, so the separation bounds are a floor, not a
preference.
"""

import numpy as np
import pytest
import yaml

from ai import identity_resolution as ir

# Measured brackets. A threshold between each pair is what makes the pipeline
# correct on the labeled data; outside them it either fuses people or names
# nobody.
IMPOSTOR_CEILING = 0.539        # highest score between two different people
TRUE_MATCH_FLOOR = 0.567        # lowest score between two clusters of one person
OVERSPLIT_FLOOR = 0.810         # lowest within-meeting score for one over-split person
CORRECT_MATCH_MIN_MARGIN = 0.440   # smallest top1-top2 gap on a correct match


@pytest.fixture(scope="module")
def bio():
    with open("config.example.yaml") as f:
        return yaml.safe_load(f)["biometrics"]


def _at_cosine(c):
    """Unit vector at exactly cosine `c` from (1, 0, 0)."""
    return np.array([c, np.sqrt(1.0 - c * c), 0.0])


def _segments(spec):
    segs, t = [], 0.0
    for speaker, n in spec:
        for _ in range(n):
            segs.append({"start": t, "end": t + 20.0, "speaker": speaker})
            t += 20.0
    return segs


def test_db_threshold_sits_inside_its_measured_bracket(bio):
    assert IMPOSTOR_CEILING < bio["db_threshold"] < TRUE_MATCH_FLOOR, (
        f"db_threshold={bio['db_threshold']} must fall between the impostor "
        f"ceiling {IMPOSTOR_CEILING} and the true-match floor {TRUE_MATCH_FLOOR}")


def test_merge_threshold_sits_inside_its_measured_bracket(bio):
    assert IMPOSTOR_CEILING < bio["merge_threshold"] < OVERSPLIT_FLOOR, (
        f"merge_threshold={bio['merge_threshold']} must separate different "
        f"people (<= {IMPOSTOR_CEILING}) from one over-split person "
        f"(>= {OVERSPLIT_FLOOR})")


def test_collapse_similarity_sits_inside_its_measured_bracket(bio):
    assert IMPOSTOR_CEILING < bio["collapse_similarity"] < OVERSPLIT_FLOOR


def test_db_margin_does_not_reject_correct_matches(bio):
    assert bio["db_margin"] < CORRECT_MATCH_MIN_MARGIN, (
        f"db_margin={bio['db_margin']} would reject a correct match, whose "
        f"smallest observed margin was {CORRECT_MATCH_MIN_MARGIN}")


def test_two_different_people_do_not_collapse_into_one_identity(bio):
    a = np.array([1.0, 0.0, 0.0])
    b = _at_cosine(IMPOSTOR_CEILING)
    mid = a + b
    db = {"Avery Chen": mid / np.linalg.norm(mid)}
    out = ir.resolve_clusters(
        _segments([("S0", 3), ("S1", 3)]), [a] * 3 + [b] * 3, db, {},
        db_threshold=bio["db_threshold"], db_margin=bio["db_margin"],
        collapse_similarity=bio["collapse_similarity"],
        min_speech=bio["min_speech"], min_turn_seconds=bio["min_turn_seconds"])
    names = {s["speaker"] for s in out}
    assert len(names) == 2, f"two different people were fused into {names}"


def test_an_over_split_person_still_rejoins(bio):
    a = np.array([1.0, 0.0, 0.0])
    b = _at_cosine(OVERSPLIT_FLOOR)
    db = {"Jordan Lee": np.array([1.0, 0.0, 0.0])}
    out = ir.resolve_clusters(
        _segments([("S0", 3), ("S1", 3)]), [a] * 3 + [b] * 3, db, {},
        db_threshold=bio["db_threshold"], db_margin=bio["db_margin"],
        collapse_similarity=bio["collapse_similarity"],
        min_speech=bio["min_speech"], min_turn_seconds=bio["min_turn_seconds"])
    assert {s["speaker"] for s in out} == {"Jordan Lee"}, \
        "one person split across two clusters should come back together"


def test_centering_is_off_for_this_backend(bio):
    # WeSpeaker separates on raw cosine; the persisted mean belongs to ECAPA's
    # space and mixing them would compare vectors from two different geometries.
    if bio["backend"] == "wespeaker_onnx":
        assert bio["center_embeddings"] is False


def test_thresholds_still_let_a_clean_match_through(bio):
    q = np.array([1.0, 0.0, 0.0])
    db = {"Avery Chen": np.array([1.0, 0.0, 0.0]),
          "Riley Stone": np.array([0.0, 1.0, 0.0])}
    out = ir.resolve_clusters(
        _segments([("S0", 3)]), [q] * 3, db, {},
        db_threshold=bio["db_threshold"], db_margin=bio["db_margin"],
        min_speech=bio["min_speech"], min_turn_seconds=bio["min_turn_seconds"])
    assert out[0]["speaker"] == "Avery Chen"
