import json

import numpy as np

from evaluation import clusters


def _sidecar(tmp_path, name, turns):
    """turns: list of (start, end, label, cluster, source)"""
    path = tmp_path / f"{name}.segments.json"
    path.write_text(json.dumps({
        "wav": str(tmp_path / f"{name}.wav"),
        "diarization": [
            {"start": s, "end": e, "label": lb, "cluster": cl, "source": src,
             "embedding": [float(i + 1), 0.0, 0.0]}
            for i, (s, e, lb, cl, src) in enumerate(turns)
        ],
    }))
    return path


def test_discover_groups_by_cluster(tmp_path):
    _sidecar(tmp_path, "meeting_2026-01-01_00-00-00", [
        (0.0, 4.0, "A", "SPEAKER_00", "visual"),
        (5.0, 6.0, "A", "SPEAKER_00", "visual"),
        (7.0, 9.0, "B", "SPEAKER_01", "unknown"),
    ])
    groups = clusters.discover(str(tmp_path), min_turns=1)
    by_cluster = {g.cluster: g for g in groups}
    assert set(by_cluster) == {"SPEAKER_00", "SPEAKER_01"}
    assert by_cluster["SPEAKER_00"].n_turns == 2
    assert by_cluster["SPEAKER_00"].total_seconds == 5.0
    assert by_cluster["SPEAKER_00"].meeting == "meeting_2026-01-01_00-00-00"


def test_discover_skips_turns_without_embeddings(tmp_path):
    p = tmp_path / "meeting_2026-01-02_00-00-00.segments.json"
    p.write_text(json.dumps({"wav": "x.wav", "diarization": [
        {"start": 0.0, "end": 4.0, "label": "A", "cluster": "S0",
         "source": "visual", "embedding": []},
        {"start": 4.0, "end": 8.0, "label": "A", "cluster": "S0",
         "source": "visual", "embedding": [1.0, 0.0]},
    ]}))
    groups = clusters.discover(str(tmp_path), min_turns=1)
    assert len(groups) == 1 and groups[0].n_turns == 1


def test_is_leaky_detects_db_prefix():
    assert clusters.is_leaky("db::Jordan Lee") is True
    assert clusters.is_leaky("SPEAKER_00") is False


def test_best_turn_prefers_a_long_one(tmp_path):
    _sidecar(tmp_path, "meeting_2026-01-03_00-00-00", [
        (0.0, 0.6, "A", "S0", "visual"),
        (10.0, 15.0, "A", "S0", "visual"),
        (20.0, 21.0, "A", "S0", "visual"),
    ])
    g = clusters.discover(str(tmp_path), min_turns=1)[0]
    assert clusters.best_turn(g) == (10.0, 15.0)


def test_best_turn_falls_back_to_longest_when_all_short(tmp_path):
    _sidecar(tmp_path, "meeting_2026-01-04_00-00-00", [
        (0.0, 0.5, "A", "S0", "visual"),
        (2.0, 3.2, "A", "S0", "visual"),
    ])
    g = clusters.discover(str(tmp_path), min_turns=1)[0]
    assert clusters.best_turn(g) == (2.0, 3.2)


def test_small_clusters_are_dropped(tmp_path):
    _sidecar(tmp_path, "meeting_2026-01-05_00-00-00", [
        (0.0, 1.0, "A", "S0", "visual"),
    ])
    assert clusters.discover(str(tmp_path), min_turns=5) == []


def test_embeddings_are_returned_as_a_matrix(tmp_path):
    _sidecar(tmp_path, "meeting_2026-01-06_00-00-00", [
        (0.0, 4.0, "A", "S0", "visual"),
        (5.0, 9.0, "A", "S0", "visual"),
    ])
    g = clusters.discover(str(tmp_path), min_turns=1)[0]
    assert isinstance(g.embeddings, np.ndarray)
    assert g.embeddings.shape == (2, 3)
