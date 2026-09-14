"""SpeakerReview groups by the stable `cluster` id, not the display label."""

import json

from ai.speaker_review import SpeakerReview


def _write_sidecar(tmp_path, diarization):
    data = {"wav": "", "note": "", "diarization": diarization, "transcript": []}
    p = tmp_path / "m.segments.json"
    p.write_text(json.dumps(data))
    return str(p), str(tmp_path / "speakers.json")


def test_groups_by_cluster_id_when_present(tmp_path):
    # Two distinct clusters that both happen to carry the label "Guest" must stay
    # two rows — grouping by label would wrongly fuse two different people.
    sidecar, db = _write_sidecar(tmp_path, [
        {"idx": 0, "start": 0, "end": 3, "label": "Guest",
         "cluster": "SPEAKER_00", "source": "visual", "embedding": [1.0, 0]},
        {"idx": 1, "start": 4, "end": 7, "label": "Guest",
         "cluster": "SPEAKER_01", "source": "visual", "embedding": [0, 1.0]},
    ])
    r = SpeakerReview(sidecar, db)
    assert len(r.list_speakers()) == 2


def test_same_cluster_is_one_row(tmp_path):
    # One person whom pyannote over-split but the DB re-merged shares a cluster id.
    sidecar, db = _write_sidecar(tmp_path, [
        {"idx": 0, "start": 0, "end": 3, "label": "Bob",
         "cluster": "db::Bob", "source": "db", "embedding": [1.0, 0]},
        {"idx": 1, "start": 4, "end": 7, "label": "Bob",
         "cluster": "db::Bob", "source": "db", "embedding": [0.9, 0.1]},
    ])
    r = SpeakerReview(sidecar, db)
    spk = r.list_speakers()
    assert len(spk) == 1 and spk[0]["name"] == "Bob" and spk[0]["n_segments"] == 2


def test_legacy_sidecar_without_cluster_falls_back_to_label(tmp_path):
    sidecar, db = _write_sidecar(tmp_path, [
        {"idx": 0, "start": 0, "end": 3, "label": "Alice", "source": "db",
         "embedding": [1.0, 0]},
        {"idx": 1, "start": 4, "end": 7, "label": "Unknown (SPEAKER_01)",
         "source": "unknown", "embedding": [0, 1.0]},
    ])
    r = SpeakerReview(sidecar, db)
    assert sorted(s["name"] for s in r.list_speakers()) == ["Alice", "Unknown (SPEAKER_01)"]
