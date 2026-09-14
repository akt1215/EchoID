"""The enrollment EMA must not fire on a name the user never confirmed.

commit() updated a voiceprint for every non-Unknown group, including one the DB
auto-filled and the user left untouched. A wrong match therefore dragged the
stored print toward the wrong person, which made the next wrong match easier —
a feedback loop observed in the live DB, where one voiceprint matched four
different clusters across three meetings above 0.70.
"""

import json

import numpy as np
import pytest

from ai.speaker_review import SpeakerReview


@pytest.fixture
def sidecar(tmp_path):
    def _make(label, source, cluster):
        wav = tmp_path / "m.wav"
        wav.write_bytes(b"")
        path = tmp_path / "m.segments.json"
        path.write_text(json.dumps({
            "wav": str(wav), "note": "", "transcript": [],
            "diarization": [
                {"start": float(i), "end": float(i) + 4.0, "label": label,
                 "cluster": cluster, "source": source,
                 "embedding": [0.0, 1.0, 0.0]}
                for i in range(20)
            ],
        }))
        db = tmp_path / "speakers.json"
        db.write_text(json.dumps({"Jordan Lee": [1.0, 0.0, 0.0]}))
        return str(path), str(db)
    return _make


def _db(path):
    with open(path) as f:
        return {k: np.asarray(v, float) for k, v in json.load(f).items()}


def test_untouched_auto_name_does_not_move_the_voiceprint(sidecar):
    side, db = sidecar("Jordan Lee", "db", "db::Jordan Lee")
    before = _db(db)["Jordan Lee"].copy()
    r = SpeakerReview(side, db, update_rate=0.5, placeholder_min_duration=1e9)
    r.commit()
    assert np.allclose(_db(db)["Jordan Lee"], before), \
        "an auto-filled name the user never confirmed poisoned the voiceprint"


def test_user_confirmed_name_does_update(sidecar):
    side, db = sidecar("Jordan Lee", "db", "db::Jordan Lee")
    before = _db(db)["Jordan Lee"].copy()
    r = SpeakerReview(side, db, update_rate=0.5, placeholder_min_duration=1e9)
    r.rename(0, "Jordan Lee")          # the user actively confirmed this row
    r.commit()
    assert not np.allclose(_db(db)["Jordan Lee"], before)


def test_trust_auto_names_restores_the_old_behavior(sidecar):
    side, db = sidecar("Jordan Lee", "db", "db::Jordan Lee")
    before = _db(db)["Jordan Lee"].copy()
    r = SpeakerReview(side, db, update_rate=0.5, placeholder_min_duration=1e9,
                      trust_auto_names=True)
    r.commit()
    assert not np.allclose(_db(db)["Jordan Lee"], before)


def test_a_brand_new_name_always_enrolls(sidecar):
    side, db = sidecar("Someone New", "visual", "SPEAKER_00")
    r = SpeakerReview(side, db, update_rate=0.5, placeholder_min_duration=1e9)
    r.commit()
    assert "Someone New" in _db(db), \
        "a first enrollment is not an EMA update and must not be gated"
