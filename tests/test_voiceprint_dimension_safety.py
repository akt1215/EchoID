"""Write paths must survive a voiceprint left over from another model.

The stale-mean crash had two siblings that write instead of read: enrollment
blends a stored print with a new centroid, and rename averages two stored
prints. Either raises when the widths differ, which happens to anyone who
switches backend without rebuilding the DB first. A print from a different model
carries no information about this one, so it is replaced rather than blended --
crashing on the user's irreplaceable DB is the worst option available.
"""

import json

import numpy as np
import pytest

from ai.speaker_directory import SpeakerDirectory
from ai.speaker_review import SpeakerReview

NEW, OLD = 8, 4


def _sidecar(tmp_path, db_contents, dim=NEW, label="Jordan Lee"):
    wav = tmp_path / "m.wav"
    wav.write_bytes(b"stub")
    note = tmp_path / "note.md"
    note.write_text("# M\n\n## Transcript\n**[[Jordan Lee]]** (00:00):\nhi\n")
    vec = (np.ones(dim) / np.sqrt(dim)).tolist()
    side = tmp_path / "meeting_2026-01-01_00-00-00.segments.json"
    side.write_text(json.dumps({
        "wav": str(wav), "note": str(note),
        "transcript": [{"start": 0.0, "end": 5.0, "text": "hi", "label": label}],
        "diarization": [
            {"start": float(i), "end": float(i) + 5.0, "label": label,
             "cluster": "db::" + label, "source": "db", "embedding": vec}
            for i in range(6)
        ],
    }))
    db = tmp_path / "speakers.json"
    db.write_text(json.dumps(db_contents))
    return str(side), str(db)


def _db(path):
    with open(path) as f:
        return json.load(f)


def test_enrolling_over_a_stale_width_print_does_not_crash(tmp_path):
    side, db = _sidecar(tmp_path, {"Jordan Lee": [0.5] * OLD})
    r = SpeakerReview(side, db, update_rate=0.5, placeholder_min_duration=1e9)
    r.rename(0, "Jordan Lee")
    r.commit()                                  # raised ValueError before
    assert len(_db(db)["Jordan Lee"]) == NEW, "the stale print should be replaced"


def test_enrolling_over_a_matching_print_still_blends(tmp_path):
    start = [1.0] + [0.0] * (NEW - 1)
    side, db = _sidecar(tmp_path, {"Jordan Lee": start})
    r = SpeakerReview(side, db, update_rate=0.5, placeholder_min_duration=1e9)
    r.rename(0, "Jordan Lee")
    r.commit()
    after = np.asarray(_db(db)["Jordan Lee"])
    assert len(after) == NEW
    assert not np.allclose(after, start), "a same-width print must still be updated"


def test_rename_merging_mismatched_prints_does_not_crash(tmp_path):
    ws = tmp_path / "workspace"
    (ws / "Obsidian_Vault").mkdir(parents=True)
    (ws / "speakers.json").write_text(json.dumps({
        "Old Name": [1.0] + [0.0] * (NEW - 1),      # current width
        "New Name": [0.5] * OLD,                    # left over from another model
    }))
    d = SpeakerDirectory(str(ws), str(ws / "Obsidian_Vault"))
    plan = d.rename_plan("Old Name", "New Name")
    d.apply_rename(plan)                            # raised ValueError before
    db = _db(str(ws / "speakers.json"))
    assert "Old Name" not in db
    assert len(db["New Name"]) == NEW, "keep the print that matches current data"


def test_rename_merging_matching_prints_still_averages(tmp_path):
    ws = tmp_path / "workspace"
    (ws / "Obsidian_Vault").mkdir(parents=True)
    a = [1.0] + [0.0] * (NEW - 1)
    b = [0.0, 1.0] + [0.0] * (NEW - 2)
    (ws / "speakers.json").write_text(json.dumps({"Old Name": a, "New Name": b}))
    d = SpeakerDirectory(str(ws), str(ws / "Obsidian_Vault"))
    d.apply_rename(d.rename_plan("Old Name", "New Name"))
    merged = np.asarray(_db(str(ws / "speakers.json"))["New Name"])
    assert merged[0] > 0.1 and merged[1] > 0.1, "both prints should contribute"
