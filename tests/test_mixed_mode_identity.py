import numpy as np
import pytest

import main as m
from ai.transcription import Transcriber


CFG = {
    "name_reader": {"backend": "ollama", "ollama_model": "fake",
                    "frames_per_cluster": 1, "min_agreement": 1},
    "visual": {"match_window": 2.0},
    "identity": {"local_names": ["Alex Morgan"]},
}
SEGMENTS = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
FRAMES = [(1.0, "/x/000001000.jpg")]


@pytest.fixture
def fake_reader(monkeypatch):
    """Stub name_clusters so the VLM's answer is fixed and no network is used."""
    calls = {}

    def fake_name_clusters(segments, unmatched, frame_events, reader, **kw):
        calls["local_names"] = kw.get("local_names")
        return {pid: "Alex Morgan" for pid in unmatched}

    monkeypatch.setattr(m, "name_clusters", fake_name_clusters)
    monkeypatch.setattr(m, "NameReader", lambda **kw: object())
    return calls


def test_dual_mode_still_blocks_the_local_name(fake_reader):
    names = m._read_cluster_names(CFG, SEGMENTS, ["SPEAKER_00"], FRAMES,
                                  mixed=False)
    assert fake_reader["local_names"] == ["Alex Morgan"]
    assert names == {"SPEAKER_00": "Alex Morgan"}


def test_mixed_mode_lifts_the_block_and_canonicalizes_to_me_local(fake_reader):
    names = m._read_cluster_names(CFG, SEGMENTS, ["SPEAKER_00"], FRAMES,
                                  mixed=True)
    assert fake_reader["local_names"] == []
    assert names == {"SPEAKER_00": "Me (Local)"}


def test_db_matched_local_name_is_canonicalized_in_mixed_mode():
    """The VLM tier already canonicalizes a local name to "Me (Local)"; the DB
    tier must too, or the same person shows as "Me (Local)" before their print
    is enrolled and under their real name in every mixed meeting after it.
    Only the display label changes -- the cluster id keeps the DB key, which is
    what enrollment and cross-meeting crediting key on."""
    segments = [
        {"speaker": "Alex Morgan", "source": "db", "cluster": "db::Alex Morgan"},
        {"speaker": "Alice", "source": "db", "cluster": "db::Alice"},
        {"speaker": "Unknown (SPEAKER_02)", "source": "unknown", "cluster": "SPEAKER_02"},
    ]
    m._canonicalize_local_label(segments, ["Alex Morgan"])
    assert [s["speaker"] for s in segments] == [
        "Me (Local)", "Alice", "Unknown (SPEAKER_02)"]
    assert segments[0]["cluster"] == "db::Alex Morgan"


def test_canonicalization_tolerates_a_differently_spelled_local_name():
    """`is_local_name` normalizes punctuation and case, so the DB key's own
    spelling need not match config's exactly."""
    segments = [{"speaker": "alex  morgan", "source": "db",
                 "cluster": "db::alex  morgan"}]
    m._canonicalize_local_label(segments, ["Alex Morgan"])
    assert segments[0]["speaker"] == "Me (Local)"


def test_energy_override_cannot_fire_on_a_mono_file(tmp_path):
    """A mixed file has no mic channel, so nothing may be forced to Me (Local)."""
    import soundfile as sf
    wav = tmp_path / "mono.wav"
    sf.write(str(wav), np.ones(48000, dtype="float32"), 48000, subtype="PCM_16")
    result = {"segments": [{"start": 0.0, "end": 1.0, "speaker": "Alice"}]}
    # Called unbound with self=None on purpose: the method touches no instance
    # state, and constructing a Transcriber would load WhisperX and its models
    # just to exercise a channel-count branch.
    Transcriber._channel_energy_override(None, result, str(wav))
    assert result["segments"][0]["speaker"] == "Alice"


def test_db_match_skips_excluded_keys():
    from ai.identity_resolution import _db_match
    db = {"Alex Morgan": [1.0, 0.0], "Alice": [0.0, 1.0]}
    q = np.array([1.0, 0.0])
    assert _db_match(q, db, 0.5)[0] == "Alex Morgan"
    assert _db_match(q, db, 0.5, exclude=("Alex Morgan",))[0] is None


def test_db_match_with_everything_excluded_is_unknown():
    from ai.identity_resolution import _db_match
    db = {"Alex Morgan": [1.0, 0.0]}
    name, score, gap = _db_match(np.array([1.0, 0.0]), db, 0.5,
                                 exclude=("Alex Morgan",))
    assert name is None
    assert score == -1.0


def test_cluster_db_names_honors_exclude():
    from ai.identity_resolution import cluster_db_names
    segs = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}]
    embs = [np.array([1.0, 0.0])]
    db = {"Alex Morgan": [1.0, 0.0]}
    assert cluster_db_names(segs, embs, db, db_threshold=0.5) == \
        {"SPEAKER_00": "Alex Morgan"}
    assert cluster_db_names(segs, embs, db, db_threshold=0.5,
                            exclude=("Alex Morgan",)) == {"SPEAKER_00": None}


def test_resolve_clusters_honors_exclude_without_precomputed_db_names():
    """The one _db_match call inside resolve_clusters itself (the branch that
    runs when the caller did not precompute db_names) must also honor exclude
    -- this is the path main.py's precomputed-db_names call does NOT cover."""
    from ai.identity_resolution import resolve_clusters
    segs = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}]
    embs = [np.array([1.0, 0.0])]
    db = {"Alex Morgan": [1.0, 0.0]}
    out = resolve_clusters(segs, embs, db, db_threshold=0.5)
    assert out[0]["speaker"] == "Alex Morgan"
    out = resolve_clusters(segs, embs, db, db_threshold=0.5,
                           exclude=("Alex Morgan",))
    assert out[0]["speaker"].startswith("Unknown (")


def _review(tmp_path, mixed, name="Me (Local)"):
    import json
    import os
    from ai.speaker_review import SpeakerReview
    os.makedirs(tmp_path, exist_ok=True)
    side = tmp_path / "m.segments.json"
    side.write_text(json.dumps({
        "wav": "", "note": "", "mixed": mixed,
        "diarization": [{"idx": 0, "start": 0.0, "end": 90.0, "label": name,
                         "cluster": "c0", "source": "visual",
                         "embedding": [1.0, 0.0]}],
        "transcript": [{"start": 0.0, "end": 90.0, "label": name, "text": "hi"}],
    }))
    return SpeakerReview(str(side), str(tmp_path / "speakers.json"),
                         llm_model=None, local_names=["Alex Morgan"])


def test_dual_mode_never_enrolls_the_local_user(tmp_path):
    """The guard keys on the REAL name, not the display label: a dual-mode
    cluster only ever carries a name the DB or the VLM produced, and
    is_local_name("Me (Local)", ...) is False. Enrollment must be refused when
    a cluster is labeled with the local user's actual name."""
    import json
    r = _review(tmp_path, mixed=False, name="Alex Morgan")
    r.commit()
    db = json.loads((tmp_path / "speakers.json").read_text())
    assert db == {}


def test_mixed_mode_enrolls_the_canonical_label_under_the_real_name(tmp_path):
    """Task 5 canonicalizes the local user's cluster to "Me (Local)" for
    display; the print must still be stored under the real name so a future
    mixed recording can match it."""
    import json
    r = _review(tmp_path, mixed=True, name="Me (Local)")
    r.commit()
    db = json.loads((tmp_path / "speakers.json").read_text())
    assert "Alex Morgan" in db
    assert "Me (Local)" not in db


def test_mixed_mode_enrolls_a_hand_typed_real_name(tmp_path):
    import json
    r = _review(tmp_path, mixed=True, name="Alex Morgan")
    r.commit()
    db = json.loads((tmp_path / "speakers.json").read_text())
    assert "Alex Morgan" in db


def test_suggestion_names_shows_the_local_name_only_in_mixed_mode(tmp_path):
    dual = _review(tmp_path / "a", mixed=False)
    dual.db["Alex Morgan"] = [1.0, 0.0]
    assert "Alex Morgan" not in dual.suggestion_names()

    mixed = _review(tmp_path / "b", mixed=True)
    mixed.db["Alex Morgan"] = [1.0, 0.0]
    assert "Alex Morgan" in mixed.suggestion_names()
