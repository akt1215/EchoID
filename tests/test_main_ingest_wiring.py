"""main.py's own ingest wiring has no other test: everything above it
(cluster_db_names/resolve_clusters honoring `exclude`, ingest's trust rules)
is covered in isolation, but nothing previously drove main() itself to check
that it actually threads `exclude=local_db_keys` into both call sites
(main.py:387, :398) or that "mixed" lands in the sidecar it writes (:457). A
reviewer once deleted all three and the full suite still passed -- see
progress.md, Task 6. These tests run the real `main()` end to end with the
heavy ML stages faked out, so a regression in that wiring fails a test instead
of silently breaking mixed-mode enrollment and cross-meeting crediting.
"""
import datetime
import copy
import json
import os

import numpy as np
import pytest
import soundfile as sf

import main as m
from core import ingest


def _write_native_wav(path):
    """A 2ch WAV carrying the recorder's provenance marker -- ingest reads it
    as a dual-channel recording of this tool's own making."""
    with sf.SoundFile(str(path), mode="w", samplerate=48000, channels=2,
                       subtype="PCM_16") as f:
        f.comment = ingest.MARKER
        f.write(np.zeros((4800, 2), dtype="float32"))


def _write_foreign_mono_wav(path):
    """A plain mono 48kHz WAV with no marker. It happens to have the exact
    shape ingest's own conversion writes, so plan_for reads it as an
    already-mixed import (a reprocess) rather than a native recording --
    never a way for an unproven file to reach mixed=False."""
    sf.write(str(path), np.zeros(4800, dtype="float32"), 48000, subtype="PCM_16")


# The fake meeting's single turn, in seconds. It must clear config.yaml's
# `biometrics.min_speech` by a wide margin: a cluster below that floor resolves
# to Unknown on duration alone, which is also what the dual-mode test asserts,
# so a turn near the floor would let that test keep passing while quietly
# losing its ability to catch an `exclude=` regression. Deliberately far above
# any plausible floor, and the margin is asserted below rather than assumed.
TURN_SECONDS = 300.0


class _FakeDiarizer:
    def __init__(self, **kw):
        pass

    def diarize(self, audio_path, target_channel=None):
        return [{"start": 0.0, "end": TURN_SECONDS, "speaker": "SPEAKER_00"}]


class _FakeBiometricsManager:
    """speaker_db pre-seeded with the local user's enrolled print; embed_all
    returns that same vector for the meeting's only turn, so the cluster
    centroid is an exact voiceprint match unless main.py's exclude wiring
    blocks it."""

    def __init__(self, **kw):
        self.speaker_db = {"Alex Morgan": [1.0, 0.0]}

    def embed_all(self, path, segments):
        return [np.array([1.0, 0.0]) for _ in segments]


class _FakeTranscriber:
    def __init__(self, **kw):
        pass

    def transcribe(self, audio_path, resolved_segments, mixed=False, channel=None):
        text = "\n".join(f"{s['speaker']}: hi" for s in resolved_segments)
        segs = [{"start": s["start"], "end": s["end"], "label": s["speaker"],
                 "text": "hi"} for s in resolved_segments]
        return text, segs


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace every heavy ML stage main() touches with a fixed, known fake;
    leave the real identity-resolution functions (the thing under test) and
    the real ingest module untouched."""
    monkeypatch.setattr(m, "Diarizer", _FakeDiarizer)
    monkeypatch.setattr(m, "BiometricsManager", _FakeBiometricsManager)
    monkeypatch.setattr(m, "Transcriber", _FakeTranscriber)
    monkeypatch.setattr(m, "_ensure_ollama_running", lambda: None)
    cfg = copy.deepcopy(m.load_config())
    cfg["identity"]["local_names"] = ["Alex Morgan"]
    monkeypatch.setattr(m, "load_config", lambda: cfg)


def _run_main(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["main.py"] + argv)
    m.main()


def _load_sidecar(out_dir, wav_path):
    stem = os.path.splitext(os.path.basename(str(wav_path)))[0]
    with open(os.path.join(str(out_dir), stem + ".segments.json")) as f:
        return json.load(f)


def test_the_fake_turn_outlasts_the_configured_min_speech():
    """The precondition both tests below depend on, asserted instead of assumed.

    `resolve_clusters` sends any cluster with less than `biometrics.min_speech`
    of usable speech straight to Unknown without consulting the DB. Raise that
    floor past the fake turn and the dual-mode test would still pass -- via the
    too-short path, no longer via the `exclude=` wiring it exists to guard.
    """
    assert m.load_config()["biometrics"]["min_speech"] < TURN_SECONDS


def test_dual_mode_sidecar_excludes_the_local_print_and_records_mixed_false(
        fake_pipeline, monkeypatch, tmp_path):
    """Dual-channel (native) recording: local_db_keys is non-empty, so the
    sole cluster -- which would otherwise voiceprint-match "Alex Morgan"
    exactly -- must be excluded and left Unknown. If main.py stopped passing
    exclude= to cluster_db_names/resolve_clusters, this cluster would silently
    resolve to the local user's name instead."""
    wav = tmp_path / "meeting_2026-01-01_00-00-00.wav"
    _write_native_wav(wav)
    out = tmp_path / "out"

    _run_main(monkeypatch, ["--from-file", str(wav), "--out", str(out),
                            "--no-finalize"])

    sidecar = _load_sidecar(out, wav)
    assert "mixed" in sidecar
    assert sidecar["mixed"] is False
    assert sidecar["diarization"][0]["label"] == "Unknown (SPEAKER_00)"
    # A native recording's note keeps the name the vault already uses: only an
    # import that could collide needs disambiguating.
    assert os.path.basename(sidecar["note"]) == "2026-01-01_00-00-00_Meeting.md"


def test_mixed_mode_sidecar_allows_the_match_and_records_mixed_true(
        fake_pipeline, monkeypatch, tmp_path):
    """A file ingest recognizes as its own mixed-mode output (mono, 48kHz) has
    local_db_keys == (); the same voiceprint DOES resolve, and the sidecar
    must record "mixed": true so review/enrollment and cross-meeting
    crediting (ai/speaker_directory.py) treat it as mixed.

    The match resolves to the local user, so it surfaces under the canonical
    "Me (Local)" label -- the same spelling the VLM tier produces, and the
    same one an unenrolled mixed meeting shows. The cluster id is what proves
    the DB tier fired at all (an excluded or missed match would leave the
    cluster on its raw pyannote id), so the exclude-wiring guard is intact.
    """
    wav = tmp_path / "imported.wav"
    _write_foreign_mono_wav(wav)
    out = tmp_path / "out"

    _run_main(monkeypatch, ["--from-file", str(wav), "--out", str(out),
                            "--no-finalize"])

    sidecar = _load_sidecar(out, wav)
    assert "mixed" in sidecar
    assert sidecar["mixed"] is True
    assert sidecar["diarization"][0]["cluster"] == "db::Alex Morgan"
    assert sidecar["diarization"][0]["label"] == "Me (Local)"


def test_two_imports_with_one_derived_timestamp_get_distinct_notes(
        fake_pipeline, monkeypatch, tmp_path):
    """Two foreign files whose derived timestamps collide must not share a note.

    `derive_timestamp` falls back to mtime, which has one-second granularity --
    a `cp -p` bulk copy ties two unrelated recordings routinely. Each gets its
    own wav and its own review sidecar; a note keyed on the timestamp alone
    would be written twice, so the second import silently replaces the first's
    note, and committing the first from its own sidecar later rewrites that
    shared file from the wrong meeting's transcript.

    The displayed date still comes from the meeting time -- only the filename
    disambiguates -- so both notes carry the same frontmatter date.
    """
    out = tmp_path / "out"
    tied = 1767225600          # identical mtime for both sources
    expected_date = datetime.datetime.fromtimestamp(tied).strftime(
        "%Y-%m-%dT%H:%M:%S")

    notes = []
    for name in ("a.wav", "b.wav"):
        src = tmp_path / name
        _write_foreign_mono_wav(src)
        os.utime(src, (tied, tied))
        _run_main(monkeypatch, ["--from-file", str(src), "--out", str(out),
                                "--no-finalize"])
        notes.append(_load_sidecar(out, src)["note"])

    assert notes[0] != notes[1]
    assert [os.path.basename(n) for n in notes] == ["a_Meeting.md", "b_Meeting.md"]
    for note in notes:
        assert os.path.exists(note)
        assert f"date: {expected_date}" in open(note).read()
