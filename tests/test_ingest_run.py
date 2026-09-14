import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from core import ingest

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                            reason="ffmpeg not installed")


def _make_m4a(path, seconds=1):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", f"sine=f=440:d={seconds}", "-c:a", "aac", "-y", str(path)],
        check=True)


def _make_wav(path, channels, seconds=1, sr=48000, values=None, marker=None):
    # `values` fills each channel with a distinct constant. Silence is unusable
    # for the destructive-overwrite tests: the mono downmix of a silent stereo
    # file is a silent mono file, so a byte comparison against zeros would pass
    # even when the file really had been overwritten.
    n = sr * seconds
    data = np.zeros((n, channels), dtype="float32")
    for c, v in enumerate(values or []):
        data[:, c] = v
    if channels == 1:
        data = data[:, 0]
    with sf.SoundFile(str(path), "w", samplerate=sr, channels=channels,
                      subtype="PCM_16") as f:
        if marker is not None:
            f.comment = marker
        f.write(data)


def _fake_ffmpeg_writes_mono(cmd, **kwargs):
    """Stand-in for a real ffmpeg decode: writes a valid mono TARGET_SR wav to
    the output path ffmpeg would have written, so `_is_ingest_product` sees a
    genuine ingest product without needing ffmpeg installed."""
    out = cmd[-1]
    sf.write(out, np.zeros(4800, dtype="float32"), ingest.TARGET_SR,
             subtype="PCM_16")
    return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


@ffmpeg
def test_m4a_becomes_a_mono_48k_wav(tmp_path):
    src = tmp_path / "memo.m4a"
    _make_m4a(src)
    ws = tmp_path / "ws"
    ws.mkdir()
    p = ingest.plan_for(str(src), str(ws))
    out = ingest.run(p)
    info = sf.info(out)
    assert info.channels == 1
    assert info.samplerate == 48000
    assert info.subtype == "PCM_16"
    assert os.path.basename(out).startswith("meeting_")


@ffmpeg
def test_foreign_stereo_wav_downmixes_under_the_mixed_flag(tmp_path):
    src = tmp_path / "quicktime.wav"
    _make_wav(src, channels=2)
    ws = tmp_path / "ws"
    ws.mkdir()
    p = ingest.plan_for(str(src), str(ws), mixed_flag=True)
    out = ingest.run(p)
    assert sf.info(out).channels == 1


def test_a_marked_recording_is_returned_untouched(tmp_path):
    """A native recording proves itself with the marker, not with channel
    count -- an unmarked stereo file gets no special treatment (see
    test_ingest_provenance.py)."""
    src = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    _make_wav(src, channels=2, marker=ingest.MARKER)
    ws = tmp_path / "ws"
    ws.mkdir()
    p = ingest.plan_for(str(src), str(ws))
    assert ingest.run(p) == str(src)
    assert sf.info(str(src)).channels == 2
    assert list(ws.iterdir()) == []


@ffmpeg
def test_existing_target_is_reused_not_redecoded(tmp_path):
    src = tmp_path / "memo.m4a"
    _make_m4a(src)
    ws = tmp_path / "ws"
    ws.mkdir()
    p = ingest.plan_for(str(src), str(ws))
    first = ingest.run(p)
    assert sf.info(first).channels == 1
    stamp = os.stat(first).st_mtime_ns
    assert ingest.run(p) == first
    assert os.stat(first).st_mtime_ns == stamp


def test_a_foreign_file_at_the_target_path_is_refused_not_overwritten(tmp_path,
                                                                     monkeypatch):
    # Existence alone is not proof ingest made the file sitting at `target`,
    # and neither is anything else readable off the filesystem: `plan` derives
    # the target from the source's filename, so an unrelated file can already
    # occupy that path. Re-converting over it (the previous behaviour) writes a
    # downmix of *this* source onto *someone else's* audio. Ingest may only
    # write over a file it made itself, so anything that isn't a mono
    # TARGET_SR ingest product is refused, untouched.
    #
    # Stubbed past the ffmpeg/ffprobe checks rather than @ffmpeg-gated: a test
    # standing between a user and a destroyed file must never silently skip.
    src = tmp_path / "quicktime.wav"
    _make_wav(src, channels=2, values=[0.1, 0.2])
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(ingest.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ingest, "has_stream", lambda source, kind: True)
    p = ingest.plan_for(str(src), str(ws), mixed_flag=True)
    os.makedirs(os.path.dirname(p.target), exist_ok=True)
    _make_wav(p.target, channels=2, values=[0.5, -0.5])
    before = open(p.target, "rb").read()

    with pytest.raises(ingest.IngestError) as excinfo:
        ingest.run(p)

    assert p.target in str(excinfo.value)
    assert "--out" in str(excinfo.value)
    # The surviving file is the point, not the exception.
    assert sf.info(p.target).channels == 2
    assert open(p.target, "rb").read() == before
    # ...and no write happened at all: no temp file, no anything, anywhere
    # under the imported subdirectory ingest converts into.
    imported_dir = os.path.dirname(p.target)
    assert [q.name for q in ws.iterdir()] == [ingest.IMPORTED_DIR]
    assert os.listdir(imported_dir) == [os.path.basename(p.target)]


def test_an_archived_copy_never_overwrites_the_workspace_original(tmp_path,
                                                                  monkeypatch):
    # The scenario this guard used to be reached through: every recording this
    # tool makes is `meeting_<ts>.wav` directly in the workspace, and
    # `derive_timestamp` reads the filename regardless of directory, so an
    # archived copy on the Desktop with the same basename used to derive a
    # target identical to the real recording's own path. Converted output now
    # always lands in the `imported/` subdirectory (see plan()), so that
    # collision cannot arise any more -- this asserts the archived copy simply
    # converts, landing beside (not onto) the original, which is left
    # completely untouched.
    ws = tmp_path / "ws"
    ws.mkdir()
    archive = tmp_path / "Desktop"
    archive.mkdir()
    name = "meeting_2026-01-02_03-04-05.wav"
    original = ws / name                       # the real, native recording
    archived = archive / name                  # unrelated audio, same basename
    _make_wav(original, channels=2, values=[0.5, -0.5], marker=ingest.MARKER)
    _make_wav(archived, channels=2, values=[0.1, 0.2])
    before = original.read_bytes()

    monkeypatch.setattr(ingest.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ingest, "has_stream", lambda source, kind: True)
    monkeypatch.setattr(ingest.subprocess, "run", _fake_ffmpeg_writes_mono)
    p = ingest.plan_for(str(archived), str(ws), mixed_flag=True)
    assert p.action == "convert"
    assert p.target != str(original)
    assert os.path.dirname(p.target).endswith(ingest.IMPORTED_DIR)

    out = ingest.run(p)

    assert out == p.target
    assert original.read_bytes() == before
    assert sf.info(str(original)).channels == 2


@ffmpeg
def test_container_without_an_audio_track_is_refused(tmp_path):
    src = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=d=1:s=64x64", "-c:v", "libx264", "-y", str(src)],
        check=True)
    ws = tmp_path / "ws"
    ws.mkdir()
    p = ingest.plan_for(str(src), str(ws))
    with pytest.raises(ingest.IngestError, match="no audio"):
        ingest.run(p)


def test_missing_ffmpeg_is_refused_with_a_hint(tmp_path, monkeypatch):
    src = tmp_path / "memo.m4a"
    src.write_bytes(b"not really aac")
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    p = ingest.plan("/x/memo.m4a", str(ws), native=False, mtime=0.0)
    with pytest.raises(ingest.IngestError, match="ffmpeg"):
        ingest.run(p)


def test_unreadable_wav_is_refused_at_ingest(tmp_path):
    src = tmp_path / "broken.wav"
    src.write_bytes(b"RIFFnope")
    with pytest.raises(ingest.IngestError):
        ingest.plan_for(str(src), str(tmp_path))


def test_a_source_already_at_the_target_path_is_refused_not_destroyed(tmp_path, monkeypatch):
    # A foreign stereo capture that happens to already carry a native
    # `meeting_<ts>.wav` name and already sits in the `imported/` subdirectory
    # ingest converts into makes plan's converted target collide with the
    # source path itself. Converting "in place" would silently replace that
    # file with a lossy downmix, so run() must refuse outright -- not convert,
    # not reuse, not touch the source at all. This is the test standing
    # between a user and a destroyed recording, so it must never silently
    # skip for lack of ffmpeg -- stub past the ffmpeg/ffprobe-dependent checks
    # instead.
    ws = tmp_path / "ws"
    imported = ws / ingest.IMPORTED_DIR
    imported.mkdir(parents=True)
    src = imported / "meeting_2026-01-02_03-04-05.wav"
    # Distinguishable values, not silence: a silent stereo file's mono downmix
    # is also silent, so a byte comparison against zeros would pass even if
    # the file really had been overwritten (see the sibling test above).
    _make_wav(src, channels=2, values=[0.3, -0.3])
    before = src.read_bytes()
    monkeypatch.setattr(ingest.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ingest, "has_stream", lambda source, kind: True)
    p = ingest.plan_for(str(src), str(ws), mixed_flag=True)
    assert p.source == p.target
    with pytest.raises(ingest.IngestError, match="onto itself"):
        ingest.run(p)
    assert sf.info(str(src)).channels == 2
    assert src.read_bytes() == before


def test_failed_conversion_leaves_no_target_behind(tmp_path, monkeypatch):
    # A garbage input makes ffmpeg fail before it ever opens an output file,
    # which would pass even without the temp-file fix. To actually exercise
    # the partial-write hazard (a killed/failed ffmpeg mid-write), stub the
    # subprocess call to write a few bytes to its output path and then report
    # failure, the way a real crash mid-encode would leave a truncated file.
    src = tmp_path / "broken.m4a"
    src.write_bytes(b"not really aac")
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(ingest.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ingest, "has_stream", lambda source, kind: True)

    def fake_run(cmd, **kwargs):
        with open(cmd[-1], "wb") as f:
            f.write(b"partial-garbage")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)
    p = ingest.plan_for(str(src), str(ws))
    with pytest.raises(ingest.IngestError):
        ingest.run(p)
    assert not os.path.exists(p.target)
    # No orphaned temp file left behind either -- anywhere. `run` does create
    # the (empty) imported/ directory before attempting the conversion, so
    # check both it and the workspace root, rather than assume a stray write
    # could only land in one or the other.
    assert [q.name for q in ws.iterdir()] == [ingest.IMPORTED_DIR]
    assert os.listdir(os.path.dirname(p.target)) == []


def test_two_sources_deriving_the_same_timestamp_get_distinct_targets(tmp_path, monkeypatch):
    """derive_timestamp falls back to mtime at one-second granularity, so a
    bulk copy that preserves mtimes can give two different foreign sources the
    same target name. Without disambiguation, the invariant that forbids
    overwriting a file ingest did not create would make the second import
    silently *reuse* the first import's audio -- never destructive, but the
    pipeline then transcribes the wrong meeting under the second import's name.
    Each conversion must record what it was made from and bump to the next
    free suffix when the target on disk belongs to someone else's source."""
    ws = tmp_path / "ws"
    ws.mkdir()
    src_a = tmp_path / "quicktime_a.wav"
    src_b = tmp_path / "quicktime_b.wav"
    _make_wav(src_a, channels=2, values=[0.1, 0.2])
    _make_wav(src_b, channels=2, values=[0.3, 0.4])
    same_mtime = 1750000000
    os.utime(src_a, (same_mtime, same_mtime))
    os.utime(src_b, (same_mtime, same_mtime))

    monkeypatch.setattr(ingest.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ingest, "has_stream", lambda source, kind: True)
    monkeypatch.setattr(ingest.subprocess, "run", _fake_ffmpeg_writes_mono)

    plan_a = ingest.plan_for(str(src_a), str(ws))
    plan_b = ingest.plan_for(str(src_b), str(ws))
    assert plan_a.target == plan_b.target      # same derived timestamp -> same first choice

    out_a = ingest.run(plan_a)
    out_b = ingest.run(plan_b)

    assert out_a == plan_a.target
    assert out_b != out_a
    root, ext = os.path.splitext(out_a)
    assert out_b == f"{root}-2{ext}"

    with open(out_a + ".ingest.json") as f:
        assert json.load(f)["source"] == os.path.abspath(str(src_a))
    with open(out_b + ".ingest.json") as f:
        assert json.load(f)["source"] == os.path.abspath(str(src_b))

    # Re-running either plan reuses its own target, never the other's.
    assert ingest.run(plan_a) == out_a
    assert ingest.run(plan_b) == out_b


def test_reprocessing_an_import_passes_through_instead_of_refusing(tmp_path):
    """A previous import's converted output already IS the shape `run` would
    produce, so reprocessing it -- pointing --from-file straight at
    imported/meeting_<ts>.wav -- is a normal thing to do, not a collision.
    Without this, plan() derives that exact same imported/meeting_<ts>.wav
    path as the *conversion target* for this *source*, and run() refuses
    "onto itself" (see test_a_source_already_at_the_target_path_is_refused_
    not_destroyed below, whose stereo source is unaffected by this change)."""
    ws = tmp_path / "ws"
    imported = ws / ingest.IMPORTED_DIR
    imported.mkdir(parents=True)
    src = imported / "meeting_2026-01-02_03-04-05.wav"
    sf.write(str(src), np.zeros(4800, dtype="float32"), ingest.TARGET_SR,
             subtype="PCM_16")
    p = ingest.plan_for(str(src), str(ws))
    assert p.action == "passthrough"
    assert p.mixed is True
    assert ingest.run(p) == str(src)


def test_an_unproven_mono_file_not_at_target_sr_still_converts(tmp_path):
    """The reuse shape-check is deliberately narrow -- mono AND exactly
    TARGET_SR -- so an ordinary foreign mono recording at some other sample
    rate (a phone voice memo, an old capture) is not mistaken for one of
    ingest's own outputs; it converts like any other foreign mono file rather
    than passing through unconverted at the wrong rate."""
    ws = tmp_path / "ws"
    ws.mkdir()
    src = tmp_path / "voice_memo.wav"
    _make_wav(src, channels=1, sr=16000)
    p = ingest.plan_for(str(src), str(ws))
    assert p.action == "convert"
    assert p.mixed is True


def test_plan_timestamp_survives_a_suffixed_target(tmp_path, monkeypatch):
    """main.py takes iplan.timestamp directly rather than re-parsing the
    resolved path's basename -- _extract_timestamp_from_path's meeting_<ts>
    regex does not match a `-2` suffix, so a refactor back to path-parsing
    would silently misdate a suffixed import's note. Guard the value plan
    actually reports, independent of whatever suffix run() lands on."""
    ws = tmp_path / "ws"
    ws.mkdir()
    src = tmp_path / "quicktime.wav"
    _make_wav(src, channels=2)
    mtime = os.path.getmtime(str(src))

    p = ingest.plan_for(str(src), str(ws))
    expected = ingest.derive_timestamp(str(src), mtime)
    assert p.timestamp == expected

    # Pre-seed the plain target as *someone else's* ingest product so `run`
    # is forced to suffix -- the physical file lands at `-2`, but the plan
    # object's timestamp must still be the un-suffixed, source-derived value.
    os.makedirs(os.path.dirname(p.target), exist_ok=True)
    sf.write(p.target, np.zeros(4800, dtype="float32"), ingest.TARGET_SR,
             subtype="PCM_16")
    other_source = tmp_path / "someone-elses-source.wav"
    other_source.write_bytes(b"unused")
    ingest._write_ingest_record(p.target, str(other_source))

    monkeypatch.setattr(ingest.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ingest, "has_stream", lambda source, kind: True)
    monkeypatch.setattr(ingest.subprocess, "run", _fake_ffmpeg_writes_mono)

    out = ingest.run(p)
    assert out != p.target                 # really did get suffixed
    assert p.timestamp == expected          # plan's timestamp is untouched by the suffix
