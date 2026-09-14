import os
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from core import ingest
from core.frames_store import frames_dir_for, read_frames_manifest

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                            reason="ffmpeg not installed")


def _make_call_mp4(path, seconds=9):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=d={seconds}:s=64x64", "-f", "lavfi",
         "-i", f"sine=f=440:d={seconds}", "-c:v", "libx264", "-c:a", "aac",
         "-y", str(path)], check=True)


def test_frame_elapsed_ms_is_zero_based_from_a_one_based_index():
    assert ingest.frame_elapsed_ms(1, 4.0) == 0
    assert ingest.frame_elapsed_ms(2, 4.0) == 4000
    assert ingest.frame_elapsed_ms(5, 2.5) == 10000


@ffmpeg
def test_extracted_frames_round_trip_through_the_manifest(tmp_path):
    src = tmp_path / "call.mp4"
    _make_call_mp4(src)
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    wav.write_bytes(b"")

    n = ingest.extract_frames(str(src), str(wav), interval=4.0)

    assert n >= 2
    events = read_frames_manifest(str(wav))
    assert len(events) == n
    assert events[0][0] == 0.0
    assert events[1][0] == 4.0
    assert all(p.endswith(".jpg") for _, p in events)
    assert frames_dir_for(str(wav)).endswith(".frames")


@ffmpeg
def test_video_without_a_video_track_extracts_nothing(tmp_path):
    src = tmp_path / "memo.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", "sine=f=440:d=2", "-c:a", "aac", "-y", str(src)], check=True)
    wav = tmp_path / "meeting_2026-01-02_03-04-05.wav"
    wav.write_bytes(b"")
    assert ingest.extract_frames(str(src), str(wav), interval=4.0) == 0


@ffmpeg
def test_run_extracts_frames_beside_the_wav_it_converts(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    src = tmp_path / "call.mp4"
    _make_call_mp4(src)

    p = ingest.plan_for(str(src), str(ws))
    assert p.extract_frames

    out = ingest.run(p)

    events = read_frames_manifest(out)
    assert len(events) >= 1


@ffmpeg
def test_run_extracts_frames_at_the_path_it_returns_not_at_p_target(tmp_path):
    """When the plain candidate target is already occupied by a *different*
    source's ingest product, run() bumps to a `-2` suffix and must key frame
    extraction off that resolved path. Keying off p.target here would write
    frames beside a file the pipeline never reads, so the name reader would
    silently see nothing for this import -- no error, just an empty manifest.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    src = tmp_path / "call.mp4"
    _make_call_mp4(src)

    p = ingest.plan_for(str(src), str(ws))
    assert p.extract_frames

    # Occupy the plain candidate target with a *different* source's ingest
    # product, forcing run() to resolve to a "-2" suffixed path instead.
    os.makedirs(os.path.dirname(p.target), exist_ok=True)
    other_source = tmp_path / "someone-elses-source.wav"
    other_source.write_bytes(b"unused")
    sf.write(p.target, np.zeros(4800, dtype="float32"), ingest.TARGET_SR,
              subtype="PCM_16")
    ingest._write_ingest_record(p.target, str(other_source))

    out = ingest.run(p)

    assert out != p.target
    assert read_frames_manifest(p.target) == []
    events = read_frames_manifest(out)
    assert len(events) >= 1


@ffmpeg
def test_reusing_a_target_does_not_re_extract_frames_already_present(tmp_path,
                                                                     monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    src = tmp_path / "call.mp4"
    _make_call_mp4(src)
    p = ingest.plan_for(str(src), str(ws))

    first = ingest.run(p)
    events_before = read_frames_manifest(first)
    assert len(events_before) >= 1

    calls = []
    monkeypatch.setattr(ingest, "extract_frames",
                         lambda *a, **k: calls.append(a) or 0)

    second = ingest.run(p)

    assert second == first
    assert calls == []          # reuse must not repeat a decode already done
    assert read_frames_manifest(first) == events_before


@ffmpeg
def test_reusing_a_target_backfills_frames_when_missing(tmp_path):
    """A reuse must not just trust that frames exist -- an import made before
    this feature existed (or a prior extraction that came up empty) has none,
    and reuse is the only path that will ever touch that target again."""
    ws = tmp_path / "ws"
    ws.mkdir()
    src = tmp_path / "call.mp4"
    _make_call_mp4(src)
    p = ingest.plan_for(str(src), str(ws))

    first = ingest.run(p)
    shutil.rmtree(frames_dir_for(first))
    assert read_frames_manifest(first) == []

    second = ingest.run(p)

    assert second == first
    assert len(read_frames_manifest(first)) >= 1
