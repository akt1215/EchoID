import datetime

import pytest

from core.ingest import VIDEO_EXTS, derive_timestamp, plan

MTIME = datetime.datetime(2026, 3, 4, 9, 8, 7).timestamp()


def test_a_native_recording_passes_through_as_dual():
    p = plan("/w/meeting_2026-01-02_03-04-05.wav", "/w", native=True, mtime=MTIME)
    assert p.action == "passthrough"
    assert p.target == "/w/meeting_2026-01-02_03-04-05.wav"
    assert p.mixed is False
    assert p.extract_frames is False


def test_stereo_wav_with_mixed_flag_is_converted_to_mono():
    p = plan("/x/quicktime.wav", "/w", native=True, mtime=MTIME, mixed_flag=True)
    assert p.action == "convert"
    assert p.mixed is True
    assert p.target == "/w/imported/meeting_2026-03-04_09-08-07.wav"


@pytest.mark.parametrize("name", ["zoom_0.mp4", "clip.MOV", "rec.mkv"])
def test_video_containers_convert_and_extract_frames(name):
    p = plan(f"/x/{name}", "/w", native=False, mtime=MTIME)
    assert p.action == "convert"
    assert p.mixed is True
    assert p.extract_frames is True


@pytest.mark.parametrize("name", ["memo.m4a", "voice.mp3", "call.aac"])
def test_audio_only_containers_convert_without_frames(name):
    p = plan(f"/x/{name}", "/w", native=False, mtime=MTIME)
    assert p.action == "convert"
    assert p.mixed is True
    assert p.extract_frames is False


def test_timestamp_prefers_a_native_meeting_stem():
    assert derive_timestamp("/w/meeting_2026-01-02_03-04-05.wav", MTIME) == \
        "2026-01-02_03-04-05"


def test_timestamp_falls_back_to_a_date_in_the_name_with_mtime_clock():
    assert derive_timestamp("/x/Zoom 2026-05-06 standup.mp4", MTIME) == \
        "2026-05-06_09-08-07"


def test_timestamp_falls_back_to_mtime_entirely():
    assert derive_timestamp("/x/zoom_0.mp4", MTIME) == "2026-03-04_09-08-07"


def test_video_exts_are_lowercase_and_dotted():
    assert ".mp4" in VIDEO_EXTS
    assert all(e.startswith(".") and e.islower() for e in VIDEO_EXTS)


def test_converted_target_lands_in_the_workspace(tmp_path):
    p = plan("/elsewhere/zoom_0.mp4", str(tmp_path), native=False, mtime=MTIME)
    assert p.target == str(tmp_path / "imported" / "meeting_2026-03-04_09-08-07.wav")


def test_a_converted_targets_stem_parses_back_to_the_same_time():
    from main import _extract_timestamp_from_path
    p = plan("/x/Zoom 2026-05-06 standup.mp4", "/w", native=False, mtime=MTIME)
    assert _extract_timestamp_from_path(p.target) == p.timestamp
