"""Tests for TUI path normalization, candidate resolution, and state persistence."""

import os
import tempfile
import zoomrecorder_tui as tui


def test_strip_invisible_text():
    raw = "meeting\u200b_recording\ufeff.wav\x00"
    assert tui._strip_invisible_text(raw) == "meeting_recording.wav"


def test_is_supported_recording_file():
    assert tui._is_supported_recording_file("clip.wav")
    assert tui._is_supported_recording_file("clip.mp4")
    assert tui._is_supported_recording_file("clip.M4A")
    assert not tui._is_supported_recording_file("clip.txt")
    assert not tui._is_supported_recording_file("clip.py")


def test_normalize_file_path():
    assert tui._normalize_file_path('"/path/to/meeting.wav"') == "/path/to/meeting.wav"
    assert tui._normalize_file_path("'/path/to/meeting.wav'") == "/path/to/meeting.wav"
    assert tui._normalize_file_path("/path/with\\ escaped\\ space.wav") == "/path/with escaped space.wav"


def test_file_path_candidates_and_resolution(tmp_path):
    target = tmp_path / "meeting audio.wav"
    target.write_text("dummy")

    # Quoted path
    candidates = tui._file_path_candidates(f'"{target}"')
    assert str(target) in candidates
    assert tui._resolve_file_path(f'"{target}"') == str(target)

    # Windows-style backslashes
    win_style = str(target).replace("/", "\\")
    assert tui._resolve_file_path(win_style) == str(target)


def test_tui_state_persistence(monkeypatch, tmp_path):
    state_file = tmp_path / ".tui_state.yaml"
    monkeypatch.setattr(tui, "_TUI_STATE_FILE", str(state_file))

    state = tui._load_tui_state()
    assert state["recent_files"] == []

    test_file = tmp_path / "rec.wav"
    test_file.write_text("data")
    recorded = tui._record_recent_files([str(test_file)])
    assert str(test_file) in recorded

    loaded = tui._load_tui_state()
    assert loaded["recent_files"] == [str(test_file)]
    assert loaded["last_file_dir"] == str(tmp_path)
