import asyncio

from unittest.mock import MagicMock

import zoomrecorder_tui as tui
from zoomrecorder_tui import RecordingScreen


class _FakeRecorder:
    instances = []

    def __init__(self, mic, bh, output_filename, **kwargs):
        self.output_filename = output_filename
        self.stopped = False
        self.ch1_silent = False
        self.stalled_side = None
        _FakeRecorder.instances.append(self)

    def start(self):
        pass

    def stop(self):
        self.stopped = True


class _FakeVision:
    def __init__(self, *a, **k):
        self.frame_count = 0

    def start(self):
        pass

    def stop(self):
        pass


def _fake_config(workspace):
    return {
        "paths": {"workspace": workspace, "obsidian_vault": "vault",
                  "speaker_db": "speakers.json"},
        "audio": {"capture_backend": "sck", "sample_rate": 48000},
        "visual": {"polling_interval": 2.0, "zoom_min_window_size": 300},
        "llm": {"model": "test-model", "timeout": 120},
    }


def _screen(tmp_path):
    scr = RecordingScreen()
    scr._finalized = False
    scr.audio_path = str(tmp_path / "meeting_2026-07-05_05-29-14.wav")
    scr.recorder = MagicMock()
    scr.vision = MagicMock()
    return scr


def test_finalize_stops_recorder_and_vision(tmp_path):
    scr = _screen(tmp_path)
    scr._finalize_recording()
    scr.recorder.stop.assert_called_once()
    scr.vision.stop.assert_called_once()


def test_finalize_is_idempotent(tmp_path):
    # Stop button then on_unmount (or quit) must not double-stop the recorder.
    scr = _screen(tmp_path)
    scr._finalize_recording()
    scr._finalize_recording()
    scr.recorder.stop.assert_called_once()


def test_on_unmount_finalizes_when_stop_never_pressed(tmp_path):
    # Quitting mid-recording used to lose the whole WAV; on_unmount must save it.
    scr = _screen(tmp_path)
    scr.on_unmount()
    scr.recorder.stop.assert_called_once()


def test_pilot_quitting_midrecording_finalizes_wav(monkeypatch, tmp_path):
    # End-to-end through Textual: verify on_unmount actually fires on app exit,
    # so quitting while recording stops the recorder and persists the OCR log
    # instead of losing the meeting. Fakes keep it off real hardware and the
    # real workspace/.
    _FakeRecorder.instances.clear()
    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    monkeypatch.setattr(tui, "load_config", lambda: _fake_config(str(tmp_path)))
    # Mounting Home fires Select.Changed; without this the TUI persists the
    # fake value into the repo's real config.yaml (conftest fails the run).
    monkeypatch.setattr(tui, "save_config", lambda *a: None)
    monkeypatch.setattr("core.audio_recorder.AudioRecorder", _FakeRecorder)
    monkeypatch.setattr("core.visual_ingestion.VisualIngestion", _FakeVision)

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.mic, app.bh = 0, None
            app.push_screen("record")
            await pilot.pause()
            assert _FakeRecorder.instances, "recorder was never constructed"
            assert not _FakeRecorder.instances[-1].stopped
            app.exit()  # quit WITHOUT pressing Stop
            await pilot.pause()

    asyncio.run(scenario())

    rec = _FakeRecorder.instances[-1]
    assert rec.stopped, "quitting mid-recording did not stop the recorder"
