"""Tests for FilePickerScreen queue, validation, and recent file interactions."""

import asyncio
import os
import pytest
from textual.widgets import Input, Button, Static
import zoomrecorder_tui as tui


def _fake_config(workspace):
    return {
        "paths": {"workspace": workspace, "obsidian_vault": "vault",
                  "speaker_db": "speakers.json"},
        "audio": {"capture_backend": "sck", "sample_rate": 48000},
        "visual": {"polling_interval": 2.0, "zoom_min_window_size": 300},
        "biometrics": {"placeholder_min_duration": 60.0},
        "llm": {"model": "test-model", "timeout": 120},
    }


def _patch_common(monkeypatch, tmp_path):
    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    monkeypatch.setattr(tui, "load_config", lambda: _fake_config(str(tmp_path)))
    monkeypatch.setattr(tui, "save_config", lambda *a: None)


def test_file_picker_queue_and_clear(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    f1 = tmp_path / "meeting1.wav"
    f2 = tmp_path / "meeting2.mp4"
    f1.write_text("data")
    f2.write_text("data")

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen("file_picker")
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, tui.FilePickerScreen)

            # Add file 1
            screen.query_one("#file_path", Input).value = str(f1)
            screen.query_one("#add_file", Button).press()
            await pilot.pause()

            # Add file 2
            screen.query_one("#file_path", Input).value = str(f2)
            screen.query_one("#add_file", Button).press()
            await pilot.pause()

            assert app.processing_queue == [str(f1), str(f2)]

            # Clear queue
            screen.query_one("#clear_queue", Button).press()
            await pilot.pause()

            assert app.processing_queue == []

    asyncio.run(scenario())


def test_file_picker_rejects_unsupported(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    bad = tmp_path / "meeting.txt"
    bad.write_text("not media")

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen("file_picker")
            await pilot.pause()

            screen = app.screen
            screen.query_one("#file_path", Input).value = str(bad)
            screen.query_one("#add_file", Button).press()
            await pilot.pause()

            assert app.processing_queue == []
            assert "Unsupported" in screen.sub_title

    asyncio.run(scenario())
