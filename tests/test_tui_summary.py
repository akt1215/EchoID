"""Tests for SummaryScreen batch results rendering and open actions."""

import asyncio
import os
import pytest
from textual.widgets import Button, Static
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


def test_summary_screen_renders_batch_results(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path)

    note1 = tmp_path / "vault" / "note1.md"
    note1.parent.mkdir(parents=True, exist_ok=True)
    note1.write_text("# Meeting 1")

    opened = []
    monkeypatch.setattr(tui, "_open_path", lambda p: opened.append(p))

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.batch_results = [
                {"file": "/path/meeting1.wav", "note": str(note1), "status": "done"},
                {"file": "/path/meeting2.wav", "note": "", "status": "failed"},
            ]
            app.push_screen("summary")
            await pilot.pause()

            screen = app.screen
            assert isinstance(screen, tui.SummaryScreen)

            # Check open note button
            open_note_btn = screen.query_one("#open_note_0", Button)
            assert not open_note_btn.disabled
            open_note_btn.press()
            await pilot.pause()

            # Check failed item has disabled open note button
            open_note_1 = screen.query_one("#open_note_1", Button)
            assert open_note_1.disabled

            # Check open output folder button
            screen.query_one("#open_output", Button).press()
            await pilot.pause()

    asyncio.run(scenario())
    assert str(note1) in opened
    assert str(tmp_path / "vault") in opened
