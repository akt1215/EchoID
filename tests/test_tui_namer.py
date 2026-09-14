"""End-to-end (Textual Pilot) check that the review row wires up typeahead."""

import asyncio
import json

from textual.widgets import Checkbox, Input

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


def test_namer_row_input_has_typeahead_from_db(monkeypatch, tmp_path):
    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    (tmp_path / "speakers.json").write_text(
        json.dumps({"Riley Stone": [1.0, 0.0], "Taylor Kim": [0.0, 1.0]}))
    sidecar = tmp_path / "m.segments.json"
    sidecar.write_text(json.dumps({
        "wav": "", "note": "", "transcript": [],
        "diarization": [
            {"idx": 0, "start": 0, "end": 3, "label": "Unknown (SPEAKER_00)",
             "cluster": "SPEAKER_00", "source": "unknown", "embedding": [0, 1.0]},
        ],
    }))
    monkeypatch.setattr(tui, "load_config", lambda: _fake_config(str(tmp_path)))
    # Mounting Home fires Select.Changed; without this the TUI persists the
    # fake value into the repo's real config.yaml (conftest fails the run).
    monkeypatch.setattr(tui, "save_config", lambda *a: None)

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen("namer", str(sidecar))
            await pilot.pause()
            await pilot.pause()
            inp = app.screen.query_one("#name_0", Input)
            assert inp.suggester is not None
            names = list(inp.suggester._suggestions)
            assert "Riley Stone" in names and "Taylor Kim" in names

    asyncio.run(scenario())


def test_remember_override_survives_a_refresh(monkeypatch, tmp_path):
    """Ticking the Remember checkbox against its duration default is an
    explicit override of the model's decision -- a subsequent Merge click (a
    no-op here, since nothing is checked to merge) still calls
    `_refresh_rows`, which must show that override rather than reverting to
    the recomputed duration default."""
    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    sidecar = tmp_path / "m.segments.json"
    sidecar.write_text(json.dumps({
        "wav": "", "note": "", "transcript": [],
        "diarization": [
            # 5s of speech, well under the 60s floor in _fake_config -- the
            # checkbox must start unticked.
            {"idx": 0, "start": 0, "end": 5, "label": "Unknown (SPEAKER_00)",
             "cluster": "SPEAKER_00", "source": "unknown", "embedding": [0, 1.0]},
        ],
    }))
    monkeypatch.setattr(tui, "load_config", lambda: _fake_config(str(tmp_path)))
    # Mounting Home fires Select.Changed; without this the TUI persists the
    # fake value into the repo's real config.yaml (conftest fails the run).
    monkeypatch.setattr(tui, "save_config", lambda *a: None)

    async def scenario():
        app = tui.ZoomRecorderApp()
        # Wide terminal: the default 80-col test size clips the row past the
        # Remember checkbox, so a click on it lands off-screen and no-ops.
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("namer", str(sidecar))
            await pilot.pause()
            await pilot.pause()

            rem = app.screen.query_one("#rem_0", Checkbox)
            assert rem.value is False  # sanity: duration default is unticked

            await pilot.click("#rem_0")
            await pilot.pause()
            assert rem.value is True  # sanity: the click registered

            await pilot.click("#merge")  # nothing checked to merge -> no-op, but refreshes
            await pilot.pause()

            rem_after = app.screen.query_one("#rem_0", Checkbox)
            assert rem_after.value is True

    asyncio.run(scenario())
