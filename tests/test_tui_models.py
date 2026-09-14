"""Pilot tests for the model-selection UI (sticky global + per-run override)."""

import asyncio
import json

from textual.widgets import Select

import zoomrecorder_tui as tui


def _fake_config(workspace):
    return {
        "paths": {"workspace": workspace, "obsidian_vault": "vault",
                  "speaker_db": "speakers.json"},
        "audio": {"capture_backend": "sck", "sample_rate": 48000},
        "visual": {"polling_interval": 2.0, "zoom_min_window_size": 300},
        "biometrics": {"placeholder_min_duration": 60.0},
        "transcription": {"backend": "local"},
        "name_reader": {"backend": "gemini"},
        "llm": {"model": "minimax-m3:cloud", "timeout": 120},
    }


def _patch_common(monkeypatch, tmp_path, save_spy):
    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    monkeypatch.setattr(tui, "load_config", lambda: _fake_config(str(tmp_path)))
    # Never touch the real config.yaml from a test.
    monkeypatch.setattr(tui, "save_config", save_spy)


def test_select_value_coerces_yaml_bool_and_falls_back():
    assert tui._select_value(False, tui.NAME_READER_OPTS, "gemini") == "off"
    assert tui._select_value("gemini", tui.NAME_READER_OPTS, "gemini") == "gemini"
    assert tui._select_value("bogus", tui.NAME_READER_OPTS, "gemini") == "gemini"


def test_app_starts_when_name_reader_is_yaml_off(monkeypatch, tmp_path):
    # `name_reader.backend: off` in YAML loads as boolean False; the Home Select
    # must not crash on it (regression: InvalidSelectValueError: False).
    calls = []
    cfg = _fake_config(str(tmp_path))
    cfg["name_reader"]["backend"] = False
    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    monkeypatch.setattr(tui, "load_config", lambda: cfg)
    monkeypatch.setattr(tui, "save_config", lambda *a: calls.append(a))

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            return app.screen.query_one("#home_name_reader", Select).value

    assert asyncio.run(scenario()) == "off"   # coerced, no crash
    assert calls == []                         # mount must not persist anything


def test_home_select_change_persists_globally(monkeypatch, tmp_path):
    calls = []
    _patch_common(monkeypatch, tmp_path, lambda *a: calls.append(a))

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.query_one("#home_transcription", Select).value = "groq"
            await pilot.pause()

    asyncio.run(scenario())
    assert any(a[1:] == ("transcription", "backend", "groq") for a in calls)


def test_model_confirm_sets_per_run_overrides(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, lambda *a: None)
    # Don't actually launch the pipeline subprocess when Process is pressed.
    monkeypatch.setattr(tui.ProcessingScreen, "run_pipeline", lambda self: None)

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.selected_file = str(tmp_path / "m.wav")
            app.push_screen("model_confirm")
            await pilot.pause()
            app.screen.query_one("#run_transcription", Select).value = "groq"
            app.screen.query_one("#run_name_reader", Select).value = "ollama"
            await pilot.pause()
            app.screen._process()
            await pilot.pause()
        return app

    app = asyncio.run(scenario())
    assert app.model_overrides["transcription_backend"] == "groq"
    assert app.model_overrides["name_reader_backend"] == "ollama"
    assert app.model_overrides["llm_model"] == "minimax-m3:cloud"  # unchanged default


def test_model_confirm_mounts_default_run_preset(monkeypatch, tmp_path):
    _patch_common(monkeypatch, tmp_path, lambda *a: None)

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.selected_file = str(tmp_path / "first.wav")
            app.push_screen("model_confirm")
            await pilot.pause()
            return app.screen.query_one("#run_preset", Select).value

    assert asyncio.run(scenario()) == "balanced"


def test_namer_applies_per_run_llm_override(monkeypatch, tmp_path):
    # The TUI summarizes in NamerScreen (subprocess runs --no-finalize), so the
    # per-run Summary-LLM override must reach SpeakerReview here, not the global.
    _patch_common(monkeypatch, tmp_path, lambda *a: None)
    (tmp_path / "speakers.json").write_text("{}")
    sidecar = tmp_path / "m.segments.json"
    sidecar.write_text(json.dumps({
        "wav": "", "note": "", "transcript": [],
        "diarization": [{"idx": 0, "start": 0, "end": 3, "label": "Unknown (SPEAKER_00)",
                         "cluster": "SPEAKER_00", "source": "unknown", "embedding": [0, 1.0]}],
    }))

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.model_overrides = {"llm_model": "gpt-oss:20b"}  # differs from global default
            app.push_screen("namer", str(sidecar))
            await pilot.pause()
            await pilot.pause()
            return app.screen.review.llm_model

    assert asyncio.run(scenario()) == "gpt-oss:20b"
