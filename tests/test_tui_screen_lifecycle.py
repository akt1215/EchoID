"""A second meeting in one TUI session must actually start.

Screens registered in `App.SCREENS` are instantiated once and cached, so
`on_mount` fires only on the first push. That made the second meeting of a
session a no-op in two places: `ProcessingScreen` never re-ran the pipeline,
and -- worse, because it fails silently mid-meeting -- `RecordingScreen` never
re-constructed the recorder while still rendering as if it were recording.

These tests push each screen twice and assert the `on_mount` side effect fired
both times.
"""

import asyncio

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


def _boot(monkeypatch, tmp_path):
    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    monkeypatch.setattr(tui, "load_config", lambda: _fake_config(str(tmp_path)))
    # Mounting Home fires Select.Changed; without this the TUI persists the
    # fake value into the repo's real config.yaml (conftest fails the run).
    monkeypatch.setattr(tui, "save_config", lambda *a: None)


def test_processing_runs_the_pipeline_on_every_push(monkeypatch, tmp_path):
    _boot(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(tui.ProcessingScreen, "run_pipeline",
                        lambda self: calls.append(self.app.selected_file))

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            for name in ("first.wav", "second.wav"):
                app.selected_file = name
                app.push_screen("processing")
                await pilot.pause()
                app.pop_screen()
                await pilot.pause()

    asyncio.run(scenario())
    assert calls == ["first.wav", "second.wav"], (
        "the second meeting never started the pipeline")


def test_recording_starts_the_recorder_on_every_push(monkeypatch, tmp_path):
    _boot(monkeypatch, tmp_path)
    starts = []
    monkeypatch.setattr(tui.RecordingScreen, "_start_recording",
                        lambda self: starts.append(self))

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.mic, app.bh = 0, None
            for _ in range(2):
                app.push_screen("record")
                await pilot.pause()
                app.pop_screen()
                await pilot.pause()

    asyncio.run(scenario())
    assert len(starts) == 2, (
        "the second recording never constructed a recorder, but the screen "
        "still renders as if it were recording")
    assert starts[0] is not starts[1], "the cached screen was reused"


def test_record_another_from_the_summary_lands_on_a_live_recording_screen(
        monkeypatch, tmp_path):
    """"Record Another" unwinds a screen stack by a hardcoded number of pops.
    Uncached screens are discarded rather than suspended by those pops, so the
    count is worth pinning: landing one screen short would push `record` on top
    of Processing, and one too far would pop past Home."""
    _boot(monkeypatch, tmp_path)
    starts = []
    monkeypatch.setattr(tui.RecordingScreen, "_start_recording",
                        lambda self: starts.append(self))
    monkeypatch.setattr(tui.ProcessingScreen, "run_pipeline", lambda self: None)

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.mic, app.bh = 0, None
            app.selected_file = str(tmp_path / "meeting.wav")
            app.push_screen("record")
            await pilot.pause()
            app.push_screen("processing")
            await pilot.pause()
            app.push_screen("summary")
            await pilot.pause()
            await pilot.click("#record")
            await pilot.pause()
            return isinstance(app.screen, tui.RecordingScreen)

    assert asyncio.run(scenario()), "Record Another did not land on the recorder"
    assert len(starts) == 2, "the second recording never started"


def test_cached_screens_keep_their_resume_behaviour(monkeypatch, tmp_path):
    """`home` and `manage_speakers` stay in SCREENS deliberately: the manager
    already re-reads the DB via `on_screen_resume`, and home is the base screen
    everything pops back to."""
    assert set(tui.ZoomRecorderApp.SCREENS) == {"home", "manage_speakers"}
    assert hasattr(tui.SpeakerManagerScreen, "on_screen_resume")
