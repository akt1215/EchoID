"""The TUI must review the sidecar the pipeline actually wrote.

`main.py` names its review sidecar after the wav it consumed, which for an
imported file is ingest's product (`meeting_<ts>.wav`, possibly bumped to
`-2`/`-3` when that name belongs to another source's import) -- never the
`.mp4`/`.m4a` the user picked. A TUI that derives the sidecar from the input
filename instead opens a path that does not exist, and NamerScreen degrades to
"Could not load review data" with its buttons hidden. Since the TUI runs the
pipeline with --no-finalize, nothing else commits either: naming, enrollment
and the summary are all unreachable, and the note keeps its placeholder
summary forever. A legacy `meeting_<ts>.wav` hides this because the two stems
coincide.
"""

import asyncio
import json

import zoomrecorder_tui as tui
from core import review_sidecar


def _fake_config(workspace):
    return {
        "paths": {"workspace": workspace, "obsidian_vault": "vault",
                  "speaker_db": "speakers.json"},
        "audio": {"capture_backend": "sck", "sample_rate": 48000},
        "visual": {"polling_interval": 2.0, "zoom_min_window_size": 300},
        "biometrics": {"placeholder_min_duration": 60.0},
        "llm": {"model": "test-model", "timeout": 120},
    }


class _FakeProc:
    """A stand-in for the pipeline subprocess that replays fixed stdout."""

    def __init__(self, lines):
        self.stdout = iter(lines)
        self.waited = False

    def wait(self):
        self.waited = True
        return 0


def _write_sidecar(path):
    path.write_text(json.dumps({
        "wav": "", "note": "", "mixed": True, "transcript": [],
        "diarization": [],
    }))


def _run(monkeypatch, tmp_path, lines, selected_file):
    """Drive Home -> Processing with a faked pipeline; return the sidecar path
    the Namer screen was opened on."""
    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    monkeypatch.setattr(tui, "load_config", lambda: _fake_config(str(tmp_path)))
    # Mounting Home fires Select.Changed; without this the TUI persists the
    # fake value into the repo's real config.yaml (conftest fails the run).
    monkeypatch.setattr(tui, "save_config", lambda *a: None)
    monkeypatch.setattr(tui.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(lines))

    captured = {}

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.selected_file = selected_file
            app.push_screen("processing")
            await pilot.pause()
            await app.workers.wait_for_complete()
            for _ in range(4):
                await pilot.pause()
            captured["screen"] = app.screen

    asyncio.run(scenario())
    return captured["screen"]


def test_namer_opens_the_sidecar_the_pipeline_announced(monkeypatch, tmp_path):
    """An import: the pipeline consumed `meeting_<ts>-2.wav`, so its sidecar is
    named for that, not for the picked `zoom_0.mp4`. The suffix is chosen by
    the run itself (it depends on what else is already in `imported/`), so
    recomputing the plan cannot recover it -- the path has to come from the
    run's own output."""
    announced = tmp_path / "meeting_2026-01-01_00-00-00-2.segments.json"
    _write_sidecar(announced)
    lines = [
        "--- Reprocessing /w/imported/meeting_2026-01-01_00-00-00-2.wav ---\n",
        "[ingest] layout: mixed (1ch, no mic channel)\n",
        f"Review sidecar written: {announced}\n",
        "--- All Done! ---\n",
    ]

    screen = _run(monkeypatch, tmp_path, lines, str(tmp_path / "zoom_0.mp4"))

    assert isinstance(screen, tui.NamerScreen)
    assert screen._sidecar == str(announced)


def test_the_announcement_survives_a_progress_bar_in_front_of_it(
        monkeypatch, tmp_path):
    """stdout and stderr share one pipe, and a progress bar's frame carries no
    terminator of its own (tqdm leads the next frame with the carriage return),
    so the announcement can arrive with those bytes in front of it. Losing it
    there would silently reinstate the guessed path -- intermittently, and only
    under the real ML stages, where no test would ever see it."""
    announced = tmp_path / "meeting_2026-03-03_10-10-10.segments.json"
    _write_sidecar(announced)
    lines = [
        f"Embedding: 100%|##########| 5/5{review_sidecar.announce(str(announced))}\n",
        "--- All Done! ---\n",
    ]

    screen = _run(monkeypatch, tmp_path, lines, str(tmp_path / "zoom_0.mp4"))

    assert isinstance(screen, tui.NamerScreen)
    assert screen._sidecar == str(announced)


def test_namer_falls_back_to_the_input_stem_when_nothing_is_announced(
        monkeypatch, tmp_path):
    """A run that prints no sidecar line -- an older main.py, or one that died
    before writing it -- must still open the path the input implies rather
    than nothing at all. That fallback is exactly right for a native
    recording, where the wav the pipeline consumes IS the one the user picked.
    """
    wav = tmp_path / "meeting_2026-02-02_09-09-09.wav"
    _write_sidecar(tmp_path / "meeting_2026-02-02_09-09-09.segments.json")
    lines = ["--- Reprocessing ---\n", "--- All Done! ---\n"]

    screen = _run(monkeypatch, tmp_path, lines, str(wav))

    assert isinstance(screen, tui.NamerScreen)
    assert screen._sidecar == str(
        tmp_path / "meeting_2026-02-02_09-09-09.segments.json")


def test_processing_cancel_button(monkeypatch, tmp_path):
    import queue
    q = queue.Queue()

    class _HangingProc:
        def __init__(self):
            self.stdout = iter(q.get, None)

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            q.put(None)

        def poll(self):
            return None

    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    monkeypatch.setattr(tui, "load_config", lambda: _fake_config(str(tmp_path)))
    monkeypatch.setattr(tui, "save_config", lambda *a: None)
    monkeypatch.setattr(tui.subprocess, "Popen", lambda *a, **k: _HangingProc())

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.selected_file = str(tmp_path / "m.wav")
            app.push_screen("processing")
            await pilot.pause()
            app.screen.query_one("#cancel", tui.Button).press()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            return app.screen, app.batch_results

    screen, results = asyncio.run(scenario())
    assert isinstance(screen, tui.SummaryScreen)
    assert results[0]["status"] == tui.BATCH_STATUS_CANCELED
