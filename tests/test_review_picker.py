"""Re-opening speaker review on a meeting that was already processed.

The visual name reader once labeled a remote cluster with the local user's own
name, read off his self-view tile. That is fixed, but three shipped meetings
still carry the wrong name, and neither repair path fits: rename-everywhere is
global by name (it would fold his own voiceprint into the other speaker's), and
a full reprocess reassigns the cluster ids the ground truth is keyed by.

`SpeakerReview` already does the right thing on a single sidecar. These tests
cover the listing that makes it reachable.
"""

import asyncio
import json

import zoomrecorder_tui as tui
from core import review_sidecar


def _sidecar(tmp_path, stem, labels, turns=3):
    """A sidecar whose transcript carries `labels`, cycled over `turns` rows."""
    path = tmp_path / f"{stem}.segments.json"
    path.write_text(json.dumps({
        "wav": str(tmp_path / f"{stem}.wav"),
        "note": "", "mixed": False,
        "diarization": [{"cluster": "SPEAKER_00", "label": labels[0],
                         "start": 0.0, "end": 1.0}],
        "transcript": [{"start": float(i), "end": i + 1.0, "text": "hi",
                        "label": labels[i % len(labels)]}
                       for i in range(turns)],
    }))
    return path


def test_lists_meetings_newest_first_with_their_labels(tmp_path):
    _sidecar(tmp_path, "meeting_2026-07-17_14-03-31", ["Alex Morgan", "Me (Local)"])
    _sidecar(tmp_path, "meeting_2026-07-22_10-13-14", ["Jordan Lee"])

    got = review_sidecar.list_reviewable(str(tmp_path))

    assert [e["meeting"] for e in got] == ["meeting_2026-07-22_10-13-14",
                                           "meeting_2026-07-17_14-03-31"]
    assert got[1]["labels"] == ["Alex Morgan", "Me (Local)"]
    assert got[1]["turns"] == 3
    assert got[1]["path"].endswith("meeting_2026-07-17_14-03-31.segments.json")


def test_unparseable_sidecar_is_skipped_not_fatal(tmp_path):
    _sidecar(tmp_path, "meeting_2026-07-17_14-03-31", ["Jordan Lee"])
    (tmp_path / "meeting_2026-07-18_09-00-00.segments.json").write_text("{ truncated")

    got = review_sidecar.list_reviewable(str(tmp_path))

    assert [e["meeting"] for e in got] == ["meeting_2026-07-17_14-03-31"]


def test_falls_back_to_diarization_labels_when_the_transcript_is_empty(tmp_path):
    """A run stopped before transcription still has clusters worth renaming."""
    path = tmp_path / "meeting_2026-07-17_14-03-31.segments.json"
    path.write_text(json.dumps({
        "wav": "", "note": "", "mixed": False,
        "diarization": [{"cluster": "SPEAKER_00", "label": "Alex Morgan",
                         "start": 0.0, "end": 9.0}],
        "transcript": [],
    }))

    got = review_sidecar.list_reviewable(str(tmp_path))

    assert got[0]["labels"] == ["Alex Morgan"]
    assert got[0]["turns"] == 1


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


def test_picking_a_meeting_opens_the_namer_on_its_sidecar(monkeypatch, tmp_path):
    sidecar = _sidecar(tmp_path, "meeting_2026-07-17_14-03-31", ["Alex Morgan"])
    _boot(monkeypatch, tmp_path)
    captured = {}

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("review_picker")
            await pilot.pause()
            await pilot.pause()
            await pilot.click("#review_0")
            await pilot.pause()
            captured["screen"] = app.screen

    asyncio.run(scenario())
    screen = captured["screen"]
    assert isinstance(screen, tui.NamerScreen)
    assert screen._sidecar == str(sidecar)


def test_every_meeting_can_be_reached_by_scrolling(monkeypatch, tmp_path):
    """A row container without `height: auto` takes the viewport's height and
    CLIPS its children instead of growing, so the scrollable parent sees nothing
    to scroll and every meeting past the first screenful is unreachable -- it is
    mounted, rendered, and impossible to click."""
    for day in range(1, 17):
        _sidecar(tmp_path, f"meeting_2026-07-{day:02d}_09-00-00", ["Jordan Lee"])
    _boot(monkeypatch, tmp_path)
    captured = {}

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen("review_picker")
            await pilot.pause()
            await pilot.pause()
            captured["rows"] = len(app.screen.query("#review_rows > *"))
            captured["scroll"] = app.screen.query_one("#reviewpicker").max_scroll_y

    asyncio.run(scenario())
    assert captured["rows"] == 16
    assert captured["scroll"] > 0, "the meetings below the fold cannot be reached"


def test_picker_explains_an_empty_workspace(monkeypatch, tmp_path):
    _boot(monkeypatch, tmp_path)
    captured = {}

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("review_picker")
            await pilot.pause()
            await pilot.pause()
            captured["text"] = str(app.screen.query_one("#review_rows").children[0].content)

    asyncio.run(scenario())
    assert "No processed meetings" in captured["text"]
