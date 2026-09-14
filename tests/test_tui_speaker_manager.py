"""End-to-end (Textual Pilot) check for the Manage Speakers screen.

Standalone screen: reads only saved data (voiceprint DB, sidecars, notes) and
never runs the pipeline. These tests exercise the index-based row ids end to
end (a name-based id would crash on "Speaker 1"), the two-press
preview-then-apply rename flow, the "no clip" status branch, and the
zero-identities degrade-gracefully path.
"""

import asyncio
import json

import numpy as np
import soundfile as sf
from textual.widgets import Input, Static

import zoomrecorder_tui as tui

MEAN = [5.0, 5.0, 5.0, 5.0]
ALICE = [1.0, 0.0, 0.0, 0.0]   # centered-space voiceprints, as in test_speaker_directory.py
BOB = [0.0, 1.0, 0.0, 0.0]     # "Speaker 1" -- never matched in the sidecar below


def _fake_config(workspace):
    return {
        "paths": {"workspace": workspace, "obsidian_vault": "vault",
                  "speaker_db": "speakers.json"},
        "audio": {"capture_backend": "sck", "sample_rate": 48000},
        "visual": {"polling_interval": 2.0, "zoom_min_window_size": 300},
        "biometrics": {"placeholder_min_duration": 60.0, "db_threshold": 0.60},
        "llm": {"model": "test-model", "timeout": 120},
    }


def _turn(start, end, cluster, label, base):
    return {"idx": 0, "start": start, "end": end, "label": label, "cluster": cluster,
            "source": "db", "embedding": list(np.array(MEAN) + np.array(base))}


def _make_workspace(tmp_path, with_speakers=True):
    """A workspace with one meeting: Alice (enrolled, matched, has a real
    clip) and, optionally, "Speaker 1" (enrolled but never matched in this
    sidecar, so it surfaces with zero clips -- exercises the "no clip"
    branch)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    wav = tmp_path / "m1.wav"
    sr = 8000
    sf.write(str(wav), np.zeros(sr * 2, dtype="float32"), sr)

    (tmp_path / "embedding_mean.json").write_text(
        json.dumps({"mean": MEAN, "count": 10, "meetings": []}))
    if with_speakers:
        (tmp_path / "speakers.json").write_text(
            json.dumps({"Alice": ALICE, "Speaker 1": BOB}))
        (tmp_path / "m1.segments.json").write_text(json.dumps({
            "wav": str(wav), "note": str(vault / "m1.md"),
            "diarization": [_turn(0, 30, "db::Alice", "Alice", ALICE)],
            "transcript": [],
        }))
    return {"workspace": str(tmp_path), "vault": str(vault)}


def _boot(monkeypatch, workspace):
    fake_devs = [{"name": "Mic", "max_input_channels": 1, "max_output_channels": 0}]
    monkeypatch.setattr(tui.sd, "query_devices", lambda *a, **k: fake_devs)
    monkeypatch.setattr(tui, "auto_detect_devices", lambda: (0, None))
    monkeypatch.setattr(tui, "load_config", lambda: _fake_config(workspace))
    # Mounting Home fires Select.Changed; without this the TUI persists the
    # fake value into the repo's real config.yaml (conftest fails the run).
    monkeypatch.setattr(tui, "save_config", lambda *a: None)
    # Never actually shell out to afplay in a test.
    monkeypatch.setattr("core.clip_player.subprocess.Popen", lambda *a, **k: None)


def test_speaker_manager_mounts_and_lists_identities(monkeypatch, tmp_path):
    ws = _make_workspace(tmp_path)
    _boot(monkeypatch, ws["workspace"])

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            # Sorted longest-speaking first: Alice (30s, matched) before
            # Speaker 1 (0s, unmatched) -- so row 0 is Alice, row 1 is Speaker 1.
            names = [app.screen.query_one(f"#mname_{i}", Input).value for i in (0, 1)]
            assert names == ["Alice", "Speaker 1"]
            # The whole point of index-based ids: a name containing a space
            # ("Speaker 1") must not have blown up widget construction.
            app.screen.query_one("#mplay_0")
            app.screen.query_one("#mren_1")

    asyncio.run(scenario())


def test_speaker_manager_play_reports_no_clip_when_unmatched(monkeypatch, tmp_path):
    ws = _make_workspace(tmp_path)
    _boot(monkeypatch, ws["workspace"])

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            await pilot.click("#mplay_1")   # "Speaker 1" -- never matched, no clip
            await pilot.pause()
            status = app.screen.query_one("#mgr_status", Static).content
            assert "No recorded clip found" in str(status)

    asyncio.run(scenario())


def test_speaker_manager_rename_previews_then_applies(monkeypatch, tmp_path):
    ws = _make_workspace(tmp_path)
    _boot(monkeypatch, ws["workspace"])
    db_path = tmp_path / "speakers.json"

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            name_input = app.screen.query_one("#mname_1", Input)   # "Speaker 1"
            name_input.value = "Jane Doe"

            # First press: preview only -- must not touch the voiceprint DB.
            await pilot.click("#mren_1")
            await pilot.pause()
            status = str(app.screen.query_one("#mgr_status", Static).content)
            assert "notes" in status and "mentions" in status
            db_after_preview = json.loads(db_path.read_text())
            assert "Jane Doe" not in db_after_preview
            assert "Speaker 1" in db_after_preview

            # Second press on the same row/value: applies for real. A Button
            # ignores a click that lands while its "-active" click animation
            # (0.2s) is still showing, so wait it out before clicking again.
            await pilot.pause(0.3)
            await pilot.click("#mren_1")
            await pilot.pause()
            db_after_apply = json.loads(db_path.read_text())
            assert "Jane Doe" in db_after_apply
            assert "Speaker 1" not in db_after_apply

            status_after = str(app.screen.query_one("#mgr_status", Static).content)
            assert "Renamed to Jane Doe" in status_after

    asyncio.run(scenario())


def test_speaker_manager_handles_zero_identities(monkeypatch, tmp_path):
    ws = _make_workspace(tmp_path, with_speakers=False)
    _boot(monkeypatch, ws["workspace"])

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            rows = app.screen.query_one("#mgr_rows")
            texts = [str(s.content) for s in rows.query(Static)]
            assert any("No enrolled speakers yet." in t for t in texts)

    asyncio.run(scenario())


def test_speaker_manager_reloads_disk_state_on_revisit(monkeypatch, tmp_path):
    """The screen is cached by Textual (registered by name in SCREENS), so
    popping it back to Home only suspends it -- on_mount never fires again on
    a later push. If the screen doesn't independently reload on revisit, a
    session shaped like Home -> Manage Speakers -> Record -> Process ->
    Manage Speakers sees a stale directory: a voiceprint enrolled or updated
    by the intervening meeting is invisible, and worse, a rename from that
    stale snapshot would silently drop it from speakers.json on write-back.
    This reproduces the revisit half of that: mutate the workspace on disk
    while the screen is popped, then push it again and require the newly
    added identity to appear.
    """
    ws = _make_workspace(tmp_path)
    _boot(monkeypatch, ws["workspace"])
    db_path = tmp_path / "speakers.json"

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            names_before = [e["name"] for e in app.screen._people]
            assert names_before == ["Alice", "Speaker 1"]

            app.pop_screen()
            await pilot.pause()

            # Simulate the intervening Record -> Process cycle enrolling a new
            # voiceprint directly in speakers.json while the screen sat popped.
            db = json.loads(db_path.read_text())
            db["Carol"] = [0.0, 0.0, 1.0, 0.0]
            db_path.write_text(json.dumps(db))

            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            names_after = [e["name"] for e in app.screen._people]
            assert "Carol" in names_after, (
                f"revisit still shows the stale snapshot from first mount: {names_after}")
            assert len(names_after) == 3

    asyncio.run(scenario())


def test_speaker_manager_rename_after_revisit_keeps_intervening_voiceprint(monkeypatch, tmp_path):
    """The consequence of the staleness bug, not just its symptom: apply_rename
    writes the WHOLE in-memory db dict back over speakers.json. If a revisit
    doesn't reload, renaming after Carol was enrolled behind the screen's back
    would silently erase Carol from speakers.json on write-back. Assert she
    survives the rename."""
    ws = _make_workspace(tmp_path)
    _boot(monkeypatch, ws["workspace"])
    db_path = tmp_path / "speakers.json"

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            app.pop_screen()
            await pilot.pause()

            db = json.loads(db_path.read_text())
            db["Carol"] = [0.0, 0.0, 1.0, 0.0]
            db_path.write_text(json.dumps(db))

            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            row = next(i for i, e in enumerate(app.screen._people) if e["name"] == "Speaker 1")
            app.screen.query_one(f"#mname_{row}", Input).value = "Jane Doe"
            await pilot.click(f"#mren_{row}")   # preview
            await pilot.pause()
            await pilot.pause(0.3)              # past the button's click-animation guard
            await pilot.click(f"#mren_{row}")   # apply
            await pilot.pause()

            db_after = json.loads(db_path.read_text())
            assert "Carol" in db_after, "the intervening voiceprint was clobbered by the rename"
            assert "Jane Doe" in db_after
            assert "Speaker 1" not in db_after

    asyncio.run(scenario())


def test_speaker_manager_revisit_clears_stale_pending_preview(monkeypatch, tmp_path):
    """A rename preview computed against the old snapshot must not survive into
    a fresh visit: press Rename once (preview only), pop without confirming,
    then revisit and confirm _pending was cleared by the reload."""
    ws = _make_workspace(tmp_path)
    _boot(monkeypatch, ws["workspace"])

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            app.screen.query_one("#mname_1", Input).value = "Jane Doe"
            await pilot.click("#mren_1")   # preview only -- sets self._pending
            await pilot.pause()
            assert app.screen._pending is not None

            app.pop_screen()
            await pilot.pause()
            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()

            assert app.screen._pending is None, (
                "a preview from a previous visit survived into the fresh one")

    asyncio.run(scenario())


def test_every_identity_can_be_reached_by_scrolling(monkeypatch, tmp_path):
    """Same defect as the review picker: a row container without `height: auto`
    takes the viewport height and clips its children, so the scrollable parent
    has nothing to scroll and every speaker past the first screenful is mounted,
    rendered and unclickable. With 16 enrolled prints this is the live state of
    the real DB, not a hypothetical."""
    ws = _make_workspace(tmp_path)
    db = {f"Person {i:02d}": ALICE for i in range(16)}
    (tmp_path / "speakers.json").write_text(json.dumps(db))
    _boot(monkeypatch, ws["workspace"])
    captured = {}

    async def scenario():
        app = tui.ZoomRecorderApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.push_screen("manage_speakers")
            await pilot.pause()
            await pilot.pause()
            captured["rows"] = len(app.screen.query("#mgr_rows > *"))
            captured["scroll"] = app.screen.query_one("#manager").max_scroll_y

    asyncio.run(scenario())
    assert captured["rows"] == 16
    assert captured["scroll"] > 0, "the speakers below the fold cannot be reached"
