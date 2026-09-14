import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CONFIG = os.path.join(ROOT, "config.yaml")


@pytest.fixture(autouse=True)
def never_write_the_real_config(monkeypatch):
    """Fail loudly if a test writes the repo's own config.yaml.

    Three TUI test files patched `load_config` but not `save_config`, so mounting
    the Home screen persisted the fake config's value into the real file — the
    backend silently flipped to `local` on every full-suite run, and the next
    real recording would have been transcribed the slow way. Reading the file is
    what missed it for two sessions; nothing pointed at the tests.

    Same hazard class as the test that once removed workspace/speakers.json:
    tests must never write user data. A test that genuinely exercises saving
    should pass a tmp_path config instead.
    """
    from core import config_store

    real_save = config_store.save_config

    def guarded(path, section, key, value):
        if os.path.abspath(path) == REAL_CONFIG:
            raise AssertionError(
                f"test wrote the real config.yaml ({section}.{key}={value!r}); "
                "patch save_config or pass a tmp_path config")
        return real_save(path, section, key, value)

    monkeypatch.setattr(config_store, "save_config", guarded)
    # The TUI imports the name directly, so patching the module alone misses it.
    try:
        import zoomrecorder_tui
    except Exception:
        return
    monkeypatch.setattr(zoomrecorder_tui, "save_config", guarded, raising=False)


@pytest.fixture
def sidecar(tmp_path):
    """A review sidecar fixture with two synthetic voice clusters lumped under
    one label, plus a matching note, wired to temp paths.

    `Unknown (SPEAKER_01)` has four turns whose 8-dim embeddings form two tight
    clusters (near [1,0,...] and near [0,1,...]), so `split(k=2)` should separate
    them 2/2. Two of the transcript segments fall under that label at timestamps
    that overlap the two clusters, so a split+rename+commit yields two different
    names across the transcript. One `Me (Local)` line proves preservation.
    """
    src = os.path.join(os.path.dirname(__file__), "fixtures", "sample_sidecar.json")
    with open(src) as f:
        data = json.load(f)

    note = tmp_path / "note.md"
    note.write_text(
        "---\n"
        "date: 2026-07-05T05:29:14\n"
        "type: meeting\n"
        'participants: ["[[Me (Local)]]"]\n'
        "tags: [zoom, auto-generated]\n"
        "---\n\n"
        "# Meeting Summary\n- something\n\n"
        "## Action Items\n\n"
        "## Transcript\n"
        "**Me (Local)** (00:00):\nlocal intro \n"
        "**Alice** (00:03):\nalice speaking \n"
        "**Unknown (SPEAKER_01)** (00:05):\ncluster one line \n"
    )

    data["note"] = str(note)
    data["wav"] = ""  # playback is not exercised in core tests
    p = tmp_path / "m.segments.json"
    p.write_text(json.dumps(data))

    return {"sidecar": str(p), "db": str(tmp_path / "speakers.json"), "note": str(note)}
