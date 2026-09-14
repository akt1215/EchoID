"""Crash-safe file writes.

These files are irreplaceable user data (the voiceprint DB, the learned
glossary, meeting sidecars, the user's own Obsidian notes), so a write that
fails must leave the previous file exactly as it was -- content AND mode.
"""

import json
import os
import stat

import pytest

from core import atomic_write


def test_write_text_replaces_the_content(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("old")
    atomic_write.write_text(str(p), "new")
    assert p.read_text() == "new"


def test_write_text_creates_a_missing_file(tmp_path):
    p = tmp_path / "sub" / "f.txt"
    atomic_write.write_text(str(p), "hello")
    assert p.read_text() == "hello"


def test_write_text_preserves_the_existing_files_permissions(tmp_path):
    # mkstemp creates 0600 and os.replace carries the temp file's mode onto the
    # target, so without care every write silently narrows the user's own files
    # to owner-only -- cumulatively, since the next write starts from 0600 too.
    p = tmp_path / "note.md"
    p.write_text("old")
    os.chmod(str(p), 0o644)
    atomic_write.write_text(str(p), "new")
    assert stat.S_IMODE(os.stat(str(p)).st_mode) == 0o644


def test_write_text_leaves_the_original_intact_when_the_write_fails(tmp_path, monkeypatch):
    p = tmp_path / "db.json"
    original = json.dumps({"Jordan Lee": [1.0, 0.0]})
    p.write_text(original)

    def boom(*a, **kw):
        raise RuntimeError("disk died mid-write")

    monkeypatch.setattr(atomic_write.os, "replace", boom)
    with pytest.raises(RuntimeError):
        atomic_write.write_text(str(p), "clobbered")
    assert p.read_text() == original


def test_write_text_does_not_leak_a_temp_file_when_the_write_fails(tmp_path, monkeypatch):
    p = tmp_path / "f.txt"
    p.write_text("old")

    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(atomic_write.os, "replace", boom)
    with pytest.raises(RuntimeError):
        atomic_write.write_text(str(p), "new")
    assert [f.name for f in tmp_path.iterdir()] == ["f.txt"]
