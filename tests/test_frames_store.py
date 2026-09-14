from core.frames_store import frames_dir_for, read_frames_manifest


def test_frames_dir_derived_from_wav():
    assert frames_dir_for("/x/meeting_1.wav") == "/x/meeting_1.frames"


def test_read_manifest_lists_and_sorts_by_elapsed(tmp_path):
    wav = tmp_path / "m.wav"
    d = tmp_path / "m.frames"
    d.mkdir()
    for ms in [5000, 1000, 12345]:
        (d / f"{ms:09d}.jpg").write_bytes(b"x")
    (d / "notes.txt").write_text("ignore")  # non-jpg ignored

    events = read_frames_manifest(str(wav))
    assert [t for t, _ in events] == [1.0, 5.0, 12.345]
    assert all(p.endswith(".jpg") for _, p in events)


def test_read_manifest_empty_when_no_dir(tmp_path):
    assert read_frames_manifest(str(tmp_path / "nope.wav")) == []
