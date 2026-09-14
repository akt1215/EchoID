"""Re-transcribing in place must preserve the user's reviewed work.

The whole reason this tool exists instead of a --from-file reprocess is that a
reprocess reassigns cluster ids and so prunes the ground truth. These pin the
parts that make it safe: reuse the resolved speakers, keep everything but the
transcript, and back up the user's reviewed work before rewriting it.
"""

import json

from tools import retranscribe as rt

DIA = [
    {"start": 0.0, "end": 4.0, "label": "Riley Stone", "cluster": "db::Riley Stone"},
    {"start": 4.0, "end": 8.0, "label": "Me (Local)", "cluster": "SPEAKER_00"},
    {"start": 8.0, "end": 9.0, "label": "Unknown (SPEAKER_03)", "cluster": "SPEAKER_03"},
    {"start": 9.0, "end": 12.0, "label": "Riley Stone", "cluster": "db::Riley Stone"},
]


def test_speaker_segments_reuse_the_reviewed_names():
    segs = rt.speaker_segments(DIA)
    assert [s["speaker"] for s in segs] == [
        "Riley Stone", "Me (Local)", "Unknown (SPEAKER_03)", "Riley Stone"]
    assert segs[0] == {"start": 0.0, "end": 4.0, "speaker": "Riley Stone"}


def test_replace_transcript_section_keeps_summary_and_frontmatter():
    note = ("---\ndate: 2026-07-17T14:03:31\n"
            'participants: ["[[Jordan Lee]]"]\n---\n\n'
            "# Meeting Summary\n- something important  (02:13)\n\n"
            "## Transcript\n**Jordan Lee** (00:00):\nold dirty text \n")
    out = rt.replace_transcript_section(note, "**Jordan Lee** (00:00):\nclean text",
                                        ["Jordan Lee"])
    assert "# Meeting Summary" in out
    assert "- something important  (02:13)" in out
    assert "date: 2026-07-17T14:03:31" in out
    assert "clean text" in out
    assert "old dirty text" not in out


def test_dry_run_writes_nothing(tmp_path, capsys):
    wav = tmp_path / "meeting_2026-07-17_14-03-31.wav"
    wav.write_bytes(b"")
    side = tmp_path / "meeting_2026-07-17_14-03-31.segments.json"
    payload = {"wav": str(wav), "note": "", "diarization": DIA,
               "transcript": [{"start": 0.0, "end": 1.0, "text": "Vocabulary.",
                               "label": "Riley Stone"}]}
    side.write_text(json.dumps(payload))
    (tmp_path / "config.yaml").write_text(json.dumps({
        "transcription": {"backend": "groq", "model": "large-v3-turbo",
                          "device": "cpu", "glossary": ["Mamba"]},
        "identity": {"local_names": ["Alex Morgan"]},
    }))
    rc = rt.main(["--workspace", str(tmp_path), "--dry-run",
                  "--config", str(tmp_path / "config.yaml")])
    assert rc == 0
    assert json.loads(side.read_text()) == payload      # byte-for-byte untouched
    assert "Riley Stone" in capsys.readouterr().out


def test_backup_is_never_overwritten(tmp_path):
    """Re-running after a filter change must not clobber the original with the
    already-rewritten file — that would destroy the only copy."""
    f = tmp_path / "m.segments.json"
    f.write_text("original")
    rt._backup_once(str(f))
    f.write_text("rewritten")
    rt._backup_once(str(f))
    assert (tmp_path / "m.segments.json.pre-retranscribe.bak").read_text() == "original"
