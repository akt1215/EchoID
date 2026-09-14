"""An UNRESOLVED transcript segment (no diarization overlap, so it never
claims a speaker) must fall back to "Me (Local)" in dual mode but "Unknown" in
mixed mode. A mixed file has no local channel at all, so defaulting an
unidentified segment to the local user was blaming it on someone who cannot
possibly have spoken it -- surfaced by a real 66-minute mixed-mode import
where 7 unresolved segments were mislabeled "Me (Local)" despite the
channel-energy override being structurally unable to fire on mono audio.
"""
from ai.transcription import Transcriber

_UNRESOLVED = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}


def test_mixed_mode_unresolved_segment_falls_back_to_unknown():
    formatted = Transcriber._format_transcript(None, _UNRESOLVED, mixed=True)
    structured = Transcriber._structured_segments(None, _UNRESOLVED, mixed=True)
    assert "**Unknown**" in formatted
    assert "Me (Local)" not in formatted
    assert structured[0]["label"] == "Unknown"


def test_dual_mode_unresolved_segment_still_falls_back_to_me_local():
    """The reference path must not regress: dual-channel diarization only ever
    misses the local user's own (ch0) speech, so defaulting to "Me (Local)"
    here is correct and must survive this change unchanged."""
    formatted = Transcriber._format_transcript(None, _UNRESOLVED, mixed=False)
    structured = Transcriber._structured_segments(None, _UNRESOLVED, mixed=False)
    assert "**Me (Local)**" in formatted
    structured_default = Transcriber._structured_segments(None, _UNRESOLVED)
    assert structured[0]["label"] == "Me (Local)"
    assert structured_default[0]["label"] == "Me (Local)"  # mixed defaults False


def test_an_explicit_me_local_label_survives_the_mixed_fallback():
    """The mode-aware fallback must only change what an UNRESOLVED segment
    becomes -- a segment the channel-energy override already stamped
    "Me (Local)" is a real signal, not a default, and must pass through
    untouched even if `mixed=True` were ever passed alongside one (structurally
    impossible in production, since the override cannot fire on mono audio,
    but this pins the fallback logic itself rather than relying on that)."""
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                            "speaker": "Me (Local)"}]}
    formatted = Transcriber._format_transcript(None, result, mixed=True)
    structured = Transcriber._structured_segments(None, result, mixed=True)
    assert "**Me (Local)**" in formatted
    assert structured[0]["label"] == "Me (Local)"


class _FakeGroqTranscriber:
    def transcribe_file(self, path):
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "unresolved speech"}],
                "language": "en"}


def test_transcribe_forwards_mixed_to_the_unresolved_fallback(tmp_path):
    """End-to-end: main.py's `mixed` must actually reach the fallback through
    Transcriber.transcribe -- an empty resolved_speaker_segments list skips
    overlap assignment and the channel-energy override entirely, leaving the
    segment genuinely unresolved."""
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "m.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)  # mono
    t = Transcriber(backend="groq")
    t._groq = _FakeGroqTranscriber()

    _, structured = t.transcribe(str(wav), resolved_speaker_segments=[], mixed=True)
    assert structured[0]["label"] == "Unknown"

    _, structured = t.transcribe(str(wav), resolved_speaker_segments=[], mixed=False)
    assert structured[0]["label"] == "Me (Local)"
