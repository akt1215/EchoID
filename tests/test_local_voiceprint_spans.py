import json
import os

import numpy as np
import pytest
import soundfile as sf

from core import ingest
from tools.build_local_voiceprint import candidate_spans, collect, verify_span

SR = 16000


def _stereo(ch0_amp, ch1_amp, seconds=2):
    n = SR * seconds
    return np.stack([np.full(n, ch0_amp), np.full(n, ch1_amp)], axis=1)


def test_mic_dominant_span_is_accepted():
    assert verify_span(_stereo(1.0, 0.1), SR, 0.0, 1.0, dominance=8.0) is True


def test_overlapping_span_is_rejected():
    # ch0/ch1 mean-square ratio here is 4x -- enough for the pipeline's 1.5x
    # transcript label, not enough to put someone else's voice in a voiceprint.
    assert verify_span(_stereo(1.0, 0.5), SR, 0.0, 1.0, dominance=8.0) is False


def test_mono_audio_is_rejected():
    mono = np.zeros((SR, 1))
    assert verify_span(mono, SR, 0.0, 1.0, dominance=8.0) is False


def test_dead_remote_channel_is_rejected():
    """A mic-only file trivially passes any ratio; it proves nothing about who
    is speaking, and its transcript is 100% Me (Local) for that reason."""
    assert verify_span(_stereo(1.0, 0.0), SR, 0.0, 1.0, dominance=8.0) is False


def test_empty_span_is_rejected():
    assert verify_span(_stereo(1.0, 0.1), SR, 1.0, 1.0, dominance=8.0) is False


def test_candidate_spans_keeps_only_long_local_turns():
    sidecar = {"transcript": [
        {"start": 0.0, "end": 3.0, "label": "Me (Local)"},
        {"start": 3.0, "end": 3.4, "label": "Me (Local)"},   # too short
        {"start": 4.0, "end": 9.0, "label": "Alice"},        # not local
    ]}
    assert candidate_spans(sidecar, min_span=1.0) == [(0.0, 3.0)]


class _FakeManager:
    """Stands in for BiometricsManager: `collect` only needs an audio loader
    and an embedder, and loading the real one pulls in the whole ML stack."""

    def __init__(self):
        self.embedded = []

    def _load_audio(self, wav):
        return np.zeros(SR), SR

    def extract_embedding(self, wav, start, end, signal=None, fs=None):
        self.embedded.append((wav, start, end))
        return np.array([1.0, 0.0])


def _write_wav(path, *, stamped):
    """A stereo WAV whose ch0 dominates ch1 by 100x -- every span in it clears
    the 8x enrollment bar -- carrying the recorder's provenance marker or not.
    """
    data = np.stack([np.full(SR * 3, 0.5), np.full(SR * 3, 0.05)], axis=1)
    with sf.SoundFile(str(path), mode="w", samplerate=SR, channels=2,
                      subtype="PCM_16") as f:
        if stamped:
            f.comment = ingest.MARKER
        f.write(data.astype("float32"))


def _write_sidecar(workspace, name, wav):
    (workspace / f"{name}.segments.json").write_text(json.dumps({
        "wav": str(wav),
        "transcript": [{"start": 0.0, "end": 2.5, "label": "Me (Local)"}],
    }))


def test_collect_skips_a_stereo_wav_that_cannot_prove_it_is_ours(tmp_path):
    """Channel count is not provenance -- the inference this branch exists to
    delete. A foreign stereo recording reprocessed before ingest existed has
    "Me (Local)" turns wherever L beat R by 1.5x, i.e. wherever the mix was
    panned left, and those spans are remote voices. Feeding them to the local
    user's voiceprint is exactly the corruption the enrollment bar is meant to
    prevent, and no energy ratio can catch it: the wav really is 2-channel.

    The stamped recording alongside it is the control: if `collect` stopped
    collecting anything at all, this test would still pass without it.
    """
    ours = tmp_path / "meeting_2026-01-01_00-00-00.wav"
    theirs = tmp_path / "quicktime.wav"
    _write_wav(ours, stamped=True)
    _write_wav(theirs, stamped=False)
    _write_sidecar(tmp_path, "meeting_2026-01-01_00-00-00", ours)
    _write_sidecar(tmp_path, "quicktime", theirs)

    manager = _FakeManager()
    out = collect(str(tmp_path), manager, dominance=8.0, min_span=1.0)

    assert [meeting for meeting, _, _ in out] == ["meeting_2026-01-01_00-00-00"]
    assert [wav for wav, _, _ in manager.embedded] == [str(ours)]


def test_collect_honours_a_layout_sidecar_stamp(tmp_path):
    """The legacy migration proves provenance with a `<wav>.layout.json` rather
    than by rewriting gigabytes of audio; the tool must honour it, or every
    pre-marker recording -- which is all the material this print is built
    from -- silently contributes nothing."""
    wav = tmp_path / "meeting_2026-02-02_00-00-00.wav"
    _write_wav(wav, stamped=False)
    st = os.stat(wav)
    (tmp_path / "meeting_2026-02-02_00-00-00.layout.json").write_text(json.dumps(
        {"layout": "dual", "size": st.st_size, "mtime": st.st_mtime}))
    _write_sidecar(tmp_path, "meeting_2026-02-02_00-00-00", wav)

    out = collect(str(tmp_path), _FakeManager(), dominance=8.0, min_span=1.0)

    assert [meeting for meeting, _, _ in out] == ["meeting_2026-02-02_00-00-00"]
