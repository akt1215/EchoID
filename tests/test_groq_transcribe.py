"""Groq transcription: chunking, timestamp stitching, and file flow (no network)."""

import numpy as np
import soundfile as sf

from ai.groq_transcribe import GroqTranscriber, chunk_audio, stitch_segments

STYLE_PROMPT = "The following is a recording of a research group meeting."


def test_chunk_audio_splits_with_offsets():
    sr = 1000
    samples = np.arange(2500.0)
    chunks = chunk_audio(samples, sr, chunk_seconds=1.0)  # 1000 samples per chunk
    assert [round(o, 3) for o, _ in chunks] == [0.0, 1.0, 2.0]
    assert [len(c) for _, c in chunks] == [1000, 1000, 500]


def test_stitch_segments_offsets_and_concatenates():
    r0 = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}], "language": "en"}
    r1 = {"segments": [{"start": 0.5, "end": 1.5, "text": "world"}]}
    out = stitch_segments([(0.0, r0), (10.0, r1)])
    assert out["segments"][0] == {"start": 0.0, "end": 1.0, "text": "hello"}
    assert out["segments"][1]["start"] == 10.5 and out["segments"][1]["end"] == 11.5
    assert out["segments"][1]["text"] == "world"
    assert out["language"] == "en"


class _FakeTranscriptions:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        return {"segments": [{"start": 0.0, "end": 0.5, "text": "chunk"}],
                "language": "en"}


class _FakeGroq:
    def __init__(self):
        self.calls = []
        self.audio = type("A", (), {"transcriptions": _FakeTranscriptions(self)})()


def test_transcribe_file_chunks_and_stitches(tmp_path):
    wav = tmp_path / "m.wav"
    sr = 16000
    sf.write(str(wav), np.zeros(sr * 3, dtype="float32"), sr)  # 3 s of mono audio
    fake = _FakeGroq()
    gt = GroqTranscriber(model="whisper-large-v3-turbo", chunk_seconds=1, client=fake)

    res = gt.transcribe_file(str(wav))

    assert len(fake.calls) == 3                     # three 1-second chunks posted
    assert len(res["segments"]) == 3
    assert res["segments"][1]["start"] == 1.0       # second chunk offset by 1 s
    assert res["segments"][2]["start"] == 2.0


def test_no_prompt_is_sent_when_none_is_configured(tmp_path):
    wav = tmp_path / "m.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    fake = _FakeGroq()
    GroqTranscriber(chunk_seconds=10, client=fake).transcribe_file(str(wav))
    assert "prompt" not in fake.calls[0]


def test_the_style_prompt_is_one_sentence_not_a_term_list(tmp_path):
    """Whisper copies the style of its prompt, so a sentence buys punctuation
    (30% of long segments come back unpunctuated without one, 7% with). A
    comma-separated term list instead puts it in list-continuation mode and it
    recites the terms over real speech -- 7-17% of every meeting's segments.
    The shape of this value is the safety property, so pin it."""
    wav = tmp_path / "m.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    fake = _FakeGroq()
    GroqTranscriber(chunk_seconds=10, client=fake,
                    style_prompt=STYLE_PROMPT).transcribe_file(str(wav))
    sent = fake.calls[0]["prompt"]
    assert sent == STYLE_PROMPT
    assert sent.count(",") <= 1 and sent.endswith(".")


def test_the_configured_style_prompt_is_still_a_sentence():
    """Guards the config itself: someone re-adding names here would silently
    reintroduce the hallucination this whole change removed."""
    import yaml
    configured = yaml.safe_load(open("config.example.yaml"))["transcription"]["style_prompt"]
    assert configured.count(",") <= 1, "style_prompt must not become a term list"
    assert configured.endswith("."), "style_prompt must be a natural sentence"


class _FakeGroqTranscriber:
    def transcribe_file(self, path):
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hello there"}],
                "language": "en"}


def _stereo_wav(tmp_path, mic_gain, remote_gain, sr=16000):
    wav = tmp_path / "m.wav"
    data = np.zeros((sr * 2, 2), dtype="float32")
    data[:, 0] = mic_gain      # ch0 = local mic
    data[:, 1] = remote_gain   # ch1 = remote/system
    sf.write(str(wav), data, sr)
    return str(wav)


def test_transcriber_groq_assigns_speaker_by_overlap(tmp_path):
    from ai.transcription import Transcriber
    wav = _stereo_wav(tmp_path, mic_gain=0.0, remote_gain=0.5)  # remote dominant
    t = Transcriber(backend="groq")          # no whisperx, no key needed
    t._groq = _FakeGroqTranscriber()
    segs = [{"start": 0.0, "end": 1.0, "speaker": "Riley Stone"}]
    formatted, structured = t.transcribe(wav, segs)
    assert "Riley Stone" in formatted and "hello there" in formatted
    assert structured[0]["label"] == "Riley Stone"


def test_transcriber_groq_channel_energy_forces_me_local(tmp_path):
    from ai.transcription import Transcriber
    wav = _stereo_wav(tmp_path, mic_gain=0.5, remote_gain=0.0)  # mic dominant
    t = Transcriber(backend="groq")
    t._groq = _FakeGroqTranscriber()
    segs = [{"start": 0.0, "end": 1.0, "speaker": "Riley Stone"}]
    _, structured = t.transcribe(wav, segs)
    assert structured[0]["label"] == "Me (Local)"


class _BoomGroq:
    def transcribe_file(self, path):
        raise RuntimeError("400 organization_restricted")


def test_transcriber_falls_back_to_local_when_groq_fails(tmp_path, monkeypatch):
    # A Groq error (e.g. restricted org) must NOT crash the pipeline — it falls
    # back to local WhisperX. Stub the local path so no model is loaded.
    from ai.transcription import Transcriber
    wav = _stereo_wav(tmp_path, mic_gain=0.0, remote_gain=0.5)
    t = Transcriber(backend="groq")
    t._groq = _BoomGroq()
    monkeypatch.setattr(t, "_transcribe_local", lambda path: {
        "segments": [{"start": 0.0, "end": 1.0, "text": "local fallback text"}],
        "language": "en"})
    segs = [{"start": 0.0, "end": 1.0, "speaker": "Riley Stone"}]
    formatted, structured = t.transcribe(wav, segs)
    assert "local fallback text" in formatted
    assert structured[0]["label"] == "Riley Stone"


def _echo_transcriber(segments):
    from ai.transcription import Transcriber
    t = Transcriber(backend="groq", style_prompt=STYLE_PROMPT)
    result = {"segments": [{"start": 0.0, "end": 1.0, "text": s} for s in segments]}
    t._strip_style_echo(result)
    return [s["text"] for s in result["segments"]]


def test_a_leading_style_prompt_echo_is_stripped():
    """Real output from meeting_2026-07-08_14-11-54: the model continued the
    prompt, then transcribed actual speech after the comma."""
    assert _echo_transcriber([
        "The following is a recording of a video of the video, and I think it "
        "gives us a pretty good estimation of the effect size."
    ]) == ["and I think it gives us a pretty good estimation of the effect size."]


def test_a_segment_that_is_only_echo_is_dropped():
    assert _echo_transcriber([STYLE_PROMPT, "real speech here"]) == ["real speech here"]


def test_the_phrase_mid_sentence_is_left_alone():
    """"research group meeting" is something people actually say; only an echo
    at the very start of a segment is an echo."""
    kept = _echo_transcriber([
        "If you have a research group meeting, IPMI, whether we publish or not.",
        "The recording is fine, by the way.",
    ])
    assert len(kept) == 2 and kept[0].startswith("If you have a research group")
