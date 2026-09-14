"""Groq-hosted Whisper transcription (opt-in cloud backend).

Sends 16 kHz-mono FLAC chunks of the meeting to Groq's `whisper-large-v3-turbo`
and stitches the per-chunk segment timestamps back onto one timeline, so the rest
of the pipeline (channel-energy override, speaker mapping, note rendering) is
unchanged. Chunking keeps each request under the free tier's 25 MB cap.

Only the audio leaves the machine, and only when this backend is selected. The
pure chunk/stitch helpers are unit-tested; the network call is injectable.
"""

import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def chunk_audio(samples, sr, chunk_seconds):
    """Split a 1-D array into (offset_seconds, chunk) pieces of chunk_seconds."""
    n = max(1, int(chunk_seconds * sr))
    return [(i / sr, samples[i:i + n]) for i in range(0, len(samples), n)]


def stitch_segments(chunk_results):
    """Merge per-chunk transcription dicts into one, shifting each chunk's segment
    timestamps by its offset. `chunk_results` is [(offset_seconds, result_dict)]."""
    segments, language = [], "en"
    for offset, res in chunk_results:
        language = res.get("language") or language
        for s in res.get("segments", []):
            segments.append({
                "start": (s.get("start") or 0.0) + offset,
                "end": (s.get("end") or 0.0) + offset,
                "text": (s.get("text") or "").strip(),
            })
    return {"segments": segments, "language": language}


def _to_dict(resp):
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    return resp


class GroqTranscriber:
    def __init__(self, model="whisper-large-v3-turbo", chunk_seconds=600, client=None,
                 style_prompt=None):
        self.model = model
        self.chunk_seconds = chunk_seconds
        self._client = client  # injected in tests; otherwise created lazily
        # One natural sentence, to set punctuation and capitalisation. Never a
        # term list -- see the note in config.yaml.
        self.style_prompt = style_prompt

    def _get_client(self):
        if self._client is None:
            import groq
            self._client = groq.Groq()  # reads GROQ_API_KEY from the environment
        return self._client

    def transcribe_file(self, wav_path):
        data, sr = sf.read(wav_path)
        if data.ndim == 2:
            data = data.mean(axis=1)              # downmix to mono
        if sr != TARGET_SR:                       # downsample to shrink the upload
            g = gcd(int(sr), TARGET_SR)
            data = resample_poly(data, TARGET_SR // g, int(sr) // g)
            sr = TARGET_SR

        client = self._get_client()
        results = []
        for offset, chunk in chunk_audio(np.asarray(data, dtype="float32"), sr,
                                         self.chunk_seconds):
            buf = io.BytesIO()
            sf.write(buf, chunk, sr, format="FLAC")
            kwargs = dict(
                model=self.model,
                file=("chunk.flac", buf.getvalue()),
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
            if self.style_prompt:
                kwargs["prompt"] = self.style_prompt
            resp = client.audio.transcriptions.create(**kwargs)
            results.append((offset, _to_dict(resp)))
        return stitch_segments(results)
