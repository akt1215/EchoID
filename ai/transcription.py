import os
import tempfile

import certifi

# Ensure SSL certificates work for torch/huggingface downloads on macOS
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import soundfile as sf

from core import channel_health, glossary


class Transcriber:
    """Transcribe a meeting and attach speaker labels.

    Two backends (config-selected): `local` runs WhisperX on CPU (default, audio
    never leaves the machine); `groq` posts the audio to Groq's hosted
    whisper-large-v3-turbo for a large speedup (opt-in — the audio leaves the
    machine). Both return the same (formatted_transcript, structured_segments)
    with the channel-energy "Me (Local)" override applied from the local WAV.
    """

    def __init__(self, backend="local", model_name="large-v3-turbo", device="cpu",
                 compute_type="default", batch_size=16, beam_size=5,
                 glossary_terms=(), style_prompt=None,
                 groq_model="whisper-large-v3-turbo", chunk_seconds=600):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.beam_size = beam_size
        # Canonical spellings applied AFTER recognition. Never sent to the model
        # as a prompt -- that made it recite the list instead of transcribing.
        self.corrector = glossary.TermCorrector(glossary_terms)
        # One natural sentence that makes Whisper punctuate and capitalise. It
        # carries no vocabulary: a term list here is what made the model recite
        # the glossary instead of transcribing (see config.yaml).
        self.style_prompt = style_prompt
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self._groq = None

        if backend == "groq":
            # No WhisperX / CT2 model loaded upfront on the cloud path; the local
            # model is loaded lazily only if Groq fails (fallback below).
            from ai.groq_transcribe import GroqTranscriber
            self._groq = GroqTranscriber(model=groq_model, chunk_seconds=chunk_seconds,
                                         style_prompt=style_prompt)
        else:
            self._ensure_local_model()

    def _ensure_local_model(self):
        if self.model is not None:
            return
        import whisperx
        asr_options = {"beam_size": self.beam_size}
        if self.style_prompt:
            asr_options["initial_prompt"] = self.style_prompt
        print(f"Loading WhisperX model ({self.model_name}) on {self.device}...")
        self.model = whisperx.load_model(
            self.model_name, device=self.device, compute_type=self.compute_type,
            asr_options=asr_options,
        )

    def _load_align_model(self, language):
        import whisperx
        if self.align_model is not None:
            return
        print("Loading forced-alignment model...")
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )

    def _transcribe_local(self, audio_path):
        import whisperx
        self._ensure_local_model()
        audio = whisperx.load_audio(audio_path)
        print(f"Running WhisperX transcription ({self.model_name})...")
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        language = result.get("language", "en")
        self._load_align_model(language)
        return whisperx.align(
            result["segments"], self.align_model, self.align_metadata,
            audio, self.device, return_char_alignments=False,
        )

    def transcribe(self, audio_path, resolved_speaker_segments=None, mixed=False,
                   channel=None):
        """Transcribe audio and merge with the resolved speaker segments. A Groq
        failure (restricted key, rate limit, network, bad chunk) falls back to
        local WhisperX so the run always completes with a usable transcript.

        `mixed` says whether `audio_path` has a local mic channel at all: it
        only changes what an UNRESOLVED segment falls back to (see
        `_format_transcript`), never a speaker diarization or the
        channel-energy override actually assigned.

        `channel` transcribes one channel alone instead of the downmix, for a
        recording whose other channel captured railed noise. Extracting it once
        here keeps both backends unchanged -- they still just read a path.
        """
        source, scratch = audio_path, None
        if channel is not None:
            fd, scratch = tempfile.mkstemp(suffix=".wav", prefix="zr-channel-")
            os.close(fd)
            source = channel_health.extract_channel(audio_path, channel, scratch)
            print(f"Transcribing channel {channel} alone "
                  f"(the other channel captured noise).")
        try:
            if self.backend == "groq":
                try:
                    print("Transcribing via Groq (whisper-large-v3-turbo)...")
                    result = self._groq.transcribe_file(source)
                except Exception as e:
                    print(f"[transcription] Groq failed ({type(e).__name__}: {e}); "
                          f"falling back to local WhisperX.")
                    result = self._transcribe_local(source)
            else:
                result = self._transcribe_local(source)
        finally:
            if scratch:
                os.unlink(scratch)

        self._strip_style_echo(result)
        self._correct_terms(result)
        if resolved_speaker_segments:
            self._assign_by_overlap(result, resolved_speaker_segments)
            self._channel_energy_override(result, audio_path)
        return (self._format_transcript(result, mixed=mixed),
                self._structured_segments(result, mixed=mixed))

    # Leading words of the style prompt that identify an echo of it. Whisper
    # occasionally opens a segment by continuing the prompt instead of
    # transcribing ("The following is a recording of a video of the video, and I
    # think it gives us a pretty good estimation of the effect size." -- the
    # clause after the comma is real speech). Measured at 2 segments in ~7,500
    # across 15 meetings, against 693 for the old term-list prompt.
    _STYLE_ECHO_WORDS = 5

    def _strip_style_echo(self, result):
        """Drop a leading echo of the style prompt from a segment.

        Unlike the term-list echo this replaced, the prompt is a known constant,
        so this matches an exact opening phrase rather than guessing at mangled
        vocabulary -- and only ever strips a PREFIX, up to the first clause
        break. Speech that merely contains the words later in the sentence is
        left alone; nobody opens a sentence with this phrase by accident.
        """
        if not self.style_prompt:
            return
        lead = " ".join(self.style_prompt.split()[: self._STYLE_ECHO_WORDS]).lower()
        kept = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if " ".join(text.lower().split()).startswith(lead):
                cut = min((i for i in (text.find(","), text.find(".")) if i > 0),
                          default=-1)
                text = text[cut + 1:].strip() if cut > 0 else ""
            if not text:
                continue
            seg["text"] = text
            kept.append(seg)
        result["segments"] = kept

    def _correct_terms(self, result):
        """Restore canonical spelling of known names and jargon in each segment.

        Runs before speaker assignment purely so every consumer downstream --
        the note, the sidecar, the LLM summary and the entities it harvests --
        sees the same corrected text.
        """
        if not self.corrector:
            return
        for seg in result.get("segments", []):
            seg["text"] = self.corrector(seg.get("text", ""))

    def _assign_by_overlap(self, result, speaker_segments):
        """Segment-level speaker assignment for backends without word alignment:
        each transcript segment takes the diarization turn it most overlaps."""
        for seg in result.get("segments", []):
            best, best_ov = None, 0.0
            for sp in speaker_segments:
                ov = min(seg.get("end", 0.0), sp["end"]) - max(seg.get("start", 0.0), sp["start"])
                if ov > best_ov:
                    best_ov, best = ov, sp["speaker"]
            if best is not None:
                seg["speaker"] = best

    def _channel_energy_override(self, result, audio_path):
        """If the mic channel (ch0) dominates the remote channel (ch1) for a
        segment, force 'Me (Local)' regardless of diarization. Reads the local
        stereo WAV, so it is independent of which backend transcribed."""
        try:
            audio_data, sr = sf.read(audio_path)
            if audio_data.ndim == 2 and audio_data.shape[1] >= 2:
                for seg in result.get("segments", []):
                    if not seg.get("speaker"):
                        continue
                    s = int(seg["start"] * sr)
                    e = int(seg["end"] * sr)
                    if e - s < sr // 20:
                        continue
                    ch0 = float(np.mean(audio_data[s:e, 0] ** 2))
                    ch1 = float(np.mean(audio_data[s:e, 1] ** 2))
                    if ch0 > ch1 * 1.5:
                        seg["speaker"] = "Me (Local)"
                        for w in seg.get("words", []):
                            w["speaker"] = "Me (Local)"
        except Exception:
            pass

    def _format_transcript(self, whisperx_result, mixed=False):
        """`mixed` picks what an unresolved segment (no diarization overlap, so
        never claimed a speaker) falls back to. Dual-channel recordings only
        ever miss the local user this way -- diarization runs on ch1 alone --
        so "Me (Local)" is the right default there. A mixed file has no local
        channel at all, so that same default would blame an unidentified
        segment on the local user; it falls back to "Unknown" instead, leaving
        it for the user to resolve like any other unmatched speaker.
        """
        fallback = "Unknown" if mixed else "Me (Local)"
        formatted = ""
        current_speaker = None
        for seg in whisperx_result.get("segments", []):
            speaker = seg.get("speaker", fallback)
            if not speaker or speaker == "Unknown":
                speaker = fallback

            start = seg.get("start", 0)
            mins = int(start // 60)
            secs = int(start % 60)
            time_str = f"{mins:02d}:{secs:02d}"

            text = seg.get("text", "").strip()
            if not text:
                continue

            if speaker != current_speaker:
                formatted += f"\n**{speaker}** ({time_str}):\n"
                current_speaker = speaker

            formatted += f"{text} "

        return formatted.strip()

    def _structured_segments(self, whisperx_result, mixed=False):
        """Per-segment transcript records for the review sidecar.

        Mirrors _format_transcript's label logic (including the mode-aware
        unresolved-segment fallback) so the label stored here matches what was
        written into the note, which the review step uses to preserve the
        local-speaker label when it rebuilds the transcript.
        """
        fallback = "Unknown" if mixed else "Me (Local)"
        segs = []
        for seg in whisperx_result.get("segments", []):
            speaker = seg.get("speaker", fallback)
            if not speaker or speaker == "Unknown":
                speaker = fallback
            text = seg.get("text", "").strip()
            if not text:
                continue
            segs.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text,
                "label": speaker,
            })
        return segs
