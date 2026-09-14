"""Detect a capture channel that recorded noise instead of audio.

`AudioRecorder` writes ch0 = local mic, ch1 = remote/system. When the system
capture fails it does not go silent -- it rails, producing a full-scale square
wave. Diarization then finds no speakers and transcription, which downmixes the
two channels, hears only the noise: the quiet mic underneath is buried and
Whisper returns hallucinations ("Thank you." on repeat, subtitle credits).

Sustained clipping is what separates the two states, and it separates them
completely. Measured over 20 recordings, every healthy one clips 0.000% of ch1
samples; the two broken captures clip 22.0% and 32.1%. `CLIP_FRACTION` sits far
from both sides on purpose -- real audio does not clip for whole seconds at a
time, and treating a good channel as dead would throw away every remote speaker.
"""

import numpy as np
import soundfile as sf

# |sample| at or above this is railed. Not 1.0 exactly: resampling and float
# conversion leave values a hair under full scale.
CLIP_LEVEL = 0.999

# Fraction of railed samples above which the channel is noise, not audio.
# Measured bracket: healthy 0.000%, broken 22.0-32.1%.
CLIP_FRACTION = 0.01

# Seconds read per probe, and how many probes are spread across the file. A
# failed capture rails for the whole recording, so a sparse sample is enough and
# avoids reading a gigabyte to answer a yes/no question.
_PROBE_SECONDS = 1
_PROBES = 20


def clipped_fraction(audio_path, channel, probes=_PROBES):
    """Fraction of `channel`'s samples that are railed, sampled across the file.

    Returns 0.0 for a file with no such channel or too short to probe.
    """
    info = sf.info(audio_path)
    if info.channels <= channel:
        return 0.0
    window = int(_PROBE_SECONDS * info.samplerate)
    if info.frames < window:
        return 0.0

    step = max(window, (info.frames - window) // max(probes, 1))
    railed = total = 0
    for start in range(0, info.frames - window, step):
        block, _ = sf.read(audio_path, start=start, stop=start + window)
        if block.ndim == 1:
            return 0.0
        column = block[:, channel]
        railed += int(np.sum(np.abs(column) >= CLIP_LEVEL))
        total += len(column)
    return railed / total if total else 0.0


def is_unusable(audio_path, channel, clip_fraction=CLIP_FRACTION):
    """True when `channel` carries railed noise rather than audio.

    Callers should fall back to the other channel rather than downmixing, which
    would bury it under the noise.
    """
    try:
        return clipped_fraction(audio_path, channel) > clip_fraction
    except (RuntimeError, OSError):
        # An unreadable file is the transcriber's problem to report, not ours;
        # never let a health probe be what fails the run.
        return False


def extract_channel(audio_path, channel, dest):
    """Write `channel` to `dest` as peak-normalized mono, and return `dest`.

    Normalizing matters here: the mic channel of a failed capture is very quiet
    (measured RMS 0.0006 against a healthy 0.05), and Whisper's frontend applies
    no gain of its own, so the speech transcribes far better once it is brought
    up to a normal level.
    """
    data, sr = sf.read(audio_path)
    mono = data[:, channel] if data.ndim == 2 else data
    peak = float(np.abs(mono).max())
    if peak > 0:
        mono = mono * (0.9 / peak)
    sf.write(dest, np.asarray(mono, dtype="float32"), sr)
    return dest
