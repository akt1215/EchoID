"""Repair delayed duplicate speech in legacy dual-clock recordings.

Older ScreenCaptureKit recordings used a sounddevice microphone and an SCK
system-audio stream.  Both advertised the same sample rate, but their hardware
clocks drifted, so speaker bleed in the mic became a second, increasingly late
copy of the remote track when the stereo WAV was played or downmixed.

This tool never overwrites its input.  It estimates the affine clock drift from
cross-channel correlation, warps ch0 onto the ch1 timeline, and gates ch0 while
ch1 contains remote speech.  Ch1 is copied bit-for-bit (modulo PCM decoding and
re-encoding); ch0 remains available during system-audio gaps for the local user.

Run:
  .venv/bin/python -m tools.deecho_native_recording workspace/meeting_....wav
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d
from scipy.signal import correlate, correlation_lags

from core import ingest


def _normalized_peak(mic, system, rate, max_lag_seconds):
    mic = mic.astype(np.float64) - float(np.mean(mic))
    system = system.astype(np.float64) - float(np.mean(system))
    norm = float(np.linalg.norm(mic) * np.linalg.norm(system))
    if norm == 0:
        return None
    corr = correlate(mic, system, mode="full", method="fft")
    lags = correlation_lags(len(mic), len(system), mode="full")
    keep = np.abs(lags) <= int(max_lag_seconds * rate)
    corr, lags = corr[keep], lags[keep]
    i = int(np.argmax(np.abs(corr)))
    return -float(lags[i]) / rate, float(abs(corr[i]) / norm)


def estimate_drift(audio_path, window_seconds=30.0, step_seconds=30.0,
                   max_lag_seconds=15.0, analysis_rate=2000,
                   min_correlation=0.08):
    """Return (initial_delay_seconds, drift_seconds_per_second, observations).

    A duplicate appearing at mic time ``t`` appears on the system track at
    ``delay + (1 + drift) * t``.  The robust fit ignores windows without a
    shared signal, so long muted stretches do not bias the clock estimate.
    """
    observations = []
    with sf.SoundFile(audio_path) as wav:
        if wav.channels < 2:
            raise ValueError("repair requires a stereo native recording")
        sr = wav.samplerate
        stride = max(1, round(sr / analysis_rate))
        rate = sr / stride
        window = int(window_seconds * sr)
        step = int(step_seconds * sr)
        for start in range(0, max(0, wav.frames - window + 1), step):
            wav.seek(start)
            data = wav.read(window, dtype="float32", always_2d=True)[::stride]
            if (np.sqrt(np.mean(data[:, 0] ** 2)) < 1e-3 or
                    np.sqrt(np.mean(data[:, 1] ** 2)) < 1e-3):
                continue
            peak = _normalized_peak(data[:, 0], data[:, 1], rate,
                                    max_lag_seconds)
            if peak is None or peak[1] < min_correlation:
                continue
            center = (start + window / 2) / sr
            observations.append((center, peak[0], peak[1]))

    if len(observations) < 3:
        raise ValueError(
            f"could not estimate drift: only {len(observations)} correlated windows")

    points = np.asarray([(t, delay) for t, delay, _ in observations])
    # Iteratively reject correlation aliases and local-speech windows.  The
    # actual clock offset is a straight line; false matches are not.
    keep = np.ones(len(points), dtype=bool)
    for _ in range(4):
        slope, intercept = np.polyfit(points[keep, 0], points[keep, 1], 1)
        residual = np.abs(points[:, 1] - (intercept + slope * points[:, 0]))
        scale = max(0.05, 1.4826 * float(np.median(residual[keep])))
        new_keep = residual <= max(0.25, 3 * scale)
        if np.array_equal(new_keep, keep) or new_keep.sum() < 3:
            break
        keep = new_keep
    slope, intercept = np.polyfit(points[keep, 0], points[keep, 1], 1)
    accepted = [obs for obs, ok in zip(observations, keep) if ok]
    return float(intercept), float(slope), accepted


def _remote_gain(audio_path, sidecar_path, frame_seconds, threshold, padding):
    env = []
    with sf.SoundFile(audio_path) as wav:
        sr = wav.samplerate
        frame = max(1, round(frame_seconds * sr))
        while True:
            data = wav.read(frame * 2000, dtype="float32", always_2d=True)
            if not len(data):
                break
            n = len(data) // frame
            if n:
                system = data[:n * frame, 1].reshape(n, frame)
                env.append(np.sqrt(np.mean(system ** 2, axis=1)))
    active = np.concatenate(env) > threshold

    if sidecar_path and os.path.exists(sidecar_path):
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        for turn in sidecar.get("diarization", []):
            start = max(0, int(float(turn["start"]) / frame_seconds))
            end = min(len(active), int(np.ceil(float(turn["end"]) / frame_seconds)))
            active[start:end] = True

    radius = max(1, int(np.ceil(padding / frame_seconds)))
    active = maximum_filter1d(active.astype(np.uint8), size=2 * radius + 1) > 0
    return (~active).astype(np.float32)


def repair(audio_path, output_path=None, sidecar_path=None, drift=None,
           activity_threshold=1e-3, padding=0.20, frame_seconds=0.02,
           block_seconds=10.0):
    """Write and return a de-echoed stereo copy plus repair statistics."""
    audio_path = os.path.abspath(audio_path)
    if output_path is None:
        stem, ext = os.path.splitext(audio_path)
        output_path = stem + "_deechoed" + ext
    output_path = os.path.abspath(output_path)
    if output_path == audio_path:
        raise ValueError("output must not overwrite the source recording")
    if os.path.exists(output_path):
        raise FileExistsError(output_path)
    if sidecar_path is None:
        sidecar_path = os.path.splitext(audio_path)[0] + ".segments.json"

    if drift is None:
        intercept, slope, observations = estimate_drift(audio_path)
    else:
        intercept, slope = drift
        observations = []
    gain = _remote_gain(audio_path, sidecar_path, frame_seconds,
                        activity_threshold, padding)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with sf.SoundFile(audio_path) as source, sf.SoundFile(audio_path) as mic_source:
        sr = source.samplerate
        block = max(1, int(block_seconds * sr))
        with sf.SoundFile(output_path, mode="w", samplerate=sr, channels=2,
                          subtype=source.subtype) as dest:
            dest.comment = ingest.MARKER
            out_start = 0
            while out_start < source.frames:
                source.seek(out_start)
                original = source.read(min(block, source.frames - out_start),
                                       dtype="float32", always_2d=True)
                out_indices = out_start + np.arange(len(original), dtype=np.float64)
                mic_positions = ((out_indices / sr - intercept) /
                                 (1.0 + slope)) * sr
                lo = max(0, int(np.floor(mic_positions.min())) - 1)
                hi = min(source.frames, int(np.ceil(mic_positions.max())) + 2)
                mic_source.seek(lo)
                mic = mic_source.read(max(0, hi - lo), dtype="float32",
                                      always_2d=True)[:, 0]
                aligned_mic = np.interp(
                    mic_positions, lo + np.arange(len(mic)), mic,
                    left=0.0, right=0.0).astype(np.float32)
                gain_positions = out_indices / (sr * frame_seconds) - 0.5
                block_gain = np.interp(
                    gain_positions, np.arange(len(gain)), gain,
                    left=float(gain[0]), right=float(gain[-1])).astype(np.float32)
                repaired = np.column_stack((aligned_mic * block_gain,
                                            original[:, 1]))
                dest.write(repaired)
                out_start += len(original)

    stats = {
        "initial_delay_seconds": intercept,
        "drift_seconds_per_second": slope,
        "final_delay_seconds": intercept + slope * (sf.info(audio_path).duration),
        "correlated_windows": len(observations),
        "mic_kept_fraction": float(np.mean(gain)),
    }
    return output_path, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path")
    parser.add_argument("--output")
    parser.add_argument("--sidecar")
    args = parser.parse_args(argv)
    output, stats = repair(args.audio_path, args.output, args.sidecar)
    print(output)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
