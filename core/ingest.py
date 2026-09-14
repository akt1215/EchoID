"""Normalize any input recording into a shape the pipeline understands.

The pipeline is built on a dual-channel invariant: ch0 = local mic, ch1 =
remote. Only recordings this tool made have that layout. Everything else — a
Zoom cloud mp4, a phone voice memo, a QuickTime stereo capture — is a single
mixed track where the local user is just another voice.

Rather than thread a mode flag through the pipeline, ingest normalizes every
foreign input to a MONO wav, and the pipeline derives its mode from the channel
count. That derivation is structural, not a heuristic: `core/audio_recorder.py`
is the only wav writer here and hardcodes `channels=2`, silence-padding a
stalled producer, so a native recording can never be mono.
"""

import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import soundfile as sf

from core.frames_store import frames_dir_for

VIDEO_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})

TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"
_MEETING_STEM = re.compile(r"^meeting_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")
_LOOSE_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")

IMPORTED_DIR = "imported"

# The only thing that entitles a WAV to be read as ch0=mic/ch1=remote.
# core/audio_recorder.py imports this back as AudioRecorder.MARKER rather than
# restating the literal, so there is exactly one string, not two that can
# drift apart. Defined here rather than the other way around so that this
# module -- which tools/stamp_legacy_recordings.py loads just to read a
# metadata string -- never pulls in the audio hardware stack
# (sounddevice, sck_capture, ...) that core/audio_recorder.py needs.
MARKER = "ZoomRecorder dual-channel: ch0=local mic, ch1=remote"


class IngestError(Exception):
    """An input this pipeline cannot process, reported before any stage runs."""


@dataclass(frozen=True)
class IngestPlan:
    source: str
    # The *candidate* conversion path, not necessarily the final answer: when
    # `action` is "convert", `run` may resolve to a different, suffixed path
    # (`meeting_<ts>-2.wav`, ...) if this candidate turns out to belong to a
    # different source's import. Callers that need the real output path must
    # use what `run` returns, not this field, for a "convert" plan.
    target: str
    action: str            # "passthrough" | "convert"
    mixed: bool
    extract_frames: bool
    timestamp: str


def derive_timestamp(source, mtime):
    """Meeting time for `source`, as a `meeting_<ts>.wav` timestamp string.

    Never the import time: a note dated when the file was converted is wrong in
    a way nobody notices. Prefers a native stem, then a date written into the
    filename (Zoom's cloud exports often carry one) with the clock from mtime,
    then mtime entirely.
    """
    stem = os.path.splitext(os.path.basename(source))[0]
    m = _MEETING_STEM.match(stem)
    if m:
        return m.group(1)
    when = datetime.datetime.fromtimestamp(mtime)
    m = _LOOSE_DATE.search(stem)
    if m:
        return f"{m.group(1)}_{when.strftime('%H-%M-%S')}"
    return when.strftime(TIMESTAMP_FMT)


def layout_marker_path(wav_path):
    """Sidecar asserting a legacy recording's layout, beside the WAV."""
    return os.path.splitext(wav_path)[0] + ".layout.json"


def is_native(source):
    """True if `source` can prove it was recorded by this tool.

    Two proofs, both positive. New recordings carry MARKER in the WAV's comment
    field, which rides inside the file and so survives being copied off the
    machine that made it. Recordings predating the marker carry a
    `<wav>.layout.json` written by tools/stamp_legacy_recordings.py; stamping
    them in place would mean rewriting gigabytes of irreplaceable audio for a
    metadata string.

    A layout sidecar names its WAV by basename only, and nothing about the
    filesystem stops a foreign file from later landing at that same path --
    replace a stamped recording with a same-named capture and a filename match
    alone would still grant it dual-channel trust. So the sidecar also records
    the WAV's size and mtime at stamp time, and both must still match before
    the record is honoured -- the same source-binding `_ingest_record_matches`
    does for `.ingest.json`. A mismatched or unreadable record fails closed.

    Anything unproven is not native. Downmixing a recording that was ours costs
    only the local speaker's channel split; trusting a file that was not ours
    corrupts speaker identity.
    """
    if os.path.splitext(source)[1].lower() != ".wav":
        return False
    try:
        with sf.SoundFile(source) as f:
            if (f.comment or "").startswith(MARKER):
                return True
    except Exception:
        return False
    try:
        with open(layout_marker_path(source)) as f:
            record = json.load(f)
    except Exception:
        return False
    if record.get("layout") != "dual":
        return False
    try:
        st = os.stat(source)
    except OSError:
        return False
    return record.get("size") == st.st_size and record.get("mtime") == st.st_mtime


def plan(source, workspace, *, native, mtime, mixed_flag=False):
    """Decide how to normalize `source`. Reads `source`'s header (via
    `_is_ingest_product`) but never writes anything.

    `native` is whether the file proved it came from this tool (see is_native).
    Split from `run` so the decision -- the part that can be silently wrong -- is
    testable without audio.

    Converted output goes in a subdirectory rather than beside the recordings.
    A source that is itself named `meeting_<ts>.wav` in the workspace -- which is
    every recording this tool has ever made -- would otherwise get a target
    identical to its own path, and converting would overwrite the user's only
    copy of that meeting with a mono downmix.
    """
    ext = os.path.splitext(source)[1].lower()
    timestamp = derive_timestamp(source, mtime)
    converted = os.path.join(workspace, IMPORTED_DIR, f"meeting_{timestamp}.wav")

    if native and not mixed_flag:
        return IngestPlan(source, source, "passthrough", False, False, timestamp)

    # Reprocessing a previous import is a normal thing to do, and the source
    # is already exactly what `run` would have produced -- there is nothing to
    # convert. This is a shape check for REUSE, never a trust grant: it can
    # only ever route to `mixed=True` (this branch is only reached once the
    # `native and not mixed_flag` passthrough above has already failed), so it
    # can never be a way for an unproven file to reach `mixed=False`. Gated on
    # `ext == ".wav"` too -- `_is_ingest_product` only checks channel count and
    # sample rate, which a mono 48kHz flac/mp3 could also match, and handing
    # anything but a wav straight to the pipeline is the exact failure this
    # feature exists to prevent.
    if ext == ".wav" and _is_ingest_product(source):
        return IngestPlan(source, source, "passthrough", True, False, timestamp)

    # Unproven provenance, or an explicit override: read it as one mixed track
    # rather than diarizing an arbitrary channel of it.
    return IngestPlan(source, converted, "convert", True, ext in VIDEO_EXTS,
                      timestamp)


TARGET_SR = 48000


def _require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise IngestError(
            "ffmpeg is required to read this file and is not on PATH. "
            "Install it with: brew install ffmpeg")


def has_stream(source, kind):
    """True if `source` carries at least one stream of `kind` ("a" or "v").

    A video-only mp4 otherwise decodes to a 0-byte wav that fails three stages
    later with an unrelated-looking error.
    """
    if shutil.which("ffprobe") is None:
        return True         # cannot check; let ffmpeg's own failure report it
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", kind,
         "-show_entries", "stream=index", "-of", "csv=p=0", source],
        capture_output=True, text=True)
    return bool(proc.stdout.strip())


def inspect_source(source):
    """(native, mtime) for `source`; see is_native for what proves provenance."""
    if not os.path.exists(source):
        raise IngestError(f"file not found: {source}")
    mtime = os.path.getmtime(source)
    if os.path.splitext(source)[1].lower() != ".wav":
        return False, mtime
    try:
        sf.info(source)
    except Exception as e:
        raise IngestError(f"cannot read {source}: {e}") from e
    return is_native(source), mtime


def _is_ingest_product(path):
    """True only if `path` has the exact shape ingest's own conversion writes.

    `run` always writes mono at TARGET_SR, and `core/audio_recorder.py` -- the
    only other wav writer here -- hardcodes `channels=2`, so a native recording
    can never look like this. Deliberately conservative: a wrong shape, an
    unreadable file, or a missing decoder all answer "not ours", because the
    only thing done to a file that is not ours is to leave it alone.
    """
    try:
        info = sf.info(path)
    except Exception:
        return False
    return info.channels == 1 and info.samplerate == TARGET_SR


def plan_for(source, workspace, *, mixed_flag=False):
    native, mtime = inspect_source(source)
    return plan(source, workspace, native=native, mtime=mtime,
                mixed_flag=mixed_flag)


def _ingest_record_path(target):
    return target + ".ingest.json"


def _write_ingest_record(target, source):
    """Record what `target` was converted from, so a later import that derives
    the same timestamp from a *different* source can tell it does not own this
    target, instead of silently reusing someone else's audio.

    Written only after a successful `os.replace` (see call site), so a crashed
    conversion can never leave a record pointing at audio that isn't there.
    A plain write, not the tmp-file/os.replace dance the wav gets: a torn
    record fails closed -- `json.load` raises, `_ingest_record_matches` treats
    that as "no match", and the next import bumps to a new suffix rather than
    trusting a half-written record.
    """
    st = os.stat(source)
    with open(_ingest_record_path(target), "w") as f:
        json.dump({"source": os.path.abspath(source), "size": st.st_size,
                   "mtime": st.st_mtime}, f)


def _ingest_record_matches(target, source):
    """True only if `target`'s `.ingest.json` names this exact `source`.

    Absent, unreadable, or naming a different path all answer False -- the
    same "unproven means not mine" stance as `is_native` -- because a reuse
    that turns out to be wrong means the pipeline silently transcribes the
    wrong meeting.
    """
    try:
        with open(_ingest_record_path(target)) as f:
            record = json.load(f)
    except Exception:
        return False
    try:
        st = os.stat(source)
    except OSError:
        return False
    return (record.get("source") == os.path.abspath(source)
            and record.get("size") == st.st_size
            and record.get("mtime") == st.st_mtime)


def frame_elapsed_ms(index, interval):
    """Elapsed milliseconds for ffmpeg's 1-based output frame `index`.

    `fps=1/interval` emits its first frame at t=0, so index 1 is time zero.
    """
    return int((index - 1) * interval * 1000)


def extract_frames(source, wav_path, interval):
    """Sample `source`'s video into `<wav>.frames/`, returning the count written.

    Zoom's cloud recordings show the active speaker's name tag, which is the
    difference between named clusters and a screen of Unknown. The frames
    store is just this directory listing, so the existing VLM name reader
    consumes these with no changes.
    """
    if not has_stream(source, "v"):
        print(f"[ingest] {source} has no video track; voiceprints only.")
        return 0

    _require_ffmpeg()
    out_dir = frames_dir_for(wav_path)
    os.makedirs(out_dir, exist_ok=True)
    staging = os.path.join(out_dir, ".staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", source,
             "-vf", f"fps=1/{interval}", "-qscale:v", "4",
             "-y", os.path.join(staging, "%09d.jpg")],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise IngestError(
                f"ffmpeg could not sample frames from {source}:\n"
                f"{proc.stderr.strip()[-800:]}")

        # ffmpeg numbers its output by frame index; the store keys on elapsed
        # ms, so each staged file is renamed on its way into the real
        # directory rather than written there directly.
        written = 0
        for name in sorted(os.listdir(staging)):
            stem, ext = os.path.splitext(name)
            try:
                index = int(stem)
            except ValueError:
                continue
            dest = os.path.join(
                out_dir, f"{frame_elapsed_ms(index, interval):09d}{ext}")
            os.replace(os.path.join(staging, name), dest)
            written += 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"[ingest] extracted {written} frame(s) from {source}")
    return written


def _frames_present(wav_path):
    """True if `<wav>.frames/` already holds at least one extracted frame.

    Backs the reuse path's idempotence check: a reuse must not repeat a full
    video decode already done by the import that created `wav_path`, but it
    must not assume the directory exists either -- an import made before
    frame extraction existed, or one that came up empty, has none.
    """
    try:
        names = os.listdir(frames_dir_for(wav_path))
    except OSError:
        return False
    return any(name.endswith(".jpg") for name in names)


def run(p, *, frame_interval=4.0):
    """Materialize `p`, returning the wav path the pipeline should consume."""
    if p.action == "passthrough":
        return p.target

    _require_ffmpeg()
    if not has_stream(p.source, "a"):
        raise IngestError(f"no audio track in {p.source}")

    # Ingest never writes over a file it did not create, and never writes onto
    # its own input. Neither is provable from a path or a shape alone --
    # `derive_timestamp` keys off the filename, ignoring the directory, so an
    # archived copy of a recording (`~/Desktop/meeting_<ts>.wav`) can derive a
    # target that collides with an unrelated existing file, and two different
    # foreign sources can derive the very same timestamp (mtime has only
    # one-second granularity, and a bulk copy that preserves mtimes can tie).
    # Every attempt to infer from a path or a shape whether overwriting was
    # safe has been wrong, twice destructively. So each candidate target is
    # checked in order, and there are exactly four outcomes, only one of which
    # writes:
    #
    #   the candidate IS `p.source`        -> refuse outright, "onto itself";
    #                                        `samefile` compares device+inode,
    #                                        so a symlink, hardlink, or
    #                                        case-insensitive APFS alias of
    #                                        the same file is still caught
    #   nothing at the candidate            -> convert; there is nothing to lose
    #   an ingest product of this source    -> reuse it; a reuse is only ever a
    #                                        reuse of the same source
    #   anything else                       -> refuse (a foreign file sits
    #                                        there, untouched) or, if it is an
    #                                        ingest product of a *different*
    #                                        source (the target-name collided
    #                                        by timestamp, not by identity),
    #                                        try the next free `-2`, `-3`, ...
    #                                        candidate and check it the same way
    #
    # This loop is what the same-file guard has to run inside of now: the
    # naming scheme changed (this suffixing), so a check made once against
    # `p.target` before the loop would only cover the first candidate and miss
    # a self-collision on a later one. Refusing is the correct outcome, not a
    # cop-out. A wrong refusal costs the user one command; a wrong overwrite --
    # or a silent reuse of the wrong meeting's audio -- costs them a meeting,
    # since `workspace/` holds the only copy of every recording this tool makes.
    root, ext = os.path.splitext(p.target)
    target = p.target
    suffix = 1
    while os.path.exists(target):
        if os.path.samefile(p.source, target):
            raise IngestError(
                f"refusing to convert {p.source} onto itself; pass --out <dir> "
                f"to write the converted copy elsewhere")
        if _is_ingest_product(target) and _ingest_record_matches(target, p.source):
            print(f"[ingest] reusing {target}")
            if p.extract_frames and not _frames_present(target):
                extract_frames(p.source, target, frame_interval)
            return target
        if not _is_ingest_product(target):
            raise IngestError(
                f"{target} already exists and was not written by ingest; "
                f"refusing to overwrite it. Pass --out <dir> to convert into a "
                f"different directory.")
        # An ingest product, but not proven to be *this* source's: the target
        # name belongs to someone else's import. Take the next one rather than
        # reusing their audio for this meeting.
        suffix += 1
        target = f"{root}-{suffix}{ext}"

    target_dir = os.path.dirname(os.path.abspath(target))
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".ingest-tmp-", suffix=".wav")
    os.close(fd)
    try:
        # -vn so a video container yields audio only; -ac 1 collapses a foreign
        # stereo mix instead of letting the pipeline diarize one arbitrary side.
        # Decoding into `tmp` rather than `target` directly means a killed or
        # failed ffmpeg never leaves a partial/truncated file at the target path.
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", p.source, "-vn",
             "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le",
             "-y", tmp],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise IngestError(
                f"ffmpeg could not decode {p.source}:\n"
                f"{proc.stderr.strip()[-800:]}")
        os.replace(tmp, target)
    finally:
        # A plain `except Exception` misses KeyboardInterrupt -- exactly what
        # a user hits aborting a long conversion -- and the default mkstemp
        # prefix would leave a bare "tmpXXXXXX.wav" in the workspace that
        # looks like a real recording to anything that globs
        # workspace/*.wav. `finally` covers every exit path, and after a
        # successful os.replace `tmp` no longer exists, so this is a no-op
        # on the success path.
        if os.path.exists(tmp):
            os.remove(tmp)
    # Only after the audio is safely at `target`: a record written earlier
    # would point at a file that a crash between os.replace and here could
    # still be missing, and the next import must never trust that.
    _write_ingest_record(target, p.source)
    print(f"[ingest] converted {p.source} -> {target} (mono {TARGET_SR} Hz)")

    if p.extract_frames:
        extract_frames(p.source, target, frame_interval)

    return target
