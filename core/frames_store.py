"""On-disk store for the meeting frames the VLM name reader consumes.

Frames are saved next to the WAV under `<wav>.frames/`, each file named by its
elapsed milliseconds since capture start (`000012345.jpg`). The "manifest" is
just that directory listing, so nothing extra needs to be written or kept in
sync — the pipeline (in-process or as a --from-file subprocess) rebuilds
`[(elapsed_seconds, path)]` by listing the directory.
"""

import glob
import os


def frames_dir_for(wav_path):
    """Directory holding a recording's captured frames."""
    return os.path.splitext(wav_path)[0] + ".frames"


def read_frames_manifest(wav_path):
    """Return [(elapsed_seconds, frame_path)] sorted by time, or [] if the
    recording has no frames directory (e.g. an old WAV, or capture was off)."""
    d = frames_dir_for(wav_path)
    events = []
    if os.path.isdir(d):
        for p in glob.glob(os.path.join(d, "*.jpg")):
            try:
                ms = int(os.path.splitext(os.path.basename(p))[0])
            except ValueError:
                continue
            events.append((ms / 1000.0, p))
    events.sort()
    return events
