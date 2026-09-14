"""Mark pre-existing recordings as this tool's own dual-channel WAVs.

Recordings made before the embedded marker carry no proof of where they came
from, and a 2-channel WAV is not proof: QuickTime, OBS and Audio Hijack all
produce stereo. Stamping the marker into the audio itself would mean rewriting
gigabytes of irreplaceable data for a metadata string, so this writes a small
`<wav>.layout.json` beside each file instead.

Provenance is PROVEN only for a WAV that is named by a `*.segments.json` review
sidecar AND whose filename matches the recorder's own `meeting_<ts>.wav` form.
Neither signal alone is sufficient: a sidecar only proves the pipeline *read*
that WAV -- main.py writes one for any `--from-file` path, including a foreign
WAV a user reprocessed while it happened to sit in `workspace/` -- and a
recorder-shaped filename alone proves nothing, since anyone can name a file
that. Together they are the strongest evidence a legacy file can offer.
Everything else needs the user to confirm, because a wrong stamp is exactly the
silent mislabel the marker exists to prevent.

Run:  .venv/bin/python -m tools.stamp_legacy_recordings            # report only
      .venv/bin/python -m tools.stamp_legacy_recordings --proven   # stamp proven
      .venv/bin/python -m tools.stamp_legacy_recordings --confirm  # ask per file
"""

import argparse
import glob
import json
import os
import re
import sys

import soundfile as sf

from core import ingest
from core.atomic_write import write_text

# What the recorder itself has always named its output (see
# derive_timestamp's native-stem branch in core/ingest.py) -- not a shared
# constant, because this describes a historical filename convention, not what
# derive_timestamp accepts or produces today.
_RECORDER_FILENAME = re.compile(
    r"^meeting_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.wav$")


def provenance(workspace):
    """{wav path: proven} for every WAV in `workspace`.

    Proven requires both a review sidecar in this workspace naming that exact
    WAV AND a filename matching the recorder's own form -- see module
    docstring for why neither alone is sufficient.
    """
    named = set()
    for p in glob.glob(os.path.join(workspace, "*.segments.json")):
        try:
            with open(p) as f:
                wav = json.load(f).get("wav", "")
        except Exception:
            continue
        if wav:
            named.add(os.path.abspath(wav))
    return {w: (os.path.abspath(w) in named
                and bool(_RECORDER_FILENAME.match(os.path.basename(w))))
            for w in sorted(glob.glob(os.path.join(workspace, "*.wav")))}


def stamp(wav_path, reason):
    """Write the dual-layout sidecar for `wav_path`. Never touches the audio.

    Records the WAV's size and mtime alongside the layout verdict: a sidecar
    that named only the basename would still grant trust after a same-named
    foreign file later replaced the stamped recording. `core.ingest.is_native`
    checks both still match before honouring the record.
    """
    st = os.stat(wav_path)
    write_text(ingest.layout_marker_path(wav_path),
               json.dumps({"layout": "dual", "reason": reason,
                           "wav": os.path.basename(wav_path),
                           "size": st.st_size, "mtime": st.st_mtime},
                          indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--proven", action="store_true",
                        help="stamp every WAV a review sidecar names")
    parser.add_argument("--confirm", action="store_true",
                        help="also ask about each unproven WAV")
    args = parser.parse_args(argv)

    found = provenance(args.workspace)
    if not found:
        print(f"No WAVs in {args.workspace}.")
        return 1

    for wav, proven in found.items():
        name = os.path.basename(wav)
        if ingest.is_native(wav):
            print(f"  [already marked] {name}")
            continue
        if proven:
            if args.proven or args.confirm:
                stamp(wav, reason="segments-sidecar")
                print(f"  [stamped, proven] {name}")
            else:
                print(f"  [proven, not stamped] {name}")
            continue
        if not args.confirm:
            print(f"  [unproven]       {name}")
            continue
        try:
            info = sf.info(wav)
        except Exception as e:
            print(f"  [unreadable]     {name} ({e})")
            continue
        answer = input(f"  {name} ({info.channels}ch, {info.duration:.0f}s) — "
                       f"recorded by ZoomRecorder? [y/N] ").strip().lower()
        if answer == "y":
            stamp(wav, reason="user-confirmed")
            print(f"  [stamped, confirmed] {name}")
        else:
            print(f"  [left unmarked]  {name}")

    if not (args.proven or args.confirm):
        print("\n(report only — pass --proven to stamp sidecar-proven files, "
              "or --confirm to review the rest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
