"""Where a run's review sidecar lives, and how a caller learns that path.

The sidecar is named after the wav the pipeline consumed. For an import that
is ingest's product, not the file the user picked -- and `run` may bump it to a
`-2`/`-3` suffix depending on what else is already sitting in `imported/`, so
the path is only knowable from the run itself. The TUI shells out to `main.py`
and must not re-derive it: re-running `ingest.plan_for` would duplicate the
target-selection logic and still miss the suffix.

So `main.py` announces the path on stdout and the TUI reads it back out of the
output it already streams. Both sides go through this module so the line's
shape is written once; a caller that cannot import `main` (the TUI must not --
`main` pulls in the whole ML stack) still shares the format.
"""

import glob
import json
import os

SUFFIX = ".segments.json"
LINE_PREFIX = "Review sidecar written: "


def list_reviewable(workspace):
    """Every past meeting whose speakers can be re-reviewed, newest first.

    One entry per sidecar: `{path, meeting, turns, labels}`. `labels` are the
    distinct speaker names the meeting currently carries, commonest first --
    that is what identifies a meeting whose names went wrong, since the stem is
    only a timestamp.

    Names come from the transcript, because those are the labels the note was
    written with. A run stopped before transcription still has clusters worth
    renaming, so it falls back to the diarization turns.

    A sidecar that will not parse is skipped rather than raised: one corrupt
    file must not cost the user access to every other meeting.
    """
    out = []
    for path in glob.glob(os.path.join(workspace, "*" + SUFFIX)):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        rows = data.get("transcript") or data.get("diarization") or []
        counts = {}
        for r in rows:
            label = r.get("label")
            if label:
                counts[label] = counts.get(label, 0) + 1
        out.append({
            "path": path,
            "meeting": os.path.basename(path)[: -len(SUFFIX)],
            "turns": len(rows),
            "labels": sorted(counts, key=lambda n: (-counts[n], n)),
        })
    # The stem is `meeting_YYYY-MM-DD_HH-MM-SS`, so a reverse string sort is a
    # reverse chronological sort -- and unlike mtime it does not reorder the
    # list every time a meeting is re-transcribed in place.
    return sorted(out, key=lambda e: e["meeting"], reverse=True)


def path_for(workspace, audio_path):
    """The sidecar path for the recording the pipeline consumed."""
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(workspace, stem + SUFFIX)


def announce(path):
    """The stdout line a pipeline run emits to name its sidecar."""
    return f"{LINE_PREFIX}{path}"


def parse_announcement(line):
    """The path announced by `line`, or None if it announces nothing.

    Matched anywhere in the line, not anchored at its start: the TUI reads the
    pipeline's stdout and stderr through one pipe, and a progress bar mid-write
    (tqdm ends a frame with no terminator -- the carriage return leads the NEXT
    one) leaves its bytes in front of whatever prints next. An anchored match
    would drop the announcement there and fall back to the guessed path, which
    is the failure this whole mechanism exists to remove -- and it would do so
    intermittently, only under real ML stages. Safe because `announce` is the
    only thing that ever writes this string.
    """
    if LINE_PREFIX not in line:
        return None
    return line.split(LINE_PREFIX, 1)[1].strip() or None


def resolve(announced, workspace, input_path):
    """The sidecar to review after a run: what the run announced, else the path
    the input implies.

    The fallback is right only when the input IS what the pipeline consumed (a
    native recording); for an import it names a file that does not exist. It
    exists so a run whose output was lost still opens something rather than
    nothing.
    """
    return announced or path_for(workspace, input_path)
