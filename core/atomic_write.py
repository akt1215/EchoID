"""Crash-safe file writes.

A plain `open(path, "w")` truncates the target the moment it is called, so a
process that dies between the open and the final flush leaves the file empty or
half-written — with the original gone. The files this project writes are
irreplaceable user data: the voiceprint DB, the learned glossary, the meeting
sidecars, and the user's own Obsidian notes.

Writing to a temp file in the same directory and then `os.replace`-ing it onto
the target makes the swap atomic on POSIX: a reader either sees the whole old
file or the whole new one, never a truncated one, and an exception before the
replace leaves the original untouched.
"""

import os
import stat
import tempfile


def write_text(path, text):
    """Atomically replace `path` with `text`.

    The temp file is created in the target's own directory so `os.replace` stays
    within one filesystem (a cross-device rename is not atomic and would fail).

    An existing target's permissions are carried over: `mkstemp` always creates
    0600 and `os.replace` gives the target the temp file's mode, so without this
    every write would silently narrow the user's own files to owner-only — and
    cumulatively, since the next write would start from 0600 too. A file this
    helper creates from scratch keeps mkstemp's restrictive 0600, which is the
    right default for the private data it holds.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        if os.path.exists(path):
            os.chmod(tmp, stat.S_IMODE(os.stat(path).st_mode))
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)   # never leak the temp file on a failed write
        raise
