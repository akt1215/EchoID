"""Cross-meeting speaker directory (model-free: numpy + json only).

The pipeline resolves identities inside one meeting. This is the view ACROSS
meetings: it maps every saved sidecar cluster back to an enrolled voiceprint, so
a person -- including a still-unidentified "Speaker N" placeholder -- can be
listened to and renamed everywhere from one place, without re-running the
pipeline. Matching uses the same centered-space comparison as resolution, so the
directory always agrees with what the pipeline decided.
"""

import glob
import json
import os
import re

import numpy as np

from ai.identity_resolution import _centroid, _db_match
from ai.name_reader import is_local_name
from core import atomic_write, embedding_mean, glossary

_PLACEHOLDER_RE = re.compile(r"^Speaker \d+$")


def _name_patterns(name):
    """The two places a speaker's name appears in a note: the transcript header
    (`**Name**`) and a wikilink (`[[Name]]`, which also covers `**[[Name]]**` and
    the `participants:` list). Both are delimiter-anchored, so renaming
    "Speaker 1" can never corrupt "Speaker 10"."""
    esc = re.escape(name)
    return ((re.compile(r"\*\*" + esc + r"\*\*"), "**{}**"),
            (re.compile(r"\[\[" + esc + r"\]\]"), "[[{}]]"))


def _count_in(text, name):
    return sum(len(rx.findall(text)) for rx, _ in _name_patterns(name))


def _rewrite(text, old, new):
    for rx, tpl in _name_patterns(old):
        text = rx.sub(tpl.format(new), text)
    return text


class SpeakerDirectory:
    def __init__(self, workspace, vault_dir, db_threshold=0.60,
                 db_margin=0.0, min_speech=0.0, min_turn_seconds=0.5,
                 local_names=None):
        self.workspace = workspace
        self.vault_dir = vault_dir
        # The same guards the pipeline applies. Without them this screen is
        # strictly more permissive than the pipeline it is meant to reflect, and
        # shows matches the pipeline itself refuses.
        self.db_threshold = db_threshold
        self.db_margin = db_margin
        self.min_speech = min_speech
        self.min_turn_seconds = min_turn_seconds
        # The local user's own display name(s) -- same config the pipeline and
        # SpeakerReview read. Used below to exclude their print from matching
        # against a dual-channel meeting's clusters (see `identities`).
        self.local_names = local_names or []
        self._multi = None      # lazily loaded set of (meeting, cluster)
        self.db_path = os.path.join(workspace, "speakers.json")
        self.glossary_path = os.path.join(workspace, "glossary_learned.json")
        self.db = {}
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                self.db = json.load(f)
        # The local user's print, if enrolled (only possible from a mixed
        # recording -- see ai/speaker_review.py). Excluded from matching against
        # a dual-channel meeting's clusters: their voice is on ch0 and never
        # embedded there, so a ch1 match is echo/speakerphone bleed naming a
        # remote speaker -- the same hazard `_db_match`'s `exclude` guards
        # against in the pipeline itself.
        self._local_db_keys = tuple(
            k for k in self.db if is_local_name(k, self.local_names))
        # Dropped when it does not match the voiceprints' width: the stored mean
        # belongs to whichever embedding model wrote it, and subtracting an
        # ECAPA mean from a WeSpeaker vector raises inside numpy far from the
        # backend setting that caused it.
        dim = len(next(iter(self.db.values()), []) or [])
        self.mean = embedding_mean.load_aligned(
            os.path.join(workspace, "embedding_mean.json"), dim)
        self.sidecars = []
        for p in sorted(glob.glob(os.path.join(workspace, "*.segments.json"))):
            try:
                with open(p) as f:
                    self.sidecars.append((p, json.load(f)))
            except (OSError, ValueError):
                continue   # a corrupt sidecar must not take the whole directory down

    def identities(self):
        """One entry per enrolled voiceprint: {name, is_placeholder, n_meetings,
        total_dur, clips:[(wav, start, end)]}, longest-speaking first.

        `n_meetings` counts distinct MEETINGS, not clusters: if a person somehow
        ends up split across two clusters within the same sidecar (rare -- a
        cluster that DB-matches at pipeline time already gets folded into a
        single "db::<name>" cluster id upstream, so this only happens when a
        voiceprint is matched retroactively against clusters that were
        unknown/visual-only when the sidecar was written), both clusters'
        turns are merged before counting so that one meeting is one meeting.
        """
        out = {n: {"name": n, "is_placeholder": bool(_PLACEHOLDER_RE.match(n)),
                   "n_meetings": 0, "total_dur": 0.0, "clips": []}
               for n in self.db}
        for path, data in self.sidecars:
            wav = data.get("wav", "")
            meeting = os.path.basename(path)[: -len(".segments.json")]
            by_cluster = {}
            for r in data.get("diarization", []):
                if r.get("embedding"):
                    by_cluster.setdefault(r.get("cluster", r.get("label")), []).append(r)

            # Group cluster-level DB matches by name so a person matching more
            # than one cluster in this same meeting still counts as ONE meeting,
            # with turns from every matching cluster merged.
            matched_by_name = {}
            for cluster, rows in by_cluster.items():
                # A cluster the user confirmed holds more than one person cannot
                # speak for any single identity: crediting its minutes, and
                # worse offering its audio as somebody's sample clip, is how
                # "Avery Chen" came to play Sam Rivera's voice.
                if self._holds_several_people(meeting, cluster):
                    continue
                speech = sum(r["end"] - r["start"] for r in rows
                             if r["end"] - r["start"] >= self.min_turn_seconds)
                if self.min_speech and speech < self.min_speech:
                    continue
                # A dual-channel meeting must never match the local user's
                # print here either: the same ch1-bleed hazard the pipeline's
                # `exclude` guards against applies retroactively across
                # meetings too, and this view also drives `best_clip` and
                # rename-everywhere -- see the comment above on
                # `_holds_several_people` for what a wrong match here costs.
                exclude = () if data.get("mixed") else self._local_db_keys
                name, _, _ = _db_match(
                    _centroid([r["embedding"] for r in rows], self.mean),
                    self.db, self.db_threshold, self.db_margin, exclude=exclude)
                if name:
                    matched_by_name.setdefault(name, []).extend(rows)

            for name, rows in matched_by_name.items():
                e = out[name]
                e["n_meetings"] += 1
                e["total_dur"] += sum(r["end"] - r["start"] for r in rows)
                longest = max(rows, key=lambda r: r["end"] - r["start"])
                e["clips"].append((wav, longest["start"], longest["end"]))
        return sorted(out.values(), key=lambda e: -e["total_dur"])

    def best_clip(self, name):
        """The longest recorded turn for `name`, for playback. None if unmatched."""
        for e in self.identities():
            if e["name"] == name and e["clips"]:
                return max(e["clips"], key=lambda c: c[2] - c[1])
        return None

    def _holds_several_people(self, meeting, cluster):
        """True when the user confirmed this cluster contains more than one
        speaker. Read lazily from the evaluation ground truth, which is the only
        record of it; absent or unreadable means "no such finding"."""
        if self._multi is None:
            self._multi = set()
            path = os.path.join(self.workspace, "eval", "ground_truth.json")
            try:
                with open(path) as f:
                    for m, clusters in json.load(f).items():
                        for c, entry in clusters.items():
                            if entry.get("verdict") == "multiple":
                                self._multi.add((m, c))
            except (OSError, ValueError, AttributeError):
                pass
        return (meeting, cluster) in self._multi

    def _current_dim(self):
        """Width of the embeddings the sidecars actually hold, or 0 if unknown.

        The sidecars are the authority on which space is current: they are
        rewritten whenever the embedding backend changes.
        """
        for _, data in self.sidecars:
            for turn in data.get("diarization", []):
                if turn.get("embedding"):
                    return len(turn["embedding"])
        return 0

    def rename_plan(self, old, new):
        """What renaming `old` -> `new` would change. Computes only; writes nothing.
        This is the preview the UI confirms before anything is touched."""
        notes = []
        for p in sorted(glob.glob(os.path.join(self.vault_dir, "*.md"))):
            try:
                with open(p) as f:
                    n = _count_in(f.read(), old)
            except OSError:
                continue
            if n:
                notes.append((p, n))

        # Not named `glossary`: that is the imported module, and shadowing it here
        # would make any later glossary.* call in this function raise.
        in_glossary = False
        if os.path.exists(self.glossary_path):
            try:
                with open(self.glossary_path) as f:
                    in_glossary = old in json.load(f)
            except (OSError, ValueError):
                in_glossary = False

        # Check transcript rows too: apply_rename rewrites labels in both, so a
        # sidecar carrying `old` only in a transcript row must still be listed.
        sidecars = [p for p, data in self.sidecars
                    if any(r.get("label") == old for r in data.get("diarization", []))
                    or any(t.get("label") == old for t in data.get("transcript", []))]

        return {"old": old, "new": new,
                "in_db": old in self.db,
                # Only a real merge if BOTH have a voiceprint -- apply_rename gates
                # the merge on in_db, so without `old` enrolled nothing is merged
                # and the preview must not claim otherwise.
                "merges_into_existing": old in self.db and new in self.db and new != old,
                "notes": notes,
                "n_mentions": sum(n for _, n in notes),
                "glossary": in_glossary,
                "sidecars": sidecars}

    def apply_rename(self, plan):
        """Execute a `rename_plan`: DB, vault notes, glossary and sidecars.

        Renaming onto a name that already exists means the placeholder turned out
        to be someone already enrolled, so the two voiceprints are averaged into one.

        Every write goes through `atomic_write.write_text`: this touches the
        voiceprint DB and the user's own notes, which no truncating write may be
        allowed to destroy halfway through.
        """
        old, new = plan["old"], plan["new"]

        if plan["in_db"]:
            vec = np.asarray(self.db.pop(old), dtype=float)
            if plan["merges_into_existing"]:
                other = np.asarray(self.db[new], dtype=float)
                # Widths differ only when one print predates a backend change and
                # so describes a different space; averaging them is meaningless
                # and raises. Keep whichever matches the current embeddings —
                # the sidecars — rather than crashing mid-rename.
                if other.shape == vec.shape:
                    vec = vec + other
                elif other.shape[0] == self._current_dim():
                    vec = other
            n = np.linalg.norm(vec)
            self.db[new] = (vec / n if n else vec).tolist()
            atomic_write.write_text(self.db_path, json.dumps(self.db, indent=4))

        for p, _ in plan["notes"]:
            with open(p) as f:
                text = f.read()
            atomic_write.write_text(p, _rewrite(text, old, new))

        if plan["glossary"]:
            with open(self.glossary_path) as f:
                store = json.load(f)
            count = store.pop(old, 0)
            store[new] = max(store.get(new, 0), count)
            glossary._save(self.glossary_path, store)

        for p in plan["sidecars"]:
            with open(p) as f:
                data = json.load(f)
            for r in data.get("diarization", []):
                if r.get("label") == old:
                    r["label"] = new
                if r.get("cluster") == f"db::{old}":
                    r["cluster"] = f"db::{new}"
            for t in data.get("transcript", []):
                if t.get("label") == old:
                    t["label"] = new
            atomic_write.write_text(p, json.dumps(data))
