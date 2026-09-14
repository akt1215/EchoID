"""Model-free speaker review over a pipeline sidecar.

Loads `workspace/<meeting>.segments.json` (per-turn embeddings + transcript) and
lets a UI rename / merge / split / reassign / reset speakers, then enroll
voiceprints and rebuild the note transcript on commit. Deliberately imports no
ML stack (torch/speechbrain/whisperx/cv2/pyannote) so it can run inside the TUI
process — all clustering and enrollment work on the stored embedding vectors.
"""

import json
import os
import re
from collections import Counter

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from ai.identity_resolution import normalize_name
from ai.name_reader import is_local_name
from core import atomic_write, embedding_mean, glossary
from core.clip_player import ClipPlayer
from export.obsidian_writer import inject_wikilinks, render_note, timestamp_to_iso


def _normalize(vec):
    vec = np.asarray(vec, dtype=float)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


_PLACEHOLDER_RE = re.compile(r"^Speaker (\d+)$")


def _next_placeholder_name(taken):
    """Lowest unused "Speaker N". Numbering is derived from the names already in
    use rather than a stored counter, so it cannot drift out of sync with the DB."""
    used = [int(m.group(1)) for m in
            (_PLACEHOLDER_RE.match(str(k)) for k in taken) if m]
    return f"Speaker {max(used, default=0) + 1}"


class SpeakerReview:
    def __init__(self, sidecar_path, db_path, update_rate=0.2,
                 llm_model=None, llm_timeout=120, placeholder_min_duration=60.0,
                 local_names=None, trust_auto_names=False, mixed=None):
        self.llm_model = llm_model
        self.llm_timeout = llm_timeout
        # The local user's own display name(s): never enrolled as a voiceprint and
        # never offered as a naming suggestion (they are channel 0 / "Me (Local)").
        self.local_names = local_names or []
        # Learned-glossary store lives beside the speaker DB (both gitignored).
        self.glossary_path = os.path.join(
            os.path.dirname(os.path.abspath(db_path)), "glossary_learned.json")
        self.sidecar_path = sidecar_path
        with open(sidecar_path) as f:
            data = json.load(f)
        self.wav = data.get("wav", "")
        self.note = data.get("note", "")
        self.transcript = data.get("transcript", [])
        # A mixed recording has no mic channel, so the local user is an ordinary
        # voice in the track: enrollable, and offerable as a suggestion. Read
        # from the sidecar so the TUI and CLI reviewers agree without either of
        # them having to thread the flag.
        self.mixed = bool(data.get("mixed", False)) if mixed is None else bool(mixed)

        # Per-turn parallel arrays: metadata in self.seg, embedding in self.emb.
        self.seg = []
        self.emb = []
        for d in data.get("diarization", []):
            self.seg.append({
                "start": float(d["start"]), "end": float(d["end"]),
                "label": d["label"], "source": d.get("source", "unknown"),
                # Group by the stable per-turn cluster id; legacy sidecars that
                # predate it fall back to the label (the old grouping key).
                "cluster": d.get("cluster", d["label"]),
            })
            self.emb.append(np.asarray(d.get("embedding", []), dtype=float))

        self.db_path = db_path
        self.update_rate = update_rate
        self.db = {}
        if os.path.exists(db_path):
            with open(db_path) as f:
                self.db = {k: np.asarray(v, dtype=float) for k, v in json.load(f).items()}

        # Global embedding mean (beside the DB). Splitting and enrollment compare
        # embeddings by cosine, which for ECAPA is only discriminative once this
        # shared mean is removed; None (no store yet) leaves the raw behavior
        # untouched. It is also dropped when its width does not match this
        # meeting's embeddings — the stored mean belongs to whichever model wrote
        # it, and a model with no need for centering leaves a stale one behind.
        mean_path = os.path.join(os.path.dirname(os.path.abspath(db_path)),
                                 "embedding_mean.json")
        dim = next((e.size for e in self.emb if e.size), 0)
        self.mean = embedding_mean.load_aligned(mean_path, dim)

        # An auto-filled name the user never touched is not evidence: enrolling
        # it lets one wrong DB match drag the voiceprint toward the wrong person,
        # which makes the next wrong match easier. Only user-confirmed rows
        # update an existing print unless this is explicitly enabled.
        self.trust_auto_names = trust_auto_names

        self._reset_names = set()
        self.placeholder_min_duration = placeholder_min_duration
        # gid -> bool override of the duration rule; absent means "no override".
        self._remember = {}

        # Clip playback: one afplay process at a time, owned by ClipPlayer so a
        # second ▶ (or leaving the review) stops the first instead of overlapping.
        self._player = ClipPlayer()

        # Working model: one group per distinct cluster id (first-seen order),
        # displayed under that cluster's resolved label. Grouping by cluster (not
        # label) keeps two speakers who share a label as two rows and lets the
        # pipeline pre-merge an over-split person into one. Merge, split and
        # rename mutate self.groups; commit reads it.
        self.groups = {}
        self._next_id = 0
        cluster_to_gid = {}
        for i, s in enumerate(self.seg):
            key = s["cluster"]
            if key not in cluster_to_gid:
                gid = self._new_id()
                cluster_to_gid[key] = gid
                self.groups[gid] = {"name": s["label"], "seg_idxs": [],
                                    "confirmed": False}
            self.groups[cluster_to_gid[key]]["seg_idxs"].append(i)

    def _new_id(self):
        gid = self._next_id
        self._next_id += 1
        return gid

    def rename(self, gid, name):
        self.groups[gid]["name"] = name
        # The user touched this row, so its name is evidence and may move the
        # stored voiceprint at commit; an untouched auto-fill may not.
        self.groups[gid]["confirmed"] = True

    def merge(self, gids):
        """Merge several groups into the first; returns the surviving group id."""
        gids = list(gids)
        target = gids[0]
        for g in gids[1:]:
            self.groups[target]["seg_idxs"].extend(self.groups[g]["seg_idxs"])
            del self.groups[g]
        return target

    def split(self, gid, k=2):
        """Split one group into k by clustering its turns' embeddings (cosine,
        average linkage). Returns the new group ids. If the group has fewer than
        k embeddable turns it is left intact and [gid] is returned. Turns whose
        embedding failed to extract (empty vector) are attached to the first
        cluster rather than dropped.
        """
        idxs = self.groups[gid]["seg_idxs"]
        valid = [i for i in idxs if self.emb[i].size]
        if len(valid) < k:
            return [gid]

        embs = np.array([self.emb[i] for i in valid])
        if self.mean is not None:
            embs = embs - self.mean       # cluster in the centered (discriminative) space
        labels = fcluster(linkage(pdist(embs, metric="cosine"), method="average"),
                          t=k, criterion="maxclust")

        clusters = {}
        for j, i in enumerate(valid):
            clusters.setdefault(int(labels[j]), []).append(i)
        leftovers = [i for i in idxs if not self.emb[i].size]

        del self.groups[gid]
        new_ids = []
        for rank, c in enumerate(sorted(clusters)):
            members = clusters[c] + (leftovers if rank == 0 else [])
            nid = self._new_id()
            self.groups[nid] = {"name": f"Unknown (split {nid})", "seg_idxs": members}
            new_ids.append(nid)
        return new_ids

    def reassign(self, gid, name):
        """Point a group at an existing DB person. Enrollment mode (refine vs
        store) is decided at commit by whether the name already exists."""
        self.rename(gid, name)

    def reset_voiceprint(self, name):
        """Mark a DB voiceprint to be overwritten (not EMA-refined) on commit,
        for a centroid that earlier wrong matches may have drifted."""
        self._reset_names.add(name)

    def set_remember(self, gid, value):
        """Override whether an unnamed group is kept as a "Speaker N" identity.
        True remembers a short group; False discards a substantial one."""
        self._remember[gid] = bool(value)

    def _should_remember(self, gid):
        """Whether group `gid` should be enrolled as a stable placeholder
        identity: an explicit `set_remember` override wins in either
        direction; absent one, falls back to whether the group has at least
        `placeholder_min_duration` of speech. This is the single place the
        override-vs-duration precedence is decided -- both
        `_assign_placeholders` (what actually happens at commit) and
        `list_speakers` (what the UI shows) call it, so the checkbox always
        matches this precedence rule. It does not itself apply the separate
        "no usable embeddings" guard in `_assign_placeholders` -- a group
        with zero embeddable turns can still show `remember: True` here even
        though commit will skip enrolling it.
        """
        override = self._remember.get(gid)
        if override is not None:
            return override
        g = self.groups[gid]
        dur = sum(self.seg[i]["end"] - self.seg[i]["start"] for i in g["seg_idxs"])
        return dur >= self.placeholder_min_duration

    def _assign_placeholders(self):
        """Give still-unknown groups a stable "Speaker N" identity so the same
        person is recognized -- and can be corrected -- in later meetings.

        Enrolls when an explicit `set_remember` override says so, else when the
        group has at least `placeholder_min_duration` of speech. Short fragments
        stay Unknown and never enter the DB.
        """
        for gid, g in self.groups.items():
            if not g["name"].startswith("Unknown ("):
                continue
            if not any(self.emb[i].size for i in g["seg_idxs"]):
                continue
            if self._should_remember(gid):
                # Recompute `taken` per group so two placeholders in one commit
                # cannot collide on the same number.
                taken = set(self.db) | {gg["name"] for gg in self.groups.values()}
                g["name"] = _next_placeholder_name(taken)

    def _canonical_db_key(self, name):
        """The existing DB key that is the same name up to punctuation/spacing/case,
        else `name` unchanged. Keeps one person from splitting into two voiceprints
        under two spellings; the existing key's display spelling wins."""
        norm = normalize_name(name)
        for k in self.db:
            if normalize_name(k) == norm:
                return k
        return name

    def commit(self):
        """Enroll confirmed voiceprints, rebuild the transcript with final names,
        and (when an LLM is configured) summarize that NAMED transcript so the
        summary and action-item assignees reflect the real speakers.

        Unnamed but substantial groups first receive a stable `Speaker N` identity.

        This is the only method that writes. Confirmed names and the summary's
        entities are also harvested into the learned glossary so future meetings
        transcribe them correctly.
        """
        self._assign_placeholders()
        for g in self.groups.values():
            name = g["name"]
            if name.startswith("Unknown ("):
                continue
            if is_local_name(name, self.local_names) and not self.mixed:
                continue  # dual-channel: the local user is ch0, never embedded
            if self.mixed and name == "Me (Local)" and self.local_names:
                # Displayed under the canonical label, stored under the real
                # name so a future mixed recording can match it by voice.
                name = self.local_names[0]
            # Fold a differently-punctuated spelling into the existing enrolled key
            # ("Riley Stone" -> "Riley Stone") so one person isn't split across
            # two near-duplicate voiceprints that then fail to match each other.
            name = self._canonical_db_key(name)
            vecs = [self.emb[i] for i in g["seg_idxs"] if self.emb[i].size]
            if not vecs:
                continue
            # Enroll in the centered space so the print is comparable to future
            # centered query centroids (the pipeline stores voiceprints centered).
            centre = np.mean(vecs, axis=0)
            if self.mean is not None:
                centre = centre - self.mean
            centroid = _normalize(centre)
            stored = self.db.get(name)
            # A print of a different width came from a different embedding model
            # and says nothing about this one, so it is replaced rather than
            # blended. Blending would raise on the user's irreplaceable DB.
            if stored is not None and np.asarray(stored).shape != centroid.shape:
                stored = None
            if stored is not None and name not in self._reset_names:
                if not (g.get("confirmed") or self.trust_auto_names):
                    continue        # unconfirmed auto-fill: never move the print
                self.db[name] = _normalize(
                    (1 - self.update_rate) * stored + self.update_rate * centroid
                )
            else:
                self.db[name] = centroid
        self._save_db()

        transcript_md, names_used, transcript_names = self._rebuild_transcript()
        self._save_sidecar(transcript_names)
        # glossary.harvest sanitizes via clean_term, so placeholder labels
        # ("Unknown ...", "Me (Local)") are rejected there — no filter needed here.
        confirmed = [g["name"] for g in self.groups.values()]

        if self.llm_model:
            # Summarize the NAMED transcript, harvest terms, re-render the note.
            from export.llm_processor import LLMProcessor
            llm_data = LLMProcessor(model_name=self.llm_model, timeout=self.llm_timeout).generate_summary(transcript_md)
            glossary.harvest(self.glossary_path, names=confirmed,
                             entities=llm_data.get("entities", []))
            if self.note and os.path.exists(self.note):
                content = render_note(self._note_date_iso(), transcript_md,
                                      llm_data, sorted(names_used))
                atomic_write.write_text(self.note, content)
        else:
            # No LLM configured (unit tests): rebuild the transcript section only.
            self._write_note(transcript_md, sorted(names_used))

    def _note_date_iso(self):
        """Reuse the existing note's frontmatter date; fall back to the filename."""
        try:
            with open(self.note) as f:
                for line in f:
                    if line.startswith("date:"):
                        return line.split("date:", 1)[1].strip()
        except Exception:
            pass
        return timestamp_to_iso(os.path.basename(self.note).replace("_Meeting.md", ""))

    def _save_db(self):
        data = {k: np.asarray(v, dtype=float).tolist() for k, v in self.db.items()}
        atomic_write.write_text(self.db_path, json.dumps(data, indent=4))

    def _rebuild_transcript(self):
        """Regenerate the transcript body, the set of names used, and the final
        name of each transcript segment. Each segment keeps its Me (Local)
        override; otherwise it takes the final name of the diarization turn it
        most overlaps (mirroring assign_word_speakers).
        """
        seg_name = {}
        for g in self.groups.values():
            for i in g["seg_idxs"]:
                seg_name[i] = g["name"]

        out = ""
        current = None
        names_used = set()
        per_segment = []
        for t in self.transcript:
            if t["label"] == "Me (Local)":
                name = "Me (Local)"
            else:
                name = self._name_for_span(t["start"], t["end"], seg_name, t["label"])
            names_used.add(name)
            per_segment.append(name)
            mm, ss = int(t["start"] // 60), int(t["start"] % 60)
            if name != current:
                out += f"\n**{name}** ({mm:02d}:{ss:02d}):\n"
                current = name
            out += f"{t['text']} "
        return out.strip(), names_used, per_segment

    def _save_sidecar(self, transcript_names):
        """Write the confirmed names back into the sidecar the review opened.

        The note alone is not enough: every tool downstream reads the sidecar,
        so a correction that lives only in the note is undone by the next
        `tools/retranscribe.py` run and shown wrong by anything that lists a
        meeting by its speakers. Only the labels and their provenance change --
        cluster ids key the evaluation ground truth and the embeddings are the
        voiceprint evidence, so both are read back from disk untouched.
        """
        if not os.path.exists(self.sidecar_path):
            return
        with open(self.sidecar_path) as f:
            data = json.load(f)

        seg_name = {}
        for g in self.groups.values():
            for i in g["seg_idxs"]:
                seg_name[i] = g["name"]
        turns = data.get("diarization", [])
        if len(turns) != len(self.seg):
            return      # the file moved under us; leave it rather than corrupt it
        for i, row in enumerate(turns):
            if i in seg_name:
                row["label"] = seg_name[i]
                # A name the user confirmed is no longer the visual reader's or
                # the DB's guess, and must not be read back as one.
                row["source"] = "review"

        rows = data.get("transcript", [])
        if len(rows) == len(transcript_names):
            for row, name in zip(rows, transcript_names):
                row["label"] = name

        atomic_write.write_text(self.sidecar_path, json.dumps(data))

    def _name_for_span(self, start, end, seg_name, fallback):
        best, best_ov = None, 0.0
        for i, s in enumerate(self.seg):
            ov = min(end, s["end"]) - max(start, s["start"])
            if ov > best_ov:
                best_ov, best = ov, i
        if best is not None and best_ov > 0:
            return seg_name.get(best, fallback)
        return fallback

    def _write_note(self, transcript_md, final_names):
        if not self.note or not os.path.exists(self.note):
            return
        with open(self.note) as f:
            content = f.read()
        linked = inject_wikilinks(transcript_md, final_names)
        head = content.split("## Transcript")[0]
        new = head + "## Transcript\n" + linked + "\n"
        participants = "[" + ", ".join(f'"[[{n}]]"' for n in final_names) + "]"
        new = re.sub(r'^participants: .*$', f'participants: {participants}', new,
                     count=1, flags=re.M)
        atomic_write.write_text(self.note, new)

    def play_clip(self, gid):
        """Play (up to ClipPlayer.MAX_SECONDS of) the group's longest turn."""
        if not self.wav:
            return
        idxs = self.groups[gid]["seg_idxs"]
        longest = max(idxs, key=lambda i: self.seg[i]["end"] - self.seg[i]["start"])
        self._player.play(self.wav, self.seg[longest]["start"], self.seg[longest]["end"])

    def stop_playback(self):
        """Stop any clip currently playing, so audio never overlaps or outlives
        the review."""
        self._player.stop()

    def suggestion_names(self):
        """Names offered as typeahead in the review UI: enrolled voiceprints, the
        learned glossary, and names already assigned this session — never
        placeholders."""
        names = set(self.db.keys())
        try:
            names |= set(glossary.active_terms(self.glossary_path))
        except Exception:
            pass
        for g in self.groups.values():
            n = g.get("name", "")
            if n and not n.startswith("Unknown (") and n != "Me (Local)":
                names.add(n)
        if self.mixed:
            return sorted(names)
        return sorted(n for n in names if not is_local_name(n, self.local_names))

    def list_speakers(self):
        out = []
        for gid, g in self.groups.items():
            idxs = g["seg_idxs"]
            total = sum(self.seg[i]["end"] - self.seg[i]["start"] for i in idxs)
            longest = max(idxs, key=lambda i: self.seg[i]["end"] - self.seg[i]["start"])
            src = Counter(self.seg[i]["source"] for i in idxs).most_common(1)[0][0]
            out.append({
                "id": gid,
                "name": g["name"],
                "n_segments": len(idxs),
                "total_dur": total,
                "rep_clip": (self.seg[longest]["start"], self.seg[longest]["end"]),
                "source": src,
                "remember": self._should_remember(gid),
            })
        return out
