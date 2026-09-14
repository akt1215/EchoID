"""Re-verify a cluster labeled `multiple`, one turn at a time.

`label_cli` judges a cluster from a single stitched clip that `loudest_turns`
assembles from bursts spread across the whole span. That montage is what put the
`multiple` verdicts in doubt: fragments taken minutes apart, played back to back
with abrupt cuts, can sound like several people even when they are not.

This tool asks the question the other way round, and aims it at the actual
failure mode. Measured on the labeled sidecars, the speech intruding into a
mixed cluster is almost entirely SHORT turns — backchannels and brief
interjections — while every long turn belongs to the dominant voice. So a review
that plays long turns can only ever confirm the dominant speaker.

Each cluster is therefore presented as:

  BASELINE  a few of the longest turns, sampled across the cluster's span and
            played individually — this is the voice the cluster claims to be;
  PROBES    the short turns least like that voice, played individually with
            their transcript text and their cosine to the dominant centroid.

Nothing is stitched and no raw window of the recording is ever played: a
contiguous slice of the meeting would contain other clusters' speech and would
induce exactly the false "multiple" answer this tool exists to rule out.

Run:  .venv/bin/python -m evaluation.verify_cluster
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

from ai.identity_resolution import _centroid
from core.clip_player import ClipPlayer
from evaluation import clusters, ground_truth

LONG_SECONDS = 3.0      # a turn long enough to embed the speaker reliably


def dominant_centroid(turns, embeddings, long_seconds=LONG_SECONDS):
    """Duration-weighted centroid of the turns at or above `long_seconds`.

    Short turns are excluded rather than down-weighted: below a second their
    embeddings fail to match even their own speaker most of the time, so
    including them is what blends a reference voice in the first place.
    Returns None when the cluster has no turn long enough to trust.
    """
    idxs = [i for i, (s, e) in enumerate(turns) if e - s >= long_seconds]
    if not idxs:
        return None
    return _centroid([embeddings[i] for i in idxs],
                     durations=[turns[i][1] - turns[i][0] for i in idxs])


def _cos(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _spread(idxs, turns, n):
    """The longest turn in each of `n` buckets across the cluster's span.

    Taking the n longest turns outright can draw them all from one stretch, and a
    speaker change elsewhere in the meeting is then never heard.
    """
    if len(idxs) <= n:
        return sorted(idxs)
    first = min(turns[i][0] for i in idxs)
    last = max(turns[i][1] for i in idxs)
    span = max(last - first, 1e-6)
    best = {}
    for i in idxs:
        b = min(n - 1, int((turns[i][0] - first) / span * n))
        cur = best.get(b)
        if cur is None or (turns[i][1] - turns[i][0]) > (turns[cur][1] - turns[cur][0]):
            best[b] = i
    return sorted(best.values())


def select_probes(turns, embeddings, long_seconds=LONG_SECONDS,
                  n_baseline=3, n_probe=12):
    """`(baseline, probes)` turn indices to play for one cluster.

    `baseline` is long turns spread across the span; `probes` is
    `[(index, cosine)]` for the short turns, least like the dominant voice first,
    so the most likely intruder is heard immediately. A cluster with no long turn
    falls back to its longest turn as the reference so it stays reviewable.
    """
    long_idxs = [i for i, (s, e) in enumerate(turns) if e - s >= long_seconds]
    reference = dominant_centroid(turns, embeddings, long_seconds)

    if reference is None:
        by_length = sorted(range(len(turns)),
                           key=lambda i: turns[i][1] - turns[i][0], reverse=True)
        baseline, rest = by_length[:1], by_length[1:]
        reference = np.asarray(embeddings[baseline[0]], dtype=float)
    else:
        baseline = _spread(long_idxs, turns, n_baseline)
        rest = [i for i in range(len(turns)) if i not in set(long_idxs)]

    scored = sorted(((i, _cos(embeddings[i], reference)) for i in rest),
                    key=lambda t: t[1])
    return baseline, scored[:n_probe]


def _transcripts(workspace, meeting):
    """[(start, end, text)] for one meeting, or [] when the sidecar has none."""
    path = os.path.join(workspace, f"{meeting}.segments.json")
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return [(float(t["start"]), float(t["end"]), t.get("text", ""))
            for t in data.get("transcript", [])]


def _text_at(transcript, start, end):
    """What was said during a turn, for the reviewer to read while it plays."""
    parts = [t for s, e, t in transcript if min(end, e) - max(start, s) > 0]
    return " ".join(p.strip() for p in parts).strip()


def _clock(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _play(player, wav, start, end, gap=0.35):
    """Play one turn and wait it out, so turns are heard as separate clips."""
    player.play(wav, start, end)
    player.wait()
    time.sleep(gap)


def _present(group, transcript, baseline, probes, player):
    print(f"\n{'=' * 72}")
    print(f"  {group.meeting}   cluster {group.cluster}")
    print(f"  {group.n_turns} turns, {group.total_seconds / 60:.1f} min of speech")

    print(f"\n  BASELINE — the voice this cluster claims to be "
          f"({len(baseline)} long turns)")
    for n, i in enumerate(baseline, 1):
        s, e = group.turns[i]
        print(f"    [{n}/{len(baseline)}] {_clock(s)}  {e - s:4.1f}s  "
              f"{_text_at(transcript, s, e)[:88]}")
        _play(player, group.wav, s, e)

    if not probes:
        print("\n  PROBES — none: every turn in this cluster is long.")
        return
    print(f"\n  PROBES — short turns least like that voice ({len(probes)})")
    print("  Listen for a DIFFERENT voice, not for a clean recording.")
    for n, (i, score) in enumerate(probes, 1):
        s, e = group.turns[i]
        print(f"    [{n}/{len(probes)}] {_clock(s)}  {e - s:4.1f}s  cos={score:+.3f}  "
              f"{_text_at(transcript, s, e)[:78]}")
        _play(player, group.wav, s, e)


def _backup_once(path, done):
    """Copy the ground truth aside before this run's first write."""
    if done or not os.path.exists(path):
        return True
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    dest = f"{path}.pre-verify_{stamp}.bak"
    shutil.copy2(path, dest)
    print(f"backed up {path} -> {dest}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--only", default=None,
                        help="restrict to clusters whose meeting or cluster id "
                             "contains this text")
    parser.add_argument("--verdict", default="multiple",
                        help="which existing verdict to revisit (default multiple)")
    parser.add_argument("--n-baseline", type=int, default=3)
    parser.add_argument("--n-probe", type=int, default=12)
    args = parser.parse_args(argv)

    gt_path = args.ground_truth or os.path.join(
        args.workspace, "eval", "ground_truth.json")
    data = ground_truth.load(gt_path)
    if not data:
        print(f"No ground truth at {gt_path}.")
        return 1

    groups = clusters.discover(args.workspace)
    # A label belongs to the diarization run it was made against; reuse of a
    # cluster id across runs would otherwise point this review at other audio.
    data, stale = ground_truth.prune_stale(data, groups)
    if stale:
        print("Dropped labels for reprocessed meetings: " + ", ".join(stale))
        ground_truth.save(gt_path, data)

    todo = []
    for g in groups:
        entry = ground_truth.get(data, g.meeting, g.cluster)
        if not entry or entry.get("verdict") != args.verdict:
            continue
        if args.only and args.only.lower() not in (g.meeting + g.cluster).lower():
            continue
        todo.append(g)
    todo.sort(key=lambda g: -g.total_seconds)

    print(f"{len(todo)} cluster(s) labeled '{args.verdict}' to re-verify.")
    if not todo:
        return 0

    player = ClipPlayer()
    backed_up = False
    try:
        for n, group in enumerate(todo, 1):
            transcript = _transcripts(args.workspace, group.meeting)
            baseline, probes = select_probes(
                group.turns, group.embeddings,
                n_baseline=args.n_baseline, n_probe=args.n_probe)
            print(f"\n[{n}/{len(todo)}]", end="")
            _present(group, transcript, baseline, probes, player)

            while True:
                print("\n  Same voice throughout -> type the NAME."
                      "\n  Genuinely more than one person -> [m]."
                      "\n  Also: [b]aseline again  [p]robes again  [s]kip  [q]uit")
                answer = input("  > ").strip()
                if answer.lower() == "b":
                    for i in baseline:
                        _play(player, group.wav, *group.turns[i])
                    continue
                if answer.lower() == "p":
                    for i, _ in probes:
                        _play(player, group.wav, *group.turns[i])
                    continue
                if answer.lower() == "q":
                    print("\nStopping. Answers so far are saved.")
                    return 0
                if answer.lower() == "m":
                    verdict, name = "multiple", None
                elif answer.lower() == "s":
                    verdict, name = "skip", None
                elif answer:
                    verdict, name = "confirmed", answer
                else:
                    print("  (empty — type a name, or one of b/p/m/s/q)")
                    continue
                backed_up = _backup_once(gt_path, backed_up)
                ground_truth.set_label(data, group.meeting, group.cluster,
                                       verdict, name)
                ground_truth.save(gt_path, data)
                break
    finally:
        player.stop()

    print(f"\nSaved to {gt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
