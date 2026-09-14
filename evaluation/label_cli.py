"""Interactive labeling: play a clip per cluster, record who it actually is.

Run:  .venv/bin/python -m evaluation.label_cli
Resumable — already-labeled groups are skipped unless --relabel is passed, and
every answer is saved immediately so an interrupted session loses nothing.
"""

import argparse
import os
import sys

from core.clip_player import ClipPlayer
from evaluation import clusters, ground_truth


SILENT_RMS = 0.002      # below this a cluster has no audible speech at all


def _prompt(group, known_names, player, spans, level):
    played = sum(e - s for s, e in spans)
    print(f"\n{'=' * 70}")
    print(f"  {group.meeting}   cluster {group.cluster}")
    print(f"  {group.n_turns} turns, {group.total_seconds / 60:.1f} min of speech")
    if clusters.is_leaky(group.cluster):
        print("  (named by the voiceprint DB — usable as a negative only)")
    print(f"  playing {played:.1f}s from {len(spans)} of the loudest turns")
    if level is not None and level < SILENT_RMS:
        # Say so rather than letting the user hunt for a voice that is not there.
        print(f"  !! this cluster is essentially SILENT (peak rms {level:.5f}) —"
              f" almost certainly a diarization artifact. [s]kip is correct here.")
    if known_names:
        print("\n  names used so far: " + ", ".join(sorted(known_names)))
    print("\n  Enter a name, or:  [r]eplay  [m]ultiple people  [s]kip  [q]uit")

    while True:
        player.play_spans(group.wav, spans)
        answer = input("  > ").strip()
        if answer.lower() == "r":
            continue
        if answer.lower() == "q":
            return None
        if answer.lower() == "m":
            return ("multiple", None)
        if answer.lower() == "s":
            return ("skip", None)
        if answer:
            return ("confirmed", answer)
        print("  (empty — type a name, or one of r/m/s/q)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--out", default=None,
                        help="ground-truth file (default <workspace>/eval/ground_truth.json)")
    parser.add_argument("--relabel", action="store_true",
                        help="revisit groups that already have a label")
    parser.add_argument("--only", default=None,
                        help="revisit just the groups whose meeting or cluster "
                             "contains this text, keeping every other answer")
    args = parser.parse_args(argv)

    out = args.out or os.path.join(args.workspace, "eval", "ground_truth.json")
    data = ground_truth.load(out)
    groups = clusters.discover(args.workspace)

    # Labels belong to the diarization run they were made against. A reprocess
    # can reuse a cluster id for different audio, so a stale verdict would be
    # applied silently to a cluster nobody ever listened to.
    data, stale = ground_truth.prune_stale(data, groups)
    if stale:
        print("These meetings were reprocessed since they were labeled, so their "
              "labels no longer describe the current clusters and were dropped:")
        for meeting in stale:
            print(f"  {meeting}")
        ground_truth.save(out, data)
    if args.only:
        needle = args.only.lower()
        todo = [g for g in groups
                if needle in g.meeting.lower() or needle in g.cluster.lower()]
    else:
        todo = [g for g in groups
                if args.relabel or not ground_truth.get(data, g.meeting, g.cluster)]

    print(f"{len(groups)} clusters found, {len(todo)} to label. Saving to {out}")
    if not todo:
        print("Nothing to do.")
        return 0

    player = ClipPlayer()
    known = {name for meeting in data.values() for entry in meeting.values()
             if (name := entry.get("name"))}
    try:
        for i, group in enumerate(todo, 1):
            spans = clusters.loudest_turns(group)
            level = clusters.peak_rms(group)
            print(f"\n[{i}/{len(todo)}]", end="")
            answer = _prompt(group, known, player, spans, level)
            if answer is None:
                print("\nStopping. Progress is saved.")
                break
            verdict, name = answer
            ground_truth.set_label(data, group.meeting, group.cluster, verdict, name)
            ground_truth.save(out, data)     # save every answer, never lose work
            if name:
                known.add(ground_truth.clean(name))
    finally:
        player.stop()

    labeled = len(ground_truth.labeled_keys(data))
    print(f"\n{labeled}/{len(groups)} clusters labeled. Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
