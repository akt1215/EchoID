"""Score every strategy on the same trials and print one comparison table.

Run:  .venv/bin/python -m evaluation.run
"""

import argparse
import os
import sys

import numpy as np

from core import embedding_mean
from evaluation import clusters, ground_truth, scorers, trials


def separation(scorer, groups, gt):
    """Positive/negative score distributions for one strategy."""
    cents = {(g.meeting, g.cluster): scorer.centroid(g) for g in groups}
    pos = [scorer.score(cents[(a.meeting, a.cluster)], cents[(b.meeting, b.cluster)])
           for a, b in trials.positives(groups, gt)]
    neg = [scorer.score(cents[(a.meeting, a.cluster)], cents[(b.meeting, b.cluster)])
           for a, b in trials.negatives(groups, gt)]
    return np.asarray(pos), np.asarray(neg)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="workspace")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--backend", default=None,
                        choices=["speechbrain", "wespeaker_onnx"],
                        help="re-extract embeddings with this backend instead of "
                             "using the ECAPA vectors stored in the sidecars")
    parser.add_argument("--model", default=None,
                        help="model id for --backend")
    parser.add_argument("--max-turns", type=int, default=40,
                        help="turns per cluster to embed when re-extracting")
    args = parser.parse_args(argv)

    gt_path = args.ground_truth or os.path.join(
        args.workspace, "eval", "ground_truth.json")
    gt = ground_truth.load(gt_path)
    if not gt:
        print(f"No ground truth at {gt_path}. Run: "
              f".venv/bin/python -m evaluation.label_cli")
        return 1

    groups = clusters.discover(args.workspace)
    mean, _ = embedding_mean.load(
        os.path.join(args.workspace, "embedding_mean.json"))
    source = "sidecar embeddings (whatever the last pipeline run wrote)"

    if args.backend:
        # A different embedding model needs its own vectors and its own mean; the
        # persisted mean belongs to ECAPA's space and would be meaningless here.
        import yaml

        from ai.biometrics import BiometricsManager, model_for
        from evaluation import embeddings as emb

        with open("config.yaml") as f:
            default_model = model_for(
                {**yaml.safe_load(f)["biometrics"], "backend": args.backend})
        path = emb.cache_path(args.workspace, args.backend)
        cache = emb.load_cache(path)
        print(f"re-extracting with {args.backend} "
              f"({len(cache)} clusters already cached)")
        manager = BiometricsManager(
            db_path=os.path.join(args.workspace, "speakers.json"),
            backend=args.backend, model_name=args.model or default_model,
            target_channel=1)
        matrices = emb.extract(groups, manager, cache, max_turns=args.max_turns)
        emb.save_cache(path, cache)
        groups = emb.regroup(groups, matrices)
        mean = None
        source = f"{args.backend} re-extracted from ch1, no global mean"

    n_pos = len(trials.positives(groups, gt))
    n_neg = len(trials.negatives(groups, gt))
    print(f"\nsource: {source}")
    print(f"{len(groups)} clusters, {len(ground_truth.labeled_keys(gt))} labeled")
    print(f"{n_pos} positive pairs, {n_neg} negative pairs")
    if n_pos < 5:
        print("!! too few positives for a reliable miss-rate estimate; "
              "treat the positive column as directional only")
    if not n_pos or not n_neg:
        print("Not enough trials to compare strategies.")
        return 1

    print(f"\n{'strategy':<12}{'pos mean':>10}{'pos min':>10}"
          f"{'neg mean':>10}{'neg max':>10}{'margin':>10}")
    print("-" * 62)
    for name, cls in scorers.SCORERS.items():
        scorer = cls(mean=mean)
        scorer.prepare(groups)
        pos, neg = separation(scorer, groups, gt)
        # The gap between the worst true match and the best impostor: positive
        # means a threshold exists that separates them, negative means none does.
        print(f"{name:<12}{pos.mean():>10.3f}{pos.min():>10.3f}"
              f"{neg.mean():>10.3f}{neg.max():>10.3f}"
              f"{pos.min() - neg.max():>10.3f}")
    print("\nmargin = worst true match minus best impostor. A negative margin "
          "means NO threshold separates them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
