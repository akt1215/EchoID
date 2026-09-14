# evaluation/ — measuring speaker identity

Development-only. Nothing here runs in the recording pipeline.

## Why ground truth has to come from a human

Every identity label already in the workspace is unusable for scoring:

- **Sidecar `label` fields** come from the vision-LLM name reader, which is the
  component under investigation. Two 2026-07 meetings have every segment labeled
  with the local user's name across three distinct clusters.
- **`db::`-prefixed cluster ids** are circular. `resolve_clusters` names a
  cluster from the voiceprint DB, then `speaker_review.commit()` updates that
  same voiceprint from that cluster's audio. Scoring a match against one reports
  ~0.97 self-similarity and proves nothing.

So `trials.positives()` excludes `db::` clusters. They are still fine as
negatives and for counting conflations. This is the easiest way to get a falsely
good number here — if a change suddenly looks great, check this first.

## Running it

```bash
# 1. Label the clusters (~20 of them, resumable, saves after every answer)
.venv/bin/python -m evaluation.label_cli

# 2. Compare scoring strategies on the labeled data
.venv/bin/python -m evaluation.run

# 3. Rebuild the voiceprint DB from confirmed clusters — scratch first, then diff
.venv/bin/python -m tools.rebuild_speakers_db --out /tmp/scratch-speakers.json
```

Ground truth lands in `workspace/eval/ground_truth.json`. Nothing except
`tools/rebuild_speakers_db.py`, run deliberately, writes to
`workspace/speakers.json`, and that backs up to `.pre-rebuild.bak` first.

## Metrics, in priority order

1. **Conflation rate** — several people fused into one identity. **Hard
   constraint, target 0.** Unrecoverable once it happens; an over-split costs
   one merge click in review.
2. **False-accept rate** — a cluster given the wrong existing name. `Unknown` is
   the safe outcome and is never counted as an error.
3. **Auto-label coverage** — share of speech time named *correctly*, maximized
   only subject to 1 and 2.

Reported together on purpose. Coverage alone would reward a change that
eliminates collisions by labeling everyone `Unknown`.

## Known limit

After the leakage exclusion, only 12 raw `SPEAKER_NN` clusters across 7 meetings
can supply cross-meeting positives, and how many actual pairs that yields depends
on how many people recur. Negatives are abundant.

**Conflation and false-accept are measured solidly; miss rate is directional.**
`evaluation/run.py` prints a warning when there are fewer than five positive
pairs. Do not report a strategy ranking on the miss axis as if it were a real
ROC.

## Modules

| file | responsibility |
|---|---|
| `ground_truth.py` | load/save human labels; verdicts `confirmed`/`multiple`/`skip` |
| `clusters.py` | discover (meeting, cluster) groups from sidecars; `is_leaky` |
| `label_cli.py` | interactive clip playback and prompting |
| `trials.py` | build positive/negative pairs, enforcing the leakage rule |
| `metrics.py` | conflation, false accepts, coverage |
| `scorers.py` | interchangeable centroid/scoring strategies for the bake-off |
| `run.py` | score every strategy on the same trials, print the table |

`multiple` is a first-class verdict: a cluster holding two people is itself a
diarization conflation, so it is recorded rather than skipped.
