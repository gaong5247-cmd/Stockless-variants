# Long NNUE training — bounded first large run

The initial experimental 0.2.0-dev pilot trained 32 games per variant. This
separate workflow runs **two 256-game shards per variant**, for five variants:
**2,560 generated games requested**, with up to 136 plies per game, 4-ply
classical search labels and 60 CPU epochs on a merged dataset for each game.

This is a *target*, not a claim that the asynchronous GitHub run has completed.
Always open the latest Actions run and inspect individual generation/manifests,
model metrics and validation results before citing final counts.

- [Long NNUE training workflow](../.github/workflows/train-long-nnue.yml)
- [GitHub Actions — all runs](https://github.com/gaong5247-cmd/Stockless-variants/actions/workflows/train-long-nnue.yml)
- [Manual workflow trigger](https://github.com/gaong5247-cmd/Stockless-variants/actions/workflows/train-long-nnue.yml)

## Design

1. One pinned Linux teacher binary is compiled and uploaded once per run.
2. Two independent generation jobs per selected variant use disjoint RNG seeds
   but precisely the same teacher executable, rules fingerprint and feature
   descriptor. Each shard is stored as its own checksum-bound artifact.
3. The merger verifies identity, hashes, byte lengths, unique game IDs, record
   counts and that every labeled position references a game in its own shard.
   It preserves original game results, including null censored outcomes.
4. Original trainer partitions by whole game, deduplicates feature-plus-side
   inputs across the split, trains one model per game with validation-only
   checkpoint selection, tests after selection and quantizes.
5. Native validation checks frozen heldout positions against the actual C++
   integer inference and four legal searches. Model artifacts include
   provenance, checksums, metrics, split, float checkpoint and .svnnue file.

The existing trainer is bounded by **100,000 positions and 512 MiB of dense
feature storage**. With 2 shards × 256 games and an 8-ply random opening,
at most 65,536 labeled positions can be generated for any single model.
For 9x9 Shogi, the largest current feature count is 1,640, so this is below
the memory guard. Do not increase shards or max plies without rewriting the
trainer for sparse streaming first.

Per-generation job timeout: 320 minutes. Per-training job: 330 minutes.
GitHub-hosted jobs have a platform limit of 6 hours. Shard artifacts are
retained for 14 days and model artifacts for 30 days; download essential
datasets and checkpoints before they expire. Artifacts are not permanently
archived in git. The workflow uses the repository's normal read-only token
and standard public-repository runners. Data volume and job limits may still
cause a run to fail; neither a target count nor training loss guarantees
strength improvement.

There is **no cross-run optimizer resumption**, schedule, automatic
retraining, automatic evaluation promotion or Elo improvement claim in this
version. Use workflow_dispatch to launch a new reproducible experiment;
save previous model/data artifacts separately. The older 32-game pilot remains
a baseline.

### Manual options

Select one variant or all five; adjust games per shard (16..256), maximum
plies (32..136), teacher depth (2..5), CPU epochs (1..120), and base seed.
A new seed generates a new dataset; do not call the resulting run a
continuation of the previous optimizer checkpoint.

**Priority after results:** richer feature encoding (especially promoted
origin and history), sparse streaming training, more diversified teacher
positions, genuine checkpoint resume, longer paired Classical/NNUE matches,
and per-variant strength statistics.
