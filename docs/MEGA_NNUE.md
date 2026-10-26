# Real million-plus native NNUE data campaign

**Target, not a result until the Actions jobs finish:** generate two separate
1,024-game native-teacher datasets for each of chess, shogi, janggi, xiangqi,
and makruk. This targets **10,240 genuinely generated games**. Every game can
produce up to 152 labeled positions after eight randomized legal opening plies,
with a 160-ply maximum and depth-4 classical teacher. The actual generated
position total is measured from the manifests, never invented.

[Full workflow](../.github/workflows/train-mega-nnue.yml) |
[GitHub Actions results](https://github.com/gaong5247-cmd/Stockless-variants/actions/workflows/train-mega-nnue.yml)

The pipeline first builds one pinned native teacher, uses it identically for
ten independent datasets, and includes rich native-context version-2
diagnostics with full data provenance. It does **not** treat context
diagnostics as deployed v2 NNUE input features. Games that hit the move
limit retain null results. Dataset artifact retention is 14 days.

Each variant then indexes two JSONL shards using bounded-memory SQLite,
keeps whole-game train/validation/test partitions disjoint, and learns the
native SVNNUE-v1 network in 24 CPU epochs. It deliberately checkpoints at
epoch 4 then resumes the full Adam moments, global step count and selected
best checkpoint to epoch 24. Checksum, variant, teacher and source drift
reject invalid resume. Actual trained-model parity and legal searches are
checked against the **same native engine** before result artifacts are
uploaded for 30 days. Experimental NNUE remains opt-in.

A final summary independently totals all five indexed source manifests and
fails when the objective of **at least 1,000,000 actual generated native
positions** has not been achieved. It never substitutes the separately
verified synthetic stress corpus for true generated data.

This is large *supervised distillation* from a depth-4 classical search
teacher, not reinforcement learning or Stockfish-strength training. The
previous [120 paired games](NNUE_ADVANCED.md) showed that early neural models
were generally much weaker than Classical. No Elo or strength gain is claimed
without new paired matches against the resulting models, including censored
games, openings, color assignment and confidence intervals.
