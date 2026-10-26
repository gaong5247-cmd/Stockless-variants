# Real million-plus native NNUE data campaign

**Verified result:** this campaign completed on 2026-10-26 at 11:32:08 UTC.
The pipeline generated **10,240 native-teacher games, 1,247,386 labeled
positions**, and retained **1,232,430 distinct NNUE inputs** after deduplication.
All five per-variant model jobs, ten shard jobs and the final exact-count
assertion completed successfully. Each game could run up to 160 plies with
an eight-ply randomized legal opening and a depth-4 classical teacher.

**[Full measured results, downloads, and NNUE-vs-Classical scores](MEGA_NNUE_RESULTS.md)**
are kept separately from the planned workflow definition.

[Full workflow](../.github/workflows/train-mega-nnue.yml) |
[GitHub Actions results](https://github.com/gaong5247-cmd/Stockless-variants/actions/workflows/train-mega-nnue.yml)

The completed pipeline first built one pinned native teacher, uses it identically for
ten independent datasets, and includes rich native-context version-2
diagnostics with full data provenance. It does **not** treat context
diagnostics as deployed v2 NNUE input features. Games that hit the move
limit retain null results. The original Actions dataset artifacts expire after 14 days, but a separate checksum-verified [research prerelease](https://github.com/gaong5247-cmd/Stockless-variants/releases/tag/nnue-corpus-1.247m-20261026) now preserves the 10 shard archives and five checkpoint/model archives.

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
were generally much weaker than Classical. The new mega models were also tested in [240 matched games](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40318979116): NNUE scored 29 wins, 16 draws, 188 losses and seven censored games. Classical remained stronger. The experiment does not establish Elo gains.
