# Five-variant, >=1,000,000 DISTINCT SVNNUE3 positions each

This campaign is a native **data generation and validation** effort for
Stockless Variants SVNNUE-v3 Lite, **not an already trained Elo gain**.
The whole workflow fails unless Chess, Shogi, Janggi, Xiangqi and Makruk
**each** contain >=1,000,000 distinct royal-perspective input states.

[GitHub Actions million-per-game workflow](../.github/workflows/v3-diverse-million.yml)
[Native v3 evaluation architecture and prior NPS](SVNNUE_V3_LITE_RESULTS.md)

## Independently certified results — 2026-10-27

**All five million-unique-input objectives have been achieved and independently
checked against the original source.** GitHub Actions
[run 40475358787](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40475358787)
generated **all 60 legal native game-data shards successfully**, but its
aggregate jobs failed because the older compact audit omitted per-shard
manifest seed metadata. The original games and model inputs themselves
were complete. The separate [recovery/certification run 40501421340](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40501421340)
individually downloaded all 60 original artifacts, recomputed game/label
integrity and input hashes, recovered actual manifest seeds, and used
disk-backed SQLite to deduplicate across **all 12 shards for each variant**.
All five verification jobs and the final certificate passed.

| Game | Source games | Native labeled positions | Unique dual-royal NNUE inputs | Duplicate rows |
| --- | ---: | ---: | ---: | ---: |
| Chess | 12,288 | 1,414,820 | **1,384,403** | 30,417 |
| Shogi | 12,288 | 1,495,557 | **1,488,624** | 6,933 |
| Janggi | 12,288 | 1,520,135 | **1,497,338** | 22,797 |
| Xiangqi | 12,288 | 1,550,649 | **1,520,092** | 30,557 |
| Makruk | 12,288 | 2,172,148 | **2,107,658** | 64,490 |
| **Total** | **61,440** | **8,153,309** | **7,998,115** | **155,194** |

Original source generator commit:
[12063059](https://github.com/gaong5247-cmd/Stockless-variants/commit/12063059cdabc355053f6c9373bfe9642858c63c).
These are native feature/side-of-turn identities, not a claim that 7.998M
human-distinct *FENs* were created. No finished model Elo claim follows
from input cardinality.

The [certified-data training run 40502083572](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40502083572)
uses a sparse feature transformer with full optimizer checkpoint recovery.
The [research-release archival run 40502256268](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40502256268)
preserves the original shard ZIPs separately from engine binaries. Training
and releases have their own acceptance gates; this data certificate alone
does not claim that either has finished.

## Verified final measurements — 2026-10-27

The original native generator's [60 independent generation jobs](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40475358787) **all succeeded**. Its initial aggregate failed *only* because audit version 1 had omitted the seed field and the artifact downloader selected only the first 100 of 121 artifacts.

A separate [fully successful 6-job recovery and acceptance workflow](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40505070066) retrieved **all 120 original source and audit artifacts by exact name**, checked the original full games and per-position file hashes against the immutable generation manifests, recovered each shard seed strictly from its SHA-256-bound original manifest and used SQLite to deduplicate each game's complete native [side,white-perspective,black-perspective] inputs.

| Native game | Actual generated positions | SHA-256-distinct royal inputs | Real game histories |
|---|---:|---:|---:|
| Chess | 1,414,820 | **1,384,403** | 12,288 |
| Shogi | 1,495,557 | **1,488,624** | 12,288 |
| Janggi | 1,520,135 | **1,497,338** | 12,288 |
| Xiangqi | 1,550,649 | **1,520,092** | 12,288 |
| Makruk | 2,172,148 | **2,107,658** | 12,288 |
| **Total** | **8,153,309** | **7,998,115** | **61,440** |

All five game-level one-million distinct-source targets passed with true verified data, rather than simply counting the requested target. The game-specific hashes represent the same source feature semantics per variant, not a claim of one universal model.

### Separate checksum-preserved original data releases

[Research archive publishing completed successfully](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40505469623). Each separate release contains 12 original legal-game + NNUE feature ZIPs, an independent original-source certificate, a corpus manifest and SHA256SUMS.

- [Chess original data](https://github.com/gaong5247-cmd/Stockless-variants/releases/tag/svnnue3-diverse-chess-1m-20261027)
- [Shogi original data](https://github.com/gaong5247-cmd/Stockless-variants/releases/tag/svnnue3-diverse-shogi-1m-20261027)
- [Janggi original data](https://github.com/gaong5247-cmd/Stockless-variants/releases/tag/svnnue3-diverse-janggi-1m-20261027)
- [Xiangqi original data](https://github.com/gaong5247-cmd/Stockless-variants/releases/tag/svnnue3-diverse-xiangqi-1m-20261027)
- [Makruk original data](https://github.com/gaong5247-cmd/Stockless-variants/releases/tag/svnnue3-diverse-makruk-1m-20261027)

These are immutable-source *experimental data* research pre-releases, not trained v3 NNUE networks, and do not affect the stable original Stockless engine, Stockless Variants main branch or its latest binary release.

### Subsequent neural training: separate success criteria

[Five-model actual million-input training workflow](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40505711978) separately loads and re-verifies all twelve source shards, merges the immutable JSONL in bounded memory, builds a whole-game-split SQLite corpus and trains each SVNNUE3 Lite network. This uses the experimental touched-row sparse Feature Transformer AdaGrad updater and Adam for the small nonlinear network tail. It resumes full optimizer state at two epochs and finishes six total epochs, with a frozen native C++ heldout-int/search gate after training.

**A completed data campaign is not a trained network, a playing-strength gain, or an Elo result.** Track the model workflow independently.

## Input sources and diversity

One fixed, pinned and SHA-256-recorded native C++ engine generates all
legal self-play positions. Each of five games receives **12 independent
1,024-game shards** (12,288 games/variant and 61,440 games total).
Games are seeded independently by variant, shard, profile and game ID.
Within each shard, every requested state exports complete native legal
moves, game-history FEN, exact depth-qualified classical teacher score,
side to move, two perspective-specific royal NNUE feature vectors, and
native outcome including censored/unfinished states as null.

Four distinct source distributions, each with three shards:

| Profile | Depth target | Opening randomness | Continuation |
| --- | --- | --- | --- |
| Balanced | per-game 3-4 | 2-10 legal random plies | Best PV, no deliberate divergence |
| MultiPV | per-game 3-4 | 6-14 | About 9% of continuation decisions sampled from legal second/third MultiPV |
| Deep | 5 | 2-12 | Depth-5 main PV plus about 4% legal top alternatives |
| Exploration | per-game 2-3 | 8-20 | About 12% legal alternative PV trajectories |

Depth, sampling profile, opening and selected move source are recorded per
game and labeled row. All alternative PV moves are checked against the
native legal move set. In variants/positions where MultiPV alternatives
are unavailable, the generator retains the best move rather than inventing
an illegal move or calling a non-exact score an exact label.

Each game runs up to 216 plies. Only real terminal outcomes are labeled
wins/draws/losses. Games stopped at the ply cap have null outcomes and
may still contribute teacher-labeled nonterminal states.

## Verified counts, not wishful counts

Each generator writes three SHA-bound files:
1. games.jsonl — real move histories and terminal results
2. positions.jsonl — native royal features, exact teacher score and profile
3. fingerprints.sha256.bin — one **full 256-bit fingerprint per input row**
   over canonical [side_to_move, features_white, features_black]

The generation job independently validates every source JSON row, each
teacher score perspective, every in-game ply, each SHA-256 fingerprint
and the game row counts before uploading both full data and a compact
audit artifact. An independent per-variant reducer downloads just the
small audit and fingerprint files, rejects mixed teacher executables,
variant rules and duplicate shard seeds, and **deduplicates all input
fingerprints on disk using SQLite**. No in-memory dataset-wide Python set
or dense feature matrix is required.

Only if the actual deduplicated count for **every single variant** meets
1,000,000 does the final workflow publish its five-game certificate.
The target is not substituted for evidence. Cross-game inputs are kept
separate; an identical-looking position in two different rule systems is
never pooled for one variant's count.

## Using the data to train the actual v3 NNUE

For a variant, retrieve its twelve audited source artifacts or published
research ZIPs, and extract each source archive into a distinct numbered folder
under one parent directory (for example: corpus/chess/0 through /11). Each
folder must contain its original games.jsonl, positions.jsonl,
fingerprints.sha256.bin, audit.json and manifest.json files.

A streaming, lossless merger is implemented in
[merge_diverse_v3.py](../tools/nnue/merge_diverse_v3.py). It checks every
original native rule descriptor, teacher binary SHA, unique independent seed,
full JSONL SHA and every position fingerprint before concatenating (no rows or
game results are fabricated, reweighted, or discarded).

    python tools/nnue/merge_diverse_v3.py \
      --input-root corpus/chess --output build/chess-diverse-merged \
      --variant chess --shards 12 --games-per-shard 1024

Once merged, the original disk-backed learner can build an SQLite index from
that manifest and learn with the v3-specific king-bucket features:

    python tools/nnue/train_v3.py \
      --data build/chess-diverse-merged \
      --output build/chess-v3-trained --epochs 8 --batch-size 64

The trainer supports resumable full Adam state, but multi-million-parameter
dense moment updates can still be CPU-expensive for over one million rows.
Actual training duration, heldout teacher error and quantization checks must
be measured; this campaign proves the data source, not the convergence or
strength of any subsequently fitted model. Only promote network weights
after native integer parity, evaluation throughput and paired matches pass.

The [multi-shard merge regression workflow](../.github/workflows/v3-merge-smoke.yml)
passed all five game variants on real four-profile examples with whole-game
heldout partitions and lossless row counts. The same merging path supports
the full twelve-shard corpora.

## Verified generator and sparse training milestones

Before launching the 60-shard campaign, the following completed checks
were published:

- [Five-game four-profile real-engine smoke](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40475004484):
  all five games passed, with native legal alternatives selected from
  MultiPV, multiple depth distributions and byte/SHA-bound FEN/feature
  exports. This verifies **the sampling procedure**, not the million target.
- [Cross-shard on-disk cardinality audit](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40475716492):
  all five groups of four independent native source shards passed a
  disk-backed SHA-256 deduplication proof.
- [Five-game lossless JSONL merge and training index](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40476852017):
  5/5 verified real native corpora combined and loaded with complete-game
  train/validation/test separation.
- [Sparse optimizer and trained native parity](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40478315847):
  all five models ran a two-epoch smoke training with exact
  checkpoint/optimizer recovery, 180 successful C++ native
  integer/search checks in total and zero failures.

### Scalable optimizer

The new --optimizer sparse_ft_adagrad uses a row-sparse AdaGrad update for
the large input feature transformer, but full Adam on the small 32->16->1
network tail. Its backward pass calculates gradients for *touched feature
rows only*, instead of allocating input_dim x 128 gradients on every
minibatch. Only actual royal-perspective CSR minibatches enter memory.
Whole-optimizer state, the dataset signature, optimizer mode, batch size,
seed and epoch/steps are checksummed; trying to resume an old dense-Adam
checkpoint with the new optimizer is explicitly rejected.

This is a different optimizer from the former pure Adam reference.
Sparse transformer rows do **not** receive stale momentum updates on
batches in which they do not appear. The old dense optimizer remains
supported and unaffected; tests compare the sparse gradient against a
fully materialized FP32 reference and require deterministic restarts.

The separate [large native training workflow](../.github/workflows/train-v3-million.yml)
retrains each independently certified 1M+ corpus, checkpointing after
the first epoch and resuming the exact optimizer state before quantization
and heldout native C++ evaluation. It is pinned to the separately successful
source recovery certificate, since the historical generation run's audit
summary failed despite all 60 data jobs succeeding. Dataset certification
must not be confused with finished model training.

An independent [data release archiver](../.github/workflows/archive-v3-diverse.yml)
can publish the 60 verified source shard ZIPs as five game-specific
research pre-releases after source certification. This is separate
from the latest engine binary Release and keeps full original move
histories and training labels, not just NNUE weights.

## Data availability and follow-on training

Each of the 60 full shard artifacts contains complete histories, feature
vectors, labels, manifests and the independent audit. Default retention
is 14 days. Five exact per-variant acceptance summaries and a complete
global campaign certificate are retained for 90 days. If needed longer,
archive verified original data to a separate immutable research GitHub
release; merely keeping a model's weights would not enable fresh training.

These files use native **SVNNUE3 royal-bucket** features, not the earlier
v1/v2 schema. They can be used to train the documented 128-per-perspective
SVNNUE3 Lite model using the disk-backed CSR loader and full Adam
resumable trainer. Native model training, quantization and paired
NNUE/Classical matches need their own measured validation; a million
diverse input states alone does not prove stronger playing strength.

The unique cardinality target is independently certified; strength and
Elo remain unmeasured until further native engine matches complete.
