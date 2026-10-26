# Independent experimental NNUE

Version `0.2.0-dev` adds **SVNNUE-v1**, a separate trainable network format and native C++ evaluator. The first targets are `chess`, `shogi`, `janggi`, `xiangqi` and `makruk`. Each target has its own data, weights, model identity and validation report. No upstream neural weights are copied, and a chess model cannot be loaded for Shogi or Janggi.

The default evaluator remains **Classical**. This is a small CPU training pipeline with actual parameter updates and quantized inference, not a playing-strength claim. A successful workflow means its numerical, data-integrity and runtime checks passed; it does not establish an Elo gain or approve a model for release.

## Run an experiment in GitHub Actions

Open [Train experimental variant NNUE](https://github.com/gaong5247-cmd/Stockless-variants/actions/workflows/train-nnue.yml), select **Run workflow**, and choose one variant or `all`.

| Input | Manual default | Allowed range |
| --- | ---: | --- |
| Variant | `all` | Five targets above, or `all` |
| Games per variant | 64 | 6–256 |
| Maximum plies per game | 120 | 16–256 |
| Classical teacher depth | 3 | 1–6 |
| Epochs | 80 | 1–500 |
| Seed | 20261026 | Unsigned 32-bit integer |

Relevant source changes on `main` also trigger a bounded pilot: **32 games, 80 plies, depth 3, 40 epochs per variant**. Up to three variant jobs run concurrently. Jobs use CPU NumPy, one BLAS thread and a 45-minute timeout. Larger manual settings can hit the timeout or the trainer's data/memory limits; they are not a promise of sufficient compute for strong networks. There is no scheduled training, GPU allocation or automatic model promotion.

Each job:

1. Builds the exact engine from pinned sources and checks the loader, integer arithmetic and incremental accumulator.
2. Generates native features and classical-search labels through legal self-play.
3. Splits whole games, trains a separate model, selects a validation checkpoint and quantizes it.
4. Loads the resulting file in that engine and checks heldout integer predictions and legal searches.
5. Uploads data, model, measurements, training scripts, license notices and a compatible Linux engine with complete corresponding C++ source.

Artifacts are named `nnue-<variant>-<commit>-<attempt>` and configured for 30-day retention. Failed jobs can also upload partial diagnostics; inspect the job conclusion and `runtime-validation.json` before using a model. The workflow has read-only repository permissions and does not commit generated weights or publish a release.

## Local commands

The engine itself needs no NumPy. The optional trainer uses Python 3.11+ and the pinned dependency in `tools/nnue/requirements.txt` (the workflow uses Python 3.12).

```bash
make build ARCH=x86-64 JOBS=2
python -m pip install -r tools/nnue/requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python tools/nnue/generate.py \
  --engine build/engine/src/stockless-variants --variant janggi \
  --games 32 --max-plies 80 --depth 3 --seed 20261026 \
  --output build/experiment/janggi/data

python tools/nnue/train.py \
  --data build/experiment/janggi/data \
  --output build/experiment/janggi/model --epochs 40 --seed 20261026

python tools/nnue/validate.py \
  --engine build/engine/src/stockless-variants \
  --data build/experiment/janggi/data \
  --model-dir build/experiment/janggi/model \
  --report build/experiment/janggi/runtime-validation.json
```

Output directories must be empty. Keep each experiment rather than overwriting previous data or a checkpoint. The dense CPU trainer caps inputs at 100,000 rows and 512 MiB for its feature matrix.

## Use the trained model

Use the engine supplied in the same experiment artifact, or a build with the same `rules_sha256`. Version `0.1.0-dev` binaries do not implement SVNNUE-v1.

```text
uci
setoption name UCI_Variant value janggi
setoption name EvalFile value /absolute/path/janggi.svnnue
setoption name EvalBackend value StocklessNNUE
isready
nnuestatus
position startpos
nnueeval
go depth 5
```

`nnuestatus` must report `active: true` and the correct `model_variant`. `nnueeval` reports whether NNUE was used, the exact integer numerator, white/side-to-move centipawns and a refreshed-accumulator comparison. `eval` also identifies the selected evaluation contribution. A normal UCI search still applies the engine's native terminal logic and subsequent evaluation scaling, so its reported score need not equal a bare `nnueeval` output.

| Option | Behavior |
| --- | --- |
| `EvalBackend=Classical` | Default; ignores neural weights for evaluation |
| `EvalBackend=StocklessNNUE` | Requires an already loaded model compatible with the current variant |
| `EvalBackend=Auto` | Uses a loaded compatible SVNNUE model when applicable; otherwise classical |
| `EvalFile` | Loads a validated `.svnnue`; loading alone does not change the selected backend |
| `Use NNUE=true` | Rejected: this inherited option refers to the incompatible FSF architecture |

Failed loads preserve the previous valid model and backend. Switching variant invalidates cached accumulators; an incompatible explicitly selected model is deactivated. Chess960 positions and unlisted/custom variants use the classical fallback. Model/backend mutations stop and join active search threads before replacing shared state or clearing the search caches.

## Native features and incremental evaluation

Features come from `nnuefeatures` in the actual rule engine, not from a second Python interpretation of FEN. This diagnostic is available in UCI general notation and exports the descriptor, native sparse features, pieces, pockets, legal moves, FEN, side to move and terminal details. USI/UCCI remain available for ordinary play; data generation uses the general UCI notation consistently.

Let `W × H` be the board geometry, `T` the count of sorted native piece-type IDs and `A = W × H`. For a piece type's index `t`, color `c` and board slot `s = rank × W + file`, the feature is:

```text
index = ((c * T + t) * (A + 1)) + s
input_dim = 2 * T * (A + 1)
```

Board slots have value one. Slot `A` is that color/type's pocket count. Feature L1 mass is limited to 512. Actual native type IDs, geometry, dimensions and a conservative rules fingerprint are part of the compatibility descriptor.

The network has one 64-unit feature transformer with clipped ReLU and two output heads selected by side to move. Each head predicts **white-perspective engine centipawns**. The runtime converts this to side-to-move engine units. These centipawns are the teacher's scale for that game; they do not compare playing strength between different games.

The accumulator belongs to each `Position`. Primitive board and pocket updates add or subtract feature columns on make/unmake. The first probe or a model-generation change refreshes it from the position. A null move changes the output head without changing board features. Both scalar and AVX2 accumulator implementations are provided; other supported CPUs use the scalar path.

### Feature limitations

This deliberately small first feature set does **not** encode castling rights, en-passant rights, move counters, repetition history, checking/chasing history or promoted-origin lineage. For example, native Shogi gold and promoted minors can share the same native movement type and thus the same board feature. The rule engine still retains their real identities, legal moves and capture/drop behavior, but this network cannot distinguish all of those states from its input alone.

There are no king-relative feature buckets, SIMD inference benchmark claims, Android-specific vector kernels or trained variants outside the five named targets. These limitations and the small shallow-teacher datasets are reasons to keep the backend experimental and disabled by default.

## Dataset integrity and labels

Generation uses `Classical`, `Use NNUE=false`, `StocklessHybrid=true`, one thread, Hash 16 MiB and MultiPV 1. Seeded random legal opening plies diversify starts. Thereafter, the classical teacher chooses its best move at the requested depth. Each exported position is reconstructed by replaying its full game history, preserving native repetition adjudication.

The manifest records variant/rules identity, the teacher executable SHA-256, observed engine identity, matched build commit/dirty state when available, generator revision, limits, seed and file checksums. Every position and game also carries the variant, rules fingerprint and teacher executable identity. A local old executable is not mislabeled with the generator checkout's newer commit.

`positions.jsonl` records FEN, sparse features, side to move, raw teacher score/unit/depth/nodes, white-perspective label, game/ply identity and outcome metadata. `games.jsonl` records the full move history and termination details. The exporter uses native immediate results, checkmate/stalemate and claimable optional results. The configured claim policy is `claim_all_native_optional`.

The last exact main-PV score at the highest completed depth is retained; bound-only results are not treated as exact. Scores are converted from side-to-move to white perspective. Mate scores receive a signed 10,000-cp target, and raw cp labels are capped at ±10,000. Training further clips targets to ±2,000 and reports how many labels were clipped.

**A game stopped at the ply limit has `result_white: null`, `truncated: true`. It is not labeled a draw.** The first trainer learns teacher scores only; game outcomes are recorded for audit and future work, not included in its loss. No blanket insufficient-material draw is invented for regional variants.

The loader validates file hashes and sizes, row/game provenance, label sign and unit, finite integer fields, move/ply ranges, outcome consistency and sparse-feature limits. The tests include semantically corrupted datasets whose outer file checksums were deliberately recomputed.

## Training, holdout and quantization gates

Whole games are partitioned into roughly two-thirds training, one-sixth validation and one-sixth test, with at least six nonempty games. Identical feature-plus-side states are removed across partitions, with training taking priority. Each surviving partition needs at least eight positions. A small manual experiment may fail this condition; it must generate more usable data rather than reuse test samples for training.

Adam updates the independent network parameters on training rows. Label centering uses training data only. The lowest validation MSE among trained epochs selects the saved checkpoint; test labels are evaluated only after that choice is frozen. The experiment records whether validation improved over initialization and does not turn a failed improvement into a strength claim. `split.json` is checksum-bound to the network manifest and checked again during native validation.

The optimizer's loss is MSE on clipped white centipawns divided by `600²`. Reports include optimizer steps, selected epoch, initial/trained weight hashes, per-epoch train/validation errors, heldout teacher-approximation errors, a train-mean baseline and Python/NumPy/platform metadata. The saved `.npz` contains selected float weights for inspection; it is not an optimizer-resume checkpoint.

After fixed-point export, both validation and test sets must satisfy these predeclared float-to-integer error limits:

| Quantization statistic | Maximum |
| --- | ---: |
| Mean absolute error | 3 cp |
| 99th percentile absolute error | 20 cp |
| Maximum absolute error | 40 cp |

The native gate replays up to 64 frozen test positions, checks exact feature and integer-score agreement with an independent NumPy int64 calculation, checks accumulator refresh equality and runs four legal depth-3 searches with the model actually active. Loader regression fixtures additionally exercise malformed/wrong-variant networks, large arithmetic values, model replacement, multithreaded search and make/unmake transitions.

## Binary format and compatibility

The file is little-endian. Its 160-byte header is followed by feature-major weights and contains:

| Offset | Field |
| ---: | --- |
| 0–7 | Magic `SVNNUE1` plus NUL |
| 8–47 | Ten uint32 fields: format version, feature version, hidden dimension, input dimension, width, height, type count, `QA`, `QB`, payload bytes |
| 48–79 | Zero-padded ASCII native variant ID |
| 80–111 | Raw rules SHA-256 |
| 112–143 | Payload SHA-256 |
| 144–159 | Reserved zero bytes |

Payload order is `W1: int16[input_dim,64]`, `b1: int32[64]`, `W2: int16[2,64]`, `b2: int32[2]`. Scales are `QA=256`, `QB=64`; transformer weights/bias use `QA`, output weights `QB` and output biases `QA*QB`. Activations clamp to `[0,256]`. The output sum uses int64 and divides by 16384 with half-away-from-zero rounding, then clamps to ±10,000 cp.

The loader validates all dimensions, reserved bytes, variant identity, file length, numerical bounds and checksums before changing the active model. The file limit is 4 MiB. Model manifests additionally bind the dataset, split, metrics, float checkpoint and quantized file.

The materializer hashes pinned upstream identities and normalized generated C++ source before adding dated notices, excluding the build-identity string and the fingerprint header itself. This avoids Windows/Linux line-ending differences and docs-only commit churn while conservatively invalidating models when engine code changes. It is a compatibility fingerprint, not a claim that every contributing source file changes game rules. A rules/architecture mismatch requires retraining or a separately reviewed migration; renaming a foreign `.nnue` file is insufficient.

## Files and license

| Output | Purpose |
| --- | --- |
| `data/manifest.json`, `positions.jsonl`, `games.jsonl` | Provenance, native features, teacher labels and complete game histories |
| `model/<variant>.svnnue` | Quantized native model |
| `model/checkpoint.npz` | Selected floating-point parameters |
| `model/metrics.json`, `split.json`, `network-manifest.json` | Measurements, heldout partition and checksum/compatibility records |
| `runtime-validation.json`, `nnue-tests.json` | Actual engine validation evidence |
| `engine/*.zip` | Matching Linux binary, corresponding C++ source, license notices and package manifest |
| `training-tools/` | The experiment's generator, trainer, exporter, validator, pinned dependency and workflow source |

New training/runtime code and generated model manifests identify `AGPL-3.0-only`; inherited engine sources retain their original notices. No third-party trained network or external game collection is required for this pilot. See [NOTICE.md](../NOTICE.md) for component provenance.

Remaining work includes richer rule-aware features, substantially larger/diverse data, deeper teachers, improved network architectures, speed measurements, regional-rule certification and controlled matches against the classical baseline. None is inferred from a successful approximation test.
