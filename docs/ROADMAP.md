# Development gates

Each phase is accepted using its own build/test evidence. A checked-in workflow or an inherited registration is not sufficient to mark a whole phase complete.

| Phase | Implemented contribution | Remaining gate |
| --- | --- | --- |
| 1. Source/license audit | Exact pins, source comparison, preserved licenses and notices | Re-audit every upstream change |
| 2. Rule core + Stockless | Actual prepared policy port, geometry-aware hints, reproducible build | Additional objective-specific policies and performance tuning |
| 3. Chess and FSF variants | Native rule runtime, perft/make-unmake/protocol regressions | Wider positions, exhaustive correctness gates, stable release qualification |
| 4. Regional games | Executable Shogi/Janggi/Xiangqi/Makruk and directed regressions | Tournament-rule certification, Shogi strict uchifuzume handling/entering-king rules, GUI conventions |
| 5. Chess.com variants | Official-rule mapping; native Duck/Torpedo/Sideways etc. classified | Capture Anything, Spell Chess, full parity fixtures |
| 6. Special architecture | Explicit unsupported classification and boundaries | Multi-board coordinator, four-player state/search, fair Fog observation model, Solo solver |
| 7. NNUE and strength | Independent SVNNUE-v1 for five native targets; native self-play export, CPU NumPy trainer, whole-game holdouts, quantization, incremental accumulation and rule/network manifests | Recorded experiment gates for exact builds/models, broader data/features, additional variants and measured strength |
| 8. Logo/GUI | Engine/frontend separation maintained | Original SVG brand and piece assets, board GUI |
| 9. Large-scale validation | Reproducible regression harness and bounded play | Long matches, race detection, paired statistics and opponent/network/hardware manifests |
| 10. Multi-platform release | Historical 0.1.0-dev five-target Linux/Windows/Android builds and corresponding-source packages | Exact 0.2.0-dev build/model evidence, Android device tests and stable-release approvals |

## Current data and network gate

The first neural targets are native `chess`, `shogi`, `janggi`, `xiangqi` and `makruk`, each trained separately. Dataset manifests identify the teacher binary/build, rules, native feature layout, search limits and seed. Every position/game carries variant, rules and teacher identity; complete move histories preserve adjudication, and unfinished games retain null outcomes. Whole-game splits and cross-split exact-input deduplication precede training.

The `.svnnue` binary header enforces the architecture, variant, geometry, rule/feature-source fingerprint, quantization and payload checksum. Its accompanying model manifest records data provenance, file hashes, license and metrics. Native replay/integer/accumulator/search checks must pass for the actual trained file. Classical evaluation stays the default.

The compact v1 features omit promoted origin, castling/en-passant rights and history/counters. Larger datasets, richer features and paired strength experiments remain separate gates. The [CPU training workflow](../.github/workflows/train-nnue.yml) is an executable experiment definition; a completed run requires its own results. [NNUE.md](NNUE.md) documents usage and recorded evidence. No training loss or quantization metric is an Elo measurement.
