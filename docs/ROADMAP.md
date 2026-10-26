# Development gates

Each phase is accepted using its own build/test evidence. A checked-in workflow or an inherited registration is not sufficient to mark a whole phase complete.

| Phase | First milestone contribution | Remaining gate |
| --- | --- | --- |
| 1. Source/license audit | Exact pins, source comparison, preserved licenses and notices | Re-audit every upstream change |
| 2. Rule core + Stockless | Actual prepared policy port, geometry-aware hints, reproducible build | Additional objective-specific policies and performance tuning |
| 3. Chess and FSF variants | Native rule runtime, perft/make-unmake/protocol regressions | Wider positions, exhaustive correctness gates, stable release qualification |
| 4. Regional games | Executable Shogi/Janggi/Xiangqi/Makruk and directed regressions | Tournament-rule certification, Shogi strict uchifuzume handling/entering-king rules, GUI conventions |
| 5. Chess.com variants | Official-rule mapping; native Duck/Torpedo/Sideways etc. classified | Capture Anything, Spell Chess, full parity fixtures |
| 6. Special architecture | Explicit unsupported classification and boundaries | Multi-board coordinator, four-player state/search, fair Fog observation model, Solo solver |
| 7. NNUE and strength | Classical fallback and incompatible-network rejection | Datasets, rule/network manifests, trainers, quantization and measured strength |
| 8. Logo/GUI | Engine/frontend separation maintained | Original SVG brand and piece assets, board GUI |
| 9. Large-scale validation | Reproducible regression harness and bounded play | Long matches, race detection, paired statistics and opponent/network/hardware manifests |
| 10. Multi-platform release | Linux/Windows/Android CI and corresponding-source package tools | Successful platform evidence, Android device tests, stable-release approvals |

## Future data/network manifest

A future training example must include variant ID, semantic rules version/hash, complete position and side to move, game result/adjudication reason, teacher identity/commit/network, search limits, and generation seed. A network manifest must bind those semantics to feature encoding, dimensions, quantization, architecture version, file hash and license. Neither schema placeholders nor a file extension are grounds to activate a model in the current executable.
