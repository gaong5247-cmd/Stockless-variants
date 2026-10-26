# Upstream architecture and licensing audit

Audit date: 2026-10-26 (Asia/Seoul). The original Stockless repository is read-only.
Exact source pins and license roles are recorded in [NOTICE](../NOTICE.md).

## Actual source/build comparison

| Area | Stockless v0.9.0 source | Fairy-Stockfish pinned source | Initial 0.1.0 integration decision |
| --- | --- | --- | --- |
| Source structure | Pinned Stockfish and Reckless submodules, three Stockless headers, `tools/materialize.py` overlay | Standalone C++ variant engine with bindings | Materialize the pinned FSF tree and apply a reviewable Stockless policy overlay |
| Rules/position | Orthodox Stockfish geometry and move representation | `Variant`, `VariantMap`, `PieceMap`, `position.*`, `movegen.cpp`, Betza parser | Reuse FSF native rules and large-board representation |
| Search | Modern Stockfish search with Stockless hybrid hooks | Older Stockfish-family search with important variant-specific guards | Retain the proven variant guards and port actual Stockless control signals |
| Time, TT, threads | Modern Stockfish implementations | FSF-native implementations | Retain FSF runtime; do not describe this as a complete current-Stockfish runtime transplant |
| NNUE | `FullThreats`, `PP_3Wide`, `HalfKAv2_hm`, modern network layers | `HalfKAv2Variants`, 512 transformed features, 16/32/1 layers | Classical evaluation; no incompatible network reuse |
| Build | Exact pin validation, source-copy/anchor patches, GNU Make/PGO | GNU Make, C++17, largeboards/all flags, Android compiler support | Reproducible non-PGO initial build; optimize only after correctness |
| Licensing | AGPLv3 integration, GPLv3 Stockfish, AGPLv3 Reckless reference | GPLv3-or-later source headers | Preserve component licenses and fulfill combined-work obligations |

## Stockless is an overlay, not a second search tree

`tools/materialize.py` verifies Stockfish pin
`0a215d6c9e48856ef630013b8ab8312941a59057` and Reckless pin
`7300f044812d80397960e3a27a4f085e9487419a`, copies the Stockfish tree, copies
`src/stockless/` to `src/stockless_core/`, then applies exact-anchor substitutions.
Reckless is an audited research reference, not a linked Rust engine or a second
NNUE whose score is averaged with Stockfish.

The reusable policy functions are `prepare_search_policy`,
`make_prepared_move_policy`, and `make_search_policy`. They consume bounded attack,
king-danger, and tactical-volatility signals. Normal operation buys back LMR
depth; optional desktop overdrive also changes pruning and re-search allowances.
Mutable depth/window/TT-PV values must remain per-move inputs. The FSF search
lacks the newer correction-history and next-ply cutoff-count inputs; neutral
inputs are explicit adaptations, not equivalent implementations.

### Reduction-unit hazard

Stockless adjustments use 1/1024-ply units. FSF `Depth` reductions are integral.
Adding a -640 policy adjustment directly to FSF depth is invalid; dividing it
first by 1024 silently disables ordinary adjustments. The integration must
combine the units deliberately and test both a disabled baseline and an enabled
behavioral difference. No strength equivalence follows from this mechanical port.

## Variant-aware threat extraction

FSF `Position::attacks_from(color, type, square)` already includes board masks,
custom movement, Xiangqi river behavior, and Janggi palace/cannon restrictions.
Use actual occupied squares rather than orthodox hard-coded piece lists.
`pinners(WHITE)` denotes white pinning pieces, matching the corrected Stockless
ownership semantics. Never unconditionally access `square<KING>()`: it asserts
exactly one king; horde, antichess, and custom rules can violate that assumption.

FSF already alters pruning for compulsory captures, atomic positions, drops,
check-counting, and walling. These guards are retained. Full multi-board state,
four-player search, hidden-information observations, and spell counters require
separate work; inherited variant names are not proof of these architectures.

## Geometry and configuration

`largeboards=yes` provides a 12x10 / 120-square envelope with 128-bit bitboards.
`all=yes` raises the move capacity to 8192. This is enough for major regional
variants but is not a general 14x14 four-player engine.

FSF configurations support inheritance and many Betza atoms/modifiers. Its
original `check` command often only prints diagnostics and still exits zero;
ordinary loading can silently ignore unknown keys. Stockless Variants performs
a complete checked pass before registering any definition, reports invalid
syntax/options/unsupported Betza, and makes failed command-line checks nonzero.
No partial registry update is permitted for an invalid input file. This does not
make every imaginable Betza extension implementable.

## Evaluation and protocol boundaries

The `0.1.0-dev` baseline shipped no neural weights and rejected neural activation;
its `Classical` and `Auto` selections both used classical evaluation. Those
historical results and the upstream architecture comparison above are unchanged.
The existing FSF NNUE architecture remains in the source, but its pretrained
format is not the new project's training or loading contract.

### Independent SVNNUE-v1 addition in 0.2.0-dev

`src/stockless_variants/nnue.*` adds an independent compact integer network with
64 hidden units, separate side-to-move output heads, and native board/pocket
features. Only `chess`, `shogi`, `janggi`, `xiangqi` and `makruk` are training
targets. Models are separate per variant, and the binary header binds their
architecture, geometry, source-derived rule/feature digest and payload checksum.
Neither a matching filename nor a matching input dimension is sufficient.

`EvalBackend=Classical` remains the default. `EvalFile` accepts a compatible
`.svnnue` model; explicit `StocklessNNUE` activation or `Auto` selects it when
applicable. Raw `Use NNUE=true` and Stockfish/FSF `.nnue` files remain unsupported.
Model loading and backend changes join search and invalidate affected caches.

`training_export.h` reads features directly from the native `Position`.
`tools/nnue/` supplies seeded classical-teacher data generation, CPU NumPy
training, whole-game holdouts, quantization and native model-validation tools.
Its compact features omit promoted-piece lineage, castling/en-passant rights
and history/counters; the full native rule state still controls legal moves and
results. The network is an experimental evaluator, with no implied inheritance
of Stockless's chess strength. See [NNUE.md](NNUE.md) for the current format and
experiment evidence. This addition does not retroactively change baseline
test counts. The [first successful remote experiment](NNUE_RESULTS.md) is
recorded with its own code, data, model and validation identities.

UCI/UCCI/USI commands that mutate a position, rule table, hash, or thread setting
stop and join an active search first. Full FEN/SFEN plus move sequences are
validated on a temporary position before replacing the live one. The native
XBoard state machine is retained, with additional option-mutation and setboard
checks. GUI-specific partner behavior and all malformed-input combinations still
require broader protocol validation; this milestone does not claim formal
compliance certification.

## License evidence

The exact pinned files were inspected: Stockless `LICENSE` and `NOTICE.md`,
FSF `Copying.txt` and source headers, Stockfish `Copying.txt`, and Reckless
`LICENSE`. They contain AGPLv3/GPLv3 as identified in NOTICE. GPLv3 section 13
and AGPLv3 section 13 allow combination while preserving the governing terms of
each component. Exact-build source availability and applicable network source
offers must accompany distribution/use as specified by those licenses.

Primary texts: [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html),
[AGPLv3](https://www.gnu.org/licenses/agpl-3.0.html). The bundled license texts
are the authoritative terms; this audit is an engineering record of what was
inspected and included.

## Reproduced runtime defects corrected during integration

The integration tests exposed upstream state/hash defects in Seirawan gating
and free drops. Separate checked patches repair reserve and material hashing,
include gate rights in position keys, and avoid a free-drop `count - 1` table
read when no reserve piece is consumed. The rule regression suite exercises
make/unmake, reconstruction, and distinct-gate positions.

The same negative-index pattern was independently reproduced in normal
`bughouse` and `supply` searches from their starting positions at depth 3,
without any partner connection. Their internal search generates virtual
checking drops that borrow reserve pieces. A narrowly instrumented
`position.cpp` build with UBSan reported an index of -1 in the 120-element
pocket-key row before the fix. `tools/patch_pocket_keys.py` replaces every read
with a signed-count accessor: valid nonnegative counts keep their original
table keys; other counts receive a deterministic mixed key without indexing
outside the table. The same two searches completed without that diagnostic
after the fix. Additional direct rule tests cover knight debt -1, pawn debt
-2, distinct reserve debts on identical boards, and complete reversal.
This repair does not supply a full multi-board coordinator or certify all
partner-protocol behavior.

Variant switching also needed cache isolation. Two custom definitions with the
same FEN and piece IDs but knight versus rook movement reproduced a wrong
cached result when switched without `ucinewgame`. `on_variant_set` now joins
search, reinitializes the rules, clears the TT/search histories, and explicitly
clears each thread's pawn and material tables. `Search::clear()` alone does not
clear those evaluation tables. A protocol regression compares the switched
search against a fresh engine using the second definition.

## Uncompleted work

- A full port of current Stockfish search/runtime/time-management internals.
- Joint multi-board, multiplayer, incomplete-information, and spell-game engines.
- Large, representative training corpora; richer variant features; tournament-
  qualified neural models and measured strength beyond the five-target CPU pilot.
- Exhaustive official tournament-rule equivalence, long matches, and strength
  measurement for every inherited or configurable variant.
- GUI assets and game interfaces beyond the engine protocol boundary.
