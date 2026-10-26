# Chess.com rules audit and implementation boundaries

Reviewed on 2026-10-26 against public Chess.com help/terms pages and Fairy-Stockfish
commit `9f778da667f6e07dae1e85d3e2ea204fc6dee94d`.

This is a development audit. A matching name, source registration, or successful
UCI handshake does not prove that every Chess.com edge case has been reproduced.
The first engine uses pinned FSF rules with an adapted Stockless search policy;
it does not reproduce a private Chess.com server or claim official affiliation.

## What the inventory means

The public [Variants help collection][collection] contains both game rules and
service/presentation modes. Its article count is not a count of implemented games.
The [navigation guide][navigation] describes the live lobby as the place where
available choices are listed. This audit uses public documentation and does not
claim an exhaustive, authenticated snapshot of every current custom lobby preset.

`tools/catalog_variants.py` reads compiled registrations from `src/variant.cpp`
and INI examples from `src/variants.ini` separately. It records build guards,
source inheritance and assigned rule expressions. Optional `--engine` querying
records the binary's actual `UCI_Variant` list. INI examples require loading and
validation; the script does not approve them merely because they parse as text.

The four classifications describe implementation work:

| Class | Meaning |
|---|---|
| Native | A rule implementation exists in the pinned core. Tests and release approval are separate. |
| Adapted | Existing rule/configuration/presentation mechanisms require a specific adaptation. |
| New Rules | Additional move, state or adjudication logic is needed in a two-player architecture. |
| Special Architecture | A coordinator, multiplayer model, single-player solver or observation-only search is needed. |

All entries keep `validation_evidence` separate from `release_approved`. The
catalog never promotes a game to a release solely from source or runtime listing.
Builds with all variants and large boards remain two-color engines with board
storage up to 12 files × 10 ranks. That is not four-player support.

## Existing two-player rules and important name differences

| Requested rule | FSF candidate | Audit finding / test requirement |
|---|---|---|
| Duck | `duck` (`ALLVARS`) | A complete move combines piece movement and duck relocation. No check/checkmate; king capture wins; a player with no legal move wins. Test blocker movement, duck relocation, castling through attack, immediate king capture and final-turn state. [Rules][duck] |
| Torpedo | `torpedo` | Repeated two-square pawn moves and en passant away from the ordinary fifth rank are required. Test blocked intermediate squares, promotion and the one-turn en passant window. [Rules][torpedo] |
| Sideways Pawns | `pawnsideways` | Sideways movement is non-capturing; ordinary pawn captures remain diagonal. Test sideways non-captures separately from promotion, en passant and initial double steps. [Rules][sideways] |
| Giveaway on Chess.com | **`antichess`** | Chess.com prohibits castling. FSF's `giveaway` allows castling; `antichess` disables it. Both names must not be presented as identical rulesets. Mandatory captures, stalemate wins and promotion to a non-royal king need fixtures. [Rules][giveaway] |
| Crazyhouse | `crazyhouse` | Captures change color into the player's own pocket. A captured promoted pawn returns as a pawn; first/last-rank pawn drops are forbidden; check and mate by drop are allowed. [Rules][crazyhouse] |
| Atomic | `atomic` | Captures remove the capturing piece and adjacent non-pawns. Kings cannot capture; adjacent kings may be legal; a winning explosion cannot also destroy one's own king. [Rules][atomic] |
| Horde | `horde` | The documented starting army has 36 white pawns. Pawns may double-step from ranks one and two, including a second move from rank two. Do not replace this with an ordinary pawn “has moved” flag. [Rules][horde] |
| Chess960, 3-Check, KOTH, Racing Kings, No Castling | `fischerandom`, `3check`, `kingofthehill`, `racingkings`, `nocastle` | Implementations exist; each exact initial position, special objective and terminal response requires tests. These are identified individually in the public collection. [Collection][collection] |
| Gothic | `gothic` | Public Chess.com documentation identifies 10×8 geometry and archbishop/chancellor compound pieces. Exact castling/start-position parity still needs fixtures. [Rules][gothic] |

Duck is already represented by FSF's compound wall-placement move machinery.
It does not need to be advertised as an unimplemented four-player-style
architecture. Search must evaluate a complete two-part turn and switch side only
after that turn; treating the two actions as opposing-player plies is incorrect.
Its branching factor and policy safety still need independent search testing.

## Chaturanga: a documented mismatch

The [help article][chaturanga-help] explicitly places the white king on d1 and
the ferz on e1. The pinned `chaturanga_variant()` uses `RNBQKBNR`, placing the
white king on e1. The article does not explicitly give the black king's square.
Interpreting the same reversal for Black would give this *provisional* symmetric
placement:

```text
rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w - - 0 1
```

That black placement is an inference, not a verified live-server fixture. An
experimental config should say it follows the help-description placement rather
than claim full Chess.com compatibility. Do not silently alias to `shatranj`:
the latter has different bare-king handling. The official [terms article][chaturanga-terms]
explicitly says the Chess.com version retains check/checkmate and the stalemated
player loses. Its discussion mixes historical and current rules, so promotion,
bare-king behavior and move limits still need carefully identified game fixtures.

## Features requiring new rule code

### Capture Anything

Chess.com permits capturing one's own pieces. FSF's INI example named `capture`
only makes captures mandatory; it does **not** add friendly captures. Implementing
this game affects target generation, legal king safety, material accounting,
make/unmake, hashing and evaluation. It cannot be produced by enabling
`mustCapture`. The sparse public help does not settle every royal-piece edge case.
[Help][capture] and [official explanation][capture-terms].

### Spell Chess

The official rules give each player five Freeze spells and two Jump spells.
Freeze applies to a 3×3 area, clipped at edges, through the opponent's next turn.
Jump selects a square that may be jumped over and remains effective through the
opponent's move. Spells have a three-full-turn cooldown. Checkmate and king capture
can win; castling through or into check remains illegal. [Rules][spell]

The implementation needs action/state fields for spell use, inventories, cooldown
and duration, plus movement and royal-safety changes. Hashes and repetition must
include all those fields. Public prose alone is insufficient for ambiguous
simultaneous spell effects or precise cooldown boundaries; write explicit,
versioned fixtures before claiming parity. This is not an existing FSF preset.

## Features requiring a separate architecture

| Game | Existing contribution | Required architecture |
|---|---|---|
| Bughouse | FSF `bughouse` supports local-board pockets supplied externally (`twoBoards`). | Two boards, four clocks, partner transfers, synchronized endings and drop-dependent mate rules. Chess.com says a blockable check is not mate even when the pocket is currently empty. A local search alone is not a whole Bughouse match. [Rules][bughouse] |
| Supply Chess | FSF `supply` provides local-board Xiangqi drop constraints with `twoBoards`. | A partner-board coordinator, not an assertion that both boards already exist inside a position. [Pinned core][core] |
| Fog of War | No-check king capture is only a small rule component. | Own pieces and reachable-square observations, temporary en passant visibility, and belief-state/information-set search. Hidden state must not enter the playing policy or its TT. Castling through attack is allowed. [Rules][fog] |
| 4PC Teams | Ordinary two-player move primitives are reusable. | Four colors and clockwise turn order; teammate across the board; no teammate captures; promotion on rank 11; team victory when the enemy player's turn reaches mate. [Rules][four-player] |
| 4PC FFA | Individual movement rules are reusable. | Four-player utilities, point scoring, eliminations and inactive pieces, rank-8 promotion, plus the special resignation/timeout state. Negating one scalar score each ply is not equivalent. [Rules][four-player] |
| Solo Chess capture puzzle | Piece geometry can be reused. | One-player solver, all pieces same color, every move a capture, per-piece limit of two captures, king as final survivor when present. It is not Antichess. [Rules][solo] |
| Chaturaji | An 8×8 board fits, but the color/turn model does not. | Four players, point scoring, king-capture elimination and boat promotion. Chess.com's version uses no dice. [Rules][chaturaji] |

The 4PC article also calls its winner-takes-all rating mode “Solo.” That mode
uses FFA game rules and must be distinguished from the separate Solo Chess
capture puzzle. [4PC documentation][four-player]

## Additional official candidates

The first catalog adds these documented choices beyond the principal requested
list. They are not all new movement-rule engines:

- **Blindfold:** standard chess with hidden piece rendering and restricted visible
  move history. This belongs in a presentation layer, unlike Fog of War.
  [Documentation][blindfold]
- **Vote Chess:** standard board rules plus timed voting, voting thresholds and
  a tie-break coordinator. The engine alone does not implement this service.
  [Documentation][vote]
- **Balanced/Odds:** standard chess from material/time/position presets.
  Chess.com's automatic balancing algorithm is not public implementation code
  and is not claimed here. [Documentation][odds]
- **Chaturaji** and **4PC Solo:** separate multiplayer candidates described above.
- **Custom Variant options:** the official editor includes selectable promotion
  rank/pieces, N-check limits and stalemate behavior. The first custom loader
  supports only documented FSF configuration capabilities and valid combinations.
  [Documentation][custom]

Generalized N-check must be bounded by the actual state and FEN parser. The pinned
parser reads single-character counters even though the internal check-count
enum has additional storage. The native 3-Check and 5-Check entries do not prove
that arbitrary multi-digit N values round-trip correctly.

## Reproducing the catalog

```sh
python3 tools/catalog_variants.py
python3 tools/catalog_variants.py --engine build/stockless-variants
python3 tools/catalog_variants.py --engine build/stockless-variants \
  --load-config config/examples.ini --validation test-results.json
```

`--source`, `--classification`, `--json` and `--markdown` select input/output paths.
`--require-all-builtins` additionally fails if the queried binary omits a source
registration; use it only for a build intended to include every upstream ID.
Validation reports are attached as evidence without automatically changing
release approval. The default outputs are `docs/variants.json` and
`docs/VARIANTS.md`.

The catalog understands the `tests/run_engine_tests.py` report's `tests` list,
including perft, search, playout and differential evidence. Each observation
retains its exact `ruleset_tested`; a successful FSF `chaturanga` test does not
certify the adapted Chess.com Chaturanga rules. Failed observations are retained.
When `--engine` is supplied, reports bearing a different executable SHA-256 are
rejected. Without a runtime binary, any report remains explicitly unbound to a
currently inspected executable. Protocol queries disable implicit INI autoload
so the inventory contains only built-ins and explicitly requested configurations.

[collection]: https://support.chess.com/en/collections/6767818-variants
[navigation]: https://support.chess.com/en/articles/8583983-how-do-i-play-chess-variants
[duck]: https://support.chess.com/en/articles/8615411-what-is-duck-chess
[torpedo]: https://support.chess.com/en/articles/8705335-what-is-torpedo-chess
[sideways]: https://support.chess.com/en/articles/8708719-what-is-sideways-pawns-chess
[giveaway]: https://support.chess.com/en/articles/8659140-what-is-giveaway-chess
[crazyhouse]: https://support.chess.com/en/articles/8614995-what-is-crazyhouse
[atomic]: https://support.chess.com/en/articles/8615237-what-is-atomic-chess
[horde]: https://support.chess.com/en/articles/8708676-what-is-horde-chess
[gothic]: https://www.chess.com/terms/gothic-chess
[chaturanga-help]: https://support.chess.com/en/articles/8648829-what-is-chaturanga
[chaturanga-terms]: https://www.chess.com/terms/chaturanga-game
[capture]: https://support.chess.com/en/articles/8708660-what-is-capture-anything-chess
[capture-terms]: https://www.chess.com/terms/capture-anything-chess
[spell]: https://support.chess.com/en/articles/8588083-what-is-spell-chess
[bughouse]: https://support.chess.com/en/articles/8615092-what-is-bughouse
[fog]: https://support.chess.com/en/articles/8708650-what-is-fog-of-war-chess
[four-player]: https://support.chess.com/en/articles/8614233-4-player-chess-4pc
[solo]: https://support.chess.com/en/articles/8588122-what-is-solo-chess
[chaturaji]: https://www.chess.com/terms/chaturaji
[blindfold]: https://support.chess.com/en/articles/8618510-what-is-blindfold-chess
[vote]: https://support.chess.com/en/articles/8614177-how-do-i-play-vote-chess
[odds]: https://www.chess.com/terms/odds-chess
[custom]: https://support.chess.com/en/articles/8609332-how-can-i-make-my-own-chess-variant
[core]: https://github.com/fairy-stockfish/Fairy-Stockfish/blob/9f778da667f6e07dae1e85d3e2ea204fc6dee94d/src/variant.cpp
