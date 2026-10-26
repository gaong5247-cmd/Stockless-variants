# Protocols

The adapters are inherited from the pinned Fairy-Stockfish core and hardened by `tools/patch_protocol.py`. The regression suite tests live subprocess behavior rather than only searching strings in source files.

## UCI / UCI_Variant

```text
uci
setoption name UCI_Variant value crazyhouse
setoption name Hash value 32
setoption name Threads value 2
setoption name MultiPV value 3
setoption name StocklessHybrid value true
isready
position startpos moves e2e4 d7d5 e4d5 d8d5
go nodes 5000
stop
quit
```

`VariantName` aliases `UCI_Variant`; `VariantConfigPath` aliases `VariantPath`. `Skill Level` remains the upstream spelling. Supported limits include depth, nodes, movetime and normal side-specific clocks/increments. Ponder and ponderhit retain their upstream semantics.

UCI_GENERAL uses ordinary one-based ranks, including `a10` on a ten-rank board. Drops use `P@e4`. Duck uses a comma-separated compound move: piece action, then a segment encoding the duck placement. Always obtain the legal notation from the engine rather than splitting a turn into two independent calls.

## USI / Shogi

```text
usi
setoption name USI_Hash value 32
isready
usinewgame
position startpos
go byoyomi 1000
quit
```

USI defaults to `shogi`. SFEN is accepted by `position sfen ...`; the core handles USI numeric-file/alphabetic-rank coordinates, `+` promotion and `P*5e` drops. `USI_Variant` selects a different registered Shogi variant.

## UCCI / Xiangqi and Janggi

```text
ucci
setoption hashsize 32
isready
position startpos
go depth 4
quit
```

UCCI defaults to `xiangqi` and uses zero-based ranks on ten-rank boards (`a0`–`i9`). UCCI option syntax omits UCI's `name`/`value` words. For a host that supports Janggi through this dialect, use `setoption UCI_Variant janggi`. The actual GUI's chosen Janggi rules and pass notation still require integration testing.

## CECP/XBoard

```text
xboard
protover 2
new
variant atomic
force
setboard rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
go
?
quit
```

Feature negotiation, variant lists, engine moves, ping/pong and the existing partner mechanisms are preserved. Partner mechanisms alone are not a full Bughouse coordinator.

## Evaluation and errors

`EvalBackend` offers `Classical` and `Auto`; both currently resolve to the classical evaluator. `Use NNUE=true` produces an explicit error and remains false. There is no misleading silently accepted incompatible network backend.

Invalid option values, unknown variants, invalid FEN/SFEN/moves and invalid INI definitions are reported. A malformed position command must not leave a partially applied move list. Mutating UCI commands stop and join an active search before changing shared state.

Additional inherited diagnostic commands include `d`, `eval`, `go perft N`, `compiler` and `check FILE.ini`. Run mutations/diagnostics between searches in normal GUI operation.
