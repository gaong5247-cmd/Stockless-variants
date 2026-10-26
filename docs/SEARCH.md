# Actual search integration

## Reused code

`tools/materialize.py` copies `src/stockless/config.h` and `src/stockless/search_policy.h` byte-for-byte from the pinned Stockless commit. These contain the real v0.9.0 atomic configuration, per-node preparation and per-move policy formulas. The source manifest records SHA-256 for every generated source file.

The FSF search remains responsible for iterative deepening, aspiration windows, PVS/alpha-beta, quiescence, TT, move ordering, history, native selective extensions and time management. This is a policy port into a variant-aware engine, not a claim that all Stockfish 19 internals have been transplanted.

## Threat adapter

The original Stockless attack helper is tied to Stockfish's orthodox piece/attack APIs. The new adapter iterates actual non-king pieces and calls `Position::attacks_from(color, type, square)`, retaining FSF's palace, cannon, blocker, pawn-direction and board-mask semantics. It does not assume 64 squares or require a king in variants that do not have one.

Signals retain Stockless's bounded 0–256 pressure, danger and volatility outputs. Atomic adds bounded contact hints for capturable pieces adjacent to a king; pocket variants add bounded possible checking-drop and pocket-availability hints. These are heuristics, not legal-move generators or neural outputs. For example, a possible checking-drop hint may include a nifu-constrained drop; the rules core remains responsible for legal actions.

Full signals use the inherited cost gate: normally depth ≥5 on desktop or ≥7 on mobile, with PV/important-node exceptions. No correction-history or next-ply-cutoff equivalent exists in this FSF revision; the corresponding Stockless inputs are explicitly neutral zero.

## Fixed-point adaptation

Stockless's LMR adjustment uses 1/1024-ply units; FSF's reductions are integer plies. Directly adding the number is invalid. Dividing it alone by 1024 erases all normal adjustments; nearest rounding also erases the mobile profile's entire 320-unit cap.

`adapted_reduction` therefore uses a stateless, deterministic threshold from the position Zobrist key and move. A bounded buyback of 320/1024 selects a one-ply reduction buyback on approximately that fraction of position/move selectors. It otherwise retains the native reduction. This preserves the scale across positions, including mobile, without a shared accumulator, random seed, or thread-schedule dependence. No individual adjustment buys back more than one ply. The native search-depth clamp still applies.

This quantization is new, **not search-equivalent to current Stockfish**. Boundary/distribution/determinism tests and ON/OFF searches establish implementation behavior, not improved playing strength. It requires future paired self-play/tuning before any strength claim.

## Variant policies

| Profile | Hybrid policy | Overdrive | Null-move condition added when hybrid option is on |
| --- | --- | --- | --- |
| Orthodox chess/Chess960 | Enabled | Opt-in desktop only | Native conditions preserved |
| Atomic | Enabled with blast-contact hints | Disabled | Disabled |
| Shogi/Crazyhouse and pocket-drop games | Enabled with pocket hints | Disabled | Disabled |
| Janggi | Enabled with FSF palace/cannon attacks | Disabled | Disabled (pass/score-sensitive) |
| Xiangqi | Enabled with FSF river/palace attacks | Disabled | Native conditions preserved |
| Check-counting games | Enabled | Disabled | Disabled |
| Other royal regional/fairy games | Enabled conservatively | Disabled | Native conditions preserved |
| Antichess/compulsory capture | Native objective-aware search | Disabled | Disabled |
| Duck/compound walling turns | Native compound-move search | Disabled | Disabled |
| Partner-board games | Native local-board search only | Disabled | Disabled |
| Alternative goals, absent/multiple kings | Native variant search | Disabled | Disabled |

The global option `StocklessHybrid=true` does not mean the royal-attack formula is suitable for every variant. The profile explicitly disables it where a dedicated objective/observation policy is absent. With `StocklessHybrid=false`, all Stockless search adjustments and the extra null-move condition are bypassed; protocol/rule correctness fixes still remain.

## Mutable state and threads

Settings/threat features are captured before each node's move loop. Depth, alpha-beta window and TT-PV state are passed per move rather than cached as stable node properties. The protocol layer stops and joins active search before mutating positions, options or rule configuration. Individual thread searches retain the existing FSF runtime.

Bounded two-thread regressions are not a race-detector certification or a long-match stability proof. Both remain future validation gates.
