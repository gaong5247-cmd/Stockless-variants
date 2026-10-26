# Explicit implementation limits

This is `0.1.0-dev`, not a stable universally validated release.

## Rules and architecture

- The inherited geometry supports a **12×10 envelope, at most two colors/players**. It is not a general 14×14 four-player state representation.
- FSF `bughouse`, `supply` and related registrations implement local board/pocket/partner behavior. A complete synchronized pair of boards, shared clocks, partner transfers and joint-team search are not implemented here.
- Fog of War has no observation/belief-state interface here. Passing a complete board to ordinary search must not be presented as fair hidden-information play.
- Four-player Teams/FFA/Chaturaji require new turn, elimination, utility and multi-player search models.
- Spell Chess and Capture Anything are not implemented. Solo Chess needs a dedicated single-agent state and capture-counter solver.
- Blindfold is a presentation mode; Vote Chess is a voting/coordinator layer; neither is a distinct piece-movement implementation.
- Duck has a native compound-move representation. Protocol moves include both piece movement and duck placement, e.g. `e2e4,e4d5`; clients must implement that dialect. Its Stockless royal-threat policy is disabled pending an appropriate adapter.
- Chess.com rule equivalence is tracked separately from an FSF name match. In particular, Chess.com Giveaway maps to FSF `antichess`; FSF `giveaway` allows castling. See the official-source comparison document.

## Regional rule caveats

- Shogi's inherited uchifuzume handling adjudicates a pawn-drop mate as a loss for the dropper; that move is not uniformly absent from the core's `MoveList<LEGAL>`. The regression suite preserves and discloses this upstream behavior. Strict illegal-move filtering for all Shogi tournament conventions remains incomplete.
- Shogi entering-king declarations/impasse and regional tournament conventions need explicit selected rulesets and dedicated tests; a generic `shogi` registration is not certification of every association's rules.
- Xiangqi inherits FSF's AXF chasing model for its standard variant. That is not a claim to match every WXF/CCA tournament convention. Derived Xiangqi variants can have different chasing settings.
- Janggi modes inherit specific pass, bikjang, repetition and material-counting behavior. GUI arrangement selection and independent federation-rule certification are not implemented. A custom initial FEN can represent a selected horse/elephant arrangement.
- Present check-counting FEN syntax uses single-digit counters. The 7-check example is supported within that format; unbounded arbitrary N is not advertised.
- Native/Adapted/New Rules/Special Architecture are implementation classifications, not test-result or release-approval labels.

## Evaluation and strength

- Only classical evaluation is enabled. No Stockless/Stockfish neural file is repurposed for Shogi, Janggi or Xiangqi.
- No variant-specific training pipeline, teacher dataset, quantization validation, new accumulator or compatible network manifest has been delivered.
- Original Stockless's measured chess results do not transfer to this engine, this evaluator or any regional variant.
- No Elo rating or improvement is claimed. Bounded test playouts are correctness exercises, not a strength match.

## Distribution and validation

- The engine has no graphical board, SVG piece catalog or new brand artwork yet.
- Android CI cross-compiles an ARM64 native executable and checks ELF architecture. Physical-device execution, host-GUI protocol compatibility and sustained thermal behavior are separate pending tests.
- Linux/Windows runtime tests and packaging status must be read from the corresponding Actions run and supplied report. A workflow definition alone is not a successful build.
- Perft agreement with the pinned FSF source is valuable regression evidence but not independent proof of all game rules.
- Long-time-control matches, statistically meaningful paired strength tests and exhaustive multi-thread race checking remain pending.
