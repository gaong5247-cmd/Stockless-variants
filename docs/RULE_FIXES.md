# Correctness repairs to the pinned rules core

These changes are a checked overlay in `tools/patch_rules.py`; the upstream
submodule remains untouched. They were discovered through actual incremental
state/FEN reconstruction tests, not inferred from a failing search result.

## Free-drop pocket hashing

Ataxx permits free placement without consuming a reserve piece. The inherited
`remove_from_hand` is a no-op in that mode, but several callers still XORed a
pocket key at `count - 1`. When the reserve count is zero, that accesses a negative
array index and corrupts or miscomputes the key.

The overlay guards reserve-key updates for free drops in actual move making,
captures/explosion bycatch and `Position::key_after`'s speculative prefetch path.
The board contribution remains intact. Tests compare incremental keys against
positions reconstructed from FEN, and compare the speculative key against a
real simple Ataxx clone with no neighboring conversion. `key_after` is not
claimed to be exact for every special move; its upstream purpose is prefetching.

## Seirawan reserve and material keys

Gating a piece out of reserve changed the board but did not update the reserve
Zobrist key. The material key used the already-incremented piece count instead
of its previous index. The overlay adds the old/new reserve-key XOR, uses the
correct material index and handles a hypothetical gated pawn's pawn key rather
than treating it as non-pawn material.

The regression reproduces Seirawan `b1a3h` and covers both Seirawan and Seirawan
Crazyhouse (`shouse`) through make/unmake and FEN-rebuild equality.

## Virtual partner-board reserve counts

An undefined-behavior-sanitized reproduction of unpatched default Bughouse and
Supply searches at depth 3 reached a pocket-key read at index -1. The inherited
interior search can borrow a future partner piece even without an external
partner connection; its reserve counts may intentionally become negative.

`tools/patch_pocket_keys.py` routes all pocket-key reads through a safe accessor.
Ordinary nonnegative table entries are unchanged, preserving normal position
keys. Counts outside the table are represented by a deterministic 64-bit mix of
the piece and signed-count bits. Distinct internal debts retain distinct key
contributions without array indexing or shared mutable state. This is a memory
safety correction for inherited local-board search, not implementation of a
complete two-board game coordinator.

## Gating eligibility in position identity

Identical boards can have different legal gate locations. Those rights must
therefore distinguish transposition and repetition keys. The inherited core
omitted them from its key. New per-color/per-square keys are included when the
position is initialized and updated using the symmetric difference between old
and new gate masks. Undo naturally restores the previous `StateInfo`.

After a Seirawan player's non-replenishing reserve is exhausted, the core clears
gating eligibility. FEN castling notation could otherwise restore latent gates
on a round trip. The loader now normalizes exhausted, non-replenishing Seirawan
gates to zero while preserving independent castling rights.

Tests distinguish otherwise identical boards with different active gate rights
and exercise longer sequences through reserve exhaustion. Existing random keys
are initialized before the new keys, preserving unrelated orthodox keys.

## Input validation repairs

The separate protocol overlay validates a whole variant configuration before
loading it, a whole position/move sequence before committing it, and rejects
unknown/malformed options explicitly. It also checks oversized FEN empty-square
runs and the final rank's width before unsafe board access.

## Variant-switch cache isolation

Custom variants can use the same piece IDs and identical FEN while giving a
piece different movement. Warming one variant and switching to another without
`ucinewgame` reproduced a different, incorrect search result compared with a
fresh engine. Position keys alone do not encode an arbitrary rule definition.

Changing the variant now stops/joins search, initializes its movement/PSQT
tables, clears TT/search history and explicitly clears each thread's pawn and
material evaluation caches. `Search::clear()` alone did not clear those two
caches. A regression warms a knight-like custom piece variant, switches to the
same FEN with a rook-like piece, and compares the result with a fresh engine.

These corrections do not establish that the entire inherited rule system is
free of defects. Remaining rule-convention and architectural boundaries are in
[LIMITATIONS.md](LIMITATIONS.md).
