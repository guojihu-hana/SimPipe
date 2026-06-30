from __future__ import annotations

# Pattern symbols (each character in the pattern string is one layer):
#   E = embedding, L = head
#   M = mamba, - = mlp, * = attn
MAMBA = "M"
ATTN = "*"
MLP = "-"
EMBEDDING = "E"
HEAD = "L"

STACK_LAYER_SYMBOLS = frozenset({MAMBA, ATTN, MLP})
LAYER_SYMBOLS = STACK_LAYER_SYMBOLS | {EMBEDDING, HEAD}
PATTERN_CHARS = frozenset({MAMBA, ATTN, MLP, EMBEDDING, HEAD})

LAYER_KIND: dict[str, str] = {
    EMBEDDING: "embedding",
    MAMBA: "mamba",
    ATTN: "attn",
    MLP: "mlp",
    HEAD: "head",
}

CHAR_TO_SYMBOL = {
    "M": MAMBA,
    "-": MLP,
    "*": ATTN,
    "E": EMBEDDING,
    "L": HEAD,
}

SYMBOL_TO_CHAR = {value: key for key, value in CHAR_TO_SYMBOL.items()}

# forward_ms / backward_ms / weight_ms may use either the single-letter symbol or the kind name.
TIMING_KEY_ALIASES: dict[str, str] = {
    "embedding": EMBEDDING,
    "mamba": MAMBA,
    "attn": ATTN,
    "attention": ATTN,
    "mlp": MLP,
    "head": HEAD,
}


def layer_kind(symbol: str) -> str:
    if symbol not in LAYER_KIND:
        raise KeyError(f"Unknown layer symbol {symbol!r}")
    return LAYER_KIND[symbol]


def normalize_timing_value(value: float | int) -> float:
    number = float(value)
    if number.is_integer():
        return number
    return float(round(number * 100))


def normalize_timing_keys(table: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in table.items():
        symbol = TIMING_KEY_ALIASES.get(key, key)
        if symbol not in LAYER_SYMBOLS:
            raise KeyError(f"Unknown timing key {key!r}")
        normalized[symbol] = normalize_timing_value(value)
    return normalized


def stack_layer_symbols(pattern: str) -> list[str]:
    """Parse pattern string: each M / - / * character is one layer."""
    symbols: list[str] = []
    for char in pattern:
        if char not in CHAR_TO_SYMBOL:
            raise ValueError(f"Unknown pattern character {char!r} in {pattern!r}")
        symbol = CHAR_TO_SYMBOL[char]
        if symbol in STACK_LAYER_SYMBOLS:
            symbols.append(symbol)
    return symbols


def tokenize_pattern(pattern: str) -> list[str]:
    """Parse all layer symbols from the pattern string."""
    symbols: list[str] = []
    for char in pattern:
        if char not in CHAR_TO_SYMBOL:
            raise ValueError(f"Unknown pattern character {char!r} in {pattern!r}")
        symbols.append(CHAR_TO_SYMBOL[char])
    return symbols


def normalize_pattern_tokens(tokens: list[str]) -> list[str]:
    """Ensure E is first and L is last when omitted from the pattern."""
    normalized = list(tokens)
    if EMBEDDING not in normalized:
        normalized.insert(0, EMBEDDING)
    if HEAD not in normalized:
        normalized.append(HEAD)
    return normalized


def encode_layer_pattern(tokens: list[str]) -> str:
    """Encode layer symbols into a pattern string (one character per layer)."""
    return "".join(SYMBOL_TO_CHAR[token] for token in tokens if token in SYMBOL_TO_CHAR)


def format_stage_layer_pattern(tokens: list[str]) -> str:
    """Format stage layers like the model pattern string (e.g. E-M-*-L)."""
    return encode_layer_pattern(tokens)


def stage_layer_pattern_strings(
    layer_counts: list[int],
    stack_symbols: list[str],
) -> list[str]:
    """Map partition layer counts to per-stage layer type strings."""
    patterns: list[str] = []
    cursor = 0
    for sid, count in enumerate(layer_counts):
        tokens: list[str] = []
        if sid == 0:
            tokens.append(EMBEDDING)
        for _ in range(count):
            if cursor >= len(stack_symbols):
                break
            tokens.append(stack_symbols[cursor])
            cursor += 1
        if sid == len(layer_counts) - 1:
            tokens.append(HEAD)
        patterns.append(format_stage_layer_pattern(tokens))
    return patterns


def pattern_tokens(pattern: str) -> list[str]:
    stack = stack_layer_symbols(pattern)
    normalized = list(stack)
    if EMBEDDING not in normalized and "E" not in pattern:
        normalized.insert(0, EMBEDDING)
    if HEAD not in normalized and "L" not in pattern:
        normalized.append(HEAD)
    return normalized


def layer_count(pattern: str) -> int:
    """Total layer count from pattern, including E and L when auto-added."""
    return len(pattern_tokens(pattern))


def stack_layer_count(pattern: str) -> int:
    """Count of mamba / attn / mlp layers (excluding E and L)."""
    return len(stack_layer_symbols(pattern))


def layer_kinds(pattern: str) -> list[str]:
    return [layer_kind(token) for token in pattern_tokens(pattern)]
