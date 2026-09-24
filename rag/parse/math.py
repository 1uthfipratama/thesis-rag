"""Math-density scoring: equations are excluded from the index, prose around them kept."""

import re
import unicodedata

from rag.parse.extract import RawBlock

MATH_DENSITY_THRESHOLD = 0.25  # plan default; checked on p05/p06 via inspect_parse
EQ_NUMBER = re.compile(r"\((\d{1,3}[a-z]?)\)\s*$")

_MATH_OPS = set("=+−-×÷±∓∑∏∫∂∇√∞≈≠≤≥∈∉⊂⊆∪∩∀∃→←↔⇒⇔′″‖·∘⊗⊕^_|<>*/")


def _is_math_char(ch: str) -> bool:
    cp = ord(ch)
    if 0x1D400 <= cp <= 0x1D7FF:  # mathematical alphanumeric symbols (italic x, bold A...)
        return True
    if 0x0370 <= cp <= 0x03FF:  # Greek
        return True
    if ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎":
        return True
    if ch in _MATH_OPS:
        return True
    return unicodedata.category(ch) == "Sm"


def math_density(raw: str) -> float:
    alpha = sum(ch.isalpha() and not _is_math_char(ch) for ch in raw)
    math = sum(_is_math_char(ch) for ch in raw)
    return math / max(1, alpha)


def _real_words(text: str) -> int:
    """Words of >= 3 letters: 'the', 'stock'. Variables like 'dt', 'Sn', 'x' don't count."""
    return len(re.findall(r"\b[A-Za-z]{3,}\b", text))


def is_equation(b: RawBlock) -> bool:
    raw, text = b.raw, b.text
    if math_density(raw) > MATH_DENSITY_THRESHOLD:
        return True
    # Display equations extract as many tiny fragments ("dS(t)", "dt", "= β(t)"),
    # each too short for density to be meaningful. A fragment with almost no
    # real words and some math/digit content is equation debris.
    words = _real_words(text)
    if words <= 1 and len(text) <= 60 and re.search(r"[=+\-−()\d^_βαγδσμλθ]", text):
        return True
    if words == 0 and len(text.strip()) <= 15:
        return True  # lone "dt", "n", "S0" left over from a display equation (p05)
    # Symbol-font equations that extract as loose letters (p17:
    # "dX t t X t dt t X t dW t ..."): mostly one-letter tokens.
    tokens = text.split()
    if len(tokens) >= 6 and sum(len(tk) == 1 for tk in tokens) / len(tokens) > 0.5:
        return True
    # "... S(t), (1)" : short line ending in an equation number with few words.
    if EQ_NUMBER.search(text) and words <= 3 and len(text) < 80:
        return True
    return False


def placeholder(blocks: list[RawBlock]) -> str:
    """One placeholder per run of adjacent equation blocks."""
    nums = []
    for b in blocks:
        for ln in b.lines:
            m = EQ_NUMBER.search(ln.text)
            if m:
                nums.append(m.group(1))
    if not nums:
        return "[equation omitted]"
    if len(nums) == 1:
        return f"[equation ({nums[0]}) omitted]"
    return f"[equations ({nums[0]})–({nums[-1]}) omitted]"
