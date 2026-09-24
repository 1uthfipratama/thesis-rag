"""Tidy recognised LaTeX for indexing.

Formula recognisers emit token-spaced, wrapper-heavy LaTeX:
    \\begin{array} { r } { \\mathbf { d } S ( t ) = r S ( t ) ... } \\end{array}\\tag{1}
The raw string is kept in Block.latex (it renders fine in KaTeX). What goes into
the indexed text is a tidier form, `dS(t) = rS(t) ...  (1)`, so BM25 tokens like
"S(t)" and the equation number match what people type and what the prose cites.
"""

import re

_WRAPPERS = re.compile(
    r"\\(begin|end)\{(array|aligned|align\*?|gathered|split|cases)\}(\s*\{\s*[lcr| ]+\s*\})?"
)
_FONT = re.compile(
    r"\\(mathbf|mathsf|mathrm|mathit|boldsymbol|mathcal|operatorname|text|textbf)\s*\{\s*([^{}]*?)\s*\}"
)
# \tag{1} (MinerU) or a trailing ", \quad (1)" (Marker)
_TAG = re.compile(r"\\tag\s*\{\s*([^{}]*)\s*\}|[,.]?\s*\\q?quad\s*\((\d{1,3}[a-z]?)\)\s*$")


_OPERATOR = re.compile(r"^(=|\+|-|<|>|\\(leq|geq|neq|approx|le|ge|times|cdot|pm|in|to)|&)$")


def _join_tokens(s: str) -> str:
    """Remove recogniser spacing ("d S" -> "dS") except where LaTeX needs it:
    after a command ("\\sigma S" must not become "\\sigmaS") and around operators."""
    toks = s.split()
    if not toks:
        return s
    out = toks[0]
    prev = toks[0]
    for t in toks[1:]:
        if re.search(r"\\[A-Za-z]+$", prev) or _OPERATOR.match(t) or _OPERATOR.match(prev):
            out += " " + t
        else:
            out += t
        prev = t
    return out


def _closing(s: str, i: int) -> int:
    """Index of the brace closing the one at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return j
    return -1


def tidy(latex: str) -> str:
    s = latex
    tag = _TAG.search(s)
    s = _TAG.sub("", s)
    s = _WRAPPERS.sub("", s)
    for _ in range(3):  # nested \mathbf{\mathrm{d}}
        s = _FONT.sub(r"\2", s)
    s = re.sub(r"\\(left|right|big|Big|bigg|Bigg)\b\s*", "", s)
    s = re.sub(r"\\(qquad|quad)\b|\\[,;:!]|~", " ", s)  # LaTeX spacing commands
    # collapse token spacing: "S ( t )" -> "S(t)", "x _ { n }" -> "x_{n}"
    s = re.sub(r"\s*([(){}\[\]^_,])\s*", r"\1", s)
    s = _join_tokens(s)
    s = re.sub(r"([_^])\{([^{}\\])\}", r"\1\2", s)  # x_{n} -> x_n (but not \bar{S})
    s = re.sub(r"\s*=\s*", " = ", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(",.")
    # strip the brace layer left by removed array wrappers: "{ ... }"
    while s.startswith("{") and _closing(s, 0) == len(s) - 1:
        s = s[1:-1].strip().rstrip(",.")
    if tag:
        s = f"{s}  ({(tag.group(1) or tag.group(2)).strip()})"
    return s
