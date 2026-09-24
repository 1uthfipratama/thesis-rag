"""Token counting in the embedding model's own tokenizer.

Chunk limits belong to the embedder: bge-small-en-v1.5 reads at most 512
WordPiece tokens and silently truncates the rest. Sizing chunks with a different
tokenizer (the plan's cl100k) let 44% of chunks overflow, so their tails were
never embedded. Counting with the embedder's tokenizer makes the limit exact.
"""

from functools import lru_cache

from tokenizers import Tokenizer

from rag.config import settings

# Model max sequence length, including [CLS] and [SEP].
EMBED_MAX_TOKENS = 512
SPECIAL_TOKENS = 2


@lru_cache(maxsize=1)
def tokenizer() -> Tokenizer:
    from fastembed import TextEmbedding

    # A copy with truncation off, so counts are real lengths; the embedding
    # model's own tokenizer object is left untouched.
    tok = Tokenizer.from_str(TextEmbedding(settings.embed_model).model.tokenizer.to_str())
    tok.no_truncation()
    tok.no_padding()
    return tok


def count(text: str) -> int:
    """Tokens without [CLS]/[SEP]."""
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def truncate(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most `limit` tokens (last resort for huge sentences)."""
    enc = tokenizer().encode(text, add_special_tokens=False)
    pieces = []
    for i in range(0, len(enc.ids), limit):
        start = enc.offsets[i][0]
        end = enc.offsets[min(i + limit, len(enc.ids)) - 1][1]
        pieces.append(text[start:end])
    return pieces
