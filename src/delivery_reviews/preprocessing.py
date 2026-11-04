"""Limpeza de texto. Mantém caixa e acentos: o BERTimbau é cased."""
from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_SPACES = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{3,}")  # 4+ repetições viram 2: "ruuuuim" -> "ruuim"


def normalize_text(text: str | None) -> str:
    """Remove URLs, comprime repetições de caractere e espaços."""
    if not text:
        return ""
    text = _URL.sub(" ", str(text))
    text = _REPEAT.sub(r"\1\1", text)
    return _SPACES.sub(" ", text).strip()


def is_usable(text: str, min_chars: int = 15, max_chars: int = 500) -> bool:
    """Filtro de qualidade: descarta textos vazios, muito curtos ou muito longos."""
    return min_chars <= len(normalize_text(text)) <= max_chars
