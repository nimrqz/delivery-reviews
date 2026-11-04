"""Esquema de rótulos e codificação de/para vetores."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

SENTIMENTS: tuple[str, ...] = ("negativo", "neutro", "positivo")
PROBLEMS: tuple[str, ...] = (
    "atraso",
    "comida_fria",
    "pedido_errado",
    "item_faltando",
    "embalagem",
    "entregador",
    "preco",
    "qualidade_comida",
)
URGENCIES: tuple[str, ...] = ("baixa", "media", "alta")


def encode_problems(problems: Sequence[str]) -> list[int]:
    """Lista de problemas -> vetor multi-hot na ordem de `PROBLEMS`."""
    unknown = set(problems) - set(PROBLEMS)
    if unknown:
        raise ValueError(f"Problemas desconhecidos: {sorted(unknown)}")
    return [int(p in problems) for p in PROBLEMS]


def decode_problems(vector: Sequence[float], threshold: float = 0.5) -> list[str]:
    """Vetor de probabilidades/binário -> nomes dos problemas acima do limiar."""
    if len(vector) != len(PROBLEMS):
        raise ValueError(f"Esperado vetor de tamanho {len(PROBLEMS)}, recebido {len(vector)}")
    return [name for name, v in zip(PROBLEMS, vector) if v >= threshold]


def encode_choice(value: str, options: Sequence[str]) -> int:
    """Rótulo categórico -> índice (levanta ValueError se inválido)."""
    if value not in options:
        raise ValueError(f"Valor inválido {value!r}; opções: {list(options)}")
    return list(options).index(value)


def validate_record(rec: Mapping[str, object]) -> None:
    """Valida um registro rotulado (sentimento, problemas, urgência)."""
    encode_choice(str(rec["sentimento"]), SENTIMENTS)
    encode_choice(str(rec["urgencia"]), URGENCIES)
    encode_problems(list(rec["problemas"]))  # type: ignore[arg-type]


def problems_matrix(records: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Matriz (n, 8) multi-hot para uma lista de registros."""
    return np.array([encode_problems(list(r["problemas"])) for r in records])  # type: ignore[arg-type]
