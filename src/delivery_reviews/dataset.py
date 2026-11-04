"""Utilitários de dataset: filtro de candidatos, (de)serialização de CSV e splits."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED
from .labels import validate_record
from .preprocessing import is_usable, normalize_text

# Palavras que sugerem avaliação ligada a entrega/comida (heurística simples).
DELIVERY_KEYWORDS = (
    "entrega", "entregador", "motoboy", "atras", "demor", "frio", "fria", "gelad",
    "embalagem", "lacre", "vazou", "derram", "faltou", "faltando", "errado", "errada",
    "comida", "pizza", "lanche", "hambúrguer", "hamburguer", "marmita", "restaurante",
    "pedido", "sabor", "cheio", "caro", "preço",
)
SEP = "|"


def filter_candidates(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Mantém textos utilizáveis que contenham alguma palavra-chave de delivery."""
    out = df.copy()
    out["text"] = out[text_col].map(normalize_text)
    mask = out["text"].map(is_usable) & out["text"].str.lower().str.contains(
        "|".join(DELIVERY_KEYWORDS), regex=True
    )
    return out[mask].drop_duplicates("text").reset_index(drop=True)[["text"]]


def problems_to_str(problems: list[str]) -> str:
    return SEP.join(problems)


def str_to_problems(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return []
    return [p.strip() for p in str(value).split(SEP) if p.strip()]


def rows_from_reviewed_csv(df: pd.DataFrame) -> list[dict]:
    """Converte CSV revisado em registros validados. Linhas não revisadas dão erro."""
    if "revisado" not in df.columns:
        raise ValueError("CSV sem coluna 'revisado'")
    pending = df[df["revisado"].astype(str).str.lower().isin({"", "nan", "0", "false", "não", "nao"})]
    if len(pending):
        raise ValueError(f"{len(pending)} linhas ainda não revisadas (marque revisado=1)")
    rows = []
    for _, r in df.iterrows():
        rec = {
            "text": r["text"],
            "sentimento": r["sentimento"],
            "problemas": str_to_problems(r["problemas"]),
            "urgencia": r["urgencia"],
        }
        validate_record(rec)
        rows.append(rec)
    return rows


def split_dataset(
    df: pd.DataFrame, val: float = 0.1, test: float = 0.1, seed: int = SEED
) -> dict[str, pd.DataFrame]:
    """Split estratificado por sentimento em train/validation/test."""
    train_val, test_df = train_test_split(
        df, test_size=test, stratify=df["sentimento"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val, test_size=val / (1 - test), stratify=train_val["sentimento"], random_state=seed
    )
    return {
        "train": train_df.reset_index(drop=True),
        "validation": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
