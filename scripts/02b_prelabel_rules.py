"""Pré-rotulagem SEM API, por regras de palavras-chave. Mesmo formato do 02_prelabel_llm.py.

Uso: python scripts/02b_prelabel_rules.py   -> data/prelabeled.csv (revisado=0)
Atenção: rótulos fracos; revise tudo manualmente antes de usar.
"""
from __future__ import annotations

import argparse

import pandas as pd

from delivery_reviews.dataset import problems_to_str
from delivery_reviews.rules import rule_label


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/candidates.csv")
    ap.add_argument("--out", default="data/prelabeled.csv")
    args = ap.parse_args()
    rows = []
    for text in pd.read_csv(args.inp)["text"]:
        rec = rule_label(text)
        rows.append({"text": text, **rec, "problemas": problems_to_str(rec["problemas"]), "revisado": 0})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"{len(rows)} linhas em {args.out}")


if __name__ == "__main__":
    main()
