"""Valida o CSV revisado e gera os splits train/validation/test (parquet em data/final/).

Fluxo de revisão: abra data/prelabeled.csv no Excel/Sheets/LibreOffice, corrija
sentimento / problemas (separados por "|") / urgencia e marque revisado=1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from delivery_reviews.dataset import rows_from_reviewed_csv, split_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/prelabeled.csv")
    ap.add_argument("--out", default="data/final")
    args = ap.parse_args()

    df = pd.DataFrame(rows_from_reviewed_csv(pd.read_csv(args.inp)))
    splits = split_dataset(df)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_json(f"{args.out}/{name}.jsonl", orient="records", lines=True, force_ascii=False)
        print(f"{name}: {len(part)}")
    print("Distribuição de sentimento:\n", df["sentimento"].value_counts().to_string())


if __name__ == "__main__":
    main()
