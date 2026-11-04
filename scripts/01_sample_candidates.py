"""Baixa o B2W-Reviews01 do Hub e salva candidatos relacionados a delivery.

Uso: python scripts/01_sample_candidates.py --n 1300 --out data/candidates.csv
Requer acesso ao huggingface.co (dataset público, não precisa de token).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from delivery_reviews.config import SEED
from delivery_reviews.dataset import filter_candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="ruanchaves/b2w-reviews01")
    ap.add_argument("--text-col", default="review_text")
    ap.add_argument("--pool", type=int, default=60000, help="reviews lidas antes do filtro")
    ap.add_argument("--n", type=int, default=1300, help="candidatos finais (sobra p/ descarte)")
    ap.add_argument("--out", default="data/candidates.csv")
    args = ap.parse_args()

    ds = load_dataset(args.source, split="train")
    df = ds.shuffle(seed=SEED).select(range(min(args.pool, len(ds)))).to_pandas()
    cand = filter_candidates(df, args.text_col)
    cand = cand.sample(n=min(args.n, len(cand)), random_state=SEED).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cand.to_csv(args.out, index=False)
    print(f"{len(cand)} candidatos salvos em {args.out}")


if __name__ == "__main__":
    main()
