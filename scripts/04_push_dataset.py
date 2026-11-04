"""Publica os splits em data/final/ no Hugging Face Hub, com dataset card.

Uso: python scripts/04_push_dataset.py [--private]   (usa HF_TOKEN e HF_USERNAME do .env)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard

from delivery_reviews.config import load_settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/final")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    s = load_settings()
    if not (s.hf_token and s.hf_username):
        raise SystemExit("Defina HF_TOKEN e HF_USERNAME no .env")
    parts = {n: pd.read_json(f"{args.dir}/{n}.jsonl", lines=True)
             for n in ("train", "validation", "test")}
    dd = DatasetDict({n: Dataset.from_pandas(df, preserve_index=False) for n, df in parts.items()})
    dd.push_to_hub(s.dataset_repo, token=s.hf_token, private=args.private)

    total = sum(len(d) for d in parts.values())
    card = (Path("data/DATASET_CARD.md").read_text(encoding="utf-8")
            .replace("__N_TOTAL__", str(total)).replace("__N_REVIEWED__", str(total))
            .replace("__LLM_MODEL__", s.llm_model))
    DatasetCard(card).push_to_hub(s.dataset_repo, token=s.hf_token)
    print(f"Publicado: https://huggingface.co/datasets/{s.dataset_repo}")


if __name__ == "__main__":
    main()
