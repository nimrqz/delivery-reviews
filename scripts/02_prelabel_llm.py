"""Pré-rotula candidatos com um LLM (configurado no .env). Retoma de onde parou.

Uso: python scripts/02_prelabel_llm.py --limit 5          # teste barato
     python scripts/02_prelabel_llm.py                    # tudo (gasta API)
Saída: data/prelabeled.csv com colunas text, sentimento, problemas ("a|b"), urgencia, revisado.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from delivery_reviews.config import load_settings
from delivery_reviews.dataset import problems_to_str
from delivery_reviews.prelabel import call_llm, parse_llm_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/candidates.csv")
    ap.add_argument("--out", default="data/prelabeled.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    settings = load_settings()
    cand = pd.read_csv(args.inp)
    out_path = Path(args.out)
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    seen = set(done["text"]) if len(done) else set()
    todo = [t for t in cand["text"] if t not in seen][: args.limit]
    print(f"{len(todo)} textos a rotular com {settings.llm_provider}/{settings.llm_model}")

    rows = done.to_dict("records") if len(done) else []
    failures = 0
    for i, text in enumerate(todo, 1):
        try:
            rec = parse_llm_json(call_llm(settings, text))
        except Exception as exc:  # noqa: BLE001 - falha isolada não derruba o lote
            failures += 1
            print(f"  [{i}] falhou: {exc}")
            continue
        rows.append({"text": text, **rec, "problemas": problems_to_str(rec["problemas"]),
                     "revisado": 0})
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"  {i}/{len(todo)}")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Concluído: {len(rows)} linhas ({failures} falhas) em {out_path}")


if __name__ == "__main__":
    main()
