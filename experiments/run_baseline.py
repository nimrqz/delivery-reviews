"""Treina e avalia o baseline TF-IDF+LR. Salva experiments/results/baseline.json.

Uso: python experiments/run_baseline.py   (requer data/final/*.jsonl)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from delivery_reviews.baseline import TfidfBaseline
from delivery_reviews.config import set_seed


def main() -> None:
    set_seed()
    train = pd.read_json("data/final/train.jsonl", lines=True)
    test = pd.read_json("data/final/test.jsonl", lines=True)
    model = TfidfBaseline().fit(train)
    t0 = time.perf_counter()
    result = model.evaluate(test)
    result["latency_ms_per_review"] = (time.perf_counter() - t0) / len(test) * 1000
    result["cost_per_1000_usd"] = 0.0
    out = Path("experiments/results/baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k]["f1_macro"] for k in ("sentimento", "urgencia", "problemas")}))


if __name__ == "__main__":
    main()
