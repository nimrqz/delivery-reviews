"""Fine-tuning do BERTimbau (multi-tarefa). Roda local ou no Colab (GPU T4 gratuita).

Uso: python scripts/05_train.py --data-dir data/final --epochs 5
Se --hf-dataset for passado, baixa os splits do Hub em vez de usar --data-dir.
Saída: outputs/bertimbau/ (modelo + tokenizer) e outputs/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from delivery_reviews.config import BASE_MODEL, load_settings
from delivery_reviews.training import train


def _download_splits(repo: str, token: str | None) -> str:
    from datasets import load_dataset

    out = Path(tempfile.mkdtemp())
    ds = load_dataset(repo, token=token)
    for split in ("train", "validation", "test"):
        ds[split].to_json(str(out / f"{split}.jsonl"), force_ascii=False)
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/final")
    ap.add_argument("--hf-dataset", default=None, help="ex.: nimrqz/delivery-reviews-ptbr")
    ap.add_argument("--output-dir", default="outputs/bertimbau")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    data_dir = _download_splits(args.hf_dataset, load_settings().hf_token) if args.hf_dataset \
        else args.data_dir
    metrics = train(data_dir, args.output_dir, args.base_model, args.epochs, args.lr, args.batch_size)
    out = Path(args.output_dir).parent / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: round(v["f1_macro"], 4) for k, v in metrics.items()}))
    print(f"Métricas em {out}")


if __name__ == "__main__":
    main()
