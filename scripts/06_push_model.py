"""Publica o modelo treinado no Hub com model card gerado a partir de outputs/metrics.json.

Uso: python scripts/06_push_model.py [--model-dir outputs/bertimbau] [--metrics outputs/metrics.json]
Recusa publicar sem métricas reais.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, ModelCard

from delivery_reviews.config import BASE_MODEL, load_settings
from delivery_reviews.inference import Predictor

CARD = """---
language: [pt]
license: mit
base_model: {base}
datasets: [{dataset}]
pipeline_tag: text-classification
tags: [multi-label, sentiment, delivery, portuguese, bertimbau]
---

# BERTimbau — avaliações de delivery

Modelo multi-tarefa baseado em [`{base}`](https://huggingface.co/{base}) que lê uma avaliação
de delivery e prevê **sentimento**, **problemas (multi-label)** e **urgência**.

## Uso

```python
# requer o pacote do projeto: pip install git+https://github.com/nimrqz/delivery-reviews
from delivery_reviews.inference import Predictor

pred = Predictor("{repo}")
print(pred.predict(["O pedido chegou frio e 1 hora atrasado"])[0]["problemas_ativos"])
```

O modelo usa uma arquitetura customizada (`MultiHeadModel`), então **não funciona** com `pipeline()` padrão.

## Treino
- Dados: [{dataset}](https://huggingface.co/datasets/{dataset}) (train/validation/test, seed 42).
- Encoder compartilhado + 3 cabeças: sentimento e urgência (cross-entropy), problemas (BCE com `pos_weight`).
- Limiar de decisão dos problemas: 0,5.

## Métricas (conjunto de teste)

{table}

## Limitações e vieses
- Treinado em reviews de **e-commerce** filtradas por palavras-chave, não em reviews reais de apps de delivery.
- Rótulos pré-gerados por regras/LLM e revisados por uma pessoa: ruído e subjetividade, sobretudo em `urgencia`.
- Classes raras têm poucos exemplos e F1 instável; confira o F1 por label acima.
- Não avaliado em gírias regionais, emojis ou textos muito curtos/longos (truncado em 128 tokens).
- Não use para decisões automáticas sobre clientes ou entregadores sem revisão humana.
"""


def _table(m: dict) -> str:
    rows = ["| Tarefa | F1 micro | F1 macro |", "|---|---|---|"]
    for task in ("sentimento", "urgencia", "problemas"):
        rows.append(f"| {task} | {m[task]['f1_micro']:.3f} | {m[task]['f1_macro']:.3f} |")
    rows += ["", "F1 por problema:", "", "| Problema | F1 |", "|---|---|"]
    rows += [f"| {k} | {v:.3f} |" for k, v in m["problemas"]["f1_per_label"].items()]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs/bertimbau")
    ap.add_argument("--metrics", default="outputs/metrics.json")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    s = load_settings()
    if not (s.hf_token and s.hf_username):
        raise SystemExit("Defina HF_TOKEN e HF_USERNAME no .env")
    mp = Path(args.metrics)
    if not mp.exists():
        raise SystemExit(f"{mp} não existe: treine antes (scripts/05_train.py)")
    Predictor(args.model_dir, device="cpu").predict(["teste rápido de carregamento"])  # sanity check

    api = HfApi(token=s.hf_token)
    api.create_repo(s.model_repo, exist_ok=True, private=args.private)
    api.upload_folder(folder_path=args.model_dir, repo_id=s.model_repo, commit_message="Modelo treinado")
    card = CARD.format(base=BASE_MODEL, repo=s.model_repo, dataset=s.dataset_repo,
                       table=_table(json.loads(mp.read_text())))
    ModelCard(card).push_to_hub(s.model_repo, token=s.hf_token)
    print(f"Publicado: https://huggingface.co/{s.model_repo}")


if __name__ == "__main__":
    main()
