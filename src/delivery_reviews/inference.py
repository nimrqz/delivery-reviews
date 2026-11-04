"""Inferência: carrega o modelo (local ou Hub) e devolve probabilidades por label."""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer

from .config import MAX_LENGTH
from .labels import PROBLEMS, SENTIMENTS, URGENCIES
from .modeling import MultiHeadModel
from .preprocessing import normalize_text


class Predictor:
    """Wrapper de inferência em CPU/GPU para o `MultiHeadModel`."""

    def __init__(self, model_path: str, device: str | None = None, threshold: float = 0.5) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MultiHeadModel.from_pretrained(model_path).to(self.device).eval()

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Para cada texto: probabilidades de sentimento/urgência/problemas e problemas ativos."""
        clean = [normalize_text(t) for t in texts]
        enc = self.tokenizer(clean, truncation=True, padding=True, max_length=MAX_LENGTH,
                             return_tensors="pt").to(self.device)
        out = self.model(**enc)
        sent = out["sent_logits"].softmax(-1).cpu().tolist()
        urg = out["urg_logits"].softmax(-1).cpu().tolist()
        prob = out["prob_logits"].sigmoid().cpu().tolist()
        return [
            {
                "sentimento": dict(zip(SENTIMENTS, s)),
                "urgencia": dict(zip(URGENCIES, u)),
                "problemas": dict(zip(PROBLEMS, p)),
                "problemas_ativos": [n for n, v in zip(PROBLEMS, p) if v >= self.threshold],
            }
            for s, u, p in zip(sent, urg, prob)
        ]
