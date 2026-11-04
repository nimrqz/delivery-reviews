"""Dados tokenizados, métricas do Trainer e função de treino."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import BASE_MODEL, MAX_LENGTH, SEED, set_seed
from .labels import SENTIMENTS, URGENCIES, encode_choice, problems_matrix
from .metrics import problems_metrics, single_label_metrics
from .modeling import MultiHeadModel


class ReviewsDataset(Dataset):
    """Dataset tokenizado com padding fixo e os 3 conjuntos de rótulos."""

    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_length: int = MAX_LENGTH) -> None:
        self.enc = tokenizer(
            list(df["text"]), truncation=True, padding="max_length", max_length=max_length
        )
        records = df.to_dict("records")
        self.sent = [encode_choice(s, SENTIMENTS) for s in df["sentimento"]]
        self.urg = [encode_choice(u, URGENCIES) for u in df["urgencia"]]
        self.prob = problems_matrix(records).astype("float32")

    def __len__(self) -> int:
        return len(self.sent)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["sentimento_labels"] = torch.tensor(self.sent[i])
        item["urgencia_labels"] = torch.tensor(self.urg[i])
        item["problemas_labels"] = torch.tensor(self.prob[i])
        return item


LABEL_NAMES = ["sentimento_labels", "problemas_labels", "urgencia_labels"]


def compute_metrics(eval_pred: Any, threshold: float = 0.5) -> dict[str, float]:
    """Resumo escalar para o Trainer (F1 macro por tarefa). Ordem dos logits: sent, prob, urg."""
    sent_l, prob_l, urg_l = eval_pred.predictions
    sent_y, prob_y, urg_y = eval_pred.label_ids
    prob_pred = (1 / (1 + np.exp(-prob_l)) >= threshold).astype(int)
    f_s = single_label_metrics(sent_y, sent_l.argmax(-1), SENTIMENTS)["f1_macro"]
    f_u = single_label_metrics(urg_y, urg_l.argmax(-1), URGENCIES)["f1_macro"]
    f_p = problems_metrics(prob_y.astype(int), prob_pred)["f1_macro"]
    return {"f1_sentimento": f_s, "f1_urgencia": f_u, "f1_problemas": f_p,
            "f1_media": (f_s + f_u + f_p) / 3}


def detailed_metrics(logits: tuple, labels: tuple, threshold: float = 0.5) -> dict[str, object]:
    """Métricas completas (por label + matrizes de confusão) para o conjunto de teste."""
    sent_l, prob_l, urg_l = logits
    sent_y, prob_y, urg_y = labels
    prob_pred = (1 / (1 + np.exp(-prob_l)) >= threshold).astype(int)
    return {
        "sentimento": single_label_metrics(sent_y, sent_l.argmax(-1), SENTIMENTS),
        "urgencia": single_label_metrics(urg_y, urg_l.argmax(-1), URGENCIES),
        "problemas": problems_metrics(prob_y.astype(int), prob_pred),
    }


def compute_pos_weight(prob_matrix: np.ndarray, cap: float = 10.0) -> torch.Tensor:
    """pos_weight por label (neg/pos), limitado por `cap`, para lidar com classes raras."""
    pos = prob_matrix.sum(0)
    neg = len(prob_matrix) - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, cap), dtype=torch.float32)


def train(
    data_dir: str | Path,
    output_dir: str | Path = "outputs/bertimbau",
    base_model: str = BASE_MODEL,
    epochs: int = 5,
    lr: float = 3e-5,
    batch_size: int = 16,
    max_length: int = MAX_LENGTH,
    seed: int = SEED,
) -> dict[str, object]:
    """Fine-tuning completo; devolve as métricas de teste e salva o melhor modelo em `output_dir`."""
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    set_seed(seed)
    data_dir = Path(data_dir)
    dfs = {n: pd.read_json(data_dir / f"{n}.jsonl", lines=True) for n in ("train", "validation", "test")}
    tok = AutoTokenizer.from_pretrained(base_model)
    ds = {n: ReviewsDataset(d, tok, max_length) for n, d in dfs.items()}

    model = MultiHeadModel.from_base(base_model)
    model.prob_pos_weight.copy_(compute_pos_weight(ds["train"].prob))

    args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
        weight_decay=0.01, warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_media", greater_is_better=True,
        save_total_limit=1, label_names=LABEL_NAMES, seed=seed, data_seed=seed,
        fp16=torch.cuda.is_available(), report_to="none", logging_steps=20,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["validation"], compute_metrics=compute_metrics)
    trainer.train()
    pred = trainer.predict(ds["test"])
    metrics = detailed_metrics(pred.predictions, pred.label_ids)
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))
    return metrics

