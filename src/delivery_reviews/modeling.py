"""Modelo multi-tarefa: encoder BERT compartilhado + 3 cabeças (sentimento, problemas, urgência)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from .labels import PROBLEMS, SENTIMENTS, URGENCIES


class MultiHeadConfig(PretrainedConfig):
    """Config com o config do encoder embutido (para salvar/carregar sem depender do Hub)."""

    model_type = "delivery-multihead"

    def __init__(self, encoder_config: dict[str, Any] | None = None, dropout: float = 0.1, **kw):
        super().__init__(**kw)
        self.encoder_config = encoder_config or {}
        self.dropout = dropout


class MultiHeadModel(PreTrainedModel):
    """Saída: `sent_logits` (3), `prob_logits` (8, sigmoid) e `urg_logits` (3)."""

    config_class = MultiHeadConfig

    def __init__(self, config: MultiHeadConfig) -> None:
        super().__init__(config)
        enc_cfg = AutoConfig.for_model(**config.encoder_config)
        self.encoder = AutoModel.from_config(enc_cfg, add_pooling_layer=True)
        hidden = enc_cfg.hidden_size
        self.drop = nn.Dropout(config.dropout)
        self.sent_head = nn.Linear(hidden, len(SENTIMENTS))
        self.prob_head = nn.Linear(hidden, len(PROBLEMS))
        self.urg_head = nn.Linear(hidden, len(URGENCIES))
        self.register_buffer("prob_pos_weight", torch.ones(len(PROBLEMS)), persistent=False)
        self.post_init()

    @classmethod
    def from_base(cls, base_name: str, dropout: float = 0.1) -> "MultiHeadModel":
        """Cria o modelo carregando os pesos pré-treinados do encoder (ex.: BERTimbau)."""
        enc = AutoModel.from_pretrained(base_name)
        model = cls(MultiHeadConfig(encoder_config=enc.config.to_dict(), dropout=dropout))
        model.encoder = enc
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        sentimento_labels: torch.Tensor | None = None,
        problemas_labels: torch.Tensor | None = None,
        urgencia_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        pooled = self.drop(out.pooler_output)
        result = {
            "sent_logits": self.sent_head(pooled),
            "prob_logits": self.prob_head(pooled),
            "urg_logits": self.urg_head(pooled),
        }
        if sentimento_labels is not None:
            ce = nn.CrossEntropyLoss()
            bce = nn.BCEWithLogitsLoss(pos_weight=self.prob_pos_weight)
            result["loss"] = (
                ce(result["sent_logits"], sentimento_labels)
                + ce(result["urg_logits"], urgencia_labels)
                + bce(result["prob_logits"], problemas_labels.float())
            )
        return result
