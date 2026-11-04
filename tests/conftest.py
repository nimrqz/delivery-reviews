"""Fixture: BERT minúsculo aleatório + tokenizer local (sem rede)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertConfig, BertTokenizerFast  # noqa: E402

from delivery_reviews.modeling import MultiHeadConfig, MultiHeadModel  # noqa: E402

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "o", "pedido", "chegou", "frio",
         "atrasado", "ótimo", "comida", "entrega", "rápida", "e", "muito"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny")
    vocab_file = path / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    enc = BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    torch.manual_seed(0)
    model = MultiHeadModel(MultiHeadConfig(encoder_config=enc.to_dict()))
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)
