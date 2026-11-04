import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from transformers import AutoTokenizer  # noqa: E402

from delivery_reviews.inference import Predictor  # noqa: E402
from delivery_reviews.labels import PROBLEMS  # noqa: E402
from delivery_reviews.modeling import MultiHeadModel  # noqa: E402
from delivery_reviews.training import ReviewsDataset, compute_pos_weight  # noqa: E402


def test_predictor_output_shape_and_probs(tiny_model_dir):
    preds = Predictor(tiny_model_dir, device="cpu").predict(
        ["O pedido chegou frio e atrasado", "ótimo entrega rápida"])
    assert len(preds) == 2
    p = preds[0]
    assert abs(sum(p["sentimento"].values()) - 1) < 1e-5
    assert abs(sum(p["urgencia"].values()) - 1) < 1e-5
    assert set(p["problemas"]) == set(PROBLEMS)
    assert all(0 <= v <= 1 for v in p["problemas"].values())


def test_forward_computes_loss_and_backward(tiny_model_dir):
    model = MultiHeadModel.from_pretrained(tiny_model_dir)
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    df = pd.DataFrame({"text": ["o pedido chegou frio", "ótimo comida"],
                       "sentimento": ["negativo", "positivo"],
                       "problemas": [["comida_fria"], []], "urgencia": ["media", "baixa"]})
    ds = ReviewsDataset(df, tok, max_length=8)
    batch = {k: torch.stack([ds[i][k] for i in range(2)]) for k in ds[0]}
    out = model(**batch)
    assert out["loss"].item() > 0
    out["loss"].backward()


def test_pos_weight_capped():
    import numpy as np
    w = compute_pos_weight(np.array([[1, 0], [0, 0], [0, 0], [0, 0]]), cap=2.0)
    assert w.tolist() == [2.0, 2.0]


def test_trainer_smoke(tiny_model_dir, tmp_path):
    """Trainer + saída em dict + label_names + compute_metrics funcionam ponta a ponta."""
    from transformers import Trainer, TrainingArguments

    from delivery_reviews.training import LABEL_NAMES, compute_metrics, detailed_metrics

    model = MultiHeadModel.from_pretrained(tiny_model_dir)
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    rows = [("o pedido chegou frio", "negativo", ["comida_fria"], "media"),
            ("ótimo entrega rápida", "positivo", [], "baixa"),
            ("pedido atrasado muito", "negativo", ["atraso"], "alta"),
            ("comida ótimo e rápida", "positivo", [], "baixa")] * 3
    df = pd.DataFrame(rows, columns=["text", "sentimento", "problemas", "urgencia"])
    ds = ReviewsDataset(df, tok, max_length=8)
    args = TrainingArguments(output_dir=str(tmp_path), num_train_epochs=1, per_device_train_batch_size=4,
                             eval_strategy="epoch", save_strategy="no", report_to="none",
                             label_names=LABEL_NAMES, use_cpu=True)
    tr = Trainer(model=model, args=args, train_dataset=ds, eval_dataset=ds, compute_metrics=compute_metrics)
    tr.train()
    res = tr.evaluate()
    assert "eval_f1_media" in res
    pred = tr.predict(ds)
    m = detailed_metrics(pred.predictions, pred.label_ids)
    assert len(m["sentimento"]["confusion_matrix"]) == 3
