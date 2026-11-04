import pandas as pd

from delivery_reviews.baseline import TfidfBaseline
from delivery_reviews.rules import rule_label


def _toy(n=40):
    pos = "Comida deliciosa e a entrega foi rápida, recomendo muito"
    neg = "O pedido atrasou muito e a comida chegou fria, péssimo"
    rows = []
    for i in range(n):
        if i % 2:
            rows.append({"text": f"{pos} {i % 3}", "sentimento": "positivo", "problemas": [], "urgencia": "baixa"})
        else:
            rows.append({"text": f"{neg} {i % 3}", "sentimento": "negativo",
                         "problemas": ["atraso", "comida_fria"], "urgencia": "media"})
    return pd.DataFrame(rows)


def test_baseline_fits_and_evaluates():
    df = _toy()
    res = TfidfBaseline().fit(df).evaluate(df)
    assert res["sentimento"]["f1_micro"] > 0.9
    assert set(res["problemas"]["f1_per_label"]) >= {"atraso", "comida_fria"}
    assert len(res["sentimento"]["confusion_matrix"]) == 3


def test_rule_label():
    r = rule_label("Pedido atrasou e a comida chegou fria")
    assert r["problemas"] == ["atraso", "comida_fria"] and r["sentimento"] == "negativo"
    assert rule_label("Adorei, excelente!")["sentimento"] == "positivo"
    assert rule_label("Nunca chegou o pedido")["urgencia"] == "alta"
