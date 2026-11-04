import pandas as pd
import pytest

from delivery_reviews.dataset import (
    filter_candidates, problems_to_str, rows_from_reviewed_csv, split_dataset, str_to_problems,
)
from delivery_reviews.prelabel import parse_llm_json


def test_parse_llm_json_with_noise():
    raw = 'Claro!\n{"sentimento":"negativo","problemas":["atraso","atraso"],"urgencia":"media"}'
    assert parse_llm_json(raw) == {
        "sentimento": "negativo", "problemas": ["atraso"], "urgencia": "media"}


@pytest.mark.parametrize("raw", ["sem json", '{"sentimento":"x","problemas":[],"urgencia":"alta"}'])
def test_parse_llm_json_invalid(raw):
    with pytest.raises(ValueError):
        parse_llm_json(raw)


def test_problems_roundtrip():
    assert str_to_problems(problems_to_str(["atraso", "preco"])) == ["atraso", "preco"]
    assert str_to_problems(float("nan")) == []


def test_filter_candidates():
    df = pd.DataFrame({"review_text": [
        "A entrega atrasou muito e a comida chegou fria", "ok", "O celular é ótimo, recomendo muito"]})
    assert len(filter_candidates(df)) == 1


def _df(n=100):
    return pd.DataFrame({
        "text": [f"texto {i}" for i in range(n)],
        "sentimento": ["negativo", "neutro", "positivo", "negativo"] * (n // 4),
        "problemas": [""] * n, "urgencia": ["baixa"] * n, "revisado": [1] * n})


def test_reviewed_csv_requires_review():
    df = _df(8)
    df.loc[0, "revisado"] = 0
    with pytest.raises(ValueError):
        rows_from_reviewed_csv(df)
    assert len(rows_from_reviewed_csv(_df(8))) == 8


def test_split_sizes_and_disjoint():
    s = split_dataset(pd.DataFrame(rows_from_reviewed_csv(_df(100))))
    assert (len(s["train"]), len(s["validation"]), len(s["test"])) == (80, 10, 10)
    assert not set(s["train"]["text"]) & set(s["test"]["text"])
