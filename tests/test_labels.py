import pytest

from delivery_reviews.labels import (
    PROBLEMS,
    decode_problems,
    encode_choice,
    encode_problems,
    validate_record,
)


def test_encode_decode_roundtrip():
    vec = encode_problems(["atraso", "preco"])
    assert len(vec) == len(PROBLEMS)
    assert decode_problems(vec) == ["atraso", "preco"]


def test_unknown_problem_raises():
    with pytest.raises(ValueError):
        encode_problems(["inexistente"])


def test_decode_threshold():
    probs = [0.9, 0.4, 0, 0, 0, 0, 0, 0.6]
    assert decode_problems(probs) == ["atraso", "qualidade_comida"]


def test_encode_choice_invalid():
    with pytest.raises(ValueError):
        encode_choice("otimo", ("negativo", "neutro", "positivo"))


def test_validate_record_ok_and_bad():
    validate_record({"sentimento": "negativo", "problemas": ["atraso"], "urgencia": "alta"})
    with pytest.raises(ValueError):
        validate_record({"sentimento": "x", "problemas": [], "urgencia": "alta"})
