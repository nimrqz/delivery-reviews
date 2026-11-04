from delivery_reviews.preprocessing import is_usable, normalize_text


def test_normalize_removes_urls_and_spaces():
    assert normalize_text("Chegou  frio   http://x.com/a ok") == "Chegou frio ok"


def test_normalize_limits_repeats():
    assert normalize_text("ruuuuim") == "ruuim"


def test_normalize_none_and_empty():
    assert normalize_text(None) == ""
    assert normalize_text("   ") == ""


def test_is_usable_bounds():
    assert not is_usable("curto")
    assert is_usable("O pedido chegou atrasado e frio")
    assert not is_usable("a" * 600)
