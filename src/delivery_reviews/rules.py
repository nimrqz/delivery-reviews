"""Pré-rotulador por regras (sem API). Gera rótulos *fracos* que precisam de revisão manual."""
from __future__ import annotations

import re

PROBLEM_PATTERNS: dict[str, str] = {
    "atraso": r"atras|demor|demorou|passou d[oa] (prazo|hora)|não chegou|nunca chegou",
    "comida_fria": r"\bfri[oa]s?\b|gelad|esfriou",
    "pedido_errado": r"errad[oa]|troca(ram|do)|não foi o que pedi|veio outro",
    "item_faltando": r"falt(ou|ando|aram)|esquec(eram|eu)|veio sem|incompleto",
    "embalagem": r"embalagem|vazou|derram|amassad|lacre|sem proteção",
    "entregador": r"entregador|motoboy|motoqueiro|mal educad|grosse|rude",
    "preco": r"\bcaro\b|\bcara\b|preço|taxa|cobr(ou|aram)|abusiv",
    "qualidade_comida": r"ruim|péssim|horrível|sem sabor|insosso|crua|queimad|estragad|azedo",
}
_NEG = re.compile(r"ruim|péssim|horrível|decepc|nunca mais|não recomendo|demor|atras|errad|frio|lament")
_POS = re.compile(r"ótim|excelente|adorei|amei|perfeit|recomendo|rápid|delici|gostoso|parabéns")
_HIGH = re.compile(r"não chegou|nunca chegou|estragad|azedo|passei mal|intoxic|cobr(ou|aram) (a mais|duas)")


def rule_label(text: str) -> dict[str, object]:
    """Rotula por palavras-chave. Heurística simples: espere erros e revise."""
    t = text.lower()
    problems = sorted(p for p, pat in PROBLEM_PATTERNS.items() if re.search(pat, t))
    neg, pos = len(_NEG.findall(t)), len(_POS.findall(t))
    if problems or neg > pos:
        sentiment = "negativo"
    elif pos > neg:
        sentiment = "positivo"
    else:
        sentiment = "neutro"
    if _HIGH.search(t):
        urgency = "alta"
    elif len(problems) >= 2 or "pedido_errado" in problems or "item_faltando" in problems:
        urgency = "media"
    else:
        urgency = "baixa"
    return {"sentimento": sentiment, "problemas": problems, "urgencia": urgency}
