"""Prompt e parser da pré-rotulagem com LLM, mais cliente configurável via .env."""
from __future__ import annotations

import json
import re
from typing import Any

from .config import Settings
from .labels import PROBLEMS, SENTIMENTS, URGENCIES, validate_record

SYSTEM_PROMPT = f"""Você rotula avaliações de clientes de delivery de comida em português do Brasil.
Responda APENAS com um JSON, sem texto extra, no formato:
{{"sentimento": "<{'|'.join(SENTIMENTS)}>", "problemas": [<zero ou mais de: {', '.join(PROBLEMS)}>], "urgencia": "<{'|'.join(URGENCIES)}>"}}

Regras:
- problemas só lista o que o cliente de fato relata; elogio não é problema.
- urgencia: alta = pedido não chegou, comida estragada/risco à saúde, cobrança indevida; media = problema que exige resposta; baixa = sem problema ou crítica leve.
- Se o texto não tem relação com delivery, use sentimento pelo tom e problemas vazio."""

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Extrai e valida o JSON da resposta do LLM. Levanta ValueError se inválido."""
    match = _JSON.search(raw or "")
    if not match:
        raise ValueError(f"Sem JSON na resposta: {raw!r}")
    try:
        rec = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido: {exc}") from exc
    rec["problemas"] = sorted(set(rec.get("problemas", [])))
    validate_record(rec)
    return {k: rec[k] for k in ("sentimento", "problemas", "urgencia")}


def call_llm(settings: Settings, text: str, max_tokens: int = 150) -> str:
    """Envia uma avaliação ao LLM configurado e devolve o texto bruto da resposta."""
    if not settings.llm_api_key:
        raise RuntimeError("LLM_API_KEY não definido (veja .env.example)")
    if settings.llm_provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=settings.llm_api_key)
        msg = client.messages.create(
            model=settings.llm_model,
            max_tokens=max_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
        return msg.content[0].text
    if settings.llm_provider == "openai_compatible":
        from openai import OpenAI

        client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content or ""
    raise ValueError(f"LLM_PROVIDER desconhecido: {settings.llm_provider}")
