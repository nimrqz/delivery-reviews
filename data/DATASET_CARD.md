---
language: [pt]
license: cc-by-nc-sa-4.0
task_categories: [text-classification]
task_ids: [multi-label-classification, sentiment-classification]
size_categories: [1K<n<10K]
pretty_name: Delivery Reviews
source_datasets: [ruanchaves/b2w-reviews01]
---

# delivery-reviews

Avaliações de clientes em português do Brasil, rotuladas com **sentimento**, **problemas (multi-label)**
e **urgência**, para classificação de feedback de delivery.

## Estrutura

| Campo | Tipo | Valores |
|---|---|---|
| `text` | string | avaliação normalizada |
| `sentimento` | string | negativo, neutro, positivo |
| `problemas` | list[string] | atraso, comida_fria, pedido_errado, item_faltando, embalagem, entregador, preco, qualidade_comida |
| `urgencia` | string | baixa, media, alta |

Splits: `train` (80%), `validation` (10%), `test` (10%), estratificados por sentimento (seed 42).
Tamanho: __N_TOTAL__ exemplos.

## Como foi construído

1. **Origem:** amostra do [B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01)
   (e-commerce, não delivery), filtrada por palavras-chave de entrega/comida/embalagem/preço.
2. **Pré-rotulagem** com LLM (`__LLM_MODEL__`), temperatura 0, prompt em `src/delivery_reviews/prelabel.py`.
3. **Revisão manual** de __N_REVIEWED__ exemplos por uma pessoa (CSV).

## Limitações e vieses

- O texto original vem de **e-commerce**, não de apps de delivery: há diferença de domínio e vocabulário.
- A seleção por palavras-chave favorece reviews que já mencionam esses termos e **superestima problemas**
  em relação a uma amostra aleatória de pedidos.
- Rótulos de urgência são subjetivos e foram definidos por regra simples (ver prompt); uma única pessoa revisou.
- Os rótulos têm viés do LLM de pré-rotulagem onde a revisão foi superficial.
- Textos podem conter linguagem ofensiva e dados pessoais citados pelos autores; não há anonimização formal.
- Classes raras (ex.: `item_faltando`, `entregador`) têm poucos exemplos.

## Licença

Derivado do B2W-Reviews01, distribuído sob **CC BY-NC-SA 4.0** (uso não comercial, compartilhamento pela mesma licença).
Este dataset herda a mesma licença. Cite o trabalho original: Real et al. (2019), *B2W-Reviews01: an open product reviews corpus*.
