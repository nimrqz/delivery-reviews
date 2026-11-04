# Classificador de avaliações de delivery (PT-BR)

Dada uma avaliação de cliente, o modelo devolve três coisas ao mesmo tempo:

- **sentimento**: negativo, neutro ou positivo
- **problemas** (multi-label): atraso, comida fria, pedido errado, item faltando, embalagem, entregador, preço, qualidade da comida
- **urgência**: baixa, média ou alta

O modelo é um [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) com um encoder compartilhado e três cabeças de classificação, treinadas juntas. Existe também um baseline TF-IDF + regressão logística para comparar.

## Status

Projeto de estudo, **em andamento**. O código do pipeline completo está pronto e testado, mas ainda não há dataset final rotulado nem modelo treinado publicado, então **este repositório não traz métricas**. Os números só entram aqui depois que o treino rodar de verdade.

## Limitações que importam

- **Domínio diferente.** Os textos vêm do [B2W-Reviews01](https://huggingface.co/datasets/ruanchaves/b2w-reviews01), que é de e-commerce, não de apps de delivery. Um filtro por palavras-chave seleciona avaliações que falam de entrega, comida e embalagem, mas o vocabulário continua sendo o de e-commerce. Não espere o mesmo desempenho em avaliações reais de iFood e similares.
- **Rótulos fracos.** Os rótulos nascem de um LLM (ou de regras de palavra-chave, sem API) e depois são revisados por uma única pessoa. Isso carrega viés do pré-rotulador e subjetividade, principalmente em urgência.
- **Seleção por palavra-chave** superestima a frequência de problemas em relação a uma amostra aleatória de pedidos.
- **Classes raras** (`item_faltando`, `entregador`) têm poucos exemplos.
- **Licença.** O B2W é CC BY-NC-SA 4.0, então o dataset derivado herda: uso não comercial. Detalhes em [`data/DATASET_CARD.md`](data/DATASET_CARD.md).

## Como funciona

```
01_sample_candidates  ->  02_prelabel_llm (ou 02b_prelabel_rules)  ->  revisão manual em CSV
        ->  03_finalize_dataset  ->  04_push_dataset (HF Hub)
        ->  experiments/run_baseline  /  05_train  ->  06_push_model (HF Hub)
```

| Etapa | Script | O que faz |
|---|---|---|
| 1 | `scripts/01_sample_candidates.py` | Baixa o B2W e salva candidatos filtrados por palavras-chave |
| 2 | `scripts/02_prelabel_llm.py` | Pré-rotula com um LLM (Anthropic ou API compatível com OpenAI), retomando de onde parou |
| 2b | `scripts/02b_prelabel_rules.py` | Alternativa sem API: regras de palavra-chave (rótulos fracos) |
| 3 | `scripts/03_finalize_dataset.py` | Valida o CSV revisado e gera `train/validation/test` (80/10/10, estratificado por sentimento, seed 42) |
| 4 | `scripts/04_push_dataset.py` | Publica o dataset e o dataset card no Hugging Face |
| 5 | `scripts/05_train.py` | Fine-tuning do BERTimbau (local ou Colab, `notebooks/train_colab.ipynb`) |
| 6 | `scripts/06_push_model.py` | Publica o modelo com model card |
| - | `experiments/run_baseline.py` | Treina e avalia o baseline TF-IDF + LR |

### Revisão manual

O passo 2 gera `data/prelabeled.csv` com a coluna `revisado=0`. Abra no Excel ou Sheets, corrija `sentimento`, `problemas` (separados por `|`) e `urgencia`, e marque `revisado=1`. O passo 3 recusa linhas não revisadas.

### Modelo

`src/delivery_reviews/modeling.py`: um encoder BERT compartilhado e três cabeças lineares. A loss é a soma de duas cross-entropies (sentimento, urgência) com uma BCE (problemas), com `pos_weight` para lidar com classes raras. A métrica principal é F1 macro por tarefa.

## Como rodar

Requer Python 3.11 ou superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/Mac: source .venv/bin/activate)
pip install -e ".[dev]"         # núcleo + testes
pip install -e ".[train]"       # torch/transformers, para treinar e inferir
pip install -e ".[llm]"         # só se for pré-rotular com LLM
cp .env.example .env            # preencha HF_TOKEN, HF_USERNAME e a chave do LLM
```

Testes (não precisam de chave de API nem de rede):

```bash
python -m pytest tests
```

Sem `torch` instalado, o teste de inferência com BERT é pulado e os demais rodam. Os testes de modelo usam um BERT minúsculo gerado na hora.

Inferência com um modelo treinado:

```python
from delivery_reviews.inference import Predictor

pred = Predictor("outputs/bertimbau")
pred.predict(["O pedido chegou frio e uma hora atrasado"])
```

## Próximos passos

- Rodar a pré-rotulagem, revisar à mão e gerar o dataset final
- Treinar o baseline e o BERTimbau e registrar as métricas aqui
- Publicar dataset e modelo no Hugging Face
- Demo em Gradio (a pasta `space/` está reservada)
