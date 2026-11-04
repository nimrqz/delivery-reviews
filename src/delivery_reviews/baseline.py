"""Baseline clássico: TF-IDF + regressão logística (uma por tarefa)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from .config import SEED
from .labels import PROBLEMS, SENTIMENTS, URGENCIES, encode_choice, problems_matrix
from .metrics import problems_metrics, single_label_metrics


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)


class TfidfBaseline:
    """Três modelos TF-IDF+LR: sentimento, urgência (softmax) e problemas (one-vs-rest)."""

    def __init__(self, seed: int = SEED) -> None:
        lr = lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)  # noqa: E731
        self.sent = Pipeline([("tfidf", _tfidf()), ("clf", lr())])
        self.urg = Pipeline([("tfidf", _tfidf()), ("clf", lr())])
        self.prob = Pipeline([("tfidf", _tfidf()), ("clf", OneVsRestClassifier(lr()))])

    def fit(self, df: pd.DataFrame) -> "TfidfBaseline":
        records = df.to_dict("records")
        self.sent.fit(df["text"], [encode_choice(s, SENTIMENTS) for s in df["sentimento"]])
        self.urg.fit(df["text"], [encode_choice(u, URGENCIES) for u in df["urgencia"]])
        self.prob.fit(df["text"], problems_matrix(records))
        return self

    def evaluate(self, df: pd.DataFrame) -> dict[str, object]:
        records = df.to_dict("records")
        return {
            "sentimento": single_label_metrics(
                [encode_choice(s, SENTIMENTS) for s in df["sentimento"]],
                self.sent.predict(df["text"]), SENTIMENTS),
            "urgencia": single_label_metrics(
                [encode_choice(u, URGENCIES) for u in df["urgencia"]],
                self.urg.predict(df["text"]), URGENCIES),
            "problemas": problems_metrics(problems_matrix(records), self.prob.predict(df["text"])),
        }
