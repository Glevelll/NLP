"""Расчёт набора метрик по выборке: средние значения и значения по каждому образцу."""
from __future__ import annotations

from typing import Any

import numpy as np

from backend.core import metrics as m

TEXT_METRICS = ["bleu", "chrf_pp", "rouge_l", "meteor", "bertscore", "exact_match"]
TASK_METRICS = ["keyword_coverage", "sentiment", "style_accuracy"]
COMPOSITE_KEYS = ["bleu", "chrf_pp", "rouge_l", "meteor", "bertscore", "keyword_coverage"]


class MetricsCalculator:
    def __init__(self, bertscore_backend: str = "auto", bertscore_model: str = "bert-base-multilingual-cased",
                 embedder=None, style_classifier: m.StyleClassifier | None = None):
        self.bertscore_backend = bertscore_backend
        self.bertscore_model = bertscore_model
        self.embedder = embedder
        self.style_classifier = style_classifier or m.StyleClassifier()

    def fit_style_classifier(self, texts: list[str], styles: list[str]) -> None:
        if len(set(styles)) >= 2:
            self.style_classifier.fit(texts, styles)

    def compute(self, predictions: list[str], references: list[str], features: list[list[str]] | None = None,
                styles: list[str] | None = None) -> dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions и references должны быть одной длины")
        n = len(predictions)
        per: dict[str, list[float]] = {
            "bleu": [m.bleu(p, r) for p, r in zip(predictions, references, strict=True)],
            "chrf_pp": [m.chrf_pp(p, r) for p, r in zip(predictions, references, strict=True)],
            "rouge_l": [m.rouge_l(p, r) for p, r in zip(predictions, references, strict=True)],
            "meteor": [m.meteor(p, r) for p, r in zip(predictions, references, strict=True)],
            "exact_match": [m.exact_match(p, r) for p, r in zip(predictions, references, strict=True)],
            "sentiment": [m.sentiment(p) for p in predictions],
        }
        bs, backend = m.bertscore(predictions, references, self.bertscore_backend, self.bertscore_model,
                                  self.embedder) if n else ([], "none")
        per["bertscore"] = bs
        if features is not None:
            per["keyword_coverage"] = [m.keyword_coverage(p, f) for p, f in zip(predictions, features, strict=True)]
        if styles is not None and self.style_classifier.fitted:
            pred_styles = self.style_classifier.predict(predictions)
            per["style_accuracy"] = [float(a == b) for a, b in zip(pred_styles, styles, strict=True)]

        summary = {k: round(float(np.mean(v)), 4) if v else 0.0 for k, v in per.items()}
        summary["corpus_bleu"] = round(m.corpus_bleu(predictions, references), 4) if n else 0.0
        comp = [summary[k] for k in COMPOSITE_KEYS if k in summary]
        summary["composite"] = round(float(np.mean(comp)), 4) if comp else 0.0
        summary["avg_length"] = round(float(np.mean([len(p) for p in predictions])), 1) if n else 0.0
        return {"metrics": summary, "per_sample": per, "bertscore_backend": backend}
