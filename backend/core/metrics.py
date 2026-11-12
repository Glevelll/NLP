"""Метрики качества генерации.

Все значения нормированы в диапазон 0..1 (BLEU и chrF++ делятся на 100), чтобы их можно было сравнивать
на одном графике. BLEU и chrF++ считаются через sacrebleu; ROUGE-L и METEOR реализованы здесь, потому что
стандартные реализации (rouge-score, nltk) рассчитаны на латиницу и английскую WordNet-синонимию.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

from backend.core.constants import NEGATIVE_WORDS, POSITIVE_WORDS
from backend.utils.text_processing import feature_coverage, normalize, stem, tokenize

logger = logging.getLogger(__name__)

_BLEU = BLEU(effective_order=True)
_CHRF_PP = CHRF(word_order=2)


# ------------------------- n-граммные метрики -------------------------
def bleu(prediction: str, reference: str) -> float:
    """Sentence BLEU (sacrebleu, токенизатор 13a, сглаживание exp)."""
    return _BLEU.sentence_score(prediction, [reference]).score / 100


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    return BLEU().corpus_score(predictions, [references]).score / 100


def chrf_pp(prediction: str, reference: str) -> float:
    """chrF++: F-мера по символьным n-граммам (n<=6) и словным биграммам."""
    return _CHRF_PP.sentence_score(prediction, [reference]).score / 100


def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str, beta: float = 1.2) -> float:
    """ROUGE-L F-мера по наибольшей общей подпоследовательности основ слов."""
    p = [stem(t) for t in tokenize(prediction)]
    r = [stem(t) for t in tokenize(reference)]
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return (1 + beta**2) * prec * rec / (rec + beta**2 * prec)


def meteor(prediction: str, reference: str, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5) -> float:
    """METEOR (Banerjee & Lavie): выравнивание по точному совпадению, затем по основе слова;
    Fmean с перевесом полноты и штраф за фрагментацию (число «кусков» выравнивания)."""
    hyp, ref = tokenize(prediction), tokenize(reference)
    if not hyp or not ref:
        return 0.0
    used_ref: set[int] = set()
    align: dict[int, int] = {}
    for matcher in (lambda w: w, stem):
        ref_index: dict[str, list[int]] = {}
        for j, w in enumerate(ref):
            if j not in used_ref:
                ref_index.setdefault(matcher(w), []).append(j)
        for i, w in enumerate(hyp):
            if i in align:
                continue
            cands = ref_index.get(matcher(w))
            if cands:
                # выбираем ближайшую по позиции свободную позицию — меньше фрагментация
                j = min(cands, key=lambda c: abs(c - i * len(ref) / len(hyp)))
                cands.remove(j)
                align[i] = j
                used_ref.add(j)
    m = len(align)
    if m == 0:
        return 0.0
    precision, recall = m / len(hyp), m / len(ref)
    fmean = precision * recall / (alpha * precision + (1 - alpha) * recall)
    pairs = sorted(align.items())
    chunks = 1 + sum(1 for (i1, j1), (i2, j2) in zip(pairs, pairs[1:], strict=False) if not (i2 == i1 + 1 and j2 == j1 + 1))
    penalty = gamma * (chunks / m) ** beta
    return fmean * (1 - penalty)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


# ------------------------- семантика -------------------------
@lru_cache(maxsize=1)
def _bert_scorer(model_type: str):
    from bert_score import BERTScorer

    return BERTScorer(model_type=model_type, lang="ru", rescale_with_baseline=False)


def bertscore(predictions: list[str], references: list[str], backend: str = "auto",
              model_type: str = "bert-base-multilingual-cased", embedder=None) -> tuple[list[float], str]:
    """BERTScore F1. Если библиотека bert-score/torch недоступна — косинусная близость эмбеддингов
    предложений (backend='embedding'), что указывается во втором элементе результата."""
    if backend in ("auto", "bert-score"):
        try:
            scorer = _bert_scorer(model_type)
            _, _, f1 = scorer.score(predictions, references)
            return [float(x) for x in f1], "bert-score"
        except Exception as exc:
            if backend == "bert-score":
                raise
            logger.info("bert-score недоступен (%s) — используется embedding-близость", exc)
    if embedder is None:
        from backend.data.embeddings import EmbeddingModel

        embedder = EmbeddingModel.get_instance()
    a, b = embedder.encode(predictions), embedder.encode(references)
    sims = np.sum(a * b, axis=1)
    return [float(max(0.0, s)) for s in sims], f"embedding:{embedder.name}"


# ------------------------- специфические для задачи -------------------------
def keyword_coverage(prediction: str, features: list[str]) -> float:
    """Доля ключевых особенностей товара, упомянутых в описании."""
    if not features:
        return 1.0
    return sum(feature_coverage(f, prediction)[0] for f in features) / len(features)


def sentiment(text: str) -> float:
    """Лексиконная тональность: доля позитивных маркеров среди всех оценочных (0.5 — нейтрально)."""
    toks = tokenize(text)
    pos = sum(1 for t in toks if any(t.startswith(w) for w in POSITIVE_WORDS))
    neg = sum(1 for t in toks if any(t.startswith(w) for w in NEGATIVE_WORDS))
    return (pos + 1) / (pos + neg + 2)


class StyleClassifier:
    """TF-IDF (символьные n-граммы) + логистическая регрессия: предсказывает стиль описания."""

    def __init__(self):
        self.model = None
        self.labels: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> StyleClassifier:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, max_features=50000),
            LogisticRegression(max_iter=2000, C=4.0),
        )
        self.model.fit(texts, labels)
        self.labels = sorted(set(labels))
        return self

    @property
    def fitted(self) -> bool:
        return self.model is not None

    def predict(self, texts: list[str]) -> list[str]:
        if self.model is None:
            raise RuntimeError("Классификатор стиля не обучен")
        return list(self.model.predict(texts))
