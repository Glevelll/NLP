import pytest

from backend.core import metrics as m
from backend.data.embeddings import EmbeddingModel
from backend.evaluation.metrics_calculator import MetricsCalculator
from backend.evaluation.significance import bootstrap_ci, compare_systems, paired_bootstrap, paired_ttest

REF = "Термокружка держит тепло восемь часов благодаря двойным стенкам."
GOOD = "Термокружка держит тепло восемь часов благодаря двойным стенкам."
CLOSE = "Термокружка сохраняет тепло восемь часов из-за двойных стенок."
BAD = "Кроссовки для бега по асфальту."


def test_identical_texts_score_max():
    assert m.bleu(GOOD, REF) == pytest.approx(1.0)
    assert m.chrf_pp(GOOD, REF) == pytest.approx(1.0)
    assert m.rouge_l(GOOD, REF) == pytest.approx(1.0)
    assert m.meteor(GOOD, REF) > 0.95
    assert m.exact_match(GOOD + "  ", REF) == 1.0


def test_metrics_are_ordered():
    for fn in (m.bleu, m.chrf_pp, m.rouge_l, m.meteor):
        assert fn(GOOD, REF) > fn(CLOSE, REF) > fn(BAD, REF)
    assert m.rouge_l("", REF) == 0.0 and m.meteor("", REF) == 0.0
    assert m.meteor("абв", "где") == 0.0
    assert m.exact_match(CLOSE, REF) == 0.0
    assert 0 < m.corpus_bleu([CLOSE], [REF]) < 1


def test_meteor_uses_stems_and_fragmentation():
    exact = m.meteor("двойные стенки держат тепло", "двойные стенки держат тепло")
    forms = m.meteor("двойными стенками держит тепло", "двойные стенки держат тепло")
    shuffled = m.meteor("тепло держат стенки двойные", "двойные стенки держат тепло")
    assert exact > forms > 0 and exact > shuffled


def test_keyword_coverage_and_sentiment():
    assert m.keyword_coverage("Двойные стенки и герметичная крышка", ["двойные стенки", "держит тепло"]) == 0.5
    assert m.keyword_coverage("текст", []) == 1.0
    assert m.sentiment("Удобный, надёжный и стильный") > 0.7
    assert m.sentiment("Плохой и неудобный, брак") < 0.3
    assert m.sentiment("Стол") == 0.5


def test_bertscore_embedding_fallback():
    emb = EmbeddingModel("hashing", "hashing", 64)
    scores, backend = m.bertscore([GOOD, BAD], [REF, REF], backend="embedding", embedder=emb)
    assert backend.startswith("embedding") and scores[0] > scores[1]
    # нет библиотеки (ImportError), нет модели в хабе (OSError) или bert-score не знает модель (KeyError)
    with pytest.raises((ImportError, OSError, ValueError, KeyError)):
        m.bertscore([GOOD], [REF], backend="bert-score", model_type="definitely/missing-model")


def test_style_classifier():
    clf = m.StyleClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(["x"])
    texts = ["Вау, огонь, лови скорее!"] * 5 + ["Безупречное качество и изысканные материалы."] * 5
    labels = ["молодёжный"] * 5 + ["премиальный"] * 5
    clf.fit(texts, labels)
    assert clf.fitted and clf.predict(["лови, это огонь"]) == ["молодёжный"]


def test_metrics_calculator():
    calc = MetricsCalculator("embedding", embedder=EmbeddingModel("hashing", "hashing", 64))
    calc.fit_style_classifier(["а а а"], ["один"])  # один класс — классификатор не обучается
    assert not calc.style_classifier.fitted
    calc.fit_style_classifier(["Вау, огонь"] * 3 + ["Изысканное качество"] * 3, ["молодёжный"] * 3 + ["премиальный"] * 3)
    res = calc.compute([GOOD, BAD], [REF, REF], [["двойные стенки"], ["двойные стенки"]], ["премиальный"] * 2)
    assert set(res["metrics"]) >= {"bleu", "chrf_pp", "rouge_l", "meteor", "bertscore", "keyword_coverage",
                                   "style_accuracy", "composite", "corpus_bleu", "avg_length"}
    assert res["per_sample"]["keyword_coverage"] == [1.0, 0.0]
    with pytest.raises(ValueError):
        calc.compute([GOOD], [])
    assert calc.compute([], [])["metrics"]["bleu"] == 0.0


def test_significance():
    lo, hi = bootstrap_ci([0.5, 0.6, 0.7, 0.8], n_resamples=200)
    assert 0.5 <= lo <= hi <= 0.8
    assert bootstrap_ci([]) == (0.0, 0.0)
    better, worse = [0.9, 0.8, 0.85, 0.95, 0.9], [0.1, 0.2, 0.15, 0.1, 0.2]
    assert paired_bootstrap(better, worse) < 0.05 and paired_bootstrap([], []) == 1.0
    assert paired_ttest(better, worse)["p_value"] < 0.01
    assert paired_ttest([1.0], [1.0]) == {"t": 0.0, "p_value": 1.0}
    res = compare_systems({"base": {"bleu": worse}, "new": {"bleu": better}}, "bleu", "base")
    assert res["new"]["delta"] > 0.5 and "ttest_p_value" in res["new"]
