"""Токенизация, нормализация и стемминг русского текста."""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

try:
    import snowballstemmer

    _STEMMER = snowballstemmer.stemmer("russian")
except Exception:  # pragma: no cover - запасной вариант без библиотеки
    _STEMMER = None

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+|\n+")
URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")

STOPWORDS: frozenset[str] = frozenset(
    """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее её мне было вот
    от меня еще ещё нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять
    уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
    будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти
    мой тем чтобы нее неё сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над
    больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда
    лучше чуть том нельзя такой им более всегда конечно всю между это также который которая которые которое
    ваш ваша ваши каждый каждая""".split()
)


def normalize(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


@lru_cache(maxsize=50_000)
def stem(word: str) -> str:
    word = word.lower().replace("ё", "е")
    if not word.isalpha() or len(word) <= 3:
        return word
    if _STEMMER is not None:
        return _STEMMER.stemWord(word)
    return word[:6]  # pragma: no cover


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def stems(text: str) -> list[str]:
    return [stem(t) for t in content_tokens(text)]


def split_sentences(text: str) -> list[str]:
    parts = (s.strip(" •-—\t") for s in SENTENCE_RE.split(text or "") if s)
    return [p for p in parts if p]


def feature_coverage(feature: str, text: str, threshold: float = 0.6) -> tuple[bool, float]:
    """Проверяет, упомянута ли особенность товара в тексте (по совпадению основ слов)."""
    feat_stems = [s for s in stems(feature) if len(s) > 1]
    if not feat_stems:
        return True, 1.0
    text_stems = set(stems(text))
    text_prefixes = {s[:5] for s in text_stems if len(s) >= 5}
    hits = sum(1 for s in feat_stems if s in text_stems or (len(s) >= 5 and s[:5] in text_prefixes))
    ratio = hits / len(feat_stems)
    return ratio >= threshold, ratio


def repeated_sentences(text: str) -> list[str]:
    counts = Counter(normalize(s) for s in split_sentences(text) if len(s) > 15)
    return [s for s, c in counts.items() if c > 1]


def overused_words(text: str, max_share: float = 0.06, min_count: int = 5) -> list[str]:
    toks = [stem(t) for t in content_tokens(text) if len(t) > 3 and t.isalpha()]
    if not toks:
        return []
    counts = Counter(toks)
    return [w for w, c in counts.items() if c >= min_count and c / len(toks) > max_share]


def contains_phrase(text: str, phrase: str) -> bool:
    """Проверка вхождения фразы с учётом словоформ (по основам, подряд)."""
    p = [stem(t) for t in tokenize(phrase)]
    if not p:
        return False
    t = [stem(x) for x in tokenize(text)]
    n = len(p)
    return any(t[i : i + n] == p for i in range(len(t) - n + 1))


def char_ngrams(text: str, n: int = 3) -> list[str]:
    t = f" {normalize(text)} "
    return [t[i : i + n] for i in range(max(len(t) - n + 1, 0))]
