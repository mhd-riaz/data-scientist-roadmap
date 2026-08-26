"""The candidate models, as a registry rather than a hardcoded ladder.

Nothing here is chosen yet. Phase C establishes baselines, Phase E adds the tree and
embedding families, and the winner is whichever has the best interval -- with ties
broken toward the simpler model, because a tie means the complexity bought nothing.
"""

from __future__ import annotations

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import SEED


class _Head(BaseEstimator, TransformerMixin):
    """Keep the first n characters. Char n-grams over a full body are not affordable."""

    def __init__(self, chars: int = 600):
        self.chars = chars

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [x[: self.chars] for x in X]

# Shared so every rung sees the same tokens and only the estimator differs.
_WORD = dict(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.6,
    sublinear_tf=True,
)
_CHAR = dict(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.6,
    sublinear_tf=True,
    lowercase=True,
)


def _majority(**_) -> Pipeline:
    return Pipeline([("clf", DummyClassifier(strategy="prior", random_state=SEED))])


def _complement_nb(**kw) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**{**_WORD, **kw.get("vectoriser", {})})),
        ("clf", ComplementNB()),
    ])


def _tfidf_logreg(**kw) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**{**_WORD, **kw.get("vectoriser", {})})),
        ("clf", LogisticRegression(
            C=kw.get("C", 5.0),
            max_iter=2000,
            class_weight=kw.get("class_weight", "balanced"),
            random_state=SEED,
        )),
    ])


def _tfidf_linear_svc(**kw) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**{**_WORD, **kw.get("vectoriser", {})})),
        ("clf", LinearSVC(
            C=kw.get("C", 1.0),
            class_weight=kw.get("class_weight", "balanced"),
            random_state=SEED,
        )),
    ])


def _word_char_svc(**kw) -> Pipeline:
    """Word and character n-grams together.

    Character n-grams buy robustness to spelling and transliteration, which matters in a
    corpus mixing Indian and international mastheads. They are applied to the **first
    600 characters only**: char 3-5 grams over a full 4,000-character body generate tens
    of millions of features and dominate fitting time for signal the word features
    already carry. The headline and lede are where spelling variation actually matters.
    """
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(**{**_WORD, **kw.get("vectoriser", {})})),
            ("char", Pipeline([
                ("head", _Head(kw.get("char_chars", 600))),
                ("tfidf", TfidfVectorizer(**_CHAR)),
            ])),
        ])),
        ("clf", LinearSVC(
            C=kw.get("C", 1.0),
            class_weight=kw.get("class_weight", "balanced"),
            random_state=SEED,
        )),
    ])


REGISTRY: dict[str, Callable[..., Pipeline]] = {
    "majority": _majority,
    "complement_nb": _complement_nb,
    "tfidf_logreg": _tfidf_logreg,
    "tfidf_linear_svc": _tfidf_linear_svc,
    "word_char_svc": _word_char_svc,
}


def build(name: str, **kw) -> Pipeline:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
