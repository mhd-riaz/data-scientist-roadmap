"""The dataset contract, and the holdout arrangement the honest number rests on."""

from __future__ import annotations

import pytest

from newsml import dataset


def _corpus(make_article, spec):
    """`spec` is (article_id, category, title) triples."""
    return [
        make_article(article_id, categories=(category,), title=title, minutes=index * 90)
        for index, (article_id, category, title) in enumerate(spec)
    ]


SPEC = [
    ("s1", "sport", "India beat Australia by six wickets in the opening Test at Galle"),
    ("s2", "sport", "City sign a midfielder from Porto for a reported forty million"),
    ("s3", "sport", "Marathon world record falls again in Berlin on a fast morning"),
    ("h1", "health", "Trial finds the new vaccine cuts recurrence of melanoma sharply"),
    ("h2", "health", "Hospitals report a rise in seasonal influenza admissions this week"),
    ("h3", "health", "Researchers link poor sleep to higher blood pressure in adults"),
]


def test_a_held_out_article_is_scored_but_never_trained_on(make_article, taxonomy):
    articles = _corpus(make_article, SPEC)
    data = dataset.build(articles, taxonomy, min_per_class=1, holdout={"s1": "sport"})

    trained = {e.article_id for e in (*data.train, *data.val, *data.test)}
    assert "s1" not in trained
    assert [e.article_id for e in data.gold] == ["s1"]
    assert data.gold[0].label_source == "human"


def test_the_whole_story_group_of_a_held_out_article_is_withheld(make_article, taxonomy):
    twin = "India beat Australia by six wickets in the opening Test at Galle"
    articles = _corpus(make_article, [*SPEC, ("s1b", "sport", twin)])
    data = dataset.build(articles, taxonomy, min_per_class=1, holdout={"s1": "sport"})

    trained = {e.article_id for e in (*data.train, *data.val, *data.test)}
    assert {"s1", "s1b"}.isdisjoint(trained), "a near-duplicate of a scored article leaked into training"
    assert data.withheld_for_gold == 1


def test_a_gold_class_the_weak_labels_never_taught_is_reported_as_unreachable(make_article, taxonomy):
    articles = _corpus(make_article, SPEC)
    data = dataset.build(articles, taxonomy, min_per_class=1, holdout={"s1": "conflict_war"})

    assert "conflict_war" not in data.classes
    assert data.unreachable == {"conflict_war": 1}


def test_gold_labels_collapse_to_the_group_the_model_predicts(make_article, taxonomy):
    articles = _corpus(make_article, SPEC)
    data = dataset.build(articles, taxonomy, min_per_class=1, holdout={"h1": "judiciary_courts"})

    assert data.gold[0].topic == "crime_justice"


def test_an_unsorted_gold_label_is_not_scored(make_article, taxonomy):
    """It is the absence of a class; scoring it adds a guaranteed zero."""
    articles = _corpus(make_article, SPEC)
    data = dataset.build(articles, taxonomy, min_per_class=1, holdout={"s1": "unsorted"})

    assert data.gold == ()
    assert data.withheld_for_gold == 1


def test_one_article_cannot_both_teach_and_score(make_article, taxonomy):
    articles = _corpus(make_article, SPEC)
    with pytest.raises(ValueError, match="both gold and holdout"):
        dataset.build(articles, taxonomy, gold={"s1": "sport"}, holdout={"s1": "sport"})


def test_disjoint_human_labels_may_teach_and_score_in_one_build(make_article, taxonomy):
    """The shipped arrangement: rare classes are taught by one slice, scored by another."""
    articles = _corpus(make_article, SPEC)
    data = dataset.build(
        articles,
        taxonomy,
        min_per_class=1,
        gold={"h1": "conflict_war", "h2": "conflict_war"},
        holdout={"s1": "conflict_war"},
    )

    taught = {e.article_id: e.topic for e in data.train}
    assert taught.get("h1") == "conflict_war", "the human label did not override the weak one"
    assert "conflict_war" in data.classes, "a class only humans can teach never reached the class list"
    assert [e.article_id for e in data.gold] == ["s1"]


def test_without_a_holdout_nothing_changes(make_article, taxonomy):
    articles = _corpus(make_article, SPEC)
    data = dataset.build(articles, taxonomy, min_per_class=1)

    assert data.gold == () and data.withheld_for_gold == 0
    assert len(data.train) + len(data.val) + len(data.test) == len(articles) - data.dropped_at_boundary


def test_child_classes_survive_when_the_collapse_is_turned_off(make_article, taxonomy):
    """Only worth doing once enough children have been hand-labelled."""
    articles = _corpus(make_article, SPEC)
    gold = {"h1": "judiciary_courts", "h2": "crime"}

    collapsed = dataset.build(articles, taxonomy, min_per_class=1, gold=gold)
    detailed = dataset.build(
        articles, taxonomy, min_per_class=1, gold=gold, collapse_to_group=False
    )

    assert "crime_justice" in collapsed.classes
    assert {"judiciary_courts", "crime"} <= set(detailed.classes)
    assert "crime_justice" not in detailed.classes


def test_dropping_weak_labels_leaves_only_the_hand_labelled_articles(make_article, taxonomy):
    articles = _corpus(make_article, SPEC)
    gold = {"s1": "sport", "s2": "sport"}

    data = dataset.build(
        articles, taxonomy, min_per_class=1, gold=gold, use_weak_labels=False
    )

    kept = {e.article_id for e in (*data.train, *data.val, *data.test)}
    assert kept == set(gold), f"an article with no human label was trained on: {kept - set(gold)}"
    assert all(e.label_source == "human" for e in data.train)
    assert data.unlabelled == len(articles) - len(gold)
