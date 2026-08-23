"""Taxonomy integrity and weak-label resolution."""

from __future__ import annotations

import pytest

from newsml.labels import LabelSource, from_categories, from_feed, is_geography_only, resolve


def test_taxonomy_has_the_agreed_thirteen_classes(taxonomy):
    assert len(taxonomy.classes) == 13


def test_every_class_is_documented(taxonomy):
    """Descriptions are what the labelling guide is generated from, so a blank
    one silently ships an unanswerable question to the annotators."""
    for topic in taxonomy.classes:
        assert topic.description, f"{topic.id} has no description"
        assert topic.iptc, f"{topic.id} has no IPTC parent"


def test_class_ids_are_unique(taxonomy):
    ids = [c.id for c in taxonomy.classes]
    assert len(ids) == len(set(ids))


def test_merge_targets_are_real_classes(taxonomy):
    for topic in taxonomy.classes:
        if topic.merges_into:
            assert topic.merges_into in taxonomy.ids, f"{topic.id} merges into unknown {topic.merges_into}"


def test_merge_chains_terminate(taxonomy):
    """A merges_into cycle would hang collapse(). Assert the graph is acyclic."""
    for topic in taxonomy.classes:
        seen, current = set(), topic.id
        while current:
            assert current not in seen, f"merge cycle through {current}"
            seen.add(current)
            current = next((c.merges_into for c in taxonomy.classes if c.id == current), "")


def test_every_mapped_category_points_at_a_real_class(taxonomy):
    for value, topic in taxonomy.category_map.items():
        assert topic in taxonomy.ids, f"category {value!r} maps to unknown class {topic!r}"


def test_every_feed_topic_points_at_a_real_class(taxonomy):
    for source, topic in taxonomy.feed_topics.items():
        assert topic in taxonomy.ids, f"feed {source!r} maps to unknown class {topic!r}"


def test_geography_and_categories_do_not_overlap(taxonomy):
    """A value cannot be both a place and a topic; if it were, the label would
    depend on which lookup ran first."""
    assert not (taxonomy.geography & set(taxonomy.category_map))


def test_non_topical_and_categories_do_not_overlap(taxonomy):
    """Regression: `gear / deals` was once both a technology label and affiliate
    content, so deals copy would have trained the technology class."""
    assert not (taxonomy.non_topical & set(taxonomy.category_map))


def test_collapse_folds_a_starved_class_into_its_parent(taxonomy):
    assert taxonomy.collapse("science_space", frozenset({"science_space"})) == "technology"


def test_collapse_follows_a_chain(taxonomy):
    """health -> science_space -> technology when both are starved."""
    assert taxonomy.collapse("health", frozenset({"health", "science_space"})) == "technology"


def test_collapse_leaves_healthy_classes_alone(taxonomy):
    assert taxonomy.collapse("health", frozenset()) == "health"


def test_collapse_stops_when_no_parent_exists(taxonomy):
    assert taxonomy.collapse("sport", frozenset({"sport"})) == "sport"


@pytest.mark.parametrize("raw", ["technology", "TECHNOLOGY", " Technology ", "technology"])
def test_canonical_accepts_case_and_whitespace_variants(taxonomy, raw):
    assert taxonomy.canonical(raw) == "technology"


@pytest.mark.parametrize("raw", ["", "  ", "tech", "not_a_class", "'; DROP TABLE", "unsorted"])
def test_canonical_rejects_anything_not_a_class(taxonomy, raw):
    assert taxonomy.canonical(raw) is None


def test_feed_label_uses_the_declared_section(taxonomy, make_article):
    label = from_feed(make_article(source_name="Wired"), taxonomy)
    assert label is not None and label.topic == "technology"
    assert label.source is LabelSource.FEED


def test_unknown_source_has_no_feed_label(taxonomy, make_article):
    assert from_feed(make_article(source_name="Some Blog"), taxonomy) is None


def test_category_label_is_independent_of_publisher_ordering(taxonomy, make_article):
    forward = from_categories(make_article(categories=("ai", "india")), taxonomy)
    reverse = from_categories(make_article(categories=("india", "ai")), taxonomy)
    assert forward is not None and forward.topic == reverse.topic == "technology"


def test_geography_only_article_yields_no_label(taxonomy, make_article):
    article = make_article(categories=("india", "karnataka", "bengaluru"))
    assert from_categories(article, taxonomy) is None
    assert is_geography_only(article, taxonomy)


def test_agreeing_signals_skip_review(taxonomy, make_article):
    article = make_article(source_name="Wired", categories=("ai",))
    candidates = [c for c in (from_feed(article, taxonomy), from_categories(article, taxonomy)) if c]
    outcome = resolve(article.id, candidates, taxonomy)

    assert outcome.topic == "technology"
    assert outcome.agreement and not outcome.needs_review


def test_conflicting_signals_route_to_review(taxonomy, make_article):
    article = make_article(source_name="Wired", categories=("cricket",))
    candidates = [c for c in (from_feed(article, taxonomy), from_categories(article, taxonomy)) if c]
    outcome = resolve(article.id, candidates, taxonomy)

    assert not outcome.agreement and outcome.needs_review


def test_a_lone_category_waits_for_a_human(taxonomy, make_article):
    article = make_article(source_name="Some Blog", categories=("cricket",))
    outcome = resolve(article.id, [from_categories(article, taxonomy)], taxonomy)
    assert outcome.topic == "sport" and outcome.needs_review


def test_no_signal_is_unsorted_and_needs_review(taxonomy, make_article):
    outcome = resolve("x", [], taxonomy)
    assert outcome.topic == taxonomy.unsorted and outcome.needs_review


def test_a_human_label_overrides_every_other_signal(taxonomy, make_article):
    from newsml.labels import Label

    article = make_article(source_name="Wired", categories=("ai",))
    candidates = [
        from_feed(article, taxonomy),
        from_categories(article, taxonomy),
        Label(article.id, "politics", LabelSource.HUMAN, "riaz"),
    ]
    outcome = resolve(article.id, [c for c in candidates if c], taxonomy)

    assert outcome.topic == "politics"
    assert outcome.source is LabelSource.HUMAN and not outcome.needs_review
