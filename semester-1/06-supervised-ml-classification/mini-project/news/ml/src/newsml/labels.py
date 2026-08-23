"""Weak labels, their provenance, and how disagreement between them is resolved.

Provenance is load-bearing: every label records where it came from, so a feed
section, a publisher's category and a human decision stay distinguishable after
the fact and can be measured against one another.

Two signals are available before any human looks at an article: the section feed
it arrived on, and the publisher's own categories. Where they agree the label is
accepted; where they disagree the article goes to review. That concentrates human
effort on the cases that change an outcome rather than spreading it over articles
two sources already agree on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .load import Article


class LabelSource(StrEnum):
    """Where a label came from. Ordered by how much it should be trusted."""

    HUMAN = "human"
    FEED = "feed"
    CATEGORY = "category"


# A human decision is final. A feed section is structural and near-deterministic.
# A publisher's own category is the noisiest of the three, since publishers
# fragment and reuse their vocabulary freely.
_PRIORITY = {
    LabelSource.HUMAN: 0,
    LabelSource.FEED: 1,
    LabelSource.CATEGORY: 2,
}


@dataclass(frozen=True, slots=True)
class TopicClass:
    id: str
    description: str
    excludes: str = ""
    iptc: str = ""
    merges_into: str = ""


@dataclass(frozen=True, slots=True)
class Label:
    article_id: str
    topic: str
    source: LabelSource
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Resolved:
    """The outcome for one article, and whether a human still needs to see it."""

    article_id: str
    topic: str
    source: LabelSource | None
    agreement: bool
    needs_review: bool
    candidates: tuple[Label, ...]


@dataclass(frozen=True, slots=True)
class Taxonomy:
    version: int
    unsorted: str
    classes: tuple[TopicClass, ...]
    feed_topics: dict[str, str]
    category_map: dict[str, str]
    geography: frozenset[str]
    non_topical: frozenset[str]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(c.id for c in self.classes)

    def collapse(self, topic: str, starved: frozenset[str]) -> str:
        """Fold a starved class into its fallback parent, repeatedly if needed.

        Applied at training time only. Labels keep the class they were assigned,
        so widening the corpus later restores the finer taxonomy without anyone
        relabelling anything.
        """
        by_id = {c.id: c for c in self.classes}
        seen: set[str] = set()
        while topic in starved and topic not in seen:
            seen.add(topic)
            parent = by_id[topic].merges_into if topic in by_id else ""
            if not parent or parent == topic:
                return topic
            topic = parent
        return topic

    def canonical(self, value: str) -> str | None:
        """Return the class id if `value` names one, else None.

        Every externally produced label passes through here. A spreadsheet
        round-trip capitalises, autocompletes and substitutes values freely, so
        nothing becomes a label without matching the allow-list.
        """
        candidate = (value or "").strip().casefold().replace("-", "_").replace(" ", "_")
        return candidate if candidate in self.ids else None


def load_taxonomy(path: Path) -> Taxonomy:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Taxonomy(
        version=int(raw["version"]),
        unsorted=str(raw.get("unsorted", "unsorted")),
        classes=tuple(
            TopicClass(id=str(c["id"]), description=str(c.get("description", "")).strip(),
                       excludes=str(c.get("excludes", "")).strip(),
                       iptc=str(c.get("iptc", "")).strip(),
                       merges_into=str(c.get("merges_into", "")).strip())
            for c in raw["classes"]
        ),
        feed_topics={str(k): str(v) for k, v in (raw.get("feed_topics") or {}).items()},
        category_map={str(k).casefold(): str(v) for k, v in (raw.get("category_map") or {}).items()},
        geography=frozenset(str(g).casefold() for g in (raw.get("geography") or [])),
        non_topical=frozenset(str(n).casefold() for n in (raw.get("non_topical") or [])),
    )


def from_feed(article: Article, taxonomy: Taxonomy) -> Label | None:
    """The section a feed names applies to every article that arrives on it."""
    topic = taxonomy.feed_topics.get(article.source_name)
    if topic is None or taxonomy.canonical(topic) is None:
        return None
    return Label(article.id, topic, LabelSource.FEED, article.source_name)


def from_categories(article: Article, taxonomy: Taxonomy) -> Label | None:
    """Map the publisher's own categories, when any of them names a topic.

    Sorted before the first match so the result never depends on the order the
    publisher happened to emit its categories in.
    """
    for raw in sorted(c.casefold() for c in article.categories):
        if (topic := taxonomy.category_map.get(raw)) and taxonomy.canonical(topic):
            return Label(article.id, topic, LabelSource.CATEGORY, raw)
    return None


def is_geography_only(article: Article, taxonomy: Taxonomy) -> bool:
    """True when an article carries categories but none of them names a topic."""
    values = {c.casefold() for c in article.categories}
    return bool(values) and all(v in taxonomy.geography or v in taxonomy.non_topical for v in values)


def resolve(article_id: str, candidates: list[Label], taxonomy: Taxonomy) -> Resolved:
    """Reconcile every signal for one article into a single decision."""
    if not candidates:
        return Resolved(article_id, taxonomy.unsorted, None, False, True, ())

    ordered = tuple(sorted(candidates, key=lambda c: (_PRIORITY[c.source], c.source, c.topic)))

    human = next((c for c in ordered if c.source is LabelSource.HUMAN), None)
    if human is not None:
        return Resolved(article_id, human.topic, LabelSource.HUMAN, True, False, ordered)

    best = ordered[0]
    topics = {c.topic for c in ordered}

    # Two independent signals concurring is the whole point: it is what lets most
    # of the corpus skip review without anyone having read it.
    if len(ordered) > 1 and len(topics) == 1:
        return Resolved(article_id, best.topic, best.source, True, False, ordered)

    if len(topics) > 1:
        return Resolved(article_id, best.topic, best.source, False, True, ordered)

    # A lone feed label is structural and stands on its own. A lone category
    # label is one unverified publisher opinion, so it waits for a human.
    solo_is_trusted = best.source is LabelSource.FEED
    return Resolved(article_id, best.topic, best.source, False, not solo_is_trusted, ordered)
