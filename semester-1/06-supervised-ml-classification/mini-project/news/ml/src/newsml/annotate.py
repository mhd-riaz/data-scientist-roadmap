"""Export articles for blind human labelling, and read the completed sheets back.

The gold set is the only thing every reported number rests on, so two properties
matter more than convenience:

* **Blind.** The sheet carries the title and summary and nothing else. No source,
  no URL, no proposed label. A labeller who sees "Wired" types `technology`
  without reading, and the feed URLs literally contain the section name, so
  either would leak the weak label into the gold set and turn the agreement
  study into a measurement of itself.
* **Reproducible.** The sample is drawn with a fixed seed from a fixed corpus, so
  the same command produces the same sheet.

One article per story group: labelling three copies of the same wire story costs
three times as much and yields one label's worth of information.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .labels import Label, LabelSource, Taxonomy
from .load import Article

COLUMNS = ("article_id", "title", "summary", "label", "notes")

# Excel guesses the encoding of a plain UTF-8 CSV and guesses wrong, mangling
# every non-ASCII character. The BOM is what stops that, and Sheets ignores it.
ENCODING = "utf-8-sig"


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a returned sheet, located precisely enough to fix."""

    sheet: str
    row: int
    article_id: str
    detail: str


def _representatives(articles: list[Article], group_of: dict[str, str] | None) -> list[Article]:
    """One article per story group, chosen by smallest id so it never varies."""
    if not group_of:
        return sorted(articles, key=lambda a: a.id)

    best: dict[str, Article] = {}
    for article in sorted(articles, key=lambda a: a.id):
        group = group_of.get(article.id, article.id)
        if group not in best:
            best[group] = article
    return [best[key] for key in sorted(best)]


def choose_sample(
    articles: list[Article],
    *,
    size: int,
    seed: int,
    group_of: dict[str, str] | None = None,
) -> list[Article]:
    """Draw a sample that covers every source rather than mirroring the corpus.

    Round-robin across sources, not proportional allocation. Proportional would
    hand ~17% of the sheet to one publisher and leave the small sources with two
    articles each, which is useless for deciding whether the taxonomy fits. The
    trade-off is that the sample over-represents small sources, so agreement
    measured on it is a per-source figure and not a corpus-wide rate.
    """
    pool = _representatives(articles, group_of)

    by_source: defaultdict[str, list[Article]] = defaultdict(list)
    for article in pool:
        by_source[article.source_id].append(article)

    rng = random.Random(seed)
    queues = {}
    for source_id in sorted(by_source):
        members = list(by_source[source_id])
        rng.shuffle(members)
        queues[source_id] = members

    chosen: list[Article] = []
    order = sorted(queues)
    while len(chosen) < size and any(queues[s] for s in order):
        for source_id in order:
            if not queues[source_id]:
                continue
            chosen.append(queues[source_id].pop())
            if len(chosen) >= size:
                break

    return sorted(chosen, key=lambda a: a.id)


def _cell(value: str) -> str:
    """Collapse whitespace so a multi-line summary stays one spreadsheet row."""
    return " ".join((value or "").split())


def write_sheets(
    articles: list[Article],
    out_dir: Path,
    *,
    shards: int = 1,
    overlap: int = 0,
    seed: int = 0,
) -> list[Path]:
    """Write one CSV per annotator, with an optional shared overlap block.

    The overlap is what makes inter-annotator agreement measurable: the same
    articles appear in every sheet, so disagreement between two people is
    visible. Without it there is no way to tell a hard taxonomy from a careless
    annotator.
    """
    if shards < 1:
        raise ValueError("shards must be at least 1")
    overlap = max(0, min(overlap, len(articles)))

    shared = articles[:overlap]
    rest = articles[overlap:]

    buckets: list[list[Article]] = [[] for _ in range(shards)]
    for index, article in enumerate(rest):
        buckets[index % shards].append(article)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for index, bucket in enumerate(buckets, start=1):
        rows = sorted({a.id: a for a in [*shared, *bucket]}.values(), key=lambda a: a.id)
        # Shuffle so the shared block is not an identifiable run at the top,
        # which would invite treating it differently from the rest.
        random.Random(seed + index).shuffle(rows)

        path = out_dir / (f"labels-{index:02d}.csv" if shards > 1 else "labels.csv")
        with path.open("w", encoding=ENCODING, newline="") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
            writer.writerow(COLUMNS)
            for article in rows:
                writer.writerow([article.id, _cell(article.title), _cell(article.summary), "", ""])
        written.append(path)

    return written


def write_guide(taxonomy: Taxonomy, path: Path, *, overlap: int = 0) -> Path:
    """Generate the annotator's instructions from the taxonomy definitions.

    Generated rather than hand-written so the humans and any other labeller are
    answering the same question, from one source of truth.
    """
    lines = [
        "# Labelling guide",
        "",
        f"Taxonomy version {taxonomy.version}. Generated from `ml/taxonomy.yaml` — do not edit by hand.",
        "",
        "Fill the **`label`** column with exactly one id from the table below.",
        "Leave `notes` for anything that felt wrong; it is read, not ignored.",
        "",
        "## How to choose",
        "",
        "The classes come in two levels. Work in two steps:",
        "",
        "1. Pick the **group** the article belongs to — the bold rows.",
        "2. Pick the **specific class** inside that group — the indented rows.",
        "",
        "**Always give the specific class.** A bare group label is a fallback for the rare",
        "article that genuinely spans two of its children: \"Prime minister campaigns while",
        "on a state visit\" is both diplomacy and elections, so it is `politics`. If you find",
        "yourself reaching for group labels often, something is wrong with this guide — say so.",
        "",
        "Groups with no indented rows beneath them are used as they are.",
        "",
        "## Rules",
        "",
        "1. Label what the article is **about**, not where it happened. "
        "A Karnataka election story is `politics_elections`, not a geography.",
        f"2. If it fits none of them, or you genuinely cannot tell, write `{taxonomy.unsorted}`. "
        "A forced guess is worse than an honest blank — it becomes silent noise no one can find later.",
        "3. Judge only from the title and summary shown. Do not search for the article. "
        "That is what the model sees, so that is what the label has to be based on.",
        "4. Pick the **dominant** subject when two apply, and say so in `notes`.",
        "5. Format is not subject. An opinion piece about cricket is still `sport`.",
        "",
        "## Classes",
        "",
        "| id | covers | does not cover |",
        "| --- | --- | --- |",
    ]
    for group in taxonomy.groups:
        children = taxonomy.children_of(group.id)
        note = " *(fallback — only if two below apply equally)*" if children else ""
        lines.append(f"| **`{group.id}`**{note} | {group.description} | {group.excludes or '—'} |")
        for child in children:
            lines.append(f"| ↳ `{child.id}` | {child.description} | {child.excludes or '—'} |")
    lines.append(f"| **`{taxonomy.unsorted}`** | Anything else, or genuinely unclear. | — |")

    if overlap:
        lines += [
            "",
            "## Why some rows repeat across sheets",
            "",
            f"{overlap} articles appear in every sheet on purpose. Comparing how differently "
            "two people labelled the same article is what tells us whether the classes are "
            "well defined. Do not coordinate on them.",
        ]

    lines += [
        "",
        "## Returning the sheet",
        "",
        "Keep the `article_id` column exactly as it is — it is how labels are matched back.",
        "Do not add, remove or reorder rows. Save as CSV, keeping UTF-8 encoding.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_sheet(
    path: Path,
    taxonomy: Taxonomy,
    *,
    known_ids: frozenset[str] | None = None,
    annotator: str = "",
) -> tuple[list[Label], list[Problem]]:
    """Parse a completed sheet, accepting only labels the taxonomy defines.

    A spreadsheet round-trip is lossy in ways that are easy to miss: autocorrect
    capitalises, autocomplete substitutes a neighbouring value, and a stray row
    gets sorted. Everything is checked rather than assumed.
    """
    allowed = taxonomy.ids | {taxonomy.unsorted}
    labels: list[Label] = []
    problems: list[Problem] = []
    seen: set[str] = set()

    with path.open("r", encoding=ENCODING, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("article_id", "label") if c not in (reader.fieldnames or [])]
        if missing:
            return [], [Problem(path.name, 0, "", f"sheet is missing column(s): {', '.join(missing)}")]

        for number, row in enumerate(reader, start=2):
            article_id = (row.get("article_id") or "").strip()
            raw = (row.get("label") or "").strip()

            if not article_id:
                problems.append(Problem(path.name, number, "", "blank article_id"))
                continue
            if article_id in seen:
                problems.append(Problem(path.name, number, article_id, "duplicate row for this article"))
                continue
            seen.add(article_id)

            if known_ids is not None and article_id not in known_ids:
                problems.append(Problem(path.name, number, article_id, "article_id is not in the corpus"))
                continue
            if not raw:
                continue  # not yet labelled; reported as a count, not an error

            resolved = taxonomy.canonical(raw) or (taxonomy.unsorted if raw.casefold() == taxonomy.unsorted else None)
            if resolved is None or resolved not in allowed:
                problems.append(Problem(path.name, number, article_id, f"not a taxonomy class: {raw!r}"))
                continue

            labels.append(Label(article_id, resolved, LabelSource.HUMAN, annotator or path.stem))

    return labels, problems


def disagreements(labels: list[Label]) -> dict[str, set[str]]:
    """Articles two annotators labelled differently, and what they each chose."""
    by_article: defaultdict[str, set[str]] = defaultdict(set)
    for label in labels:
        by_article[label.article_id].add(label.topic)
    return {article: topics for article, topics in by_article.items() if len(topics) > 1}
