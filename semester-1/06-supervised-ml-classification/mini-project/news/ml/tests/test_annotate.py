"""Labelling sheets: blind by construction, and validated on the way back in."""

from __future__ import annotations

import csv

from newsml import annotate
from newsml.annotate import ENCODING


def _articles(make_article, count=40):
    return [
        make_article(f"a{i:03d}", source_id=f"src-{i % 5}", source_name=f"Source {i % 5}", minutes=i * 30)
        for i in range(count)
    ]


def _read(path):
    with path.open("r", encoding=ENCODING, newline="") as handle:
        return list(csv.DictReader(handle))


def test_sheet_never_reveals_the_source_or_url(tmp_path, make_article):
    """The blindness guarantee. A visible source name turns labelling into
    transcription, and the feed URLs contain the section name outright."""
    sheets = annotate.write_sheets(_articles(make_article, 8), tmp_path)
    header = _read(sheets[0])[0].keys()

    assert set(header) == set(annotate.COLUMNS)
    assert "source_name" not in header and "url" not in header

    body = sheets[0].read_text(encoding=ENCODING)
    assert "Source 0" not in body and "example.test" not in body


def test_label_column_is_empty_for_the_annotator(tmp_path, make_article):
    rows = _read(annotate.write_sheets(_articles(make_article, 6), tmp_path)[0])
    assert all(row["label"] == "" for row in rows)


def test_shards_partition_the_sample_with_a_shared_overlap(tmp_path, make_article):
    articles = _articles(make_article, 40)
    sheets = annotate.write_sheets(articles, tmp_path, shards=4, overlap=8)

    assert len(sheets) == 4
    per_sheet = [{row["article_id"] for row in _read(sheet)} for sheet in sheets]

    shared = set.intersection(*per_sheet)
    assert len(shared) == 8, "the overlap block must appear in every sheet"

    everything = set.union(*per_sheet)
    assert everything == {a.id for a in articles}

    # Outside the overlap, no article is labelled twice.
    once = [aid for aid in everything if sum(aid in s for s in per_sheet) == 1]
    assert len(once) == len(articles) - 8


def test_export_is_reproducible(tmp_path, make_article):
    articles = _articles(make_article, 20)
    first = annotate.write_sheets(articles, tmp_path / "a", shards=2, overlap=4, seed=7)
    second = annotate.write_sheets(articles, tmp_path / "b", shards=2, overlap=4, seed=7)

    for left, right in zip(first, second, strict=True):
        assert left.read_text(encoding=ENCODING) == right.read_text(encoding=ENCODING)


def test_sample_covers_every_source(make_article):
    sample = annotate.choose_sample(_articles(make_article, 50), size=10, seed=1)
    assert len({a.source_id for a in sample}) == 5


def test_sample_takes_one_article_per_story_group(make_article):
    articles = _articles(make_article, 20)
    group_of = {a.id: ("dupes" if i < 10 else a.id) for i, a in enumerate(articles)}

    sample = annotate.choose_sample(articles, size=20, seed=1, group_of=group_of)
    assert len(sample) == 11, "the ten duplicates should contribute one article"


def test_summary_newlines_do_not_break_the_row(tmp_path, make_article):
    article = make_article("a1", summary="First line.\nSecond line,\twith a tab.")
    rows = _read(annotate.write_sheets([article], tmp_path)[0])
    assert rows[0]["summary"] == "First line. Second line, with a tab."


def _sheet(tmp_path, rows, name="sheet.csv"):
    path = tmp_path / name
    with path.open("w", encoding=ENCODING, newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(annotate.COLUMNS)
        writer.writerows(rows)
    return path


def test_reads_back_valid_labels(tmp_path, taxonomy):
    path = _sheet(tmp_path, [["a1", "t", "s", "sport", ""], ["a2", "t", "s", "politics", ""]])
    labels, problems = annotate.read_sheet(path, taxonomy)

    assert not problems
    assert {(lbl.article_id, lbl.topic) for lbl in labels} == {("a1", "sport"), ("a2", "politics")}


def test_accepts_the_capitalisation_a_spreadsheet_introduces(tmp_path, taxonomy):
    path = _sheet(tmp_path, [["a1", "t", "s", " Sport ", ""], ["a2", "t", "s", "Crime-Justice", ""]])
    labels, problems = annotate.read_sheet(path, taxonomy)

    assert not problems
    assert {lbl.topic for lbl in labels} == {"sport", "crime_justice"}


def test_rejects_a_value_that_is_not_a_class(tmp_path, taxonomy):
    path = _sheet(tmp_path, [["a1", "t", "s", "sprot", ""]])
    labels, problems = annotate.read_sheet(path, taxonomy)

    assert not labels
    assert problems[0].row == 2 and "sprot" in problems[0].detail


def test_unsorted_is_accepted_as_an_honest_answer(tmp_path, taxonomy):
    path = _sheet(tmp_path, [["a1", "t", "s", "unsorted", ""]])
    labels, problems = annotate.read_sheet(path, taxonomy)

    assert not problems and labels[0].topic == taxonomy.unsorted


def test_blank_labels_are_skipped_not_errors(tmp_path, taxonomy):
    path = _sheet(tmp_path, [["a1", "t", "s", "", ""], ["a2", "t", "s", "sport", ""]])
    labels, problems = annotate.read_sheet(path, taxonomy)

    assert not problems and len(labels) == 1


def test_unknown_article_id_is_reported(tmp_path, taxonomy):
    path = _sheet(tmp_path, [["ghost", "t", "s", "sport", ""]])
    labels, problems = annotate.read_sheet(path, taxonomy, known_ids=frozenset({"a1"}))

    assert not labels and "not in the corpus" in problems[0].detail


def test_duplicate_rows_are_reported(tmp_path, taxonomy):
    path = _sheet(tmp_path, [["a1", "t", "s", "sport", ""], ["a1", "t", "s", "politics", ""]])
    labels, problems = annotate.read_sheet(path, taxonomy)

    assert len(labels) == 1 and "duplicate" in problems[0].detail


def test_a_sheet_missing_the_label_column_fails_loudly(tmp_path, taxonomy):
    path = tmp_path / "broken.csv"
    with path.open("w", encoding=ENCODING, newline="") as handle:
        csv.writer(handle).writerows([["article_id", "title"], ["a1", "t"]])

    labels, problems = annotate.read_sheet(path, taxonomy)
    assert not labels and "missing column" in problems[0].detail


def test_disagreement_between_annotators_is_detected(taxonomy):
    from newsml.labels import Label, LabelSource

    labels = [
        Label("a1", "sport", LabelSource.HUMAN, "riaz"),
        Label("a1", "politics", LabelSource.HUMAN, "friend"),
        Label("a2", "sport", LabelSource.HUMAN, "riaz"),
        Label("a2", "sport", LabelSource.HUMAN, "friend"),
    ]
    assert annotate.disagreements(labels) == {"a1": {"sport", "politics"}}


def test_guide_lists_every_class_and_the_escape_hatch(tmp_path, taxonomy):
    guide = annotate.write_guide(taxonomy, tmp_path / "guide.md").read_text(encoding="utf-8")

    for topic in taxonomy.classes:
        assert f"`{topic.id}`" in guide
    assert f"`{taxonomy.unsorted}`" in guide
