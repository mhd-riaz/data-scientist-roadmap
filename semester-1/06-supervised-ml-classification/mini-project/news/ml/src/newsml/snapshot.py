"""Freeze a dataset: clean, admit, group, split, and write it to disk.

A snapshot is the boundary between this phase and every later one. After it
exists, training never touches MongoDB — it reads a directory whose contents are
a pure function of `(git sha, cleaning_version, seed, corpus)`.

Byte-identical rebuilds are an acceptance criterion, so the data files are JSONL
with sorted keys and a fixed row order, and every mutable fact (timestamps, host,
paths) lives in the manifest rather than in the data.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import admit, boilerplate, clean, neardup, splits
from .config import CLEANING_VERSION, SEED
from .load import Article


def git_sha(repo: Path) -> str:
    """The commit a snapshot was built from. 'unknown' outside a work tree."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    """Write rows in the given order with sorted keys, and return the digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    return _sha256(path)


@dataclass(frozen=True, slots=True)
class Result:
    directory: Path
    manifest: dict


def build(
    articles: list[Article],
    *,
    snapshot_id: str,
    out_root: Path,
    variant: str,
    repo: Path,
    check_language: bool = True,
    apply_boilerplate: bool = True,
) -> Result:
    """Run the whole Phase 2 pipeline and write the snapshot directory."""
    candidates = boilerplate.discover(articles, variant) if apply_boilerplate else []
    lookup = boilerplate.as_lookup(candidates)

    pairs = [(a, clean.clean(a.text(variant), boilerplate=lookup.get(a.source_id))) for a in articles]
    admitted, rejected = admit.partition(pairs, check_language=check_language)

    grouping = neardup.group({a.article.id: a.cleaned.text for a in admitted})

    rows = [
        splits.SplitRow(
            article_id=a.article.id,
            group_id=grouping.group_of[a.article.id],
            published_at=a.article.published_at,
        )
        for a in admitted
    ]
    assignment = splits.make_splits(rows)
    split_of = assignment.assignment()

    spanning = splits.groups_spanning_splits(rows, assignment)
    if spanning:
        raise AssertionError(f"{len(spanning)} story group(s) span more than one split: {sorted(spanning)[:5]}")

    article_rows = [
        {
            "article_id": a.article.id,
            "source_id": a.article.source_id,
            "source_name": a.article.source_name,
            "url": a.article.url,
            "title": a.article.title,
            "text": a.cleaned.text,
            "word_count": a.cleaned.word_count,
            "dateline_city": a.cleaned.dateline_city,
            "wire_agency": a.cleaned.wire_agency,
            "categories": sorted(c.casefold() for c in a.article.categories),
            "language_declared": a.article.language,
            "language_detected": a.detected_language,
            "country": a.article.country,
            "published_at": a.article.published_at.isoformat(),
            "collected_at": a.article.collected_at.isoformat(),
            "scrape_status": a.article.scrape_status,
            "story_group_id": grouping.group_of[a.article.id],
            "split": split_of.get(a.article.id, "dropped_at_boundary"),
        }
        for a in sorted(admitted, key=lambda x: x.article.id)
    ]

    rejection_rows = [
        {"article_id": r.article_id, "source_id": r.source_id, "reason": str(r.reason), "detail": r.detail}
        for r in sorted(rejected, key=lambda r: (r.article_id, str(r.reason)))
    ]

    directory = out_root / snapshot_id
    directory.mkdir(parents=True, exist_ok=True)
    digests = {
        "articles.jsonl": _write_jsonl(directory / "articles.jsonl", article_rows),
        "rejections.jsonl": _write_jsonl(directory / "rejections.jsonl", rejection_rows),
    }

    reason_counts: dict[str, int] = {}
    for row in rejection_rows:
        reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1

    manifest = {
        "snapshot_id": snapshot_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(repo),
        "cleaning_version": CLEANING_VERSION,
        "seed": SEED,
        "text_variant": variant,
        "counts": {
            "input": len(articles),
            "admitted": len(admitted),
            "rejected": len(rejected),
            "story_groups": grouping.group_count,
            "near_duplicate_pairs": len(grouping.pairs),
            "train": len(assignment.train),
            "val": len(assignment.val),
            "test": len(assignment.test),
            "dropped_at_boundary": len(assignment.dropped_at_boundary),
        },
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "boilerplate_lines": len(candidates),
        "digests": digests,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "data-card.md").write_text(_data_card(manifest, article_rows), encoding="utf-8")

    return Result(directory=directory, manifest=manifest)


def _data_card(manifest: dict, rows: list[dict]) -> str:
    counts = manifest["counts"]
    sources: dict[str, int] = {}
    for row in rows:
        sources[row["source_name"]] = sources.get(row["source_name"], 0) + 1
    dates = sorted(row["published_at"] for row in rows)

    lines = [
        f"# Data card — `{manifest['snapshot_id']}`",
        "",
        "Generated by `make ml-snapshot`. Do not edit by hand.",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Git SHA | `{manifest['git_sha']}` |",
        f"| Cleaning version | `{manifest['cleaning_version']}` |",
        f"| Seed | `{manifest['seed']}` |",
        f"| Text variant | `{manifest['text_variant']}` |",
        f"| Date range | {dates[0] if dates else '—'} → {dates[-1] if dates else '—'} |",
        "",
        "## Counts",
        "",
        "| Stage | Articles |",
        "| --- | --- |",
        f"| Input (bronze) | {counts['input']:,} |",
        f"| Admitted | {counts['admitted']:,} |",
        f"| Rejected | {counts['rejected']:,} |",
        f"| Story groups | {counts['story_groups']:,} |",
        f"| Train / val / test | {counts['train']:,} / {counts['val']:,} / {counts['test']:,} |",
        f"| Dropped at split boundary | {counts['dropped_at_boundary']:,} |",
        "",
        "## Rejections by reason",
        "",
        "| Reason | Articles |",
        "| --- | --- |",
    ]
    lines += [f"| `{k}` | {v:,} |" for k, v in manifest["rejection_reasons"].items()] or ["| — | 0 |"]

    lines += ["", "## Sources", "", "| Source | Articles |", "| --- | --- |"]
    lines += [f"| {k} | {v:,} |" for k, v in sorted(sources.items(), key=lambda kv: -kv[1])]

    lines += [
        "",
        "## Known limitations",
        "",
        "- `language_declared` comes from the source configuration, not from the text.",
        "  `language_detected` is the measured value; they disagree for some articles.",
        "- Story groups come from MinHash over shingles, which finds near-identical",
        "  copy. Two independent reports of the same event do not group.",
        "- Splits are grouped and temporal, so articles straddling a cut point are",
        "  dropped rather than assigned. The count is above.",
        "- No labels. The taxonomy and gold set are deferred until the corpus is",
        "  large enough to support them — see `docs/plan.md`.",
    ]
    return "\n".join(lines) + "\n"
