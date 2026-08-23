"""One entrypoint for every offline task. `python -m newsml <command>`.

Ground rule 10 applied to tooling as well as libraries: a config-driven CLI and a
directory of results is enough at this scale. No experiment-tracking server.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import admit as admit_mod
from . import annotate as annotate_mod
from . import boilerplate as boilerplate_mod
from . import clean as clean_mod
from . import neardup as neardup_mod
from . import profile as profile_mod
from . import snapshot as snapshot_mod
from .config import (
    ARTIFACT_DIR,
    CLEANING_VERSION,
    DATA_DIR,
    DEFAULT_TEXT_VARIANT,
    REPORT_DIR,
    SEED,
    SNAPSHOT_DIR,
    TAXONOMY_PATH,
    TEXT_VARIANTS,
    mongo_uri,
)
from .labels import load_taxonomy
from .load import load_articles

REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_snapshot_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d}-{CLEANING_VERSION.replace('.', '')}"


def cmd_profile(args: argparse.Namespace) -> int:
    articles = load_articles(args.uri, limit=args.limit)
    if not articles:
        print("no articles found — is the collector's MongoDB reachable?", file=sys.stderr)
        return 1
    report = profile_mod.write(articles, Path(args.out))
    print(f"{len(articles):,} articles profiled → {report}")
    return 0


def cmd_boilerplate(args: argparse.Namespace) -> int:
    import yaml

    articles = load_articles(args.uri, limit=args.limit)
    candidates = boilerplate_mod.discover(articles, args.variant)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    document = boilerplate_mod.to_yaml_document(candidates)
    out.write_text(
        "# Learned per-source boilerplate. Regenerate with `make ml-boilerplate`.\n"
        "# Review before trusting: every line here is removed from every article\n"
        "# of its source, and a false positive silently deletes real content.\n"
        + yaml.safe_dump(document, sort_keys=True, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    print(f"{len(candidates)} boilerplate line(s) across {len(document['sources'])} source(s) → {out}")
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    articles = load_articles(args.uri, limit=args.limit)
    if not articles:
        print("no articles found — is the collector's MongoDB reachable?", file=sys.stderr)
        return 1
    result = snapshot_mod.build(
        articles,
        snapshot_id=args.id,
        out_root=Path(args.out),
        variant=args.variant,
        repo=REPO_ROOT,
        check_language=not args.no_language_check,
    )
    print(json.dumps(result.manifest["counts"], indent=2, sort_keys=True))
    print(f"snapshot → {result.directory}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Rebuild a snapshot into a scratch directory and compare digests.

    This is the acceptance criterion "rebuilds byte-identically", run as a
    command rather than asserted in prose.
    """
    import shutil
    import tempfile

    original = Path(args.out) / args.id / "manifest.json"
    if not original.exists():
        print(f"no snapshot at {original}", file=sys.stderr)
        return 1
    expected = json.loads(original.read_text(encoding="utf-8"))["digests"]

    articles = load_articles(args.uri, limit=args.limit)
    scratch = Path(tempfile.mkdtemp(prefix="newsml-verify-"))
    try:
        rebuilt = snapshot_mod.build(
            articles,
            snapshot_id=args.id,
            out_root=scratch,
            variant=args.variant,
            repo=REPO_ROOT,
            check_language=not args.no_language_check,
        )
        mismatched = {k: (v, rebuilt.manifest["digests"].get(k)) for k, v in expected.items()
                      if rebuilt.manifest["digests"].get(k) != v}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if mismatched:
        for name, (want, got) in sorted(mismatched.items()):
            print(f"MISMATCH {name}\n  expected {want}\n  rebuilt  {got}", file=sys.stderr)
        return 1
    print(f"snapshot {args.id} rebuilds byte-identically ({len(expected)} file(s))")
    return 0


def cmd_export_labels(args: argparse.Namespace) -> int:
    """Draw a sample and write blind labelling sheets, one per annotator."""
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    articles = load_articles(args.uri, limit=args.limit)
    if not articles:
        print("no articles found \u2014 is the collector's MongoDB reachable?", file=sys.stderr)
        return 1

    # Only admitted articles: no one should spend attention labelling a
    # horoscope that the pipeline is going to drop anyway.
    pairs = [(a, clean_mod.clean(a.text(args.variant))) for a in articles]
    admitted, _ = admit_mod.partition(pairs, check_language=False)
    grouping = neardup_mod.group({a.article.id: a.cleaned.text for a in admitted})

    sample = annotate_mod.choose_sample(
        [a.article for a in admitted], size=args.size, seed=SEED, group_of=grouping.group_of
    )
    out = Path(args.out)
    sheets = annotate_mod.write_sheets(
        sample, out, shards=args.shards, overlap=args.overlap, seed=SEED
    )
    guide = annotate_mod.write_guide(taxonomy, out / "labelling-guide.md", overlap=args.overlap)

    print(f"{len(sample)} articles sampled from {len(admitted)} admitted ({grouping.group_count} story groups)")
    for sheet in sheets:
        print(f"  {sheet}")
    print(f"guide  {guide}")
    return 0


def cmd_import_labels(args: argparse.Namespace) -> int:
    """Read completed sheets, validate every cell, and write the gold set."""
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    known = frozenset(a.id for a in load_articles(args.uri, limit=args.limit))

    labels: list = []
    problems: list = []
    for raw in args.sheets:
        path = Path(raw)
        if not path.exists():
            print(f"no such sheet: {path}", file=sys.stderr)
            return 1
        found, issues = annotate_mod.read_sheet(path, taxonomy, known_ids=known, annotator=path.stem)
        labels.extend(found)
        problems.extend(issues)

    for problem in problems:
        print(f"  {problem.sheet}:{problem.row} {problem.article_id} \u2014 {problem.detail}", file=sys.stderr)

    disagreed = annotate_mod.disagreements(labels)
    counts: dict[str, int] = {}
    for label in labels:
        counts[label.topic] = counts.get(label.topic, 0) + 1

    print(f"{len(labels)} label(s) across {len({lbl.article_id for lbl in labels})} article(s)")
    print(f"{len(problems)} problem(s), {len(disagreed)} article(s) with annotator disagreement")
    for topic in sorted(taxonomy.ids | {taxonomy.unsorted}):
        starved = "" if counts.get(topic, 0) >= args.min_per_class else "  <-- starved"
        print(f"  {topic:<22} {counts.get(topic, 0):>5}{starved}")

    if problems and not args.force:
        print("refusing to write while problems remain; fix them or pass --force", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for label in sorted(labels, key=lambda x: (x.article_id, x.detail)):
            handle.write(json.dumps({
                "article_id": label.article_id,
                "topic": label.topic,
                "label_source": str(label.source),
                "annotator": label.detail,
                "taxonomy_version": taxonomy.version,
            }, sort_keys=True) + "\n")
    print(f"gold set -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="newsml", description=__doc__)
    parser.add_argument("--uri", default=None, help=f"MongoDB URI (default: {mongo_uri()})")
    parser.add_argument("--limit", type=int, default=None, help="read at most N articles")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="write the corpus profile report and figures")
    p.add_argument("--out", default=str(REPORT_DIR))
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("boilerplate", help="discover per-source template lines for review")
    p.add_argument("--out", default=str(ARTIFACT_DIR / "boilerplate.yaml"))
    p.add_argument("--variant", default="title_summary_content", choices=TEXT_VARIANTS)
    p.set_defaults(func=cmd_boilerplate)

    p = sub.add_parser("export-labels", help="write blind labelling sheets for human annotators")
    p.add_argument("--out", default=str(DATA_DIR / "labels" / "pilot"))
    p.add_argument("--size", type=int, default=150, help="articles to sample")
    p.add_argument("--shards", type=int, default=4, help="one sheet per annotator")
    p.add_argument("--overlap", type=int, default=20, help="articles shared by every sheet")
    p.add_argument("--variant", default=DEFAULT_TEXT_VARIANT, choices=TEXT_VARIANTS)
    p.set_defaults(func=cmd_export_labels)

    p = sub.add_parser("import-labels", help="validate completed sheets and write the gold set")
    p.add_argument("--sheets", nargs="+", required=True)
    p.add_argument("--out", default=str(DATA_DIR / "labels" / "gold.jsonl"))
    p.add_argument("--min-per-class", type=int, default=40)
    p.add_argument("--force", action="store_true", help="write even if problems were reported")
    p.set_defaults(func=cmd_import_labels)

    for name, func, help_text in (
        ("snapshot", cmd_snapshot, "freeze a dataset"),
        ("verify", cmd_verify, "rebuild a snapshot and compare digests"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--id", default=_default_snapshot_id())
        p.add_argument("--out", default=str(SNAPSHOT_DIR))
        p.add_argument("--variant", default=DEFAULT_TEXT_VARIANT, choices=TEXT_VARIANTS)
        p.add_argument("--no-language-check", action="store_true", help="skip langdetect (much faster)")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    return args.func(args)
