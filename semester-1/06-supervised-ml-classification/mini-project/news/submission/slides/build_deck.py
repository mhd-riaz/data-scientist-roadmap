"""Build the mini-project slide deck.

Run with no install:

    uv run --with python-pptx python submission/slides/build_deck.py

Numbers are read from the trained model's `metrics.json` wherever possible, so the deck
cannot quote a figure the shipped model does not produce. The few that come from closed
experiments (the model ladder, the ensemble oracle) are constants here and are marked as
such in `docs/plan.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
METRICS = PROJECT / "ml-v2" / "artifacts" / "models" / "v2-001" / "metrics.json"
FIGURES = PROJECT / "submission" / "report" / "figures"
OUT = HERE / "news-topic-classifier.pptx"

INK = RGBColor(0x1B, 0x22, 0x2C)
MUTED = RGBColor(0x5A, 0x66, 0x75)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
GOOD = RGBColor(0x1E, 0x7A, 0x4B)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xDD, 0xE2, 0xE8)

W, H = Inches(13.333), Inches(7.5)


def deck() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _text(slide, left, top, width, height, size=18, bold=False, color=INK,
          align=PP_ALIGN.LEFT, space_after=8):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    para = frame.paragraphs[0]
    para.alignment = align
    para.space_after = Pt(space_after)
    run = para.add_run()
    run.font.size, run.font.bold, run.font.color.rgb = Pt(size), bold, color
    return frame, run


def heading(slide, title: str, kicker: str = "") -> None:
    if kicker:
        _, run = _text(slide, Inches(0.7), Inches(0.42), Inches(11.9), Inches(0.3),
                       size=13, bold=True, color=ACCENT)
        run.text = kicker.upper()
    _, run = _text(slide, Inches(0.7), Inches(0.72), Inches(11.9), Inches(0.8),
                   size=30, bold=True)
    run.text = title
    line = slide.shapes.add_shape(1, Inches(0.7), Inches(1.62), Inches(11.9), Emu(9525))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()


def bullets(slide, items: list, top=Inches(2.0), left=Inches(0.7),
            width=Inches(11.9), size=18) -> None:
    box = slide.shapes.add_textbox(left, top, width, Inches(4.6))
    frame = box.text_frame
    frame.word_wrap = True
    for i, item in enumerate(items):
        text, level = (item, 0) if isinstance(item, str) else item
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.level = level
        para.space_after = Pt(12 if level == 0 else 6)
        run = para.add_run()
        run.text = ("" if level else "") + text
        run.font.size = Pt(size if level == 0 else size - 3)
        run.font.color.rgb = INK if level == 0 else MUTED


def table(slide, rows: list[list[str]], left=Inches(0.7), top=Inches(2.1),
          width=Inches(11.9), height=Inches(0.4), highlight: int | None = None,
          size=14) -> None:
    shape = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width,
                                   height * len(rows))
    tbl = shape.table
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            # add_run rather than cell.text: an empty string creates no run at all,
            # and the font settings below would then have nothing to apply to.
            run = para.add_run()
            run.text = str(value)
            run.font.size = Pt(size)
            run.font.bold = r == 0 or r == highlight
            run.font.color.rgb = ACCENT if r == highlight else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xF2, 0xF5, 0xF8) if r == 0 else PAPER


def note(slide, text: str, top=Inches(6.5)) -> None:
    _, run = _text(slide, Inches(0.7), top, Inches(11.9), Inches(0.6),
                   size=14, color=MUTED)
    run.text = text


def picture(slide, name: str, left, top, width) -> bool:
    path = FIGURES / name
    if not path.is_file():
        return False
    slide.shapes.add_picture(str(path), left, top, width=width)
    return True


def build(m: dict) -> Presentation:
    val, test, ab = m["validation"], m["test"], m["body_ab"]
    prs = deck()

    # --- 1. title -----------------------------------------------------------
    s = _blank(prs)
    _, run = _text(s, Inches(1.0), Inches(2.1), Inches(11.3), Inches(1.4),
                   size=44, bold=True)
    run.text = "Reading the Body"
    _, run = _text(s, Inches(1.0), Inches(3.2), Inches(11.3), Inches(0.9), size=24,
                   color=MUTED)
    run.text = ("A calibrated, abstaining classifier for 13-class news topic "
                "assignment")
    _, run = _text(s, Inches(1.0), Inches(4.6), Inches(11.3), Inches(0.5), size=18)
    run.text = "Mohamed Riaz  ·  PES1PGE25DS037"
    _, run = _text(s, Inches(1.0), Inches(5.1), Inches(11.3), Inches(0.5), size=16,
                   color=MUTED)
    run.text = ("Supervised Machine Learning — Classification  ·  Mini Project  ·  "
                "Department of CSE, PES University")

    # --- 2. problem ---------------------------------------------------------
    s = _blank(prs)
    heading(s, "The same story, five times, filed five ways", kicker="Problem")
    bullets(s, [
        "A reader following several papers sees one wire story repeated, each publisher "
        "filing it under its own section names.",
        "Sections are a publishing convention, not a property of the article — there is "
        "no shared taxonomy across mastheads.",
        "So: assign every article a topic from one fixed 13-class taxonomy, "
        "independently of who published it.",
        ("The constraint I set myself: classical supervised ML only. No large language "
         "model anywhere in the pipeline — every prediction has to be explainable from "
         "the model's own weights.", 1),
    ])
    note(s, "13 classes: politics · business_economy · crime_justice · technology · sport · "
            "entertainment_arts · health · education · science_space · environment_climate · "
            "disaster_accident · conflict_war · society_lifestyle")

    # --- 3. the question ----------------------------------------------------
    s = _blank(prs)
    heading(s, "The research question", kicker="Hypothesis")
    bullets(s, [
        "An RSS feed gives you a headline and maybe a summary — about 33 words. "
        "That is what news classifiers are normally trained on.",
        "My collector follows the link and scrapes the body: a median of 3,765 "
        "characters, roughly 100× more text.",
        "Is the body worth it? Not obviously — bodies also carry house style, "
        "boilerplate and author biographies, which are shortcuts a classifier will "
        "happily learn instead of the topic.",
    ])
    _, run = _text(s, Inches(0.7), Inches(5.3), Inches(11.9), Inches(1.0), size=26,
                   bold=True, color=ACCENT)
    run.text = ("Does 650 words of body beat 33 words of headline — "
                "and can I prove it rather than assume it?")

    # --- 4. system ----------------------------------------------------------
    s = _blank(prs)
    heading(s, "What was built", kicker="System")
    stages = [
        ("Collector", "Go service, 97 RSS/Atom feeds\n→ MongoDB, deduplicated"),
        ("Preparation", "clean · admit · group\nnear-duplicate stories"),
        ("Snapshot", "frozen Parquet corpus\n+ manifest of every input"),
        ("Model", "word+char TF-IDF → linear SVM\n+ isotonic calibration"),
        ("Demo UI", "Streamlit: classify, explain,\nabstain, validate"),
    ]
    for i, (name, detail) in enumerate(stages):
        left = Inches(0.7 + i * 2.44)
        card = s.shapes.add_shape(5, left, Inches(2.6), Inches(2.2), Inches(1.9))
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(0xF2, 0xF5, 0xF8)
        card.line.color.rgb = RULE
        card.text_frame.word_wrap = True
        para = card.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        run = para.add_run()
        run.text = name
        run.font.size, run.font.bold, run.font.color.rgb = Pt(17), True, INK
        sub = card.text_frame.add_paragraph()
        sub.alignment = PP_ALIGN.CENTER
        srun = sub.add_run()
        srun.text = detail
        srun.font.size, srun.font.color.rgb = Pt(11), MUTED
        if i < len(stages) - 1:
            arrow = s.shapes.add_shape(13, left + Inches(2.24), Inches(3.4),
                                       Inches(0.16), Inches(0.2))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = MUTED
            arrow.line.fill.background()
    note(s, "Nothing is deployed. This is a proof of concept: build it, measure it "
            "honestly, then decide whether it earns a deployment conversation.",
         top=Inches(5.0))

    # --- 5. data ------------------------------------------------------------
    s = _blank(prs)
    heading(s, "The data", kicker="Dataset")
    table(s, [
        ["", "Value"],
        ["Corpus at the frozen cut", "14,189 articles, 40 publishers"],
        ["Human labels", "8,001 — single label, 13 classes"],
        ["Body coverage on labelled articles", "94.3% (median 3,765 chars)"],
        ["Class imbalance", "6.7 : 1"],
        ["Labelled split (grouped + temporal)",
         f"train {m['splits'].get('train', 0):,} / val {m['splits'].get('val', 0):,} "
         f"/ test {m['splits'].get('test', 0):,}"],
        ["Rejected by admission rules", "4.1% (582 articles)"],
    ], top=Inches(2.1), height=Inches(0.46))
    note(s, "Labels were produced over three rounds with a targeted sampler, because "
            "random sampling could never reach the rare classes — conflict_war is 1.7% "
            "of news, so 150 labels would need ~9,000 random draws.", top=Inches(5.6))

    # --- 6. preparation -----------------------------------------------------
    s = _blank(prs)
    heading(s, "Preparation is where the accuracy came from", kicker="Method")
    bullets(s, [
        "Boilerplate discovery, per source. 355 repeated furniture lines across 47 "
        "sources — “Story continues below this ad” is in 75% of one publisher's bodies.",
        ("Including multi-sentence author biographies, which a length-based rule "
         "cannot reach. A cricket correspondent's bio inside a business article drags "
         "it toward sport.", 1),
        "Nine admission gates, each a switch so its cost can be measured, not assumed. "
        "4.1% rejected.",
        "Near-duplicate story grouping: TF-IDF cosine blocking, then verification with "
        "time-gap and boilerplate guards. Precision 0.86 at recall 0.81 on 43 "
        "hand-judged pairs.",
        ("Grouping on bodies folds 999 articles vs 708 on headlines, +41%. This matters: "
         "a syndicated copy of a training article landing in validation is leakage.", 1),
    ])

    # --- 7. methodology -----------------------------------------------------
    s = _blank(prs)
    heading(s, "How the evaluation avoids fooling itself", kicker="Validation")
    bullets(s, [
        "Grouped + temporal splits. No story group spans two splits; every test article "
        "was collected after every training article.",
        "Bootstrap intervals resample story groups, not articles — syndicated copies of "
        "one story are not independent observations.",
        "Model comparisons use McNemar's test, never a subtraction of two accuracies.",
        "Two publisher holdouts scored on every run: The Hindu (in-distribution) and "
        "The Guardian (out-of-distribution), each refit without that publisher.",
        "The test split was opened exactly once, through one function that refuses to "
        "run without an explicit flag — after the model was frozen.",
    ])
    note(s, "Any delta under ~0.03 on a 1,120-article validation split is noise. "
            "The harness exists to stop that being read as signal.")

    # --- 8. the result ------------------------------------------------------
    s = _blank(prs)
    heading(s, "The body wins — measured, not assumed", kicker="Results")
    table(s, [
        ["Model", "title+summary", "title+body", "McNemar p"],
        ["majority baseline", "0.025", "0.025", "—"],
        ["ComplementNB", "0.635", "0.594", "2.8e-02"],
        ["TF-IDF + LogisticRegression", "0.698", "0.752", "3.0e-05"],
        ["TF-IDF + LinearSVC", "0.696", "0.753", "1.3e-04"],
        ["word+char SVM  (final model)", f"{ab['title_summary']:.3f}",
         f"{ab['title_body']:.3f}", f"{ab['mcnemar_p']:.1e}"],
    ], top=Inches(2.1), height=Inches(0.48), highlight=5)
    _, run = _text(s, Inches(0.7), Inches(5.2), Inches(11.9), Inches(0.6), size=24,
                   bold=True, color=GOOD)
    run.text = (f"+{ab['delta']:.3f} macro-F1, non-overlapping intervals, "
                f"p = {ab['mcnemar_p']:.1e}")
    note(s, "ComplementNB gets WORSE with the body — its term-independence assumption is "
            "length-sensitive. “More text” is not universally better.", top=Inches(6.0))

    # --- 9. what failed -----------------------------------------------------
    s = _blank(prs)
    heading(s, "Everything else I tried, and lost", kicker="Results")
    table(s, [
        ["Alternative", "Result", "Kept?"],
        ["Entity / geography scrubbing (spaCy)", "no rule cleared the bar", "no"],
        ["Up-weighting the title", "monotonically worse", "no"],
        ["Tuning C over a 6× range", "moves macro-F1 by 0.002", "no"],
        ["XGBoost on 256 SVD components", "−0.036, 5× slower", "no"],
        ["Random forest / extra trees", "−0.041 to −0.087", "no"],
        ["MiniLM sentence embeddings", "−0.061", "no"],
        ["Majority vote over 10 models", "+0.001, p = 1.00", "no"],
        ["Per-class confidence cuts", "no better than one global cut", "no"],
    ], top=Inches(2.0), height=Inches(0.42))
    note(s, "A negative result you can defend is worth more than a positive one you "
            "cannot. Every row carries an interval or a paired test.", top=Inches(6.1))

    # --- 10. ensemble paradox ----------------------------------------------
    s = _blank(prs)
    heading(s, "The ensemble paradox", kicker="Finding")
    bullets(s, [
        "The models DO disagree. A perfect selector over the ten candidates would score "
        "0.900 [0.878, 0.919] — disjoint from the incumbent's 0.771.",
        "And every reachable vote lands on 0.771. Four different member sets, four ties, "
        "McNemar p = 1.0000.",
        "Why: the disagreement is asymmetric. MiniLM rescues 70 of the incumbent's "
        "errors and destroys 148 of its correct answers — worse than 2:1 against.",
        ("A vote has no way to know which side of that trade it is on. So I refused to "
         "build the stacked ensemble the plan had scheduled, and recorded why.", 1),
    ])
    _, run = _text(s, Inches(0.7), Inches(5.6), Inches(11.9), Inches(0.7), size=20,
                   bold=True, color=ACCENT)
    run.text = ("The 0.129 oracle gap says a SELECTOR would pay where an AVERAGE does "
                "not — which is exactly what abstention is.")

    # --- 11. calibration ----------------------------------------------------
    s = _blank(prs)
    heading(s, "Novelty 1 — the confidence number is honest", kicker="Innovation")
    if not picture(s, "calibration.png", Inches(1.2), Inches(2.0), Inches(10.9)):
        bullets(s, ["(figures/calibration.png missing — run "
                    "`uv run python scripts/build_figures.py` in ml-v2/)"])
    note(s, f"A LinearSVC margin is a signed distance from a hyperplane: no scale, not "
            f"comparable across classes. Isotonic calibration on 5 grouped folds of "
            f"train makes it a probability. ECE {val['ece']:.3f} on validation, "
            f"{test['ece']:.3f} on test — the calibration generalised BETTER than the "
            f"accuracy did.", top=Inches(5.9))

    # --- 12. abstention -----------------------------------------------------
    s = _blank(prs)
    heading(s, "Novelty 2 — it is allowed to say “I don't know”", kicker="Innovation")
    table(s, [
        ["On the held-out test split", "coverage", "accuracy on what it files"],
        ["forced to answer everything", "100%", f"{test['accuracy_without_abstention']:.1%}"],
        [f"abstain below {test['cut']:.3f}", f"{test['coverage']:.1%}",
         f"{test['accuracy_filed']:.1%}"],
    ], top=Inches(2.2), height=Inches(0.55), highlight=2, size=16)
    bullets(s, [
        "The cut is fitted on training out-of-fold probabilities for 90% precision — "
        "never on anything it is scored against.",
        "Per-class cuts bought nothing (0.879 vs 0.891 at matched coverage): once "
        "calibration makes scores comparable across classes, one global cut is enough.",
        "It abstains on the right things: on the 63 articles humans labelled "
        "“unsorted”, median confidence is 0.70 against 0.81 on labelled articles.",
    ], top=Inches(4.2), size=17)

    # --- 13. test split -----------------------------------------------------
    s = _blank(prs)
    heading(s, "The held-out test split, opened once", kicker="Results")
    table(s, [
        ["", "validation", "test"],
        ["articles", f"{val['n']:,}", f"{test['n']:,}"],
        ["macro-F1",
         f"{val['macro_f1']:.3f} [{val['macro_f1_low']:.3f}, {val['macro_f1_high']:.3f}]",
         f"{test['macro_f1']:.3f} [{test['macro_f1_low']:.3f}, {test['macro_f1_high']:.3f}]"],
        ["accuracy", f"{val['accuracy']:.3f}", f"{test['accuracy']:.3f}"],
        ["expected calibration error", f"{val['ece']:.3f}", f"{test['ece']:.3f}"],
        ["coverage at the shipping cut", f"{0.761:.1%}", f"{test['coverage']:.1%}"],
        ["accuracy on filed articles", f"{0.879:.3f}", f"{test['accuracy_filed']:.3f}"],
    ], top=Inches(2.1), height=Inches(0.46), highlight=2)
    _, run = _text(s, Inches(0.7), Inches(5.6), Inches(11.9), Inches(0.8), size=20,
                   bold=True, color=GOOD)
    run.text = (f"Delta {test['macro_f1'] - val['macro_f1']:+.3f} against a "
                "pre-registered ±0.05 guard, intervals overlapping. Nothing was changed "
                "in response.")

    # --- 14. per class ------------------------------------------------------
    s = _blank(prs)
    heading(s, "Per class — and which scores mean anything", kicker="Results")
    charted = picture(s, "per-class.png", Inches(0.9), Inches(1.9), Inches(5.6))
    bullets(s, [
        "sport 0.95, entertainment_arts 0.89 — tight intervals, real measurements.",
        ("education 0.71 has 29 validation articles and an interval a quarter of an F1 "
         "point wide. That is noise, and it is labelled as noise.", 1),
        "society_lifestyle 0.42 is a definitional grab-bag: community + labour + "
        "lifestyle glued together.",
        ("But it still calibrates honestly — 91% precision at a high cut. Weak at "
         "RANKING, fine at KNOWING. Abstention protects it.", 1),
        "18.6% of errors sit on class pairs where human annotators disagreed with each "
        "other. Macro-F1 has a real ceiling below 1.0.",
    ], top=Inches(2.0),
        left=Inches(6.9) if charted else Inches(0.7),
        width=Inches(5.7) if charted else Inches(11.9),
        size=15 if charted else 18)

    # --- 15. demo -----------------------------------------------------------
    s = _blank(prs)
    heading(s, "The demo", kicker="Live")
    shown = picture(s, "demo.png", Inches(0.9), Inches(1.9), Inches(7.4))
    bullets(s, [
        "Paste any news article — it classifies, scores its own confidence, and "
        "either files it or holds it for review.",
        "Explains itself: the model is linear over TF-IDF, so the decision is a plain "
        "sum of weight × term-frequency. The largest terms ARE the reasons — an exact "
        "explanation, not an approximation.",
        "The abstention dial is live: move the cut, watch coverage and accuracy trade.",
        "A Validation tab with the real per-class intervals, reliability diagram and "
        "confusion matrix — the evidence, not a claim.",
        "And a batch tab that runs the model over real collected articles nobody "
        "labelled.",
    ], top=Inches(2.0),
        left=Inches(8.5) if shown else Inches(0.7),
        width=Inches(4.2) if shown else Inches(11.9),
        size=13 if shown else 18)
    note(s, "uv run streamlit run app/streamlit_app.py", top=Inches(6.6))

    # --- 16. limits ---------------------------------------------------------
    s = _blank(prs)
    heading(s, "What I would not claim", kicker="Limitations")
    bullets(s, [
        "Four-day collection window. Nothing here measures topic drift over weeks — "
        "the publisher holdouts carry the whole generalisation argument.",
        "English only, and India-heavy: 40 publishers, mostly Indian mastheads plus "
        "The Guardian, BBC and France 24.",
        "conflict_war has 120 training articles and moved −0.14 on test. A thin class "
        "behaving like a thin class — flagged in advance, not explained afterwards.",
        "The label set is single-annotator for most rounds, so label noise cannot be "
        "estimated from agreement.",
        "Next lever is a fine-tuned transformer encoder — deliberately sequenced AFTER "
        "this A/B, so any gain can be attributed to one change.",
    ])

    # --- 17. conclusion -----------------------------------------------------
    s = _blank(prs)
    heading(s, "What the project actually showed", kicker="Conclusion")
    bullets(s, [
        f"Article bodies beat headlines by +{ab['delta']:.3f} macro-F1, "
        f"p = {ab['mcnemar_p']:.1e}. The hypothesis held.",
        "No fancier model family or ensemble improved on a linear SVM over TF-IDF. "
        "The classifier was never the bottleneck — data preparation was.",
        f"Calibration plus a rejection option turned a {test['accuracy_without_abstention']:.1%}"
        f"-accurate classifier into one that files {test['coverage']:.1%} of unseen "
        f"articles at {test['accuracy_filed']:.1%} accuracy.",
        f"Test macro-F1 {test['macro_f1']:.3f} "
        f"[{test['macro_f1_low']:.3f}, {test['macro_f1_high']:.3f}], on a split opened "
        "once and never reopened.",
    ])
    _, run = _text(s, Inches(0.7), Inches(5.8), Inches(11.9), Inches(0.8), size=22,
                   bold=True, color=ACCENT)
    run.text = ("The most useful output of this project is the list of things that "
                "did not work, each with a p-value attached.")

    return prs


def main() -> int:
    if not METRICS.is_file():
        raise SystemExit(f"no metrics at {METRICS}; run: uv run newsmlv2 train --id v2-001")
    prs = build(json.loads(METRICS.read_text(encoding="utf-8")))
    prs.save(OUT)
    print(f"wrote {OUT} ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
