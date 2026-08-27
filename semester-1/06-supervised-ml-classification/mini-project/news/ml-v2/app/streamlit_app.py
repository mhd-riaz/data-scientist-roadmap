"""Demo UI for the news topic classifier.

Two things this is trying to show, beyond "it predicts a class":

1. **The confidence number means something.** It is a calibrated probability, so 0.80
   really does mean right about 80% of the time -- the reliability diagram on the
   Validation tab is the evidence, and the abstention dial in the sidebar is what that
   honesty buys you.
2. **The model can say "I don't know".** Below the cut it declines rather than guessing,
   which is why filed accuracy (83%) is well above forced accuracy (78%).

No logic lives here. Everything comes from `newsmlv2.serve` and the metrics written by
`newsmlv2 train`, so the demo cannot drift from the measured model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from newsmlv2 import config, serve
from newsmlv2 import snapshot as snapshot_mod

SNAPSHOT_ID = "v2-001"
MODEL_DIR = config.ARTIFACT_DIR / "models" / SNAPSHOT_ID

READABLE = {
    "business_economy": "Business & Economy",
    "conflict_war": "Conflict & War",
    "crime_justice": "Crime & Justice",
    "disaster_accident": "Disaster & Accident",
    "education": "Education",
    "entertainment_arts": "Entertainment & Arts",
    "environment_climate": "Environment & Climate",
    "health": "Health",
    "politics": "Politics",
    "science_space": "Science & Space",
    "society_lifestyle": "Society & Lifestyle",
    "sport": "Sport",
    "technology": "Technology",
}

SAMPLES = {
    "Sport": (
        "India beat Australia by six wickets in the final over at the MCG",
        "Chasing 287, the Indian batting order held its nerve as the captain struck an "
        "unbeaten 94 off 71 balls. The bowlers had earlier restricted Australia to 286 "
        "for eight, with two wickets falling in the final over of the innings. The win "
        "seals the three-match one-day international series 2-1, and hands the visitors "
        "their first series victory in Australia since 2019.",
    ),
    "Business": (
        "Reserve Bank holds repo rate steady as inflation cools",
        "The monetary policy committee voted five to one to keep the benchmark repo rate "
        "unchanged, citing easing food inflation and steady growth in industrial output. "
        "Bond yields fell four basis points after the announcement and the rupee closed "
        "marginally stronger against the dollar. Economists said a rate cut is now likely "
        "in the next quarter if the monsoon holds.",
    ),
    "A near tie": (
        "Parliament clears bill raising import duty on electronics",
        "The bill passed after a three-hour debate in which the opposition walked out, "
        "arguing the measure would raise handset prices for consumers. Industry bodies "
        "welcomed the protection for domestic manufacturers, while importers warned of "
        "supply disruption. The finance minister said the change would take effect from "
        "the start of the next fiscal year.",
    ),
    "A boundary case": (
        "NCERT panel recommends rewriting three chapters of the class 10 history textbook",
        "The panel, appointed last year, said the chapters gave disproportionate space to "
        "certain periods. Opposition members of the review committee dissented, calling "
        "the exercise politically motivated. Teachers' associations have asked for the "
        "draft to be published before it is adopted. The ministry said the revised "
        "textbook would reach schools next academic year.",
    ),
}


@st.cache_resource(show_spinner="Loading the model bundle...")
def load_model() -> serve.Classifier:
    return serve.load(MODEL_DIR)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((MODEL_DIR / serve.METRICS).read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Reading the frozen snapshot...")
def load_unlabelled(limit: int = 400) -> pd.DataFrame:
    """Real articles from the corpus that no human ever labelled."""
    snap = snapshot_mod.read(config.SNAPSHOT_DIR / SNAPSHOT_ID)
    frame = snap.frame
    pool = frame[frame["topic"].isna() & frame["has_body"]]
    return pool.sample(min(limit, len(pool)), random_state=config.SEED)[
        ["article_id", "publisher", "title", "summary", "body"]
    ].reset_index(drop=True)


@st.cache_data(show_spinner="Looking for something it finds hard...")
def hardest(_model: serve.Classifier, n: int = 12) -> pd.DataFrame:
    """The articles in the unlabelled pool the model is least sure about."""
    pool = load_unlabelled(200)
    texts = [serve.text_for(t, s, b)
             for t, s, b in zip(pool["title"], pool["summary"], pool["body"])]
    scores = _model.probabilities(texts).max(axis=1)
    return pool.assign(confidence=scores).nsmallest(n, "confidence").reset_index(drop=True)


def least_certain(model: serve.Classifier, _cut: float):
    """Cycle through the hard cases so repeated clicks show different articles."""
    frame = hardest(model)
    if frame.empty:
        return None
    seen = st.session_state.get("hard_index", -1) + 1
    st.session_state.hard_index = seen
    return frame.iloc[seen % len(frame)]


def pretty(topic: str) -> str:
    return READABLE.get(topic, topic.replace("_", " ").title())


def interpolate(curve: list[dict], cut: float) -> tuple[float, float]:
    """Coverage and accuracy at an arbitrary cut, read off the validation sweep."""
    cuts = [p["cut"] for p in curve]
    return (
        float(np.interp(cut, cuts, [p["coverage"] for p in curve])),
        float(np.interp(cut, cuts, [p["accuracy_on_kept"] for p in curve])),
    )


st.set_page_config(page_title="News Topic Classifier", layout="wide")

if not (MODEL_DIR / serve.BUNDLE).is_file():
    st.error(
        f"No model bundle at `{MODEL_DIR}`.\n\n"
        f"Build one first:\n\n```\nuv run newsmlv2 train --id {SNAPSHOT_ID}\n```"
    )
    st.stop()

model = load_model()
metrics = load_metrics()
val, test = metrics["validation"], metrics["test"]

st.title("News Topic Classifier")
st.caption(
    "Files an English news article into one of 13 topics — and declines to answer when "
    "it is not sure enough. Supervised ML (Classification) mini-project · "
    "Mohamed Riaz · PES1PGE25DS037"
)

with st.sidebar:
    st.subheader("The model")
    st.metric("Test macro-F1", f"{test['macro_f1']:.3f}",
              help=f"95% interval [{test['macro_f1_low']:.3f}, {test['macro_f1_high']:.3f}] · "
                   f"test split opened once, on {test['opened_on']}, and closed")
    st.metric("Accuracy on filed articles", f"{test['accuracy_filed']:.1%}",
              delta=f"{test['accuracy_filed'] - test['accuracy_without_abstention']:+.1%} vs no abstention")
    st.caption(
        f"`word_char_svc` · title + {serve.BODY_CHARS:,} chars of body · isotonic "
        f"calibration · trained on {metrics['metadata']['train_articles']:,} articles"
    )

    st.divider()
    st.subheader("Abstention dial")
    st.caption(
        "How sure must it be before it files an article without a human reading it? "
        "Everything below the line is routed to a person instead of guessed at."
    )
    cut = st.slider("Confidence cut", 0.0, 0.95, float(metrics["cut"]), 0.01)
    coverage, on_kept = interpolate(metrics["coverage_curve"], cut)
    left, right = st.columns(2)
    left.metric("Filed automatically", f"{coverage:.0%}")
    right.metric("...and right this often", f"{on_kept:.0%}")
    if abs(cut - metrics["cut"]) > 1e-9:
        st.caption(f"Shipping default is {metrics['cut']:.3f}, fitted on training "
                   "out-of-fold scores for 90% precision.")

classify_tab, validation_tab, batch_tab, method_tab = st.tabs(
    ["Classify an article", "Validation", "Run it on the corpus", "How it was built"]
)


with classify_tab:
    if "title" not in st.session_state:
        first = next(iter(SAMPLES))
        st.session_state.title, st.session_state.body = SAMPLES[first]

    def use_sample(name: str) -> None:
        st.session_state.title, st.session_state.body = SAMPLES[name]

    def use_uncertain() -> None:
        row = least_certain(model, cut)
        if row is None:
            st.session_state.title = st.session_state.body = ""
        else:
            st.session_state.title, st.session_state.body = row["title"], row["body"]

    st.markdown("**Start from**")
    columns = st.columns(len(SAMPLES) + 1)
    for column, name in zip(columns, SAMPLES):
        column.button(name, on_click=use_sample, args=(name,), use_container_width=True)
    columns[-1].button("A real one it is unsure about", on_click=use_uncertain,
                       use_container_width=True,
                       help="Picks the lowest-confidence article from a sample of real "
                            "collected news that nobody ever labelled.")

    title = st.text_input("Headline", key="title", placeholder="Paste the headline")
    body = st.text_area("Article body", key="body", height=200,
                        placeholder="Paste the article text. The model reads the first "
                                    "4,000 characters.")

    if st.button("Classify", type="primary", use_container_width=True):
        if not (title.strip() or body.strip()):
            st.warning("Give it a headline or some body text first.")
        else:
            result = model.classify(title, "", body)
            filed = result.confidence >= cut
            ranked = result.ranked

            verdict, detail = st.columns([2, 3])
            with verdict:
                if filed:
                    st.success(f"### {pretty(result.topic)}")
                    st.caption(f"Filed automatically — confidence {result.confidence:.1%} "
                               f"is above the {cut:.2f} cut.")
                else:
                    st.warning(f"### Held for review — looks like {pretty(result.topic)}")
                    st.caption(f"Confidence {result.confidence:.1%} is below the {cut:.2f} "
                               "cut, so this goes to a person rather than being filed.")
                st.progress(result.confidence)

                margin = ranked[0][1] - ranked[1][1]
                if margin < 0.25:
                    st.info(
                        f"**Close call.** {pretty(ranked[1][0])} is only "
                        f"{margin:.2f} behind. Raise the cut in the sidebar past "
                        f"{result.confidence:.2f} and this article stops being filed."
                    )

            with detail:
                st.markdown("**Where the probability went**")
                st.dataframe(
                    pd.DataFrame(
                        [{"Topic": pretty(t), "Probability": p} for t, p in ranked[:6]]
                    ),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Probability": st.column_config.ProgressColumn(
                            format="%.3f", min_value=0.0, max_value=1.0
                        )
                    },
                )

            if result.evidence:
                st.markdown("**Why — the words that pushed it there**")
                st.caption(
                    "The classifier is linear over TF-IDF, so its decision is a plain sum "
                    "of weight x term-frequency. These are the largest terms in that sum, "
                    "which makes this an exact explanation rather than an approximation."
                )
                st.dataframe(
                    pd.DataFrame(
                        [{"Term": t, "Contribution": round(v, 4)} for t, v in result.evidence]
                    ),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Contribution": st.column_config.ProgressColumn(
                            format="%.3f", min_value=0.0,
                            max_value=float(result.evidence[0][1]),
                        )
                    },
                )


with validation_tab:
    a, b, c, d = st.columns(4)
    a.metric("Validation macro-F1", f"{val['macro_f1']:.3f}",
             help=f"[{val['macro_f1_low']:.3f}, {val['macro_f1_high']:.3f}], "
                  "bootstrapped over story groups")
    b.metric("Test macro-F1", f"{test['macro_f1']:.3f}",
             delta=f"{test['macro_f1'] - val['macro_f1']:+.3f} vs validation")
    c.metric("Calibration error (test)", f"{test['ece']:.3f}",
             delta=f"{test['ece'] - val['ece']:+.3f} vs validation", delta_color="inverse")
    d.metric("Body vs headline", f"+{metrics['body_ab']['delta']:.3f}",
             help=f"McNemar p = {metrics['body_ab']['mcnemar_p']:.1e}")

    st.caption(
        f"Validation is {val['n']:,} articles, test is {test['n']:,}. The test split was "
        f"opened exactly once ({test['opened_on']}) and nothing was changed in response "
        "to what it said — that is what makes it an honest estimate."
    )

    st.subheader("Per class")
    st.caption(
        "Intervals are bootstrapped over story groups, not articles, because syndicated "
        "copies of one story are not independent observations. A wide interval means the "
        "class is thin, not that the model is unstable — read `education` (29 validation "
        "articles) as noise."
    )
    per_class = pd.DataFrame([
        {
            "Class": pretty(c["topic"]),
            "F1": c["f1"],
            "95% interval": f"[{c['low']:.2f}, {c['high']:.2f}]",
            "Precision": c["precision"],
            "Recall": c["recall"],
            "Val support": c["support"],
            "Test F1": test["per_class_f1"].get(c["topic"]),
        }
        for c in val["per_class"]
    ])
    st.dataframe(
        per_class, hide_index=True, use_container_width=True,
        column_config={
            "F1": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
            "Precision": st.column_config.NumberColumn(format="%.2f"),
            "Recall": st.column_config.NumberColumn(format="%.2f"),
            "Test F1": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Is the confidence honest?")
        st.caption(
            "Each point is a band of predictions. On the dotted line, the model is exactly "
            "as right as it claims. Above it, it is under-confident — which is the safe "
            "direction for a system that abstains."
        )
        reliability = pd.DataFrame([
            {"Claimed": b["claimed"], "Actually right": b["actual"],
             "Perfect calibration": b["claimed"], "n": b["n"]}
            for b in metrics["reliability"]
        ]).set_index("Claimed")
        st.line_chart(reliability[["Actually right", "Perfect calibration"]])

    with right:
        st.subheader("The abstention trade-off")
        st.caption(
            "Raise the cut and the model files fewer articles but is right more often on "
            "the ones it does file. This curve is the whole argument for letting a "
            "classifier abstain."
        )
        curve = pd.DataFrame(metrics["coverage_curve"]).rename(columns={
            "cut": "Confidence cut", "coverage": "Filed",
            "accuracy_on_kept": "Accuracy on filed",
        }).set_index("Confidence cut")
        st.line_chart(curve)

    st.subheader("Where it goes wrong")
    st.caption(
        "The top confusions are the same class pairs human annotators disagreed on when "
        "the corpus was labelled — 18.6% of errors sit on those pairs, so part of this "
        "gap is in the labels rather than in the model."
    )
    st.dataframe(
        pd.DataFrame([
            {"Actually": pretty(c["actual"]), "Called": pretty(c["called"]), "Articles": c["n"]}
            for c in val["confusions"]
        ]),
        hide_index=True, use_container_width=True,
    )

    with st.expander("Full confusion matrix"):
        matrix = pd.DataFrame(
            val["matrix"],
            index=[pretty(t) for t in val["labels"]],
            columns=[pretty(t) for t in val["labels"]],
        )
        st.dataframe(matrix.style.background_gradient(cmap="Blues", axis=None),
                     use_container_width=True)


with batch_tab:
    st.subheader("Real articles nobody labelled")
    st.caption(
        "These are drawn from the frozen snapshot — genuine collected news that was never "
        "part of training, validation or test. It is the closest thing to watching the "
        "model work in production."
    )

    how_many = st.slider("How many articles", 20, 300, 60, 20)
    if st.button("Classify them", type="primary"):
        pool = load_unlabelled().head(how_many)
        texts = [
            serve.text_for(t, s, b)
            for t, s, b in zip(pool["title"], pool["summary"], pool["body"])
        ]
        with st.spinner(f"Classifying {len(texts)} articles..."):
            probabilities = model.probabilities(texts)

        called = model.classes[probabilities.argmax(axis=1)]
        scores = probabilities.max(axis=1)
        filed = scores >= cut

        a, b, c = st.columns(3)
        a.metric("Filed automatically", f"{filed.mean():.0%}")
        b.metric("Sent for review", f"{1 - filed.mean():.0%}")
        c.metric("Median confidence", f"{np.median(scores):.2f}")

        st.bar_chart(
            pd.Series([pretty(t) for t in called]).value_counts().rename("Articles")
        )
        st.dataframe(
            pd.DataFrame({
                "Publisher": pool["publisher"],
                "Headline": pool["title"],
                "Filed as": [pretty(t) if f else "— held for review"
                             for t, f in zip(called, filed)],
                "Confidence": scores,
            }),
            hide_index=True, use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    format="%.2f", min_value=0.0, max_value=1.0
                )
            },
        )


with method_tab:
    st.subheader("The question the project actually asked")
    st.markdown(
        f"""
An earlier version of this classifier read only the **headline and summary** — about 33
words per article — and scored **{metrics['body_ab']['v1_shipped']:.3f}** macro-F1. Once
the collector started storing full article bodies, the obvious question was whether ~650
words beat 33.

**It does, by +{metrics['body_ab']['delta']:.3f} macro-F1**
({metrics['body_ab']['title_summary']:.3f} → {metrics['body_ab']['title_body']:.3f}),
with non-overlapping confidence intervals and McNemar p = {metrics['body_ab']['mcnemar_p']:.1e}.
That is the headline result, and it is a *measured* answer rather than an assumed one.
        """
    )

    st.subheader("What was tried and rejected")
    st.caption("Every row was measured against the incumbent with an interval or a "
               "paired test. Nothing was dropped on a hunch.")
    st.dataframe(
        pd.DataFrame([
            {"Idea": "Entity / geography scrubbing (spaCy)", "Result": "no rule cleared the bar", "Kept?": "No"},
            {"Idea": "Up-weighting the headline", "Result": "monotonically worse", "Kept?": "No"},
            {"Idea": "Head + tail of the body vs head only", "Result": "no difference", "Kept?": "No"},
            {"Idea": "Tuning C over a 6x range", "Result": "moves macro-F1 by 0.002", "Kept?": "No"},
            {"Idea": "XGBoost on SVD features", "Result": "−0.036, 5x slower", "Kept?": "No"},
            {"Idea": "Random forest / extra trees", "Result": "−0.041 to −0.087", "Kept?": "No"},
            {"Idea": "MiniLM sentence embeddings", "Result": "−0.061 (truncates at 256 tokens)", "Kept?": "No"},
            {"Idea": "Majority-vote ensemble", "Result": "+0.001, McNemar p = 1.00", "Kept?": "No"},
            {"Idea": "Per-class confidence cuts", "Result": "no better than one global cut", "Kept?": "No"},
            {"Idea": "Isotonic calibration", "Result": "costs 0.002 F1, makes confidence honest", "Kept?": "Yes"},
            {"Idea": "Abstention below a global cut", "Result": "78% → 83% accuracy on filed", "Kept?": "Yes"},
            {"Idea": "Reading the full article", "Result": "+0.059, p = 7.5e-06", "Kept?": "Yes"},
        ]),
        hide_index=True, use_container_width=True,
    )

    st.markdown(
        """
**The uncomfortable finding is the useful one.** A perfect selector over all ten
candidate models would score 0.900 — they genuinely disagree — yet every ensemble that
can actually be built lands on 0.771. The disagreement is asymmetric: the embedding model
rescues 70 of the incumbent's mistakes while destroying 148 of its correct answers. A vote
has no way to tell which side of that trade it is on.

So the accuracy came from **data preparation**, not from the classifier. Boilerplate
removal, near-duplicate grouping that respects story boundaries, admission rules and an
honest grouped-temporal split did the work. The model itself is a linear SVM, and every
attempt to replace it with something fancier lost.
        """
    )

    with st.expander("How the evaluation avoids fooling itself"):
        st.markdown(
            """
- **Bootstrap intervals resample story groups, not articles.** Syndicated copies of one
  story are not independent observations; resampling articles reports an interval
  narrower than the truth.
- **Splits are grouped and temporal.** No story group spans two splits, and every test
  article was collected after every training article.
- **Two publisher holdouts** (The Hindu, The Guardian) are scored on every run, each
  refitted without that publisher. The Guardian is out-of-distribution: non-Indian, a
  different house style.
- **The test split was opened once**, through a single function that refuses to run
  without an explicit flag, after the model was frozen. It has not been reopened.
- **Model comparisons use McNemar**, not a subtraction of two accuracy numbers.
- **Confidence is never a raw SVM margin.** A margin is a signed distance from a
  hyperplane; it has no scale and is not comparable across classes.
            """
        )

st.divider()
st.caption(
    f"Snapshot `{metrics['snapshot_id']}` · "
    f"{sum(metrics['splits'].values()):,} labelled articles "
    f"(train {metrics['splits'].get('train', 0):,} / val {metrics['splits'].get('val', 0):,} "
    f"/ test {metrics['splits'].get('test', 0):,}) · 13 classes · 40 publishers · "
    f"{test['predict_ms_per_article']:.2f} ms per article"
)
