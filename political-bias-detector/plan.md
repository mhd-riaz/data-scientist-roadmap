# Political Bias Detector — Plan

Status: **DRAFT — awaiting approval on the five decisions in [§3](#3-known-inconsistencies--decisions-needed).**
Stage: `plan` (see [Definition of done](#9-definition-of-done) before moving to `explore`).

## 1. Problem statement

Retrieve politics-related articles from the `news` corpus, analyse `title` / `summary` /
`content`, and produce a **bias score** per article: a continuous value in `[0, 1]`,
thresholded at a single named constant `BIAS_THRESHOLD = 0.7` into **biased** /
**unbiased**. The threshold is a default to be justified with a sweep, not asserted.

This is a **separate project** from
`semester-1/06-supervised-ml-classification/mini-project/news` (the "mini-project"). We
read its live MongoDB database and its notes for reference; we do not import, depend on,
or edit any of its code (`ml/`, `taxonomy.yaml`, `newsml`, etc.).

## 2. Data inventory (from the live database, 2026-08-25)

Connected via the MongoDB MCP tool to `192.168.31.233:27017` (VS Code extension
connection — same homelab instance the mini-project's memory notes describe), database
`news`, collection `articles`. All figures below are measured, not assumed.

| Fact                | Value                                                                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Total articles      | **10,379**                                                                                                                                           |
| Distinct sources    | **97**                                                                                                                                               |
| `collected_at` span | 2026-08-22 17:27 UTC → 2026-08-25 07:40 UTC (~2.6 days; the collector is continuously running, so re-running this notebook later will see more rows) |
| `published_at` span | 2008-06-29 → 2026-08-25 (long tail; bulk is recent per the mini-project's notes)                                                                     |
| `language`          | **100% `en`** (10,379/10,379) — no language filtering needed                                                                                         |

**Field coverage** (non-empty, by character length, not just non-null):

| Field        | Non-empty       | Coverage            | Notes                                     |
| ------------ | --------------- | ------------------- | ----------------------------------------- |
| `title`      | 10,379 / 10,379 | 100.0%              | never empty                               |
| `summary`    | 7,639 / 10,379  | 73.6%               | avg length 376 chars where present        |
| `content`    | 9,429 / 10,379  | 90.8%               | avg length 3,714 chars where present      |
| `categories` | 6,930 / 10,379  | 66.8% (≥1 category) | free-text tags, not a controlled taxonomy |

**Schema** (`articles`, 26 fields): identity/dedup (`_id`, `dedup_id`, `content_hash`,
`normalized_url`, `feed_guid`), source linkage (`source_id`, `source_name`), content
(`title`, `summary`, `content`, `categories`, `authors`, `image_url`), provenance
(`url`, `canonical_url`, `language`, `country`, `state`, `city`), pipeline state
(`processing_status`, `scrape_status`, `scrape_attempts`, `scraped_at`, `next_scrape_at`,
`collected_at`, `published_at`). **`source_id`, `source_name`, `url`, `canonical_url`,
`country`, `state`, `city` are the leakage-risk fields per ground rule 4** — never used
as model features.

### How politics-related articles can be identified

Three approaches were measured directly against the corpus:

| Method                                                                                                                                                                                       | Matched | Share | Publisher concentration                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `categories` contains "polit" (any tag)                                                                                                                                                      | **123** | 1.2%  | **~95% from a single publisher** (The Guardian's various sections: World, Politics, Society, Business, ...)                                                                                 |
| Generic political keyword regex on `title`+`summary` (election, parliament, minister, president, senate, coalition, lawmaker, ...)                                                           | 1,614   | 15.5% | broad, but tuned on Western political vocabulary only                                                                                                                                       |
| `categories` OR **India-aware** keyword regex (adds Lok Sabha, Rajya Sabha, BJP, chief minister, panchayat, assembly election, etc. — the corpus is majority Indian regional/national press) | **872** | 8.4%  | spread across 15+ sources, no single source above ~12% (The Hindu — National 106, The New Indian Express 84, Deccan Herald 63, The Guardian — World 54, HT — India 47, India Today 46, ...) |

**Category tags alone are unusable as the politics filter**: 1.2% coverage is too sparse
to sample from, and using it would mean the "politics filter" is really detecting "is
this a Guardian article" — the corpus's only publisher that runs a dedicated Politics
RSS section. See [§3.5](#35-political-is-itself-a-classification-problem) for the
recommended filter and how its own precision will be measured.

## 3. Known inconsistencies — decisions needed

Each one is stated with options, trade-offs, and a recommendation. **None of these are
decided yet** — this section is the approval gate. See [§8](#8-decision-log) for what
gets recorded once you approve or redirect each one.

### 3.1 No bias label exists in the corpus

This is the single biggest decision in the project — nothing past this can start.

| Option                                                                                                      | What it means                                                                                                                                                                        | Trade-off                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Hand-label a gold set**                                                                                | An analyst reads a stratified sample of the 872 politics-filtered articles and rates each (rubric TBD: loaded language, one-sided sourcing, opinion framing presented as news, etc.) | Directly measures what we claim to measure; labour-bound (~300–400 articles is a realistic session); single-annotator noise ceiling, as the mini-project found (52–87% self-agreement across its rounds)                                                                                                                                                                                                                                                                           |
| **B. Weak labels from publisher-level lean priors** (e.g. AllSides, Ad Fontes Media, Media Bias/Fact Check) | Every article from a rated publisher inherits that publisher's rating                                                                                                                | Scales to the whole corpus for free, **but** those rating sites are curated almost entirely for US/UK national outlets — this corpus is dominated by Indian regional/national press (The Hindu, IE, HT, Deccan Herald, ToI, NDTV, India Today, city desks) that such rating sites generally do not cover. Where it does apply, the label is a **publisher fingerprint**, which is exactly the leakage ground rule 4 forbids unless every evaluation is strictly publisher-held-out |
| **C. External labelled bias dataset** (e.g. a hyperpartisan-news or media-bias corpus)                      | Train on someone else's labelled news, apply here                                                                                                                                    | Different domain (US-partisan English news vs. our India-heavy corpus) — a domain-transfer problem with no guarantee of validity, and we'd be importing a labelling methodology we can't audit                                                                                                                                                                                                                                                                                     |

**Recommendation: A, primary.** Hand-label a stratified sample of the 872-article
politics set (stratified by source and time) as the actual supervision — not just a
validation slice, since there is no usable weak signal to validate against here. This
mirrors the pattern that worked for the mini-project's topic classifier (gold labels as
ground truth, weak labels only where independently available), except here B's coverage
gap makes weak labels not viable as the primary signal for *this* corpus. Treat B/C as
optional secondary signal only if A proves too small to train on.

### 3.2 Score versus class

**Recommendation:** train a binary classifier on the gold labels (biased=1 /
unbiased=0), and take the model's **calibrated probability of the positive class** as
the bias score — this is exactly the vocabulary of
[03-logistic-regression.md §2.2](../semester-1/06-supervised-ml-classification/notes/03-logistic-regression.md#22-reading-the-output-as-a-probability)
($h_\theta(x) = P(y=1\mid x;\theta)$), and lets every model in the ladder ([§6](#6-model-ladder))
be compared on the same footing after calibration (Platt scaling / isotonic regression —
API to be confirmed via Context7 before use). `class = score > BIAS_THRESHOLD`. This
avoids needing a labelled bias *magnitude* scale, which is harder to annotate
consistently than a binary judgement.

### 3.3 The 0.7 cut is arbitrary on an uncalibrated classifier

**Recommendation:** keep `BIAS_THRESHOLD = 0.7` as the single named constant the spec
requires, calibrate the classifier's output first (so "0.7" means something), then run
the threshold sweep from
[07-performance-metrics.md §4](../semester-1/06-supervised-ml-classification/notes/07-performance-metrics.md#4-the-classification-threshold-and-the-precisionrecall-trade-off)
(0.1 → 0.9) and report precision/recall/F1 at each step, with support counts. The
evaluation stage states plainly whether 0.7 sits near the F1 peak, trades recall for
precision, or is simply not defensible on this data.

### 3.4 `content` missing for a meaningful share, and it tracks source, not topic

Measured: `content` is empty for 9.2% of the corpus (950 articles) — much better than
the mini-project's earlier snapshots (they saw ~35% and later ~9–10% as their scraper
caught up), but not zero, and not necessarily uniform across sources.

**Recommendation:** build a fallback text field, `text = content, else summary, else
title` (in that priority order) — independently implemented for this project, not
imported from the mini-project's equivalent `lede` field. Restricting to `content` alone
would silently and non-randomly exclude ~9% of articles. The exploration stage will
report content availability **per source** before this is finalised, since if the gap is
concentrated on a handful of publishers, that is itself a leakage risk to flag under
ground rule 4.

### 3.5 "Political" is itself a classification problem

**Recommendation:** ship the rule-based filter from [§2](#2-data-inventory-from-the-live-database-2026-08-25)
(category tag OR India-aware keyword regex, 872 articles) as v1, and **measure its own
precision** with a manual audit — a human reads a random sample of ~60 matched articles
and ~40 near-miss articles (matched by only one signal, or rejected but topically
adjacent, e.g. "government policy on X" without an explicit political keyword) and
reports estimated precision/recall for the filter itself, per ground rule 5. A full
second classifier for this step is not proportionate to its role in the project.

## 4. Feature plan

- **Primary text representation**: TF-IDF over the fallback `text` field
  ([§3.4](#34-content-missing-for-a-meaningful-share-and-it-tracks-source-not-topic)),
  word n-grams (1–2) as the default rung; char n-grams considered as a secondary
  experiment if word n-grams underperform. `TfidfVectorizer` signature to be confirmed
  via Context7 before use.
- **Stylistic/linguistic features**, each added only with a stated reason: text length,
  sentence count and average sentence length, exclamation/question-mark rate, quoted-text
  share (proxy for on-the-record sourcing vs. unsourced assertion), first/second-person
  pronoun rate, a subjectivity/sentiment lexicon score (library TBD — `nltk` or
  `spacy`/`textblob`, confirmed via Context7 at implementation time and only added if it
  measurably helps).
- **Never as features** (ground rule 4): `source_id`, `source_name`, `url`,
  `canonical_url`, `country`, `state`, `city`, or anything derived from the RSS feed
  section. Split before fitting; where sample size allows, add a publisher-held-out
  check the same way the mini-project validated its topic classifier against leakage.

## 5. Model ladder

Following the algorithms taught in
[semester-1/06-supervised-ml-classification/notes](../semester-1/06-supervised-ml-classification/notes/00-study-checklist.md),
in increasing strength, never a single unexplained model:

1. **Majority-class baseline** — establishes the accuracy-trap floor per
   [07-performance-metrics.md §1](../semester-1/06-supervised-ml-classification/notes/07-performance-metrics.md#1-why-accuracy-alone-is-not-enough).
2. **Logistic Regression** on TF-IDF + stylistic features — the taught model that
   natively emits $P(y=1\mid x)$.
3. **K-Nearest Neighbours** (scaled features, per
   [04-knn.md](../semester-1/06-supervised-ml-classification/notes/04-knn.md)).
4. **Decision Tree** (entropy / information gain, per
   [05-decision-trees-and-id3.md](../semester-1/06-supervised-ml-classification/notes/05-decision-trees-and-id3.md)) —
   interpretable, gives a feature-importance view.
5. **Ensemble** — Random Forest and/or Gradient Boosting (per
   [06-ensemble-learning.md](../semester-1/06-supervised-ml-classification/notes/06-ensemble-learning.md) /
   [06b](../semester-1/06-supervised-ml-classification/notes/06b-ensemble-methods-deep-dive.md)) —
   the strongest rung, feeding the "what works" conclusion.

Every rung's output is calibrated to a genuine probability before it is called a "bias
score", regardless of which rung wins the ladder.

## 6. Evaluation plan

Using [07-performance-metrics.md](../semester-1/06-supervised-ml-classification/notes/07-performance-metrics.md)'s
exact vocabulary:

- Confusion matrix (biased vs. unbiased), TP/FP/FN/TN labelled explicitly.
- Precision, recall, specificity, F1 **per class**, support printed beside every one —
  a recall of 1.000 on 7 articles is noise and will be labelled as such.
- Macro vs. micro averaging, noting micro = accuracy in this binary case.
- ROC curve and AUC.
- Threshold sweep table centred on 0.7 (0.1 → 0.9, step 0.1, 0.7 called out explicitly),
  matching the "Worked sweep" style of
  [07 §4](../semester-1/06-supervised-ml-classification/notes/07-performance-metrics.md#4-the-classification-threshold-and-the-precisionrecall-trade-off).
- Stratified train/validation/test split; publisher-held-out check per
  [§4](#4-feature-plan) if sample size allows once the gold set size is known.

## 7. Stage list

| #   | Stage    | Deliverable                                                                                                              | Status                                                                               |
| --- | -------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| 1   | Plan     | This document                                                                                                            | **Draft — awaiting approval on §3**                                                  |
| 2   | Explore  | Notebook §1: counts, coverage, politics-filter precision audit, class balance, publisher/time distribution, sample reads | Not started                                                                          |
| 3   | Clean    | Notebook §2: HTML stripping, Unicode/whitespace normalisation, dedup, exclusion rules, each with its removal count       | Not started                                                                          |
| 4   | Label    | Export stratified gold-labelling sample, write rubric, import returned labels                                            | Not started — depends on §3.1 approval and an annotator (you, or someone you assign) |
| 5   | Features | Notebook §3: TF-IDF + stylistic features, no-leakage check                                                               | Not started                                                                          |
| 6   | Model    | Notebook §4: baseline ladder                                                                                             | Not started                                                                          |
| 7   | Evaluate | Notebook §5: confusion matrix, per-class metrics, ROC/AUC, threshold sweep                                               | Not started                                                                          |
| 8   | Conclude | Notebook §6: honest number, what works, what doesn't, next steps                                                         | Not started                                                                          |
| 9   | Verify   | Headless `nbconvert --execute`, zero stored outputs                                                                      | Not started                                                                          |
| 10  | Report   | File list, metrics with support, decisions, open questions                                                               | Not started                                                                          |

## 8. Decision log

| Date       | Decision                                                                                                                          | Rationale                                                                                                                                          |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-08-25 | Data inventory established directly from the live `news` database (10,379 articles, 97 sources, 100% English) rather than assumed | Ground rule: report from the database, not assumption                                                                                              |
| 2026-08-25 | Category tag alone rejected as the politics filter (1.2% coverage, ~95% one publisher)                                            | Would make the filter detect "is this The Guardian", not "is this political"                                                                       |
| 2026-08-25 | Proposed politics filter: category tag OR India-aware keyword regex (872 articles, 15+ sources)                                   | Balances coverage against publisher concentration; precision still needs a manual audit ([§3.5](#35-political-is-itself-a-classification-problem)) |
| *pending*  | §3.1 labelling strategy                                                                                                           | Awaiting your approval of recommendation A (hand-label a gold set)                                                                                 |
| *pending*  | §3.2 score-vs-class framing                                                                                                       | Awaiting your approval of recommendation (calibrated probability)                                                                                  |
| *pending*  | §3.3 threshold justification method                                                                                               | Awaiting your approval of recommendation (sweep, keep 0.7 as shipped default)                                                                      |
| *pending*  | §3.4 text field choice                                                                                                            | Awaiting your approval of recommendation (`content` → `summary` → `title` fallback)                                                                |
| *pending*  | §3.5 politics-filter validation                                                                                                   | Awaiting your approval of recommendation (rule-based v1 + manual precision audit)                                                                  |

## 9. Definition of done (for this `plan` stage)

- [x] `plan.md` exists under `political-bias-detector/`.
- [x] Data inventory is measured from the live database, not assumed.
- [x] All five known inconsistencies are stated with options, trade-offs, and a
      recommendation.
- [ ] **You have approved, amended, or redirected each of the five decisions in §3.**
- [ ] Decision log ([§8](#8-decision-log)) reflects your actual choices, not just the
      proposals.

Nothing under `explore` starts until the checkboxes above are complete.

## Open questions (beyond the five known inconsistencies)

1. Who hand-labels the gold set, and how many articles is realistic in one pass? (300–400
   is the plan's placeholder, not a commitment.)
2. Should the label rubric allow a third bucket (e.g. "opinion piece — excluded from
   scoring") or stay strictly binary?
3. Does a single annotator label everything (as the mini-project's early rounds did,
   later found to have a 52–87% self-agreement ceiling), or should a small overlap slice
   be double-labelled to measure annotator noise before trusting the model's numbers
   against it?
4. Is the bias score ever computed for non-politics articles, or is scope strictly
   "politics-filtered articles only" for this whole project?
