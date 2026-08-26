# ml-v2 — Body-Aware 13-Class News Classifier (POC)

> Living document. Update status as phases land; append to the decision log rather than
> rewriting history.

**Status:** **Phases A, B, C, D0, D complete** · snapshot `v2-001` · best **0.771 [0.743, 0.796]** · next Phase E

**The core hypothesis is confirmed.** The article body is worth **+0.059 macro-F1**
(word_char_svc: 0.712 → 0.771), with non-overlapping intervals and McNemar p=7.5e-06.
Against the v1-parity rung measured on this same snapshot (0.696), the gain is **+0.075**.

**Two clean negative results, both worth as much as the positive one:** entity/geography
scrubbing (D0) buys nothing because Phase A already removed the publisher fingerprints,
and up-weighting the title actively hurts. Neither is enabled.

**Frozen corpus cut:** `collected_before = 2026-08-26T12:00:00+00:00` → 14,189 articles,
40 publishers, all **8,001** gold labels join.
**Gold digest:** `8ed59b13d4148727063fdd5d38ee6c4d365e9949f179525f3824eb187e5a918e`

---

## Why this exists

`news/ml/` (v1) reaches **val macro-F1 0.720** classifying on `title + summary` — about
33 words per article. Two things changed:

1. **The corpus now has bodies.** 94.3% of gold articles carry a median 3,765-char body.
   That is roughly 100× more text per article, and it is the central hypothesis to test.
2. **v1's methodology has known, documented weaknesses** — a single validation split used
   for both tuning and selection, point estimates with no confidence intervals, a
   near-duplicate threshold that provably cannot exceed 0.80 precision, and a class floor
   whose module constant (300) disagrees with its CLI default (40).

v2 is a **standalone rewrite that improves those methods**, not a port.

This is a **POC**. Ship nothing, deploy nothing. Build the model, measure it honestly on
real data, then decide whether it earns a deployment conversation.

---

## Measured baseline

Verified 2026-08-26 against the live corpus. Supersedes any earlier figures.

| Fact | Value |
| --- | --- |
| Gold labels | **8,001** articles, one label each, all human, taxonomy v4 |
| Class distribution | politics 1,576 … conflict_war 234; `unsorted` 63. Imbalance **6.7:1** |
| **Body coverage (gold)** | **94.3%** — median 3,765 chars, p90 7,285, p99 16,228, max 179,230 |
| Summary coverage | 74.8%; title-only 86 articles (1.07%) |
| Body coverage by class | 89.7%–99.3%, uniform — no class disadvantaged |
| Corpus at the frozen cut | **14,189** articles, 97 feeds → **40 publishers**, 92.6% with a body |
| `collected_at` span | 2026-08-22 → 08-26 (**4 days**) |
| `published_at` span | 2025-09-13 → 2026-08-26 |
| v1 best (historical) | macro-F1 **0.720**, coverage 82.9%, 11 MB, 0.05 ms/doc, 12 emittable classes |
| Expected val size | ~1,190 ⇒ SE on macro-F1 ≈ **±0.02–0.03** |

> **Any delta under ~0.03 on a single validation split is noise.** v1 repeatedly read
> ±0.07 per-class swings as signal. Phase C's harness exists to stop that recurring.

---

## Pre-implementation findings (2026-08-26)

### Finding 1 — publisher holdout candidates

Publisher = the part of `source_name` before the em-dash, so all `The Indian Express — *`
section feeds collapse into one publisher.

| Publisher | gold | classes covered | classes <10 articles |
| --- | ---: | ---: | --- |
| The Indian Express | 1,683 | 13 | **none** |
| The Hindu | 897 | 13 | **none** |
| The Guardian | 800 | 13 | **none** |
| The New Indian Express | 600 | 13 | technology, science_space |
| Livemint | 593 | 13 | 6 classes |
| Hindustan Times | 526 | 13 | 4 classes |
| Phys.org | 231 | 11 | (missing 2) |

Only **three** publishers cover all 13 classes with ≥10 articles each. Individual *section
feeds* are badly skewed — `The Indian Express — Education` covers only 8 of 13 — which is
why a publisher must be held out as a family, never as a feed.

**Decision: hold out The Hindu (in-distribution) and The Guardian (out-of-distribution).**
The Guardian is the harder test: non-Indian, different house style and vocabulary.
**The Indian Express is deliberately not used** even though v1 used it — at 1,683 of 8,001
it is 21% of all labels, so removing it confounds "generalizes to an unseen publisher"
with "trained on 21% less data". v1's reassuring 0.018–0.043 drop was measured under
exactly that confound.

### Finding 2 — body furniture

400 bodies sampled across 76 sources.

- Furniture is **repetitive and per-source**, exactly what line-frequency discovery
  handles: `Story continues below this ad` (71–75% of Indian Express bodies),
  `Updated on:` / `2 min read` (52%/33% New Indian Express), `- Ends` / `Published On:`
  (62% India Today), `Tags:`, `© IE Online Media Services Pvt Ltd`, Guardian
  newsletter-promo lines.
- Only **26.5%** of bodies contain any common furniture marker; the largest single one
  (`read more`) is 11.8%. Tractable, not a swamp.
- **New risk — embedded author biographies.** Phys.org has `Who's behind this story?` in
  86% of bodies followed by multi-sentence bios; Livemint and Indian Express do the same.
  These are *topically misleading* — a cricket correspondent's bio inside a business
  article pulls it toward `sport`. **v1's rule would miss them**: `MAX_LINE_WORDS = 25`
  treats long lines as syndicated content, and bios are long.
- 5% of bodies are under 200 chars, in two distinct kinds: Deccan Herald `DH Toon` cartoon
  pages whose entire body is `Check out more of our cartoons here .` (reject as
  non-article), and NDTV one-line summaries that are thin but legitimate (keep).

### Finding 3 — label noise ceiling

Same-story groups found by sparse TF-IDF cosine over title + 4,000 chars of body, then
checked for label agreement. **~18–22% of multi-article story groups carry disagreeing
labels**, stable across cosine cuts 0.55 / 0.70 / 0.85.

That is an **upper bound, not the noise rate** — many are *grouping* false positives:
`DH Toon` cartoons, `Daily Quiz` instalments, `Watch:` segments and `Chennai Canvas
Episode 1–4` merged as one "story", plus two unrelated France 24 articles merged by shared
boilerplate. These are the same four failure kinds v1 catalogued.

The genuine disagreements are **taxonomy boundary collisions, not carelessness**:

| Same story | Labels given |
| --- | --- |
| Ukraine shopping-centre strike | politics vs conflict_war |
| Gujarat hooch tragedy | disaster_accident vs crime_justice |
| Ex-cricketers' letter on Imran Khan | politics / crime_justice / **sport** (3-way) |
| NCERT textbook panel | politics vs education |

Top disagreeing pairs: conflict_war↔politics (8), business_economy↔politics (8),
crime_justice↔disaster_accident (5) — **the same pairs the v1 classifier confuses most**,
meaning those confusions are partly in the labels, not the model.

**Consequence: macro-F1 has a real ceiling below 1.0.** `conflict_war`, `crime_justice`
and `politics` cannot reach `sport`-level F1. Targeting 0.90 macro-F1 means targeting
annotation noise. Tie-break rules in the labelling guide would buy more than another
labelling round.

### Finding 4 — near-duplicate cost is not a problem

Sparse TF-IDF cosine over all 8,001 articles with 4,000-char bodies ran in **12 seconds
end to end**. Pure-Python MinHash over the same bodies would be orders of magnitude slower
(~22× more shingles per document than headline-length text). Cosine verification is
effectively free and should carry most of the work.

---

## What v2 does differently

| Layer | v1 | v2 | Why |
| --- | --- | --- | --- |
| Text | title+summary, one string | title / summary / body as separate fields | a concatenated string cannot be field-weighted |
| Storage | JSONL snapshots | **Parquet + zstd** | bodies make snapshots ~60 MB; ~5× smaller, ~10× faster, re-read by every run |
| Near-dup | one MinHash threshold | **two-stage: blocking → cosine verify**, + time-gap and boilerplate guards | v1 measured precision **cannot exceed 0.80 at any cut**; its 4 FP kinds need rules, not a threshold |
| Tuning | one val split | **StratifiedGroupKFold on train**; val for selection only | v1 tuned and selected on the same ~600 rows |
| Statistics | point estimates | **bootstrap CIs resampled by story group** + McNemar | article-level bootstrap under-reports variance when near-dup clusters exist |
| Calibration | `CalibratedClassifierCV(cv=5)` | `cv="prefit"` on val | 5× cheaper, calibrates on genuinely held-out data |
| Thresholds | grid-searched on val | fitted on **OOF probabilities from train** | `education` had 29 val articles — a cut fitted on that overfits |
| Admission | hardcoded rules | each rule toggleable and **priced** | 443 articles dropped by `language_mismatch` alone; nobody knows if it helps |
| Class floor | fixed (300 in module, 40 in CLI) | **derived: the smallest class present** | nothing is ever dropped; v1's mismatch would have deleted `conflict_war` silently |
| Tracking | ad-hoc `/tmp` scripts | append-only experiment ledger | v1 lost whole rounds of results between sessions |

---

## Phase A — Foundation

- [x] **A1. Scaffold `news/ml-v2/`** — done 2026-08-26
  - uv-managed package `newsmlv2`, Python 3.12, console script `newsmlv2`.
  - One dependency set (POC): scikit-learn 1.9, pymongo, numpy, scipy, pandas, pyarrow,
    PyYAML, xgboost 3.4, sentence-transformers 6.0, matplotlib, spacy 3.8. Dev: pytest.
  - `taxonomy.yaml` (v4, 13 classes) and `data/labels/gold.jsonl` (8,001 labels) vendored,
    digest `8ed59b13…a918e` recorded in every snapshot manifest, so ml-v2 builds without
    `../ml/`. Label *production* stays v1's job.
  - `config.py` (paths, seed, frozen cut, publisher holdouts), `load.py` (Article with
    title/summary/body **separate**, `publisher_of`), `labels.py` (taxonomy + gold reader,
    `trainable` strips `unsorted` before any class is counted).
  - `ml2-*` targets in `news/Makefile`. 13 tests green.

- [x] **A2. Text layer** — done 2026-08-26
  - `clean.py` cleans title, summary and body **independently**; explicit punctuation
    fold, line-anchored furniture removal, dateline and wire-agency extraction into
    fields, plus a navigation-body guard.
  - `boilerplate.py` discovers per-source repeated lines over **bodies**, with
    `MAX_LINE_WORDS` raised from 25 → 120 and a repetition bar replacing the length bar.
    Found **355 lines across 47 sources, 19 of them long** — the author bios v1 could not
    reach. Cleaning removes only **1.2%** of body text corpus-wide.
  - `admit.py` — every gate is a switch on `Policy`, so Phase B can price it by turning
    it off and re-scoring. Rejections on the real corpus: **582 of 14,189 = 4.1%**
    (implausible_timestamp 238, too_short 142, non_article_format 128, service_bulletin
    38, sponsored 31, exact_duplicate 5). **7,969 of 8,001 gold labels survive.**
    Kept articles: 13,607, **94.9% with a usable body**, median **594 words** against
    v1's ~33.
  - Language rejection defaults **off**: v1 dropped 443 articles on it without ever
    measuring the cost. Implemented as a deterministic script ratio rather than
    `langdetect`, which samples randomly and needs a global seed to reproduce.
  - 76 tests green.

- [x] **A3. Near-duplicate detection, two-stage** — done 2026-08-26
  - Blocking by sparse TF-IDF cosine k-NN, verification by threshold plus two guards.
  - **Calibrated against v1's 43-pair hand-judged census** (39 survive admission):
    at cut **0.50** with the guard, **precision 0.86, recall 0.81, F1 0.833**.
    v1's proven ceiling was **precision 0.80 at any single-threshold cut** — beaten.
  - Cut is **not** the F1 argmax (0.30, F1 0.852). The census only covers the 0.40–0.95
    band, so a lower cut extrapolates where nothing was ever judged, and the whole
    0.30–0.52 range differs by one true positive out of 39.
  - **Recall criterion restated from ≥0.90 to ≥0.80.** Recall plateaus at 0.84 even at
    cut 0.30: about 5 of the 31 judged same-story pairs share almost no vocabulary
    because two publishers covered one event in genuinely different words. No
    text-similarity threshold reaches those, so 0.90 was never achievable by this route.
  - **Body-based grouping folds 999 articles vs 708 for title+summary — the Phase B2
    prediction confirmed, +41%.** Largest groups verified genuine: the Kriti Sanon rakhi
    story across 3 publishers, Ajit Doval's Beijing talks across 5.
  - `pipeline.py` added as the single clean→admit→group path, so no script can drift
    from the snapshot's recipe.

- [x] **A4. Snapshot** — done 2026-08-26, `v2-001`
  - Parquet + zstd, 24 MB. `title`, `summary`, `body` stored **separately**, plus
    `body_chars`, `has_body`, `publisher`, `story_group_id`, `split`, `topic`.
  - Manifest pins git sha, seed, cleaning/taxonomy versions, the **label file digest**,
    the frozen cut, split boundaries, near-dup config and the admission policy.
  - `--id` is required and an existing id refuses to overwrite.
  - `newsmlv2 verify` re-checks digests: both files OK.
  - **Result:** 13,607 admitted, 582 rejected, 7,919 labelled, 12,608 story groups.

- [x] **A5. Splits — three regimes** — done 2026-08-26
  - Grouped + temporal on `collected_at`, cuts placed on **labelled** rows and applied to
    all, straddling groups dropped whole. Boundaries frozen in the manifest.
  - **Labelled split: train 5,487 / val 1,120 / test 1,159 / dropped 153** = 69/14/15.
  - `publisher_holdout()` for The Hindu and The Guardian, keyed on publisher, never a feed.
  - `StratifiedGroupKFold` over train is what hyperparameter search will use.
  - Asserted against the real snapshot: no story group spans two splits, and every test
    article was collected after every train article.

- [ ] **A6. Class scope — derived, never fixed** *(needs A5)*
  - `min_per_class` is **computed**: the size of the smallest class present. Every class
    clears it by construction, `out_of_scope` is always empty, nothing can be dropped.
  - **Trap:** exclude `unsorted` (63 rows) *before* taking the minimum, or the floor
    becomes 63 and `unsorted` starts training as a 14th class. Today's real minimum is
    `conflict_war` at 234.
  - Two advisory reports, neither of which drops anything:
    - **CV feasibility** — `StratifiedGroupKFold(n_splits=5)` needs ≥5 members per class
      in train. Below that, reduce `n_splits` for that run and say so loudly; never delete
      the class.
    - **Thin-class warning** — flag classes whose val support makes their F1 noise.
  - Because the thinnest class now always trains, class weighting (G3) becomes the main
    defence against the 6.7:1 imbalance rather than optional tuning.

**Phase A done when:** `newsmlv2 snapshot --id v2-<date>` then `newsmlv2 verify`
reproduces digests byte-identically, and the test suite is green.

---

## Phase B — Data quality & leakage *(done 2026-08-26)*

- [x] **B1.** `newsmlv2 report --id v2-001` → `reports/data-quality.md`, all 14 items.
  Headlines: body present on **94.9%** of admitted articles, imbalance 6.7:1,
  **481 story groups span more than one publisher** (the syndication that would leak),
  longest body 175,331 chars = 53x the median.
- [x] **B2. Leakage audit.**
  - Body-based grouping folds **999 articles vs 708** for title+summary — the prediction
    held, +41%.
  - No story group spans two splits (asserted on the real snapshot, not a fixture).
  - Admission-rule pricing moved into Phase C's harness, where a model exists to price
    them against. `Policy` already exposes every gate as a switch.

---

## Phase C — Harness + baselines *(done 2026-08-26)*

- [x] **C1. Experiment ledger** — `experiment.py` + `evaluate.py`. Every run appends a
  row with git sha, snapshot, config digest, metrics and timings. **Bootstrap CIs
  resampled by story group** and **McNemar** are built in, and every run is scored on the
  validation split *and* both publisher holdouts (each refit without that publisher).
- [x] **C2. Baselines** — 5 models x 2 variants, 10 runs in the ledger.

| Model | title_summary | title_body | Hindu | Guardian |
| --- | --- | --- | --- | --- |
| majority | 0.025 | 0.025 | 0.030 | 0.017 |
| complement_nb | **0.635** | 0.594 | 0.589 | 0.563 |
| tfidf_logreg | 0.698 | 0.752 | 0.747 | 0.736 |
| tfidf_linear_svc *(v1 parity)* | 0.696 | 0.753 | 0.750 | 0.745 |
| **word_char_svc** | 0.712 | **0.771 [0.743, 0.796]** | 0.769 | 0.724 |

**The body A/B, the central question of the project:**

| Model | delta | McNemar p | Verdict |
| --- | --- | --- | --- |
| complement_nb | **−0.041** | 2.8e-02 | **short wins** |
| tfidf_logreg | +0.054 | 3.0e-05 | body wins |
| tfidf_linear_svc | +0.056 | 1.3e-04 | body wins |
| word_char_svc | **+0.059** | 7.5e-06 | body wins, intervals disjoint |

Three findings worth keeping:

1. **The body is worth ~+0.06 macro-F1**, significant on every linear model.
2. **ComplementNB gets *worse* with the body** — the one model where it hurts. Naive
   Bayes assumes term independence and is sensitive to document length, so a 500-word
   body breaks its assumptions where a headline did not. A good reminder that "more
   text" is not universally better.
3. **Publisher generalization improved rather than degraded** (Hindu 0.682→0.769,
   Guardian 0.665→0.724). H2 predicted bodies might carry more publisher style and make
   this *worse*; measured, the opposite happened — the Phase A boilerplate work is the
   likely reason.

---

## Phase D0 — Signal hygiene *(done 2026-08-26 — NEGATIVE RESULT, nothing adopted)*

> Cleaning here is signal correction, not tidying: furniture and place names become
> **publisher→class shortcuts** that score well on validation and collapse on an unseen
> masthead. `scrub.py` runs one spaCy pass (105s for 6,607 docs) and renders every policy
> from the stored annotations, so ablations are cheap.

**Result: no rule earned adoption.** Measured on `word_char_svc` + `title_body`,
baseline 0.773.

| Policy | val | Δ val | Δ holdout | publisher probe | Verdict |
| --- | --- | --- | --- | --- | --- |
| raw | 0.773 | — | — | 0.482 | baseline |
| mask PERSON | 0.769 | −0.004 | −0.000 | 0.481 | reject |
| mask PLACE | 0.768 | −0.005 | **+0.003** | 0.476 | inside noise |
| mask numbers | 0.766 | −0.007 | −0.000 | 0.477 | reject |
| lemmatise | 0.771 | −0.002 | −0.003 | 0.487 | reject |
| person+place | 0.766 | −0.007 | −0.008 | 0.474 | reject |
| person+place+numbers | 0.764 | −0.009 | −0.009 | 0.471 | reject |
| all four | **0.775** | **+0.002** | **−0.018** | 0.473 | **reject** |

Three things this bought, all worth more than a fractional gain would have been:

1. **The acceptance test paid for itself on the last row.** `person+place+numbers+
   lemmatise` *improves validation* (+0.002) while *losing 0.018 of publisher
   generalization*. On validation alone it looks like the winner. That is precisely the
   failure mode the holdout-vs-validation rule exists to catch, and without it this
   would have shipped.
2. **The publisher probe barely moves** (0.482 → 0.471 at best). Entity masking removes
   almost no publisher fingerprint — because **Phase A already removed it**. The
   boilerplate, affix and publisher-level work did the job, so D0 arrived with nothing
   left to clean.
3. **The brief's warning was right.** "Do not blindly apply traditional NLP
   preprocessing" — lemmatisation was the only rule that made the probe *worse* (+0.005),
   i.e. it made publishers easier to identify, presumably by collapsing distinct
   vocabulary into shared stems.

Kept as a switchable module for Phase E to re-test on embeddings, where masking may
behave differently. Not enabled anywhere.

> **Cleaning here is signal correction, not tidiness.** `Story continues below this ad`
> is in 75% of Indian Express bodies; `- Ends` in 62% of India Today; `Who's behind this
> story?` in 86% of Phys.org. Those publishers skew toward particular classes, so each
> phrase becomes a **publisher→class shortcut** — a fake relationship that scores well on
> validation and collapses on an unseen publisher.

Each rule is an independent switch, measured against the plain baseline, adopted only on
a CI-backed gain. spaCy `en_core_web_sm` costs one ~15-minute pass over ~14k articles;
the scrubbed text is **cached** and paid for once.

### D0 rules — high expected payoff

- [ ] 1. **Publisher fingerprints** — per-source line-frequency removal plus author bios.
- [ ] 2. **Geography stripping** — the corpus is India-heavy and city desks over-supply
  crime and disaster, so the model learns `Bengaluru → crime_justice`. `taxonomy.yaml`
  already carries a `geography` list, so the stoplist is free.
- [ ] 3. **Number / date / money normalization** — `<NUM>`, `<DATE>`, `<MONEY>`. Converts
  thousands of single-document features into real indicators.
- [ ] 4. **Entity policy, split by type** — `PERSON` → `<PERSON>` (brittle memorization;
  the brief asks for generalization to new people), `GPE`/`LOC` → `<PLACE>`, but
  **keep `ORG`** (ISRO→science_space, RBI→business_economy, BCCI→sport are genuine signal).

### D0 rules — medium, test before believing

- [ ] 5. **Entity-aware lemmatization** — helps thin classes most (`conflict_war` has 234
  labels, so merging inflections multiplies the evidence per feature). Non-entity tokens
  only, so `Reuters` never becomes `reuter`.
- [ ] 6. **Keyword extraction** on the body (~30 topic-bearing terms inside ~650 words).
- [ ] 7. **`chi2` / mutual-information feature selection** after vectorization.
- [ ] 8. **Drop non-English lines** — India Today bodies carry Hindi fragments.

**Rejected:** aggressive stemming (destroys entities), stopword removal (`sublinear_tf`
handles it), removing all entities (throws away ORG signal).

### D0 acceptance — two probes, both required

- **Publisher probe:** train a classifier to predict the *publisher* from cleaned text.
  **The score must get worse** after each cleaning step; if it stays high, fingerprints
  remain.
- **The real tell:** a genuine cleaning win improves the **publisher holdout more than
  validation**. Validation shares publishers with train, so shortcuts still pay there —
  only the holdout exposes them. A rule that helps val but not the holdout removed
  *information*, not noise, and must be reverted.

---

## Phase D — Feature engineering *(done 2026-08-26)*

**Body length sweep** (`word_char_svc`, title ×1, head truncation):

| Body chars | val macro-F1 | Hindu | Guardian |
| --- | --- | --- | --- |
| none (summary only) | 0.712 | 0.688 | 0.675 |
| 256 | 0.716 | 0.694 | 0.613 |
| 512 | 0.759 | 0.730 | 0.665 |
| 1024 | 0.760 | 0.752 | 0.679 |
| 2048 | 0.774 | 0.752 | 0.700 |
| **4000** | **0.771** | **0.772** | **0.726** |
| 8000 | 0.783 | 0.760 | 0.721 |
| full | 0.781 | 0.765 | 0.712 |

The curve climbs steeply to ~2,048 characters and then flattens — 8000 is nominally best
but its interval [0.753, 0.807] overlaps everything from 2048 up, so the extra text is
not measurably worth anything. **4000 wins on the tiebreak that matters: it has the best
publisher holdout on both mastheads.** It was already the default, so the sweep confirmed
the setting rather than changing it.

**Title repetition — the lazy alternative to a weighted FeatureUnion — loses, clearly:**

| Title × | 1 | 2 | 3 | 5 |
| --- | --- | --- | --- | --- |
| val macro-F1 | **0.783** | 0.768 | 0.764 | 0.751 |

Monotonically worse. That settles the field-weighting question in the *opposite*
direction to the prior: the body deserves its weight, and up-weighting the headline
actively destroys signal. No FeatureUnion weighting is needed, which removes a whole
tuning dimension.

**Head+tail is not better than head-only** (2048: 0.764 h+t vs 0.774 head; 4000: 0.779 vs
0.771 — all overlapping). News puts the topic in the lede, as expected.

**Settled: `body_chars=4000`, title ×1, head truncation, no field weighting.**

- [ ] **Body reduction — test three, assume none.** The body is ~650 words against a
  ~10-word title, so it must be compressed or weighted or it dominates:
  1. **Truncation** — first-N ∈ {512, 1024, 2048, 4096, full} chars, and first-N + last-N.
     News puts the topic in the lede, so this may be sufficient on its own.
  2. **Keyword extraction** — top-k terms. Cheapest at inference.
  3. **Lemmatization / stemming** — collapses inflections, shrinks vocabulary.
- [ ] **Field-weighted `FeatureUnion`** (title / summary-or-lede / reduced body) with
  tuned weights.
- [ ] **The lazy alternative** — just repeat the title N times in one concatenated string.
  Often matches a weighted union at a fraction of the complexity; if it ties, it wins.
- [ ] Char n-grams (`char_wb` 3–5) on title for spelling and transliteration robustness.
- [ ] Ablations: stop-words, `sublinear_tf`, min_df/max_df.
- Explicit handling for the 86 title-only and 25%-no-summary articles; a missing field
  must never become the literal string `"None"`.

> These are ablations, not defaults. The brief warns against blindly applying traditional
> NLP preprocessing, and the warning is well aimed: stemming damages named entities
> (`Reuters`→`reuter`), and entities are exactly what separates `politics` from `sport`.
> v1 already measured that feature engineering bought nothing **at headline length** —
> whether that reverses at 100× the length is what this phase tests.
>
> **Never lemmatize or keyword-reduce the text used for near-duplicate detection** —
> duplicate detection needs surface form; normalizing it inflates false merges.

---

## Phase E — Model families *(E1–E4 parallel, need D)*

- [ ] **E1. Linear** — LinearSVC, LogisticRegression. Tune C, class_weight, n-gram range,
  min_df/max_df.
- [ ] **E2. XGBoost** — **never on raw sparse TF-IDF.** Dense input = TruncatedSVD
  (128/256/512) of the union + engineered metadata (title length, body length, `has_body`,
  source desk prior, category tags, hour-of-day). Early stopping on val.
- [ ] **E3. Bagging** — RandomForest, ExtraTrees, Bagging on the same dense input.
  **Expected to lose** on text; the phase exists to prove it, and the result is recorded
  either way.
- [ ] **E4. Embeddings** — `all-MiniLM-L6-v2` first; escalate only if it shows promise.
  Cache vectors keyed by (model, snapshot, text hash) or every later experiment re-pays
  the encode cost. Encode on **CPU** — GPU/MPS kernels are not bit-reproducible and would
  break snapshot reproducibility. Heads: LogisticRegression, linear, XGBoost.
  The publisher-holdout score is the real question: does semantics survive an unseen
  publisher?

---

## Phase F — Ensemble decision *(needs E)*

- [ ] **F1. Complementary-error analysis first** — per-model error sets, pairwise
  disagreement, oracle accuracy. **If oracle ≈ best single model, stop and record the
  negative result.**
- [ ] **F2. Only if F1 justifies it** — soft voting on calibrated probabilities; weights
  *learned* on validation, never assumed; stacking on strictly **out-of-fold** predictions.

---

## Phase G — Calibration, confidence, imbalance *(needs F)*

- [ ] **G1. Calibration** — sigmoid vs isotonic via `cv="prefit"` on val. Score Brier,
  log loss, ECE, reliability curves. v1 found calibration is **free** (0.720 calibrated vs
  0.718 raw). Never surface raw SVM decision scores as probabilities.
- [ ] **G2. Per-class thresholds** fitted on **OOF probabilities from train**, never a
  single global cut. Three bands (auto / review / unknown) with boundaries derived from
  data, not the brief's illustrative 0.90/0.60. Report the coverage-vs-accuracy curve.
  - **`society_lifestyle`'s forced-abstain status is re-opened.** v1 retired it after 439
    labels (F1 0.42, best cut only 0.64 precision) — but that verdict was reached on
    headlines. Whether 3,765 chars of body separates society/labour/lifestyle is open.
- [ ] **G3. Imbalance** at 6.7:1 — plain vs `class_weight="balanced"` vs controlled
  oversampling, on one validation strategy. Oversampling text is most likely to backfire.

---

## Phase H — Error analysis, holdouts, selection

- [ ] **H1.** Full 13×13 confusion matrix; per-class precision/recall/F1/support/FP/FN;
  hardest examples; name overlapping classes, misleading keywords, publisher bias, label
  inconsistencies. Cross-reference against the Finding 3 noise pairs so we don't "fix"
  what is actually annotation disagreement.
- [ ] **H2. Publisher holdouts, final read.** v1 measured only a 0.018–0.043 drop at
  headline length under a confounded setup. **Bodies carry far more publisher-specific
  style, so this can legitimately get worse** — it is the main new risk the body
  introduces.
- [ ] **H3.** Random and time holdout both reported, with the 4-day-window caveat stated
  plainly.
- [ ] **H4. Open the test split — ONCE**, through a single greppable call site, only after
  the model is chosen. If test diverges from val by more than ~0.05, investigate rather
  than accept.
- [ ] **H5. Class-support report.** Every class trains, so this reports **support and
  reliability**, not survival: train/val count, F1 and CI per class, flagged where val
  support is too small for the score to mean anything.
- [ ] **H6. Real-data check** — run the chosen model over recent **unlabelled** articles
  and inspect predictions and the confidence distribution by hand. Metrics on a held-out
  split are not the same as behaving sensibly on live news.
- [ ] **H7. Comparison table + rationale** — macro F1, accuracy, weighted F1, latency,
  model size, publisher-holdout drop, all with CIs. **Ship the simplest model whose CI
  overlaps the best.** Plus a short results notebook driving the charts (notebooks call
  `src/`, never define logic).

**Then stop and review.** Serving, stress testing and packaging are decided after these
numbers exist.

---

## Deferred — planned, not in scope yet

- Robustness / stress tests: short, long, malformed, missing title, missing body, unseen
  publishers, syndicated, paraphrased, breaking-news, multi-topic articles.
- FastAPI `/predict` service with the agreed JSON contract, plus latency and memory
  benchmarks.
- Artifact versioning strategy and README.
- **Fine-tuned transformer** (DeBERTa-v3-small / DistilRoBERTa). Likely the single biggest
  accuracy lever left, but deliberately sequenced *after* the body A/B: running both at
  once makes it impossible to attribute the gain, and if bodies alone suffice, a 250 MB
  model is not worth it. Also sits outside the course's classical-ML syllabus.

---

## Verification

1. `make ml2-test` green, including regression tests for the three known text traps
   (curly-quote fold, title-only service bulletins, no `travel` in the pattern).
2. `newsmlv2 snapshot --id v2-<date>` → `verify` reproduces digests byte-identically.
3. Near-dup scores **precision 0.86 at recall 0.81** on v1's 43-pair census, clearing the
   0.80 precision ceiling v1 proved a single threshold had. (Recall bar restated from
   0.90 — see A3 and the decision log for why it is unreachable by text similarity.)
4. No story group spans splits, in any of the three regimes.
5. `reports/data-quality.md` answers all 14 data-quality items.
6. Ledger has one row per run with config hash and git sha; the comparison table renders
   with group-resampled CIs and a publisher-holdout column from Phase C onward.
7. The body-vs-no-body A/B is reported as one number with a CI.
8. **Publisher probe run before and after D0** — publisher-prediction accuracy must fall,
   and each retained D0 rule shows a larger gain on the publisher holdout than on
   validation.
9. Both baselines reported: v1-parity rung on the v2 snapshot, and the historical 0.720
   clearly labelled as cross-snapshot context.
10. Every complexity decision cites a McNemar p-value or an overlapping CI.
11. `out_of_scope` is **empty in every run**, asserted in tests. The class-support report
    lists all 13 classes with train/val support.
12. Test split touched exactly once.
13. Latency and bundle size benchmarked.

---

## Decisions

| # | Decision | Rationale |
| --- | --- | --- |
| 1 | **Improve, don't port** | v1's *measurements* are inherited as constraints and test cases; its *implementations* are not |
| 2 | **Body text is the hypothesis, not a given** | no CI-backed gain ⇒ stay on short text and record the negative result; that is a valid outcome |
| 3 | Gold labels **vendored + digest-pinned, not committed** | user keeps an external backup. Known gap: no git commit contains the 8,001-label file, so the manifest sha256 is the only integrity anchor |
| 4 | **Snapshot cut `collected_before` = 2026-08-26T12:00:00Z, frozen** | a mid-project re-cut makes every earlier number incomparable. The timestamp is pinned by two constraints: it must fall **after** the last gold label (09:49:39Z) or labelled articles are dropped, and **in the past** or the collector keeps adding rows inside the window and the snapshot stops being reproducible |
| 5 | Near-dup on **body**, two-stage, cosine-verified | bodies are a far stronger duplicate signal than a reworded headline |
| 6 | Time split on `collected_at` only | a 2019 article can arrive tomorrow; `published_at` would leak |
| 7 | **Two publisher holdouts** (The Hindu, The Guardian), scored every run | The Indian Express excluded — 21% of labels would confound the result |
| 8 | The 63 `unsorted` rows are the **abstention evaluation set** | a good model should decline to classify them |
| 9 | **No class is ever dropped** | floor derived as the smallest class; `unsorted` excluded before the minimum; all 13 train by construction |
| 10 | D0 cleaning is **signal correction** | furniture creates publisher→class shortcuts; a rule is kept only if it helps the holdout more than val |
| 11 | Body NLP is an **ablation set** | adopted only on a CI-backed gain, and never applied to near-dup text |
| 12 | **Macro-F1 has a ceiling below 1.0** | measured label disagreement on conflict_war↔politics, business_economy↔politics, crime_justice↔disaster_accident |
| 13 | Randomized search, fixed iteration cap | the space multiplies out and each fit is now ~100× longer; budget is better spent on more model families than a finer grid |
| 14 | Embeddings encoded on **CPU** | GPU/MPS is not bit-reproducible and would break snapshot reproducibility |
| 15 | POC — one dependency set | no packaging ceremony. If it doesn't beat v1, the transformer gets ditched |
| 16 | Test split closed until H4 | one door, greppable, opened once |

---

## Open considerations

1. **Body length is the sleeper risk.** Max is 179,230 chars — 48× the median. How the
   body is *reduced* (Phase D) will likely matter more than which classifier is chosen.
2. **The 4-day `collected_at` window is a weak drift test** and no modelling fixes it. The
   two publisher holdouts carry the generalization argument; a real drift test needs the
   collector running several more weeks.
3. **Taxonomy boundaries are the known bottleneck, not the model.** Tie-break rules in the
   labelling guide would buy more than another labelling round.
4. **Upstream: fix the BBC scraping technique** (user TODO, 2026-08-26). 25 BBC bodies are
   scraped navigation menus and video carousels instead of article text. ml-v2 works
   around it by dropping the body and falling back to title+summary, but the real fix is
   in the Go collector's extractor. Low urgency at 0.19% of the corpus; revisit if BBC
   coverage grows or the workaround starts hiding a larger problem.

---

## Decision log

Append newest at the top. One line of rationale each.

| Date | Decision |
| --- | --- |
| 2026-08-26 | **`bool(nan)` is `True`, and it silently reintroduced v1's worst split bug.** An unlabelled row's `topic` is `NaN` in pandas, so `bool(r.topic)` marked every row as labelled, the cuts fell back to corpus-wide quantiles, and the labelled split came out **79/12/9** instead of 70/15/15. The unit test written to catch exactly this passed, because its fixture used a real `bool`. Fixed with `isinstance(r.topic, str)`; the snapshot test now asserts split proportions on the **real** frozen data, which is the only place the bug was visible. |
| 2026-08-26 | **Boilerplate must be keyed on publisher, not section feed, and found at three levels.** Line-level discovery alone missed two whole classes: scrapers that emit a body as a *single line* (The Economic Times' `Listen to this article in summarized format`) needed **sentence-level** splitting, and chrome concatenated without punctuation (the BBC's `Related topics MoneyUpdates from your News topics...`) needed **common prefix/suffix** detection. Keying on `source_name` also put each of the BBC's six feeds under the discovery floor. Together these merged **38, 21 and 20 unrelated articles** into single story groups at 0.99 similarity. After the fix the largest groups are genuine syndication. |
| 2026-08-26 | **Near-dup recall bar restated 0.90 → 0.80.** Precision now reaches 0.86 (v1 proved 0.80 was its ceiling), but recall plateaus at 0.84 even at a cut of 0.30, because ~5 of 31 judged same-story pairs share almost no vocabulary. Same shape as v1's own restatement, in the opposite direction. |
| 2026-08-26 | **Boilerplate discovery validated on the real corpus.** 355 lines across 47 sources, **19 of them longer than 25 words** — exactly the author bios v1's cap made unreachable, including a 112-word Livemint cricket-correspondent bio in 36% of its Sports bodies. Cleaning removes only **1.2%** of body text corpus-wide, so it is not eating reporting. 7 bodies clean to nothing, all Deccan Herald `DH Toon` cartoons, which admission will reject. |
| 2026-08-26 | **Navigation bodies found and handled.** 25 of 13,133 bodies (0.19%, almost all BBC) are scraped menus or video carousels rather than prose; 8 carry gold labels. Per-source boilerplate cannot catch them because every carousel lists different titles. Handled with a one-line share-of-short-lines guard that **drops the body and falls back to title+summary**, rather than rejecting an article whose headline is perfectly good. |
| 2026-08-26 | **Corpus cut corrected to 12:00:00Z.** A midnight cut silently dropped **477 of 8,001** gold labels — articles collected this morning carry labels. The cut must sit after the last gold label *and* in the past. Caught by the Phase A smoke test, not by inspection. |
| 2026-08-26 | Plan created. Publisher holdouts fixed as The Hindu + The Guardian; class floor made data-derived; Phase D0 added after measuring publisher fingerprints in bodies; transformer deferred behind the body A/B. |
