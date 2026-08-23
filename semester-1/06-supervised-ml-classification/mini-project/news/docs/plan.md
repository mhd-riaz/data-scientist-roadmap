# Phase plan — Phase 2 onward

Phase 1 (collection) is complete and is described in [readme.md](../readme.md). This
document plans everything after it.

## North star

> Showcase a **combination of classical ML models**, presented as a **dynamic
> e-newspaper**.

The combination is the deliverable, not any single model. Two consequences that
decide every trade-off in this document:

1. **Every model must earn a visible feature in the paper.** A model with no
   surface in the product is not showcasing anything, and does not belong in
   scope.
2. **Every model must carry a defensible number.** A feature with no evaluation
   is a demo, not ML work.

Where those two rules conflict with breadth, breadth loses. Four models that are
built, surfaced and measured beat seven that are gestured at.

---

## How an agent should use this document

- Work **one phase at a time**, in order. A phase is done when every one of its
  acceptance criteria is met and the evidence exists in the repository — not when
  the code appears finished.
- **Do not start a phase whose predecessor's criteria are unmet.** The one
  exception is stated explicitly in Phase 5.
- Acceptance thresholds marked *provisional* are first estimates made before any
  data was seen. Recalibrate them against the first honest baseline, record the
  new number and the reason in the decision log, and move on. Do not silently
  lower a threshold to make a run pass.
- If a phase's plan turns out to be wrong once real data is in hand, stop and say
  so rather than working around it. The plan is a hypothesis.
- Read [Ground rules](#ground-rules) before writing anything. They are hard
  constraints, not preferences.

---

## Ground rules

| #   | Rule                                                                                        | Why                                                                                                                                                                 |
| --- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **No LLMs**, at training or inference. Classical ML only.                                   | Project constraint. Transformer *encoders* for embeddings are a boundary case — see the decision log; currently excluded.                                           |
| 2   | Serving must fit **2 GB RAM**, on a host with 8 GB total.                                   | Homelab target. Cap MongoDB's WiredTiger cache explicitly so the collector is never the process that gets OOM-killed.                                               |
| 3   | **Training happens offline**, on a laptop, never on the homelab.                            | The 2 GB budget is a serving budget. Do not let it distort training choices.                                                                                        |
| 4   | **No JavaScript build step.** No npm, no bundler, no Node at runtime.                       | Settled decision. The UI is Go `html/template` plus hand-written CSS. Up to ~30 lines of dependency-free vanilla JS for telemetry only, as progressive enhancement. |
| 5   | **Bronze is immutable.** Never mutate `articles` text in place.                             | Cleansing is a derived artifact keyed by `cleaning_version`. Mirrors the existing rule that `ContentHash` is computed once and never recomputed.                    |
| 6   | **The test split is touched exactly once**, at the end of a phase.                          | Everything else uses validation. A test split consulted during tuning is a validation split wearing a disguise.                                                     |
| 7   | Every Go change passes `make check` and `make test-race`.                                   | Existing gate. Unit tests never contact a network service.                                                                                                          |
| 8   | Python work is **pinned, seeded and deterministic**. Same inputs produce identical outputs. | Reproducibility is an acceptance criterion, not a nicety.                                                                                                           |
| 9   | **ML failure must never break collection.** Scoring is best-effort and retried.             | Phase 1 is the asset. Do not put it behind a model.                                                                                                                 |
| 10  | No new heavyweight dependency without a one-line justification in the decision log.         | Prefer the standard library, then a small focused package, then a framework — in that order.                                                                        |
| 11  | Do not write per-change markdown docs. Update this plan and the readme.                     | Keeps the documentation surface small enough to stay true.                                                                                                          |

### Security rules

- Render every article field through `html/template`. Its contextual escaping is
  what stands between scraped third-party text and stored XSS. Never build HTML
  by string concatenation.
- Secrets stay in the environment. `configs/config.yaml` must never hold one —
  the loader already refuses this.
- The read-event endpoint is the only state-changing browser request. Reject
  cross-origin writes by validating `Origin` / `Sec-Fetch-Site`.
- Publish figures, metrics and derived features. **Never redistribute the raw
  article corpus** — it is third-party copyrighted text collected for study.

---

## Repository context an agent needs

**Stack.** Go 1.26+, MongoDB, Docker Compose. Deployed as a single image to a
Coolify homelab by [.github/workflows/news-collector.yml](../../../../../.github/workflows/news-collector.yml).

**Commands.** `make help` lists everything. The ones that matter:

| Command                                   | Purpose                                                    |
| ----------------------------------------- | ---------------------------------------------------------- |
| `make check`                              | `gofmt` check + `go vet` + unit tests. The milestone gate. |
| `make test-race`                          | Unit tests under the race detector.                        |
| `make test-integration`                   | Requires MongoDB on `localhost:27017`.                     |
| `make cover`                              | Coverage summary per package.                              |
| `make up` / `make down`                   | Full local stack.                                          |
| `make seed` / `make collect` / `make run` | Feeds, one collection pass, API.                           |

**Layout.** `cmd/` holds `api`, `migrate`, `seed`, `collector`, `scrape`.
`internal/` holds `domain`, `repository`, `service`, `handler`, `collector`,
`extract`, `processor`, `scheduler`, `mongodb`, `httpclient`, `robots`,
`ratelimit`, `observability`, `config`, `app`.

**New directories this plan introduces:**

```text
docs/plan.md              this file
ml/                       Python, offline only — never deployed
  pyproject.toml          pinned dependencies
  src/                    library code
  data/snapshots/<id>/    frozen datasets (gitignored except manifests)
  artifacts/<version>/    model bundles
  reports/                figures and metrics
internal/web/             templates, CSS, embedded assets (Phase 4.5+)
```

**Existing domain facts that the ML phases depend on:**

- `Article` carries `Title`, `Summary`, `Content`, `Categories`, `Authors`,
  `SourceID`/`SourceName`, `Language`, `Country`, `State`, `City`,
  `PublishedAt`, `CollectedAt`, `ContentHash`, `ProcessingStatus`,
  `ScrapeStatus`.
- Only `scrape_status ∈ {success, not_needed}` articles have full body text.
  NDTV is permanently `blocked` and is summary-only. Mixing the two creates a
  length↔source confound — Phase 2 must decide how to handle this, not ignore it.
- `Language` comes from the **source configuration**, not from detection. It is
  an assumption, not a measurement.
- `MinScrapedWords = 80`, `FullTextWords = 500` are enrichment thresholds, not
  ML-suitability thresholds.
- Sources are Indian English news. **Wire syndication (PTI/ANI/IANS) is heavy**:
  the same story appears near-identically across outlets. `ContentHash` is exact
  match and catches none of it.

---

## The model portfolio

| #   | Model                    | Family                         | Visible feature              | Evaluation                         | Phase |
| --- | ------------------------ | ------------------------------ | ---------------------------- | ---------------------------------- | ----- |
| 1   | Topic classifier         | Supervised classification      | **Sections** of the paper    | Macro-F1 on gold test              | 3     |
| 2   | Near-duplicate detector  | Hashing / LSH                  | Syndicated copies collapsed  | Precision/recall on labelled pairs | 2     |
| 3   | Event clustering         | Unsupervised, online           | **"7 SOURCES COVERED THIS"** | B-cubed P/R/F1                     | 5     |
| 4   | Extractive summariser    | Graph ranking (TextRank + MMR) | **Story digest**             | ROUGE vs hand summaries            | 5     |
| 5   | Ranker                   | Heuristic → learning-to-rank   | **Front page**               | NDCG@10 (needs clicks)             | 6, 8  |
| 6   | Dimensionality reduction | SVD / UMAP                     | **"Today's news map"**       | Variance explained + qualitative   | 6     |

Models 1–4 are the committed core; they need no user data and each has a clean
evaluation. Model 5 ships as a transparent heuristic and upgrades only if click
data accumulates. Model 6 is nearly free once vectors exist.

**Explicitly out of scope**: bias analysis, NER-based location matching,
abstractive summarisation, recommendation beyond the single-user case. Record
them as future work; do not start them.

---

## Phase 2 — Data readiness

**Goal.** A frozen, versioned dataset that a training script consumes without
ever touching MongoDB.

### In scope

- **Corpus profile.** Counts by source × `scrape_status` × `processing_status`;
  word-count distribution per source (p5/p50/p95); how many articles hit
  `MaxArticleContentLength`; `published_at` vs `collected_at` deltas;
  `categories` cardinality and cross-source overlap; missing-field matrix;
  character-set histogram.
- **Cleansing (Silver).** Deterministic, versioned, re-runnable. Unicode NFKC;
  dateline extraction (`NEW DELHI:` → structured field, then removed); byline and
  wire-agency extraction (`PTI`, `ANI`, `IANS` → `wire_agency`); cross-promo
  removal (`Also Read:`, `Watch:`); trailing disclaimer and CTA removal.
- **Learned boilerplate.** Per source, hash every line across all its articles and
  count distinct articles containing each. Lines above a frequency threshold are
  template furniture. Emit the discovered list as a **reviewable artifact** before
  applying it. This replaces hand-written regexes as the primary mechanism; the
  seven patterns in `internal/extract/text.go` stay as a floor.
- **Admission filters.** Length floor for the full-text corpus; soft-paywall
  detection; live blogs, galleries, listicles, horoscopes, scorecards and
  sponsored content; actual language detection with quarantine on mismatch;
  timestamp sanity.
- **Near-duplicate detection (model #2).** MinHash/LSH over shingles → story
  groups. Needed here, not later, because grouped splits depend on it.
- **Label taxonomy.** 8–14 canonical classes, plus a per-source mapping table
  from RSS `categories` and URL path segments. Decide single- vs multi-label.
- **Gold set.** Hand-labelled, stratified, from a random sample.
- **Frozen splits.** Temporal, grouped by story cluster, stratified where the
  temporal ordering allows.
- **Snapshot export.** Parquet or JSONL under `ml/data/snapshots/<snapshot_id>/`
  with a manifest and a data card.

### Out of scope

Training anything. Any cleaning rule that exists only because it might help a
model that has not been built yet.

### Deliverables

1. Profile report with figures, under `ml/reports/`.
2. Cleansing pipeline with a `cleaning_version` constant and a rejection log
   carrying a reason code per dropped article.
3. Reviewed per-source boilerplate list, checked in.
4. `taxonomy.yaml` — canonical classes + per-source mapping.
5. Gold label set, checked in.
6. Snapshot + manifest + data card.

### Acceptance criteria

Phase 2 is split in two. Everything that does not depend on corpus size is
**built and enforced by tests** (`make ml-test`, 53 tests). Everything that
depends on having enough articles per class is **deferred** until the homelab has
accumulated ~10K articles — see the decision log.

Built and verified:

- [x] Profile report exists and is referenced by the decisions made below it.
      `ml/reports/corpus-profile.md`, regenerated by `make ml-profile`.
- [x] Cleansing is deterministic: running it twice on the same input produces
      byte-identical output.
- [x] Rejection log accounts for **100%** of the difference between input and
      output counts, by reason code. `admit.partition` raises rather than
      returning an unbalanced result.
- [x] Splits are grouped by story cluster. **Zero** story clusters span two
      splits — asserted by `groups_spanning_splits`, which the snapshot builder
      calls before writing.
- [x] Splits are temporal: every test article is published after every training
      article.
- [x] Snapshot rebuilds byte-identically from `(git SHA, cleaning_version, seed)`.
      `make ml-verify` rebuilds into a scratch directory and compares digests.
- [x] Data card records article count, date range, sources, cleaning version and
      known limitations. Class distribution is added when labels exist.

Deferred until the corpus is large enough:

- [ ] No single cleansing rule silently removes more than **5%** of any source's
      articles without an explicit, recorded justification. *(provisional —
      cannot be measured yet: learned boilerplate found 0 lines because 65% of
      articles have no body text. Re-run after the scrape backlog drains.)*
- [ ] Taxonomy has **8–14 classes**; every class has **≥ 300 weakly-labelled
      articles** in the snapshot. *(provisional — needs 2,400–4,200 articles;
      1,142 exist and only 795 carry any category)*
- [ ] Gold set has **≥ 800 articles**, **≥ 40 per class**, drawn by stratified
      random sample, labelled without reference to the weak label.
- [ ] Weak-vs-gold label agreement is **measured and reported**. This number is
      the ceiling on what Phase 3 can achieve and must be known before training.
- [ ] Near-duplicate detection evaluated on **≥ 200 hand-labelled pairs**;
      precision **≥ 0.90**. *(provisional — the harness exists and runs on
      synthetic pairs; the threshold itself is uncalibrated, see open question 7)*

### Verification

`make check` passes. A single documented command rebuilds the snapshot from
scratch. The grouped-split and determinism assertions run as tests.

---

## Phase 3 — Topic classifier (model #1)

**Goal.** A versioned model artifact plus a metrics report, reproducible from a
snapshot ID and a git SHA.

### In scope

- Experiment harness: one config-driven entrypoint, one directory of results per
  run. **Do not install MLflow** — a JSON per run and a results table is enough
  at this scale.
- Baseline ladder, in order, each required to beat the last or to explain why
  not: majority class → Complement NB → TF-IDF + LinearSVC → HashingVectorizer +
  `SGDClassifier(loss='modified_huber')` → fastText (quantised).
- Hyperparameter search on **validation only**.
- Error analysis: confusion matrix, and the top 30 misclassified documents read
  by hand. Expect a meaningful share to be label noise, not model error.
- Leakage audit: top coefficients per class; a source-holdout run.
- Robustness: title-only vs title+summary vs full-text; performance by source;
  performance by article length.
- Artifact packaging and the model card.
- `swiss.mplstyle` matplotlib stylesheet, so figures share the UI's design
  tokens. See Phase 7.

### Out of scope

Serving, integration, any Go code.

### The artifact contract

A model bundle is self-describing. Design this **before** training:

```text
model_version, trained_at, git_sha
dataset_snapshot_id, cleaning_version
vectorizer config (n_features, ngram range, lowercase, ...)
idf array | fasttext binary
coefficients + intercepts
label map (index → canonical class)
per-class decision thresholds
metrics
```

If a prediction in production cannot be traced to the exact data and code that
produced it, the contract is incomplete.

### Acceptance criteria

- [ ] Every rung of the baseline ladder is run and recorded in one comparison
      table. A single model with no ladder is not acceptable.
- [ ] Macro-F1 on the **gold test split** ≥ **0.75**. *(provisional — recalibrate
      against the weak-vs-gold agreement measured in Phase 2; a model cannot
      meaningfully exceed its label ceiling)*
- [ ] Per-class F1 reported. No class below **0.50** without a recorded
      explanation. *(provisional)*
- [ ] Leakage audit performed and its result stated. Top features per class must
      be topically plausible. If the top features are `pti`, `ist`, `bengaluru`
      or a source name, the run is invalid regardless of its score.
- [ ] Source-holdout result reported, even if poor. It answers whether the model
      learned topic or publisher style.
- [ ] Test split touched exactly once. The harness should make this checkable.
- [ ] Artifact ≤ **100 MB**; inference ≤ **20 ms/article** single-threaded.
      *(provisional)*
- [ ] Model card complete, including known weaknesses.
- [ ] Full retrain from snapshot reproduces the reported metrics.
- [ ] Confidence thresholds chosen per class, with an explicit "unsorted" route
      for low-confidence articles. A live paper needs "I don't know" more than it
      needs a forced guess.

---

## Phase 4 — Inference integration

**Goal.** Every new article is scored automatically, and re-scoring the whole
corpus under a new model version is a routine operation.

### In scope

- Serving decision (see decision log) and artifact loading.
- Schema additions to `Article`: `predicted_topics[]`, `confidences[]`,
  `model_version`, `scored_at`, `cluster_id`, and a vector reference.
- **Backfill mechanism.** Reuse the pattern that already exists — `scrape_status`
  / `next_scrape_at` becomes `score_status` / `scored_with_version`. Shipping a
  new model marks the corpus stale and a worker drains the backlog. Do not invent
  a second mechanism.
- **Train/serve parity fixtures.** The single largest risk in this phase.
- Migration for new indexes.

### Acceptance criteria

- [ ] **Golden fixture set**: N articles with expected vectors and predicted
      labels, checked in under `fixtures/`, asserted by **both** the Python tests
      and the Go tests. Any divergence fails the build. Without this, a cleaning
      mismatch between training and serving degrades accuracy invisibly.
- [ ] Every newly collected article is scored within **one scheduler interval**
      of collection.
- [ ] Backfill re-scores the full corpus and is resumable after a kill.
- [ ] Scoring failure never blocks collection — prove it with a test that fails
      the scorer and asserts collection still succeeds.
- [ ] Resident memory of the serving process stays under **2 GB** with the model
      loaded, measured and recorded.
- [ ] A missing or corrupt artifact degrades gracefully: the API starts, serves
      unscored articles, and reports the condition on `/health/ready`.
- [ ] `make check`, `make test-race` and `make test-integration` pass.

---

## Phase 4.5 — Thin UI slice

**Goal.** Schedule insurance. From here on there is always something to
demonstrate, and read-event data begins accumulating.

This phase is deliberately small and deliberately ugly. **Do not style it.**

### In scope

- `internal/web/`: `html/template` templates, `go:embed`, three routes — feed
  list, article detail (`/articles/{id}`), cluster view (`/clusters/{id}`).
- Card content: topic label, title, server-truncated summary (~180–220 chars at a
  word boundary), source, relative time.
- Browser auth via the **existing Basic auth** — no new auth code. Session
  cookies are a later polish item.
- **Read-event logging**, from day one:
  - Click is free — a `GET /articles/{id}` is the event.
  - Impressions and dwell need ~30 lines of dependency-free JS:
    `IntersectionObserver` for impressions, a timer plus `visibilitychange` for
    dwell, batched and flushed with `navigator.sendBeacon`.
  - Record `article_id`, `cluster_id`, `timestamp`, `position_in_feed`, `dwell`.
    **Position matters** — it is what lets Phase 8 correct for the fact that
    items at the top get clicked regardless of quality.

### Acceptance criteria

- [ ] Three routes render real data from MongoDB.
- [ ] Page works fully with JavaScript disabled. Telemetry is the only thing lost.
- [ ] Read events persist and are queryable.
- [ ] Impressions are recorded, not just clicks. **A shown-and-not-clicked card is
      the only source of negative labels Phase 8 will ever have.**
- [ ] Cross-origin writes to the read-event endpoint are rejected.
- [ ] All article text renders through `html/template` escaping.
- [ ] No JavaScript build step introduced.

---

## Phase 5 — Grouping and summarisation (models #3, #4)

**Goal.** Stories are grouped live and each carries a digest of what every outlet
said.

This phase depends on Phase 2 (vectors, near-dup) but **not** on Phase 3 —
clustering is unsupervised. It may proceed in parallel with Phase 3 or 4 if that
helps the schedule.

### In scope

- **Online leader-follower clustering.** For each new article, cosine against the
  centroid of every *active* cluster; join above a threshold, else start a new
  cluster. Single pass, no fixed *k*, new events create their own clusters — this
  is what makes the paper dynamic rather than retrained.
- Time-windowed active set so memory stays flat. Clusters age out.
- Cluster metadata: size, source count, first/last seen, span, representative
  article.
- **Multi-document extractive summarisation.** TextRank over the sentence graph
  across all articles in a cluster, with MMR to remove redundancy.
- Threshold tuning against a hand-grouped sample.

### Known failure modes to guard against

- **Centroid drift** — long-lived clusters wander until "cricket" absorbs "sports
  politics". Mitigate with a cluster age cap, a frozen centroid after N members,
  or max-similarity-to-any-member instead of centroid similarity.
- **Order dependence** — single-pass clustering gives different results for
  different arrival orders. Acceptable for a live feed; it does mean results are
  not reproducible from the same corpus. Know this before it causes confusion.

### Acceptance criteria

- [ ] **≥ 200 articles hand-grouped** into events as an evaluation set.
- [ ] B-cubed F1 ≥ **0.70** on that set. *(provisional)* Silhouette score is not
      acceptable as the primary metric here.
- [ ] Active cluster memory is **bounded and measured** — state the ceiling in
      articles/day terms and show it holds.
- [ ] Clustering threshold chosen from the evaluation set, not by eye, and the
      sensitivity curve recorded.
- [ ] Summaries are ≤ 5 sentences, contain no sentence twice, and draw from more
      than one source when the cluster has more than one.
- [ ] Summary quality measured: ROUGE against **≥ 30 hand-written** cluster
      summaries, or a documented human evaluation.
- [ ] Cluster view in the UI shows all versions ordered by time.

---

## Phase 6 — Edition assembly (models #5, #6)

**Goal.** The paper assembles itself on a schedule, and it looks like a paper.

### In scope

- **Edition concept.** A generated edition with a masthead stating date, time,
  article count and story count. This is what makes "dynamic" tangible.
- **Front page ← ranker.** Ship the transparent heuristic first:

  ```text
  score = w1·interest + w2·importance + w3·freshness + w4·novelty + w5·source_quality
  ```

  - `importance` = cluster size and source count — how many outlets covered it.
    This is a real signal available on day one, with no user data.
  - `freshness` = exponential decay on `published_at`.
  - `novelty` = penalty for similarity to already-shown clusters.
  - `interest` = cosine to a time-decayed centroid of read articles.
- **Diversity.** MMR re-ranking plus hard caps: max 2 per source, max 3 per
  topic, one representative per cluster with the rest collapsed.
- **Sections ← classifier.**
- **Today's news map ← SVD/UMAP**, a 2D topic landscape rendered as a full-page
  data visual.

### Acceptance criteria

- [ ] An edition generates on a schedule and is retrievable by date.
- [ ] Front page respects every diversity cap — assert in a test, not by eye.
- [ ] Every ranking weight is documented with its rationale. An unexplained
      constant is not acceptable.
- [ ] Ranker is evaluated against at least one baseline (pure recency, and random)
      on whatever click data exists. If data is too thin for NDCG, say so
      explicitly and report the comparison qualitatively — **do not report a
      number the data cannot support**.
- [ ] Forced exploration quota of ~**15%** in the feed. Personalisation on a
      single user's clicks collapses to one topic within weeks without it. Build
      it in now, not as a later fix.
- [ ] News map renders and is legible at both mobile and desktop widths.
- [ ] Edition generation is idempotent for a given timestamp.

---

## Phase 7 — Swiss design system

**Goal.** The e-newspaper looks like an e-newspaper. Pure styling over markup
that already works.

The full design system specification lives outside this document. What the plan
fixes:

### Architecture

- Hand-written CSS with **custom properties as the token layer**. ~400 lines.
  No Tailwind, no build step.
- **Two registers.** The design system as supplied is a landing-page system
  (hero, testimonials, FAQ). A newspaper is information-dense. Split it:
  - *Editorial* — masthead, front page lead, section heads, article detail, news
    map. Full extreme type scale, generous padding, geometric compositions.
  - *Dense* — feed listings, cluster contents, metric tables. Tighter scale,
    borders doing the structural work instead of whitespace.
  Both keep zero radius, thick borders, uppercase, flush-left, no shadows.
- Four CSS textures: grid (24px), dots (16px), diagonals, SVG noise.
- Inline lucide SVGs. No icon font, no package.
- Inter, self-hosted, variable, latin subset, woff2, preloaded, `go:embed`ed.
  Do not call Google Fonts.

### Dark mode — re-derived, not inverted

Straight inversion gives `#FFF` on `#000` at 21:1, which causes halation and
reads as cheap.

| Token              | Light      | Dark       | Note                                                                |
| ------------------ | ---------- | ---------- | ------------------------------------------------------------------- |
| Background         | `#FFFFFF`  | `~#0E0D0C` | Off-black, slightly warm                                            |
| Foreground         | `#000000`  | `~#EDEBE8` | Off-white, warm-neutral                                             |
| Muted surface      | `#F2F2F2`  | `~#1A1917` | **Lighter** than bg — elevation by lightness, never shadow          |
| Border, structural | `#000000`  | `~#EDEBE8` |                                                                     |
| Border, subtle     | —          | `~#2E2C29` | Dark mode needs a quiet rule light mode does not                    |
| Accent             | `#FF3000`  | `~#FF5C3D` | Lifted, slightly desaturated — saturated red on near-black vibrates |
| Accent, body text  | `~#D42400` | `#FF5C3D`  | See contrast note                                                   |

Textures need re-tuning, not inversion: light-on-dark patterns read stronger at
identical opacity. Drop to ~2%, and consider dropping the noise overlay entirely
in dark mode, where it reads as sensor noise rather than paper grain.

Theme comes from a cookie, rendered server-side into the initial response, so
there is **zero flash by construction**.

### The red budget

Red is a functional signal, never decoration. On a 40-card feed that rule is hard
to hold. Red is permitted on exactly these:

1. Source count on a cluster — `7 SOURCES`
2. Unread / new since last visit
3. Active filter state
4. Section number prefixes (`01.`, `02.`) — from the system
5. Hover feedback — transient, so it does not compete

**Not red**: topic labels, timestamps, source names, confidence scores, bylines.

### Acceptance criteria

- [ ] All tokens are CSS custom properties in one place, consumed by both the UI
      and `swiss.mplstyle`.
- [ ] **Contrast**: `#FF3000` on white is ≈ 3.7:1 — it **fails** AA for
      body-size text and passes only for large text (≥24px, or ≥19px bold) and
      UI boundaries. Body-size red must use the darkened variant. Verify every
      pairing with a checker; do not assume.
- [ ] Keyboard navigable throughout; visible 2px focus ring.
- [ ] Touch targets ≥ 44×44px on mobile.
- [ ] `prefers-reduced-motion` respected.
- [ ] Correct heading hierarchy and semantic HTML5.
- [ ] Both themes verified at mobile, tablet and desktop widths.
- [ ] Page weight ≤ **100 KB**, font included. *(provisional)*
- [ ] Borders never thin below 4px, and type never loses uppercase or tight
      tracking, at any breakpoint.

---

## Phase 8 — Learned ranker (stretch)

Only reachable if read data has accumulated. Replace the heuristic weights with
logistic regression or LightGBM over the same features, plus impressions as
negatives and position as a bias-correction feature.

### Acceptance criteria

- [ ] **≥ 2,000 read events** before training. Below that, do not train — report
      the heuristic and say why.
- [ ] Beats the heuristic on NDCG@10 with a temporal split.
- [ ] Position bias explicitly handled and the handling documented.
- [ ] Model ≤ 10 MB.

---

## Cross-cutting

- **Reproducibility.** Pinned seeds, pinned dependency versions, one documented
  command to rebuild dataset and retrain from scratch.
- **Drift.** News vocabulary shifts fast; a model trained in August decays by
  November. Monitor rolling accuracy on recent labels or PSI on the prediction
  distribution. Decide the retraining trigger.
- **Rollback.** Pin `model_version` the way `NEWS_IMAGE` pins a SHA tag.
- **Resource guards.** Cap the WiredTiger cache; memory-limit the scoring
  process.
- **Retention.** Decide how long articles, vectors and read events are kept.

### Report deliverables

The `report/generate_report.py` + `figures/` convention from the sibling
assignments applies. Figures worth generating:

| Figure                                    | Why                                           |
| ----------------------------------------- | --------------------------------------------- |
| 2D UMAP coloured by predicted topic       | The single most compelling visual available   |
| Confusion matrix + per-class F1           | Expected                                      |
| Learning curve                            | Answers "do you need more data" with evidence |
| Class distribution before/after cleansing | Justifies Phase 2                             |
| Cleansing funnel with drop reasons        | Almost nobody does this                       |
| Story timeline across sources             | Proves clustering better than any metric      |
| Cluster size distribution                 | Shows the importance signal is real           |
| Top features per class                    | Interpretability and leakage evidence         |
| Baseline ladder table                     | Shows method, not just result                 |
| Latency and memory under the 2 GB cap     | Addresses the stated constraint directly      |

---

## Decision log

| Date       | Decision                                                  | Rationale                                                                                                                                                                                                                                                                                                 |
| ---------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-08-23 | No LLMs; classical ML only                                | Project constraint                                                                                                                                                                                                                                                                                        |
| 2026-08-23 | 2 GB serving budget; training offline on laptop           | Homelab has 8 GB total, shared with MongoDB                                                                                                                                                                                                                                                               |
| 2026-08-23 | Go `html/template` + hand-written CSS; **React rejected** | React was considered and declined: `go:embed` made it viable, but it added a second toolchain, a browser auth redesign (an API key cannot live in a JS bundle), CORS, and ~3× page weight. Its one real advantage — dwell and impression telemetry — is recoverable with ~30 lines of dependency-free JS. |
| 2026-08-23 | Browser auth reuses existing Basic auth                   | Zero new code. Session cookies deferred to polish.                                                                                                                                                                                                                                                        |
| 2026-08-23 | Card shows title + truncated summary; click → detail page | Not a modal, not inline expand: bookmarkable, survives reload, and gives the cluster somewhere to live                                                                                                                                                                                                    |
| 2026-08-23 | Dark mode: re-derived tokens, not inverted                | Pure white on pure black causes halation and reads cheap                                                                                                                                                                                                                                                  |
| 2026-08-23 | Near-dup detection moved into Phase 2                     | Grouped splits depend on it; doing it later would invalidate Phase 3's metrics                                                                                                                                                                                                                            |
| 2026-08-23 | Thin UI slice inserted at 4.5                             | The UI is on the demo critical path; deferring all of it to the end risks having no demo                                                                                                                                                                                                                  |
| 2026-08-23 | Bias analysis and NER location matching cut               | Two more models with no evaluation budget left                                                                                                                                                                                                                                                            |
| 2026-08-23 | Extractive summarisation added                            | Distinct classical model family, strongest demo feature, and it proves the clustering                                                                                                                                                                                                                     |
| 2026-08-23 | **Corpus trains on `title + summary`**, not full text     | Answers open question 6. Profile showed `content` empty for 65.5% of articles and present for only 5.9% at full length, and availability correlates with the *source*, not the topic. Training on full text would confound topic with publisher. Title is 100% present, summary 77%. Full text becomes a variant, not the default. |
| 2026-08-23 | Scrape backlog identified as never run, not failing       | 806 articles `pending` with `scrape_attempts: 0` and zero `success`/`failed` rows. The missing body text is an undrained queue, not scraper breakage — the 57 `not_needed` articles prove extraction works. Recoverable, and worth ~5x more usable text.                                                    |
| 2026-08-23 | "Indian English news" corrected in the corpus description | Of 20 sources, 7 are Indian general news and ~45% are global tech (Wired, Ars, TechCrunch, Verge, ZDNET, Engadget, Register, MIT TR) plus BBC/Guardian/NPR/France 24. PTI/ANI/IANS wire syndication applies only to the Indian subset.                                                                       |
| 2026-08-23 | Taxonomy, gold set and split freeze **deferred**          | 8–14 classes x >=300 weak labels needs 2,400–4,200 articles; 1,142 exist and only 795 carry any category. Labelling now would be redone. Size-independent machinery built first; the taxonomy is frozen once the homelab has accumulated ~10K articles.                                                     |
| 2026-08-23 | Splits cut on **time first**, then drop straddling groups | The first implementation ordered groups then set the boundary at `max(published_at)` of train. One group spanning the corpus collapsed the later splits to nothing — observed as train=721, val=0, test=1. Choosing cut times first and dropping only straddling groups whole gives 718/154/154, 3 dropped. |
| 2026-08-23 | MinHash/LSH hand-rolled rather than `datasketch`          | ~80 lines, and determinism is the point: library defaults seed from Python's `hash()`, which is randomised per process and would break byte-identical snapshots. It is also one of the models being showcased.                                                                                              |
| 2026-08-23 | Snapshots are **JSONL**, not Parquet                      | Byte-identical rebuild is an acceptance criterion. JSONL with sorted keys and fixed row order is trivially deterministic and greppable; Parquet adds `pyarrow` plus writer-version determinism risk for no benefit at this scale.                                                                            |
| 2026-08-23 | Punctuation folded explicitly on top of NFKC              | NFKC does **not** fold curly quotes or en/em dashes. Two outlets running the same wire copy with different quote styles must produce identical shingles, or near-duplicate detection misses the pair.                                                                                                        |
| 2026-08-23 | Ground rule 1 relaxation granted, then **not used**       | An LLM was briefly approved for pre-filling training labels only. Superseded the same day by blind human labelling, so no LLM is used anywhere in the project and rule 1 stands unmodified. Recorded because the reasoning is worth keeping if the question returns.                                          |
| 2026-08-23 | **Gold set is labelled blind**                            | The sheet carries `title` and `summary` only — no source, no URL, no proposed label. A suggestion seen first anchors an annotator into accepting ~95% of it, errors included, and the feed URLs contain the section name. Either would turn the agreement study into a measurement of itself.                  |
| 2026-08-23 | Every label carries a `label_source`                       | `feed`, `category` or `human`, plus the taxonomy version. Without it the weak-vs-gold agreement study cannot be run, because the two kinds of label become indistinguishable after the fact.                                                                                                                  |
| 2026-08-23 | Feed and category labels cross-check each other            | The section a feed declares and the publisher's own categories are independent signals. Agreement auto-accepts; disagreement routes to human review. This concentrates review effort where it changes an outcome instead of spreading it over articles two signals already agree on.                          |
| 2026-08-23 | Gold sheets carry an **overlap block** across annotators   | The same articles appear in every sheet. Without it there is no way to separate a genuinely ambiguous taxonomy from a careless annotator, and inter-annotator agreement is unmeasurable.                                                                                                                     |

## Open questions

1. **Taxonomy size and single- vs multi-label.** Still open, and still the
   highest-risk item — but no longer blocking, since the pipeline is built and the
   taxonomy is now frozen at the *end* of data accumulation rather than the
   start. Note that `categories` carries two interleaved axes: geography
   (`india` 236, `karnataka` 38, `bengaluru` 33) and topic (`gear` 36, `ai`,
   `security`). The taxonomy needs one topical axis; geography becomes a
   separate field.
2. ~~**Corpus sufficiency.**~~ Answered: 1,142 articles is roughly a third of
   what the stated thresholds need. The homelab run accumulates ~10K/week, after
   which the taxonomy is frozen and the gold set drawn.
3. **Serving mechanism for Phase 4** — Go-native artifact export, ONNX via
   `onnxruntime-go`, or a Python sidecar. Changes Phase 3's export format, so
   decide before Phase 3 packaging. Go-native is preferred: a hashing vectoriser
   is murmurhash3 plus a sign bit, and linear inference is a dot product. The
   catch is matching scikit-learn's murmurhash3 variant exactly — verifiable with
   the Phase 4 golden fixtures.
4. **Vector representation.** TF-IDF + SVD (fully classical) versus fastText
   document vectors versus a sentence-transformer encoder. The third is excluded
   under ground rule 1 unless that rule is consciously relaxed and recorded here.
5. **Deadline.** Determines how much of Phases 6–8 is real versus described.
6. ~~**Handling of summary-only sources.**~~ Answered: train everything on
   `title + summary` so the field is level across all sources. See the decision
   log.
7. **Near-duplicate threshold is uncalibrated.** 0.72 Jaccard over 5-word
   shingles is a derivation from the LSH banding (16 bands x 8 rows), not a
   measurement. On short `title + summary` text a reworded headline is a large
   share of the shingles and drags similarity down, so the real threshold is
   probably lower. Only 3 near-duplicate pairs were found in 1,142 articles,
   which is implausibly few for a corpus with this much wire syndication.
   Calibrate against the >= 200 hand-labelled pairs the acceptance criteria
   already require.
