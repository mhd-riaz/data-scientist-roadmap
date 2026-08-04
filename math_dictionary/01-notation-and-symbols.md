# 01 — Notation and Symbols

This is the "alphabet." None of this is hard math — it's just shorthand.
Once these stop looking scary, entire equations become readable.

---

### Variables, subscripts, and superscripts

- **Symbol**: `x`, `x_i`, `x^{(i)}`, `x'`
- **Pronounced**: "x", "x sub i", "x superscript i" or "the i-th x", "x prime"
- **Plain-English meaning**: A variable is just a named box holding a value —
  same idea as a variable in code. A subscript (`x_i`) picks out *one item*
  from a collection, like indexing an array: `x_i` ≈ `x[i]`. A superscript in
  parentheses (`x^{(i)}`) is used in ML specifically to mean "the i-th
  **training example**" (not a power!) so it doesn't get confused with
  squaring. A prime (`x'`) usually means "a modified/updated version of x."
- **Formula**: `x_i` = element at position `i` in vector/list `x`
- **Why it's used**: You need a way to talk about "one specific data point out
  of a thousand" without writing out all thousand names.
- **Real-life analogy**: Same as saying "the 3rd person in line" instead of
  giving every person in line a unique English name.
- **What happens without it**: You'd have to write a separate symbol for every
  single data point — impossible for datasets with thousands/millions of rows.
- **Where you'll see it in ML/AI**: `x^{(i)}` = i-th training example, `y_i` =
  i-th label, `w_j` = j-th weight of a model.

---

### Greek letters (they're just variable names)

Greek letters are used because Latin letters (a-z) ran out, and by convention
certain Greek letters *tend* to mean certain kinds of things — but they are
still just variables.

- **α (alpha)** — pronounced "AL-fuh". Commonly used for a **learning rate**
  (how big a step a model takes while learning) or a significance level in
  statistics.
- **β (beta)** — pronounced "BAY-tuh". Commonly a **coefficient/weight** in a
  regression model, or a momentum term in optimizers.
- **θ (theta)** — pronounced "THAY-tuh". The generic symbol for "**all the
  parameters of a model**" — e.g. `θ` might represent every weight in a
  neural network bundled together.
- **λ (lambda)** — pronounced "LAM-duh". Commonly a **regularization
  strength** (how much to penalize an overly complex model) or a rate
  parameter in probability distributions.
- **μ (mu)** — pronounced "myoo". Almost always the **mean** (average) of a
  distribution.
- **σ (sigma, lowercase)** — pronounced "SIG-muh". Almost always the
  **standard deviation** (how spread out data is). `σ²` = variance.
- **Σ (sigma, capital)** — pronounced "sum" or "capital sigma". Means
  **summation** — see its own entry below.
- **ε (epsilon)** — pronounced "EP-sih-lon". A **tiny number**, usually
  representing error/noise, or "an arbitrarily small positive number" in
  proofs.
- **δ (delta, lowercase) / Δ (delta, capital)** — pronounced "DEL-tuh".
  Means **change in** something. `Δx` = "change in x" (final x minus initial
  x) — this is literally what a derivative formalizes.
- **π (pi)** — pronounced "pie". The constant 3.14159...; also reused in
  probability/ML to mean "a policy" (in reinforcement learning) — context
  tells you which.
- **ρ (rho)** — pronounced "row". Commonly used for **correlation**.
- **∇ (nabla)** — pronounced "del" or "nabla". Means **gradient** — see its
  own entry below.
- **Why it's used**: Convention — it lets anyone reading a paper guess the
  *role* of a symbol before reading the definition (see `σ` for spread, `μ`
  for center, `θ` for "the model's dials").
- **What happens without it**: Nothing breaks — you could use any letters —
  but every paper/course would use different naming and be much harder to
  skim/compare.
- **Where you'll see it in ML/AI**: Everywhere — `θ` in gradient descent
  formulas, `λ` in regularized loss functions, `μ`/`σ` in normalization
  (`(x - μ) / σ`), `α` as the learning rate in every training loop.

---

### Σ — Summation

- **Symbol**: `Σ` (capital sigma), written as $\sum_{i=1}^{n} x_i$
- **Pronounced**: "sum from i equals 1 to n, of x sub i" or just "sigma"
- **Plain-English meaning**: "Add up a bunch of things that follow a pattern."
  It is exactly a `for` loop with a running total.
- **Formula**: $\sum_{i=1}^{n} x_i = x_1 + x_2 + \dots + x_n$
- **Why it's used**: Writing out `x_1 + x_2 + x_3 + ... + x_1000000` by hand
  is absurd. `Σ` compresses "add all of these" into one symbol regardless of
  how many terms there are.
- **Real-life analogy**: A shopping receipt's "Total" line — you don't write
  out "item 1 price plus item 2 price plus item 3 price..."; the total is a
  summation over all line items.
- **What happens without it**: You'd need to write an explicit list of every
  term every single time, and formulas couldn't generalize to "n" items — you'd
  have a different formula for a 3-item dataset than a 3-million-item one.
- **Where you'll see it in ML/AI**: The average loss over a whole dataset,
  e.g. Mean Squared Error: $\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
  — code equivalent: `sum((y[i] - y_pred[i])**2 for i in range(n)) / n`.

---

### ∏ — Product

- **Symbol**: `∏` (capital pi), written as $\prod_{i=1}^{n} x_i$
- **Pronounced**: "product from i equals 1 to n, of x sub i"
- **Plain-English meaning**: Same idea as `Σ`, but **multiplying** instead of
  adding.
- **Formula**: $\prod_{i=1}^{n} x_i = x_1 \times x_2 \times \dots \times x_n$
- **Why it's used**: Compact way to write repeated multiplication over a
  list, same reasoning as `Σ` for addition.
- **Real-life analogy**: Compound interest across multiple years — you
  multiply your balance by `(1 + rate)` once per year, i.e. multiply a growth
  factor across every year in sequence.
- **What happens without it**: Same problem as `Σ` — you'd need to write out
  every multiplication term by hand for every dataset size.
- **Where you'll see it in ML/AI**: Computing the **likelihood** of a whole
  dataset under a model — the probability of *all* independent data points
  occurring together is the product of each one's individual probability:
  $L(\theta) = \prod_{i=1}^{n} P(x_i \mid \theta)$.

---

### ∫ — Integral

- **Symbol**: `∫`, written as $\int f(x)\,dx$
- **Pronounced**: "the integral of f of x, dx"
- **Plain-English meaning**: Adding up infinitely many, infinitely thin
  slices — think of `Σ` taken to the extreme where your "items" aren't
  discrete rows but a smooth continuous curve. Practically: the **area under
  a curve**.
- **Formula**: $\int_a^b f(x)\,dx$ = area under the curve `f(x)` between `x=a`
  and `x=b`.
- **Why it's used**: Some quantities (probability of a continuous variable,
  total distance from a speed curve, total area) can't be computed by adding
  a finite list of numbers — you need to account for every point on a
  continuous range.
- **Real-life analogy**: If your car's speed changes every instant, the total
  distance you traveled is the "area under the speed-vs-time graph" — you're
  summing up tiny `speed × tiny-time-slice` pieces across the whole trip.
- **What happens without it**: You couldn't precisely compute probabilities
  for continuous distributions (like "probability height is between 170cm
  and 175cm") or areas/totals for anything that varies continuously.
- **Where you'll see it in ML/AI**: Computing probabilities under continuous
  distributions (e.g. area under a Normal/Gaussian curve), and in the
  mathematical derivation (not the day-to-day coding) of some loss functions.
  You will *read* this symbol far more often than you'll compute it by hand —
  in practice, code and libraries do this for you.

---

### ∂ — Partial derivative symbol

- **Symbol**: `∂`, written as $\dfrac{\partial f}{\partial x}$
- **Pronounced**: "partial f, partial x" or "the partial derivative of f with
  respect to x"
- **Plain-English meaning**: "If I nudge **only** `x` a tiny bit and freeze
  every other variable, how much does the output change?" It's a derivative
  (see [03 — Calculus](03-calculus.md)) applied to a function of *several*
  variables, one variable at a time.
- **Formula**: For $f(x, y)$, $\dfrac{\partial f}{\partial x}$ treats `y` as a
  constant and differentiates with respect to `x` only.
- **Why it's used**: Real models have many parameters at once (thousands to
  billions of weights). You need to know how the error changes with respect
  to *each individual* weight, one at a time, to know how to adjust it.
- **Real-life analogy**: A recipe with salt and sugar both affecting taste —
  "if I only add more salt, keeping sugar exactly the same, how much saltier
  does it get?" That isolated question is a partial derivative.
- **What happens without it**: You couldn't isolate the effect of one
  parameter among thousands — you'd only know "the total taste changed," not
  which ingredient to adjust and by how much.
- **Where you'll see it in ML/AI**: The core mechanism of **backpropagation**
  — for every weight in a neural network, you compute `∂loss/∂weight` to know
  which direction and how much to adjust that weight.

---

### √ — Square root

- **Symbol**: `√x` or $\sqrt{x}$
- **Pronounced**: "the square root of x" or "root x"
- **Plain-English meaning**: "What number, multiplied by itself, gives me
  `x`?" The reverse operation of squaring.
- **Formula**: $\sqrt{x} = y$ such that $y \times y = x$
- **Why it's used**: Undoes squaring — needed anywhere you squared something
  earlier (e.g. to remove negative signs) and now want to get back to the
  original scale/units.
- **Real-life analogy**: If a square room has an area of 9 m², the length of
  one side is $\sqrt{9} = 3$ m — square root converts "area" back to "side
  length."
- **What happens without it**: Error/spread metrics computed via squaring
  (like variance) would stay in "squared units" forever and be hard to
  interpret intuitively.
- **Where you'll see it in ML/AI**: Root Mean Squared Error (RMSE) — you
  square errors to make them positive and penalize big errors more, then take
  the square root at the end to bring the metric back to the original units
  (e.g. dollars, not dollars²). Also: standard deviation = √variance.

---

### `| |` — Absolute value, and `‖ ‖` — Norm

- **Symbol**: `|x|` (absolute value, for a single number), `‖x‖` (norm, for a
  vector)
- **Pronounced**: "absolute value of x" / "the norm of x" (or "the length of
  x")
- **Plain-English meaning**: `|x|` strips the sign off a number — "how far is
  it from zero, ignoring direction." `‖x‖` is the same idea generalized to a
  vector (list of numbers) — "how long is this vector," ignoring which
  direction it points.
- **Formula**: $|x| = x$ if $x \ge 0$, else $-x$. Euclidean norm:
  $\|x\| = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$
- **Why it's used**: Often you care about the **size** of an error or a
  vector, not whether it's positive or negative — being "5 too high" and "5
  too low" should count as an equally bad error of magnitude 5.
- **Real-life analogy**: Being 5 minutes early and 5 minutes late are both
  "5 minutes off schedule" — you don't care about the sign, just the
  magnitude of the deviation.
- **What happens without it**: Positive and negative errors could cancel each
  other out when summed (a +10 error and a -10 error would look like "0 total
  error"), hiding how wrong a model actually is.
- **Where you'll see it in ML/AI**: L1 loss/regularization uses `|x|`,
  distances between data points use `‖x - y‖` (Euclidean distance), and
  gradient magnitude checks use norms.

---

### ∈, ⊆/⊂, ∀, ∃ — Set and logic symbols

- **Symbol / Pronounced**:
  - `∈` — "is an element of" / "belongs to" — e.g. $x \in \mathbb{R}$ = "x
    belongs to the real numbers" (i.e., x is any real number, like `2.7` or
    `-5`).
  - `⊆` / `⊂` — "is a subset of" — e.g. training set `⊂` full dataset.
  - `∀` — "for all" — e.g. $\forall i$ = "for every i."
  - `∃` — "there exists" — e.g. $\exists x$ = "there is at least one x such
    that..."
- **Plain-English meaning**: These are just compressed English phrases used
  to state rules precisely and briefly, borrowed from formal logic/set
  theory.
- **Why it's used**: Prose like "for every data point in the training set,
  the following must hold..." gets replaced by `∀ x ∈ X, ...` — shorter and
  unambiguous.
- **Real-life analogy**: Legal contracts use precise phrasing ("the party of
  the first part") instead of casual language to avoid ambiguity — these
  symbols do the same for math statements.
- **What happens without it**: Statements would be wordier and more prone to
  ambiguity, especially in dense proofs or formal definitions.
- **Where you'll see it in ML/AI**: Mostly in formal definitions/papers
  (e.g. "for all training examples..."), rarely in day-to-day applied ML
  code — but common enough in course notes that recognizing them removes a
  lot of unnecessary fear.

---

### ≈ and ∝ — Approximation and proportionality

- **Symbol**: `≈` ("approximately equal to"), `∝` ("proportional to")
- **Pronounced**: "approximately equals" / "is proportional to"
- **Plain-English meaning**: `≈` means "close enough to, not exactly equal."
  `∝` means "grows/shrinks in lockstep with" — if `y ∝ x`, doubling `x`
  doubles `y` (they scale together, even if we don't know the exact
  multiplier).
- **Formula**: `y ∝ x` means `y = k·x` for some constant `k` we may not care
  to specify.
- **Why it's used**: Sometimes the exact constant doesn't matter for the
  argument being made — only the *relationship* (more of X leads to more of
  Y) matters.
- **Real-life analogy**: "The more hours you study, the higher your score" —
  you're stating a proportional relationship without claiming an exact
  formula.
- **What happens without it**: You'd be forced to either state an exact
  (possibly unknown or irrelevant) constant, or use much wordier language to
  express "roughly" or "scales with."
- **Where you'll see it in ML/AI**: "Probability of the data ∝ likelihood ×
  prior" (Bayes' theorem, ignoring the normalizing constant), or "training
  time ∝ dataset size" in complexity discussions.

---

### `!` — Factorial

- **Symbol**: `n!`
- **Pronounced**: "n factorial"
- **Plain-English meaning**: Multiply every whole number from `n` down to 1.
  Counts "how many ways can you arrange/order n distinct things."
- **Formula**: $n! = n \times (n-1) \times (n-2) \times \dots \times 1$, and
  $0! = 1$ by definition.
- **Why it's used**: Counting arrangements/combinations comes up constantly
  in probability (e.g. "how many ways can 5 cards be ordered?").
- **Real-life analogy**: If you have 4 friends and want to know how many
  different orders you could seat them in a row, it's $4! = 24$ ways.
- **What happens without it**: You'd have to manually enumerate every
  possible ordering/arrangement instead of computing a count directly.
- **Where you'll see it in ML/AI**: Probability distributions used in
  statistics (e.g. combinations `n choose k` inside the Binomial
  distribution), rarely in day-to-day deep learning code.

---

### `^` (hat) — Estimate/prediction marker

- **Symbol**: `ŷ` (y-hat), `θ̂` (theta-hat)
- **Pronounced**: "y-hat", "theta-hat"
- **Plain-English meaning**: A little hat over a symbol means "this is our
  model's **estimate/prediction** of the true value," as opposed to the real,
  actual value (which has no hat).
- **Formula**: `y` = the true/actual label. `ŷ` = the model's predicted
  label. The whole point of training is to make `ŷ` as close to `y` as
  possible.
- **Why it's used**: You constantly need to distinguish "what actually
  happened" from "what the model guessed" in the same equation — the hat is
  a one-character way to do that instead of writing "predicted" every time.
- **Real-life analogy**: A weather forecast ("70% chance of rain tomorrow")
  is a `ŷ` — an estimate — versus what actually happens tomorrow, which is
  `y`.
- **What happens without it**: Every formula comparing predictions to reality
  (i.e., every loss function) would need clunky wording like
  "predicted_y minus actual_y" instead of a clean `ŷ - y`.
- **Where you'll see it in ML/AI**: Every single loss function:
  $(y - \hat{y})^2$, classification outputs, evaluation metrics.

---

### ∞ — Infinity

- **Symbol**: `∞`
- **Pronounced**: "infinity"
- **Plain-English meaning**: Not a specific number — a concept meaning "grows
  without bound / larger than any number you can name."
- **Formula**: Used in limits, e.g. $\lim_{n \to \infty} f(n)$ = "what value
  does `f(n)` approach as `n` keeps growing forever?"
- **Why it's used**: Needed to describe trends/behavior "in the long run" or
  "at the extreme," e.g. what happens to a model's error as you get
  infinitely more training data.
- **Real-life analogy**: "The line for the theme park ride never ends" — you
  don't need an exact count to reason about what happens as the line keeps
  growing.
- **What happens without it**: You couldn't reason about limiting/asymptotic
  behavior (e.g. "does this algorithm's runtime explode as data grows?")
  without a concept for unbounded growth.
- **Where you'll see it in ML/AI**: Big-O complexity discussions, convergence
  guarantees ("as the number of iterations approaches infinity, the model
  converges to..."), and numerical stability caveats (avoiding
  division-by-zero blowing up "to infinity").
