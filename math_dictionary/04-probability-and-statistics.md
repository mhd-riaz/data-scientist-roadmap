# 04 — Probability and Statistics

Probability and statistics are the math of **uncertainty and description**.
Data is messy and noisy; this is the toolkit for reasoning about it honestly
instead of pretending it's perfectly clean.

---

### Probability

- **Symbol**: $P(A)$
- **Pronounced**: "the probability of A"
- **Plain-English meaning**: A number between 0 and 1 (or 0% to 100%)
  representing how likely an event is. `0` = impossible, `1` = certain.
- **Formula**: $P(A) = \dfrac{\text{number of favorable outcomes}}{\text{total number of possible outcomes}}$
  (for equally likely outcomes)
- **Why it's used**: The world (and data) is inherently uncertain — you
  rarely know things with 100% certainty, so you need a formal way to
  quantify "how confident" you are.
- **Real-life analogy**: A weather forecast saying "70% chance of rain" — a
  number capturing uncertainty, not a guarantee either way.
- **What happens without it**: You'd be stuck making binary yes/no
  statements ("it will rain" / "it won't") with no way to express degrees of
  confidence — and no way for a model to say "I'm not sure, but probably X."
- **Where you'll see it in ML/AI**: Classification models output
  probabilities (e.g. "85% chance this email is spam"), not just hard
  yes/no answers.

---

### Conditional probability

- **Symbol**: $P(A \mid B)$
- **Pronounced**: "the probability of A given B"
- **Plain-English meaning**: How likely is `A`, **once you already know**
  `B` happened? Knowing `B` can change how likely `A` seems.
- **Formula**: $P(A \mid B) = \dfrac{P(A \text{ and } B)}{P(B)}$
- **Why it's used**: New information should update your beliefs — the
  probability of rain tomorrow changes once you know today was unusually
  humid. Conditional probability formalizes "updating belief given evidence."
- **Real-life analogy**: "What's the chance I'm late for work?" is different
  from "What's the chance I'm late for work, GIVEN that I already missed my
  train?" — the second, more specific question is a conditional probability.
- **What happens without it**: A model couldn't incorporate context/evidence
  into its predictions — every prediction would ignore everything else you
  already know about the situation.
- **Where you'll see it in ML/AI**: Practically every model output is a
  conditional probability: $P(\text{label} \mid \text{input features})$ — "the
  probability of this class, GIVEN these specific input values."

---

### Bayes' theorem

- **Symbol**: $P(A \mid B) = \dfrac{P(B \mid A) \, P(A)}{P(B)}$
- **Pronounced**: "Bayes' theorem" (rhymes with "days")
- **Plain-English meaning**: A formula for **flipping** a conditional
  probability around — if you know "probability of evidence given a cause,"
  it tells you how to compute "probability of the cause given the evidence,"
  which is usually the thing you actually want to know.
- **Formula**: $P(\text{cause} \mid \text{evidence}) = \dfrac{P(\text{evidence} \mid \text{cause}) \times P(\text{cause})}{P(\text{evidence})}$
- **Why it's used**: You often know one direction of a relationship (e.g.
  "if someone has a disease, there's a 95% chance the test is positive") but
  want the reverse (e.g. "given a positive test, what's the chance they
  actually have the disease?") — Bayes' theorem is the only correct way to
  flip it.
- **Real-life analogy**: A doctor knows "95% of sick patients test positive"
  (easy to measure via clinical trials), but a patient really wants to know
  "given I tested positive, how likely am I actually sick?" — these are NOT
  the same number, and Bayes' theorem is how you correctly compute the
  second from the first (it also depends on how rare the disease is overall).
- **What happens without it**: People (and models) routinely confuse "P(A|B)"
  with "P(B|A)" — a very common and serious real-world reasoning error (e.g.
  overestimating disease risk from a positive test on a rare disease) —
  without this formula you have no rigorous way to avoid that mistake.
- **Where you'll see it in ML/AI**: Naive Bayes classifiers (a whole family
  of ML models), spam filters, medical diagnosis models, and the theoretical
  foundation of "updating beliefs" in Bayesian machine learning.

---

### Random variable

- **Symbol**: $X$ (capital letter)
- **Pronounced**: "random variable X"
- **Plain-English meaning**: A variable whose value is the outcome of some
  random/uncertain process — not one fixed number, but a description of "all
  the possible values it could take, and how likely each one is."
- **Formula**: `X` = the result of rolling a die → could be 1, 2, 3, 4, 5, or
  6, each with probability 1/6.
- **Why it's used**: Lets you do math on uncertain quantities (compute
  averages, spreads, combine them with other uncertain quantities) using the
  same algebraic tools as ordinary variables.
- **Real-life analogy**: "Tomorrow's temperature" isn't one fixed number
  today — it's a random variable: a range of possible values, each with some
  likelihood.
- **What happens without it**: You couldn't formally reason about or combine
  uncertain quantities mathematically — you'd be limited to describing
  uncertainty in words only.
- **Where you'll see it in ML/AI**: Model inputs and outputs are often
  treated as random variables when reasoning about noise, uncertainty
  estimation, and generative models.

---

### Expectation (expected value)

- **Symbol**: $E[X]$ or $\mathbb{E}[X]$
- **Pronounced**: "the expectation of X" / "expected value of X" / "E of X"
- **Plain-English meaning**: The long-run **average** outcome if you repeated
  a random process many, many times — a weighted average where more likely
  outcomes count more.
- **Formula**: $E[X] = \sum_{i} x_i \, P(x_i)$ — each possible value times its
  probability, summed up.
- **Why it's used**: Gives you a single, meaningful summary number for an
  uncertain quantity, so you can compare/optimize/reason about it without
  tracking every possible outcome individually.
- **Real-life analogy**: A casino game's "expected value" — if a $1 bet wins
  $3 with 20% chance and $0 otherwise, the expected value is $0.60; you
  wouldn't win exactly $0.60 on any single play, but that's your average
  outcome over many, many plays.
- **What happens without it**: You couldn't summarize or compare random
  outcomes with a single number — you'd have to compare entire lists of
  possible-outcome/probability pairs every time, which doesn't scale.
- **Where you'll see it in ML/AI**: The loss function you minimize during
  training is (conceptually) the **expected** loss over the whole data
  distribution — you're trying to minimize the average error the model would
  make on any data point, not just the ones you've seen.

---

### Variance and standard deviation

- **Symbol**: $\text{Var}(X)$ or $\sigma^2$ (variance); $\sigma$ (standard
  deviation)
- **Pronounced**: "variance of X" / "sigma squared"; "sigma" / "standard
  deviation"
- **Plain-English meaning**: How **spread out** values are around the
  average. Low variance = values cluster tightly near the mean; high
  variance = values are scattered widely. Standard deviation is just the
  square root of variance, which brings it back to the same units as the
  original data (variance is in "squared units," which isn't intuitive).
- **Formula**: $\sigma^2 = \dfrac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$ (average
  squared distance from the mean `μ`); $\sigma = \sqrt{\sigma^2}$
- **Why it's used**: The average alone can be misleading — two datasets can
  have the identical mean but be wildly different in consistency. You need a
  number that captures "how reliable/consistent is this" separately from
  "what's typical."
- **Real-life analogy**: Two bus routes both average 20 minutes, but one is
  always between 18-22 minutes (low variance/reliable) while the other
  ranges from 5-40 minutes (high variance/unpredictable) — same average,
  very different experience.
- **What happens without it**: You'd have no way to quantify consistency or
  risk — a model reporting only "the average prediction is X" without any
  spread gives no sense of confidence or reliability.
- **Where you'll see it in ML/AI**: Feature normalization/standardization
  (scaling data using `σ` so all features are on comparable scales — see
  below), measuring model uncertainty, and evaluating how consistent a
  model's errors are.

---

### Standardization / Normalization using mean and std dev

- **Symbol**: $z = \dfrac{x - \mu}{\sigma}$
- **Pronounced**: "z equals x minus mu over sigma" / "z-score"
- **Plain-English meaning**: Rescale a value to say "how many standard
  deviations away from average is this," so different features (which may
  have wildly different original scales, like age in years vs. income in
  dollars) become directly comparable.
- **Formula**: $z = \dfrac{x - \mu}{\sigma}$ — subtract the mean, divide by
  the standard deviation.
- **Why it's used**: Models often compare/combine multiple features at once
  — if one feature ranges 0-1 and another ranges 0-1,000,000, the second will
  unfairly dominate unless everything is rescaled to comparable units first.
- **Real-life analogy**: Comparing a student's math test score (out of 100)
  to their essay score (out of 5) directly is unfair — converting both to
  "how many standard deviations above/below the class average" puts them on
  the same footing.
- **What happens without it**: Features with naturally larger numeric ranges
  would dominate training just because of their scale, not because they're
  actually more important — this quietly biases and often breaks model
  training (especially for gradient-based methods and distance-based
  methods).
- **Where you'll see it in ML/AI**: Standard preprocessing step before
  training almost any model — `StandardScaler` in scikit-learn, batch
  normalization in neural networks.

---

### Covariance and correlation

- **Symbol**: $\text{Cov}(X, Y)$; $\rho$ (rho) or `r` for correlation
- **Pronounced**: "covariance of X and Y"; "rho" / "correlation coefficient"
- **Plain-English meaning**: Covariance measures whether two variables tend
  to move **together** (both up, both down) or **oppositely**, but its raw
  number is hard to interpret. Correlation is the same idea, rescaled to
  always sit between -1 and +1, making it easy to interpret: +1 = perfectly
  move together, -1 = perfectly move opposite, 0 = no linear relationship.
- **Formula**: $\rho = \dfrac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$
- **Why it's used**: Helps you discover which variables are related to each
  other — critical for understanding data before modeling, and for spotting
  redundant/duplicate information (highly correlated features).
- **Real-life analogy**: Ice cream sales and sunscreen sales rise and fall
  together across the year (high positive correlation) — not because one
  causes the other, but both are driven by temperature.
- **What happens without it**: You couldn't quantify how strongly two
  variables relate — you'd be reduced to eyeballing scatter plots, with no
  precise, comparable number.
- **Where you'll see it in ML/AI**: Feature selection (dropping redundant,
  highly-correlated features), exploratory data analysis, and as a reminder
  that **correlation is not causation** — a correlated feature isn't
  necessarily a *cause* of the outcome you're predicting.

---

### Normal (Gaussian) distribution

- **Symbol**: $X \sim N(\mu, \sigma^2)$
- **Pronounced**: "X is distributed as a Normal with mean mu and variance
  sigma squared" — informally "the bell curve"
- **Plain-English meaning**: A very common pattern where most values cluster
  around an average (`μ`), with values farther from the average becoming
  increasingly rare, in the familiar symmetric "bell" shape.
- **Formula**: $f(x) = \dfrac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
  (you rarely need to compute this by hand — recognizing the shape and
  meaning matters far more than memorizing the formula)
- **Why it's used**: An enormous number of natural and human phenomena
  (heights, measurement errors, test scores, noise in sensors) follow
  approximately this pattern, and it has convenient mathematical properties
  that make it the default assumption in many statistical methods.
- **Real-life analogy**: Adult human height — most people cluster near the
  average, very short and very tall people both exist but are increasingly
  rare the further from average you go, and the pattern is roughly symmetric.
- **What happens without it**: You'd have no reasonable "default" shape to
  assume for natural variation/noise, making many statistical formulas and
  guarantees (like confidence intervals) far harder to derive or justify.
- **Where you'll see it in ML/AI**: Initializing neural network weights
  (often sampled from a Normal distribution), assumed noise in linear
  regression, and the basis of many statistical tests used to evaluate
  models.

---

### Mean and median

- **Symbol**: $\mu$ or $\bar{x}$ (mean); no standard symbol for median, often
  just called "the median"
- **Pronounced**: "mu" or "x-bar"; "median"
- **Plain-English meaning**: Mean = the everyday "average" (add up all
  values, divide by how many there are). Median = the middle value when all
  values are sorted in order — half the data is above it, half below.
- **Formula**: $\bar{x} = \dfrac{1}{n}\sum_{i=1}^{n} x_i$
- **Why it's used**: Both summarize "what's typical," but the mean is
  sensitive to extreme outliers while the median isn't — you need both tools
  to describe data honestly depending on the situation.
- **Real-life analogy**: If 9 people in a room earn $50k and one earns $10M,
  the **mean** salary looks like ~$1M (misleadingly high, dragged up by one
  outlier), while the **median** salary is still $50k (reflects what's
  actually typical).
- **What happens without it**: You'd have no simple, agreed-upon way to
  describe "the typical value" of a dataset — every comparison between
  datasets would be far more manual and error-prone.
- **Where you'll see it in ML/AI**: Filling in missing data (imputing with
  mean or median), understanding whether a dataset has skew/outliers before
  choosing a model, and reporting typical model errors.
