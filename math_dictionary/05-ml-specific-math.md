# 05 — ML-Specific Math

This is where notation, linear algebra, calculus, and probability all get
assembled into the actual machinery of machine learning. If you understand
[01](01-notation-and-symbols.md)–[04](04-probability-and-statistics.md),
everything here is just those building blocks combined for a specific
purpose: **making a model learn from data.**

---

### Loss function / cost function

- **Symbol**: $J(\theta)$ or $L(\theta)$
- **Pronounced**: "J of theta" / "the loss" / "the cost function"
- **Plain-English meaning**: A single number measuring **how wrong** a
  model's predictions currently are, given its current parameters `θ`. Small
  loss = good predictions; large loss = bad predictions. Training a model is
  entirely about making this number smaller.
- **Formula**: e.g. Mean Squared Error: $J(\theta) = \dfrac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$
- **Why it's used**: You can't improve what you can't measure. A loss
  function turns "is this model good?" into a single, precise, comparable
  number that an algorithm can systematically try to minimize.
- **Real-life analogy**: A golf score — one single number capturing "how far
  from ideal" your round was, letting you compare rounds and know if you're
  improving, without needing to describe every single shot.
- **What happens without it**: There would be no objective, automatable
  target for a model to improve against — "training" would have no
  well-defined goal to optimize toward.
- **Where you'll see it in ML/AI**: Every single model you'll ever train
  defines one: Mean Squared Error (regression), Cross-Entropy (classification),
  etc. It's the very first thing defined when building any model.

---

### Gradient descent

- **Symbol**: $\theta := \theta - \alpha \nabla J(\theta)$
- **Pronounced**: "theta becomes theta minus alpha times the gradient of J"
- **Plain-English meaning**: The core learning algorithm. Repeatedly: (1)
  measure how wrong the model is and which direction makes it worse
  (the gradient), then (2) take a small step in the **opposite** direction
  (downhill on the loss), scaled by a step size `α` (the learning rate).
  Repeat until the loss stops improving much.
- **Formula**: $\theta_{\text{new}} = \theta_{\text{old}} - \alpha \, \nabla J(\theta_{\text{old}})$
  — new parameters = old parameters, nudged opposite to the gradient.
- **Why it's used**: For models with thousands to billions of parameters,
  there's no way to solve directly for "the perfect parameters" — instead,
  you iteratively nudge parameters a little at a time in the direction that
  reliably reduces error, using calculus (the gradient) to know which way
  that is.
- **Real-life analogy**: Descending a foggy mountain with only a flashlight
  showing the ground right at your feet — you can't see the whole
  mountain/valley at once, but at each step you feel which direction is
  downhill (the gradient) and take a small step that way (`α` controls how
  big a step you dare take). Repeat until you're near the bottom.
- **What happens without it**: You'd have no scalable, general-purpose way
  to train models with many parameters — you'd be reduced to guesswork or
  exhaustive brute-force search over parameter combinations, which becomes
  impossible as models grow.
- **Where you'll see it in ML/AI**: The literal training loop of virtually
  every ML model, from simple linear regression to today's largest neural
  networks (via variants like SGD, Adam, RMSProp — all are gradient descent
  with extra tricks).

---

### Learning rate

- **Symbol**: $\alpha$ (alpha) or `η` (eta)
- **Pronounced**: "alpha" / "eta" / "the learning rate"
- **Plain-English meaning**: How big a step to take at each round of
  learning. Too big and you overshoot/bounce around wildly without settling;
  too small and learning takes forever.
- **Formula**: appears directly in gradient descent's update rule above,
  scaling the gradient step.
- **Why it's used**: The gradient only tells you *direction*, not how far to
  actually move — you need a separate, tunable setting for step size.
- **Real-life analogy**: Adjusting a shower's temperature knob — turn it too
  far each time and you overshoot between scalding and freezing forever;
  turn it in tiny increments and you'll eventually settle on the right
  temperature, but slowly.
- **What happens without it**: The gradient alone gives no notion of "how
  far" to step — without a controllable step size, training would either
  never converge (steps too big, forever overshooting) or take impractically
  long (steps too small).
- **Where you'll see it in ML/AI**: One of the most important settings
  ("hyperparameters") you'll tune when training literally any model.

---

### Sigmoid function

- **Symbol**: $\sigma(z) = \dfrac{1}{1 + e^{-z}}$
- **Pronounced**: "sigmoid of z" / "the sigmoid function"
- **Plain-English meaning**: Squashes any real number (from -infinity to
  +infinity) into a value between 0 and 1 — perfect for representing a
  probability. Very negative inputs → near 0; very positive inputs → near 1;
  0 input → exactly 0.5.
- **Formula**: $\sigma(z) = \dfrac{1}{1 + e^{-z}}$
- **Why it's used**: Raw model outputs (like a weighted sum) can be any
  number, but you often want the result to be interpretable as a
  probability (which must be between 0 and 1) — sigmoid converts one into
  the other smoothly.
- **Real-life analogy**: A dimmer switch that saturates at fully off (0) or
  fully on (1) no matter how hard you push past those extremes — very
  strong signals in either direction get squashed toward the limits, while
  signals near the middle stay sensitive to small changes.
- **What happens without it**: A model's raw output (which can be any real
  number, e.g. -50 or +1000) couldn't be directly interpreted as "probability
  of belonging to class A" — you'd have no principled way to convert scores
  into probabilities for binary classification.
- **Where you'll see it in ML/AI**: Logistic regression, and the output
  layer of binary classifiers (spam vs. not spam, fraud vs. not fraud).

---

### Softmax function

- **Symbol**: $\text{softmax}(z_i) = \dfrac{e^{z_i}}{\sum_{j} e^{z_j}}$
- **Pronounced**: "softmax"
- **Plain-English meaning**: The multi-class version of sigmoid — takes a
  list of raw scores (one per possible class) and converts them into a
  proper probability distribution: all values between 0 and 1, and they all
  add up to exactly 1.
- **Formula**: For each class `i`, exponentiate its score and divide by the
  sum of all classes' exponentiated scores.
- **Why it's used**: For problems with more than 2 possible categories (e.g.
  "cat / dog / bird"), you need each class's probability to be comparable
  and all of them to sum to 100% — softmax guarantees this.
- **Real-life analogy**: Splitting a fixed pie of "confidence" (100%) among
  several competing candidates based on how strong each one's raw score
  is — the strongest scores get the biggest slices, but every candidate
  gets *some* non-zero share, and it all adds up to the whole pie.
- **What happens without it**: You'd have several unrelated raw scores with
  no guarantee they behave like probabilities (summing to 1, all
  non-negative) — you couldn't cleanly say "there's a 70% chance it's a cat."
- **Where you'll see it in ML/AI**: The output layer of virtually every
  multi-class classification neural network (image classifiers, next-word
  prediction in language models, etc.).

---

### Cross-entropy loss

- **Symbol**: $L = -\sum_i y_i \log(\hat{y}_i)$
- **Pronounced**: "cross-entropy loss"
- **Plain-English meaning**: Measures how different the model's predicted
  probability distribution is from the true answer. Heavily penalizes being
  **confidently wrong** (e.g. predicting 99% "dog" when it's actually a cat)
  much more than being cautiously wrong.
- **Formula**: $L = -\sum_i y_i \log(\hat{y}_i)$ — where `y_i` is 1 for the
  correct class and 0 for the others, so effectively this simplifies to
  `-log(predicted probability of the correct class)`.
- **Why it's used**: For classification, Mean Squared Error doesn't punish
  overconfident wrong answers harshly enough — cross-entropy's use of `log`
  makes the penalty blow up as a confident wrong prediction approaches 0%
  probability for the correct answer.
- **Real-life analogy**: A weather forecaster who confidently says "0% chance
  of rain" and then it pours should be penalized far more harshly than one
  who said "40% chance" — cross-entropy formalizes "confidently wrong is much
  worse than cautiously wrong."
- **What happens without it**: Classification models trained with a
  mismatched loss function (like plain squared error) would learn more
  slowly and be less sensitive to dangerously overconfident wrong
  predictions.
- **Where you'll see it in ML/AI**: The standard, default loss function for
  virtually all classification models (binary and multi-class), paired with
  sigmoid or softmax outputs.

---

### Regularization (L1 and L2)

- **Symbol**: L1: $\lambda \sum_i |\theta_i|$; L2: $\lambda \sum_i \theta_i^2$
- **Pronounced**: "L1 regularization" / "Lasso"; "L2 regularization" /
  "Ridge"; `λ` = "lambda"
- **Plain-English meaning**: An extra penalty added to the loss function
  that discourages the model's weights from growing too large/complex. `λ`
  controls how strongly this penalty is enforced. L1 tends to push some
  weights all the way to exactly zero (effectively removing unimportant
  features); L2 shrinks all weights smoothly toward zero without eliminating
  them entirely.
- **Formula**: $J_{\text{regularized}}(\theta) = J(\theta) + \lambda \sum_i |\theta_i|$
  (L1) or $J(\theta) + \lambda \sum_i \theta_i^2$ (L2)
- **Why it's used**: Without a penalty, a model can grow arbitrarily complex
  weights that fit the training data perfectly but fail to generalize to new
  data (**overfitting** — essentially memorizing noise instead of learning
  the real pattern). Regularization discourages this by making complexity
  "cost" something in the loss.
- **Real-life analogy**: A company budget cap that discourages every
  department from spending freely, forcing them to justify large expenses —
  keeps the whole system simpler and less wasteful, similar to how
  regularization keeps weights from growing unnecessarily large.
- **What happens without it**: Models tend to overfit — memorizing quirks
  and noise specific to the training data, performing great on data they've
  already seen but poorly on new, unseen data.
- **Where you'll see it in ML/AI**: Ridge regression (L2), Lasso regression
  (L1), weight decay in neural network optimizers, and as one of the first
  things you tune when a model overfits.

---

### Likelihood and Maximum Likelihood Estimation (MLE)

- **Symbol**: $L(\theta) = \prod_i P(x_i \mid \theta)$
- **Pronounced**: "the likelihood of theta" / "maximum likelihood
  estimation" / "M-L-E"
- **Plain-English meaning**: "Likelihood" flips the usual question around:
  instead of "given these parameters, how probable is this data," it asks
  "given this **observed** data, how plausible is each possible setting of
  the parameters?" MLE is the process of picking the parameter values that
  make the data you actually observed look as plausible/likely as possible.
- **Formula**: $L(\theta) = \prod_{i=1}^n P(x_i \mid \theta)$ (product of
  probability of each data point under the assumed model, given parameters
  `θ`); MLE finds $\hat{\theta} = \arg\max_\theta L(\theta)$.
- **Why it's used**: Gives a principled, general recipe for choosing "the
  best" model parameters, given only observed data — rather than guessing
  parameters arbitrarily.
- **Real-life analogy**: If you found a coin that landed heads 8 out of 10
  flips, MLE asks "what 'probability of heads' setting for this coin would
  make observing 8/10 heads the most unsurprising/plausible outcome?" (the
  answer: 0.8) — you're reverse-engineering the most plausible explanation
  for what you actually saw.
- **What happens without it**: You'd have no principled, general-purpose
  procedure to fit model parameters to data — many common training
  procedures (including plain least-squares regression) are secretly special
  cases of MLE, without which they'd feel like arbitrary, disconnected
  tricks.
- **Where you'll see it in ML/AI**: The theoretical justification behind why
  many standard loss functions (like cross-entropy and mean squared error)
  are the "right" choice — they're derived from maximizing likelihood under
  certain assumptions about the data.

---

### argmax / argmin

- **Symbol**: $\arg\max_x f(x)$ / $\arg\min_x f(x)$
- **Pronounced**: "arg max" / "arg min"
- **Plain-English meaning**: NOT "what's the biggest/smallest value of the
  function" — instead, "**which input** `x` produces that biggest/smallest
  value." You're asking for the *location* of the best result, not the
  result itself.
- **Formula**: If $f(1)=5, f(2)=9, f(3)=3$, then $\arg\max f = 2$ (because
  input 2 gives the biggest output, 9) — note the answer is `2`, not `9`.
- **Why it's used**: In ML you usually care about *which* class/parameter/
  action is best, not merely how good the best score is.
- **Real-life analogy**: A judging panel doesn't just want to know "the
  highest score was 9.8" — they want to know **which contestant** scored
  9.8, so they know who wins. `argmax` gives you the winner's identity, not
  just the winning score.
- **What happens without it**: A classifier that computes a probability for
  every class (via softmax) would have no standard way to state "therefore,
  the predicted class IS ___" — you'd have probabilities but no final
  decision/answer extracted from them.
- **Where you'll see it in ML/AI**: Turning a softmax's probability list
  into an actual predicted class: `predicted_class = argmax(probabilities)`
  — used at the very last step of every classification model's prediction.
