# 03 — Calculus

Calculus, for ML purposes, boils down to one question asked over and over:
**"If I change this input a tiny bit, how much does the output change?"**
That single question is the entire engine behind how models learn.

---

### Function notation

- **Symbol**: $f(x)$
- **Pronounced**: "f of x"
- **Plain-English meaning**: A function is a machine/recipe: you feed it an
  input `x`, it gives you back an output `f(x)`. Exactly like a function in
  code: `def f(x): return ...`.
- **Formula**: $f(x) = x^2 + 3$ means "take x, square it, add 3."
- **Why it's used**: Almost everything in ML is expressed as "some function of
  the input" — a model IS a function that maps inputs to predictions.
- **Real-life analogy**: A vending machine — put in money (`x`), get out a
  snack (`f(x)`). Same input always gives the same output.
- **What happens without it**: You couldn't describe *any* relationship
  between an input and an output precisely — this notation is the foundation
  everything else in this file builds on.
- **Where you'll see it in ML/AI**: The model itself is written as
  $\hat{y} = f(x; \theta)$ — "the prediction is a function of the input `x`,
  controlled by parameters `θ`."

---

### Derivative

- **Symbol**: $\dfrac{dy}{dx}$, $f'(x)$
- **Pronounced**: "dee y dee x" or "f prime of x" — "the derivative of f"
- **Plain-English meaning**: The **rate of change** — how fast the output
  changes as you nudge the input, at a single instant. Geometrically, it's
  the slope of the curve at one exact point.
- **Formula**: $f'(x) = \lim_{h \to 0} \dfrac{f(x+h) - f(x)}{h}$ — "change in
  output divided by change in input, as that change shrinks to zero."
- **Why it's used**: Tells you the **direction and steepness** to move `x` in
  to make `f(x)` bigger or smaller — this is exactly what's needed to
  "improve" something step by step.
- **Real-life analogy**: Your car's speedometer — speed is the derivative of
  position with respect to time (how fast your location is changing right
  now, not on average over the whole trip).
- **What happens without it**: You'd only know the *average* change over a
  whole range, never the instantaneous direction/steepness at a specific
  point — you couldn't do the tiny, precise adjustments needed to gradually
  improve a model.
- **Where you'll see it in ML/AI**: Every single training step of every model
  — the derivative of the loss function tells you exactly which way to adjust
  each weight to reduce error.

---

### Partial derivative

See also: [`∂` in 01 — Notation and Symbols](01-notation-and-symbols.md).

- **Symbol**: $\dfrac{\partial f}{\partial x}$
- **Pronounced**: "partial f partial x"
- **Plain-English meaning**: A derivative, but for a function with **several**
  inputs — you find the rate of change with respect to just ONE input,
  holding all the others perfectly still.
- **Formula**: For $f(x, y) = x^2 y$, $\dfrac{\partial f}{\partial x} = 2xy$
  (treat `y` as a fixed number while differentiating).
- **Why it's used**: Real models have thousands/millions of parameters at
  once — you need to know the individual effect of each one, isolated from
  the rest.
- **Real-life analogy**: A cake recipe with both oven temperature and baking
  time affecting how done the cake is. "If I only change the temperature,
  keeping time fixed, how much more done does it get?" — isolating one
  variable's effect.
- **What happens without it**: You couldn't figure out how to adjust ONE
  specific weight among millions without accidentally attributing the
  combined effect of everything at once.
- **Where you'll see it in ML/AI**: Backpropagation computes a partial
  derivative of the loss with respect to every single weight in a neural
  network, one at a time (in practice, done efficiently all together via the
  chain rule).

---

### Gradient

- **Symbol**: $\nabla f$ (nabla f)
- **Pronounced**: "gradient of f" / "del f" / "nabla f"
- **Plain-English meaning**: Bundle up ALL the partial derivatives of a
  function (with respect to every one of its inputs) into a single vector.
  That vector points in the direction of **steepest increase** of the
  function.
- **Formula**: $\nabla f = \left( \dfrac{\partial f}{\partial x_1}, \dfrac{\partial f}{\partial x_2}, \dots, \dfrac{\partial f}{\partial x_n} \right)$
- **Why it's used**: With many parameters at once, you need one combined
  "which direction should I move ALL of them" answer, not separate isolated
  answers — the gradient IS that combined direction.
- **Real-life analogy**: Standing on a hillside blindfolded — the gradient is
  the direction that feels steepest uphill under your feet, combining the
  slope in every direction (north-south AND east-west) into one "steepest
  direction" arrow.
- **What happens without it**: You'd have to adjust each parameter separately
  and check results one at a time by trial and error, instead of taking one
  coordinated, mathematically-optimal step across all parameters at once.
- **Where you'll see it in ML/AI**: **Gradient descent** — the core training
  algorithm for virtually every ML model. You compute the gradient of the
  loss function and step in the *opposite* direction (downhill, to
  **minimize** loss) — see [05 — ML-Specific Math](05-ml-specific-math.md).

---

### Chain rule

- **Symbol**: $\dfrac{dy}{dx} = \dfrac{dy}{du} \cdot \dfrac{du}{dx}$
- **Pronounced**: "the chain rule"
- **Plain-English meaning**: If one function feeds into another (output of
  one is the input of the next), the overall rate of change is just the
  product of each step's individual rate of change. It lets you break "how
  does the far end change when I nudge the far start" into small, manageable
  local pieces.
- **Formula**: If $y = f(u)$ and $u = g(x)$, then
  $\dfrac{dy}{dx} = f'(u) \cdot g'(x)$.
- **Why it's used**: Neural networks are literally functions stacked inside
  functions inside functions (layer after layer). The chain rule is the only
  practical way to compute how the final error depends on a weight buried
  deep inside many layers.
- **Real-life analogy**: A factory assembly line where each station's output
  quality depends on the previous station's output quality — to know how a
  change at station 1 affects the final product, you multiply the "sensitivity"
  of each station in the chain, one after another.
- **What happens without it**: Training any network with more than one layer
  would be computationally infeasible — you'd have no way to propagate the
  error signal backward through multiple stacked functions.
- **Where you'll see it in ML/AI**: This IS **backpropagation** — the
  algorithm that trains virtually all neural networks is the chain rule
  applied systematically, layer by layer, from the output back to the input.

---

### Integral

See also: [`∫` in 01 — Notation and Symbols](01-notation-and-symbols.md) for
the core intuition (area under a curve / sum of infinitely many, infinitely
small pieces).

- **Symbol**: $\int_a^b f(x)\,dx$
- **Pronounced**: "the integral of f of x from a to b"
- **Plain-English meaning**: The reverse operation of a derivative — if a
  derivative tells you the instantaneous rate of change, an integral
  reconstructs the **total accumulated amount** from a rate.
- **Formula**: If $F'(x) = f(x)$, then $\int_a^b f(x)\,dx = F(b) - F(a)$
  (the Fundamental Theorem of Calculus).
- **Why it's used**: Needed to go from "rate" back to "total" — e.g. from a
  probability density (how likely values are, per tiny sliver of range) to an
  actual probability (over a whole range).
- **Real-life analogy**: If you know your car's speed at every instant, the
  integral of speed over time gives you total distance traveled — going from
  "rate" (speed) back to "accumulated total" (distance).
- **What happens without it**: You couldn't compute exact probabilities or
  totals for anything that varies continuously — you'd be stuck with rough
  approximations by chopping things into finite (not infinitely fine) chunks.
- **Where you'll see it in ML/AI**: Computing probabilities from continuous
  distributions (e.g. area under a Normal/Gaussian curve equals the
  probability of falling in that range). Mostly appears in the *theory*
  behind ML — day-to-day applied ML code rarely computes integrals directly,
  libraries/approximations handle it.
