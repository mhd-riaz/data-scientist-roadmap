# Math Dictionary — A Plain-English Reference for ML/AI Math

If you freeze up every time you see a Greek letter or a Σ in a paper or a course,
this folder is for you. It is **not** a math textbook. It's a personal
"translator" between formal math notation and normal human language, built for
one purpose: so math never blocks you from learning ML/AI again.

Every entry in this dictionary answers the same set of questions, in the same
order, so you can scan it fast:

## The Entry Template

> ### Name of the concept
> - **Symbol**: the actual symbol(s) used
> - **Pronounced**: how you'd say it out loud in a lecture/meeting
> - **Plain-English meaning**: what it actually *means*, no jargon
> - **Formula**: how it's written formally
> - **Why it's used**: the problem it solves / why mathematicians bothered inventing it
> - **Real-life analogy**: how you, a human, already use this idea daily without calling it "math"
> - **What happens without it**: what breaks or gets clumsy if this didn't exist
> - **Where you'll see it in ML/AI**: the exact context you'll meet it in courses/code

Read entries top to bottom the first time. After that, just use this as a
lookup — Ctrl+F the symbol or the concept name.

## How to Read Any Equation (do this before panicking)

1. **Ignore the symbols first, read the shape.** Every equation is just
   "this thing equals a recipe made of other things." Find the `=` sign first.
2. **Every Greek letter is just a variable name.** Mathematicians ran out of
   English letters, so `θ` (theta) is just as ordinary as `x`. Don't let the
   font scare you.
3. **Subscripts and superscripts are just labels/indexes**, like array indices
   in code. `x_i` means "the i-th item in x" — same as `x[i]` in Python.
4. **Most ML formulas are loops or aggregations in disguise.** `Σ` (sigma) is
   a `for` loop that adds things up. `∏` (pi) is a `for` loop that multiplies.
   `∫` (integral) is a sum over infinitely many infinitely small pieces.
5. **If you can write it as code, you understand it.** Translating a formula
   to a `for` loop or a NumPy expression is a legitimate and complete way to
   "get" the math — you don't need a proof-level understanding to use it.

## Index

### [01 — Notation and Symbols](01-notation-and-symbols.md)
The "alphabet" — variables, subscripts, Greek letters, Σ, ∏, ∫, ∂, √, `| |`,
`‖ ‖`, ∈, ∀, ∃, ≈, ∝, and other symbols you'll see constantly before you even
get to the interesting math.

### [02 — Linear Algebra](02-linear-algebra.md)
Scalars, vectors, matrices, transpose, dot product, matrix multiplication,
identity matrix, norms, eigenvalues/eigenvectors — the language data is
stored and transformed in.

### [03 — Calculus](03-calculus.md)
Functions, derivatives, partial derivatives, gradients, the chain rule,
integrals — the language of "how much does output change when input
changes," which is the entire mechanism behind training a model.

### [04 — Probability and Statistics](04-probability-and-statistics.md)
Probability, conditional probability, Bayes' theorem, random variables,
expectation, variance, standard deviation, covariance/correlation, normal
distribution — the language of uncertainty and data description.

### [05 — ML-Specific Math](05-ml-specific-math.md)
Loss/cost functions, gradient descent, sigmoid, softmax, cross-entropy,
regularization (L1/L2), likelihood/MLE, argmax/argmin — where all the above
building blocks get assembled into actual machine learning.

## Ground Rule for Adding New Entries

When you hit a new symbol or concept anywhere in `semester-1/` (or anywhere
else) that trips you up, add it here using the exact template above, in the
most relevant file (or a new one if it's a new topic area). Keep the same
"why / analogy / what happens without it" structure — that's what makes this
useful instead of just another textbook.
