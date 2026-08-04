# 02 — Linear Algebra

Linear algebra is just the math of **organizing and transforming lists of
numbers**. In ML, every image, every sentence, every row of a spreadsheet
becomes a list of numbers — linear algebra is how you do useful things to
those lists efficiently.

---

### Scalar

- **Symbol**: a plain lowercase letter, e.g. `a`, `x`, `5`
- **Pronounced**: just the letter/number itself
- **Plain-English meaning**: A single number. Nothing more.
- **Formula**: `x = 5`
- **Why it's used**: It's the baseline unit everything else (vectors,
  matrices) is built from.
- **Real-life analogy**: Your age, your height in cm, the price of one item —
  a single quantity.
- **What happens without it**: N/A — this is the most basic building block;
  without it there's nothing to build vectors/matrices from.
- **Where you'll see it in ML/AI**: A single weight, a single learning rate
  value, a single prediction for one data point.

---

### Vector

- **Symbol**: a lowercase bold/arrow letter, e.g. **x** or $\vec{x}$, written
  as a column: $\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$
- **Pronounced**: "vector x"
- **Plain-English meaning**: An ordered list of numbers — exactly like a
  Python list or a single row/column of a spreadsheet. Geometrically, it can
  also represent a point in space or an arrow with direction and length.
- **Formula**: $\vec{x} = (x_1, x_2, \dots, x_n)$
- **Why it's used**: Real-world things usually need more than one number to
  describe — a house needs (size, bedrooms, age, location-score), not just
  one number. A vector bundles all those numbers as a single object you can
  do math on together.
- **Real-life analogy**: A GPS coordinate (latitude, longitude) is a
  2-dimensional vector. A grocery list with quantities is a vector over item
  types.
- **What happens without it**: You'd have to track every single feature
  (age, income, height, ...) as a separate independent variable with its own
  name, instead of one bundled object — code and formulas become unmanageable
  past a few features.
- **Where you'll see it in ML/AI**: A single data row (e.g. one house's
  features) is a vector. A word embedding (a word turned into numbers) is a
  vector. Model weights for one layer are often a vector.

---

### Matrix

- **Symbol**: uppercase bold letter, e.g. **X** or $A$, written as a grid:
  $\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$
- **Pronounced**: "matrix A"
- **Plain-English meaning**: A 2D grid of numbers — rows and columns — exactly
  like a spreadsheet or a 2D array/table.
- **Formula**: $A_{ij}$ = the number in row `i`, column `j` of matrix `A`.
- **Why it's used**: A whole dataset is naturally a table: rows = data
  points/examples, columns = features. A matrix lets you treat "the entire
  dataset" as one mathematical object instead of thousands of separate
  vectors.
- **Real-life analogy**: A spreadsheet of students (rows) with columns for
  age, grade, attendance — that whole spreadsheet is a matrix.
- **What happens without it**: You'd have to process every row one at a time
  with separate formulas instead of applying one operation to the whole
  dataset at once — much slower and harder to reason about (this is also why
  GPUs, which are matrix-multiplication machines, make ML fast).
- **Where you'll see it in ML/AI**: Your entire training dataset `X` (rows =
  examples, columns = features), the weight matrices inside every neural
  network layer, image data (pixels arranged in a grid).

---

### Transpose

- **Symbol**: $A^T$ (also written $A'$)
- **Pronounced**: "A transpose"
- **Plain-English meaning**: Flip a matrix so rows become columns and columns
  become rows — like rotating a spreadsheet 90 degrees.
- **Formula**: If $A_{ij} = v$, then $A^T_{ji} = v$.
- **Why it's used**: Needed to make matrix shapes compatible for
  multiplication, and to switch between "row of features per example" and
  "column of features per example" layouts.
- **Real-life analogy**: Turning a list of "name: score" pairs written as
  rows into the same data written as columns — same information, different
  layout, so it lines up with whatever else you're combining it with.
- **What happens without it**: Many matrix operations (like dot products
  between rows and columns) would be impossible to express — shapes simply
  wouldn't line up.
- **Where you'll see it in ML/AI**: Almost every neural network layer formula
  includes a transpose somewhere, e.g. $W^T x$, and it shows up constantly in
  derivations of backpropagation.

---

### Dot product (inner product)

- **Symbol**: $a \cdot b$ or $a^T b$
- **Pronounced**: "a dot b"
- **Plain-English meaning**: Multiply two vectors element-by-element, then
  add up all the results into a single number. It measures how much two
  vectors "point in the same direction" / how aligned they are.
- **Formula**: $a \cdot b = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \dots$
- **Why it's used**: It's the mathematical operation behind "weighted sum" —
  multiply each input by its importance (weight) and add them up. This is
  the single most common operation in ML.
- **Real-life analogy**: Calculating a final grade from weighted components:
  `0.3×exam + 0.5×homework + 0.2×participation` — that's a dot product between
  the scores vector and the weights vector.
- **What happens without it**: You'd have to write out "multiply and add"
  manually every time, term by term, instead of using one compact,
  hardware-optimized operation.
- **Where you'll see it in ML/AI**: This IS how a neuron computes its output
  before an activation function: $z = w \cdot x + b$ (weights dot inputs, plus
  bias). Also used to measure similarity between vectors (e.g. word
  embeddings, recommendation systems).

---

### Matrix multiplication

- **Symbol**: $AB$ or $A \times B$
- **Pronounced**: "A times B" / "A B"
- **Plain-English meaning**: Doing many dot products at once — each entry of
  the result is the dot product of a row from `A` and a column from `B`. It's
  how you apply a "transformation" (like a weight layer) to many data points
  simultaneously.
- **Formula**: $(AB)_{ij} = \sum_k A_{ik} B_{kj}$; requires columns of `A` to
  match rows of `B` in count.
- **Why it's used**: Lets you process an entire batch of data through an
  entire layer of weights in a single operation, instead of looping row by
  row — this is exactly what makes neural networks fast on GPUs.
- **Real-life analogy**: A factory assembly line applying the same set of
  operations to every item on the conveyor belt at once, rather than
  processing one item fully before starting the next.
- **What happens without it**: You'd need explicit loops over every data
  point and every weight — mathematically identical, but painfully slow and
  verbose, and unable to use hardware acceleration effectively.
- **Where you'll see it in ML/AI**: Every layer of every neural network:
  `output = activation(X @ W + b)` — the `X @ W` is matrix multiplication
  applying weights to a whole batch of inputs at once.

---

### Identity matrix

- **Symbol**: $I$
- **Pronounced**: "the identity matrix" / "I"
- **Plain-English meaning**: The matrix version of the number 1 — a grid of
  0s with 1s down the diagonal. Multiplying anything by it changes nothing.
- **Formula**: $I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ (for 2×2);
  $AI = A$ and $IA = A$.
- **Why it's used**: Needed as a mathematical "do-nothing" placeholder, e.g.
  to define matrix inverses ($A A^{-1} = I$) or as a starting point in
  regularization formulas.
- **Real-life analogy**: Multiplying a price by 1 — it's the "neutral"
  operation that leaves things unchanged.
- **What happens without it**: You couldn't formally define what an "inverse"
  matrix means (an inverse is defined as "the thing that gets you back to
  identity"), and some regularization formulas couldn't be written.
- **Where you'll see it in ML/AI**: Ridge regression regularization term
  ($\lambda I$ added to a matrix before inverting it), and in derivations
  involving matrix inverses.

---

### Norm (vector length)

- **Symbol**: $\|x\|$ or $\|x\|_2$ (L2 norm), $\|x\|_1$ (L1 norm)
- **Pronounced**: "the norm of x" / "L2 norm of x" / "L1 norm of x"
- **Plain-English meaning**: A single number representing "how big" a vector
  is overall, regardless of direction. L2 is straight-line ("as the crow
  flies") length; L1 is "taxicab" distance — sum of absolute values, like
  walking city blocks.
- **Formula**: $\|x\|_2 = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}$,
  $\|x\|_1 = |x_1| + |x_2| + \dots + |x_n|$
- **Why it's used**: You need a single scalar to compare/measure/penalize
  vectors — "how large are these weights overall," "how far apart are these
  two points."
- **Real-life analogy**: L2 is the straight-line distance a bird flies
  between two points on a map; L1 is the distance a taxi drives along a city
  grid of streets (can't cut diagonally through buildings).
- **What happens without it**: You couldn't summarize a whole vector's
  magnitude with one comparable number — you'd be stuck comparing lists of
  numbers element by element with no overall sense of "bigger" or "smaller."
- **Where you'll see it in ML/AI**: Regularization (L1/L2 penalties on model
  weights to prevent overfitting — see [05 — ML-Specific Math](05-ml-specific-math.md)),
  measuring distance between data points (e.g. in k-nearest-neighbors,
  clustering).

---

### Eigenvalues and eigenvectors

- **Symbol**: $Av = \lambda v$ (`A` = matrix, `v` = eigenvector, `λ` =
  eigenvalue)
- **Pronounced**: "eigenvalue" (EYE-gen-value), "eigenvector" (EYE-gen-vector)
- **Plain-English meaning**: For most vectors, multiplying by a matrix
  changes both their length AND direction. An eigenvector is a special
  direction that a matrix only **stretches or shrinks** (by a factor of `λ`,
  the eigenvalue) without rotating/changing its direction at all.
- **Formula**: $Av = \lambda v$ — applying matrix `A` to vector `v` gives the
  same result as just scaling `v` by the number `λ`.
- **Why it's used**: Finds the "natural axes" along which a transformation
  behaves simply (pure scaling) — this reveals the most important underlying
  directions of variation in data.
- **Real-life analogy**: Spinning a football — it has a natural axis it
  prefers to spin around stably (the "eigen-axis"); spin it any other way and
  it wobbles unpredictably. The eigenvector is that stable axis.
- **What happens without it**: You couldn't identify the dominant
  directions/patterns hidden inside high-dimensional data, making dimension
  reduction and understanding "what matters most in the data" much harder.
- **Where you'll see it in ML/AI**: Principal Component Analysis (PCA) — used
  to compress high-dimensional data down to its most important few
  directions (dimensionality reduction) by finding the eigenvectors of the
  data's covariance matrix.

---

### Determinant (brief mention)

- **Symbol**: $\det(A)$ or $|A|$
- **Pronounced**: "the determinant of A"
- **Plain-English meaning**: A single number summarizing how much a matrix
  "scales area/volume" when used as a transformation, and whether it flips
  orientation. If it's `0`, the matrix squashes space into a lower dimension
  (information is lost / not invertible).
- **Formula**: For a 2×2 matrix $\begin{bmatrix} a & b \\ c & d
  \end{bmatrix}$: $\det = ad - bc$.
- **Why it's used**: Tells you whether a matrix can be "undone" (inverted) —
  a determinant of 0 means it can't be, which matters for solving systems of
  equations.
- **Real-life analogy**: If you flatten a 3D box into a 2D sheet of paper,
  you've lost the ability to recover the original box's volume — a
  determinant of 0 signals this kind of information loss.
- **What happens without it**: You couldn't quickly check whether a system of
  equations has a unique solution, or whether a matrix inversion (needed in
  some ML formulas like linear regression's closed-form solution) is even
  possible.
- **Where you'll see it in ML/AI**: Checking whether the closed-form solution
  to linear regression ($\theta = (X^TX)^{-1}X^Ty$) is computable — it
  requires $X^TX$ to be invertible, i.e. have a non-zero determinant.
