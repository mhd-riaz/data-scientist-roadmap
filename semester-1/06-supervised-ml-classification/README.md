# Supervised ML — Classification

## Overview

Predicting categorical outcomes: from logistic regression to ensemble methods.

## Topics Covered

Covered by the notes in this folder:

- [x] ML foundations — $\langle P,T,E\rangle$, learning styles, train/validation/test
- [x] Linear regression & gradient descent (prerequisite for logistic regression)
- [x] Logistic regression — sigmoid, decision boundary, one-vs-all
- [x] k-Nearest Neighbors (k-NN), including k-NN regression
- [x] Decision trees — entropy, information gain, ID3, pruning
- [x] Model evaluation — confusion matrix, precision/recall/F1, ROC-AUC
- [x] Ensembles — bagging, boosting, Random Forest, AdaBoost, GBM, XGBoost
- [x] Ensemble methods deep dive — weak-learner diversity/taxonomy, full AdaBoost algorithm, Gradient Boosting for classification, Random Forest proximity-matrix missing-data imputation

Not present in the source material for this subject:

- [ ] Naive Bayes
- [ ] Support Vector Machines (SVM)
- [ ] Explicit class-imbalance handling (SMOTE, class weights)
- [ ] Cross-validation

## Notes

See [notes/](notes/) for the full reference text. Read in order the first time — each chapter
assumes the previous one.

- [00 — Module Map & Study Checklist](notes/00-study-checklist.md) — start here
- [01 — Machine Learning Foundations](notes/01-ml-foundations.md) — $\langle P,T,E\rangle$, learning styles, notation, data splits
- [02 — Linear Regression & Gradient Descent](notes/02-linear-regression-and-gradient-descent.md) — residuals, cost function, optimisation, convergence
- [03 — Logistic Regression](notes/03-logistic-regression.md) — sigmoid, probability, decision boundary, one-vs-all
- [04 — K-Nearest Neighbours](notes/04-knn.md) — distance, choosing K, encoding, scaling
- [05 — Decision Trees & ID3](notes/05-decision-trees-and-id3.md) — entropy, information gain, inductive bias, pruning
- [06 — Ensemble Learning](notes/06-ensemble-learning.md) — bagging, boosting, Random Forest, AdaBoost, XGBoost
- [06b — Ensemble Methods Deep Dive](notes/06b-ensemble-methods-deep-dive.md) — full AdaBoost algorithm, Gradient Boosting for classification, Random Forest internals (OOB, voting, proximity-matrix imputation)
- [07 — Performance Metrics](notes/07-performance-metrics.md) — confusion matrix, precision/recall/F1, ROC, AUC
- [08 — Exam Preparation](notes/08-exam-preparation.md) — model answers, practice questions, revision, gaps

## Notebooks

See [notebooks/](notebooks/) for hands-on practice.

## Resources

- Tom M. Mitchell, *Machine Learning* — chapters 2–3 cover the inductive-bias and
  decision-tree material listed as gaps in [08 §5](notes/08-exam-preparation.md#5-gaps-to-look-up).
- [Math dictionary](../../math_dictionary/README.md) for notation and symbols.

## Key Takeaways

- Classification is decided by the **target's data type**, not by the algorithm you prefer.
- Logistic regression reuses linear regression's optimiser unchanged — only the hypothesis
  and cost function differ.
- KNN is pure distance arithmetic, so **feature scaling is mandatory**; decision trees never
  need it.
- ID3 is a greedy search with a preference bias for **short trees**; unpruned, it overfits.
- Bagging reduces **variance**, boosting reduces **bias** — that single sentence explains
  every design difference between Random Forest and XGBoost.
- **Never judge an imbalanced classifier by accuracy.** Report precision, recall, F1 and AUC.

Back to [Semester 1 index](../README.md).
