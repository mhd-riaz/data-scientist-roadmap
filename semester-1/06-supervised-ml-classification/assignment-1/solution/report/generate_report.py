"""Generates the Lab 1 PDF report (report/Lab1_Report.pdf) from the fixed
results obtained by executing the two completed notebooks. Run with:
    python generate_report.py
"""
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image, PageBreak, ListFlowable, ListItem)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "Lab1_Report.pdf")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBig", fontSize=20, leading=26, alignment=1, spaceAfter=12))
styles.add(ParagraphStyle(name="CenterMeta", fontSize=11, leading=16, alignment=1))
styles.add(ParagraphStyle(name="H1", fontSize=15, leading=20, spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#1a3d5c")))
styles.add(ParagraphStyle(name="H2", fontSize=12.5, leading=17, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#2a5a82")))
styles.add(ParagraphStyle(name="Body", fontSize=10.2, leading=14.5, spaceAfter=6))
styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=8, leading=10.5, backColor=colors.HexColor("#f2f2f2")))

story = []

# ---------------------------------------------------------------- Title page
story.append(Spacer(1, 5 * cm))
story.append(Paragraph("Lab 1: Model Selection and Comparative Analysis", styles["TitleBig"]))
story.append(Paragraph("Manual Grid Search vs. Scikit-learn GridSearchCV for Classifier Hyperparameter Tuning", styles["CenterMeta"]))
story.append(Spacer(1, 2 * cm))
meta = [
    "<b>Name:</b> Mohamed Riaz",
    "<b>Registration Number:</b> PES1PGE25DS037",
    "<b>Department:</b> Data Science &amp; AI",
    "<b>Course:</b> UX25CS635A &ndash; Supervised Machine Learning: Classification",
    "<b>Submission Date:</b> 14 August 2026",
]
for m in meta:
    story.append(Paragraph(m, styles["CenterMeta"]))
story.append(PageBreak())

# ---------------------------------------------------------------- Intro
story.append(Paragraph("1. Introduction", styles["H1"]))
story.append(Paragraph(
    "This lab implements a complete machine learning pipeline for binary classification and "
    "compares two approaches to hyperparameter tuning: a Grid Search implemented from scratch "
    "using nested loops and stratified k-fold cross-validation, and scikit-learn's built-in "
    "<b>GridSearchCV</b>. Three classifiers &ndash; Decision Tree, k-Nearest Neighbours (kNN), and "
    "Logistic Regression &ndash; are tuned inside a <b>StandardScaler &rarr; SelectKBest &rarr; "
    "Classifier</b> pipeline and evaluated on two datasets: <b>HR Attrition</b> and <b>Banknote "
    "Authentication</b>. Individual models are also combined into soft-voting ensembles, and all "
    "models are compared using Accuracy, Precision, Recall, F1-Score and ROC AUC.", styles["Body"]))

# ---------------------------------------------------------------- Dataset description
story.append(Paragraph("2. Dataset Description", styles["H1"]))

story.append(Paragraph("2.1 HR Attrition", styles["H2"]))
story.append(Paragraph(
    "Predicts whether an employee will leave the company (target: <b>Attrition</b>, Yes/No). "
    "The raw IBM dataset has 1470 instances and 34 work-related and personal attributes "
    "(e.g. age, monthly income, job role, overtime, years at company). After dropping the "
    "constant/ID column <i>EmployeeNumber</i> and one-hot encoding categorical variables, the "
    "feature space expands to <b>46 numeric features</b>. The target is imbalanced, with an "
    "attrition rate of about 16.1% in the training split (1029 train / 441 test, 70/30 stratified split).",
    styles["Body"]))

story.append(Paragraph("2.2 Banknote Authentication", styles["H2"]))
story.append(Paragraph(
    "Distinguishes genuine from forged banknotes (target: <b>class</b>, 0 = authentic, 1 = forged) "
    "using <b>4 features</b> extracted from wavelet-transformed banknote images: variance, "
    "skewness, curtosis and entropy. The dataset has 1372 instances (960 train / 412 test, "
    "70/30 stratified split) and is close to balanced, with a forged rate of about 44.5% in the "
    "training split.", styles["Body"]))

# ---------------------------------------------------------------- Methodology
story.append(Paragraph("3. Methodology", styles["H1"]))
story.append(Paragraph(
    "<b>Hyperparameter Tuning &amp; Grid Search:</b> Hyperparameters (e.g. tree depth, number of "
    "neighbours, regularization strength) are not learned from data directly and must be chosen "
    "externally. Grid Search evaluates every combination in a predefined parameter grid and picks "
    "the combination that yields the best cross-validated score.", styles["Body"]))
story.append(Paragraph(
    "<b>K-Fold Cross-Validation:</b> To get a robust performance estimate for each hyperparameter "
    "combination, the training data is split into 5 stratified folds. Each fold is used once as a "
    "validation set while the pipeline is fit on the remaining 4 folds; the mean ROC AUC across "
    "the 5 folds is used as the score for that combination.", styles["Body"]))
story.append(Paragraph(
    "<b>Pipeline:</b> Every model uses the same three-stage pipeline: <i>StandardScaler</i> "
    "(zero mean, unit variance) &rarr; <i>SelectKBest</i> (top-k features by the f_classif ANOVA "
    "F-statistic, with k itself tuned) &rarr; <i>Classifier</i> (Decision Tree / kNN / Logistic "
    "Regression). Fitting the scaler and feature selector only within each training fold (never "
    "on the validation fold) prevents data leakage.", styles["Body"]))
story.append(Paragraph(
    "<b>Part 1 &ndash; Manual Implementation:</b> For each classifier, all combinations of its "
    "parameter grid are generated with <i>itertools.product</i>. For every combination, the "
    "pipeline is rebuilt, its parameters set with <i>set_params</i>, fit on each training fold and "
    "scored with <i>roc_auc_score</i> on the corresponding validation fold. The combination with "
    "the highest mean AUC is kept and a final pipeline is refit on the full training set.", styles["Body"]))
story.append(Paragraph(
    "<b>Part 2 &ndash; Scikit-learn Implementation:</b> The same pipeline and parameter grid are "
    "passed to <i>GridSearchCV</i> with <i>scoring='roc_auc'</i> and the same "
    "<i>StratifiedKFold(n_splits=5)</i> cross-validator, then fit on the training data. "
    "GridSearchCV parallelizes the search with <i>n_jobs=-1</i> and directly exposes "
    "<i>best_estimator_</i>, <i>best_params_</i> and <i>best_score_</i>.", styles["Body"]))
story.append(Paragraph(
    "<b>Voting Classifiers:</b> The three tuned models are combined into a soft-voting ensemble. "
    "In the manual implementation this is done by hand (majority vote of predicted class labels, "
    "average of predicted probabilities); in the built-in implementation, scikit-learn's "
    "<i>VotingClassifier(voting='soft')</i> is used, which averages predicted probabilities and "
    "assigns the class directly &mdash; a subtle behavioural difference discussed in Section 4.", styles["Body"]))

# ---------------------------------------------------------------- Results
story.append(Paragraph("4. Results and Analysis", styles["H1"]))


def metrics_table(rows, col_labels):
    data = [col_labels] + rows
    t = Table(data, colWidths=[3.0*cm, 2.6*cm, 2.1*cm, 2.1*cm, 1.9*cm, 1.9*cm, 2.1*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
    ]))
    return t


col_labels = ["Model", "Method", "Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]

hr_rows = [
    ["Decision Tree", "Manual", "0.8231", "0.3333", "0.0986", "0.1522", "0.7107"],
    ["Decision Tree", "Built-in", "0.8231", "0.3333", "0.0986", "0.1522", "0.7107"],
    ["kNN", "Manual", "0.8186", "0.3784", "0.1972", "0.2593", "0.7236"],
    ["kNN", "Built-in", "0.8186", "0.3784", "0.1972", "0.2593", "0.7236"],
    ["Logistic Regression", "Manual", "0.8798", "0.7368", "0.3944", "0.5138", "0.8187"],
    ["Logistic Regression", "Built-in", "0.8798", "0.7368", "0.3944", "0.5138", "0.8187"],
    ["Voting Classifier", "Manual", "0.8299", "0.4231", "0.1549", "0.2268", "0.7966"],
    ["Voting Classifier", "Built-in", "0.8458", "0.5556", "0.2113", "0.3061", "0.7966"],
]

bank_rows = [
    ["Decision Tree", "Manual", "0.9854", "0.9784", "0.9891", "0.9837", "0.9856"],
    ["Decision Tree", "Built-in", "0.9854", "0.9784", "0.9891", "0.9837", "0.9856"],
    ["kNN", "Manual", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
    ["kNN", "Built-in", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
    ["Logistic Regression", "Manual", "0.9903", "0.9786", "1.0000", "0.9892", "0.9999"],
    ["Logistic Regression", "Built-in", "0.9903", "0.9786", "1.0000", "0.9892", "0.9999"],
    ["Voting Classifier", "Manual", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
    ["Voting Classifier", "Built-in", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
]

story.append(Paragraph("4.1 HR Attrition", styles["H2"]))
story.append(Paragraph(
    "Best hyperparameters found (identical for manual and built-in search, as expected since both "
    "use the same StratifiedKFold(n_splits=5, random_state=42) splitter and the same grid): "
    "Decision Tree &rarr; k=5, max_depth=3, min_samples_split=2 (CV AUC 0.7152); kNN &rarr; k=10, "
    "n_neighbors=9, weights=distance (CV AUC 0.7226); Logistic Regression &rarr; k=all, C=0.1, "
    "penalty=l2 (CV AUC 0.8329).", styles["Body"]))
story.append(metrics_table(hr_rows, col_labels))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "<b>Manual vs. built-in:</b> for all three individually-tuned classifiers the two "
    "implementations are identical on every metric, confirming the manual search reproduces "
    "GridSearchCV exactly when the same grid and CV splitter are used. The <i>Voting Classifier</i> "
    "is the only place they diverge (Accuracy 0.8299 vs 0.8458): both average the same predicted "
    "probabilities (hence identical ROC AUC = 0.7966), but the manual implementation assigns the "
    "final label from a majority vote of each model's hard predictions, while scikit-learn's "
    "VotingClassifier(voting='soft') thresholds the averaged probability directly at 0.5 &ndash; a "
    "small but real algorithmic difference, not a bug.", styles["Body"]))
story.append(Paragraph(
    "<b>Best model:</b> Logistic Regression achieves the highest test ROC AUC (0.8187), "
    "outperforming both the Decision Tree, kNN and even the Voting Classifier. HR Attrition is a "
    "mostly linear-separable, highly imbalanced problem (~16% positive class) with many one-hot "
    "encoded categorical features; Logistic Regression's L2-regularized linear decision boundary "
    "generalizes better here than the higher-variance Decision Tree/kNN, and dragging those weaker "
    "models into the vote actually pulls the ensemble's performance down.", styles["Body"]))
story.append(Image(os.path.join(FIG, "hr_plot_1.png"), width=16.5*cm, height=6.5*cm))
story.append(Paragraph("Figure 1: ROC curves and confusion matrix &ndash; HR Attrition (Manual).", styles["Body"]))
story.append(Image(os.path.join(FIG, "hr_plot_2.png"), width=16.5*cm, height=6.5*cm))
story.append(Paragraph("Figure 2: ROC curves and confusion matrix &ndash; HR Attrition (Built-in).", styles["Body"]))

story.append(PageBreak())
story.append(Paragraph("4.2 Banknote Authentication", styles["H2"]))
story.append(Paragraph(
    "Best hyperparameters found (again identical across manual and built-in search): Decision "
    "Tree &rarr; k=3, max_depth=10, min_samples_split=10 (CV AUC 0.9869); kNN &rarr; k=3, "
    "n_neighbors=5, weights=distance (CV AUC 1.0000); Logistic Regression &rarr; k=3, C=10, "
    "penalty=l2 (CV AUC 0.9995).", styles["Body"]))
story.append(metrics_table(bank_rows, col_labels))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "<b>Manual vs. built-in:</b> every model, including the Voting Classifier this time, matches "
    "exactly between the two implementations. With only 4 highly-informative wavelet features and "
    "a near-perfectly separable dataset, the averaged-probability and majority-vote strategies "
    "agree on every test sample, so the earlier source of divergence does not appear here.", styles["Body"]))
story.append(Paragraph(
    "<b>Best model:</b> kNN and the Voting Classifier both reach a perfect test ROC AUC of 1.0000; "
    "Logistic Regression is close behind (0.9999) and the Decision Tree is the weakest (0.9856). "
    "The banknote features are strongly and fairly simply correlated with authenticity, so a "
    "distance-based method like kNN with only 3 selected features can separate the classes almost "
    "perfectly, and the Decision Tree's axis-aligned splits are the least efficient way to capture "
    "that structure, explaining why it trails the other two.", styles["Body"]))
story.append(Image(os.path.join(FIG, "bank_plot_1.png"), width=16.5*cm, height=6.5*cm))
story.append(Paragraph("Figure 3: ROC curves and confusion matrix &ndash; Banknote Authentication (Manual).", styles["Body"]))
story.append(Image(os.path.join(FIG, "bank_plot_2.png"), width=16.5*cm, height=6.5*cm))
story.append(Paragraph("Figure 4: ROC curves and confusion matrix &ndash; Banknote Authentication (Built-in).", styles["Body"]))

# ---------------------------------------------------------------- Screenshots
story.append(PageBreak())
story.append(Paragraph("5. Screenshots of Execution Output", styles["H1"]))
story.append(Paragraph(
    "Full, cell-by-cell console output (grid search progress, best parameters, per-model and "
    "voting classifier metrics) is preserved in the executed notebooks under "
    "<i>solution/notebooks/</i>. Representative excerpts are reproduced below.", styles["Body"]))

hr_snippet = (
    "RUNNING MANUAL GRID SEARCH FOR HR ATTRITION\n"
    "--- Manual Grid Search for Decision Tree ---\n"
    "Best parameters for Decision Tree: {'feature_selection__k': 5, 'classifier__max_depth': 3,\n"
    " 'classifier__min_samples_split': 2}\n"
    "Best cross-validation AUC: 0.7152\n"
    "--- Manual Grid Search for Logistic Regression ---\n"
    "Best parameters for Logistic Regression: {'feature_selection__k': 'all', 'classifier__C': 0.1,\n"
    " 'classifier__penalty': 'l2'}\n"
    "Best cross-validation AUC: 0.8329\n\n"
    "EVALUATING BUILT-IN MODELS FOR HR ATTRITION\n"
    "Logistic Regression:\n"
    "  Accuracy: 0.8798  Precision: 0.7368  Recall: 0.3944  F1-Score: 0.5138  ROC AUC: 0.8187\n"
    "Built-in Voting Classifier:\n"
    "  Accuracy: 0.8458, Precision: 0.5556, Recall: 0.2113, F1: 0.3061, AUC: 0.7966"
)
bank_snippet = (
    "RUNNING BUILT-IN GRID SEARCH FOR BANKNOTE AUTHENTICATION\n"
    "Best params for kNN: {'classifier__n_neighbors': 5, 'classifier__weights': 'distance',\n"
    " 'feature_selection__k': 3}\n"
    "Best CV score: 1.0000\n\n"
    "EVALUATING BUILT-IN MODELS FOR BANKNOTE AUTHENTICATION\n"
    "kNN:\n"
    "  Accuracy: 1.0000  Precision: 1.0000  Recall: 1.0000  F1-Score: 1.0000  ROC AUC: 1.0000\n"
    "Built-in Voting Classifier:\n"
    "  Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1: 1.0000, AUC: 1.0000"
)

for title, snippet in [("HR Attrition (excerpt)", hr_snippet), ("Banknote Authentication (excerpt)", bank_snippet)]:
    story.append(Paragraph(title, styles["H2"]))
    story.append(Paragraph(snippet.replace("\n", "<br/>"), styles["Mono"]))
    story.append(Spacer(1, 8))

# ---------------------------------------------------------------- Conclusion
story.append(Paragraph("6. Conclusion", styles["H1"]))
story.append(ListFlowable([
    ListItem(Paragraph(
        "The manual grid search reproduces scikit-learn's GridSearchCV exactly for every "
        "individually-tuned classifier on both datasets, confirming a correct understanding of "
        "nested cross-validation and pipeline parameter passing.", styles["Body"])),
    ListItem(Paragraph(
        "GridSearchCV is far more practical for real work: it is shorter to write, parallelizes "
        "the search across folds/combinations with n_jobs=-1, and exposes best_estimator_/"
        "best_params_/best_score_ directly, whereas the manual version required explicitly looping "
        "over folds and combinations and re-fitting the final pipeline by hand.", styles["Body"])),
    ListItem(Paragraph(
        "Voting classifiers help most when the base models are already strong and diverse; on HR "
        "Attrition, voting actually underperformed the single best model (Logistic Regression) "
        "because it was dragged down by the much weaker Decision Tree and kNN models, while on the "
        "near-perfectly-separable Banknote dataset voting matched the best individual model.", styles["Body"])),
    ListItem(Paragraph(
        "The only observed manual-vs-built-in discrepancy was in the Voting Classifier on HR "
        "Attrition, and it was traced to a genuine algorithmic difference (majority-vote-of-labels "
        "vs. threshold-on-averaged-probability) rather than an implementation bug &ndash; a useful "
        "reminder that scikit-learn's convenience functions can encode subtly different design "
        "choices than a naive from-scratch implementation.", styles["Body"])),
    ListItem(Paragraph(
        "Overall, this lab reinforced how feature scaling, feature selection and hyperparameter "
        "tuning must be chained inside a single cross-validated pipeline to avoid data leakage, and "
        "how the right classifier choice depends heavily on the dataset's separability and class "
        "balance.", styles["Body"])),
], bulletType="bullet"))

doc = SimpleDocTemplate(OUT, pagesize=A4,
                         leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
                         title="Lab 1 Report - Mohamed Riaz - PES1PGE25DS037")
doc.build(story)
print("Report written to", OUT)
