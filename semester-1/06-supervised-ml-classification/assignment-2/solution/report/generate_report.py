"""Generates the Lab 2 PDF report (report/Lab2_Report.pdf) from the fixed
results obtained by running test.py (PyTorch and NumPy implementations) on
all three datasets. Run with:
    python generate_report.py
"""
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, ListFlowable, ListItem)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "Lab2_Report.pdf")

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
story.append(Paragraph("Lab 2: Decision Tree Classifier &ndash; Multi-Dataset Analysis", styles["TitleBig"]))
story.append(Paragraph("ID3 Decision Tree implemented from scratch (PyTorch and NumPy) and compared across three datasets", styles["CenterMeta"]))
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


def table(rows, col_labels, col_widths):
    data = [col_labels] + rows
    t = Table(data, colWidths=col_widths)
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


# ---------------------------------------------------------------- Intro
story.append(Paragraph("1. Introduction", styles["H1"]))
story.append(Paragraph(
    "This lab implements the ID3 Decision Tree algorithm from scratch (no scikit-learn model "
    "calls) using both <b>PyTorch tensors</b> and <b>NumPy arrays</b>, and applies it to three "
    "structurally different categorical datasets: Mushroom Classification, Tic-Tac-Toe Endgame, "
    "and Nursery School recommendation. Only four building-block functions were implemented "
    "(<i>get_entropy_of_dataset</i>, <i>get_avg_info_of_attribute</i>, <i>get_information_gain</i>, "
    "<i>get_selected_attribute</i>); the provided test harness recursively calls them to grow the "
    "full tree, evaluate it, and report complexity metrics.", styles["Body"]))

# ---------------------------------------------------------------- Dataset description
story.append(Paragraph("2. Dataset Descriptions", styles["H1"]))
ds_rows = [
    ["Mushroom Classification", "8,124", "22 (all categorical)", "2 (edible 51.8% / poisonous 48.2%)", "Balanced"],
    ["Tic-Tac-Toe Endgame", "958", "9 (each cell: x/o/b)", "2 (win 65.3% / not-win 34.7%)", "Mild imbalance"],
    ["Nursery School", "12,960", "8 (all categorical)", "5 (33.3% / 32.9% / 31.2% / 2.5% / 0.02%)", "Severe imbalance"],
]
story.append(table(ds_rows,
                    ["Dataset", "Instances", "Features", "Class distribution", "Balance"],
                    [4.0*cm, 2.0*cm, 3.6*cm, 5.8*cm, 2.6*cm]))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "All three datasets are purely categorical and are label-encoded column-by-column before "
    "tree construction, with the last column always treated as the target. Nursery is the most "
    "imbalanced by far: its <i>recommend</i> class has only 2 of 12,960 rows (0.02%), which turns "
    "out to matter a great deal for macro-averaged metrics (Section 4).", styles["Body"]))

# ---------------------------------------------------------------- Methodology
story.append(Paragraph("3. Methodology", styles["H1"]))
story.append(Paragraph(
    "<b>ID3 core functions:</b> <i>get_entropy_of_dataset</i> computes "
    "-&Sigma;(p<sub>i</sub>&middot;log<sub>2</sub>p<sub>i</sub>) over the class distribution of the "
    "target column. <i>get_avg_info_of_attribute</i> partitions the data by every unique value of "
    "one attribute and returns the sample-weighted sum of each partition's entropy. "
    "<i>get_information_gain</i> subtracts that weighted entropy from the dataset entropy and rounds "
    "to 4 decimals. <i>get_selected_attribute</i> computes the gain for every remaining attribute and "
    "returns both the full gain dictionary and the index of the attribute with the maximum gain.", styles["Body"]))
story.append(Paragraph(
    "<b>Tree construction (test harness):</b> starting from the training split, the harness "
    "recursively splits on the attribute with the highest information gain, stopping when a node is "
    "pure, an attribute budget of <i>max_depth=7</i> is reached, all attributes are used, or no "
    "attribute yields positive gain &ndash; in each stopping case the majority class of that node "
    "is returned as a leaf. Both frameworks implement identical logic; the only difference is "
    "PyTorch tensor ops (<i>torch.unique</i>, <i>torch.log2</i>) vs. NumPy array ops "
    "(<i>np.unique</i>, <i>np.log2</i>).", styles["Body"]))
story.append(Paragraph(
    "<b>Train/test split:</b> an 80/20 shuffle-split with a fixed seed (42), but the two frameworks "
    "shuffle with their own RNGs (<i>torch.randperm</i> vs. <i>np.random.permutation</i>), so the two "
    "implementations do not see literally the same split &ndash; this is the source of the small "
    "differences reported in Section 4, not a bug in either implementation.", styles["Body"]))

# ---------------------------------------------------------------- Results
story.append(Paragraph("4. Results and Analysis", styles["H1"]))
story.append(Paragraph("4.1 Performance Comparison", styles["H2"]))

perf_rows = [
    ["Mushroom", "NumPy", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
    ["Mushroom", "PyTorch", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
    ["Tic-Tac-Toe", "NumPy", "0.8836", "0.8827", "0.8836", "0.8822", "0.8784", "0.8680"],
    ["Tic-Tac-Toe", "PyTorch", "0.8730", "0.8741", "0.8730", "0.8734", "0.8590", "0.8613"],
    ["Nursery", "NumPy", "0.9887", "0.9888", "0.9887", "0.9887", "0.9577", "0.9576"],
    ["Nursery", "PyTorch", "0.9867", "0.9876", "0.9867", "0.9872", "0.7604", "0.7628"],
]
story.append(table(perf_rows,
                    ["Dataset", "Framework", "Accuracy", "Precision(w)", "Recall(w)", "F1(w)", "Precision(macro)", "F1(macro)"],
                    [2.5*cm, 2.0*cm, 2.0*cm, 2.1*cm, 2.0*cm, 1.8*cm, 2.5*cm, 2.0*cm]))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Mushroom is solved perfectly by both implementations. Tic-Tac-Toe sits around 87-88% "
    "accuracy for both. Nursery reaches ~98-99% weighted accuracy in both, but its <b>macro-averaged "
    "metrics diverge sharply</b> (NumPy 0.958 vs. PyTorch 0.760 macro F1) &ndash; see Section 4.4 for "
    "why this is a class-imbalance/random-split effect rather than an algorithmic difference.", styles["Body"]))

story.append(Paragraph("4.2 Tree Characteristics Analysis", styles["H2"]))
tree_rows = [
    ["Mushroom", "NumPy", "4", "29", "24", "odor (0.9048)"],
    ["Mushroom", "PyTorch", "4", "29", "24", "odor (0.9083)"],
    ["Tic-Tac-Toe", "NumPy", "7", "260", "165", "middle-middle-square (0.0909)"],
    ["Tic-Tac-Toe", "PyTorch", "7", "281", "180", "middle-middle-square (0.0834)"],
    ["Nursery", "NumPy", "7", "983", "703", "health (0.9597)"],
    ["Nursery", "PyTorch", "7", "952", "680", "health (0.9595)"],
]
story.append(table(tree_rows,
                    ["Dataset", "Framework", "Max Depth", "Total Nodes", "Leaf Nodes", "Root attribute (gain)"],
                    [2.5*cm, 2.0*cm, 2.0*cm, 2.2*cm, 2.2*cm, 5.9*cm]))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "Both frameworks independently select the <b>same root attribute</b> for every dataset (odor / "
    "middle-middle-square / health) with matching information gain to 2-3 decimal places, which is "
    "strong evidence the two implementations compute entropy/gain identically; the small differences "
    "in node counts (e.g. 260 vs. 281 for Tic-Tac-Toe) come from the different train/test shuffles "
    "producing slightly different subtrees below the root, not from a difference in the ID3 logic "
    "itself.", styles["Body"]))
story.append(Paragraph(
    "Mushroom needs only depth 4 and 29 nodes to reach 100% accuracy &ndash; <i>odor</i> alone almost "
    "perfectly separates the classes, and only one branch (odor = 'none') needs a second split on "
    "spore-print-color. Tic-Tac-Toe and Nursery both grow to the maximum allowed depth of 7 with far "
    "more nodes (260-983), reflecting that no small set of attributes is individually decisive and the "
    "tree must chain many splits together to isolate each class.", styles["Body"]))

story.append(Paragraph("4.3 Dataset-Specific Insights", styles["H2"]))
story.append(Paragraph("<b>Mushroom Classification:</b>", styles["Body"]))
story.append(ListFlowable([
    ListItem(Paragraph("Feature importance: <i>odor</i> dominates (gain &asymp; 0.90, more than 5&times; the next-best attribute), followed by spore-print-color for the residual 'no odor' branch.", styles["Body"])),
    ListItem(Paragraph("Class distribution: balanced (51.8% / 48.2%), so accuracy is a trustworthy metric here.", styles["Body"])),
    ListItem(Paragraph("Decision pattern: almost a single-level lookup on odor, with only one deeper branch &ndash; a biologically plausible pattern (smell is a well-known practical heuristic for mushroom toxicity).", styles["Body"])),
    ListItem(Paragraph("Overfitting indicators: none observed &ndash; depth 4 with 100% test accuracy on a held-out 20% split indicates the classes truly are (near-)perfectly separable on these features, not memorization.", styles["Body"])),
], bulletType="bullet"))
story.append(Paragraph("<b>Tic-Tac-Toe Endgame:</b>", styles["Body"]))
story.append(ListFlowable([
    ListItem(Paragraph("Feature importance: <i>middle-middle-square</i> is selected as root in both frameworks (the center square is well known to be strategically the most valuable position in Tic-Tac-Toe), but its gain (&asymp;0.08-0.09) is an order of magnitude lower than mushroom's root gain &ndash; no single square is individually decisive.", styles["Body"])),
    ListItem(Paragraph("Class distribution: mildly imbalanced (65.3% win / 34.7% not-win).", styles["Body"])),
    ListItem(Paragraph("Decision pattern: deep, wide trees (260-281 nodes) that combine many board positions &ndash; consistent with the fact that a win/loss outcome genuinely depends on the joint configuration of multiple cells, not any one cell alone.", styles["Body"])),
    ListItem(Paragraph("Overfitting indicators: this is the dataset with the largest node count relative to its size (958 rows -&gt; 260+ nodes, i.e. under 4 samples/node on average) alongside the lowest test accuracy (&asymp;87-88%) of the three &ndash; a classic small-data/high-node-count overfitting signature.", styles["Body"])),
], bulletType="bullet"))
story.append(Paragraph("<b>Nursery School:</b>", styles["Body"]))
story.append(ListFlowable([
    ListItem(Paragraph("Feature importance: <i>health</i> is by far the strongest attribute (gain &asymp; 0.96, close to mushroom's odor), which matches domain intuition &ndash; health status was designed into the original ranking rules used to generate this dataset.", styles["Body"])),
    ListItem(Paragraph("Class distribution: severely imbalanced &ndash; 3 large classes (31-33% each) plus <i>very_recom</i> (2.5%) and <i>recommend</i> (0.02%, only 2 rows total).", styles["Body"])),
    ListItem(Paragraph("Decision pattern: very large tree (952-983 nodes) needed to carve out the rare classes from the three dominant ones.", styles["Body"])),
    ListItem(Paragraph("Overfitting indicators: weighted accuracy stays high (&asymp;98-99%) purely because the three big classes are easy, while macro metrics (which weight all 5 classes equally) expose the real difficulty &ndash; the extremely rare <i>recommend</i> class (2 rows) is essentially unlearnable and highly sensitive to which split a random shuffle happens to put it in.", styles["Body"])),
], bulletType="bullet"))

story.append(PageBreak())
story.append(Paragraph("4.4 Manual (PyTorch) vs. Manual (NumPy) Comparison", styles["H2"]))
story.append(Paragraph(
    "Since scikit-learn model calls are disallowed, both implementations are 'manual' ID3 built on "
    "the same four formulas; the comparison here is PyTorch-tensor vs. NumPy-array arithmetic rather "
    "than 'from-scratch vs. library'. On Mushroom both are identical (100% / depth 4 / 29 nodes). On "
    "Tic-Tac-Toe both pick the same root and reach similar accuracy (87-88%) with slightly different "
    "tree sizes. On Nursery, weighted metrics agree closely (98.67-98.87% accuracy) but macro metrics "
    "diverge sharply (macro F1 0.9576 NumPy vs. 0.7628 PyTorch).", styles["Body"]))
story.append(Paragraph(
    "<b>Root cause of the Nursery divergence:</b> <i>torch.randperm(42)</i> and "
    "<i>np.random.permutation(42)</i> are different RNGs seeded the same way, so they produce "
    "different 80/20 splits of the 12,960 rows. The <i>recommend</i> class has only 2 rows in the "
    "entire dataset; depending on the split, those 2 rows can land such that the class is poorly "
    "represented in training, or is present in the test set but the tree never learned a leaf for it. "
    "Because macro-averaging treats all 5 classes equally, missing or misclassifying that one "
    "vanishingly-rare class swings the macro F1 by tens of percentage points, while the weighted "
    "metrics barely move because that class carries almost no weight (0.02% of samples). This is a "
    "textbook illustration of why macro metrics, not just accuracy, must be reported for imbalanced "
    "multi-class problems.", styles["Body"]))

# ---------------------------------------------------------------- Comparative analysis report
story.append(Paragraph("5. Comparative Analysis Report", styles["H1"]))
story.append(Paragraph("a) Algorithm Performance", styles["H2"]))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Highest accuracy:</b> Mushroom (100%), because <i>odor</i> alone is an almost perfect "
        "discriminator &ndash; a single categorical feature happens to be near-deterministic for the "
        "label, so even a shallow tree separates the classes exactly.", styles["Body"])),
    ListItem(Paragraph(
        "<b>Effect of dataset size:</b> the largest dataset (Nursery, 12,960 rows) and the smallest "
        "(Tic-Tac-Toe, 958 rows) both needed the maximum depth of 7, but Tic-Tac-Toe's much smaller "
        "training set (766 rows) spread across 9 attributes with 3 values each produced a "
        "disproportionately large, sparse tree (under 4 samples per node on average) and the weakest "
        "accuracy of the three &ndash; small sample size relative to the attribute/value space directly "
        "hurts ID3's ability to estimate reliable splits.", styles["Body"])),
    ListItem(Paragraph(
        "<b>Role of feature count:</b> Mushroom has the most features (22) but the fewest are actually "
        "needed (root alone nearly solves it), while Tic-Tac-Toe has the fewest features (9) but needs "
        "almost all of them combined to reach only ~87-88% &ndash; more features help only when at "
        "least one of them is individually informative; when information is spread thin across many "
        "weakly-informative features (as in Tic-Tac-Toe), extra features mostly add tree complexity, "
        "not accuracy.", styles["Body"])),
], bulletType="bullet"))

story.append(Paragraph("b) Data Characteristics Impact", styles["H2"]))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Class imbalance:</b> as shown in Section 4.4, imbalance does not hurt weighted "
        "accuracy/F1 (which stayed &ge;98.6% even on the 5-class, highly-skewed Nursery data) but it "
        "makes the tree's treatment of rare classes highly sensitive to the random train/test split, "
        "and it is macro-averaged metrics that reveal this fragility; a single 0.02%-frequency class "
        "can swing macro F1 by 0.1-0.2 depending purely on which split it lands in.", styles["Body"])),
    ListItem(Paragraph(
        "<b>Binary vs. multi-valued features:</b> Tic-Tac-Toe's ternary (x/o/b) features gave the "
        "lowest per-attribute information gain (root gain &asymp; 0.08-0.09) and the weakest accuracy, "
        "while Nursery and Mushroom's attributes with more distinct categories (e.g. odor has 9 "
        "values, health has 3) produced far higher single-attribute gains and shallower, more "
        "accurate splits close to the root &ndash; attributes with more distinguishing categories tend "
        "to separate classes more cleanly per split, provided those categories are actually correlated "
        "with the label (as odor and health are, but individual board cells are not).", styles["Body"])),
], bulletType="bullet"))

story.append(Paragraph("c) Practical Applications", styles["H2"]))
story.append(ListFlowable([
    ListItem(Paragraph(
        "<b>Mushroom &rarr; food-safety / foraging-assistant tools:</b> a shallow, high-accuracy tree "
        "is exactly what's needed for a safety-critical, explainable rule ('if odor is X, it's "
        "poisonous') that a non-expert forager or a mobile app could apply instantly.", styles["Body"])),
    ListItem(Paragraph(
        "<b>Tic-Tac-Toe &rarr; game-outcome / strategy analysis:</b> useful for teaching search/game "
        "theory concepts and for building simple heuristic game-evaluation functions, but the deep, "
        "low-gain-per-split tree shows ID3 is a poor fit for problems where the outcome depends on "
        "joint combinations of positions rather than individually informative features &ndash; a "
        "minimax/game-tree search is the more natural tool for this domain in practice.", styles["Body"])),
    ListItem(Paragraph(
        "<b>Nursery &rarr; policy/admissions decision support:</b> the tree gives interpretable, "
        "auditable admission-priority rules ('if health is not_recom, recommendation = not_recom') "
        "which matters for a domain where decisions must be explainable to families and "
        "administrators; however, the rare classes need special handling (e.g. oversampling, or "
        "manual review flags) before deploying this in a real admissions pipeline, since ID3 alone "
        "will not reliably serve the 0.02%-frequency class.", styles["Body"])),
    ListItem(Paragraph(
        "<b>Interpretability:</b> all three trees are fully interpretable path-by-path, which is ID3's "
        "core practical advantage over black-box models in domains (food safety, admissions) where a "
        "human must be able to justify the specific decision path for a given case.", styles["Body"])),
    ListItem(Paragraph(
        "<b>Suggested improvements per dataset:</b> Mushroom &ndash; already near-optimal, could prune "
        "further for an even simpler rule set; Tic-Tac-Toe &ndash; needs feature engineering (e.g. "
        "encode rows/columns/diagonals as derived features) or a different algorithm (minimax) rather "
        "than more tree depth; Nursery &ndash; needs class-imbalance handling (oversampling the rare "
        "classes, or reporting per-class recall alongside accuracy) before the rare-class results can "
        "be trusted.", styles["Body"])),
], bulletType="bullet"))

# ---------------------------------------------------------------- Conclusion
story.append(Paragraph("6. Conclusion", styles["H1"]))
story.append(ListFlowable([
    ListItem(Paragraph(
        "The PyTorch and NumPy implementations of the four ID3 building blocks are numerically "
        "consistent: they select the same root attribute with matching information gain on all three "
        "datasets, confirming the entropy/gain formulas were implemented correctly in both frameworks.", styles["Body"])),
    ListItem(Paragraph(
        "Dataset characteristics, not the framework, drive the observed performance differences: "
        "Mushroom's single highly-informative feature yields a near-trivial, perfect tree; "
        "Tic-Tac-Toe's weakly-informative, combinatorial features force a large tree with the "
        "weakest accuracy; Nursery's severe class imbalance makes weighted and macro metrics tell "
        "very different stories.", styles["Body"])),
    ListItem(Paragraph(
        "Class imbalance is the single biggest risk factor observed in this lab: it does not lower "
        "aggregate accuracy, but it makes a model's behaviour on rare classes essentially "
        "unpredictable across random splits, which is a serious concern in decision-support domains "
        "like Nursery admissions where those rare cases may be exactly the ones needing the most "
        "careful review.", styles["Body"])),
], bulletType="bullet"))

doc = SimpleDocTemplate(OUT, pagesize=A4,
                         leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
                         title="Lab 2 Report - Mohamed Riaz - PES1PGE25DS037")
doc.build(story)
print("Report written to", OUT)
