"""Figures for the IEEE report and the slide deck.

Reads the metrics written by `newsmlv2 train`, so a figure can never quote a number the
shipped model does not produce.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from newsmlv2 import config, serve  # noqa: E402

SNAPSHOT_ID = "v2-001"
OUT = config.ML_ROOT.parent / "submission" / "report" / "figures"

plt.rcParams.update({
    "font.size": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def reliability_and_coverage(metrics: dict) -> None:
    """The two charts that carry the calibration + abstention argument."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    bins = metrics["reliability"]
    claimed = [b["claimed"] for b in bins]
    actual = [b["actual"] for b in bins]
    left.plot([0, 1], [0, 1], "--", color="0.5", lw=0.8, label="perfect calibration")
    left.plot(claimed, actual, "o-", color="#c0392b", ms=3.5, lw=1.2, label="measured")
    left.set_xlabel("claimed confidence")
    left.set_ylabel("observed accuracy")
    left.set_title(f"(a) Reliability  (ECE = {metrics['validation']['ece']:.3f})")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.legend(frameon=False, loc="upper left")

    curve = metrics["coverage_curve"]
    cuts = [p["cut"] for p in curve]
    right.plot(cuts, [p["coverage"] for p in curve], color="#2c3e50", lw=1.2,
               label="articles filed")
    right.plot(cuts, [p["accuracy_on_kept"] for p in curve], color="#c0392b", lw=1.2,
               label="accuracy on filed")
    right.axvline(metrics["cut"], color="0.4", ls=":", lw=1.0)
    right.annotate(f"shipping cut {metrics['cut']:.2f}", xy=(metrics["cut"], 0.32),
                   xytext=(metrics["cut"] + 0.03, 0.28), fontsize=7, color="0.3")
    right.set_xlabel("confidence cut")
    right.set_ylabel("proportion")
    right.set_title("(b) The abstention trade-off")
    right.set_xlim(0, 0.95)
    right.set_ylim(0.25, 1.02)
    right.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", bbox_inches="tight")
    plt.close(fig)


def per_class(metrics: dict) -> None:
    """Per-class F1 with bootstrap intervals, so a thin class cannot be misread."""
    rows = sorted(metrics["validation"]["per_class"], key=lambda c: c["f1"])
    names = [r["topic"].replace("_", " ") for r in rows]
    f1 = [r["f1"] for r in rows]
    low = [r["f1"] - r["low"] for r in rows]
    high = [r["high"] - r["f1"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.barh(names, f1, color="#4a6fa5", height=0.62)
    ax.errorbar(f1, names, xerr=[low, high], fmt="none", ecolor="0.25",
                elinewidth=0.8, capsize=2)
    for y, row in enumerate(rows):
        ax.text(0.02, y, f"n={row['support']}", va="center", fontsize=6, color="white")
    ax.set_xlabel("validation macro-F1 (95% CI)")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "per-class.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    path = config.ARTIFACT_DIR / "models" / SNAPSHOT_ID / serve.METRICS
    if not path.is_file():
        raise SystemExit(f"no metrics at {path}; run: uv run newsmlv2 train --id {SNAPSHOT_ID}")

    metrics = json.loads(path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    reliability_and_coverage(metrics)
    per_class(metrics)
    print(f"wrote {OUT}/calibration.png and {OUT}/per-class.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
