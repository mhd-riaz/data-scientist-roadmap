"""Constants that make a run reproducible.

Every artifact written by this package records CLEANING_VERSION and SEED, so any
row in a snapshot can be traced back to the code that produced it.
"""

from __future__ import annotations

import os
from pathlib import Path

# Bump on any change to clean.py or admit.py that alters their output. Snapshots
# are keyed by this: a bump makes existing snapshots stale, not wrong.
CLEANING_VERSION = "1.0.0"

# Fixed seed for every sampling decision. Ground rule 8.
SEED = 20260823

ML_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ML_ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
ARTIFACT_DIR = ML_ROOT / "artifacts"
REPORT_DIR = ML_ROOT / "reports"
TAXONOMY_PATH = ML_ROOT / "taxonomy.yaml"
SOURCES_PATH = ML_ROOT.parent / "configs" / "sources.yaml"

DEFAULT_MONGO_URI = "mongodb://127.0.0.1:27017/news"

# Title+summary is the only field populated across every source; content is
# present for a minority and correlates with the source, not the topic. Phase 2
# therefore builds the corpus on title+summary and treats full text as a bonus
# variant. See the decision log entry for 2026-08-23.
DEFAULT_TEXT_VARIANT = "title_summary"
TEXT_VARIANTS = ("title", "title_summary", "title_summary_content")


def mongo_uri() -> str:
    """Connection string for the bronze corpus. Read-only by convention."""
    return os.environ.get("NEWS_ML_MONGO_URI", DEFAULT_MONGO_URI)
