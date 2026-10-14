"""Repo-relative paths. Import this before transformers/datasets so the HF cache env is set in time."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "out"
CACHE = ROOT / "cache"

os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "hf" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE / "hf" / "datasets"))
