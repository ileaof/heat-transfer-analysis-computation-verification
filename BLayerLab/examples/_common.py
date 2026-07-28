"""Shared helpers for the example scripts.

Puts the repository root on ``sys.path`` (so the examples run without an
install) and provides a single output directory for figures and data files.
"""

from __future__ import annotations

import os
import sys

# Make ``import blayerlab`` work when running an example directly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#: Directory where all example outputs (figures, CSV, Tecplot) are written.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(*parts: str) -> str:
    """Return a path inside :data:`OUTPUT_DIR` (creating sub-folders)."""
    path = os.path.join(OUTPUT_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
