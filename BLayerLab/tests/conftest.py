"""pytest configuration: make ``blayerlab`` importable without installation
and force a headless Matplotlib backend so tests never open a window.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
