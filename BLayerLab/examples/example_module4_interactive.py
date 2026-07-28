"""Example -- Module 4: Interactive Boundary Layer Laboratory.

Opens the interactive Matplotlib window.  Drag the sliders (free-stream
velocity, plate length, wall and free-stream temperatures) and pick a fluid to
watch every boundary-layer quantity update in real time.

Run::

    python examples/example_module4_interactive.py

Requires an interactive Matplotlib backend (the default on a desktop).  This
example is skipped by the batch ``run_all.py`` because it blocks on a window.
"""

from __future__ import annotations

from _common import out  # noqa: F401  (ensures blayerlab is importable)

from blayerlab.interactive import InteractiveLab


def main() -> None:
    print("Launching the Interactive Boundary Layer Laboratory ...")
    print("Close the window to exit.")
    InteractiveLab().launch()


if __name__ == "__main__":
    main()
