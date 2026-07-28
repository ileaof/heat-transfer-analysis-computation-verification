"""Run every non-interactive example in one batch.

Uses the non-interactive ``Agg`` Matplotlib backend, so it produces all figures
and data files without opening any window.  Module 4 (the interactive lab) is
intentionally excluded because it blocks on a GUI window; run it separately with
``python examples/example_module4_interactive.py``.

Run::

    python examples/run_all.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window

from _common import OUTPUT_DIR  # noqa: E402

import example_module1_blasius as m1  # noqa: E402
import example_module2_thermal as m2  # noqa: E402
import example_module3_calculator as m3  # noqa: E402
import example_module5_verification as m5  # noqa: E402
import example_module6_parametric as m6  # noqa: E402


def main() -> None:
    for label, module in [
        ("MODULE 1  Blasius momentum boundary layer", m1),
        ("MODULE 2  Thermal boundary layer", m2),
        ("MODULE 3  Flat-plate calculator", m3),
        ("MODULE 5  Correlation verification", m5),
        ("MODULE 6  Parametric studies", m6),
    ]:
        print("\n" + "#" * 70)
        print("# " + label)
        print("#" * 70)
        module.main()

    print("\nAll outputs written under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
