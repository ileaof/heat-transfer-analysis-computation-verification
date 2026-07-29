"""Example -- Module 3 (GUI): graphical flat-plate heat-transfer calculator.

Opens an entry-field calculator window.  Type the free-stream velocity, plate
length, and wall / free-stream temperatures, pick a fluid, and press
**Calculate** to see the full boundary-layer and heat-transfer report, the
numerical-vs-correlation comparison, and the streamwise distribution plots.
Use **Export CSV** / **Export .dat** to save the distributions.

Run::

    python examples/example_module3_calculator_gui.py

Requires an interactive Matplotlib backend (the default on a desktop).  The
non-interactive batch equivalent is ``example_module3_calculator.py``.
"""

from __future__ import annotations

from _common import OUTPUT_DIR

from blayerlab.calculator_gui import CalculatorGUI


def main() -> None:
    print("Launching the graphical Flat-Plate Heat-Transfer Calculator ...")
    print(f"Exports will be written to: {OUTPUT_DIR}")
    print("Close the window to exit.")
    # Route the export buttons to the examples/outputs directory.
    CalculatorGUI(output_dir=OUTPUT_DIR).launch()


if __name__ == "__main__":
    main()
