"""Tests for the Module 3 GUI (headless, Agg backend from conftest)."""

import os

import matplotlib.pyplot as plt
import pytest

from blayerlab.calculator_gui import CalculatorGUI


@pytest.fixture
def gui(tmp_path):
    g = CalculatorGUI(output_dir=str(tmp_path))
    yield g
    plt.close(g.fig)


def test_builds_and_computes_defaults(gui):
    """Constructing the GUI computes the default operating point."""
    assert gui._last is not None
    assert gui._last.Re_L > 0
    assert gui._last.nu_local_L > 0


def test_recompute_on_input_change(gui):
    """Editing a field and calculating updates the stored result."""
    gui.tb_u.set_val("4.0")
    gui.calculate()
    assert gui._last.u_inf == pytest.approx(4.0)


def test_invalid_input_is_ignored(gui):
    """A non-numeric entry leaves the previous valid result intact."""
    gui.tb_u.set_val("5.0")
    gui.calculate()
    good = gui._last
    gui.tb_u.set_val("not-a-number")
    gui.calculate()
    assert gui._last is good  # unchanged


def test_fluid_switch(gui):
    """Selecting a different fluid recomputes with that fluid's properties."""
    gui._on_fluid("Water")
    assert gui._last.fluid.name == "Water"
    assert gui._last.Pr == pytest.approx(gui.fluids["water"].Pr, rel=1e-9)


def test_exports_write_files(gui, tmp_path):
    """Export buttons write CSV and Tecplot files to the output directory."""
    gui._on_export_csv(None)
    gui._on_export_tecplot(None)
    files = os.listdir(tmp_path)
    assert any(f.endswith(".csv") for f in files)
    assert any(f.endswith(".dat") for f in files)
