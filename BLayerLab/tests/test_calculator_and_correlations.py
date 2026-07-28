"""Tests for Module 3 (calculator), the correlations and I/O helpers."""

import numpy as np
import pytest

from blayerlab import correlations as corr
from blayerlab import export_csv, export_tecplot, get_fluid
from blayerlab.calculator import FlatPlateCalculator


@pytest.fixture(scope="module")
def result():
    return FlatPlateCalculator().compute(
        u_inf=2.0, fluid=get_fluid("air"), length=0.5, t_wall=350.0, t_inf=300.0
    )


def test_reynolds_and_prandtl(result):
    fluid = get_fluid("air")
    assert result.Re_L == pytest.approx(2.0 * 0.5 / fluid.nu, rel=1e-12)
    assert result.Pr == pytest.approx(fluid.Pr, rel=1e-12)


def test_nusselt_matches_correlation(result):
    """Numerical Nu_x agrees with Churchill-Ozoe within 2%."""
    ref = corr.nu_local_churchill_ozoe(result.Re_L, result.Pr)
    assert result.nu_local_L == pytest.approx(ref, rel=0.02)


def test_average_is_twice_local(result):
    """For laminar flow the average Nu and h are twice the trailing-edge value."""
    assert result.nu_avg == pytest.approx(2.0 * result.nu_local_L, rel=1e-6)
    assert result.h_avg == pytest.approx(2.0 * result.h_local_L, rel=1e-6)
    assert result.cf_avg == pytest.approx(2.0 * result.cf_local_L, rel=1e-6)


def test_heat_flux_definition(result):
    """q'' = h (T_w - T_inf)."""
    dT = result.t_wall - result.t_inf
    assert result.q_local_L == pytest.approx(result.h_local_L * dT, rel=1e-12)


def test_thermal_thicker_than_momentum_for_air(result):
    """For Pr < 1 (air) the thermal layer is thicker than the momentum layer."""
    assert result.delta_t_L > result.delta_L


def test_reynolds_analogy():
    """St = C_f/2 at Pr = 1 (Reynolds analogy)."""
    Re = 1e5
    assert corr.reynolds_analogy_st(Re) == pytest.approx(corr.cf_local(Re) / 2.0)


def test_chilton_colburn():
    """St Pr^{2/3} = C_f/2 (Colburn j-factor)."""
    Re, Pr = 1e5, 7.0
    st = corr.chilton_colburn_st(Re, Pr)
    assert st * Pr ** (2.0 / 3.0) == pytest.approx(corr.cf_local(Re) / 2.0, rel=1e-12)


def test_export_roundtrip(tmp_path, result):
    """CSV and Tecplot exports create non-empty files with the right header."""
    data = result.to_distribution_dict()
    csv_path = export_csv(str(tmp_path / "d.csv"), data)
    tec_path = export_tecplot(str(tmp_path / "d.dat"), data, title="T")

    with open(csv_path) as fh:
        header = fh.readline()
    assert "x [m]" in header

    with open(tec_path) as fh:
        content = fh.read()
    assert content.startswith('TITLE = "T"')
    assert "VARIABLES" in content and "ZONE" in content


def test_fluid_database_prandtl_span():
    """The database spans a wide Prandtl-number range (liquid metal to oil)."""
    prs = [get_fluid(f).Pr for f in ("mercury", "air", "water", "engine_oil")]
    assert min(prs) < 0.1 < 1.0 < max(prs)
    assert get_fluid("mercury").Pr < get_fluid("air").Pr < get_fluid("engine_oil").Pr
