"""Export helpers: comma-separated values (CSV) and Tecplot ASCII.

Every solver in the Boundary Layer Laboratory can dump its results to disk so
that figures can be regenerated or post-processed in external tools.  Two
formats are supported, using only the Python standard library and NumPy:

* **CSV** -- universally readable (spreadsheets, pandas, gnuplot).
* **Tecplot ASCII (POINT format)** -- the de-facto standard in the CFD
  community; directly loadable by Tecplot, ParaView and VisIt.
"""

from __future__ import annotations

import csv
import os
from typing import Mapping, Sequence

import numpy as np


def _as_columns(data: Mapping[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    """Convert a ``{name: array}`` mapping into aligned column arrays.

    Raises
    ------
    ValueError
        If the columns do not all share the same length.
    """
    names = list(data.keys())
    columns = [np.asarray(data[name], dtype=float).ravel() for name in names]
    lengths = {col.size for col in columns}
    if len(lengths) != 1:
        raise ValueError(
            "All columns must have equal length; got lengths "
            f"{ {n: c.size for n, c in zip(names, columns)} }."
        )
    matrix = np.column_stack(columns) if columns else np.empty((0, 0))
    return names, matrix


def export_csv(
    path: str,
    data: Mapping[str, Sequence[float]],
    header_comment: str | None = None,
) -> str:
    """Write column data to a CSV file.

    Parameters
    ----------
    path : str
        Output file path.  Parent directories are created if needed.
    data : mapping of str to sequence
        Ordered mapping ``{column_name: values}``.  All value arrays must have
        equal length.
    header_comment : str, optional
        Free-form text written as leading ``#`` comment lines.

    Returns
    -------
    str
        The path written (for convenient chaining/logging).
    """
    names, matrix = _as_columns(data)
    _ensure_parent(path)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        if header_comment:
            for line in header_comment.splitlines():
                fh.write(f"# {line}\n")
        writer = csv.writer(fh)
        writer.writerow(names)
        for row in matrix:
            writer.writerow([f"{value:.10g}" for value in row])
    return path


def export_tecplot(
    path: str,
    data: Mapping[str, Sequence[float]],
    title: str = "Boundary Layer Laboratory",
    zone_name: str = "ZONE 001",
) -> str:
    """Write column data as a Tecplot ASCII (POINT format) file.

    Parameters
    ----------
    path : str
        Output file path.
    data : mapping of str to sequence
        Ordered mapping ``{variable_name: values}``.
    title : str, optional
        Dataset title written to the ``TITLE`` record.
    zone_name : str, optional
        Name of the single ordered zone.

    Returns
    -------
    str
        The path written.

    Notes
    -----
    The emitted file has the structure::

        TITLE = "..."
        VARIABLES = "eta", "f", ...
        ZONE T="...", I=<npoints>, F=POINT
        <row 0>
        <row 1>
        ...
    """
    names, matrix = _as_columns(data)
    npoints = matrix.shape[0]
    _ensure_parent(path)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f'TITLE = "{title}"\n')
        fh.write("VARIABLES = " + ", ".join(f'"{name}"' for name in names) + "\n")
        fh.write(f'ZONE T="{zone_name}", I={npoints}, F=POINT\n')
        for row in matrix:
            fh.write(" ".join(f"{value: .10e}" for value in row) + "\n")
    return path


def _ensure_parent(path: str) -> None:
    """Create the parent directory of ``path`` if it does not yet exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


__all__ = ["export_csv", "export_tecplot"]
