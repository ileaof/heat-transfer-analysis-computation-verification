"""Example -- Module 5: Engineering correlation verification.

Runs the full verification suite (momentum constants, thermal wall gradients,
Reynolds and Chilton-Colburn analogies), prints the report, and exports the
per-item error table to CSV.

Run::

    python examples/example_module5_verification.py
"""

from __future__ import annotations

from _common import out

from blayerlab import export_csv
from blayerlab.verification import VerificationSuite


def main() -> None:
    suite = VerificationSuite(re_x=1.0e5)
    print(suite.report())

    # Flatten all items into a single error table.
    groups = suite.run_all()
    names, nums, refs, kinds, abs_e, rel_e = [], [], [], [], [], []
    for group, items in groups.items():
        for it in items:
            names.append(f"{group}:{it.quantity}")
            nums.append(it.numerical)
            refs.append(it.reference)
            kinds.append(0.0 if it.reference_kind == "analytical" else 1.0)
            abs_e.append(it.abs_error)
            rel_e.append(it.rel_error)

    csv_path = export_csv(
        out("module5", "verification_errors.csv"),
        {
            "numerical": nums,
            "reference": refs,
            "reference_is_correlation": kinds,
            "absolute_error": abs_e,
            "relative_error": rel_e,
        },
        header_comment="Verification error table (row order matches report). "
        "reference_is_correlation: 0=analytical, 1=correlation.",
    )
    print("\nwrote", csv_path)


if __name__ == "__main__":
    main()
