"""
Part 1 — Full pipeline runner (robust)

Runs the following scripts IN ORDER:

1) calculate_phasor.py
2) phasor_plots.py
3) mask_analysis.py
4) mask_flim_parameters.py
5) plot_final_mask.py

This runner is robust to scripts that:
- define main()
- OR define a driver like process_patient(...)
- OR only execute inside `if __name__ == "__main__":` (fallback via runpy)

Usage:
    python run_all.py
"""

from __future__ import annotations

import sys
import runpy
import importlib
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent


def _banner(msg: str):
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)


def run_module(module_name: str):
    """
    Try to run a step in this order:
      1) module.main()
      2) module.process_patient(PATIENT_DIR, OUT_ROOT)  (if those exist)
      3) fallback: run the script file via runpy.run_path (as __main__)
    """
    py_path = THIS_DIR / f"{module_name}.py"
    if not py_path.exists():
        raise FileNotFoundError(f"Missing script: {py_path}")

    # Import module (so we can call functions if they exist)
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))

    module = importlib.import_module(module_name)

    # 1) main()
    if hasattr(module, "main") and callable(getattr(module, "main")):
        module.main()
        return

    # 2) process_patient(PATIENT_DIR, OUT_ROOT)
    # (matches your calculate_phasor.py)
    if hasattr(module, "process_patient") and callable(getattr(module, "process_patient")):
        # Try to reuse module-level constants if they exist
        patient_dir = getattr(module, "PATIENT_DIR", None)
        out_root = getattr(module, "OUT_ROOT", None)

        if patient_dir is None or out_root is None:
            raise RuntimeError(
                f"{module_name}.py has process_patient() but does not expose PATIENT_DIR and OUT_ROOT.\n"
                f"Fix: define PATIENT_DIR and OUT_ROOT at module level (not only inside __main__),\n"
                f"or add a main()."
            )

        module.process_patient(Path(patient_dir), Path(out_root))
        return

    # 3) fallback: run as script (executes its __main__ section)
    runpy.run_path(str(py_path), run_name="__main__")


def main():
    _banner("🚀 STARTING Part 1 — Phasor computation + masks + FLIM parameters")

    steps = [
        ("1) Phasor computation and calibration", "calculate_phasor"),
        ("2) Median filtering + QC phasor plots", "phasor_plots"),
        ("3) Mask mosaic assembly + instance segmentation", "mask_analysis"),
        ("4) Per-instance FLIM parameters + τ filtering", "mask_flim_parameters"),
        ("5) QC visualization of final instance masks", "plot_final_mask"),
    ]

    for title, mod in steps:
        _banner(f"▶ RUNNING: {title}  ({mod}.py)")
        run_module(mod)
        print(f"✔ COMPLETED: {title}")

    _banner("✅ PART 1 PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()