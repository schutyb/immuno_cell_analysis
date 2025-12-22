#!/usr/bin/env python3
"""
preflight_check.py

Run this BEFORE run_all.py to ensure:
- Each step script exists
- It can be imported
- It defines a callable main()

Optional:
- Check presence of key config variables per script (helps catch misconfigured paths early)

Usage:
  python preflight_check.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import importlib.util
import traceback
from typing import Optional


# ============================================================
# CONFIG
# ============================================================
# Put this file in the same folder as the step scripts (recommended).
HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "calculate_phasor.py",
    "phasor_plots.py",          # (or phasor_plot.py if that’s your real name)
    "mask_analysis.py",
    "mask_flim_parameters.py",
    "plot_final_mask.py",       # (or plot_final.py if that’s your real name)
]

# If your filenames differ, edit them above.

# Optional: check key config variables exist in each script module
CHECK_CONFIG_VARS = False

# Per script: list of variables that should exist at module-level.
# Only checks presence (not validity of paths).
REQUIRED_VARS = {
    "calculate_phasor.py": ["PATIENT_DIR", "OUT_ROOT"],
    "phasor_plots.py":     [],  # add if you want
    "mask_analysis.py":    [],  # add if you want
    "mask_flim_parameters.py": [],
    "plot_final_mask.py": ["TIFF_PATH"],  # since you pass one TIFF path
}


# ============================================================
# Helpers
# ============================================================
def import_module_from_path(py_path: Path):
    """
    Import a module from an arbitrary .py file path without needing it in PYTHONPATH.
    """
    if not py_path.exists():
        raise FileNotFoundError(py_path)

    mod_name = py_path.stem  # e.g. calculate_phasor
    spec = importlib.util.spec_from_file_location(mod_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def check_main(module, file_name: str) -> Optional[str]:
    """
    Return None if OK, otherwise an error string.
    """
    if not hasattr(module, "main"):
        return f"Module '{file_name}' does NOT define main()."
    if not callable(getattr(module, "main")):
        return f"Module '{file_name}' has main but it is NOT callable."
    return None


def check_required_vars(module, file_name: str) -> list[str]:
    """
    Return list of missing variable names for that module.
    """
    missing = []
    req = REQUIRED_VARS.get(file_name, [])
    for var in req:
        if not hasattr(module, var):
            missing.append(var)
    return missing


# ============================================================
# Main
# ============================================================
def main() -> int:
    print("🧪 Preflight check — validating scripts before run_all\n")

    failures = 0
    for fname in SCRIPTS:
        py_path = HERE / fname

        print("=" * 90)
        print(f"▶ Checking: {fname}")
        print(f"  Path: {py_path}")

        # 1) exists
        if not py_path.exists():
            print(f"  ❌ MISSING FILE")
            failures += 1
            continue

        # 2) import
        try:
            module = import_module_from_path(py_path)
            print(f"  ✅ Import OK: module={module.__name__}")
        except Exception as e:
            print(f"  ❌ Import FAILED: {e}")
            print("  --- traceback ---")
            traceback.print_exc()
            failures += 1
            continue

        # 3) main()
        err = check_main(module, fname)
        if err:
            print(f"  ❌ {err}")
            failures += 1
        else:
            print(f"  ✅ main() exists")

        # 4) optional config vars
        if CHECK_CONFIG_VARS:
            missing_vars = check_required_vars(module, fname)
            if missing_vars:
                print(f"  ❌ Missing config variables: {missing_vars}")
                failures += 1
            else:
                print(f"  ✅ Required config vars present")

    print("=" * 90)
    if failures == 0:
        print("\n✅ Preflight PASSED — safe to run run_all.py")
        return 0

    print(f"\n❌ Preflight FAILED — {failures} issue(s) found. Fix them before running run_all.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())