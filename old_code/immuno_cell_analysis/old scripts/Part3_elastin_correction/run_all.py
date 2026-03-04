"""
run_all.py
==========

Run-all driver for Part 3 — Elastin-based correction + QC plots.

Execution order:
1) elastin_phase_mod_correction.py
2) plot_elastin_centroids_only.py
3) corrected_phasor_plot.py

This script:
- Imports each module
- Verifies that `main()` exists
- Stops immediately if any step fails

Usage:
    python run_all.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
SCRIPTS = [
    ("elastin_phase_mod_correction", "Elastin phase/mod correction"),
    ("plot_elastin_centroids_only", "Elastin centroids QC"),
    ("corrected_phasor_plot", "Corrected vs Final phasor plots"),
]


# ============================================================
# Helpers
# ============================================================
def run_step(module_name: str, title: str):
    print("\n" + "=" * 90)
    print(f"▶ RUNNING: {title}")
    print("=" * 90)

    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        raise RuntimeError(f"Failed to import '{module_name}': {e}")

    if not hasattr(module, "main"):
        raise RuntimeError(f"Module '{module_name}.py' does NOT define main()")

    try:
        module.main()
    except Exception as e:
        raise RuntimeError(f"Error while running {module_name}.main(): {e}")

    print(f"✅ COMPLETED: {module_name}")


# ============================================================
# Main
# ============================================================
def main():
    print("\n🚀 STARTING Part 3 — Elastin correction + QC plots\n")

    # Ensure local folder is on PYTHONPATH
    here = Path(__file__).parent.resolve()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    for module_name, title in SCRIPTS:
        run_step(module_name, title)

    print("\n🎉 Part 3 COMPLETE — all steps finished successfully")


if __name__ == "__main__":
    main()