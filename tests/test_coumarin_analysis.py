from __future__ import annotations

import numpy as np
import pandas as pd

from coumarin_analysis.analyze_coumarin import (
    differences_from_channels,
    modality_summary,
    parse_visit_folder,
)


def test_visit_folder_parser() -> None:
    assert parse_visit_folder("p427-v01") == ("p427", "visit01")
    assert parse_visit_folder("P449-v4") == ("p449", "visit04")
    assert parse_visit_folder("not-a-visit") is None


def test_coumarin_differences_and_modality_means() -> None:
    base = {
        "visit_folder": "p1-v01",
        "experiment": "coumarin",
        "pair_index": 1,
        "replicate_index": 1,
    }
    rows = [
        {
            **base,
            "patient": "p1",
            "visit": "visit01",
            "job_id": "sp1",
            "acquisition_type": "Sp",
            "channel": "green",
            "phase_correction_deg": -12.0,
        },
        {
            **base,
            "patient": "p1",
            "visit": "visit01",
            "job_id": "sp1",
            "acquisition_type": "Sp",
            "channel": "blue",
            "phase_correction_deg": -10.0,
        },
        {
            **base,
            "patient": "p2",
            "visit": "visit02",
            "job_id": "sp2",
            "acquisition_type": "Sp",
            "channel": "green",
            "phase_correction_deg": -11.0,
        },
        {
            **base,
            "patient": "p2",
            "visit": "visit02",
            "job_id": "sp2",
            "acquisition_type": "Sp",
            "channel": "blue",
            "phase_correction_deg": -8.0,
        },
        {
            **base,
            "patient": "p3",
            "visit": "visit01",
            "job_id": "a1a0",
            "acquisition_type": "A1_A0",
            "channel": "green",
            "phase_correction_deg": -9.0,
        },
        {
            **base,
            "patient": "p3",
            "visit": "visit01",
            "job_id": "a1a0",
            "acquisition_type": "A1_A0",
            "channel": "blue",
            "phase_correction_deg": -8.0,
        },
    ]
    channels = pd.DataFrame(rows)

    differences = differences_from_channels(channels)
    summary = modality_summary(channels, differences).set_index("acquisition_type")

    sp = differences[differences["acquisition_type"] == "Sp"]
    assert np.allclose(sp["green_minus_blue_correction_deg"], [-2.0, -3.0])
    assert np.isclose(summary.loc["Sp", "mean_green_minus_blue_deg"], -2.5)
    assert np.isclose(summary.loc["Sp", "recommended_green_offset_deg"], 2.5)
    assert np.isclose(summary.loc["A1_A0", "recommended_green_offset_deg"], 1.0)
