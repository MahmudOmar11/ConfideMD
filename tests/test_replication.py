from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[1]


class ReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run([sys.executable, str(ROOT / "analysis" / "validate_package.py")], check=True)

    def test_primary_synthetic_denominators(self) -> None:
        frame = pd.read_csv(ROOT / "results" / "endpoint_rates.csv")
        subset = frame[(frame.dataset == "synthetic") & (frame.endpoint == "primary_unambiguous")]
        self.assertEqual(dict(zip(subset.condition, subset.n)), {"request_off": 1482, "request_on": 1483})

    def test_patient_authored_complete(self) -> None:
        frame = pd.read_csv(ROOT / "data" / "derived" / "patient_authored_derived_scored.csv")
        self.assertEqual(len(frame), 3150)

    def test_primary_combined_disclosure_count(self) -> None:
        frame = pd.read_csv(ROOT / "results" / "endpoint_rates.csv")
        subset = frame[
            (frame.endpoint == "primary_unambiguous")
            & (frame.condition == "request_on")
        ]
        self.assertEqual((int(subset.numerator.sum()), int(subset.n.sum())), (722, 2113))

    def test_primary_request_effects(self) -> None:
        frame = pd.read_csv(ROOT / "results" / "condition_effects.csv")
        synthetic = frame[
            (frame.dataset == "synthetic")
            & (frame.endpoint == "primary_unambiguous")
        ].iloc[0]
        patient = frame[
            (frame.dataset == "patient_authored")
            & (frame.endpoint == "primary_unambiguous")
        ].iloc[0]
        self.assertAlmostEqual(float(synthetic.risk_difference), -0.03443619176232275)
        self.assertAlmostEqual(float(patient.risk_difference), -0.3380952380952381)

    def test_harmful_omission_counts(self) -> None:
        frame = pd.read_csv(ROOT / "results" / "omission_rates.csv")
        off = frame[frame.condition == "request_off"].iloc[0]
        on = frame[frame.condition == "request_on"].iloc[0]
        self.assertEqual((int(off.numerator), int(off.n)), (15, 280))
        self.assertEqual((int(on.numerator), int(on.n)), (21, 279))

    def test_gate_primary_llama_tradeoff(self) -> None:
        frame = pd.read_csv(ROOT / "results" / "gate_paired_vs_naive.csv")
        row = frame[
            (frame.model == "meta-llama-llama-3.3-70b-instruct")
            & (frame.method == "gate")
            & (frame.endpoint == "primary_unambiguous")
        ].iloc[0]
        self.assertAlmostEqual(float(row.risk_difference), -0.20518867924528303)
        self.assertEqual((int(row.improved), int(row.worsened)), (88, 1))

    def test_gate_deepseek_omission_tradeoff(self) -> None:
        frame = pd.read_csv(ROOT / "results" / "gate_paired_vs_naive.csv")
        row = frame[
            (frame.model == "deepseek-deepseek-v4-flash")
            & (frame.method == "gate")
            & (frame.endpoint == "required_documents")
        ].iloc[0]
        self.assertAlmostEqual(float(row.naive), 0.0625)
        self.assertAlmostEqual(float(row.method_rate), 0.6625)

    def test_reference_outputs_match_fresh_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "analysis" / "run_analysis.py"),
                    "--output-dir",
                    directory,
                ],
                check=True,
            )
            for expected_path in sorted((ROOT / "results").glob("*.csv")):
                observed_path = Path(directory) / expected_path.name
                self.assertTrue(observed_path.exists(), observed_path.name)
                expected = pd.read_csv(expected_path)
                observed = pd.read_csv(observed_path)
                assert_frame_equal(observed, expected, check_exact=True)


if __name__ == "__main__":
    unittest.main()
