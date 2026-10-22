from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from confidemd.release import scan_release, verify_checksums


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_release_tree_has_no_privacy_or_meta_findings(self) -> None:
        self.assertEqual(scan_release(ROOT), [])

    def test_release_scan_rejects_a_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe = "api" + "_key = 'not-a-real-release-secret'"
            (root / "unsafe.txt").write_text(unsafe)
            findings = scan_release(root)
            self.assertTrue(any("credential" in finding for finding in findings))

    def test_git_release_scan_ignores_untracked_environment_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "safe.txt").write_text("tracked")
            (root / ".gitignore").write_text(".venv/\n")
            environment = root / ".venv"
            environment.mkdir()
            (environment / "generated.txt").write_text("api" + "_key = 'not-a-release-secret'")
            subprocess.run(["git", "-C", str(root), "add", "safe.txt", ".gitignore"], check=True)
            self.assertEqual(scan_release(root), [])

    def test_release_scan_rejects_private_columns_in_any_data_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "derived").mkdir(parents=True)
            (root / "data" / "derived" / "extra.csv").write_text("case_id,quote\ncase-001,text\n")
            findings = scan_release(root)
            self.assertTrue(any("private columns in data/derived/extra.csv" in finding for finding in findings))

    def test_release_scan_rejects_patient_requests_with_more_than_the_request(self) -> None:
        source = (ROOT / "data" / "withholding_requests.csv").read_text().splitlines()
        header, rows = source[0], source[1:]
        patient = next(row for row in rows if row.startswith("patient_authored,"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            requests = root / "data" / "withholding_requests.csv"
            requests.write_text("\n".join([header, patient]) + "\n")
            self.assertEqual(scan_release(root), [])
            leaked = patient.replace(",final line,", ",line 2 of 3,")
            requests.write_text("\n".join([header, leaked]) + "\n")
            self.assertTrue(any("carries more than the request" in finding for finding in scan_release(root)))
            renamed = patient.replace("real_case_", "real-case-", 1)
            requests.write_text("\n".join([header, renamed]) + "\n")
            self.assertTrue(any("carries more than the request" in finding for finding in scan_release(root)))

    def test_checksum_verification_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("frozen")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest = root / "SHA256SUMS"
            manifest.write_text(f"{digest}  artifact.txt\n")
            self.assertEqual(verify_checksums(root, manifest), [])
            artifact.write_text("changed")
            self.assertTrue(any("checksum mismatch" in finding for finding in verify_checksums(root, manifest)))

    def test_release_cli_accepts_a_clean_frozen_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("frozen")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest = root / "SHA256SUMS"
            manifest.write_text(f"{digest}  artifact.txt\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "analysis" / "validate_release.py"),
                    "--root",
                    str(root),
                    "--checksums",
                    str(manifest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("release validation: PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
