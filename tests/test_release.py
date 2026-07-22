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
