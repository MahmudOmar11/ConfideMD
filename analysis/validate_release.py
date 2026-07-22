from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from confidemd.release import scan_release, verify_checksums


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public ConfideMD release tree.")
    parser.add_argument("--root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--checksums", type=Path)
    args = parser.parse_args()
    checksum_path = args.checksums or args.root / "SHA256SUMS"

    findings = scan_release(args.root)
    if not checksum_path.exists():
        findings.append(f"missing checksum manifest: {checksum_path.name}")
    else:
        findings.extend(verify_checksums(args.root, checksum_path))
    if findings:
        print("release validation: FAIL")
        for finding in sorted(set(findings)):
            print(f"- {finding}")
        raise SystemExit(1)
    print("release validation: PASS")


if __name__ == "__main__":
    main()
