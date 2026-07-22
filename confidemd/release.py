from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".gitignore", ".json", ".md", ".py", ".toml", ".txt", ".lock"}
FORBIDDEN_DIRECTORIES = {
    ".ai" + "der",
    ".claude",
    ".codex",
    ".continue",
    ".cursor",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
FORBIDDEN_FILENAMES = {".DS_Store", ".env"}
PRIVATE_COLUMNS = {
    "email",
    "name",
    "patient_text",
    "phone",
    "quote",
    "secret",
    "source",
    "source_row",
    "transcript",
    "withhold_line",
}

CREDENTIAL_PATTERNS = (
    re.compile(r"(?<![A-Za-z])sk-(?:live-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
    re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(?:api[_-]?key|password|bearer[_-]?token)\s*[:=]\s*['\"][^'\"]{12,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
PERSONAL_PATH = re.compile(r"/(?:Users|home)/[^/\s]+/")
_META_TERMS = (
    "created by " + "codex",
    "created by " + "claude",
    "claude " + "code",
    "cursor " + "agent",
    "co" + "pilot",
    "ai" + "der",
    "chain of " + "thought",
    "second " + "brain",
    "internal " + "project",
    "scratch" + "pad",
    "check" + "point",
    "TO" + "DO",
    "FIX" + "ME",
)
META_TEXT = re.compile(r"(?i)\b(?:" + "|".join(re.escape(term) for term in _META_TERMS) + r")\b")


def _release_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if ".git" in relative_parts:
            continue
        if path.is_file() and path.name != "SHA256SUMS":
            files.append(path)
    return sorted(files)


def scan_release(root: str | Path) -> list[str]:
    root = Path(root)
    findings: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            findings.append(f"symlink: {relative}")
        if path.is_dir() and path.name in FORBIDDEN_DIRECTORIES:
            findings.append(f"forbidden directory: {relative}")
        if path.is_file() and (
            path.name in FORBIDDEN_FILENAMES
            or path.suffix in {".bak", ".key", ".pem", ".pyc", ".pyo", ".tmp"}
            or path.name.lower().startswith(("credential", "secret"))
        ):
            findings.append(f"forbidden file: {relative}")

    for path in _release_files(root):
        relative = path.relative_to(root)
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore", "LICENSE"}:
            findings.append(f"unexpected binary or file type: {relative}")
            continue
        text = path.read_text(errors="replace")
        for pattern in CREDENTIAL_PATTERNS:
            if pattern.search(text):
                findings.append(f"credential pattern: {relative}")
                break
        if PERSONAL_PATH.search(text):
            findings.append(f"personal path: {relative}")
        if path.name != ".gitignore" and META_TEXT.search(text):
            findings.append(f"internal or meta text: {relative}")

    patient_path = root / "data" / "derived" / "patient_authored_derived_scored.csv"
    if patient_path.exists():
        with patient_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            forbidden = sorted(columns & PRIVATE_COLUMNS)
            if forbidden:
                findings.append(f"private patient columns: {', '.join(forbidden)}")
            for row_number, row in enumerate(reader, start=2):
                if not re.fullmatch(r"real_case_\d{3}", row.get("case_id", "")):
                    findings.append(f"nonpseudonymous patient case_id at row {row_number}")
                    break
    return sorted(set(findings))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_lines(root: str | Path) -> list[str]:
    root = Path(root)
    return [f"{file_sha256(path)}  {path.relative_to(root).as_posix()}" for path in _release_files(root)]


def verify_checksums(root: str | Path, manifest: str | Path) -> list[str]:
    root = Path(root)
    manifest = Path(manifest)
    findings: list[str] = []
    expected: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError:
            findings.append(f"malformed checksum line {line_number}")
            continue
        expected[relative] = digest
    actual_paths = {path.relative_to(root).as_posix(): path for path in _release_files(root)}
    for relative, digest in expected.items():
        path = actual_paths.get(relative)
        if path is None:
            findings.append(f"missing file: {relative}")
        elif file_sha256(path) != digest:
            findings.append(f"checksum mismatch: {relative}")
    for relative in sorted(set(actual_paths) - set(expected)):
        findings.append(f"unlisted file: {relative}")
    return findings
