"""Download the IR presentation PDFs listed in docs/sources.csv into data/raw/.

Run with:  uv run python scripts/download_sources.py [--dry-run]

- Files that already exist are skipped, so the script can be re-run safely.
- Each download is checked to be a real PDF (by its first bytes).
- SHA-256 hashes are written to data/raw/checksums.csv so later runs can detect
  that a company silently replaced a file.
- Requests are spaced out to be polite to the IR sites.
"""

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "docs" / "sources.csv"
RAW_DIR = ROOT / "data" / "raw"
CHECKSUMS = RAW_DIR / "checksums.csv"

# Identify honestly as a script. If a site refuses it (e.g. 403), don't work around
# the block: the file is reported as failed and gets downloaded by hand in a browser.
USER_AGENT = "multimodal-rag-research/0.1 (personal research; https://github.com/Rimcat-JA)"
DELAY_SECONDS = 2.0
TIMEOUT_SECONDS = 60


def target_path(row: dict[str, str]) -> Path:
    # e.g. data/raw/toyota/fy2025_full.pdf
    return RAW_DIR / row["company_id"] / f"{row['period']}.pdf"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return resp.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    args = parser.parse_args()

    with SOURCES.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    checksums: dict[str, str] = {}
    if CHECKSUMS.exists():
        with CHECKSUMS.open(encoding="utf-8") as f:
            checksums = {r["path"]: r["sha256"] for r in csv.DictReader(f)}

    failures = []
    for i, row in enumerate(rows, 1):
        dest = target_path(row)
        rel = dest.relative_to(ROOT).as_posix()
        label = f"[{i}/{len(rows)}] {row['company_id']}/{row['period']}"

        if dest.exists():
            # Files saved by hand in a browser get their hash recorded here.
            if rel not in checksums:
                checksums[rel] = hashlib.sha256(dest.read_bytes()).hexdigest()
                print(f"{label}: exists (added by hand), checksum recorded")
            else:
                print(f"{label}: exists, skipped")
            continue
        if args.dry_run:
            print(f"{label}: would download {row['url']}")
            continue

        try:
            data = fetch(row["url"])
        except Exception as e:  # network errors, 404, timeouts
            print(f"{label}: FAILED ({e})")
            failures.append(label)
            continue

        if not data.startswith(b"%PDF"):
            print(f"{label}: FAILED (response is not a PDF)")
            failures.append(label)
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        checksums[rel] = hashlib.sha256(data).hexdigest()
        print(f"{label}: saved {len(data) / 1024**2:.1f} MB -> {rel}")
        time.sleep(DELAY_SECONDS)

    if not args.dry_run:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with CHECKSUMS.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "sha256"])
            writer.writerows(sorted(checksums.items()))

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
