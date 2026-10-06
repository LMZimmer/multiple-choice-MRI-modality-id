#!/usr/bin/env python3
"""Read the modality selections from a filled-in review PDF and write them to a CSV file.

Takes one or more PDFs created by make_review_pdf.py and writes one row per image series
with the columns path, series_uid, modality and notes.

Usage:
    python read_review_pdf.py modality_review.pdf -o modality_mapping.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from pypdf import PdfReader

# Form field names are "<prefix>_<page number>" and the path is percent-encoded, as written by make_review_pdf.py
FIELD_PATH, FIELD_UID, FIELD_MODALITY, FIELD_NOTES = "path", "uid", "modality", "notes"


def text_value(field) -> str:
    value = field.get("/V") if field is not None else None
    return "" if value is None else str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def radio_value(field) -> str:
    """Selected option of a radio group, or "" if nothing is selected."""
    if field is None:
        return ""
    value = field.get("/V")
    if value in (None, "", "/Off"):
        # Some viewers only update the state of the buttons and not the value of the group
        for kid in field.get("/Kids", []):
            state = kid.get_object().get("/AS")
            if state not in (None, "/Off"):
                value = state
                break
        else:
            return ""
    return str(value).lstrip("/")


def read_pdf(path: Path) -> list[dict]:
    fields = PdfReader(path).get_fields() or {}
    pages = {}
    for name, field in fields.items():
        prefix, _, number = name.rpartition("_")
        if prefix in (FIELD_PATH, FIELD_UID, FIELD_MODALITY, FIELD_NOTES) and number.isdigit():
            pages.setdefault(int(number), {})[prefix] = field
    rows = []
    for number, page in sorted(pages.items()):
        if FIELD_PATH not in page:
            continue
        rows.append({
            "path": unquote(text_value(page[FIELD_PATH])),
            "series_uid": text_value(page.get(FIELD_UID)),
            "modality": radio_value(page.get(FIELD_MODALITY)),
            "notes": text_value(page.get(FIELD_NOTES)),
            "page": f"{path.name} page {number}",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pdf", type=Path, nargs="+", help="filled-in PDF(s) created by make_review_pdf.py")
    parser.add_argument("-o", "--output", type=Path, default=Path("modality_mapping.csv"), help="output CSV (default: %(default)s)")
    args = parser.parse_args()

    rows = []
    for path in args.pdf:
        found = read_pdf(path)
        if not found:
            sys.exit(f"{path}: no review form fields found. Was the PDF saved with its form fields "
                     "(not printed or flattened)?")
        rows.extend(found)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["path", "series_uid", "modality", "notes"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["modality"] or "(not selected)" for row in rows)
    print(f"Wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    for modality, count in counts.most_common():
        print(f"  {modality:<15}{count}", file=sys.stderr)
    unlabeled = [row for row in rows if not row["modality"]]
    if unlabeled:
        print(f"\n{len(unlabeled)} image(s) without a selected modality:", file=sys.stderr)
        for row in unlabeled:
            print(f"  {row['page']}: {row['path']}", file=sys.stderr)


if __name__ == "__main__":
    main()
