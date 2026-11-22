"""Read legacy and metadata-bearing sweep CSVs."""

import csv
import json
from pathlib import Path

Point = tuple[float, float]
COLUMNS = ("Voltage (V)", "Current (A)")


def read_csv(path: str | Path) -> tuple[list[Point], dict]:
    metadata = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        line = source.readline()
        while line.startswith("#"):
            key, separator, value = line[1:].partition(":")
            if separator:
                metadata[key.strip()] = json.loads(value)
            line = source.readline()
        if tuple(next(csv.reader([line]), [])) != COLUMNS:
            raise ValueError("CSV must have Voltage (V) and Current (A) columns.")
        points = [
            (float(v), float(i)) for v, i in (row for row in csv.reader(source) if row)
        ]
    return points, metadata


def write_csv(path: str | Path, points: list[Point], metadata: dict) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as target:
        for key, value in metadata.items():
            target.write(f"# {key}: {json.dumps(value)}\n")
        writer = csv.writer(target)
        writer.writerow(COLUMNS)
        writer.writerows(points)
