"""Reproduce both figures from the measured sweep in this directory."""

from pathlib import Path

from langmuir_probe.analysis import analyze, save_plot
from langmuir_probe.data import read_csv


def main() -> None:
    examples = Path(__file__).resolve().parent
    output = examples.parent / "docs" / "img"
    output.mkdir(parents=True, exist_ok=True)
    points, _ = read_csv(examples / "voltage_current_data_23.csv")
    result = analyze(points)
    save_plot(output / "sweep-23-iv.png", points, result)
    save_plot(output / "sweep-23-semilog.png", points, result, semilog=True)
    print(result.summary())


if __name__ == "__main__":
    main()
