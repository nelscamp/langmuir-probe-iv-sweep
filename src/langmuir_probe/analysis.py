"""Estimate single-probe parameters from measured current versus voltage."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .data import Point, read_csv

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class Analysis:
    floating_potential: float
    plasma_potential: float
    ion_current: float
    electron_temperature: float
    temperature_uncertainty: float
    ion_slope: float
    ion_intercept: float
    fit_start: float
    fit_end: float
    log_slope: float
    log_intercept: float

    def summary(self) -> str:
        return (
            f"Floating potential: {self.floating_potential:.4g} V\n"
            f"Plasma potential estimate: {self.plasma_potential:.4g} V\n"
            f"Ion saturation current estimate: {self.ion_current:.4g} A\n"
            f"Electron temperature: {self.electron_temperature:.4g} "
            f"+/- {self.temperature_uncertainty:.2g} eV\n"
            f"Fit window: {self.fit_start:.6g} to {self.fit_end:.6g} V"
        )


def analyze(
    points: list[Point], *, fit_range: tuple[float, float] | None = None
) -> Analysis:
    data = np.asarray(points, dtype=float)
    if (
        data.shape != (len(points), 2)
        or len(points) < 16
        or not np.isfinite(data).all()
    ):
        raise ValueError("Analysis requires at least 16 finite I-V measurements.")

    voltage, inverse = np.unique(data[:, 0], return_inverse=True)
    current = np.bincount(inverse, weights=data[:, 1]) / np.bincount(inverse)
    if len(voltage) < 16:
        raise ValueError("Analysis requires at least 16 distinct voltages.")
    crossings = np.flatnonzero((current[:-1] <= 0) & (current[1:] > 0))
    if not len(crossings):
        raise ValueError("The trace must cross from negative to positive current.")
    crossing = crossings[0]
    floating = np.interp(
        0, current[crossing : crossing + 2], voltage[crossing : crossing + 2]
    )

    tail = voltage <= voltage[0] + 0.2 * (floating - voltage[0])
    if np.count_nonzero(tail) < 3:
        raise ValueError("The trace has too few points below the floating potential.")
    ion_slope, ion_intercept = np.polyfit(voltage[tail], current[tail], 1)
    electron = current - (ion_slope * voltage + ion_intercept)

    # Compare slope on a uniform grid so adaptive samples do not bias knee detection.
    grid = np.linspace(voltage[0], voltage[-1], min(max(len(voltage), 100), 2000))
    uniform = np.interp(grid, voltage, electron)
    derivative = np.gradient(uniform, grid)
    derivative = np.convolve(derivative, np.ones(5) / 5, mode="valid")
    probe_grid = grid[2:-2]
    derivative[probe_grid <= floating] = -np.inf
    knee_index = int(np.argmax(derivative))
    if knee_index in (0, len(derivative) - 1) or derivative[knee_index] <= 0:
        raise ValueError("No interior plasma-potential knee was resolved.")
    knee = probe_grid[knee_index]
    if fit_range is None:
        knee_current = np.interp(knee, voltage, electron)
        fit = (
            (voltage > floating)
            & (voltage < knee)
            & (electron >= 0.05 * knee_current)
            & (electron <= 0.75 * knee_current)
        )
    else:
        low, high = fit_range
        if not np.isfinite(fit_range).all() or low >= high:
            raise ValueError(
                "Fit range requires two finite voltages in increasing order."
            )
        fit = (voltage >= low - 1e-9) & (voltage <= high + 1e-9)
    fit &= electron > 0
    if np.count_nonzero(fit) < 4:
        raise ValueError("Too few usable points in the electron-retarding region.")
    (slope, intercept), covariance = np.polyfit(
        voltage[fit], np.log(electron[fit]), 1, cov=True
    )
    if slope <= 0:
        raise ValueError("The retarding-region fit must have a positive slope.")
    return Analysis(
        floating,
        knee,
        float(np.mean(current[tail])),
        1 / slope,
        float(np.sqrt(covariance[0, 0])) / slope**2,
        ion_slope,
        ion_intercept,
        voltage[fit][0],
        voltage[fit][-1],
        slope,
        intercept,
    )


def draw_trace(
    points: list[Point],
    axis: "Axes",
    *,
    semilog: bool = False,
    result: Analysis | None = None,
    microamps: bool = False,
) -> None:
    axis.clear()
    axis.set_xlabel("Probe voltage (V)")
    quantity = "Electron current" if semilog else "Probe current"
    unit = "µA" if microamps else "A"
    scale = 1e6 if microamps else 1
    axis.set_ylabel(f"{quantity} ({unit})")
    axis.grid(alpha=0.25)
    if not points:
        return
    voltage, current = np.asarray(sorted(points)).T
    if semilog:
        if result is None:
            raise ValueError("Semi-log analysis requires a fitted ion baseline.")
        electron = current - (result.ion_slope * voltage + result.ion_intercept)
        positive = electron > 0
        axis.semilogy(
            voltage[positive],
            electron[positive] * scale,
            ".",
            ms=3,
            label="Ion-corrected data",
        )
        fit_voltage = np.linspace(result.fit_start, result.fit_end, 50)
        axis.semilogy(
            fit_voltage,
            np.exp(result.log_slope * fit_voltage + result.log_intercept) * scale,
            color="#ad4628",
            lw=2,
            label="Retarding-region fit",
        )
    else:
        axis.plot(voltage, current * scale, color="#146f8c", lw=1.5)
        axis.axhline(0, color="0.5", lw=0.7)
        if result is not None:
            axis.axvline(
                result.floating_potential,
                ls="--",
                color="0.45",
                label="Floating potential",
            )
            axis.axvline(
                result.plasma_potential, ls=":", color="#ad4628", label="Knee estimate"
            )
    if result is not None:
        axis.legend(loc="best")


def save_plot(
    path: str | Path, points: list[Point], result: Analysis, *, semilog: bool = False
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.2, 4.2), layout="constrained")
    FigureCanvasAgg(figure)
    draw_trace(
        points, figure.add_subplot(), semilog=semilog, result=result, microamps=True
    )
    figure.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--plot", type=Path, help="Write an annotated I-V plot.")
    parser.add_argument(
        "--fit-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Override the automatic retarding-region fit window (volts).",
    )
    args = parser.parse_args(argv)
    try:
        points, _ = read_csv(args.csv)
        fit_range = tuple(args.fit_range) if args.fit_range is not None else None
        result = analyze(points, fit_range=fit_range)
        if args.plot:
            save_plot(args.plot, points, result)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Analysis failed: {error}\n")
    print(result.summary())
    return 0
