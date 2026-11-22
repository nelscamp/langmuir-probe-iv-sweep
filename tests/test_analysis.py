from pathlib import Path

import pytest

from dev.simulated_smu import SimulatedSMU
from langmuir_probe.analysis import analyze, main
from langmuir_probe.data import read_csv, write_csv
from langmuir_probe.sweep import SweepControl, SweepSettings, run_sweep


@pytest.mark.parametrize("temperature", [0.65, 0.9, 1.5])
def test_recovers_electron_temperature(temperature: float) -> None:
    instrument = SimulatedSMU(temperature=temperature)
    points = []
    run_sweep(
        instrument,
        SweepSettings(0, 22, step=0.05, settle_seconds=0),
        SweepControl(),
        lambda v, i: points.append((v, i)),
    )
    result = analyze(points)
    assert result.electron_temperature == pytest.approx(temperature, rel=0.05)
    assert result.plasma_potential == pytest.approx(15, abs=0.5)
    assert result.floating_potential < result.plasma_potential
    assert result.ion_current < 0
    assert result.temperature_uncertainty > 0


def test_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sweep.csv"
    points = [(0.0, -1e-6), (1.0, 2e-6)]
    metadata = {
        "date": "2025-05-01T00:00:00+00:00",
        "instrument_id": "SMU4201",
        "settings": {"current_limit": 0.001},
    }
    write_csv(path, points, metadata)
    assert read_csv(path) == (points, metadata)


def test_example_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    example = (
        Path(__file__).resolve().parents[1] / "examples" / "voltage_current_data_23.csv"
    )
    plot = tmp_path / "sweep.png"
    assert main([str(example), "--plot", str(plot)]) == 0
    assert "Electron temperature:" in capsys.readouterr().out
    assert plot.read_bytes().startswith(b"\x89PNG")

    assert main([str(example), "--fit-range", "55", "59"]) == 0
    assert "Fit window: 55 to 59 V" in capsys.readouterr().out

    with pytest.raises(SystemExit) as failure:
        main([str(example), "--fit-range", "59", "55"])
    assert failure.value.code == 2
    assert "in increasing order" in capsys.readouterr().err
