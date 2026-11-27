from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from dev.simulated_smu import SimulatedSMU
from langmuir_probe.sweep import (
    SweepControl,
    SweepSettings,
    at_current_limit,
    run_sweep,
)

FAST = {"settle_seconds": 0, "first_settle_seconds": 0}


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0, 1, [0, 0.3, 0.6, 0.9, 1]),
        (1, 0, [1, 0.7, 0.4, 0.1, 0]),
        (1, 1, [1]),
    ],
)
def test_fixed_steps(start: float, end: float, expected: list[float]) -> None:
    smu = SimulatedSMU()
    run_sweep(
        smu,
        SweepSettings(start, end, 0.3, **FAST),
        SweepControl(),
        lambda *_: None,
    )
    assert smu.setpoints == pytest.approx(expected)
    assert not smu.output_enabled


def test_adaptive_refines_steep_region() -> None:
    smu = SimulatedSMU(noise=0)
    run_sweep(
        smu,
        SweepSettings(0, 18, adaptive=True, **FAST),
        SweepControl(),
        lambda *_: None,
    )
    steps = [
        (v, following - v) for v, following in zip(smu.setpoints, smu.setpoints[1:])
    ]
    assert max(step for v, step in steps if 2 < v < 8) > 0.25
    assert min(step for v, step in steps if 14 < v < 15) < 0.2
    assert smu.setpoints.count(18) == 1


@pytest.mark.parametrize("resume", [False, True])
def test_pause_and_stop(resume: bool) -> None:
    control = SweepControl()
    first_point = Event()
    second_point = Event()
    smu = SimulatedSMU()

    def on_point(voltage: float, current: float) -> None:
        if voltage == 0:
            control.pause()
            first_point.set()
        else:
            second_point.set()
            control.stop()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            run_sweep, smu, SweepSettings(0, 3, **FAST), control, on_point
        )
        try:
            assert first_point.wait(2)
            assert not second_point.wait(0.05)
            if resume:
                control.resume()
                assert second_point.wait(2)
        finally:
            control.stop()
        future.result(timeout=2)
    assert smu.setpoints == ([0, 0.25] if resume else [0])
    assert not smu.output_enabled


def test_disables_output_on_failure() -> None:
    smu = SimulatedSMU()

    def fail(voltage: float, current: float) -> None:
        raise RuntimeError("acquisition interrupted")

    with pytest.raises(RuntimeError, match="acquisition interrupted"):
        run_sweep(smu, SweepSettings(0, 3, **FAST), SweepControl(), fail)
    assert not smu.output_enabled


def test_first_point_waits_longer(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    monkeypatch.setattr("langmuir_probe.sweep.sleep", waits.append)
    run_sweep(
        SimulatedSMU(),
        SweepSettings(0, 1, 0.5, settle_seconds=0.1, first_settle_seconds=1.0),
        SweepControl(),
        lambda *_: None,
    )
    assert waits == [1.0, 0.1, 0.1]


def test_current_limit_detection() -> None:
    assert at_current_limit(-1e-3, 1e-3)
    assert at_current_limit(0.995e-3, 1e-3)
    assert not at_current_limit(0.9e-3, 1e-3)
