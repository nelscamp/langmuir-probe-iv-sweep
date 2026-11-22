"""Sweep settings, cooperative controls, and acquisition loop."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from threading import Condition
from time import sleep

from .instrument import Instrument


@dataclass(frozen=True)
class SweepSettings:
    start: float
    end: float
    step: float = 0.25
    current_limit: float = 0.001
    adaptive: bool = False
    sensitivity: float = 5e-6
    settle_seconds: float = 0.1

    def __post_init__(self) -> None:
        values = (
            self.start,
            self.end,
            self.step,
            self.current_limit,
            self.sensitivity,
            self.settle_seconds,
        )
        if not all(map(isfinite, values)):
            raise ValueError("Sweep settings must be finite.")
        if min(self.step, self.current_limit, self.sensitivity) <= 0:
            raise ValueError("Step, current limit, and sensitivity must be positive.")
        if self.settle_seconds < 0:
            raise ValueError("Settle time cannot be negative.")
        scale = max(abs(self.start), abs(self.end))
        if scale + self.step / 10 == scale:
            raise ValueError("Step is too small for this voltage range.")


class SweepControl:
    def __init__(self) -> None:
        self._condition = Condition()
        self._paused = False
        self._stopped = False

    @property
    def stopped(self) -> bool:
        with self._condition:
            return self._stopped

    def pause(self) -> None:
        with self._condition:
            self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def wait_until_ready(self) -> bool:
        with self._condition:
            self._condition.wait_for(lambda: self._stopped or not self._paused)
            return not self._stopped


def run_sweep(
    instrument: Instrument,
    settings: SweepSettings,
    control: SweepControl,
    on_point: Callable[[float, float], None],
) -> None:
    """Acquire one point per setpoint, including the stop voltage."""
    if not control.wait_until_ready():
        return
    direction = 1 if settings.end >= settings.start else -1
    voltage = settings.start
    previous_current = None
    try:
        instrument.configure(
            settings.current_limit, max(abs(voltage), abs(settings.end))
        )
        while control.wait_until_ready():
            instrument.set_voltage(voltage)
            sleep(settings.settle_seconds)
            measured_voltage, current = instrument.read_point()
            if not (isfinite(measured_voltage) and isfinite(current)):
                raise ValueError("Instrument returned a non-finite measurement.")
            on_point(measured_voltage, current)
            if voltage == settings.end:
                break
            step = settings.step
            if settings.adaptive and previous_current is not None:
                ratio = 2 / (1 + abs(current - previous_current) / settings.sensitivity)
                step *= max(0.1, ratio)
            previous_current = current
            next_voltage = voltage + direction * step
            if direction * (settings.end - next_voltage) <= step * 1e-9:
                next_voltage = settings.end
            voltage = next_voltage
    finally:
        instrument.disable_output()
