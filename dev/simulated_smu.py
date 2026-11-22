"""Development-only SMU with an ion baseline and one or two electron populations."""

from math import exp, log1p
from random import Random


class SimulatedSMU:
    identity = "Development SMU"

    def __init__(
        self,
        *,
        temperature: float = 0.9,
        plasma_potential: float = 15.0,
        ion_current: float = -6e-6,
        electron_current: float = 100e-6,
        sheath_slope: float = 6e-8,
        hot_fraction: float = 0.0,
        hot_temperature: float = 3.0,
        noise: float = 2e-8,
        seed: int = 23,
    ) -> None:
        self.temperature = temperature
        self.plasma_potential = plasma_potential
        self.ion_current = ion_current
        self.electron_current = electron_current
        self.sheath_slope = sheath_slope
        self.hot_fraction = hot_fraction
        self.hot_temperature = hot_temperature
        self.noise = noise
        self.random = Random(seed)
        self.output_enabled = False
        self.voltage = 0.0
        self.current_limit = float("inf")
        self.setpoints: list[float] = []

    def configure(self, current_limit: float, max_abs_voltage: float) -> None:
        self.current_limit = current_limit
        self.output_enabled = True

    def set_voltage(self, voltage: float) -> None:
        if not self.output_enabled:
            raise RuntimeError("Output is disabled.")
        self.voltage = voltage
        self.setpoints.append(voltage)

    def read_point(self) -> tuple[float, float]:
        bias = self.voltage - self.plasma_potential
        width = 0.35
        # Round the change between regimes while preserving the exponential tail.
        retarding_bias = min(0.0, bias)
        if abs(bias) < width:
            retarding_bias = -((width - bias) ** 2) / (4 * width)
        cold = exp(retarding_bias / self.temperature)
        hot = exp(retarding_bias / self.hot_temperature)
        electron = (1 - self.hot_fraction) * cold + self.hot_fraction * hot
        electron += 0.09 * log1p(bias - retarding_bias)
        current = (
            self.ion_current
            + self.sheath_slope * bias
            + self.electron_current * electron
        )
        current += self.random.gauss(0, self.noise)
        return self.voltage, min(self.current_limit, max(-self.current_limit, current))

    def disable_output(self) -> None:
        self.output_enabled = False

    def close(self) -> None:
        self.disable_output()
