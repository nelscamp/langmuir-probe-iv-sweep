"""VISA discovery and the Aim-TTi SMU4000-series SCPI adapter."""

import re
from typing import Protocol


class Instrument(Protocol):
    @property
    def identity(self) -> str: ...
    def configure(self, current_limit: float, max_abs_voltage: float) -> None: ...
    def set_voltage(self, voltage: float) -> None: ...
    def read_point(self) -> tuple[float, float]: ...
    def disable_output(self) -> None: ...
    def close(self) -> None: ...


def visa_resources() -> tuple[str, ...]:
    import pyvisa

    with pyvisa.ResourceManager() as manager:
        return manager.list_resources()


class SMU4201:
    """Aim-TTi SMU4000-series adapter, written for the SMU4201 over VISA."""

    def __init__(self, address: str) -> None:
        import pyvisa

        self._manager = pyvisa.ResourceManager()
        try:
            self._resource = self._manager.open_resource(address)
            # Explicit line feeds work on GPIB and on the USB virtual COM port, where
            # VISA's serial defaults can stall reads until the timeout.
            self._resource.write_termination = "\n"
            self._resource.read_termination = "\n"
            self._resource.timeout = 10_000
            self._identity = self._resource.query("*IDN?").strip()
            if not re.search(r"SMU4\d{3}", self._identity.upper()):
                raise ValueError(
                    "Expected an Aim-TTi SMU4000-series instrument. "
                    f"Device reported {self._identity!r}."
                )
        except BaseException:
            self._manager.close()
            raise

    @property
    def identity(self) -> str:
        return self._identity

    def configure(self, current_limit: float, max_abs_voltage: float) -> None:
        commands = (
            "OUTPut:STATe OFF",
            "SOURce:FUNCtion:MODE VOLTage",
            f"SOURce:VOLTage:CURRent:LIMit {current_limit:.12g}",
            "SOURce:VOLTage:RANGe:CURRent:AUTO ON",
            "SOURce:VOLTage:MEASure:PRIMary VOLTage",
            "SOURce:VOLTage:MEASure:SECondary CURRent",
            f"SYSTem:MODE:HV:STATe {int(max_abs_voltage >= 42.0)}",
            "SOURce:VOLTage:FIXed 0",
            "OUTPut:STATe ON",
        )
        for command in commands:
            self._resource.write(command)

    def set_voltage(self, voltage: float) -> None:
        self._resource.write(f"SOURce:VOLTage:FIXed {voltage:.12g}")

    def read_point(self) -> tuple[float, float]:
        return (
            float(self._resource.query("MEASure:PRIMary:LIVEdata?")),
            float(self._resource.query("MEASure:SECondary:LIVEdata?")),
        )

    def disable_output(self) -> None:
        self._resource.write("OUTPut:STATe OFF")

    def close(self) -> None:
        try:
            self._resource.close()
        finally:
            self._manager.close()
