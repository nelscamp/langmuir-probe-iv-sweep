"""VISA discovery and the Aim-TTi SMU4201 SCPI adapter."""

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
    def __init__(self, address: str) -> None:
        import pyvisa

        self._manager = pyvisa.ResourceManager()
        try:
            self._resource = self._manager.open_resource(address)
            self._resource.timeout = 3000
            self._identity = self._resource.query("*IDN?").strip()
            if "SMU4201" not in self._identity.upper():
                raise ValueError(
                    f"Expected an SMU4201. Device reported {self._identity!r}."
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
