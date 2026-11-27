"""Check the SCPI the driver sends, using a stand-in for PyVISA (no hardware)."""

import sys
from types import SimpleNamespace

import pytest

from langmuir_probe.instrument import SMU4201
from langmuir_probe.sweep import SweepControl, SweepSettings, run_sweep


class FakeResource:
    def __init__(self, identity: str) -> None:
        self.identity = identity
        self.writes: list[str] = []
        self.closed = False

    def write(self, command: str) -> None:
        self.writes.append(command)

    def query(self, command: str) -> str:
        replies = {
            "*IDN?": self.identity,
            "MEASure:PRIMary:LIVEdata?": "40.0\n",
            "MEASure:SECondary:LIVEdata?": "-5.2e-06\n",
        }
        return replies[command]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def visa(monkeypatch: pytest.MonkeyPatch):
    state = SimpleNamespace(identity="Aim-TTi,SMU4201,000000,1.00\n", managers=[])

    class ResourceManager:
        def __init__(self) -> None:
            self.closed = False
            self.resource = FakeResource(state.identity)
            state.managers.append(self)

        def open_resource(self, address: str) -> FakeResource:
            return self.resource

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(
        sys.modules, "pyvisa", SimpleNamespace(ResourceManager=ResourceManager)
    )
    return state


def test_connection_settings(visa) -> None:
    smu = SMU4201("ASRL3::INSTR")
    resource = visa.managers[0].resource
    assert resource.write_termination == "\n"
    assert resource.read_termination == "\n"
    assert resource.timeout == 10_000
    assert smu.identity.startswith("Aim-TTi,SMU4201")


def test_accepts_other_smu4000_models(visa) -> None:
    visa.identity = "Aim-TTi,SMU4001,000000,1.00"
    assert "SMU4001" in SMU4201("GPIB0::1::INSTR").identity


def test_rejects_other_instruments(visa) -> None:
    visa.identity = "KEITHLEY INSTRUMENTS, MODEL 2400, 1, C30"
    with pytest.raises(ValueError, match="SMU4000-series"):
        SMU4201("GPIB0::24::INSTR")
    assert visa.managers[0].closed


@pytest.mark.parametrize(("end", "hv"), [(40, 0), (75, 1), (-100, 1)])
def test_sweep_command_sequence(visa, end: float, hv: int) -> None:
    smu = SMU4201("GPIB0::1::INSTR")
    run_sweep(
        smu,
        SweepSettings(0, end, step=abs(end), settle_seconds=0, first_settle_seconds=0),
        SweepControl(),
        lambda *_: None,
    )
    assert visa.managers[0].resource.writes == [
        "OUTPut:STATe OFF",
        "SOURce:FUNCtion:MODE VOLTage",
        "SOURce:VOLTage:CURRent:LIMit 0.001",
        "SOURce:VOLTage:RANGe:CURRent:AUTO ON",
        "SOURce:VOLTage:MEASure:PRIMary VOLTage",
        "SOURce:VOLTage:MEASure:SECondary CURRent",
        f"SYSTem:MODE:HV:STATe {hv}",
        "SOURce:VOLTage:FIXed 0",
        "OUTPut:STATe ON",
        "SOURce:VOLTage:FIXed 0",
        f"SOURce:VOLTage:FIXed {end}",
        "OUTPut:STATe OFF",
    ]
