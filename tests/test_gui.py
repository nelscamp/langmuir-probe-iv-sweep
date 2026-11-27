"""Drive the real window with the development SMU, headless."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PyQt5.QtCore")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

from dev.simulated_smu import SimulatedSMU  # noqa: E402
from langmuir_probe.gui import MainWindow  # noqa: E402

SWEEP = Path(__file__).resolve().parents[1] / "examples" / "voltage_current_data_23.csv"


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def run_gui_sweep(window: MainWindow) -> None:
    loop = QtCore.QEventLoop()
    window.connect_instrument()
    window.start_sweep()
    window.worker.finished.connect(lambda: QtCore.QTimer.singleShot(0, loop.quit))
    QtCore.QTimer.singleShot(20_000, loop.quit)
    loop.exec_()


def make_window(smu: SimulatedSMU) -> MainWindow:
    window = MainWindow(lambda: ("GPIB0::1::INSTR",), lambda _: smu, 0)
    window.start_voltage.setValue(0)
    window.end_voltage.setValue(20)
    window.step.setValue(0.5)
    return window


def test_full_sweep(app) -> None:
    smu = SimulatedSMU()
    window = make_window(smu)
    run_gui_sweep(window)
    assert window.worker is None
    assert len(window.points) == 41
    assert window.status.text() == "Sweep complete: 41 points"
    assert window.metadata["points_at_current_limit"] == 0
    assert not smu.output_enabled
    window.log_abs.setChecked(True)
    window.analyze_trace()
    assert "Te:" in window.summary.text()
    window.semilog.setChecked(True)


def test_warns_at_current_limit(app) -> None:
    window = make_window(SimulatedSMU())
    window.compliance.setValue(0.05)  # 50 µA, below the simulated electron current
    run_gui_sweep(window)
    assert window.metadata["points_at_current_limit"] > 0
    assert "at current limit" in window.status.text()


def test_open_and_analyze_csv(app) -> None:
    window = MainWindow(lambda: (), lambda _: SimulatedSMU())
    assert "No VISA resources found" in window.status.text()
    window.load_csv(SWEEP)
    assert window.status.text() == "Opened CSV: 176 points"
    window.analyze_trace()
    assert "Floating potential: 51.09 V" in window.summary.text()
