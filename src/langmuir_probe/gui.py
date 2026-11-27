"""PyQt instrument picker, live sweep controls, and trace analysis."""

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from sys import argv

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

from .analysis import Analysis, analyze, draw_trace
from .data import Point, read_csv, write_csv
from .instrument import SMU4201, Instrument, visa_resources
from .sweep import SweepControl, SweepSettings, at_current_limit, run_sweep


class SweepWorker(QtCore.QThread):
    point = QtCore.pyqtSignal(float, float)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        instrument: Instrument,
        settings: SweepSettings,
        control: SweepControl,
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        self.instrument = instrument
        self.settings = settings
        self.control = control

    def run(self) -> None:
        try:
            run_sweep(self.instrument, self.settings, self.control, self.point.emit)
        except Exception as error:
            self.failed.emit(str(error))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        discover: Callable[[], tuple[str, ...]] = visa_resources,
        connect: Callable[[str], Instrument] = SMU4201,
        settle_seconds: float = 0.1,
    ) -> None:
        super().__init__()
        self.discover = discover
        self.connect = connect
        self.instrument: Instrument | None = None
        self.worker: SweepWorker | None = None
        self.control = SweepControl()
        self.points: list[Point] = []
        self.current_limit = float("inf")
        self.clipped = 0
        self.metadata: dict = {}
        self.result: Analysis | None = None
        self.paused = False
        self.failed = False
        self.plot_pending = False

        self.setWindowTitle("Langmuir Probe I–V Sweep")
        self.resize(1100, 680)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        sidebar = QtWidgets.QVBoxLayout()
        layout.addLayout(sidebar)
        plot = QtWidgets.QVBoxLayout()
        layout.addLayout(plot, 1)
        figure = Figure(layout="constrained", dpi=130)
        self.canvas = FigureCanvasQTAgg(figure)
        self.axis = figure.add_subplot()
        plot.addWidget(NavigationToolbar2QT(self.canvas, self))
        plot.addWidget(self.canvas, 1)

        device_group = QtWidgets.QGroupBox("Instrument")
        device_layout = QtWidgets.QVBoxLayout(device_group)
        self.resources = QtWidgets.QComboBox()
        self.resources.setAccessibleName("VISA resource")
        self.resources.setEditable(True)
        self.resources.lineEdit().setPlaceholderText("Select or enter VISA address")
        device_layout.addWidget(self.resources)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_resources)
        device_layout.addWidget(self.refresh_button)
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_instrument)
        device_layout.addWidget(self.connect_button)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setMaximumWidth(295)
        device_layout.addWidget(self.status)
        sidebar.addWidget(device_group)

        settings_group = QtWidgets.QGroupBox("Sweep")
        settings_form = QtWidgets.QFormLayout(settings_group)
        self.start_voltage = self._number(settings_form, "Start (V)", 40, -200, 200)
        self.end_voltage = self._number(settings_form, "End (V)", 75, -200, 200)
        self.step = self._number(settings_form, "Step (V)", 0.25, 0.001, 200)
        self.compliance = self._number(
            settings_form, "Current limit (mA)", 1, 0.001, 100
        )
        self.settle = self._number(
            settings_form, "Settle time (ms)", settle_seconds * 1000, 0, 10_000, 0
        )
        self.sensitivity = self._number(settings_form, "Sensitivity (µA)", 5, 0.1, 25)
        self.adaptive = QtWidgets.QCheckBox("Adaptive steps")
        self.adaptive.toggled.connect(self.sensitivity.setEnabled)
        self.sensitivity.setEnabled(False)
        settings_form.addRow(self.adaptive)
        sidebar.addWidget(settings_group)

        transport = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.start_button.clicked.connect(self.start_sweep)
        self.pause_button.clicked.connect(self.pause_resume)
        self.stop_button.clicked.connect(self.stop_sweep)
        for button in (self.start_button, self.pause_button, self.stop_button):
            transport.addWidget(button)
        sidebar.addLayout(transport)

        self.import_button = QtWidgets.QPushButton("Open CSV")
        self.import_button.clicked.connect(self.import_csv)
        self.export_button = QtWidgets.QPushButton("Save CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.analysis_button = QtWidgets.QPushButton("Analyze trace")
        self.analysis_button.clicked.connect(self.analyze_trace)
        views = QtWidgets.QHBoxLayout()
        self.log_abs = QtWidgets.QCheckBox("Log |I|")
        self.log_abs.setToolTip("Raw current magnitude on a log axis, live")
        self.semilog = QtWidgets.QCheckBox("Semi-log fit")
        self.semilog.setToolTip("Ion-corrected electron current and Te fit")
        for box in (self.log_abs, self.semilog):
            box.toggled.connect(self.refresh_plot)
            views.addWidget(box)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setMaximumWidth(295)
        for widget in (
            self.import_button,
            self.export_button,
            self.analysis_button,
        ):
            sidebar.addWidget(widget)
        sidebar.addLayout(views)
        sidebar.addWidget(self.summary)
        sidebar.addStretch()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_live_plot)
        self.timer.start(100)
        self.refresh_resources()
        self.refresh_plot()

    @staticmethod
    def _number(
        form: QtWidgets.QFormLayout,
        label: str,
        value: float,
        minimum: float,
        maximum: float,
        decimals: int = 3,
    ) -> QtWidgets.QDoubleSpinBox:
        field = QtWidgets.QDoubleSpinBox()
        field.setDecimals(decimals)
        field.setRange(minimum, maximum)
        field.setValue(value)
        field.setKeyboardTracking(False)
        form.addRow(label, field)
        return field

    def refresh_resources(self) -> None:
        previous = self.resources.currentText()
        self.resources.clear()
        try:
            addresses = self.discover()
        except Exception as error:
            self.status.setText(
                f"VISA unavailable. Check your VISA installation.\n{error}"
            )
        else:
            self.resources.addItems(addresses)
            if addresses:
                self.status.setText("Select a VISA resource and connect.")
            else:
                self.status.setText(
                    "No VISA resources found. Connect your instrument and refresh."
                )
        if previous:
            self.resources.setCurrentText(previous)
        if self.instrument is not None:
            self.status.setText(f"Connected to {self.instrument.identity}")
        self._update_controls()

    def connect_instrument(self) -> None:
        address = self.resources.currentText().strip()
        if not address:
            self.status.setText(
                "Connect your instrument, then refresh or enter its VISA address."
            )
            return
        try:
            if self.instrument is not None:
                previous, self.instrument = self.instrument, None
                previous.close()
            self.instrument = self.connect(address)
        except Exception as error:
            self.status.setText(f"Connection failed.\n{error}")
        else:
            self.status.setText(f"Connected to {self.instrument.identity}")
        self._update_controls()

    def _update_controls(self) -> None:
        running = self.worker is not None
        self.start_button.setEnabled(self.instrument is not None and not running)
        self.pause_button.setEnabled(running and not self.control.stopped)
        self.stop_button.setEnabled(running and not self.control.stopped)
        for widget in (
            self.refresh_button,
            self.connect_button,
            self.resources,
            self.import_button,
        ):
            widget.setEnabled(not running)
        self.export_button.setEnabled(bool(self.points) and not running)
        self.analysis_button.setEnabled(bool(self.points) and not running)
        self.semilog.setEnabled(self.result is not None)

    def _clear_analysis(self) -> None:
        self.result = None
        self.summary.clear()
        self.semilog.setChecked(False)

    def start_sweep(self) -> None:
        if self.instrument is None or self.worker is not None:
            return
        try:
            settings = SweepSettings(
                start=self.start_voltage.value(),
                end=self.end_voltage.value(),
                step=self.step.value(),
                current_limit=self.compliance.value() / 1000,
                adaptive=self.adaptive.isChecked(),
                sensitivity=self.sensitivity.value() * 1e-6,
                settle_seconds=self.settle.value() / 1000,
            )
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.points.clear()
        self.current_limit = settings.current_limit
        self.clipped = 0
        self.status.setToolTip("")
        self._clear_analysis()
        self.metadata = {
            "date": datetime.now(timezone.utc).isoformat(),
            "instrument_id": self.instrument.identity,
            "settings": asdict(settings),
        }
        self.control = SweepControl()
        self.paused = self.failed = False
        self.pause_button.setText("Pause")
        self.worker = SweepWorker(self.instrument, settings, self.control, self)
        self.worker.point.connect(self._receive_point)
        self.worker.failed.connect(self._sweep_error)
        self.worker.finished.connect(self._finish_sweep)
        self.status.setText("Acquiring measurements…")
        self.refresh_plot()
        self._update_controls()
        self.worker.start()

    @QtCore.pyqtSlot(float, float)
    def _receive_point(self, voltage: float, current: float) -> None:
        self.points.append((voltage, current))
        self.clipped += at_current_limit(current, self.current_limit)
        self.plot_pending = True

    def _update_live_plot(self) -> None:
        if self.plot_pending:
            self.refresh_plot()

    def refresh_plot(self) -> None:
        draw_trace(
            self.points,
            self.axis,
            semilog=self.semilog.isChecked(),
            log_abs=self.log_abs.isChecked(),
            result=self.result,
            microamps=True,
        )
        self.canvas.draw_idle()
        self.plot_pending = False

    def pause_resume(self) -> None:
        if self.paused:
            self.control.resume()
            self.status.setText("Acquiring measurements…")
            self.pause_button.setText("Pause")
        else:
            self.control.pause()
            self.status.setText("Paused. Instrument output remains enabled.")
            self.pause_button.setText("Resume")
        self.paused = not self.paused

    def stop_sweep(self) -> None:
        self.control.stop()
        self.status.setText("Stopping sweep…")
        self._update_controls()

    @QtCore.pyqtSlot(str)
    def _sweep_error(self, error: str) -> None:
        self.failed = True
        self.status.setText(f"Sweep failed.\n{error}")

    @QtCore.pyqtSlot()
    def _finish_sweep(self) -> None:
        self.worker.deleteLater()
        self.worker = None
        outcome = (
            "failed"
            if self.failed
            else "stopped"
            if self.control.stopped
            else "complete"
        )
        self.metadata["outcome"] = outcome
        self.metadata["points_at_current_limit"] = self.clipped
        if not self.failed:
            message = f"Sweep {outcome}: {len(self.points)} points"
            if self.clipped:
                message += f"\nWarning: {self.clipped} at current limit"
                self.status.setToolTip(
                    "Those readings are clipped. "
                    "Raise the limit or narrow the voltage range."
                )
            self.status.setText(message)
        self.refresh_plot()
        self._update_controls()

    def import_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open sweep", filter="CSV (*.csv)"
        )
        if path:
            self.load_csv(path)

    def load_csv(self, path: str | Path) -> None:
        try:
            self.points, self.metadata = read_csv(path)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.warning(self, "CSV import failed", str(error))
            return
        self.status.setText(f"Opened CSV: {len(self.points)} points")
        self.status.setToolTip(str(path))
        self._clear_analysis()
        self.refresh_plot()
        self._update_controls()

    def export_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save sweep", filter="CSV (*.csv)"
        )
        if path:
            try:
                write_csv(path, self.points, self.metadata)
            except OSError as error:
                QtWidgets.QMessageBox.warning(self, "CSV export failed", str(error))

    def analyze_trace(self) -> None:
        try:
            self.result = analyze(self.points)
        except ValueError as error:
            self.summary.setText(str(error))
            return
        result = self.result
        self.summary.setText(
            f"Floating potential: {result.floating_potential:.2f} V\n"
            f"Plasma potential: {result.plasma_potential:.1f} V\n"
            f"Ion current: {result.ion_current * 1e6:.2f} µA\n"
            f"Te: {result.electron_temperature:.2f} ± "
            f"{result.temperature_uncertainty:.2f} eV\n"
            f"Fit window: {result.fit_start:.1f} to {result.fit_end:.1f} V"
        )
        self.refresh_plot()
        self._update_controls()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker is not None:
            self.stop_sweep()
            event.ignore()
            return
        if self.instrument is not None:
            try:
                self.instrument.close()
            except Exception as error:
                QtWidgets.QMessageBox.warning(
                    self, "Instrument close failed", str(error)
                )
        super().closeEvent(event)


def main() -> int:
    application = QtWidgets.QApplication(argv)
    window = MainWindow()
    window.show()
    return application.exec_()
