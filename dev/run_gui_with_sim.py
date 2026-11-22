import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dev.simulated_smu import SimulatedSMU  # noqa: E402
from langmuir_probe.gui import MainWindow  # noqa: E402


def main() -> int:
    capture = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    app = QtWidgets.QApplication(sys.argv[:1])
    window = MainWindow(
        lambda: ("GPIB0::1::INSTR",),
        lambda _: SimulatedSMU(plasma_potential=59.6, hot_fraction=0.12),
        0.0,
    )
    window.show()
    if capture is not None:

        def finish() -> None:
            if window.failed or len(window.points) != 141:
                app.exit(1)
                return
            window.analyze_trace()
            window.canvas.draw()
            capture.parent.mkdir(parents=True, exist_ok=True)
            app.exit(0 if window.grab().save(str(capture)) else 1)

        def start() -> None:
            window.connect_instrument()
            window.start_sweep()
            window.worker.finished.connect(finish)

        QtCore.QTimer.singleShot(0, start)
        QtCore.QTimer.singleShot(15_000, lambda: app.exit(1))
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
