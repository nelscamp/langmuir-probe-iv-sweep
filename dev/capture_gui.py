"""Capture the README screenshot: the app with measured sweep 23 opened and analyzed.

Run in a desktop session (Qt's offscreen platform renders no text on Windows):
    python dev/capture_gui.py docs/img/sweep-23-app.png
"""

import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from langmuir_probe.gui import MainWindow

SWEEP = Path(__file__).resolve().parents[1] / "examples" / "voltage_current_data_23.csv"


def main() -> int:
    capture = Path(sys.argv[1])
    app = QtWidgets.QApplication(sys.argv[:1])
    window = MainWindow(lambda: ("GPIB0::1::INSTR",))
    window.step.setValue(0.2)  # match the recorded sweep's step
    window.resize(1600, 760)
    window.show()

    def grab() -> None:
        window.load_csv(SWEEP)
        window.analyze_trace()
        window.canvas.draw()
        QtCore.QTimer.singleShot(300, save)

    def save() -> None:
        capture.parent.mkdir(parents=True, exist_ok=True)
        app.exit(0 if window.grab().save(str(capture)) else 1)

    QtCore.QTimer.singleShot(500, grab)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
