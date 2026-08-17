"""Qt window that shows the board at exactly one image pixel per screen pixel.

Kept apart from `tagcal.screen` so the metric logic stays importable without the
optional GUI dependency.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QGuiApplication, QImage, QKeyEvent, QPixmap, QScreen
from PySide6.QtWidgets import QApplication, QLabel

from tagcal.screen import ScreenLayout, configure_physical_pixels, render_board


class BoardWindow(QLabel):
    """Borderless-capable board view. Keys: Q/Esc close, I info, R ruler."""

    closed = Signal()

    def __init__(
        self,
        layout: ScreenLayout,
        *,
        show_info: bool = True,
        show_ruler: bool = False,
        fullscreen: bool = False,
        frameless: bool = False,
        on_top: bool = False,
        position: tuple[int, int] | None = None,
        warn: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._layout = layout
        self._show_info = show_info
        self._show_ruler = show_ruler
        self._fullscreen = fullscreen
        self._position = position
        self._warn = warn or (lambda message: print(message))
        self._buffer: NDArray[np.uint8] | None = None

        flags = Qt.WindowType.Window
        if frameless:
            flags |= Qt.WindowType.FramelessWindowHint
        if on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setWindowTitle("tagcal board")
        self.setScaledContents(False)  # any scaling would destroy the metric size
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setStyleSheet("background-color: white;")

        screen = self._target_screen()
        if screen is not None:
            self.setScreen(screen)
        self.refresh()
        self._place(screen)

    def _target_screen(self) -> QScreen | None:
        monitor = self._layout.monitor
        if monitor is None:
            return QGuiApplication.primaryScreen()
        screen = next(
            (item for item in QGuiApplication.screens() if item.name() == monitor.name),
            None,
        )
        if screen is None:
            self._warn(f"Qt does not know monitor {monitor.name}; using the primary screen")
            return QGuiApplication.primaryScreen()

        ratio = screen.devicePixelRatio()
        if abs(ratio - 1.0) > 1e-6:
            self._warn(
                f"devicePixelRatio={ratio}: logical and physical pixels differ, "
                "so the displayed size will not be metric"
            )
        geometry = screen.geometry()
        if (geometry.width(), geometry.height()) != (monitor.width_px, monitor.height_px):
            self._warn(
                f"Qt reports {geometry.width()}x{geometry.height()} but xrandr reports "
                f"{monitor.width_px}x{monitor.height_px}"
            )
        return screen

    def _place(self, screen: QScreen | None) -> None:
        geometry = screen.geometry() if screen is not None else None
        if self._fullscreen:
            self.setCursor(Qt.CursorShape.BlankCursor)
            if geometry is not None:
                self.setGeometry(geometry)
            self.showFullScreen()
            return

        pixmap = self.pixmap()
        width, height = pixmap.width(), pixmap.height()
        if geometry is not None:
            if self._position is None:
                # Bottom-right by default so the board stays clear of workspace windows.
                x = geometry.x() + geometry.width() - width - 40
                y = geometry.y() + geometry.height() - height - 80
            else:
                x = geometry.x() + self._position[0]
                y = geometry.y() + self._position[1]
            self.move(max(geometry.x(), x), max(geometry.y(), y))
        self.show()

    def refresh(self) -> None:
        size = None
        monitor = self._layout.monitor
        if self._fullscreen and monitor is not None:
            size = (monitor.width_px, monitor.height_px)
        image = np.ascontiguousarray(
            render_board(
                self._layout,
                show_info=self._show_info,
                show_ruler=self._show_ruler,
                size=size,
            )
        )
        height, width = image.shape
        qimage = QImage(image.data, width, height, width, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        pixmap.setDevicePixelRatio(1.0)  # never let Qt reinterpret these as logical pixels
        self._buffer = image  # QImage does not own the buffer it wraps
        self.setPixmap(pixmap)
        if not self._fullscreen:
            self.setFixedSize(width, height)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in {Qt.Key.Key_Escape, Qt.Key.Key_Q}:
            self.close()
        elif key == Qt.Key.Key_I:
            self._show_info = not self._show_info
            self.refresh()
        elif key == Qt.Key.Key_R:
            self._show_ruler = not self._show_ruler
            self.refresh()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)


def show_board_window(
    layout: ScreenLayout,
    *,
    show_info: bool = True,
    show_ruler: bool = False,
    fullscreen: bool = False,
    frameless: bool = False,
    on_top: bool = False,
    position: tuple[int, int] | None = None,
) -> int:
    """Run a standalone board window until the user closes it."""
    configure_physical_pixels()  # read by Qt when the application object is built
    app = QApplication.instance() or QApplication(sys.argv)
    window = BoardWindow(
        layout,
        show_info=show_info,
        show_ruler=show_ruler,
        fullscreen=fullscreen,
        frameless=frameless,
        on_top=on_top,
        position=position,
    )
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    return int(app.exec())  # type: ignore[union-attr]
