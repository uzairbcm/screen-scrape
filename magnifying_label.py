from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6 import QtGui
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QMouseEvent, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel

if TYPE_CHECKING:
    from screenshot_app import ScreenshotApp


class MagnifyingLabel(QLabel):
    def __init__(self, parent_: ScreenshotApp) -> None:
        self.parent_ = parent_
        super().__init__(parent_)
        self.gray_pen = QtGui.QPen(QtGui.QColor(128, 128, 128))
        self.gray_pen.setWidth(2)
        self.setMouseTracking(True)
        self.screen_scale_factor = self.calculate_scale_factor()

    def calculate_scale_factor(self) -> float:
        screen_geo = QApplication.primaryScreen().geometry()  # type: ignore  # noqa: PGH003
        min_dimension = min(screen_geo.width(), screen_geo.height())
        return min_dimension / 1080

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        super().mousePressEvent(ev)
        if ev is not None:
            self.parent_.capture_click(ev)

    def mouseMoveEvent(self, ev: QMouseEvent | None) -> None:
        super().mouseMoveEvent(ev)

        if self.pixmap() is None or ev is None:
            return

        magnify_size = int(120 * self.screen_scale_factor)
        magnification = 2
        x, y = ev.pos().x(), ev.pos().y()
        pixmap = self.pixmap()
        half_size = magnify_size // (2 * magnification)

        # Calculate region to magnify
        if self.alignment() == Qt.AlignmentFlag.AlignTop:
            left = max(0, x - half_size)
            top = max(0, y - half_size)
            right = min(pixmap.width(), x + half_size)
            bottom = min(pixmap.height(), y + half_size)
            width, height = right - left, bottom - top
        else:
            left, top = 0, 0
            width, height = pixmap.width(), pixmap.height()

        if width <= 0 or height <= 0:
            return

        try:
            # Create magnified image
            rect = QRect(left, top, width, height)
            magnified_pixmap = pixmap.copy(rect).scaled(
                magnify_size, magnify_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )

            # Create final image with crosshair
            final_pixmap = QPixmap(magnify_size, magnify_size)
            final_pixmap.fill(Qt.GlobalColor.white)

            # Draw magnified content
            x_offset = (magnify_size - magnified_pixmap.width()) // 2
            y_offset = (magnify_size - magnified_pixmap.height()) // 2
            painter = QPainter(final_pixmap)
            painter.drawPixmap(x_offset, y_offset, magnified_pixmap)

            # Draw crosshair
            painter.setPen(self.gray_pen)
            crosshair_x = magnify_size // 2
            crosshair_y = magnify_size // 2
            crosshair_length = int(10 * self.screen_scale_factor)
            painter.drawLine(crosshair_x - crosshair_length, crosshair_y, crosshair_x + crosshair_length, crosshair_y)
            painter.drawLine(crosshair_x, crosshair_y - crosshair_length, crosshair_x, crosshair_y + crosshair_length)
            painter.end()

            # Position and display magnifier
            cursor_offset = 5
            magnifier_x = min(x + cursor_offset, self.parent_.width() - magnify_size)
            magnifier_y = max(y - magnify_size - cursor_offset, 0)

            self.parent_.magnifier_label.setPixmap(final_pixmap)
            self.parent_.magnifier_label.resize(magnify_size, magnify_size)
            self.parent_.magnifier_label.move(magnifier_x, magnifier_y)
            self.parent_.magnifier_label.setVisible(True)
            self.parent_.magnifier_label.raise_()

        except Exception as e:
            print(f"Error creating magnifier: {e}")
