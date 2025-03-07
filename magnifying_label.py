from PyQt5 import QtGui
from PyQt5.QtCore import QRect, Qt, QSize
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QLabel, QDesktopWidget


class MagnifyingLabel(QLabel):
    def __init__(self, parent=None):
        super(MagnifyingLabel, self).__init__(parent)
        self.gray_pen = QtGui.QPen(QtGui.QColor(128, 128, 128))
        self.gray_pen.setWidth(2)
        self.setMouseTracking(True)
        self.screen_scale_factor = self.calculate_scale_factor()

    def calculate_scale_factor(self) -> float:
        screen_geo = QDesktopWidget().screenGeometry()
        min_dimension = min(screen_geo.width(), screen_geo.height())
        return min_dimension / 1080

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.parent() and hasattr(self.parent(), "capture_click"):
            self.parent().capture_click(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

        if self.pixmap() is not None and self.parent() and hasattr(self.parent(), "magnifier_label"):
            magnify_size = int(120 * self.screen_scale_factor)
            magnification = 2

            x = event.pos().x()
            y = event.pos().y()

            pixmap = self.pixmap()

            half_size = magnify_size // (2 * magnification)

            if self.alignment() == Qt.AlignTop:
                left = max(0, x - half_size)
                top = max(0, y - half_size)
                right = min(pixmap.width(), x + half_size)
                bottom = min(pixmap.height(), y + half_size)

                width = right - left
                height = bottom - top
            else:
                left = 0
                top = 0
                width = pixmap.width()
                height = pixmap.height()

            if width <= 0 or height <= 0:
                return

            rect = QRect(left, top, width, height)
            try:
                magnified_pixmap = pixmap.copy(rect)
                magnified_pixmap = magnified_pixmap.scaled(magnify_size, magnify_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                final_pixmap = QPixmap(magnify_size, magnify_size)
                final_pixmap.fill(Qt.white)

                x_offset = (magnify_size - magnified_pixmap.width()) // 2
                y_offset = (magnify_size - magnified_pixmap.height()) // 2

                painter = QPainter(final_pixmap)
                painter.drawPixmap(x_offset, y_offset, magnified_pixmap)
                painter.setPen(self.gray_pen)

                crosshair_x = magnify_size // 2
                crosshair_y = magnify_size // 2
                crosshair_length = int(10 * self.screen_scale_factor)

                painter.drawLine(crosshair_x - crosshair_length, crosshair_y, crosshair_x + crosshair_length, crosshair_y)
                painter.drawLine(crosshair_x, crosshair_y - crosshair_length, crosshair_x, crosshair_y + crosshair_length)
                painter.end()

                cursor_offset = 5
                magnifier_x = x + cursor_offset
                magnifier_y = y - magnify_size - cursor_offset

                parent_width = self.parent().width()
                parent_height = self.parent().height()

                if magnifier_x + magnify_size > parent_width:
                    magnifier_x = parent_width - magnify_size

                if magnifier_y < 0:
                    magnifier_y = 0

                self.parent().magnifier_label.setPixmap(final_pixmap)
                self.parent().magnifier_label.resize(magnify_size, magnify_size)
                self.parent().magnifier_label.move(magnifier_x, magnifier_y)
                self.parent().magnifier_label.setVisible(True)
                self.parent().magnifier_label.raise_()

            except Exception as e:
                print(f"Error creating magnifier: {e}")
