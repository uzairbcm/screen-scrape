import os
import re
import sys
import traceback

import cv2
import pandas as pd
import pytesseract
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDesktopWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from image_processor import *  # process_image, process_image_with_grid, mse_between_loaded_images, hconcat_resize
from magnifying_label import MagnifyingLabel

# Note: May need to be imported on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class BatteryRow:
    @classmethod
    def headers(cls):
        return [
            "Full path",
            "File name",
            "ID",
            "Date from file name",
            "Date from image",
            "Time from image",
            *list(range(24)),
            "Total",
        ]

    def __init__(self, full_path, file_name, date_from_image, time_from_ui, rows):
        super().__init__()
        self.full_path = full_path
        self.file_name = file_name
        self.date_from_image = date_from_image
        self.time_from_ui = time_from_ui
        self.rows = rows

    def date_extracted_from_file_name(self):
        match = re.search(r"\d{1,2}[\.|-]\d{1,2}[\.|-]\d{2,4}", self.full_path)

        if match:
            return match.group(0)
        return ""

    def subject_id_extracted_from_file_path(self):
        directory = os.path.dirname(self.full_path)
        # print("ID:", directory)
        return os.path.basename(directory)

    def to_csv_row(self):
        return [
            self.full_path,
            self.file_name,
            self.subject_id_extracted_from_file_path(),
            self.date_extracted_from_file_name(),
            self.date_from_image,
            self.time_from_ui,
            *self.rows,
        ]


class ScreenTimeRow:
    @classmethod
    def headers(cls):
        return ["Full path", "Filename", "ID", "Date", "App Title"] + [str(i) for i in range(24)] + ["Total"]

    def __init__(self, full_path, file_name, app_title, rows):
        super().__init__()
        self.full_path = full_path
        self.file_name = file_name
        self.app_title = app_title
        self.rows = rows

    def date_extracted_from_file_name(self):
        match = re.search(r"\d{1,2}[\.|-]\d{1,2}[\.|-]\d{2,4}", self.full_path)

        if match:
            return match.group(0)
        return ""

    def subject_id_extracted_from_file_path(self):
        path_parts = self.full_path.split(os.sep)
        if len(path_parts) > 3:
            return path_parts[-3]
        else:
            return ""

    def to_csv_row(self):
        return [
            self.full_path,
            self.file_name,
            self.subject_id_extracted_from_file_path(),
            self.date_extracted_from_file_name(),
            self.app_title,
            *self.rows,
        ]


class ScreenshotApp(QWidget):
    """
    Can adjust length_dimension for best visibility on your computer
    """

    length_dimension = 800

    def __init__(self):
        super().__init__()
        self.time_mapping = None
        self.extracted_text_label = None
        self.extracted_total_label = None
        self.mode = "Battery"
        self.graph_image_label = None
        self.snap_to_grid_checkbox = None
        self.auto_process_images_checkbox = None
        self.remove_duplicates_automatically_checkbox = None
        self.skip_daily_usage_checkbox = None
        self.magnifier_label = None
        self.next_button = None
        self.previous_button = None
        self.skip_button = None
        self.time_label = None
        self.time_slider = None
        self.extracted_text_edit = None
        self.image_name_line_edit = None
        self.cropped_image_label = None
        self.image_label = None
        self.instruction_label = None
        self.folder_button = None
        self.images = []
        self.current_image_index = 0
        self.click_count = 0
        self.coordinates = []
        self.folder_name = ""

        self.screen_scale_factor = self.calculate_scale_factor()
        self.init_ui()

        self.current_row = None
        self.last_row = None
        self.graph_issue = None
        self.title_issue = None
        self.total_issue = None
        self.invalid_title_list = ["", " ", None]

    def calculate_scale_factor(self) -> float:
        screen_geo = QDesktopWidget().screenGeometry()
        min_dimension = min(screen_geo.width(), screen_geo.height())
        return min_dimension / 1080

    def init_ui(self):
        self.setWindowTitle("Screenshot Slideshow")
        base_font_size = int(14 * self.screen_scale_factor)
        self.setStyleSheet(f"QWidget {{ font-size: {base_font_size}px; }} QPushButton {{ font-weight: bold; }}")
        layout = QVBoxLayout(self)

        btn_battery = QPushButton("Select Folder of Battery Images")
        btn_battery.clicked.connect(lambda: self.open_folder("Battery"))
        btn_battery.setMinimumHeight(int(40 * self.screen_scale_factor))

        btn_screen_time = QPushButton("Select Folder of Screen Time Images")
        btn_screen_time.clicked.connect(lambda: self.open_folder("ScreenTime"))
        btn_screen_time.setMinimumHeight(int(40 * self.screen_scale_factor))

        layout.addWidget(btn_battery)
        layout.addWidget(btn_screen_time)

        self.instruction_label = QLabel(
            "Click Next/Save if the graphs match, otherwise click the upper left corner of the graph in the left image.",
        )
        instruction_font = QFont()
        instruction_font.setPointSize(int(base_font_size * 1.1))
        instruction_font.setBold(True)
        self.instruction_label.setFont(instruction_font)
        self.instruction_label.setStyleSheet("background-color:rgb(255,255,150); padding: 10px;")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setFixedHeight(int(60 * self.screen_scale_factor))
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)
        self.instruction_label.hide()

        main_image_layout = QHBoxLayout()
        layout.addLayout(main_image_layout)

        original_image_layout = QVBoxLayout()

        self.image_label = MagnifyingLabel(self)
        self.image_label.setCursor(Qt.CrossCursor)
        self.image_label.setAlignment(Qt.AlignTop)
        original_image_layout.addWidget(self.image_label)
        main_image_layout.addLayout(original_image_layout)

        self.magnifier_label = QLabel(self)
        magnifier_size = int(100 * self.screen_scale_factor)
        self.magnifier_label.resize(magnifier_size, magnifier_size)
        self.magnifier_label.setFrameShape(QFrame.StyledPanel)
        self.magnifier_label.setStyleSheet("background-color: white; border: 2px solid gray;")
        self.magnifier_label.raise_()
        self.magnifier_label.setVisible(True)

        cropped_image_layout = QVBoxLayout()

        self.cropped_image_label = QLabel("No cropped image loaded.")
        self.cropped_image_label.setAlignment(Qt.AlignCenter)
        cropped_image_layout.addWidget(self.cropped_image_label)

        self.graph_image_label = QLabel("No graph extracted.")
        self.graph_image_label.setAlignment(Qt.AlignCenter)
        cropped_image_layout.addWidget(self.graph_image_label)

        cropped_image_layout.setAlignment(Qt.AlignCenter)
        main_image_layout.addLayout(cropped_image_layout)

        text_fields_layout = QVBoxLayout()
        text_fields_layout.setAlignment(Qt.AlignCenter)
        text_fields_layout.setSpacing(10)

        image_name_label = QLabel("Image Name:")
        image_name_label.setFont(instruction_font)
        text_fields_layout.addWidget(image_name_label)

        self.image_name_line_edit = QLineEdit("Image_Name_Placeholder.png")
        self.image_name_line_edit.setMinimumHeight(int(30 * self.screen_scale_factor))
        text_fields_layout.addWidget(self.image_name_line_edit)

        self.extracted_text_label = QLabel("Extracted Title/App Name:")
        self.extracted_text_label.setFont(instruction_font)
        text_fields_layout.addWidget(self.extracted_text_label)

        self.extracted_text_edit = QLineEdit("")
        self.extracted_text_edit.setMinimumHeight(int(30 * self.screen_scale_factor))
        self.extracted_text_edit.textEdited.connect(self.check_title)
        text_fields_layout.addWidget(self.extracted_text_edit)

        self.extracted_total_label = QLabel("Extracted Total: ")
        total_font = QFont()
        total_font.setPointSize(int(base_font_size * 1.2))
        total_font.setBold(True)
        self.extracted_total_label.setFont(total_font)
        self.extracted_total_label.setStyleSheet("color: #0066CC; padding: 5px;")
        text_fields_layout.addWidget(self.extracted_total_label)
        self.extracted_total_label.hide()

        self.extracted_total_image = QLabel("No total image")
        self.extracted_total_image.setAlignment(Qt.AlignCenter)
        self.extracted_total_image.setMinimumHeight(int(60 * self.screen_scale_factor))
        self.extracted_total_image.setFrameShape(QFrame.Box)
        self.extracted_total_image.setStyleSheet("border: 2px solid #0066CC; padding: 5px;")
        text_fields_layout.addWidget(self.extracted_total_image)
        self.extracted_total_image.hide()

        self.snap_to_grid_checkbox = QCheckBox("Automatically snap to grid", self)
        self.snap_to_grid_checkbox.setMinimumHeight(int(30 * self.screen_scale_factor))
        text_fields_layout.addWidget(self.snap_to_grid_checkbox)

        self.auto_process_images_checkbox = QCheckBox("Automatically process images (minimized, until an error occurs)", self)
        self.auto_process_images_checkbox.setMinimumHeight(int(30 * self.screen_scale_factor))
        text_fields_layout.addWidget(self.auto_process_images_checkbox)
        self.auto_process_images_checkbox.hide()

        self.remove_duplicates_automatically_checkbox = QCheckBox("Remove duplicates in csv output, keeping last saved (based on image path)", self)
        self.remove_duplicates_automatically_checkbox.setMinimumHeight(int(30 * self.screen_scale_factor))
        text_fields_layout.addWidget(self.remove_duplicates_automatically_checkbox)

        self.skip_daily_usage_checkbox = QCheckBox("Skip daily usage images", self)
        self.skip_daily_usage_checkbox.setToolTip("If checked, images with 'Daily Total' title will be automatically skipped")
        self.skip_daily_usage_checkbox.setMinimumHeight(int(30 * self.screen_scale_factor))
        text_fields_layout.addWidget(self.skip_daily_usage_checkbox)
        self.skip_daily_usage_checkbox.hide()

        self.skip_button = QPushButton("Skip (no saving)")
        self.skip_button.clicked.connect(self.skip_current_image)
        self.skip_button.setMinimumHeight(int(40 * self.screen_scale_factor))
        text_fields_layout.addWidget(self.skip_button)

        main_image_layout.addLayout(text_fields_layout)
        main_image_layout.setStretch(0, 1)
        main_image_layout.setStretch(1, 1)
        main_image_layout.setStretch(2, 1)

        self.time_mapping = {
            0: "Midnight",
            1: "3 AM",
            2: "6 AM",
            3: "9 AM",
            4: "12 PM",
            5: "3 PM",
            6: "6 PM",
            7: "9 PM",
        }
        slider_layout = QHBoxLayout()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.time_mapping) - 1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setMinimumHeight(int(40 * self.screen_scale_factor))
        slider_layout.addWidget(self.slider)

        layout.addLayout(slider_layout)
        self.slider.hide()

        label_layout = QHBoxLayout()
        label_layout.addStretch()
        self.time_label = QLabel("First time displayed in screenshot: Midnight")
        self.time_label.setFont(instruction_font)
        self.time_label.setAlignment(Qt.AlignCenter)
        label_layout.addWidget(self.time_label)
        label_layout.addStretch()
        layout.addLayout(label_layout)
        self.time_label.hide()

        nav_layout = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.show_previous_image)
        self.previous_button.setMinimumHeight(int(50 * self.screen_scale_factor))
        nav_layout.addWidget(self.previous_button)

        self.next_button = QPushButton("Next/Save")
        self.next_button.clicked.connect(self.show_next_image_manual_button_press)
        self.next_button.setMinimumHeight(int(50 * self.screen_scale_factor))
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)

        self.calculate_and_set_window_size()

        self.slider.valueChanged.connect(self.update_time_label)
        self.slider.hide()

        self.calculate_and_set_window_size()

    def calculate_and_set_window_size(self):
        """Calculate and set the window size based on screen dimensions"""
        screen_geo = QDesktopWidget().screenGeometry()
        width = int(screen_geo.width() * 0.85)
        height = int(screen_geo.height() * 0.85)

        min_width = 1200
        min_height = 800

        width = max(width, min_width)
        height = max(height, min_height)

        self.resize(width, height)

        self.length_dimension = int(min(width, height) * 0.7)

    def update_interface(self):
        if not self.graph_issue and not self.title_issue and not self.total_issue:
            self.instruction_label.setText(
                "Click Next/Save if ALL the graphs match (including the one on the left), otherwise click the upper left corner of the graph in the left image to reselect."
            )
            self.instruction_label.setStyleSheet("background-color:rgb(255,255,150)")
        else:
            if self.title_issue:
                self.instruction_label.setText(
                    "A title issue occurred. Please enter the title correctly and click Next/Save when finished.\nIf the title is for the daily view, please enter 'Daily Total'."
                )
                self.instruction_label.setStyleSheet("background-color:rgb(255,165,0)")
            elif self.total_issue:
                self.instruction_label.setText(
                    "A total time discrepancy issue occurred. The extracted total was either not found, was overestimated by the calculated total, or differs from the calculated total by more than 5 minutes. You can either proceed or reselect the graph for better accuracy."
                )
                self.instruction_label.setStyleSheet("background-color:rgb(255,179,179)")
            elif self.graph_issue:
                self.instruction_label.setText(
                    "A graph detection issue occurred. To start reselecting, first click the upper left corner of the graph in the left image."
                )
                self.instruction_label.setStyleSheet("background-color:rgb(255,0,0)")
            self.setWindowState(self.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
            self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
            self.show()
            self.raise_()
            self.activateWindow()

    def adjust_ui_for_image_type(self, image_type):
        if image_type == "Battery":
            self.extracted_text_label.setText("Date of first displayed time:")
            self.slider.show()
            self.time_label.hide()
            self.instruction_label.hide()
            self.extracted_total_label.hide()
            self.extracted_total_image.hide()
            self.auto_process_images_checkbox.hide()
            self.remove_duplicates_automatically_checkbox.hide()
            self.skip_daily_usage_checkbox.hide()
        elif image_type == "ScreenTime":
            self.extracted_text_label.setText("Extracted Title/App Name:")
            self.slider.hide()
            self.time_label.hide()
            self.instruction_label.show()
            self.extracted_total_label.show()
            self.extracted_total_image.show()
            self.auto_process_images_checkbox.show()
            self.remove_duplicates_automatically_checkbox.show()
            self.skip_daily_usage_checkbox.show()

    def capture_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.coordinates.append((x, y))
        self.click_count += 1

        if self.click_count == 1:
            self.instruction_label.setText("Now click the bottom right corner of the graph in the left image to finish the selection.")
            self.instruction_label.setStyleSheet("background-color:rgb(0,255,0)")
        elif self.click_count == 2:
            self.process_coordinates(self.coordinates[0], self.coordinates[1])
            self.coordinates = []
            self.click_count = 0

            self.update_interface()

    def process_coordinates(self, upper_left, lower_right):
        image_path = self.images[self.current_image_index]
        original_pixmap = QPixmap(image_path)

        original_width = original_pixmap.width()
        original_height = original_pixmap.height()

        scalar = self.length_dimension / original_height

        display_width = original_width * scalar
        display_height = original_height * scalar

        scale_x = original_width / display_width
        scale_y = original_height / display_height

        true_upper_left = (int(upper_left[0] * scale_x), int(upper_left[1] * scale_y))
        true_lower_right = (
            int(lower_right[0] * scale_x),
            int(lower_right[1] * scale_y),
        )

        print(f"True Upper Left: {true_upper_left}, True Lower Right: {true_lower_right}")
        print(f"Orig width: {original_width}, Orig height: {original_height}")
        print(f"Display width: {display_width}, Display height: {display_height}")

        print("Processing image from clicks...")
        processed_image_path, graph_image_path, row, title, total, total_image_path = process_image_with_grid(
            image_path,
            true_upper_left,
            true_lower_right,
            self.mode == "Battery",  # bool
            self.snap_to_grid_checkbox.isChecked(),
        )

        self.update_(title, total, row, processed_image_path, graph_image_path, total_image_path)

    def update_time_label(self, value):
        selected_time = self.time_mapping.get(value, "Midnight")
        self.time_label.setText(f"First time displayed in screenshot: {selected_time}")

    def open_folder(self, selection_type):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.folder_name = folder_path
        self.mode = selection_type
        self.adjust_ui_for_image_type(selection_type)
        if folder_path:
            self.load_images(folder_path)
            if self.images:
                self.show_image(0)
            else:
                QMessageBox.information(self, "No Images Found", "No images found in the folder.")

    def load_images(self, folder_path):
        self.images = []
        ignore_list = ["Do Not Use", "debug"]
        for root, _dirs, files in os.walk(folder_path):
            for filename in files:
                if all(ignored not in os.path.join(root, filename) for ignored in ignore_list):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".jfif")):
                        self.images.append(os.path.join(root, filename))
        if not self.images:
            self.image_label.setText("No image loaded.")
            self.cropped_image_label.setText("No cropped image loaded.")
            self.image_name_line_edit.setText("Image name will appear here.")

    def check_title(self):
        """
        Checks if the entered title is valid and updates the title_issue accordingly.
        """
        title = self.extracted_text_edit.text()
        if title in self.invalid_title_list:
            self.title_issue = True
        else:
            self.title_issue = False
            self.update_interface()

    def parse_hours_and_minutes_time_string(self, total_str):
        """
        Parse the extracted total time string into minutes.
        Handles various formats like "2h 30m", "45m", etc.
        Returns -1 if parsing fails.
        """
        if not total_str or total_str == "N/A":
            return -1

        # Clean the string
        cleaned_str = re.sub(r"[^\w\s]", " ", total_str).lower()

        total_minutes = 0

        # Look for hours pattern (e.g., "2h", "2 hours", etc.)
        hours_match = re.search(r"(\d+)\s*h", cleaned_str)
        if hours_match:
            total_minutes += int(hours_match.group(1)) * 60

        # Look for minutes pattern (e.g., "30m", "30 min", etc.)
        minutes_match = re.search(r"(\d+)\s*m", cleaned_str)
        if minutes_match:
            total_minutes += int(minutes_match.group(1))

        return total_minutes

    def update_(self, title, total, row, processed_image_path, graph_image_path, total_image_path=None) -> None:
        self.extracted_text_edit.setText(title)
        self.extracted_total_label.setText(f"Extracted Total: {total}")

        # Display the total image if available
        if total_image_path and os.path.exists(total_image_path):
            total_pixmap = QPixmap(total_image_path)
            # Scale the image to a reasonable size
            scaled_pixmap = total_pixmap.scaled(300, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.extracted_total_image.setPixmap(scaled_pixmap)
        else:
            self.extracted_total_image.setText("No total image available")

        # Compare extracted and calculated totals for ScreenTime mode
        if self.mode == "ScreenTime" and total != "N/A":
            # Parse extracted total
            extracted_total_minutes = self.parse_hours_and_minutes_time_string(total)

            # Get calculated total
            calculated_total_minutes = row[-1] if row else 0

            # Set the total_issue flag if difference exceeds 5 minutes
            if (calculated_total_minutes > extracted_total_minutes) or (abs(extracted_total_minutes - calculated_total_minutes) > 5):
                self.total_issue = True
            else:
                self.total_issue = False
        else:
            self.total_issue = False

        if self.mode == "Battery":
            self.current_row = BatteryRow(
                full_path=self.images[self.current_image_index],
                file_name=self.image_name_line_edit.text(),
                date_from_image=self.extracted_text_edit.text(),
                time_from_ui=self.time_mapping[self.slider.value()],
                rows=row,
            )
        elif self.mode == "ScreenTime":
            self.current_row = ScreenTimeRow(
                full_path=self.images[self.current_image_index],
                file_name=self.image_name_line_edit.text(),
                app_title=self.extracted_text_edit.text(),
                rows=row,
            )

        if processed_image_path:
            processed_pixmap = QPixmap(processed_image_path)

            self.cropped_image_label.setPixmap(
                processed_pixmap.scaled(
                    self.length_dimension,
                    self.length_dimension,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

            self.graph_issue = False
            self.update_interface()

        else:
            self.cropped_image_label.setText("No cropped image could be loaded from the selection.")
            self.graph_issue = True
            self.check_title()
            self.update_interface()

        if graph_image_path:
            processed_graph_pixmap = QPixmap(graph_image_path)

            self.graph_image_label.setPixmap(
                processed_graph_pixmap.scaled(
                    self.length_dimension,
                    self.length_dimension,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

            self.graph_issue = False
            self.check_title()
            self.update_interface()

        else:
            self.graph_image_label.setText("No graph could be extracted from the selection.")
            self.graph_issue = True
            self.update_interface()

        if processed_image_path and graph_image_path:
            processed_image = cv2.imread(processed_image_path)
            graph_image = cv2.imread(graph_image_path)

            compare_blue_in_images(image1=processed_image, image2=graph_image)

            if mse_between_loaded_images(processed_image, graph_image) > 100:
                check_folder = "./debug/check/"

                os.makedirs(check_folder, exist_ok=True)
                try:
                    combined_image = cv2.vconcat([processed_image, graph_image])
                    original_screenshot_image = cv2.imread(self.images[self.current_image_index])
                    combined_image = hconcat_resize([original_screenshot_image, combined_image])

                    cv2.imwrite(
                        f"{check_folder}/{os.path.basename(processed_image_path)}_combined.jpg",
                        combined_image,
                    )
                except Exception:
                    print(traceback.format_exc())

    def show_image(self, index):
        # Reset all issue flags when loading a new image
        self.total_issue = False
        self.graph_issue = False
        self.title_issue = False

        if 0 <= index < len(self.images):
            image_path = self.images[index]
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.length_dimension,
                    self.length_dimension,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            self.image_name_line_edit.setText(os.path.basename(image_path))
            self.current_image_index = index

            try:
                processed_image_path, graph_image_path, row, title, total, total_image_path = process_image(
                    image_path,
                    self.mode == "Battery",
                    self.snap_to_grid_checkbox.isChecked(),
                )

                self.update_(title, total, row, processed_image_path, graph_image_path, total_image_path)

                if (
                    self.mode == "ScreenTime"
                    and self.skip_daily_usage_checkbox.isChecked()
                    and title == "Daily Total"
                    and not self.graph_issue
                    and not self.title_issue
                    and not self.total_issue
                ):
                    print(f"Skipping Daily Total image: {image_path}")
                    self.skip_current_image()
                    return

                """These lines are what causes the app to loop automatically over images until an error occurs"""
                if self.auto_process_images_checkbox.isChecked() and not self.graph_issue and not self.title_issue and not self.total_issue:
                    self.showMinimized()
                    self.show_next_image()

            except Exception as e:
                print(f"Error during image loading or processing: {traceback.format_exc()}")
                self.graph_issue = True
                self.title_issue = False
                self.total_issue = False
                self.cropped_image_label.setText("No cropped image loaded.")
                self.graph_image_label.setText("No graph extracted")
                self.image_name_line_edit.setText("No image available.")
                self.update_interface()

        else:
            self.image_label.setText("No image loaded.")
            self.cropped_image_label.setText("No cropped image loaded.")
            self.graph_image_label.setText("No graph extracted")
            self.image_name_line_edit.setText("No image available.")

    def skip_current_image(self):
        if self.current_image_index + 1 < len(self.images):
            self.show_image(self.current_image_index + 1)

    def show_next_image_manual_button_press(self):
        if self.total_issue:
            self.total_issue = False
        self.show_next_image()

    def show_next_image(self):
        if (
            self.current_row
            and not self.graph_issue
            and not self.title_issue
            and not (self.total_issue and self.auto_process_images_checkbox.isChecked())
        ):
            self.save_current_row()

            if self.auto_process_images_checkbox.isChecked():
                self.showMinimized()

            if self.current_image_index + 1 < len(self.images):
                self.show_image(self.current_image_index + 1)
                if self.current_image_index + 1 == len(self.images):
                    self.final_row = self.current_row

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)

    def save_current_row(self):
        if self.current_row:
            csv_path = (
                os.path.dirname(sys.argv[0])
                + "/output/"
                + os.path.basename(self.images[self.current_image_index].split("\\")[-3])
                + " Arcascope Output.csv"
            )

            if self.mode == "Battery":
                headers = BatteryRow.headers()
                self.current_row.date_from_image = self.extracted_text_edit.text()
                self.current_row.time_from_ui = self.time_mapping[self.slider.value()]
            elif self.mode == "ScreenTime":
                headers = ScreenTimeRow.headers()
                self.current_row.app_title = self.extracted_text_edit.text()
            else:
                raise ValueError(f"Invalid mode: {self.mode}")

            # Check if the CSV file exists
            if not os.path.exists(csv_path):
                df = pd.DataFrame(columns=headers)
                df.to_csv(csv_path, index=False)
                print("CSV file created with headers.")

        # Create a new row to check
        new_row_to_check = pd.DataFrame([self.current_row.to_csv_row()], columns=headers).fillna("")
        old_df = pd.read_csv(csv_path).fillna("")

        # Ensure the columns match
        new_row_to_check = new_row_to_check[old_df.columns]

        # Concatenate old dataframe with new row to check
        combined_df = pd.concat([old_df, new_row_to_check], axis=0)

        if self.remove_duplicates_automatically_checkbox.isChecked():
            # Define the columns to check for duplicates
            duplicate_columns = ["Full path"]

            # Find duplicates based on specific columns
            duplicates = combined_df.duplicated(subset=duplicate_columns, keep="last")
            num_duplicates = duplicates.sum()

            # Remove duplicates based on specific columns while preserving the order, keeping the last updated entry
            combined_df = combined_df.drop_duplicates(subset=duplicate_columns, keep="last")

            print(f"{num_duplicates} duplicate(s) based on full image path removed, kept the last updated entries.")

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Save the updated dataframe back to the CSV
        combined_df.to_csv(csv_path, index=False)

        print(f"CSV file {csv_path} updated")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = ScreenshotApp()
    ex.show()
    sys.exit(app.exec_())
