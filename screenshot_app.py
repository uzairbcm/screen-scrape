from __future__ import annotations

import os
import re
import sys
import traceback
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import pandas as pd
import pytesseract
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
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

from image_processor import compare_blue_in_images, hconcat_resize, mse_between_loaded_images, process_image, process_image_with_grid
from issue import (
    GraphDetectionIssue,
    TitleMissingIssue,
    TotalIssue,
    TotalNotFoundIssue,
    TotalOverestimationLargeIssue,
    TotalOverestimationSmallIssue,
    TotalParseErrorIssue,
    TotalUnderestimationLargeIssue,
    TotalUnderestimationSmallIssue,
)
from issue_manager import IssueManager
from magnifying_label import MagnifyingLabel

if TYPE_CHECKING:
    from collections.abc import Sequence

# Note: May need to be imported on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class ImageType(StrEnum):
    BATTERY = "Battery"
    SCREEN_TIME = "Screen Time"


class BaseRow(ABC):
    """Abstract base class for row data used in the screen scrape application."""

    @classmethod
    @abstractmethod
    def headers(cls) -> list[str]:
        """Return headers for CSV output."""

    def __init__(self, full_path: Path | str, file_name: Path | str, rows: Sequence[float]) -> None:
        """Initialize the base row with common attributes."""
        self.full_path = full_path
        self.file_name = file_name
        self.rows = rows

    def date_extracted_from_file_name(self) -> str:
        """Extract date from file path using regex."""
        date_match = re.search(r"\d{1,2}[\.|-]\d{1,2}[\.|-]\d{2,4}", str(self.full_path))
        if date_match:
            return date_match.group(0)
        return ""

    @abstractmethod
    def subject_id_extracted_from_file_path(self) -> str:
        """Extract subject ID from file path."""

    def get_common_row_fields(self) -> list[Any]:
        """Get the common fields for CSV rows."""
        return [
            self.full_path,
            self.file_name,
            self.subject_id_extracted_from_file_path(),
            self.date_extracted_from_file_name(),
        ]

    @abstractmethod
    def get_specific_row_fields(self) -> list[Any]:
        """Get class-specific fields for CSV rows."""

    def to_csv_row(self) -> list[Any]:
        """Convert object to CSV row data using template method pattern."""
        return self.get_common_row_fields() + self.get_specific_row_fields() + list(self.rows)


class BatteryRow(BaseRow):
    @classmethod
    def headers(cls) -> list[str]:
        return ["Full path", "File name", "ID", "Date from file name", "Date from image", "Time from image"] + [str(i) for i in range(24)] + ["Total"]

    def __init__(self, full_path: Path | str, file_name: Path | str, date_from_image: str, time_from_ui: str, rows: Sequence[float]) -> None:
        super().__init__(full_path, file_name, rows)
        self.date_from_image = date_from_image
        self.time_from_ui = time_from_ui

    def subject_id_extracted_from_file_path(self) -> str:
        return Path(self.full_path).parent.name

    def get_specific_row_fields(self) -> list[Any]:
        return [
            self.date_from_image,
            self.time_from_ui,
        ]


class ScreenTimeRow(BaseRow):
    @classmethod
    def headers(cls) -> list[str]:
        return ["Full path", "Filename", "ID", "Date", "App Title"] + [str(i) for i in range(24)] + ["Total"]

    def __init__(self, full_path: Path | str, file_name: Path | str, app_title: str, rows: Sequence[float]) -> None:
        super().__init__(full_path, file_name, rows)
        self.app_title = app_title

    def subject_id_extracted_from_file_path(self) -> str:
        path_parts = Path(self.full_path).parts
        if len(path_parts) > 3:
            return path_parts[-3]
        else:
            return ""

    def get_specific_row_fields(self) -> list[Any]:
        return [
            self.app_title,
        ]


class ScreenshotApp(QWidget):
    """
    Can adjust length_dimension for best visibility on your computer
    """

    length_dimension = 800

    def __init__(self) -> None:
        super().__init__()
        # self.time_mapping = None
        # self.extracted_text_label = None
        # self.extracted_total_label = None
        self.image_mode = ImageType.BATTERY
        # self.graph_image_label = None
        # self.snap_to_grid_checkbox = None
        # self.auto_process_images_checkbox = None
        # self.remove_duplicates_automatically_checkbox = None
        # self.skip_daily_usage_checkbox = None
        # self.magnifier_label = None
        # self.next_button = None
        # self.previous_button = None
        # self.skip_button = None
        # self.time_label = None
        # self.time_slider = None
        # self.extracted_text_edit = None
        # self.image_name_line_edit = None
        # self.cropped_image_label = None
        # self.image_label = None
        # self.instruction_label = None
        # self.folder_button = None
        self.images = []
        self.current_image_index = 0
        self.current_row = BatteryRow("", "", "", "", [])
        self.click_count = 0
        self.coordinates = []
        self.folder_name = ""

        self.screen_scale_factor = self.calculate_scale_factor()
        self.init_ui()

        self.issue_manager = IssueManager()
        self.issue_manager.register_observer(self.update_interface)

        self.invalid_title_list = ["", " ", None]

    @property
    def screen_geometry(self) -> QtCore.QRect:
        screen_geo = None
        if QApplication.primaryScreen() is not None:
            screen = QApplication.primaryScreen()
            if screen is not None:
                screen_geo = screen.geometry()

        if screen_geo is None:
            screen_geo = QtCore.QRect(0, 0, 1920, 1080)

        return screen_geo

    def calculate_scale_factor(self) -> float:
        min_dimension = min(self.screen_geometry.width(), self.screen_geometry.height())
        return min_dimension / 1080

    def calculate_and_set_window_size(self) -> None:
        """Calculate and set the window size based on screen dimensions"""
        width = int(self.screen_geometry.width() * 0.85)
        height = int(self.screen_geometry.height() * 0.85)

        min_width = 1200
        min_height = 800

        width = max(width, min_width)
        height = max(height, min_height)

        self.resize(width, height)

        self.length_dimension = int(min(width, height) * 0.7)

    def init_ui(self) -> None:
        self.setWindowTitle("Screenshot Slideshow")
        base_font_size = int(14 * self.screen_scale_factor)
        self.setStyleSheet(f"QWidget {{ font-size: {base_font_size}px; }} QPushButton {{ font-weight: bold; }}")
        layout = QVBoxLayout(self)

        btn_battery = QPushButton("Select Folder of Battery Images")
        btn_battery.clicked.connect(lambda: self.open_folder(ImageType.BATTERY))
        btn_battery.setMinimumHeight(int(40 * self.screen_scale_factor))

        btn_screen_time = QPushButton("Select Folder of Screen Time Images")
        btn_screen_time.clicked.connect(lambda: self.open_folder(ImageType.SCREEN_TIME))
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
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setFixedHeight(int(60 * self.screen_scale_factor))
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label)
        self.instruction_label.hide()

        main_image_layout = QHBoxLayout()
        layout.addLayout(main_image_layout)

        original_image_layout = QVBoxLayout()

        self.image_label = MagnifyingLabel(self)
        self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        original_image_layout.addWidget(self.image_label)
        main_image_layout.addLayout(original_image_layout)

        self.magnifier_label = QLabel(self)
        magnifier_size = int(100 * self.screen_scale_factor)
        self.magnifier_label.resize(magnifier_size, magnifier_size)
        self.magnifier_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.magnifier_label.setStyleSheet("background-color: white; border: 2px solid gray;")
        self.magnifier_label.raise_()
        self.magnifier_label.setVisible(True)

        cropped_image_layout = QVBoxLayout()

        self.cropped_image_label = QLabel("No cropped image loaded.")
        self.cropped_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cropped_image_layout.addWidget(self.cropped_image_label)

        self.graph_image_label = QLabel("No graph extracted.")
        self.graph_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cropped_image_layout.addWidget(self.graph_image_label)

        cropped_image_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_image_layout.addLayout(cropped_image_layout)

        text_fields_layout = QVBoxLayout()
        text_fields_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        self.extracted_total_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.extracted_total_image.setMinimumHeight(int(60 * self.screen_scale_factor))
        self.extracted_total_image.setFrameShape(QFrame.Shape.Box)
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

        self.slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.time_mapping) - 1)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setMinimumHeight(int(40 * self.screen_scale_factor))
        slider_layout.addWidget(self.slider)

        layout.addLayout(slider_layout)
        self.slider.hide()

        label_layout = QHBoxLayout()
        label_layout.addStretch()
        self.time_label = QLabel("First time displayed in screenshot: Midnight")
        self.time_label.setFont(instruction_font)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

    def adjust_ui_for_image_type(self, image_type: ImageType) -> None:
        if image_type == ImageType.BATTERY:
            self.extracted_text_label.setText("Date of first displayed time:")
            self.slider.show()
            self.time_label.hide()
            self.instruction_label.hide()
            self.extracted_total_label.hide()
            self.extracted_total_image.hide()
            self.auto_process_images_checkbox.hide()
            self.remove_duplicates_automatically_checkbox.hide()
            self.skip_daily_usage_checkbox.hide()
        elif image_type == ImageType.SCREEN_TIME:
            self.extracted_text_label.setText("Extracted Title/App Name:")
            self.slider.hide()
            self.time_label.hide()
            self.instruction_label.show()
            self.extracted_total_label.show()
            self.extracted_total_image.show()
            self.auto_process_images_checkbox.show()
            self.remove_duplicates_automatically_checkbox.show()
            self.skip_daily_usage_checkbox.show()

    def capture_click(self, event: QMouseEvent) -> None:
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

    def update_time_label(self, value: int) -> None:
        selected_time = self.time_mapping.get(value, "Midnight")
        self.time_label.setText(f"First time displayed in screenshot: {selected_time}")

    def open_folder(self, selection_type) -> None:
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.folder_name = folder_path
        self.image_mode = selection_type
        self.adjust_ui_for_image_type(selection_type)
        if folder_path:
            self.load_images(folder_path)
            if self.images:
                self.show_image(0)
            else:
                QMessageBox.information(self, "No Images Found", "No images found in the folder.")

    def load_images(self, folder_path: Path | str) -> None:
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

    def skip_current_image(self) -> None:
        if self.current_image_index + 1 < len(self.images):
            self.show_image(self.current_image_index + 1)

    def show_previous_image(self) -> None:
        if self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)

    def save_current_row(self) -> None:
        if self.current_row:
            csv_path = Path(sys.argv[0]).parent / "output" / Path(self.images[self.current_image_index]).parents[2].name / " Arcascope Output.csv"

            if self.image_mode == ImageType.BATTERY:
                if not isinstance(self.current_row, BatteryRow):
                    msg = f"Expected BatteryRow object for Battery mode, got {type(self.current_row)}"
                    raise ValueError(msg)

                headers = BatteryRow.headers()
                self.current_row.date_from_image = self.extracted_text_edit.text()
                self.current_row.time_from_ui = self.time_mapping[self.slider.value()]

            elif self.image_mode == ImageType.SCREEN_TIME:
                if not isinstance(self.current_row, ScreenTimeRow):
                    msg = f"Expected ScreenTimeRow object for Screen Time mode, got {type(self.current_row)}"
                    raise ValueError(msg)

                headers = ScreenTimeRow.headers()
                self.current_row.app_title = self.extracted_text_edit.text()

            else:
                msg = f"Invalid image mode: {self.image_mode}"
                raise ValueError(msg)

            if csv_path.exists():
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

            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the updated dataframe back to the CSV
            combined_df.to_csv(csv_path, index=False)

            print(f"CSV file {csv_path} updated")

    def check_title(self) -> None:
        """Checks if the entered title is valid and updates issues accordingly."""
        title = self.extracted_text_edit.text()
        if title in self.invalid_title_list:
            self.issue_manager.add_issue(TitleMissingIssue("Title is missing or invalid. Please enter a valid title."))
        else:
            self.issue_manager.remove_issues_of_class(TitleMissingIssue)

    def parse_hours_and_minutes_time_string(self, total_str: str):
        """
        Parse the extracted total time string into minutes.
        Handles various formats like "2h 30m", "45m", etc.
        Returns -1 if parsing fails.
        """
        if not total_str or total_str == "N/A":
            return -1

        # Clean the string
        cleaned_str = re.sub(r"[^\w\s]", " ", total_str).lower()

        # Check if the string contains only hours and minutes
        valid_format = bool(re.match(r"^\s*(\d+\s*h\s+)?\d+\s*m\s*$", cleaned_str))

        total_minutes = 0

        # Look for hours pattern (e.g., "2h", "2 hours", etc.)
        hours_match = re.search(r"(\d+)\s*h", cleaned_str)
        if hours_match:
            total_minutes += int(hours_match.group(1)) * 60

        # Look for minutes pattern (e.g., "30m", "30 min", etc.)
        minutes_match = re.search(r"(\d+)\s*m", cleaned_str)
        if minutes_match:
            total_minutes += int(minutes_match.group(1))

        # If we found a valid time but it wasn't in the expected format
        if total_minutes > 0 and not valid_format:
            self.issue_manager.add_issue(
                TotalParseErrorIssue(f"The extracted total '{total_str}' is not in the expected format (e.g., '2h 30m' or '45m').")
            )

        return total_minutes

    def update_(
        self,
        title: str,
        total: str,
        row: list[int],
        processed_image_path: Path | str,
        graph_image_path: Path | str,
        total_image_path: Path | str | None = None,
    ) -> None:
        # Reset issues
        self.issue_manager.remove_all_issues()

        self.extracted_text_edit.setText(title)
        self.extracted_total_label.setText(f"Extracted Total: {total}")

        # Display the total image if available
        if total_image_path and Path(total_image_path).exists():
            total_pixmap = QPixmap(total_image_path)
            # Scale the image to a reasonable size
            scaled_pixmap = total_pixmap.scaled(300, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.extracted_total_image.setPixmap(scaled_pixmap)
        else:
            self.extracted_total_image.setText("No total image available")

        # Compare extracted and calculated totals for ScreenTime mode
        if self.image_mode == ImageType.SCREEN_TIME and total != "N/A":
            # Parse extracted total
            extracted_total_minutes = self.parse_hours_and_minutes_time_string(total)

            # Get calculated total
            calculated_total_minutes = row[-1] if row else 0

            if extracted_total_minutes == -1:
                self.issue_manager.add_issue(TotalNotFoundIssue("No total time could be extracted from the image."))
            else:
                diff = extracted_total_minutes - calculated_total_minutes

                if diff < 0:  # Underestimation
                    if abs(diff) < 2:
                        self.issue_manager.add_issue(TotalUnderestimationSmallIssue(f"The extracted total underestimates by {abs(diff)} minute(s)."))
                    else:
                        self.issue_manager.add_issue(
                            TotalUnderestimationLargeIssue(f"The extracted total significantly underestimates by {abs(diff)} minute(s).")
                        )
                elif diff > 0:  # Overestimation
                    if diff == 1:
                        self.issue_manager.add_issue(TotalOverestimationSmallIssue("The extracted total overestimates by 1 minute."))
                    else:
                        self.issue_manager.add_issue(TotalOverestimationLargeIssue(f"The extracted total overestimates by {diff} minutes."))

        # Check for graph issues
        if not processed_image_path:
            self.cropped_image_label.setText("No cropped image could be loaded from the selection.")
            self.issue_manager.add_issue(GraphDetectionIssue("Failed to extract a valid graph from the image."))
        else:
            processed_pixmap = QPixmap(processed_image_path)

            self.cropped_image_label.setPixmap(
                processed_pixmap.scaled(
                    self.length_dimension,
                    self.length_dimension,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        if not graph_image_path:
            self.graph_image_label.setText("No graph could be extracted from the selection.")
            self.issue_manager.add_issue(GraphDetectionIssue("Failed to extract a valid graph from the image."))
        else:
            processed_graph_pixmap = QPixmap(graph_image_path)

            self.graph_image_label.setPixmap(
                processed_graph_pixmap.scaled(
                    self.length_dimension,
                    self.length_dimension,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        # Compare the images if both exist
        if processed_image_path and graph_image_path:
            processed_image = cv2.imread(str(processed_image_path))
            graph_image = cv2.imread(str(graph_image_path))

            compare_blue_in_images(image1=processed_image, image2=graph_image)

            if mse_between_loaded_images(processed_image, graph_image) > 100:
                check_folder = "./debug/check/"

                Path(check_folder).mkdir(parents=True, exist_ok=True)
                try:
                    combined_image = cv2.vconcat([processed_image, graph_image])
                    original_screenshot_image = cv2.imread(self.images[self.current_image_index])
                    combined_image = hconcat_resize([original_screenshot_image, combined_image])

                    cv2.imwrite(
                        f"{check_folder}/{Path(processed_image_path).name}_combined.jpg",
                        combined_image,
                    )
                except Exception:
                    print(traceback.format_exc())

        # Check for title issues
        self.check_title()

        # Update the current row
        if self.image_mode == ImageType.BATTERY:
            self.current_row = BatteryRow(
                full_path=self.images[self.current_image_index],
                file_name=self.image_name_line_edit.text(),
                date_from_image=self.extracted_text_edit.text(),
                time_from_ui=self.time_mapping[self.slider.value()],
                rows=row,
            )
        elif self.image_mode == ImageType.SCREEN_TIME:
            self.current_row = ScreenTimeRow(
                full_path=self.images[self.current_image_index],
                file_name=self.image_name_line_edit.text(),
                app_title=self.extracted_text_edit.text(),
                rows=row,
            )

    def update_interface(self) -> None:
        """Update the UI based on current issues."""
        if not self.issue_manager.has_issues():
            self.instruction_label.setText(
                "Click Next/Save if ALL the graphs match (including the one on the left), otherwise click the upper left corner of the graph in the left image to reselect."
            )
            self.instruction_label.setStyleSheet("background-color:rgb(255,255,150)")
            return

        # Get the most severe issue (those that prevent continuation)
        issue = self.issue_manager.get_most_important_issue()
        if issue:
            message, style = issue.get_styled_message()
            self.instruction_label.setText(message)
            self.instruction_label.setStyleSheet(style)

        # Ensure window is visible if there are blocking issues
        if self.issue_manager.has_blocking_issues():
            self.bring_window_to_front()

    def bring_window_to_front(self) -> None:
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.show()
        self.raise_()
        self.activateWindow()

    def show_next_image_manual_button_press(self) -> None:
        self.issue_manager.remove_issues_of_class(TotalIssue)
        self.show_next_image()

    def show_next_image(self) -> None:
        if self.current_row and not self.issue_manager.has_blocking_issues():
            self.save_current_row()

            if self.auto_process_images_checkbox.isChecked():
                self.showMinimized()

            if self.current_image_index + 1 < len(self.images):
                self.show_image(self.current_image_index + 1)
                if self.current_image_index + 1 == len(self.images):
                    self.final_row = self.current_row

    def show_image(self, index: int) -> None:
        # Reset all issues when loading a new image
        self.issue_manager.remove_all_issues()

        if 0 <= index < len(self.images):
            image_path = self.images[index]
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.length_dimension,
                    self.length_dimension,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.image_name_line_edit.setText(Path(image_path).name)
            self.current_image_index = index

            try:
                processed_image_path, graph_image_path, row, title, total, total_image_path = process_image(
                    image_path,
                    self.image_mode == "Battery",
                    self.snap_to_grid_checkbox.isChecked(),
                )

                self.update_(title or "", total, row, processed_image_path, graph_image_path, total_image_path)

                if (
                    self.image_mode == ImageType.SCREEN_TIME
                    and self.skip_daily_usage_checkbox.isChecked()
                    and title == "Daily Total"
                    and not self.issue_manager.has_blocking_issues()
                ):
                    print(f"Skipping Daily Total image: {image_path}")
                    self.skip_current_image()
                    return

                """These lines are what causes the app to loop automatically over images until an error occurs"""
                if self.auto_process_images_checkbox.isChecked() and not self.issue_manager.has_blocking_issues():
                    self.showMinimized()
                    self.show_next_image()

            except Exception as e:
                print(f"Error during image loading or processing: {traceback.format_exc()}")
                self.issue_manager.add_issue(GraphDetectionIssue("Failed to process the image: " + str(e)))
                self.cropped_image_label.setText("No cropped image loaded.")
                self.graph_image_label.setText("No graph extracted")
                self.image_name_line_edit.setText("No image available.")
        else:
            self.image_label.setText("No image loaded.")
            self.cropped_image_label.setText("No cropped image loaded.")
            self.graph_image_label.setText("No graph extracted")
            self.image_name_line_edit.setText("No image available.")

    def process_coordinates(self, upper_left: tuple[int, int], lower_right: tuple[int, int]) -> None:
        image_path = self.images[self.current_image_index]
        original_pixmap = QPixmap(image_path)

        original_width = original_pixmap.width()
        original_height = original_pixmap.height()

        scalar = self.length_dimension / original_height

        display_width = original_width * scalar
        display_height = original_height * scalar

        scale_x = original_width / display_width
        scale_y = original_height / display_height

        true_upper_left = (
            int(upper_left[0] * scale_x),
            int(upper_left[1] * scale_y),
        )
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
            self.image_mode == ImageType.BATTERY,  # bool
            self.snap_to_grid_checkbox.isChecked(),
        )

        self.update_(title, total, row, processed_image_path, graph_image_path, total_image_path)


if __name__ == "__main__":
    sys.argv += ["-platform", "windows:darkmode=1"]
    app = QApplication(sys.argv)
    ex = ScreenshotApp()
    ex.show()
    sys.exit(app.exec())
