from __future__ import annotations

import contextlib
import os
import re
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import pandas as pd
import pytesseract
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QWidget,
)

from enums import ImageType
from image_processor import compare_blue_in_images, hconcat_resize, mse_between_loaded_images, process_image, process_image_with_grid
from issue import (
    GraphDetectionIssue,
    NonBlockingIssue,
    TitleMissingIssue,
    TotalNotFoundIssue,
    TotalOverestimationLargeIssue,
    TotalOverestimationSmallIssue,
    TotalParseErrorIssue,
    TotalUnderestimationLargeIssue,
    TotalUnderestimationSmallIssue,
)
from issue_manager import IssueManager
from ui import ScreenshotAppUI
from utils import convert_dark_mode, find_screenshot_title

if TYPE_CHECKING:
    from collections.abc import Sequence

# Note: May need to be imported on Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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
    def __init__(self) -> None:
        super().__init__()
        # Initialize data attributes
        self.images = []
        self.current_image_index = 0
        self.current_row = BatteryRow("", "", "", "", [])
        self.click_count = 0
        self.coordinates = []
        self.folder_name = ""
        self.image_mode = ImageType.BATTERY
        self.invalid_title_list = ["", " ", None]

        # Create the UI
        self.ui = ScreenshotAppUI(self)

        # Initialize issue manager
        self.issue_manager = IssueManager()
        self.issue_manager.register_observer(self.update_interface)

    # Add property getters for UI elements needed by other classes
    @property
    def magnifier_label(self):
        return self.ui.magnifier_label

    @property
    def image_label(self):
        return self.ui.image_label

    @property
    def screen_geometry(self) -> QtCore.QRect:
        screen_geo = None
        if QApplication.primaryScreen() is not None:
            screen = QApplication.primaryScreen()
            if screen is not None:
                screen_geo = screen.geometry()

        if screen_geo is None:
            screen_geo = QtCore.QRect(0, 0, 800, 600)

        return screen_geo

    def adjust_ui_for_image_type(self, image_type: ImageType) -> None:
        if image_type == ImageType.BATTERY:
            self.ui.extracted_text_label.setText("Date of first displayed time:")
            self.ui.slider.show()
            self.ui.time_label.hide()
            self.ui.instruction_label.hide()
            self.ui.extracted_total_label.hide()
            self.ui.extracted_total_image.hide()
            self.ui.auto_process_images_checkbox.hide()
            self.ui.remove_duplicates_automatically_checkbox.hide()
            self.ui.skip_daily_usage_checkbox.hide()
        elif image_type == ImageType.SCREEN_TIME:
            self.ui.extracted_text_label.setText("Extracted Title/App Name:")
            self.ui.slider.hide()
            self.ui.time_label.hide()
            self.ui.instruction_label.show()
            self.ui.extracted_total_label.show()
            self.ui.extracted_total_image.show()
            self.ui.auto_process_images_checkbox.show()
            self.ui.remove_duplicates_automatically_checkbox.show()
            self.ui.skip_daily_usage_checkbox.show()

    def capture_click(self, event: QMouseEvent) -> None:
        x = event.pos().x()
        y = event.pos().y()
        self.coordinates.append((x, y))
        self.click_count += 1

        if self.click_count == 1:
            self.ui.instruction_label.setText("Now click the bottom right corner of the graph in the left image to finish the selection.")
            self.ui.instruction_label.setStyleSheet("background-color:rgb(0,255,0)")
        elif self.click_count == 2:
            self.process_coordinates(self.coordinates[0], self.coordinates[1])
            self.coordinates = []
            self.click_count = 0

            self.update_interface()

    def update_time_label(self, value: int) -> None:
        selected_time = self.ui.time_mapping.get(value, "Midnight")
        self.ui.time_label.setText(f"First time displayed in screenshot: {selected_time}")

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
            self.ui.image_label.setText("No image loaded.")
            self.ui.cropped_image_label.setText("No cropped image loaded.")
            self.ui.image_name_line_edit.setText("Image name will appear here.")

    def skip_current_image(self) -> None:
        if self.current_image_index + 1 < len(self.images):
            self.show_image(self.current_image_index + 1)

    def show_previous_image(self) -> None:
        if self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)

    def save_current_row(self) -> None:
        if self.current_row:
            csv_path = Path(sys.argv[0]).parent / "output" / f"{Path(self.images[self.current_image_index]).parents[1].name} Arcascope Output.csv"

            if self.image_mode == ImageType.BATTERY:
                if not isinstance(self.current_row, BatteryRow):
                    msg = f"Expected BatteryRow object for Battery mode, got {type(self.current_row)}"
                    raise ValueError(msg)

                headers = BatteryRow.headers()
                self.current_row.date_from_image = self.ui.extracted_text_edit.text()
                self.current_row.time_from_ui = self.ui.time_mapping[self.ui.slider.value()]

            elif self.image_mode == ImageType.SCREEN_TIME:
                if not isinstance(self.current_row, ScreenTimeRow):
                    msg = f"Expected ScreenTimeRow object for Screen Time mode, got {type(self.current_row)}"
                    raise ValueError(msg)

                headers = ScreenTimeRow.headers()
                self.current_row.app_title = self.ui.extracted_text_edit.text()

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

            if self.ui.remove_duplicates_automatically_checkbox.isChecked():
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
        title = self.ui.extracted_text_edit.text()
        if title in self.invalid_title_list:
            self.issue_manager.add_issue(TitleMissingIssue("Title is missing or invalid. Please enter a valid title."))
        else:
            self.issue_manager.remove_issues_of_class(TitleMissingIssue)

    def apply_letter_to_number_ocr_corrections(self, time_str: str) -> str:
        """Apply common OCR error corrections to a time string."""
        # Common OCR error substitutions
        substitutions = {
            "A": "4",
            "a": "4",
            "T": "1",
            "t": "1",
            "I": "1",
            "i": "1",
            "l": "1",
            "O": "0",
            "o": "0",
            "B": "8",
            "b": "8",
            "S": "5",
            "s": "5",
            "Z": "2",
            "z": "2",
            "G": "6",
            "g": "6",
        }

        corrected = time_str

        # Apply substitutions to possible numeric parts
        for match in re.finditer(r"([A-Za-z0-9]+)\s*[hm]", time_str, re.IGNORECASE):
            original_number = match.group(1)
            position = match.start(1)

            # Check each character
            for i, char in enumerate(original_number):
                if char in substitutions:
                    corrected_char = substitutions[char]
                    # Replace at the correct position
                    corrected = corrected[: position + i] + corrected_char + corrected[position + i + 1 :]

        return corrected

    def parse_hours_and_minutes_time_string(self, total_str: str, calculated_total: float | None = None) -> int:
        """
        Parse the extracted total time string into minutes.
        Handles various formats like "2h 30m", "45m", etc.
        Returns -1 if parsing fails.

        If calculated_total is provided, tries to correct OCR errors when parsed total doesn't match.
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
            with contextlib.suppress(ValueError):
                total_minutes += int(hours_match.group(1)) * 60

        # Look for minutes pattern (e.g., "30m", "30 min", etc.)
        minutes_match = re.search(r"(\d+)\s*m", cleaned_str)
        if minutes_match:
            with contextlib.suppress(ValueError):
                total_minutes += int(minutes_match.group(1))

        # If we found a valid time but it wasn't in the expected format
        if total_minutes > 0 and not valid_format:
            self.issue_manager.add_issue(
                TotalParseErrorIssue(f"The extracted total '{total_str}' is not in the expected format (e.g., '2h 30m' or '45m').")
            )

        # If calculated_total is provided and differs significantly from parsed total, try OCR corrections
        if calculated_total is not None and total_minutes > 0 and abs(total_minutes - calculated_total) > 1:
            corrected_str = self.apply_letter_to_number_ocr_corrections(total_str)

            if corrected_str != total_str:
                # Parse the corrected string
                corrected_minutes = 0

                # Look for hours in corrected string
                hours_match = re.search(r"(\d+)\s*h", corrected_str.lower())
                if hours_match:
                    with contextlib.suppress(ValueError):
                        corrected_minutes += int(hours_match.group(1)) * 60

                # Look for minutes in corrected string
                minutes_match = re.search(r"(\d+)\s*m", corrected_str.lower())
                if minutes_match:
                    with contextlib.suppress(ValueError):
                        corrected_minutes += int(minutes_match.group(1))

                # Use corrected value if it's closer to the calculated total
                if corrected_minutes > 0 and abs(corrected_minutes - calculated_total) < abs(total_minutes - calculated_total):
                    self.issue_manager.add_issue(
                        TotalParseErrorIssue(f"Corrected OCR errors in '{total_str}' to '{corrected_str}' (closer to calculated total)")
                    )
                    return corrected_minutes

        return total_minutes

    def compare_totals_and_add_issues(self, extracted_total: str, row: list[int]) -> None:
        calculated_total_minutes = row[-1] if row else 0

        extracted_total_minutes = self.parse_hours_and_minutes_time_string(extracted_total, calculated_total_minutes)

        if extracted_total_minutes == -1:
            self.issue_manager.add_issue(TotalNotFoundIssue("The displayed total was unable to be extracted from the image."))
            return

        diff = calculated_total_minutes - extracted_total_minutes
        abs_diff = abs(diff)

        # Skip processing if there's no difference
        if abs_diff == 0:
            return

        # Calculate percentage difference
        percent_diff = abs_diff / max(1, extracted_total_minutes) * 100

        small_total_threshold = 30  # minutes
        small_total_diff_threshold = 5  # minutes
        large_total_percent_threshold = 3  # percent

        # For totals less than 30 minutes
        if extracted_total_minutes < small_total_threshold:
            if abs_diff < small_total_diff_threshold:
                # Small absolute difference for small totals → Small issue
                if diff < 0:  # Underestimation
                    self.issue_manager.add_issue(
                        TotalUnderestimationSmallIssue(
                            f"The bar total underestimated the displayed total by {abs_diff} minute(s) ({percent_diff:.1f}%)."
                        )
                    )
                else:  # Overestimation
                    self.issue_manager.add_issue(
                        TotalOverestimationSmallIssue(
                            f"The bar total overestimated the displayed total by {abs_diff} minute(s) ({percent_diff:.1f}%)."
                        )
                    )
            # Large absolute difference for small totals → Large issue
            elif diff < 0:  # Underestimation
                self.issue_manager.add_issue(
                    TotalUnderestimationLargeIssue(
                        f"The bar total significantly underestimated the displayed total by {abs_diff} minute(s) ({percent_diff:.1f}%)."
                    )
                )
            else:  # Overestimation
                self.issue_manager.add_issue(
                    TotalOverestimationLargeIssue(
                        f"The bar total significantly overestimated the displayed total by {abs_diff} minute(s) ({percent_diff:.1f}%)."
                    )
                )
        # For totals 30 minutes or more
        elif percent_diff >= large_total_percent_threshold:
            # Large percentage difference for large totals → Large issue
            if diff < 0:  # Underestimation
                self.issue_manager.add_issue(
                    TotalUnderestimationLargeIssue(
                        f"The bar total significantly underestimated the displayed total by {abs_diff} minute(s) ({percent_diff:.1f}%)."
                    )
                )
            else:  # Overestimation
                self.issue_manager.add_issue(
                    TotalOverestimationLargeIssue(
                        f"The bar total significantly overestimated the displayed total by {abs_diff} minute(s) ({percent_diff:.1f}%)."
                    )
                )
            # If less than 3% difference for large totals → No issue raised

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

        self.ui.extracted_text_edit.setText(title)
        self.ui.extracted_total_label.setText(f"Extracted Total: {total}")

        # Display the total image if available
        if total_image_path and Path(total_image_path).exists():
            total_pixmap = QPixmap(total_image_path)
            # Scale the image to a reasonable size
            scaled_pixmap = total_pixmap.scaled(300, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.ui.extracted_total_image.setPixmap(scaled_pixmap)
        else:
            self.ui.extracted_total_image.setText("No total image available")

        # Compare extracted and calculated totals for ScreenTime mode
        if self.image_mode == ImageType.SCREEN_TIME and total != "N/A":
            self.compare_totals_and_add_issues(total, row)

        # Check for graph issues
        if not processed_image_path:
            self.ui.cropped_image_label.setText("No cropped image could be loaded from the selection.")
            self.issue_manager.add_issue(GraphDetectionIssue("Failed to extract a valid graph from the image."))
        else:
            processed_pixmap = QPixmap(processed_image_path)

            self.ui.cropped_image_label.setPixmap(
                processed_pixmap.scaled(
                    self.ui.length_dimension,
                    self.ui.length_dimension,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        if not graph_image_path:
            self.ui.graph_image_label.setText("No graph could be extracted from the selection.")
            self.issue_manager.add_issue(GraphDetectionIssue("Failed to extract a valid graph from the image."))
        else:
            processed_graph_pixmap = QPixmap(graph_image_path)

            self.ui.graph_image_label.setPixmap(
                processed_graph_pixmap.scaled(
                    self.ui.length_dimension,
                    self.ui.length_dimension,
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
                file_name=self.ui.image_name_line_edit.text(),
                date_from_image=self.ui.extracted_text_edit.text(),
                time_from_ui=self.ui.time_mapping[self.ui.slider.value()],
                rows=row,
            )
        elif self.image_mode == ImageType.SCREEN_TIME:
            self.current_row = ScreenTimeRow(
                full_path=self.images[self.current_image_index],
                file_name=self.ui.image_name_line_edit.text(),
                app_title=self.ui.extracted_text_edit.text(),
                rows=row,
            )

    def update_interface(self) -> None:
        """Update the UI based on current issues."""
        if not self.issue_manager.has_issues():
            self.ui.instruction_label.setText(
                "Click Next/Save if ALL the graphs match (including the one on the left), otherwise click the upper left corner of the graph in the left image to reselect."
            )
            self.ui.instruction_label.setStyleSheet("background-color:rgb(255,255,150)")
            return

        # Ensure window is visible if there are blocking issues
        if self.issue_manager.has_issues():
            # Get the most severe issue (those that prevent continuation)
            issue = self.issue_manager.get_most_important_issue()
            if issue:
                message, style = issue.get_styled_message()
                self.ui.instruction_label.setText(message)
                self.ui.instruction_label.setStyleSheet(style)
                self.bring_window_to_front()

    def bring_window_to_front(self) -> None:
        # First, restore the window if it's minimized
        self.setWindowState(self.windowState() & ~QtCore.Qt.WindowState.WindowMinimized | QtCore.Qt.WindowState.WindowActive)

        # Set the window to stay on top by adding the flag (using OR operation)
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        # Setting window flags hides the window, so we need to show it again
        self.show()

        # Raise and activate the window
        self.raise_()
        self.activateWindow()

    def show_next_image_manual_button_press(self) -> None:
        self.issue_manager.remove_issues_of_class(NonBlockingIssue)
        self.show_next_image()

    def show_next_image(self) -> None:
        if self.current_row and not self.issue_manager.has_issues():
            self.save_current_row()
            if self.current_image_index + 1 < len(self.images):
                self.show_image(self.current_image_index + 1)
                if self.current_image_index + 1 == len(self.images):
                    self.final_row = self.current_row

    def show_image(self, index: int) -> None:
        self.issue_manager.remove_all_issues()

        if 0 <= index < len(self.images):
            image_path = self.images[index]
            pixmap = QPixmap(image_path)
            self.ui.image_label.setPixmap(
                pixmap.scaled(
                    self.ui.length_dimension,
                    self.ui.length_dimension,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.ui.image_name_line_edit.setText(Path(image_path).name)
            self.current_image_index = index

            # Check if we should skip this image based on title before full processing
            if self.image_mode == ImageType.SCREEN_TIME and self.ui.skip_daily_usage_checkbox.isChecked():
                try:
                    # Basic title extraction with minimal processing
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        img = convert_dark_mode(img)
                        title = find_screenshot_title(img)

                        if title == "Daily Total":
                            print(f"Skipping Daily Total image: {image_path}")
                            self.skip_current_image()
                            return
                except Exception as e:
                    print(f"Error checking title for skipping: {e}")
                    # Continue with normal processing if title check fails

            # Normal image processing
            try:
                processed_image_path, graph_image_path, row, title, total, total_image_path = process_image(
                    image_path,
                    self.image_mode == ImageType.BATTERY,
                    self.ui.snap_to_grid_checkbox.isChecked(),
                )

                self.update_(title or "", total, row, processed_image_path, graph_image_path, total_image_path)

                # Automatic processing for next image if no issues
                if self.ui.auto_process_images_checkbox.isChecked() and not self.issue_manager.has_issues():
                    self.showMinimized()
                    self.show_next_image()

            except Exception as e:
                print(f"Error during image loading or processing: {traceback.format_exc()}")
                self.issue_manager.add_issue(GraphDetectionIssue("Failed to process the image: " + str(e)))
                self.ui.cropped_image_label.setText("No cropped image loaded.")
                self.ui.graph_image_label.setText("No graph extracted")
                self.ui.image_name_line_edit.setText("No image available.")
        else:
            self.ui.image_label.setText("No image loaded.")
            self.ui.cropped_image_label.setText("No cropped image loaded.")
            self.ui.graph_image_label.setText("No graph extracted")
            self.ui.image_name_line_edit.setText("No image available.")

    def process_coordinates(self, upper_left: tuple[int, int], lower_right: tuple[int, int]) -> None:
        image_path = self.images[self.current_image_index]
        original_pixmap = QPixmap(image_path)

        original_width = original_pixmap.width()
        original_height = original_pixmap.height()

        scalar = self.ui.length_dimension / original_height

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
            self.ui.snap_to_grid_checkbox.isChecked(),
        )

        self.update_(title, total, row, processed_image_path, graph_image_path, total_image_path)


if __name__ == "__main__":
    sys.argv += ["-platform", "windows:darkmode=1"]
    app = QApplication(sys.argv)
    ex = ScreenshotApp()
    ex.show()
    sys.exit(app.exec())
