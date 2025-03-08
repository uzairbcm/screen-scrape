from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pytesseract import Output, pytesseract

from utils import (
    adjust_contrast_brightness,
    convert_dark_mode,
    find_left_anchor,
    find_right_anchor,
    find_screenshot_title,
    find_screenshot_total_usage,
    get_pixel,
    get_text,
    remove_all_but,
    slice_image,
)


class ImageProcessingError(Exception):
    """Custom exception for image processing failures."""

    def __init__(self, message: str, errors=None) -> None:
        super().__init__(message)
        self.errors = errors


def load_and_validate_image(filename: Path | str) -> np.ndarray:
    """Load an image and validate its existence."""
    img = cv2.imread(str(filename))
    img = convert_dark_mode(img)
    if img is None:
        msg = "Failed to load image."
        raise ImageProcessingError(msg)
    return img


def calculate_roi_from_clicks(
    upper_left: tuple[int, int],
    lower_right: tuple[int, int],
    snap_to_grid_func: Callable | None = None,
    img: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """Calculate the region of interest based on user input, optionally snapping to grid."""
    roi_width = lower_right[0] - upper_left[0]
    roi_height = lower_right[1] - upper_left[1]

    if roi_width <= 0 or roi_height <= 0:
        msg = "Invalid region of interest dimensions."
        raise ImageProcessingError(msg)

    # TODO: Re-add snap-to-grid capabilities
    # if snap_to_grid_func and img is not None:
    #     return snap_to_grid_func(img, upper_left[0], upper_left[1], roi_width, roi_height)
    # else:
    return upper_left[0], upper_left[1], roi_width, roi_height


def process_image_with_grid(
    filename: Path | str,
    upper_left: tuple[int, int],
    lower_right: tuple[int, int],
    is_battery: bool,
    snap_to_grid,
) -> tuple:
    try:
        img = load_and_validate_image(filename)
        if img is not None:
            img_copy = img.copy()
        else:
            msg = "Failed to load image."
            raise ImageProcessingError(msg)

        img = adjust_contrast_brightness(img, contrast=2.0, brightness=-220)

        snap_func = snap_to_grid if snap_to_grid else None

        print("Calculating region of interest from clicks...")
        upper_left_x, upper_left_y, roi_width, roi_height = calculate_roi_from_clicks(upper_left, lower_right, snap_func, img)

        if upper_left_x < 0 or upper_left_y < 0:
            msg = "ROI coordinates are out of image bounds."
            raise ImageProcessingError(msg)

        roi_x = upper_left_x
        roi_y = upper_left_y

        if is_battery:
            print("Extracting time...")
            title = find_time(img_copy, roi_x, roi_y, roi_width, roi_height)
            total = "N/A"
            total_image_path = None
        else:
            print("Extracting title...")
            title = find_screenshot_title(img)
            total, total_image_path = find_screenshot_total_usage(img)

        filename, row, graph_filename = save_image(filename, roi_x, roi_y, roi_width, roi_height, is_battery)

    except ImageProcessingError:
        print(f"Error: {traceback.format_exc()}")
        return None, None, list(range(25)), "", "", None

    else:
        return filename, graph_filename, row, title, total, total_image_path


def process_image(filename: str, is_battery: bool, snap_to_grid) -> tuple[str, str, list, str, str, str | None]:
    """Process an image to extract grid data and relevant information."""
    img = load_and_validate_image(filename)
    return apply_processing(filename, img, is_battery, snap_to_grid)


def apply_processing(filename: str, img: np.ndarray, is_battery: bool, snap_to_grid) -> tuple[str, str, list, str, str, str | None]:
    """Apply image processing to identify and extract grid data from screenshots."""
    # Prepare image and perform OCR
    img = adjust_contrast_brightness(img, contrast=2.0, brightness=-220)
    img_copy = img.copy()
    img_left, img_right, right_offset = prepare_image_chunks(img)

    d_left, d_right = perform_ocr(img_left, img_right)
    adjust_anchor_offsets(d_right, right_offset)

    # Find grid anchors and calculate ROI
    roi_params = find_grid_anchors_and_calculate_roi(d_left, d_right, img, img_copy, snap_to_grid)
    if roi_params is None:
        msg = "Couldn't find graph anchors!"
        raise ValueError(msg)

    roi_x, roi_y, roi_width, roi_height = roi_params

    # Extract content based on image type
    if is_battery:
        title = find_time(img_copy, roi_x, roi_y, roi_width, roi_height)
        total, total_image_path = "N/A", None
    else:
        title = find_screenshot_title(img)
        total, total_image_path = find_screenshot_total_usage(img)

    # Process and save extracted grid
    filename, row, graph_filename = save_image(filename, roi_x, roi_y, roi_width, roi_height, is_battery)
    return filename, graph_filename, row, title, total, total_image_path


def adjust_anchor_offsets(data: dict, offset: int) -> None:
    """Adjust the left coordinates of detected text by the specified offset."""
    for i in range(len(data["level"])):
        data["left"][i] += offset


def find_grid_anchors_and_calculate_roi(
    d_left: dict,
    d_right: dict,
    img: np.ndarray,
    img_copy: np.ndarray,
    snap_to_grid,
) -> tuple[int, int, int, int] | None:
    """Find grid anchors and calculate region of interest from them."""
    # First attempt with no skipping
    found_12, lower_left_x, lower_left_y = find_left_anchor(d_left, img, img_copy, detections_to_skip=0)
    found_60, upper_right_x, upper_right_y = find_right_anchor(d_right, img, img_copy)

    # Try to calculate ROI if anchors are found
    if found_12 and found_60:
        try:
            return calculate_roi(
                lower_left_x, upper_right_y, upper_right_x - lower_left_x, lower_left_y - upper_right_y, img, snap_to_grid=snap_to_grid
            )
        except Exception as e:
            print(f"First attempt failed: {e}")

    # Retry with different skip values
    for skip_value in range(1, 4):
        found_12, lower_left_x, lower_left_y = find_left_anchor(d_left, img, img_copy, detections_to_skip=skip_value)
        found_60, upper_right_x, upper_right_y = find_right_anchor(d_right, img, img_copy)

        if found_12 and found_60:
            try:
                return calculate_roi(
                    lower_left_x,
                    upper_right_y,
                    upper_right_x - lower_left_x,
                    lower_left_y - upper_right_y,
                    img,
                    snap_to_grid=snap_to_grid,
                )
            except Exception as e:
                print(f"Attempt with skip={skip_value} failed: {e}")
                continue

    # No anchors found after all attempts
    return None


def find_time(
    img: np.ndarray,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
) -> str:
    text1, text2, is_pm = get_text(img, roi_x, roi_y, roi_width, roi_height)

    return text1


def prepare_image_chunks(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    img_chunk_num = 3
    img_width, img_height = img.shape[1], img.shape[0]
    top_removal = int(img_height * 0.05)  # Removing top 10% of the image to minimize risk of failed grid detection

    # Split image into left and right for anchor detection
    img_left = img[:, : int(img_width / img_chunk_num)]
    img_right = img[:, -int(img_width / img_chunk_num) :]

    # Overwrite the top part of the image to simplify the image for OCR
    img_left[0:top_removal, :] = get_pixel(img_left, 1)
    img_right[0:top_removal, :] = get_pixel(img_right, 1)
    right_offset = img_width - int(img_width / img_chunk_num)

    return img_left, img_right, right_offset


def perform_ocr(
    img_left: np.ndarray,
    img_right: np.ndarray,
) -> tuple[dict, dict]:
    d_left = pytesseract.image_to_data(img_left, config="--psm 12", output_type=Output.DICT)
    d_right = pytesseract.image_to_data(img_right, config="--psm 12", output_type=Output.DICT)
    return d_left, d_right


def calculate_roi(
    lower_left_x: int,
    upper_right_y: int,
    roi_width: int,
    roi_height: int,
    img: np.ndarray,
    snap_to_grid,
) -> tuple[int, int, int, int]:
    if snap_to_grid:
        lower_left_x, upper_right_y, roi_width, roi_height = snap_to_grid(img, lower_left_x, upper_right_y, roi_width, roi_height)

    if lower_left_x < 0:
        msg = f"Invalid ROI lower left x coordinate: {lower_left_x}"
        raise ValueError(msg)
    elif upper_right_y < 0:
        msg = f"Invalid ROI upper right y coordinate: {upper_right_y}"
        raise ValueError(msg)
    elif roi_width < 0:
        msg = f"Invalid ROI width value: {roi_width}"
        raise ValueError(msg)
    elif roi_height < 0:
        msg = f"Invalid ROI height value: {roi_height}"
        raise ValueError(msg)

    return lower_left_x, upper_right_y, roi_width, roi_height


def save_image(
    filename: Path | str,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    is_battery: bool,
) -> tuple[str, list, str]:
    print("Preparing to extract grid")
    img = load_and_validate_image(filename)
    if img is not None:
        img_copy = img.copy()
    else:
        msg = "Failed to load image."
        raise ImageProcessingError(msg)

    if is_battery:
        print("Removing all but the dark blue color...")
        img_new = remove_all_but(img_copy, np.ndarray([255, 121, 0]))
        no_dark_blue_detected = np.sum(255 - img_new[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]) < 10
        if no_dark_blue_detected:
            print("No dark blue color detected; assuming dark mode...")
            img_copy = img.copy()
            img_new = remove_all_but(img_copy, np.ndarray([0, 255 - 121, 255]))
        img = img_new

    row, img, scale_amount = slice_image(img, roi_x, roi_y, roi_width, roi_height)

    print("Saving processed image...")
    selection_save_path = save_processed_image(img, roi_x, roi_y, roi_width, roi_height, filename, scale_amount)

    debug_folder = "debug"
    Path(debug_folder).mkdir(parents=True, exist_ok=True)

    graph_save_path = Path(debug_folder) / "graph_" / Path(filename).name
    graph_save_path = str(graph_save_path).replace(".jfif", ".jpg")

    plt.figure(figsize=(8, 3))

    total_minutes = row[-1]
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)

    total_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

    ax = plt.gca()

    plt.xlabel(f"Calculated Total: {total_text}", ha="center", va="center", labelpad=20, fontsize=16, fontweight="bold", color="#0066CC")

    x = range(len(row[:-1]))
    height = row[:-1]
    plt.bar(np.array(x) + 0.5, height, color="#4682B4")
    plt.ylim([0, 60])
    plt.xlim([0, 24])

    tick_positions = np.array(range(24)) + 0.5
    tick_labels = [f"{int(x)}" for x in height]

    plt.xticks(tick_positions, tick_labels, fontsize=9, fontweight="bold")

    # Remove y-axis labels and ticks
    plt.yticks([])
    ax.yaxis.set_visible(False)

    # Keep only the bottom x-axis visible
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    for x in range(25):  # 0 to 24 for 24 hours + right edge
        plt.axvline(x=x, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)

    plt.tight_layout(pad=0)

    plt.savefig(graph_save_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close()

    return selection_save_path, row, graph_save_path


def save_processed_image(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    original_filename: str | Path,
    scale_amount: int,
) -> str:
    """Save the ROI of the image to a file in the debug directory."""
    debug_folder = "debug"
    Path(debug_folder).mkdir(parents=True, exist_ok=True)
    save_name = Path(debug_folder) / Path(original_filename).name
    roi = image[
        scale_amount * y : scale_amount * y + scale_amount * height,
        scale_amount * x : scale_amount * x + scale_amount * width,
    ]

    save_name = str(save_name).replace(".jfif", ".jpg")
    cv2.imwrite(save_name, roi)
    return save_name


def mse_between_loaded_images(image1: np.ndarray, image2: np.ndarray) -> float:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    height, width, _ = image1.shape
    diff = cv2.subtract(image1, image2)
    error = np.sum(diff**2)
    mean_squared_error = error // float(height * width)
    print("MSE between selection and graph:", mean_squared_error)
    return float(mean_squared_error)


def hconcat_resize(
    img_list: list[np.ndarray],
    interpolation: cv2.InterpolationFlags = cv2.INTER_CUBIC,
) -> np.ndarray:
    # take minimum hights
    h_max = max(img.shape[0] for img in img_list)

    # image resizing
    im_list_resize = [
        cv2.resize(
            img,
            (int(img.shape[1] * h_max / img.shape[0]), h_max),
            interpolation=interpolation,
        )
        for img in img_list
    ]

    # return final image
    return cv2.hconcat(im_list_resize)


def compare_blue_in_images(
    image1_path: str | Path | None = None,
    image2_path: str | Path | None = None,
    image1: np.ndarray | None = None,
    image2: np.ndarray | None = None,
) -> None:
    if image1_path is not None and image2_path is not None:
        image1 = cv2.imread(str(image1_path))
        image2 = cv2.imread(str(image2_path))
    elif image1 is not None and image2 is not None:
        pass
    else:
        msg = "Incorrect argument set entered."
        raise ValueError(msg)

    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv2, lower_blue, upper_blue)

    blue_only_image1 = cv2.bitwise_and(image1, image1, mask=mask1)
    blue_only_image2 = cv2.bitwise_and(image2, image2, mask=mask2)

    # Convert the blue_only images to grayscale
    gray_image1 = cv2.cvtColor(blue_only_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(blue_only_image2, cv2.COLOR_BGR2GRAY)

    # Binarize the grayscale images
    _, binary_image1 = cv2.threshold(gray_image1, 1, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(gray_image2, 1, 255, cv2.THRESH_BINARY)

    binary_image2 = cv2.resize(binary_image2, (binary_image1.shape[1], binary_image1.shape[0]))

    height, width = binary_image1.shape
    diff = cv2.subtract(binary_image1, binary_image2)
    error = np.sum(diff**2)
    mean_squared_error = error / float(height * width)

    # cv2.imshow('image1',binary_image1)
    # cv2.imshow('image2',binary_image2)
    # cv2.imshow('image3',diff)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("MSE between selection and graph based on blue bars:", mean_squared_error)
