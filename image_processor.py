import traceback
import pytesseract
import os
import matplotlib.pyplot as plt
from utils import *

import cv2
import numpy as np


class ImageProcessingError(Exception):
    """Custom exception for image processing failures."""

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


def load_and_validate_image(filename):
    """Load an image and validate its existence."""
    img = cv2.imread(filename)
    img = convert_dark_mode(img)
    if img is None:
        raise ImageProcessingError("Failed to load image.")
    return img


def calculate_roi_from_clicks(upper_left, lower_right, snap_to_grid_func=None, img=None):
    """Calculate the region of interest based on user input, optionally snapping to grid."""
    roi_width = lower_right[0] - upper_left[0]
    roi_height = lower_right[1] - upper_left[1]

    if roi_width <= 0 or roi_height <= 0:
        raise ImageProcessingError("Invalid region of interest dimensions.")

    # TODO: Re-add snap-to-grid capabilities
    # if snap_to_grid_func and img is not None:
    #     return snap_to_grid_func(img, upper_left[0], upper_left[1], roi_width, roi_height)
    # else:
    return upper_left[0], upper_left[1], roi_width, roi_height


def process_image_with_grid(filename, upper_left, lower_right, is_battery=False, snap_to_grid=False):
    try:
        img = load_and_validate_image(filename)
        img_copy = img.copy()

        img = adjust_contrast_brightness(img, contrast=2.0, brightness=-220)

        if snap_to_grid:
            snap_func = snap_to_grid
        else:
            snap_func = None

        print("Calculating region of interest from clicks...")
        upper_left_x, upper_left_y, roi_width, roi_height = calculate_roi_from_clicks(upper_left, lower_right, snap_func, img)

        if upper_left_x < 0 or upper_left_y < 0:
            raise ImageProcessingError("ROI coordinates are out of image bounds.")

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
        return filename, graph_filename, row, title, total, total_image_path

    except ImageProcessingError as e:
        print(f"Error: {traceback.format_exc()}")
        return None, None, list(range(25)), "", "", None


def process_image(filename, is_battery=False, snap_to_grid=False):
    img = load_and_validate_image(filename)
    return apply_processing(filename, img, is_battery, snap_to_grid)


def adjust_anchor_offsets(d_right, offset):
    for i in range(len(d_right["level"])):
        d_right["left"][i] += offset


def apply_processing(filename, img, is_battery, snap_to_grid):
    img = adjust_contrast_brightness(img, contrast=2.0, brightness=-220)
    img_copy = img.copy()
    img_left, img_right, right_offset = prepare_image_chunks(img)

    # Perform OCR and adjust offsets
    d_left, d_right = perform_ocr(img_left, img_right)
    adjust_anchor_offsets(d_right, right_offset)

    found_12, lower_left_x, lower_left_y = find_left_anchor(d_left, img, img_copy, skip_detections=0)
    found_60, upper_right_x, upper_right_y = find_right_anchor(d_right, img, img_copy)

    # Try to calculate ROI if anchors are found
    if found_12 and found_60:
        try:
            roi_x, roi_y, roi_width, roi_height = calculate_roi(
                lower_left_x,
                upper_right_y,
                upper_right_x - lower_left_x,
                lower_left_y - upper_right_y,
                snap_to_grid,
                img,
            )
        except Exception as e:
            print(e)
            found_12 = found_60 = False

    # Retry finding anchors if initial detection failed
    if not found_12 or not found_60:
        for i in range(1, 4):
            found_12, lower_left_x, lower_left_y = find_left_anchor(d_left, img, img_copy, skip_detections=i)
            found_60, upper_right_x, upper_right_y = find_right_anchor(d_right, img, img_copy)
            if found_12 and found_60:
                try:
                    roi_x, roi_y, roi_width, roi_height = calculate_roi(
                        lower_left_x,
                        upper_right_y,
                        upper_right_x - lower_left_x,
                        lower_left_y - upper_right_y,
                        snap_to_grid,
                        img,
                    )
                    break
                except Exception as e:
                    print(e)
                    continue

    # Raise error if anchors are still not found
    if not found_12 or not found_60:
        raise ValueError("Couldn't find graph anchors!")

    # Determine title based on battery status
    if is_battery:
        title = find_time(img_copy, roi_x, roi_y, roi_width, roi_height)
        total = "N/A"
        total_image_path = None
    else:
        title = find_screenshot_title(img)
        total, total_image_path = find_screenshot_total_usage(img)

    # Save the processed image and return
    filename, row, graph_filename = save_image(filename, roi_x, roi_y, roi_width, roi_height, is_battery)
    return filename, graph_filename, row, title, total, total_image_path


def find_time(img, roi_x, roi_y, roi_width, roi_height):
    text1, text2, is_pm = get_text(img, roi_x, roi_y, roi_width, roi_height)

    return text1


def prepare_image_chunks(img):
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


def perform_ocr(img_left, img_right):
    d_left = pytesseract.image_to_data(img_left, config="--psm 12", output_type=Output.DICT)
    d_right = pytesseract.image_to_data(img_right, config="--psm 12", output_type=Output.DICT)
    return d_left, d_right


def calculate_roi(lower_left_x, upper_right_y, roi_width, roi_height, snap_to_grid, img):
    if snap_to_grid:
        lower_left_x, upper_right_y, roi_width, roi_height = snap_to_grid(img, lower_left_x, upper_right_y, roi_width, roi_height)

    if lower_left_x < 0:
        raise ValueError(f"Invalid ROI lower left x coordinate: {lower_left_x}")
    elif upper_right_y < 0:
        raise ValueError(f"Invalid ROI upper right y coordinate: {upper_right_y}")
    elif roi_width < 0:
        raise ValueError(f"Invalid ROI width value: {roi_width}")
    elif roi_height < 0:
        raise ValueError(f"Invalid ROI height value: {roi_height}")

    return lower_left_x, upper_right_y, roi_width, roi_height


def save_image(filename, roi_x, roi_y, roi_width, roi_height, is_battery):
    print("Preparing to extract grid")
    img = load_and_validate_image(filename)
    img_copy = img.copy()

    if is_battery:
        print("Removing all but the dark blue color...")
        img_new = remove_all_but(img_copy, [255, 121, 0])
        no_dark_blue_detected = np.sum(255 - img_new[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]) < 10
        if no_dark_blue_detected:
            print("No dark blue color detected; assuming dark mode...")
            img_copy = img.copy()
            img_new = remove_all_but(img_copy, [0, 255 - 121, 255])
        img = img_new

    row, img, scale_amount = slice_image(img, roi_x, roi_y, roi_width, roi_height)

    print("Saving processed image...")
    selection_save_path = save_processed_image(img, roi_x, roi_y, roi_width, roi_height, filename, scale_amount)

    # Create debug directory
    debug_folder = "debug"
    os.makedirs(debug_folder, exist_ok=True)

    # Define path for the graph image
    graph_save_path = os.path.join(debug_folder, "graph_" + os.path.basename(filename))
    graph_save_path = graph_save_path.replace(".jfif", ".jpg")

    # Create a separate figure just for the bar chart
    plt.figure(figsize=(8, 2))

    # Format and display the total in the middle subplot
    total_minutes = row[-1]
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)

    total_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

    ax = plt.gca()

    plt.xlabel(f"Calculated Total: {total_text}", ha="center", va="center", fontsize=8, fontweight="bold")

    # Create the bar chart
    x = range(len(row[:-1]))
    height = row[:-1]
    bars = plt.bar(np.array(x) + 0.5, height)
    plt.ylim([0, 60])
    plt.xlim([0, 24])

    # Set up x-ticks for every hour with minute values
    tick_positions = np.array(range(24)) + 0.5
    tick_labels = [f"{int(x)}" for x in height]

    # Set the minute value labels on the x-axis (horizontal)
    plt.xticks(tick_positions, tick_labels, fontsize=8)

    # Remove y-axis labels and ticks
    plt.yticks([])
    ax.yaxis.set_visible(False)

    # Keep only the bottom x-axis visible
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    # Add subtle grid lines for readability
    plt.grid(which="both", axis="x", linestyle="--", linewidth=0.5, alpha=0.3)

    # Save just the graph image
    plt.savefig(graph_save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return selection_save_path, row, graph_save_path


def save_processed_image(image, x, y, width, height, original_filename, scale_amount):
    """Save the ROI of the image to a file in the debug directory."""
    debug_folder = "debug"
    os.makedirs(debug_folder, exist_ok=True)
    save_name = os.path.join(debug_folder, os.path.basename(original_filename))
    roi = image[
        scale_amount * y : scale_amount * y + scale_amount * height,
        scale_amount * x : scale_amount * x + scale_amount * width,
    ]

    save_name = save_name.replace(".jfif", ".jpg")
    cv2.imwrite(save_name, roi)
    return save_name


def mse_between_loaded_images(image1, image2):
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    height, width, _ = image1.shape
    diff = cv2.subtract(image1, image2)
    error = np.sum(diff**2)
    mean_squared_error = error // float(height * width)
    print("MSE between selection and graph:", mean_squared_error)
    return mean_squared_error


def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
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


def compare_blue_in_images(image1_path=None, image2_path=None, image1=None, image2=None, *args, **kwargs):
    if image1_path and image2_path:
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
    elif image1.any() and image2.any():
        pass
    else:
        raise ValueError("Incorrect argument set entered.")

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
