import os
from glob import iglob

import pandas as pd

from utils import *


def extract_parental_controls(folder):
    disallowed_types = [".db"]
    file_list = [f for f in iglob(folder + "Parental controls/*", recursive=True) if
                 os.path.isfile(f) and not any(key in f for key in disallowed_types)]

    export_columns = []
    export_rows = []

    for filename in file_list:

        print("Running " + filename + "...")
        img = cv2.imread(filename)
        img_copy = img.copy()

        last_y = 0
        new_row_threshold = 10
        text_row = ""

        # Darken image to improve likelihood of text extraction
        img = adjust_contrast_brightness(img, contrast=2.0, brightness=-220)
        d = pytesseract.image_to_data(img, config='--psm 4', output_type=Output.DICT)

        n_boxes = len(d['level'])

        all_rows_for_file = []

        # Loop over all text, add words that are in a similar vertical position to the same "row"
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(d['text'][i]) > 1:
                if np.abs(last_y - y) < new_row_threshold:
                    text_row = text_row + " " + d['text'][i]
                else:
                    all_rows_for_file.append(text_row)
                    text_row = d['text'][i]
                last_y = y

        all_rows_for_file.append(text_row)

        # Append contents to export file
        column_name = filename.split("/")[-1][:-4]
        export_columns.append(column_name)
        export_rows.append(all_rows_for_file)

    df = pd.DataFrame(export_rows)
    df = df.transpose()
    df.columns = export_columns

    save_name = folder.split("/")[-2]
    with pd.ExcelWriter("output/Parental Controls " + save_name + '.xlsx') as writer:
        df.to_excel(writer, sheet_name='sheet1')


def extract_all_parental_controls():
    root_directory = 'data/*/'

    folder_list = [f for f in iglob(root_directory, recursive=False) if os.path.isdir(f)]
    for folder in folder_list:
        extract_parental_controls(folder)


if __name__ == '__main__':
    extract_all_parental_controls()
