from enum import StrEnum


class ImageType(StrEnum):
    BATTERY = "Battery"
    SCREEN_TIME = "Screen Time"


class LineExtractionMode(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
