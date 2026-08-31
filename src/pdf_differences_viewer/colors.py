"""Shared semantic colors for difference layers and change annotations."""


class DifferenceColors:
    """Palette for additions and removals in Qt (RGB) and OpenCV (BGR)."""

    ADDITION_RGB = (0, 102, 255)  # bright blue
    REMOVAL_RGB = (245, 45, 45)   # bright red
    ADDITION_BGR = ADDITION_RGB[::-1]
    REMOVAL_BGR = REMOVAL_RGB[::-1]
