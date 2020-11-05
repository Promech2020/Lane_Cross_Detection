import get_line
import cv2
import pytest

image = cv2.imread("../support_images/11.png")

def test_getLine():
    line = get_line.getLine(image)
    assert line is not None and len(line) == 5