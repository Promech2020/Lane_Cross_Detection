import line_de
import cv2
import pytest
import numpy as np

image = cv2.imread("../support_images/11.png")
hsl_image = None
hcs = None
gs = None
gas = None

def test_convert_hsl():
    global hsl_image
    hsl_image = line_de.convert_hsl(image)
    assert type(hsl_image) == type(image)

def test_HSL_color_selection():
    global hcs
    global hsl_image
    hcs = line_de.HSL_color_selection(hsl_image)
    assert type(hcs) == type(image)

def test_gray_scale():
    global gs
    global hcs
    gs = line_de.gray_scale(hcs)
    assert type(gs) == type(image)

def test_gaussian_smoothing():
    global gas
    global gs
    gas = line_de.gaussian_smoothing(gs)
    assert type(gas) == type(image)

def test_execute():
    x = 838
    y = 364
    w = 1017
    h = 1082
    # data = line_de.execute(image, x, y, w, h)
    global gas
    rs = line_de.region_selection(gas, x, y, w, h)
    skel = line_de.skeleton(rs)
    data = line_de.hough_transform(skel)
    houghLines = np.array([[[ 938,  364,  951, 1079]],

       [[ 880, 1078,  917,  364]],

       [[ 939,  364,  951, 1066]],

       [[ 881, 1079,  911,  493]],

       [[ 942,  487,  952, 1063]],

       [[ 882, 1070,  907,  592]],

       [[ 910,  479,  916,  364]]])
    assert type(data) == type(houghLines)

# def test_region_selection():
#     assert type(line_de.region_selection(image)) == type(image)

# def test_skeleton():
#     assert type(line_de.skeleton(image)) == type(line_de.skeleton(image))

# def test_hough_transform():
#     assert type(line_de.hough_transform(image)) == type(line_de.hough_transform(image))