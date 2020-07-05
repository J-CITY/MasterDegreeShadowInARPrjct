import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import resize, ratio
from config import IS_DEBUG, DRAW_CONTOUROS

def detectEdges(img, min_val, max_val):
    img = cv2.cvtColor(resize(img), cv2.COLOR_BGR2GRAY)
    # Applying blur and threshold
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)
    #implt(img, 'gray', 'Adaptive Threshold')

    # Median blur replace center pixel by median of pixels under kelner
    # => removes thin details
    img = cv2.medianBlur(img, 11)

    # Add black border - detection of border touching pages
    # Contour can't touch side of image
    img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #implt(img, 'gray', 'Median Blur + Border')

    return cv2.Canny(img, min_val, max_val)

def cornersSort(pts):
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contourOffset(cnt, offset):
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt


def findPageContours(edges, img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Finding biggest rectangle otherwise return original corners
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * 0.3
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    max_area = MIN_COUNTOUR_AREA
    page_contour = np.array([[0, 0],
                            [0, height-5],
                            [width-5, height-5],
                            [width-5, 0]])

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # Page has 4 corners and it is convex
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                max_area < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
            
            max_area = cv2.contourArea(approx)
            page_contour = approx[:, 0]

    # Sort corners and offset them
    page_contour = cornersSort(page_contour)
    return contourOffset(page_contour, (-5, -5))


def getPaperCoords(image):
    edges_image = detectEdges(image, 200, 250)

    edges_image = cv2.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
    #implt(edges_image, 'gray', 'Edges')
    page_contour = findPageContours(edges_image, resize(image))

    if IS_DEBUG:
        print("PAGE CONTOUR:")
        print(page_contour)
    #implt(cv2.drawContours(resize(image), [page_contour], -1, (0, 255, 0), 3))

    # Recalculate to original scale
    #page_contour = page_contour.dot(ratio(image))

    return page_contour, image

def drawContours(image, coords, color=(0,255,0)):
    image = cv2.drawContours(resize(image), [coords], -1, color, 3)
    return image

def getPaperCoords2(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    th, threshed = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)

    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    canvas  = img.copy()
    cnts = sorted(cnts, key = cv2.contourArea)
    cnt = cnts[-1]

    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02* arclen, True)
    #cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 1, cv2.LINE_AA)
    approx = approx[:, 0]
    if IS_DEBUG:
        print("PAGE CONTOUR:")
        print(approx)
    return approx, canvas