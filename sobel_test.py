import cv2 as cv
import numpy as np




def nothing(x):
    pass

img = cv.imread("debug_imgs/otsu-thresh.png")
cv.namedWindow("frame", cv.WINDOW_AUTOSIZE)

img2 = img.copy()
img2 = cv.Canny(img2, 100, 200)
img2 = cv.morphologyEx(img2, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)), iterations = 2)
cv.imwrite("what.png", img2)


cv.imshow('frame', img2)
while True:
    k = cv.waitKey(100) & 0xFF
    if k == 27:
        break
