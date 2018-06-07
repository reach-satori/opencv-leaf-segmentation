import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#just gets the biggest contour, might make a mistake
#there's probably a one- or two-line elegant way to do this but
#let's do it the simple way with a loop
def get_biggest_contour(cnts):
    maxarea = 0
    maxindex = 0
    for i, cnt in enumerate(cnts):
        area = cv.contourArea(cnt)
        if area > maxarea:
            maxarea = area
            maxindex = i

    return cnts[maxindex]

def get_area(img):
    proc = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    (h, s, v) = cv.split(proc)
    proc = cv.subtract(s, v)
    ret, proc = cv.threshold(proc, proc.max()*0.2, 255, cv.THRESH_BINARY)
    proc = cv.dilate(proc, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations = 5)
    proc, cnts, hierarchy = cv.findContours(proc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    biggest = get_biggest_contour(cnts)
    x, y, w, h = cv.boundingRect(biggest)
    cropped = img[y:y+h, x:x+w]
    for i in biggest:
        i[0][0] -= x
        i[0][1] -= y

    mask = np.zeros((h, w), np.uint8)
    cv.drawContours(mask, [biggest],0, 255, -1)
    # mask = np.resize(mask, cropped.shape)
    cropped = cv.bitwise_and(cropped, cropped, mask=mask)

    return cropped


