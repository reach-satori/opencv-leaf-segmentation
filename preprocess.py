import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#for filtering out small noisy contours and improving sort speed
MIN_CONTOUR_AREA = 600
#how many samples to take to figure out average contour-centroid distance
DIST_SAMPLES = 40
CONTOURS_TO_CHECK = 2
def get_centroid(cnt):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return np.array([cx, cy])

#unused
def get_plant_contour(cnts, imgsize):
    imgcenter = np.array([imgsize[0]/2, imgsize[1]/2])
    distances = []
    for i in range(CONTOURS_TO_CHECK):
        avg = 0
        for j in range(DIST_SAMPLES):
            randint = np.random.randint(0, cnts[i].shape[0])
            avg += np.linalg.norm(imgcenter - cnts[i][randint, 0])
        avg /= DIST_SAMPLES
        distances.append(avg)
    return cnts[distances.index(min(distances))]


def remove_noisy_contours(img):
    _, cnts, __ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = list(filter(lambda x: cv.contourArea(x) > MIN_CONTOUR_AREA, cnts))
    cnts = np.array(sorted(cnts, key=cv.contourArea, reverse=True))
    return cnts



#cuts out dark soil background (needs denoising)
def get_plants(img):
    b, g, r = cv.split(img)
    g_b = cv.subtract(g, b)
    _, otsu = cv.threshold(g_b,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    otsu = cv.morphologyEx(otsu, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7)), iterations=2)
    _, cnts, __ = cv.findContours(otsu, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#filter noise to improve sorting speed, sort from largest to smallest
    cnts = list(filter(lambda x: cv.contourArea(x) > MIN_CONTOUR_AREA, cnts))
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    # plantcnt = get_plant_contour(cnts, img.shape[0:2])
    # if len(cnts) >= 2:
    #     plantcnt = np.concatenate((cnts[0], cnts[1]),axis=0)
    # else:
    plantcnt = cnts[0]

    x, y, w, h = cv.boundingRect(plantcnt)
    cropped = img[y:y+h, x:x+w]

    return cropped

#cuts out bright table background
def get_area(img):
    proc = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    (h, s, v) = cv.split(proc)
    proc = cv.subtract(s, v)
    ret, proc = cv.threshold(proc, 255//3, 255, cv.THRESH_BINARY)
    proc = cv.morphologyEx(proc, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7)), iterations=4)
    proc = cv.morphologyEx(proc, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))
    # cv.imwrite("binary.jpg", proc)
    cnts = remove_noisy_contours(proc)
    x, y, w, h = cv.boundingRect(cnts[0])
    cropped = img[y:y+h, x:x+w]

    for i in cnts[0]:
        i[0][0] -= x
        i[0][1] -= y
    mask = np.zeros((h, w), np.uint8)
    cv.drawContours(mask, cnts, 0, 255, -1)
    # mask = np.resize(mask, cropped.shape)
    cropped = cv.bitwise_and(cropped, cropped, mask=mask)

    #At this point we're assuming the largest contour(cnts[0]) is either the plant or contains the plant.
    #One more step is necessary to make sure we're getting only the plant contour
    cropped = get_plants(cropped)
    return cropped



