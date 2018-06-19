import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from os.path import basename, dirname, realpath
from preprocess import get_area, remove_noisy_contours
DEBUG_IMGS = True


def leaf_segment(filename):
#basic input
    img = cv.imread(filename)
#preprocess
    img = get_area(img)
    if DEBUG_IMGS: cv.imwrite("debug_imgs/cropped.png", img)

#g-b highlights the plants nicely
    (b, g, r) = cv.split(img)
    img2 = cv.subtract(g, b)
    if DEBUG_IMGS: cv.imwrite("debug_imgs/green-blue.png", img2)

#otsu threshold it to get a decent binary segmentation (but doesn't get different leaves)
    ret1, otsu = cv.threshold(img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    img2 = cv.bitwise_and(img2, img2, mask=otsu)
    cnts = remove_noisy_contours(otsu)
    otsu = np.zeros_like(otsu)
    cv.drawContours(otsu, cnts, -1, 255, -1)
    if DEBUG_IMGS: cv.imwrite("debug_imgs/otsu-thresh.png", otsu)
    img2 = cv.bitwise_and(img2, img2, mask=otsu)
    if DEBUG_IMGS: cv.imwrite("debug_imgs/cutout-plants.png", img2)

#try k-means instead
    # imgflat = img2.reshape((-1, 1))
    # imgflat = np.float32(imgflat)
    # criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K=3
    # ret, label, center = cv.kmeans(imgflat, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # center = np.uint8(center)
    # otsu = center[label.flatten()]
    # otsu = otsu.reshape((img2.shape))

    # otsu[otsu<40] = 0
    # cnts = remove_noisy_contours(otsu)
    # otsu = np.zeros_like(otsu)
    # cv.drawContours(otsu, cnts, 0, 255, -1)


    # img2 = cv.equalizeHist(img2)
# otsu = cv.morphologyEx(otsu, cv.MORPH_OPEN, kernel, iterations=2)

#get sure background
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    bg = cv.morphologyEx(otsu, cv.MORPH_DILATE, kernel)

#get distance transform for watershed seeds
    dist = cv.distanceTransform(otsu, cv.DIST_L2, 3)
    dist = cv.normalize(dist,dist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    dist = dist*255
    dist = dist.astype(np.uint8)
    if DEBUG_IMGS: cv.imwrite("debug_imgs/dist-transform.png", dist)

#watershed seeds and labeling them
    ret, sure_fg = cv.threshold(dist, 0.4*dist.max(), 255, 0)
    ret, markers = cv.connectedComponents(sure_fg)
    if DEBUG_IMGS: cv.imwrite("debug_imgs/watershed-seeds.png", sure_fg)

    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    bg = cv.subtract(bg, sure_fg)
    markers += 1
    markers[bg==255] = 0
    markers = cv.watershed(img2, markers)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    vis = np.zeros_like(markers)
    vis[markers==-1] = 255
    vis = vis.astype(np.uint8)
    vis = cv.morphologyEx(vis, cv.MORPH_DILATE, kernel)
    if DEBUG_IMGS: cv.imwrite("debug_imgs/contour.JPG", vis)

    img[vis==255] = [255, 0 , 0]
    img[:,:,0] = cv.add(img[:,:,0], vis)
    return img

if __name__ == "__main__":
    scriptpath = dirname(realpath(sys.argv[0]))
    files = []
    if len(sys.argv) == 1:
        imgs = glob.glob(scriptpath +"/raw/*.JPG")
    else:
        imgs = [realpath(sys.argv[1])]

    for img in imgs:
        output = leaf_segment(img)
        cv.imwrite(scriptpath + "/processed/" + basename(img), output)
        msg = "writing file to " + scriptpath + "/processed/" + basename(img)
        print(msg)



