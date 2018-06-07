import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import basename
from preprocess import get_area


def leaf_segment(filename):
#basic input
    img = cv.imread(filename)
#preprocess
    img = get_area(img)

#g-b highlights the plants nicely
    (b, g, r) = cv.split(img)
    img2 = cv.subtract(g, b)
    # cv.imwrite("green-blue.png", img2)

#otsu threshold it to get a decent binary segmentation (but doesn't get different leaves)
#then remove noise (not sure if its necessary yet)
    ret1, otsu = cv.threshold(img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # cv.imwrite("otsu-thresh.png", otsu)
# otsu = cv.morphologyEx(otsu, cv.MORPH_OPEN, kernel, iterations=2)

#get sure background
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    bg = cv.morphologyEx(otsu, cv.MORPH_DILATE, kernel)
# bg = cv.bitwise_not(bg)

#get distance transform for watershed seeds
    dist = cv.distanceTransform(otsu, cv.DIST_L2, 3)
    dist = cv.normalize(dist,dist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    dist = dist*255
    dist = dist.astype(np.uint8)
    # cv.imwrite("dist-transform.png", dist)

#watershed seeds and labeling them
    ret, sure_fg = cv.threshold(dist, 0.4*dist.max(), 255, 0)
    ret, markers = cv.connectedComponents(sure_fg)
    # cv.imwrite("watershed-seeds.png", sure_fg)

# img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    bg = cv.subtract(bg, sure_fg)
    markers += 1
    markers[bg==255] = 0
    markers = cv.watershed(img, markers)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    vis = np.zeros_like(markers)
    vis[markers==-1] = 255
    vis = vis.astype(np.uint8)
    vis = cv.morphologyEx(vis, cv.MORPH_DILATE, kernel)
    cv.imwrite("aff.JPG", vis)

    # img[vis==255] = [255, 0 , 0]
    img[:,:,0] = cv.add(img[:,:,0], vis)
    return img

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect usage.")
        exit()

    img = leaf_segment(sys.argv[1])
    cv.imwrite("processed/" + basename(sys.argv[1]), img)
    msg = "writing file to processed/" + basename(sys.argv[1])
    print(msg)

