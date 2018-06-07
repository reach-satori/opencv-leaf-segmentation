import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from preprocess import get_area


#basic input
img = cv.imread("raw.JPG")
#preprocess
img = get_area(img)

#g-b highlights the plants nicely
(b, g, r) = cv.split(img)
img2 = cv.subtract(g, b)
cv.imwrite("green-blue.png", img2)

#otsu threshold it to get a decent binary segmentation (but doesn't get different leaves)
#then remove noise (not sure if its necessary yet)
ret1, otsu = cv.threshold(img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imwrite("otsu-thresh.png", otsu)
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
cv.imwrite("dist-transform.png", dist)

#watershed seeds and labeling them
ret, sure_fg = cv.threshold(dist, 0.4*dist.max(), 255, 0)
ret, markers = cv.connectedComponents(sure_fg)
cv.imwrite("watershed-seeds.png", sure_fg)

# img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
bg = cv.subtract(bg, sure_fg)
markers += 1
markers[bg==255] = 0
markers = cv.watershed(img, markers)
img[markers==-1] = [255, 0 , 0]
plt.imshow(img)
plt.show()
# img[markers == -1] = [255,0,0]

