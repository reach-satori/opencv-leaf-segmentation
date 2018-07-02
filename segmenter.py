import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from consts import *

#Takes in a greyscale image and returns that greyscale image histogram-equalized (excluding all input 0 pixels)
def mask_hist_eq(mat):
    hist = cv.calcHist([mat], [0], mat, [256], [0, 256]).ravel().astype(np.uint8)
    lut = hist.cumsum()
    lut = 255 * lut / lut[-1]
    np.round(lut).astype(np.uint8)
    mat = cv.LUT(mat, lut)
    return mat


class Segmenter(object):
    def __init__(self, img):
        #least modified necessary image at each step
        self.proc_img = img
        self.grid_scale = None
        self.segmentation_map = None

        #threshold to ignore contours during various contour manipulations
        self.MIN_CONTOUR_AREA = MIN_CONTOUR_AREA
        #output debug images, yes or no
        self.DEBUG_IMGS = DEBUG_IMGS
        #brightness percentile below which pixels in the get_grid fourier transform get ignored
        self.FOURIER_PERCENTILE = FOURIER_PERCENTILE
        #Canny detections thresholds for get_grid
        self.GRID_CANNY_LOWER = GRID_CANNY_LOWER
        self.GRID_CANNY_UPPER = GRID_CANNY_UPPER

    #not so good approach
    #histeq -> median blur -> canny -> fft -> threshold -> ifft ...
    #there has to be a better way
    def get_grid(self):
        PERCENTILE = 99.
        f = self.proc_img.copy()

        f = cv.split(f)
        for channel in f:
            channel = cv.equalizeHist(channel)
        f = cv.merge(f)

        # f = cv.medianBlur(f, 3)
        f = cv.Canny(f, self.GRID_CANNY_LOWER, GRID_CANNY_UPPER)
        # fourier transform the edge detection
        f = np.fft.fft2(f)
        absfft = np.abs(f)
        # only let pixels brighter than FOURIER_PERCENTILE through
        perc = np.percentile(absfft, self.FOURIER_PERCENTILE)
        f[absfft < perc] = 0
        f = np.abs(np.fft.ifft2(f))
        f = np.interp(f, (f.min(), f.max()), (0., 255.))
        f = f.astype(np.uint8)
        # at this point f should be the canny detected image with only the grid (high freq) highlighted
        # (+ some noise due to rounding)

        _, f = cv.threshold(f, 170, 255, cv.THRESH_BINARY)
        f = cv.blur(f, (20, 20))
        _, f = cv.threshold(f, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        f = cv.morphologyEx(f, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7)), iterations=2)
        _, cnts, __ = cv.findContours(f, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = list(filter(lambda x: cv.contourArea(x) > self.MIN_CONTOUR_AREA, cnts))
        cnts = np.array(sorted(cnts, key=cv.contourArea, reverse=True))

        mask = np.zeros_like(f)
        cv.drawContours(mask, cnts, 0, 255, -1)
        f = cv.bitwise_and(self.proc_img, self.proc_img, mask=mask)
        cnt = cnts[0]
        (x, y, w, h) = cv.boundingRect(cnt)
        f = f[y:y+h, x:x+w]
        f = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        f = mask_hist_eq(f)
        # self.proc_img=f
        self.grid_scale = f

    def run(self):
        self.get_grid()
        #self.proc_img (usually) gets modified at each stage
        # self.get_soil_area()
        # self.get_plant_area()
        # self.get_segmentation_map()
        # self.get_visualization()
        return self.grid_scale

    def get_visualization(self):
        self.proc_img[self.segmentation_map==255] = [255, 0 , 0]
        self.proc_img[:,:,0] = cv.add(self.proc_img[:,:,0], self.segmentation_map)


    def get_segmentation_map(self):
        #cut out leaves
        (b, g, r) = cv.split(self.proc_img)
        g_b = cv.subtract(g, b)
        g_b = cv.normalize(g_b, g_b, 0, 255, cv.NORM_MINMAX, -1)
        _,  otsu = cv.threshold(g_b, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        otsu = cv.morphologyEx(otsu, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7)))

        #remove noise from contours
        _, cnts, __ = cv.findContours(otsu, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cnts = list(filter(lambda x: cv.contourArea(x) > self.MIN_CONTOUR_AREA, cnts))
        cnts = np.array(sorted(cnts, key=cv.contourArea, reverse=True))

        #redraw contours, leaving out noise
        otsu = np.zeros_like(otsu)
        cv.drawContours(otsu, cnts, -1, 255, -1)
        g_b = cv.bitwise_and(g_b, g_b, mask=otsu)
        if self.DEBUG_IMGS: cv.imwrite("debug_imgs/otsu-thresh.png", g_b)

        #get sure bg
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
        bg = cv.morphologyEx(otsu, cv.MORPH_DILATE, kernel)

        #get distance transform for watershed seeds
        dist = cv.distanceTransform(otsu, cv.DIST_L2, 3)
        dist = cv.normalize(dist, dist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        dist *= 255
        dist = dist.astype(np.uint8)
        if self.DEBUG_IMGS: cv.imwrite("debug_imgs/dist-transform.png", dist)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        _, sure_fg = cv.threshold(dist, 0.6*dist.max(), 255, 0)

        #cut out barriers from watershed seeds
        canny = g_b.copy()
        canny = cv.Canny(canny, 70, 255)
        canny = cv.morphologyEx(canny, cv.MORPH_DILATE, kernel, iterations=2)
        sure_fg = cv.subtract(sure_fg, canny)

        _, markers = cv.connectedComponents(sure_fg)
        if self.DEBUG_IMGS: cv.imwrite("debug_imgs/watershed-seeds.png", sure_fg)

        g_b = cv.cvtColor(g_b, cv.COLOR_GRAY2BGR)
        bg = cv.subtract(bg, sure_fg)
        markers += 1
        markers[bg==255] = 0
        markers = cv.watershed(g_b, markers)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        vis = np.zeros_like(markers)
        vis[markers==-1] = 255
        vis = vis.astype(np.uint8)
        vis = cv.morphologyEx(vis, cv.MORPH_DILATE, kernel)
        if self.DEBUG_IMGS: cv.imwrite("debug_imgs/contour.JPG", vis)
        self.segmentation_map = vis


    def get_plant_area(self):
        b, g, r = cv.split(self.proc_img)
        g_b = cv.subtract(g, b)
        _, otsu = cv.threshold(g_b,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        otsu = cv.morphologyEx(otsu, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7)), iterations=2)

        #filter noise to improve sorting speed, sort from largest to smallest
        _, cnts, __ = cv.findContours(otsu, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cnts = list(filter(lambda x: cv.contourArea(x) > self.MIN_CONTOUR_AREA, cnts))
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)

        #bad... just gets the biggest contour again, failure-prone
        plantcnt = cnts[0]

        x, y, w, h = cv.boundingRect(plantcnt)
        self.proc_img = self.proc_img[y:y+h, x:x+w]
        if self.DEBUG_IMGS: cv.imwrite("debug_imgs/plant_only.png", self.proc_img)


    def get_soil_area(self):
        #get HSV, do s-v, threshold, dilate to remove noise
        proc = cv.cvtColor(self.proc_img, cv.COLOR_BGR2HSV)
        (h, s, v) = cv.split(proc)
        proc = cv.subtract(s, v)
        ret, proc = cv.threshold(proc, 255//2, 255, cv.THRESH_BINARY)
        proc = cv.morphologyEx(proc, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)), iterations=4)
        proc = cv.morphologyEx(proc, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)), iterations=3)

        #remove small contours and sort from biggest to smallest
        _, cnts, __ = cv.findContours(proc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = list(filter(lambda x: cv.contourArea(x) > self.MIN_CONTOUR_AREA, cnts))
        cnts = np.array(sorted(cnts, key=cv.contourArea, reverse=True))

        #failure point: just gets the biggest contour, crude approach
        x, y, w, h = cv.boundingRect(cnts[0])
        cropped = self.proc_img[y:y+h, x:x+w]
        for i in cnts[0]:
            i[0][0] -= x
            i[0][1] -= y
        mask = np.zeros((h, w), np.uint8)
        cv.drawContours(mask, cnts, 0, 255, -1)

        #output
        self.proc_img = cv.bitwise_and(cropped, cropped, mask=mask)
        if self.DEBUG_IMGS: cv.imwrite("debug_imgs/soil_only.png", self.proc_img)

