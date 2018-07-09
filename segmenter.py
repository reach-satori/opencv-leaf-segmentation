import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from consts import *
from math import ceil

#Takes in a greyscale image and returns that greyscale image histogram-equalized (excluding all input 0 pixels)
def mask_hist_eq(mat):
    hist = cv.calcHist([mat], [0], mat, [256], [0, 256]).ravel().astype(np.uint8)
    lut = hist.cumsum()
    lut = 255 * lut / lut[-1]
    lut = np.round(lut).astype(np.uint8)
    mat = cv.LUT(mat, lut)
    return mat

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    rot_mat = cv.getRotationMatrix2D((nw/2., nh/2.), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw-w)/2., (nh-h)/2.,0]))
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv.warpAffine(src, rot_mat, (int(ceil(nw)), int(ceil(nh))), flags=cv.INTER_LANCZOS4)


class Segmenter(object):

    class RotatedRect(object):
        #this convenience object is meant to be initiated with the return from cv2.minAreaRect
        #it's basically just a POD struct
        #it's initiated here because it's not meant to be reused outside Segmenter
        # ((center_x, center_y), (width, height), angle)
        def __init__(self, rect_tuple):
            self.center = rect_tuple[0]
            self.width = rect_tuple[1][1]
            self.height = rect_tuple[1][0]
            self.angle = rect_tuple[2]

        def get_points(self):
            cx = self.center[0]
            cy = self.center[1]
            vertices = [
                    [-self.width/2, -self.height/2],
                    [-self.width/2, self.height/2],
                    [self.width/2,  self.height/2],
                    [self.width/2,  -self.height/2],
                    ]

            for pt in vertices:
                temp0 = pt[0]*np.cos(self.angle) - pt[1]*np.sin(self.angle) + cx
                temp1 = pt[0]*np.sin(self.angle) + pt[1]*np.cos(self.angle) + cy
                pt[0] = temp0
                pt[1] = temp1

            return tuple(vertices)


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

        f = cv.Canny(f, self.GRID_CANNY_LOWER, self.GRID_CANNY_UPPER)
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
        #at this point f is (with about 90% reliability) the milimiter paper grid
        # cropped out, masked out and histogram equalized
        self.grid_scale = f

    def get_scale(self):
        # first step to get scale from equalized grid image is to remove the border squares that are likely to be malformed
        # print(self.grid_scale.dtype, self.grid_scale.shape)
        overlap_outer = self.grid_scale.copy()
        _, overlap_outer = cv.threshold(overlap_outer, 1, 255, cv.THRESH_BINARY)
        overlap_outer = cv.morphologyEx(overlap_outer, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)))
        overlap_outer = cv.bitwise_not(overlap_outer)
        overlap_outer = cv.morphologyEx(overlap_outer, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))

        overlap_inner = self.grid_scale.copy()
        overlap_inner = cv.adaptiveThreshold(overlap_inner, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, 0)
        overlap = cv.bitwise_and(overlap_inner, overlap_outer)

        _, labels = cv.connectedComponents(overlap_inner, connectivity = 4)
        discarded = np.unique(labels[overlap != 0])
        flattened = labels.flatten()
        for i, pix in enumerate(flattened):
            if pix in discarded:
                flattened[i] = 0
        labels = flattened.reshape(labels.shape).astype(np.uint8)

        _, cnts, __ = cv.findContours(labels, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # here rectangles should be sufficiently well-formed to be a dataset of sorts
        # after just a small filtering out of outliers:
        #first remove non-square contours
        square_threshold = 1.25
        #solve division by 0
        cnts[:] = [cnt for cnt in cnts if cv.minAreaRect(cnt)[1][1] != 0]
        cnts[:] = [cnt for cnt in cnts if abs(1-cv.minAreaRect(cnt)[1][0]/cv.minAreaRect(cnt)[1][1]) < square_threshold]

        # discard contours with area "m"+ sigma away from mean
        areas = np.array([cv.contourArea(x) for x in cnts])
        m = 3
        cnts[:] = [cnt for cnt in cnts if abs(cv.contourArea(cnt)-np.mean(areas)) < m * np.std(areas)]

        #get the angle the grid is at by applying HoughLines to a Sobel detection in one direction only
        anglemeasure = np.zeros_like(labels)
        cv.drawContours(anglemeasure, cnts, -1, 255, -1)
        labels = anglemeasure.copy()
        anglemeasure = cv.Scharr(anglemeasure, -1, 0, 1)
        gridangle = 0
        for linethresh in range(300, 0, -20):
            lines = cv.HoughLines(anglemeasure, 1, np.pi/360, linethresh)
            # if lines is not None:print(lines.shape)
            # else:print(linethresh)
            if lines is None or lines.shape[0] < 3:
                continue
            gridangle = np.median(np.squeeze(lines)[:, 1])
            # visualization:
            # for rho,theta in lines:
            #     a = np.cos(theta)
            #     b = np.sin(theta)
            #     x0 = a*rho
            #     y0 = b*rho
            #     x1 = int(x0 + 1000*(-b))
            #     y1 = int(y0 + 1000*(a))
            #     x2 = int(x0 - 1000*(-b))
            #     y2 = int(y0 - 1000*(a))
            #     cv.line(labels,(x1,y1),(x2,y2),127,1)
            break

        labels = rotate_about_center(labels, gridangle)
        self.grid_scale = labels



    def run(self):
        self.get_grid()
        self.get_scale()

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
        #necessarily should have just one contour because that's how we get the grid in the first place
        cnt = cnts[0]

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

        _, markers = cv.connectedComponents(sure_fg, connectivity=4)
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

