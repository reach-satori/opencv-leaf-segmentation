import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



class Segmenter(object):
    def __init__(self, img):
        #least modified necessary image at each step
        self.proc_img = img
        self.segmentation_map = None
        self.MIN_CONTOUR_AREA = 800
        self.DEBUG_IMGS = False

    def run(self):
        #self.proc_img gets modified after each stage
        self.get_soil_area()
        self.get_leaves()
        self.get_segmentation_map()
        self.get_visualization()
        return self.proc_img

    def get_visualization(self):
        self.proc_img[self.segmentation_map==255] = [255, 0 , 0]
        self.proc_img[:,:,0] = cv.add(self.proc_img[:,:,0], self.segmentation_map)


    def get_segmentation_map(self):
        ####### cut out leaves ########
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
        _, sure_fg = cv.threshold(dist, 0.4*dist.max(), 255, 0)
        canny = g_b.copy()
        canny = cv.Canny(canny, 100, 200)
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


    def get_leaves(self):
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

