# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:30:08 2017

@author: iCV
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:04:00 2017

@author: iCV
"""

# -*- coding: utf-8 -*-
#
#      Day 2 Demo 1:
#   Car lane detection:
#       Colorspaces
#       Thresholding
#       Line detection

#import important libraries
import numpy as np
import cv2
import cv2.ximgproc as ximgproc

#Open camera session
cap = cv2.VideoCapture(0)
#cap.release()
cv2.namedWindow("frame",cv2.WINDOW_AUTOSIZE)




def nothing(*arg):
    pass
cv2.createTrackbar('Number of Superpixels', 'frame', 301, 1000, nothing)
cv2.createTrackbar('Iterations', 'frame', 4, 12, nothing)

seeds = None
display_mode = 0
num_superpixels = 301
prior = 2
num_levels = 4
num_histogram_bins = 5





#detector = cv2.SimpleBlobDetector()
#try:

#start reading frames
#while(cap.isOpened()):
    
    #read frame
 #   ret, frame = cap.read()
    
    #end if no frame detected
  #  if (not ret): 
   #     break

frame = cv2.imread("debug_imgs/plant_only.png")
    
while cv2.waitKey(100):
    num_superpixels = cv2.getTrackbarPos('Number of Superpixels', 'frame')
    num_iterations = cv2.getTrackbarPos('Iterations', 'frame')
    converted_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height,width,channels = converted_img.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
    color_img = np.zeros((height,width,3), np.uint8)
    color_img[:] = (0, 0, 255)
    seeds.iterate(converted_img, num_iterations)
    labels = seeds.getLabels()
    # labels output: use the last x bits to determine the color
    
    
    num_label_bits = 2
    #labels &= (1<<num_label_bits)-1
    #labels *= 1<<(16-num_label_bits)
    mask = seeds.getLabelContourMask(False)
    # stitch foreground & background together
    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
    result = cv2.add(result_bg, result_fg)

    cv2.imshow('frame',result)


    #Stop and exit on 'Esc'
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

#except:
#Release camera and close all windows
#    cap.release()
#    cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
