# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 13:31:31 2020

@author: Sudeshna Bhakat
"""

import cv2 as cv
import numpy as np

def getTextOverlay(input_image):
    '''create a mask image'''
    mask = np.zeros(input_image.shape, dtype=np.uint8)
    '''create a white background image in which the text will be overlayed'''
    output=np.zeros(input_image.shape, dtype=np.uint8)
    output[:]=(255,255,255)
    '''convert the image to grayscale image'''
    img=cv.cvtColor(input_image,cv.COLOR_BGR2GRAY)
    '''binarize the image based on a threshold value, this threshold value will differ with different image'''
    ret, thresh = cv.threshold(img, 10, 255, cv.THRESH_BINARY)
    '''find the contours in the binary image'''
    contours,hierarchy=cv.findContours(thresh,cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    '''calculate the area of each contour and discard those contours which are not greater than minArea and
    not less than maxArea, keep the contours which maintain this criteria in a list called gooContours,
    this minArea and maxArea will differ with different image'''
    minArea=500
    maxArea=400000
    goodContours=list()
    for i in range(0,len(contours)):
        area=cv.contourArea(contours[i])
        if(area>minArea and area<maxArea):
            goodContours.append(contours[i])
    '''create a mask from the contours list'''
    cv.drawContours(mask, goodContours, -1, (255, 255, 255), cv.FILLED)
    '''zero the background where the text needs to be overlayed'''
    output[mask>0]=0
    '''add the input image to zeroed out space'''
    output += input_image*(mask>0)
    return output

if __name__ == '__main__':
    image = cv.imread('simpsons_frame0.png')
    output = getTextOverlay(image)
    cv.imwrite('simpons_edge1.png', output)