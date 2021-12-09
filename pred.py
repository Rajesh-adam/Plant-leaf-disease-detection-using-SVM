import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from matplotlib import pyplot as plt
%matplotlib inline
def pred(img):
    img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
    gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25,25),0)
    ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((50,50),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    
    #Shape features
    contours, _=cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
        
    #Color features
                    
    red_channel = img[:,:,0]
    green_channel = img[:,:,1]
    blue_channel = img[:,:,2]

        
    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
                    
    # std deviation
    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)
        
    gr = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    boundaries = [([30,0,0],[70,255,255])]
    for (lower, upper) in boundaries:
        mask = cv2.inRange(gr, (36, 0, 0), (70, 255,255))
        ratio_green = cv2.countNonZero(mask)/(img.size/3)
        f1=np.round(ratio_green, 2)
    f2=1-f1
                    
    #Texture features using GLCM matrix
    glcm = greycomatrix(gs, 
    distances=[1], 
    angles=[0],
    symmetric=True,
    normed=True)

    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']
    contrast = greycoprops(glcm, properties[0])
    energy = greycoprops(glcm, properties[1])
    homogeneity = greycoprops(glcm, properties[2])
    correlation = greycoprops(glcm, properties[3])
    dissimilarity = greycoprops(glcm, properties[4])
        
    vector = [area,perimeter,w,h,aspect_ratio,\
    red_mean,green_mean,blue_mean,f1,f2,red_std,green_std,blue_std,\
    contrast[0][0],energy[0][0],homogeneity[0][0],correlation[0][0],dissimilarity[0][0],label
    ]
        
    print(vector)