
import random
import cv2
import cv2.cv as cv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import argparse
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.namedWindow(wndwName, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(wndwName, img.shape[1]/resizeFactor, img.shape[0]/resizeFactor)
		cv2.imshow(wndwName, img)

 


def rszDisp(img, wndwName, resizeFactor):
    cv2.namedWindow(wndwName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wndwName, img.shape[1]/resizeFactor, img.shape[0]/resizeFactor)
    cv2.imshow(wndwName,img)
    return 0
    
def Histogram(img, dispPlot):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    
    if dispPlot == True:
        plt.figure()
        plt.plot(cdf_normalized, color = 'b')
        plt.hist(img.flatten(),256,[0,256], color = 'r')
        plt.xlim([0,256])
        plt.legend(('cdf','histogram'), loc = 'upper left')
        plt.show()
    
    return cdf
    
    
##MAIN

f_cropNew = 0

#
#Cropping Phase
#
if f_cropNew == 1:
    # load the image, clone it, and setup the mouse callback function
    img = cv2.imread("Radiographs/04.tif", cv2.CV_8UC1)
    clone = img.copy()
    
    resizeFactor = 2
    
    wndwName = "image"
    cv2.namedWindow(wndwName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wndwName, img.shape[1]/resizeFactor, img.shape[0]/resizeFactor)
    cv2.imshow(wndwName,img)
    cv2.setMouseCallback(wndwName, click_and_crop)
    
    # keep looping until the 'q' key is pressed
    while True:
   	# display the image and wait for a keypress
   	cv2.namedWindow(wndwName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wndwName, img.shape[1]/resizeFactor, img.shape[0]/resizeFactor)
   	cv2.imshow(wndwName, img)
   	key = cv2.waitKey(1) & 0xFF
    
   	# if the 'r' key is pressed, reset the cropping region
   	if key == ord("r"):
  		image = clone.copy()
    
   	# if the 'c' key is pressed, break from the loop
   	elif key == ord("c"):
  		break
    
    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(refPt) == 2:
   	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
   	cv2.imshow("ROI", roi)
   	cv2.imwrite("ROI.jpg", roi)
   	cv2.waitKey(0)
    
    # close all open windows
    cv2.destroyAllWindows()
    

if f_cropNew == 0:
    roi = cv2.imread("ROI.jpg", cv2.CV_8UC1)
    
f_dispImages = 1

img = cv2.imread("Radiographs/01.tif", cv2.CV_16UC1)
Histogram(img, True)

cv2.imshow('ciao', img)
Histogram(img, True)


raw_img = roi
hSt = np.zeros((roi.shape[0],1),np.uint8)

img = raw_img
#hSt = np.hstack((hSt,img))

#noise removal
flt_img = cv2.GaussianBlur(img, (5,5), 10.0)
img = flt_img
hSt = np.hstack((hSt,img))

#instogram equalization
cdf = Histogram(img, True)
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
cl_img = cdf[img]
img = cl_img
hSt = np.hstack((hSt,img))

'''
clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(16,16))
cl_img = clahe.apply(img)
img = cl_img
hSt = np.hstack((hSt,img))


#unsharp masking
gaussian_3 = cv2.GaussianBlur(img, (55,55), 50.0)
us_img = cv2.addWeighted(img, 1, gaussian_3, -1, 0, img) 
img = us_img
hSt = np.hstack((hSt,img))

#instogram equalization
clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
cl2_img = clahe2.apply(img)
img = cl2_img
hSt = np.hstack((hSt,img))

cv2.imshow('clahe_2.jpg',hSt)
'''

hTresh = 70
img = cv2.imread("preProc_test3.tif", cv2.CV_8UC1)
edges = cv2.Canny(img,hTresh/2,hTresh)
canny_result = np.copy(img)
canny_result[edges.astype(np.bool)]=0
#cv2.imshow('img',canny_result)
cv2.waitKey(0)
'''
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side

cv2.imshow('res.png',res)


gaussian_3 = cv2.GaussianBlur(img, (45,45), 50.0)
cv2.imshow("ROI_blurred", gaussian_3)

unsharp_img = cv2.addWeighted(img, 2, gaussian_3, -1, 0, img)
cv2.imshow("ROI_unsharpMask", unsharp_img)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()