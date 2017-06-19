
import random
import cv2
import cv2.cv as cv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math



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
totalMouths = 14

vImgsSizes = np.zeros(totalMouths, dtype = np.int)
vPad = np.zeros(totalMouths, dtype = np.int)
vMeans = np.zeros(totalMouths, dtype = np.double)

for mth in range(totalMouths):
    if mth+1 < 10:
        fileName = str(0)+str(mth+1)+".tif"
    else: 
        fileName = str(mth+1)+".tif"
        
    filePath = "Radiographs/" + fileName 
      
    vImgsSizes[mth] = cv2.imread(filePath).flatten().shape[0]
    #vImgs[:,mth] = cv2.imread(filePath).flatten()
    
maxSize = vImgsSizes.max()
vPad = abs(vImgsSizes-maxSize)
print vPad
vImgs = np.zeros((maxSize,totalMouths), dtype = np.uint8)

for mth in range(totalMouths):
    if mth+1 < 10:
        fileName = str(0)+str(mth+1)+".tif"
    else: 
        fileName = str(mth+1)+".tif"
        
    filePath = "Radiographs/" + fileName 
    img = cv2.imread(filePath).flatten()
    padImg = np.pad(img, (0,vPad[mth]), 'constant')
    vImgs[:,mth] = padImg
    vMeans[mth] = np.mean(img)    

meansMean = np.mean(vMeans)
mulCoeffs = meansMean/vMeans

for mth in range(totalMouths):
    if mth+1 < 10:
        fileName = str(0)+str(mth+1)+".tif"
    else: 
        fileName = str(mth+1)+".tif"
        
    filePath = "Radiographs/" + fileName 
    savePath = "Radiographs/meanTest" + fileName
    
    img = cv2.imread(filePath, cv2.CV_8UC1)
    Histogram(img, True)
    img = img*mulCoeffs[mth]
    cv2.imwrite(savePath, img)
    
    

print "vMeans: ",vMeans
print "meansMean: ",meansMean
print "mulCoeffs", mulCoeffs

'''
roi = cv2.imread("ROI.jpg", cv2.CV_8UC1)
    
f_dispImages = 1

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


hTresh = 70
img = cv2.imread("preProc_test3.tif", cv2.CV_8UC1)
edges = cv2.Canny(img,hTresh/2,hTresh)
canny_result = np.copy(img)
canny_result[edges.astype(np.bool)]=0
cv2.imshow('img',canny_result)
cv2.waitKey(0)

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