'''
Testing landmark positioning
'''


import cv2
import cv2.cv as cv
import numpy as np
import scipy.cluster.vq as spclstr
import matplotlib.pyplot as plt
import math
import time


def rszDisp(img, wndwName, resizeFactor):
    cv2.namedWindow(wndwName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wndwName, img.shape[1]/resizeFactor, img.shape[0]/resizeFactor)
    cv2.imshow(wndwName,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0
    
def readToothLndmrks(mouth, tooth):
    fileName = "Landmarks\original\landmarks" + str(mouth) + "-" + str(tooth) + ".txt"
    with open(fileName) as lndmrksFile:
        rawPoints = lndmrksFile.readlines()

    points = np.zeros( (len(rawPoints)/2, 2), dtype=np.double )

    for i in range(len(rawPoints)/2):
        points[i,:] = (rawPoints[2*i],rawPoints[2*i+1]);
    
    return points
    
def dispMouthLandmarks(mouth, landmarks):
    
    if mouth < 10:
        fileName = str(0)+str(mouth+1)+".tif"
    else: 
        fileName = str(mouth+1)+".tif"
        
    filePath = "Radiographs/" + fileName
    print filePath
    
    img = cv2.imread(filePath)
    imgShape = img.shape
    resizeFactor = 2
    wndwName = 'image'
    
    for th in range(landmarks.shape[2]):
        for lm in range(landmarks.shape[0]):
            cv2.circle(img, (int(landmarks[lm,0,th]), int(landmarks[lm,1,th])), 8, cv2.cv.CV_RGB(255, 0, 0), -1, 8, 0 )
    
    rszDisp(img, wndwName, resizeFactor)


dispImgs = 0

landmarksPerTooth = 40
teethPerMouth = 8
totalMouths = 14

allLandmarks = np.zeros( (landmarksPerTooth, 2, teethPerMouth, totalMouths), dtype=np.double )

for mth in range(totalMouths):
    for th in range(teethPerMouth):
        points = readToothLndmrks(mth+1,th+1)
        allLandmarks[:,:,th,mth] = points

mouthToDisplay = 6
dispMouthLandmarks(mouthToDisplay, allLandmarks[:,:,:,mouthToDisplay])

#displaying (inpractical)
'''
for th in range(1,teethPerMouth+1):
    points = readToothLndmrks(1,th)
    allLandmarks[:,:,th-1] = points
    print th
    for i in range(points.shape[0]):
        cv2.circle(img, (int(points[i,0]), int(points[i,1])), 8, cv2.cv.CV_RGB(255, 0, 0), -1, 8, 0 )

rszDisp(img, wndwName, resizeFactor)
'''

