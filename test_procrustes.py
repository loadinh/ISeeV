'''
Testing landmark positioning
'''

import random
import cv2
import cv2.cv as cv
import numpy as np
import scipy as sp
import scipy.spatial as spsptl
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
    
def readToothLndmrks(mouth, tooth, isMirrored):
    if isMirrored == 'mirrored':
        mouth = mouth + 14
    
    
    fileName = "Landmarks/" + isMirrored + "/landmarks" + str(mouth) + "-" + str(tooth) + ".txt"
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
    #print filePath
    
    img = cv2.imread(filePath)
    resizeFactor = 2
    wndwName = 'image'
    
    #print landmarks
    
    for th in range(landmarks.shape[2]):
        for lm in range(landmarks.shape[0]):
            cv2.circle(img, (int(landmarks[lm,0,th]), int(landmarks[lm,1,th])), 8, cv2.cv.CV_RGB(255, 0, 0), -1, 8, 0 )
    
    rszDisp(img, wndwName, resizeFactor)

def GPA(specTooth, disp):
    
    #specTooth is a matrix containing the shapes of the 14 teeth in a specific position, 40x2x1x14 matrix
    #keeping the third dimension, more practical to handle
    
    #choose initial reference shape randomly
    initShapePos = random.randint(1, 14)
    
    #initialize point matrices
    res = np.zeros( (40, 2, 1, 1), dtype=np.double )
    ComparedShape = specTooth[:,:,0,initShapePos]
    sumShape = np.zeros( (40, 2), dtype=np.double )
    
    #initialize variables for checking the disparity reduction
    prev_meanDisp = 0
    dispReduct = 1
     
    #cycles through until the difference beetween the mean disparity of the cicle i and the mean disparity of the cycle i-1
    #gets smaller than the selected treshold
    thresh = 1e-06
    
    while abs(dispReduct) > thresh:
        
        cyc = 1
        sumShape = np.zeros( (40, 2), dtype=np.double )
        sumDisp = 0
        
        for inst in range(13):
            [stdMat1, orientation, disparity] = spsptl.procrustes(ComparedShape,specTooth[:,:,0,inst])
            sumShape = sumShape + orientation
            sumDisp = sumDisp + disparity
            if disp:
                plt.axis([-0.3, 0.3, -0.3, 0.3])
                plt.plot(orientation[:,0], orientation[:,1], 'ro', hold = True)
                plt.pause(0.05)
                plt.show()
                
        meanShape = sumShape / 14
        meanDisp = sumDisp / 14  
        dispReduct = meanDisp-prev_meanDisp
        
        print 'disparity reduction, cycle ', cyc, ': ',dispReduct   
        print 'mean disparity, cycle ', cyc, ': ',meanDisp
            
        if disp:    
            plt.plot(stdMat1[:,0], stdMat1[:,1], 'bo', hold = True)
            plt.plot(meanShape[:,0], meanShape[:,1], 'yo', hold = True)
            plt.pause(0.05)
            plt.show()
            plt.figure()
            plt.axis([-0.3, 0.3, -0.3, 0.3])
        
        
        ComparedShape = meanShape
        prev_meanDisp = meanDisp
        cyc += 1
        
    return [meanShape, meanDisp]




#
# MAIN
#

dispImgs = 0

landmarksPerTooth = 40
teethPerMouth = 8
totalMouths = 14

originalLandmarks = np.zeros( (landmarksPerTooth, 2, teethPerMouth, totalMouths), dtype=np.double )
mirroredLandmarks = np.zeros( (landmarksPerTooth, 2, teethPerMouth, totalMouths), dtype=np.double )

for mth in range(totalMouths):
    for th in range(teethPerMouth):
        points = readToothLndmrks(mth+1,th+1, 'original')
        originalLandmarks[:,:,th,mth] = points
        points = readToothLndmrks(mth+1,th+1, 'mirrored')
        mirroredLandmarks[:,:,th,mth] = points
''' 
#compute mean shape
specTooth = np.zeros( (landmarksPerTooth, 2, 1, totalMouths), dtype=np.double )
specTooth[:,:,0,:] = originalLandmarks[:,:,2,:]
print specTooth.shape
meanShape = (np.sum(specTooth, axis = 3))/14
'''



specTooth = np.zeros( (landmarksPerTooth, 2, 1, totalMouths), dtype=np.double )
outShape = np.zeros( (landmarksPerTooth, 2, 1, 1), dtype=np.double )
specTooth[:,:,0,:] = originalLandmarks[:,:,2,:]
[meanShape, meanDisp] = GPA(specTooth, True)
outShape[:,:,0,0] = meanShape




