# -*- coding: utf-8 -*-
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
import numpy.linalg as npLA
import math
import time

#GLOBAL CONSTANTS
landmarksPerTooth = 40
teethPerMouth = 8
totalMouths = 14

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
    #initShapePos = random.randint(0, 13)
    initShapePos = 0
    
    #initialize point matrices
    ComparedShape = specTooth[:,:,initShapePos]
    sumShape = np.zeros( (40, 2), dtype=np.double )
    normalizedShapes = np.zeros( (40, 2, 14), dtype=np.double )
    
    #initialize variables for checking the disparity reduction
    prev_meanDisp = 0
    dispReduct = 1
    cyc = 1
     
    #cycles through until the difference beetween the mean disparity of the cicle i and the mean disparity of the cycle i-1
    #gets smaller than the selected treshold
    thresh = 1e-06
    
    
    while abs(dispReduct) > thresh:
        
        sumShape = np.zeros( (40, 2), dtype=np.double )
        sumDisp = 0
        
        for inst in range(13):
            [stdMat1, orientation, disparity] = spsptl.procrustes(ComparedShape,specTooth[:,:,inst])
            #normalizedShapes[:,:,inst] = stdMat1
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

def NormalizeDataset(Landmarks, meanShapes):
    
    global teethPerMouth
    global totalMouths
    
    normalizedLandmarks = np.zeros( (landmarksPerTooth, 2, teethPerMouth, totalMouths), dtype=np.double )
    
    for tooth in range(teethPerMouth):
        for inst in range(totalMouths):
            [mat1,normalizedLandmarks[:,:,tooth,inst], disparity] = spsptl.procrustes(meanShapes[:,:,tooth],Landmarks[:,:,tooth,inst])
    #print "orientation",orientation
        
    return normalizedLandmarks
  
def pca(X, nb_components=0):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    X = np.transpose(X)
    
    [n,d] = X.shape
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n

    #get image size
    imgSz = X.shape[1]
    
    #initialize result variables
    largestEigenVals = np.zeros(nb_components)
    largestEigenVects = np.zeros((imgSz,nb_components))
    
    #Compute mean image
    mu = np.mean(X, axis=0)
    
    #since computing the covariance matrix takes too much time, we can 
    #approximate it by multiplying X by X transposed, then normalizing it
    Xcov = np.dot(X, np.transpose(X))/(X.shape[1])

    #compute eigenvalues and eigenvectors
    eigenVals, eigenVects = npLA.eig(Xcov)

    #compute actual eigenvectors
    eigenVects = np.dot(np.transpose(X),eigenVects)
    
    #define a data type to sort the eigenvalues and vectors
    #based on eigenvalues magnitude
    Eigen_dType = np.dtype(  [('evals',np.float),('evects', np.float, (imgSz) )]  )
    eigensList = np.zeros(n, dtype = Eigen_dType)
    for i in xrange(n):
        eigensList[i] = (eigenVals[i],eigenVects[:,i])
    
    #sort eigenvalues and vectors by descending order
    eigensList = np.sort(eigensList,order = ['evals'])
    
    #keep only the bigger eigenvectors and normalize them
    for i in xrange(nb_components):
        largestEigenVals[i] = eigensList[i][0]
        largestEigenVects[:,i] = eigensList[i][1] / npLA.norm(eigensList[i][1])
    
    return largestEigenVals, largestEigenVects, mu  
    
    
def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    result = np.dot(X-mu,W)
    
    return result

def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    result = np.dot(W,Y)+mu

    return result

def dispProjection(eigVects, sample, mu, inter):
    x = project(eigVects, sample, mu)
    y = reconstruct(eigVects, x, mu)
    
    if inter == True:
        plt.plot(sample[0::2], sample[1::2], 'ro', hold = True)
        plt.plot(y[0::2], y[1::2], 'yo', hold = True)
        plt.pause(0.05)
        plt.show()
    else:
        plt.plot(sample[0:40], sample[40:80], 'ro', hold = True)
        plt.plot(y[0:40], y[40:80], 'go', hold = True)
        plt.pause(0.05)
        plt.show()

    return y
    
def reshapeDataset(dataset):
    global teethPerMouth
    global totalMouths
    
    res = np.zeros((80,teethPerMouth,totalMouths), dtype = np.double)
    
    for inst in range(totalMouths):
        for th in range(teethPerMouth):
            res[:,th,inst] = np.transpose(np.hstack((dataset[:,0,tooth,inst],dataset[:,1,tooth,inst])))
    
    return res
    
def reshapeInterleaved(dataset):
    global teethPerMouth
    global totalMouths
    
    temp_Vect = np.zeros((80,teethPerMouth,totalMouths), dtype = np.double)
    res= np.zeros((80,teethPerMouth,totalMouths), dtype = np.double)
    
    for inst in range(totalMouths):
        for th in range(teethPerMouth):
            temp_Vect[:,th,inst] = np.transpose(np.hstack((dataset[:,0,tooth,inst],dataset[:,1,tooth,inst])))
            res[0::2,th,inst] = temp_Vect[0:40,th,inst]
            res[1::2,th,inst] = temp_Vect[40:80,th,inst]
    return res


#
# MAIN
#

dispImgs = 0

originalLandmarks = np.zeros( (landmarksPerTooth, 2, teethPerMouth, totalMouths), dtype=np.double )
mirroredLandmarks = np.zeros( (landmarksPerTooth, 2, teethPerMouth, totalMouths), dtype=np.double )

normalizedLandmarks = np.zeros( (landmarksPerTooth, 2, teethPerMouth, totalMouths), dtype=np.double )

for mth in range(totalMouths):
    for th in range(teethPerMouth):
        points = readToothLndmrks(mth+1,th+1, 'original')
        originalLandmarks[:,:,th,mth] = points
        points = readToothLndmrks(mth+1,th+1, 'mirrored')
        mirroredLandmarks[:,:,th,mth] = points


#obtain normalized shapes for every tooth
specTooth = np.zeros( (landmarksPerTooth, 2, 1, totalMouths), dtype=np.double )
outShape = np.zeros( (landmarksPerTooth, 2, 1, 1), dtype=np.double )
#specTooth[:,:,0,:] = originalLandmarks[:,:,2,:]
#[normalizedShapes, meanDisp] = GPA(originalLandmarks[:,:,2,:], False)

meanShapes = np.zeros((landmarksPerTooth,2,teethPerMouth), dtype=np.double)

for tooth in range (teethPerMouth):
    print "tooth n° ", tooth+1, ":"
    [meanShapes[:,:,tooth], meanDisp] = GPA(originalLandmarks[:,:,tooth,:], False)
    print "\n"

dispMean = 0
if dispMean == 1:
    for tooth in range (teethPerMouth):
        dispx = ((math.floor(tooth)/4-(tooth//4))*1.8)-0.9
        dispy = (tooth//4)*(-0.6)+0.3
        plt.axis([-1.2, 1.2, -0.6, 0.6])
        plt.plot(meanShapes[:,0,tooth]+dispx, meanShapes[:,1,tooth]+dispy, 'ro', hold = True)
        plt.pause(0.05)
        plt.show()

normalizedLandmarks = NormalizeDataset(originalLandmarks, meanShapes)

#print normalizedShapes
dispPlots = 0
if dispPlots == 1:
    for tooth in range (teethPerMouth):
        dispx = ((math.floor(tooth)/4-(tooth//4))*1.8)-0.9
        dispy = (tooth//4)*(-0.6)+0.3
        for inst in range (totalMouths):
            plt.axis([-1.2, 1.2, -0.6, 0.6])
            plt.plot(normalizedLandmarks[:,0,tooth,inst]+dispx, normalizedLandmarks[:,1,tooth,inst]+dispy, 'ro', hold = True)
            plt.pause(0.05)
            plt.show()
#           
#do PCA  
#  
consComps = 4
consComps2 = 13
isInterleaved = False

reshapedLandmarks = np.zeros((80,teethPerMouth,totalMouths), dtype = np.double)
ireshapedLandmarks = np.zeros((80,teethPerMouth,totalMouths), dtype = np.double)

eigVects = np.zeros((80,consComps,teethPerMouth), dtype = np.double)
eigVals = np.zeros((consComps,teethPerMouth), dtype = np.double)
mus = np.zeros((80,teethPerMouth), dtype = np.double)

ieigVects = np.zeros((80,consComps2,teethPerMouth), dtype = np.double)
ieigVals = np.zeros((consComps2,teethPerMouth), dtype = np.double)
imus = np.zeros((80,teethPerMouth), dtype = np.double)

'''
if isInterleaved:
    reshapedLandmarks = reshapeInterleaved(normalizedLandmarks)
else:    
    reshapedLandmarks = reshapeDataset(normalizedLandmarks)

ireshapedLandmarks = reshapeInterleaved(normalizedLandmarks)
'''
reshapedLandmarks = reshapeDataset(normalizedLandmarks)
ireshapedLandmarks = reshapeInterleaved(normalizedLandmarks)

print "reshaped dataset shape", reshapedLandmarks.shape

#pca on every tooth separately
for th in range(teethPerMouth):
    eigVals,eigVects[:,:,th], mus[:,th] = pca(reshapedLandmarks[:,th,:], nb_components = consComps)


for th in range(teethPerMouth):
    ieigVals,ieigVects[:,:,th], imus[:,th] = pca(ireshapedLandmarks[:,th,:], nb_components = consComps2)


print "eigVals", eigVals.shape
print "eigVects matrix shape", eigVects.shape
print "mu shape", mus.shape


f_dispProj = 1
testTh = 0
testInst = 2
if f_dispProj == 1:
    dispProjection(eigVects[:,:,testTh],reshapedLandmarks[:,testTh,testInst], mus[:,testTh], False)
    dispProjection(ieigVects[:,:,testTh],ireshapedLandmarks[:,testTh,testInst], imus[:,testTh], True)




#x and y values working, not interleaved
'''   
pcaInput = np.zeros((80, 14), dtype = np.double)

for inst in range(totalMouths):
    pcaInput[:,inst] = np.transpose(np.hstack((normalizedLandmarks[:,0,0,inst],normalizedLandmarks[:,1,0,inst])))
    
print pcaInput.shape
plt.plot(pcaInput[0:40,0], pcaInput[40:80,0], 'ro', hold = True)
plt.show()      
      
compToConsider = 6
eigVals, eigVects, mu = pca(pcaInput,nb_components = compToConsider)
print "eigvects shape",eigVects.shape      

Ya = project(eigVects, pcaInput[:,0], mu )
print Ya
print "Ya shape", Ya.shape
Xa = reconstruct(eigVects, Ya, mu)      

print "Xa shape",Xa.shape

plt.plot(Xa[0:40], Xa[40:80], 'bo', hold = True)
plt.show()
'''

#only x values, working          
'''
xVals = normalizedLandmarks[:,0,0,:]
print xVals.shape

plt.plot(xVals[:,0], np.linspace(0,39,40), 'ro', hold = True)
plt.show()

compToConsider = 6
eigVals, eigVects, mu = pca(xVals,nb_components = compToConsider)
print "eigvects shape",eigVects.shape

#plt.plot(np.sum(eigVects,1), np.linspace(0,39,40), 'ro', hold = True)
plt.plot(eigVects[:,2], np.linspace(0,39,40), 'ro', hold = True)
plt.show()

Ya = project(eigVects, normalizedLandmarks[:,0,0,0], mu )
print Ya
print "Ya shape", Ya.shape
Xa = reconstruct(eigVects, Ya, mu)

print "Xa shape",Xa.shape

plt.plot(Xa, np.linspace(0,39,40), 'bo', hold = True)
plt.show()
'''

