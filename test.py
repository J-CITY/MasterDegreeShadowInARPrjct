import random
import os
import cv2
import numpy as np

import dbrar.dbrar as dbrar
import ADNET.adnet as adnet
import ycbcr
import lab
import hsi

hsiModel = hsi.HSI()
dbrarModel = dbrar.DBRAR()
ycbcrModel = ycbcr.YCBCR()
labModel = lab.LAB()
adnetModel = adnet.ADNET()

originalImagesPath = "C:/Users/333da/Desktop/STCGANs/ISTD_Dataset/test/test_A/"
originalMasksPath = "C:/Users/333da/Desktop/STCGANs/ISTD_Dataset/test/test_B/"

def getImagesPaths(sz):
    files = []
    for r, d, f in os.walk(originalImagesPath):
        for file in f:
            if '.png' in file:
                files.append(file)
    res = []
    for i in range(sz):
        f = random.choice(files)
        res.append((originalImagesPath+f, originalMasksPath+f))
    return res

def goTest(model):
    sz = 1
    paths = getImagesPaths(sz)
    
    result = 0.0
    
    for p in paths:
        TP=0
        TN=0
        FP=0
        FN=0
        _mask = cv2.imread(p[1])
        im = cv2.imread(p[0])
        
        #_pred = model.getShadow(im)
        _pred = model.getShadow(im, 640, 480)
        (thresh, _pred) = cv2.threshold(_pred, 127, 255, cv2.THRESH_BINARY)
 
        
        for i in range(0, len(_mask)):
            for j in range(0, len(_mask[0])):
                res = _pred[i][j]#[-1]
                exp = _mask[i][j][-1]
                #print(res, exp)
                if res==0 and exp ==0:
                    TP+=1
                if res == 255 and exp == 255:
                    TP+=1
                if res ==0 and exp==255:
                    FN+=1
                if res ==255 and exp ==0:
                    FP+=1
        print(TP, TN, FP, FN)
        result += 1-0.5*(TP/(TP+FN+0.0000001) + TN/(TN+FP+0.000001))
                
    print(result / sz)
    

#goTest(hsiModel)
#goTest(dbrarModel)
#goTest(ycbcrModel)
#goTest(labModel)
goTest(adnetModel)
