import cv2
from PIL import  Image
import numpy as np
import math

class HSI:
    def __init__(self):
        pass
        
    def getShadow(self, image):
        h, w, c = image.shape
        image = cv2.resize(image,(100,100))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(image,9,75,75)
        #print(blur[300,300].astype(float))
        blur = np.divide(blur.astype(float), 255.0)
        #print(blur[300,300])
        cv2.imwrite("image.jpg",blur)
        hsi = np.zeros((blur.shape[0],blur.shape[1],blur.shape[2]),dtype=np.float)
        ratio_map = np.zeros((blur.shape[0],blur.shape[1]),dtype=np.uint8)
        
        for i in range(blur.shape[0]):
            for j in range(blur.shape[1]):
                #print(hsi[i][j])
                #if (hsi[i][j][0]==0.0 and hsi[i][j][1]==0.0 and hsi[i][j][2]==0.0):
                #    continue
                hsi[i][j][2] = (blur[i][j][0]+blur[i][j][1]+blur[i][j][2])/3
                hsi[i][j][0] = math.acos(((blur[i][j][2]-blur[i][j][1])*(blur[i][j][2]-blur[i][j][0])) /
                    (0.000001 + 2*math.sqrt((blur[i][j][2]-blur[i][j][1])*(blur[i][j][2]-blur[i][j][1])+(blur[i][j][2]-blur[i][j][0])*(blur[i][j][1]-blur[i][j][0]))))
                hsi[i][j][1] = 1 - 3*min(blur[i][j][0],blur[i][j][1],blur[i][j][2])/hsi[i][j][2]
                #print(blur[i][j][2], blur[i][j][1], blur[i][j][0])
                ratio_map[i][j] = hsi[i][j][0]/(hsi[i][j][2]+0.01)
    
        hist = np.histogram(ratio_map.ravel(),256,[0,256])
        ret,th = cv2.threshold(ratio_map,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        median = cv2.medianBlur(th,15)
        median = cv2.resize(median,(w,h))
        image = cv2.resize(image,(w,h))
        return median
        