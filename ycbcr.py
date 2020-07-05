import numpy as np
import cv2

class YCBCR:
    def __init__(self):
        pass
        
    def getShadow(self, image):
        h, w, c = image.shape
        image = cv2.resize(image,(320,240))
        y_cb_cr_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        binary_mask = np.zeros((y_cb_cr_img.shape[0],y_cb_cr_img.shape[1]),dtype=np.uint8)
        y_mean = np.mean(cv2.split(y_cb_cr_img)[0])
        
        y_std = np.std(cv2.split(y_cb_cr_img)[0])
        
        for i in range(y_cb_cr_img.shape[0]):
            for j in range(y_cb_cr_img.shape[1]):
                if y_cb_cr_img[i, j, 0] < y_mean - (y_std / 3):
                    binary_mask[i, j] = 255
                else:
                    binary_mask[i, j] = 0
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(binary_mask, kernel, iterations=1)
        ret,th = cv2.threshold(erosion,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        median = cv2.medianBlur(th,15)
        median = cv2.resize(median,(w,h))
        image = cv2.resize(image,(w,h))
        return median
