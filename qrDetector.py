import cv2
import numpy as np
import sys
import time
import config
#Draw border
def drawQrCodeBorder(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[ (j+1) % n][0]), (255,0,0), 3)
    return im#cv2.imshow("Results", im)


qrDecoder = cv2.QRCodeDetector()

def qrDetect(inputImage):
    data,bbox,rectifiedImage = qrDecoder.detectAndDecode(inputImage)
    if len(data) > 0:
        print("Decoded Data : {}".format(data))
        if config.IS_DEBUG:
            rectifiedImage = drawQrCodeBorder(inputImage, bbox)
        rectifiedImage = np.uint8(rectifiedImage)
        print(bbox)
        return (rectifiedImage, bbox)
    else:
        print("QR Code not detected")
        return (inputImage, bbox)
