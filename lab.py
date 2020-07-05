import cv2 as cv2
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image

class LAB:
    def __init__(self):
        pass
        
    def getShadow(self, image):
        rgb = image
        image_B = np.copy(rgb[:, :, 0])
        image_G = np.copy(rgb[:, :, 1])
        image_R = np.copy(rgb[:, :, 2]) 
        s=np.shape(rgb)
        
        #Converting RGB to LAB color space
        lab = color.rgb2lab(rgb)
        image_b = np.copy(lab[:, :, 0])
        image_a = np.copy(lab[:, :, 1])
        image_l = np.copy(lab[:, :, 2])
        
        lm=np.mean(lab[:,:,0], axis=(0, 1))
        am=np.mean(lab[:,:,1], axis=(0, 1))
        bm=np.mean(lab[:,:,2], axis=(0, 1))
        
        #Creating empty mask for masking shadow
        mas = np.empty([rgb.shape[0], rgb.shape[1]], dtype = bool)
        lb=lab[:,:,0]+lab[:,:,2]
        #Hand crafted thresholds: Dataset specific
        #if (am+bm)*100<=6:
        #mas[(image_l <=(lm-(np.std(image_l))/15))] = False
        #else:
        mas[(image_l+image_b)<=10] = False
        B_masked = np.ma.masked_array(image_b, mask = mas)
        G_masked = np.ma.masked_array(image_G, mask = mas)
        R_masked = np.ma.masked_array(image_R, mask = mas) 
        mam = np.dstack([rgb, (~mas).astype(np.uint8)*255])
        #mam = cv2.cvtColor(mam, cv2.COLOR_RGBA2RGB)
        return mam
        
        