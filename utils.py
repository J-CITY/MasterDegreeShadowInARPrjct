import numpy as np
import cv2
from PIL import Image
import torch

SMALL_HEIGHT = 800

def resize(img, height=SMALL_HEIGHT, allways=False):
    if (img.shape[0] > height or allways):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))
    return img

def ratio(img, height=SMALL_HEIGHT):
    return img.shape[0] / height


def convert(frame):
	A_img = Image.fromarray(frame).convert('RGB')
	ow = A_img.size[0]
	oh = A_img.size[1]
	A_img = A_img.resize((256,256))
	A_img = np.array(A_img,np.float32)
	A_img = np.log(A_img +1)
	A_img = torch.from_numpy(A_img.transpose(2, 0, 1)).div(np.log(256))
	A_img = A_img-0.5
	A_img = A_img*2
	A = A_img.unsqueeze(0)
	return {'A': A, 'w':ow,'h':oh}