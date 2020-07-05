import sys
sys.path.insert(0, "./dbrar")
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from dbrar.model import BDRAR
import cv2


class DBRAR:
    def __init__(self):
        self.img_transform = transforms.Compose([
            transforms.Resize(416),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.to_pil = transforms.ToPILImage()

        self.net = BDRAR().cuda()
        self.net.load_state_dict(torch.load('dbrar/model.pth', map_location='cuda:0'))
        self.net.eval()

    def getShadow(self, image):
        with torch.no_grad():
            img = Image.fromarray(image)
            w, h = img.size
            img_var = Variable(self.img_transform(img).unsqueeze(0)).cuda()
            res = self.net(img_var)
            prediction = np.array(transforms.Resize((h, w))(self.to_pil(res.data.squeeze(0).cpu())))
    
            #resim = cv2.resize(resim, (h, w))
            return prediction
            

#img = cv2.imread("shadow1.jpg")
#res = getDbrarShadow(img)
#cv2.imwrite('out.png',res)