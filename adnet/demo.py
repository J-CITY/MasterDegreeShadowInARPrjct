from unet256 import Unet256Model
from data.single_dataset import SingleDataset
import numpy as np
from PIL import Image
import torch

def convert(frame):
    A_img = frame.convert('RGB')
    ow = A_img.size[0]
    oh = A_img.size[1]
    A_img = A_img.resize((256,256))
    A_img = np.array(A_img,np.float32)
    #print(A_img)
    #print(A_img.transpose(2, 0, 1))
    A_img = np.log(A_img +1)
    A_img = torch.from_numpy(A_img.transpose(2, 0, 1)).div(np.log(256))
    A_img = A_img-0.5
    A_img = A_img*2
    A = A_img.unsqueeze(0)
    print(A.shape)#1,3,256,256
    return {'A': A, 'w':ow,'h':oh, 'imname': "testp.jpg"}

model = Unet256Model(load_model='135_net_D.pth')
#model.print_net()

#for i,data in enumerate(dataset):
#    out = model.test(data)
#    im_out = out[0].cpu().float().numpy()
#    im_out = np.transpose(im_out,(1,2,0))
#    im_out = (im_out+1)/2*255
#    im_out = im_out.astype('uint8')
#    
#    A = Image.fromarray(np.squeeze(im_out,axis =2)).resize((data['w'],data['h']))
#    A.save('out/'+data['imname'])
im = convert(Image.open("testp.jpg"))
out = model.test(im)
im_out = out[0].cpu().float().numpy()
im_out = np.transpose(im_out,(1,2,0))
im_out = (im_out+1)/2*255
im_out = im_out.astype('uint8')

A = Image.fromarray(np.squeeze(im_out, axis =2)).resize((im['w'], im['h']))
A = A.transpose(Image.ROTATE_270)
A.save('out/'+im['imname'])
