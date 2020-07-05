import torch
from torchvision.models import mobilenet_v2

from unet256 import Unet256Model
from data.single_dataset import SingleDataset
import numpy as np
from PIL import Image
from torch.autograd import Variable
def convert(frame):
	A_img = frame.convert('RGB')
	ow = A_img.size[0]
	oh = A_img.size[1]
	A_img = A_img.resize((256,256))
	A_img = np.array(A_img,np.float32)
	A_img = np.log(A_img +1)
	A_img = torch.from_numpy(A_img.transpose(2, 0, 1)).div(np.log(256))
	A_img = A_img-0.5
	A_img = A_img*2
	A = A_img.unsqueeze(0)
	return {'A': A, 'w':ow,'h':oh, 'imname': "testp.jpg"}


#model = mobilenet_v2(pretrained=True)
model = Unet256Model(load_model='135_net_D.pth')

model.net.eval()
#input_tensor = torch.rand(1,3,224,224)


input_tensor = convert(Image.open("testp.jpg"))

script_model = torch.jit.trace(model.net,
    Variable(input_tensor['A'],requires_grad = 0))
script_model.save("amodel.pt")