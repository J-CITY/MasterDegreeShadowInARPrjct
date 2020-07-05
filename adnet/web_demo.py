from unet256 import Unet256Model
from data.single_dataset import SingleDataset
import numpy as np
from PIL import Image
import cv2
import torch
#dataset = SingleDataset('../datasets/SBUsd/Test/TestA/')
#print('Dataset size: ' + str(len(dataset)))

model = Unet256Model(load_model='135_net_D.pth', isCPU=False)
#model.print_net()

#for i,data in enumerate(dataset):
#	out = model.test(data)
#	im_out = out[0].cpu().float().numpy()
#	im_out = np.transpose(im_out,(1,2,0))
#	im_out = (im_out+1)/2*255
#	im_out = im_out.astype('uint8')
#	
#	A = Image.fromarray(np.squeeze(im_out,axis =2)).resize((data['w'],data['h']))
#	A.save('out/'+data['imname'])


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


cap = cv2.VideoCapture('input2.mp4')

#out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

width = cap.get(3)
height = cap.get(4)

while(cap.isOpened()):
	ret, frame = cap.read()
	
	if ret == False:
		break
	
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	outim = model.test(convert(frame))
	im_out = outim[0].cpu().float().numpy()
	im_out = np.transpose(im_out, (1,2,0))
	im_out = (im_out+1)/2*255
	im_out = im_out.astype('uint8')
	
	gray = Image.fromarray(np.squeeze(im_out, axis =2)).resize((int(width), int(height)))
	
	open_cv_image = np.array(gray)
	#open_cv_image = open_cv_image[:, :, ::-1].copy() 
	
	out.write(open_cv_image)
	cv2.imshow('frame', open_cv_image)
	
	#EXIT
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	

cap.release()
out.release()
cv2.destroyAllWindows()
	
	
	