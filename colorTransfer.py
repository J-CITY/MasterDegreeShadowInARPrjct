import numpy as np
import cv2
import os

def getMeanStd(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std

def colorTransfer(_source, _target):
	source = cv2.cvtColor(_source, cv2.COLOR_BGR2LAB)
	target = cv2.cvtColor(_target, cv2.COLOR_BGR2LAB)
	sMean, sStd = getMeanStd(source)
	tMean, tStd = getMeanStd(target)
	height, width, channel = source.shape
	for i in range(0,height):
		for j in range(0,width):
			for k in range(0,channel):
				x = source[i,j,k]
				x = ((x-sMean[k])*(tStd[k]/sStd[k]))+tMean[k]
				# round or +0.5
				x = round(x)
				# boundary check
				x = 0 if x<0 else x
				x = 255 if x>255 else x
				source[i,j,k] = x
	source = cv2.cvtColor(source,cv2.COLOR_LAB2RGB)
	return source

def brightnessTransfer(_source, _target, box):
	source =  cv2.cvtColor(_source, cv2.COLOR_BGR2HLS)
	target =  cv2.cvtColor(_target, cv2.COLOR_BGR2HLS)

	h, w, _ = target.shape

	target = cv2.resize(target, (source.shape[1], source.shape[0]))
	
	maxX =-100000
	minX = 100000
	maxY =-100000
	minY = 100000
	if box is not None:
		for p in box:
			if p[1] > maxY:
				maxY = p[1]
			if p[1] < minY:
				minY = p[1]
			if p[0] > maxX:
				maxX = p[0]
			if p[0] < minX:
				minX = p[0]
	if maxX == -100000:
		maxX = _source.shape[1]-1
	if minX == 100000:
		minX = 0
	if maxY == -100000:
		maxY = _source.shape[0]-1
	if minY == 100000:
		minY = 0

	_box = source[minX:maxX,minY:maxY,1]
	#cv2.imwrite("out.png", _box)
	mean = np.mean(_box)
	#target[:,:,1] = mean
	#print(mean)
	alpha = 0.2
	target[:,:,1] = alpha * mean + (1.0 - alpha) * target[:,:,1]

	target =  cv2.cvtColor(target, cv2.COLOR_HLS2BGR)
	target = cv2.resize(target, (w, h))
	return target

def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std

def color_transfer(ws, we, hs, he, s, t):
	s_mean, s_std = get_mean_and_std(s)
	t_mean, t_std = get_mean_and_std(t)
	height, width, channel = s.shape
	for i in range(hs,he):
		for j in range(ws,we):
			for k in range(2,channel):
				x = s[i,j,k]
				x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
				# round or +0.5
				x = round(x)
				# boundary check
				x = 0 if x<0 else x
				x = 255 if x>255 else x
				s[i,j,k] = x
	s = cv2.cvtColor(s,cv2.COLOR_LAB2RGB)
	return s

def brightnessTransfer2(_source, _target, box, shift):
	source =  cv2.cvtColor(_source, cv2.COLOR_BGR2HLS)
	target =  cv2.cvtColor(_target, cv2.COLOR_BGR2HLS)
	h, w, _ = source.shape
	#cv2.imwrite("out.png", target)
	target = cv2.resize(target, (source.shape[1], source.shape[0]))
	
	maxX =-100000
	minX = 100000
	maxY =-100000
	minY = 100000
	if box is not None:
		for p in box:
			if p[0] > maxY:
				maxY = p[0]
			if p[0] < minY:
				minY = p[0]
			if p[1] > maxX:
				maxX = p[1]
			if p[1] < minX:
				minX = p[1]
	if maxX == -100000 or maxX >= _source.shape[1]:
		maxX = _source.shape[1]-1
	if minX == 100000 or minX < 0:
		minX = 0
	if maxY == -100000 or maxY >= _source.shape[0]:
		maxY = _source.shape[0]-1
	if minY == 100000 or minY < 0:
		minY = 0


	alpha = 0.22
	alpha2 = 0.05
	step = 10
	
	for y in range(0, h-step-1, step):
		for x in range(0, w-step-1, step):
			if (y-shift[1] < 0 or x-shift[0] < 0 or y+step-shift[1] < 0 or x+step-shift[0] < 0):
				continue
			#mean = np.mean(source[y:y+step, x:x+step, 1])
			#target[y-shift[1]:y+step-shift[1], x-shift[0]:x+step-shift[0], 1] = alpha * mean + (1.0 - alpha) * \
			#	target[y-shift[1]:y+step-shift[1], x-shift[0]:x+step-shift[0], 1]

			mean = np.mean(source[y:y+step, x:x+step, 2])
			target[y-shift[1]:y+step-shift[1], x-shift[0]:x+step-shift[0], 2] = alpha * mean + (1.0 - alpha) * \
				target[y-shift[1]:y+step-shift[1], x-shift[0]:x+step-shift[0], 2]
			
			#mean = np.mean(source[y:y+step, x:x+step, 0])
			#target[y-shift[1]:y+step-shift[1], x-shift[0]:x+step-shift[0], 0] = alpha2 * mean + (1.0 - alpha2) * \
			#	target[y-shift[1]:y+step-shift[1], x-shift[0]:x+step-shift[0], 0]

	target =  cv2.cvtColor(target, cv2.COLOR_HLS2RGB)


	#s = cv2.cvtColor(_source,cv2.COLOR_BGR2LAB)
	#t = cv2.cvtColor(target,cv2.COLOR_BGR2LAB)
	#target = color_transfer(w-shift[0], w-shift[0] , y-shift[1], h-shift[1], t, s)

	return target

#s = cv2.imread("in1.jpg")
#s = cv2.cvtColor(s,cv2.COLOR_BGR2LAB)
#t = cv2.imread("in2.jpg")
#t = cv2.cvtColor(t,cv2.COLOR_BGR2LAB)
#
#cv2.imwrite('res.png', colorTransfer(s, t))