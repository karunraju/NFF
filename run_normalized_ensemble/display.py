import torch
import cv2
import sys
import numpy as np

model = torch.load("ensemble_model_2.pth",map_location='cpu')
big = []
small=[]
for key in model.keys():
	if 'weight' in key:
		big.append(model[key].numpy().max())
		small.append(model[key].numpy().min())
biggest = max(big)
smallest = min(small)

for key in model.keys():
	if 'weight' in key:
		image = model[key].unsqueeze(-1).numpy()
		image1 = image*255/(image.max()-image.min())
		image2 = image*255
		image3 = image*255/(biggest-smallest)
		cv2.imwrite("{}.png".format(key), np.concatenate([image1,image2,image3],-1))
